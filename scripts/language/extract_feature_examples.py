#!/usr/bin/env python3
"""
Extract example sentences with activations for Figure 8 features.

This script:
1. Loads the fw-medium model and SAEs
2. Finds top-activating examples for features 3834 and 751
3. Computes the correlation between true and rank-2 predicted activations
4. Saves results as JSON for use in LaTeX tables

Usage:
    python scripts/language/extract_feature_examples.py --output results/language/feature_examples.json
"""

import sys
from pathlib import Path
import argparse
import json
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from language.utils import Sight
from sae.sae import SAE

from src.utils import (
    get_device,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs_streaming,
    predict_activation_from_eigenpairs,
)
from src.language.context import LanguageContext


def extract_examples(
    model,
    sae_out: SAE,
    layer: int,
    tokenizer,
    feature_indices: list,
    device: str,
    n_samples: int = 5000,
    top_k_examples: int = 10,
    chunk_size: int = 256,
):
    """
    Extract top-activating examples for specified features.
    
    Args:
        model: Transformer model
        sae_out: Output SAE
        layer: Layer index
        tokenizer: Model tokenizer
        feature_indices: List of feature indices to analyze
        device: Device for computation
        n_samples: Number of text samples to process
        top_k_examples: Number of top examples to extract per feature
        chunk_size: Chunk size for streaming Q computation
    
    Returns:
        Dict mapping feature index to list of examples with activations
    """
    sight = Sight(model)
    sae_out_device = sae_out.to(device)
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
    except Exception:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    
    n_ctx = 256
    
    # Collect activations for each feature
    feature_activations = {feat: [] for feat in feature_indices}
    
    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Processing samples")):
            text = example["text"]
            
            # Tokenize
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=n_ctx,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Forward pass
            with sight.trace(tokens, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            # Get SAE activations
            z_true = sae_out_device.encode(mlp_out.flatten(0, 1).float())  # [seq, d_features]
            
            # For each feature, find activating positions
            for feat_idx in feature_indices:
                feat_activations = z_true[:, feat_idx]  # [seq]
                
                # Find non-zero activations
                active_mask = feat_activations > 0
                if not active_mask.any():
                    continue
                
                # Get the max activation position
                max_pos = feat_activations.argmax().item()
                max_activation = feat_activations[max_pos].item()
                
                # Decode tokens around the activation
                input_ids = tokens["input_ids"][0].cpu().tolist()
                
                # Get context window around the activation
                context_start = max(0, max_pos - 5)
                context_end = min(len(input_ids), max_pos + 10)
                context_ids = input_ids[context_start:context_end]
                
                # Decode the context
                context_text = tokenizer.decode(context_ids, skip_special_tokens=False)
                
                # Get the specific token
                if max_pos < len(input_ids):
                    token_text = tokenizer.decode([input_ids[max_pos]], skip_special_tokens=False)
                else:
                    token_text = ""
                
                feature_activations[feat_idx].append({
                    "sample_idx": i,
                    "position": max_pos,
                    "activation": max_activation,
                    "token": token_text.strip(),
                    "context": context_text.strip(),
                    "full_text": text[:200] + "..." if len(text) > 200 else text,
                })
    
    # Sort by activation and take top-k for each feature
    results = {}
    for feat_idx in feature_indices:
        examples = feature_activations[feat_idx]
        examples.sort(key=lambda x: x["activation"], reverse=True)
        results[feat_idx] = examples[:top_k_examples]
    
    return results


def compute_rank2_correlations(
    model,
    sae_out: SAE,
    layer: int,
    tokenizer,
    feature_indices: list,
    device: str,
    n_samples: int = 2000,
    chunk_size: int = 256,
):
    """
    Compute rank-2 correlation for specified features.
    
    Returns:
        Dict mapping feature index to correlation value
    """
    sight = Sight(model)
    sae_out_device = sae_out.to(device)
    
    # Load dataset
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
    except Exception:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    
    n_ctx = 256
    
    # Accumulate activations
    all_mlp_in = []
    all_z_true = []
    
    print("Accumulating activations for correlation computation...")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Processing"):
            text = example["text"]
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=n_ctx,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            with sight.trace(tokens, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            all_mlp_in.append(mlp_in.flatten(0, 1).cpu())
            all_z_true.append(sae_out_device.encode(mlp_out.flatten(0, 1).float()).cpu())
    
    mlp_in_all = torch.cat(all_mlp_in, dim=0)
    z_true_all = torch.cat(all_z_true, dim=0)
    
    # Compute correlation for each feature
    correlations = {}
    
    print("Computing rank-2 correlations...")
    
    for feat_idx in tqdm(feature_indices, desc="Features"):
        out_direction = sae_out_device.w_enc.weight[feat_idx, :]
        
        # Get eigenpairs
        eigenpairs = get_interaction_eigenpairs_streaming(
            model=model,
            layer=layer,
            feat_idx=feat_idx,
            out_direction=out_direction,
            chunk_size=chunk_size,
            compute_device=device,
        )
        
        # Get rank-2 approximation
        eigenvalues_k = eigenpairs.eigenvalues[:2]
        eigenvectors_k = eigenpairs.eigenvectors[:, :2]
        
        # Get true activations for this feature
        z_true_feat = z_true_all[:, feat_idx]
        
        # Filter for active positions
        active_mask = z_true_feat > 0
        if active_mask.sum() < 10:
            correlations[feat_idx] = float('nan')
            continue
        
        x_active = mlp_in_all[active_mask]
        z_active = z_true_feat[active_mask]
        
        # Predict activation
        z_pred = predict_activation_from_eigenpairs(x_active, eigenvalues_k, eigenvectors_k)
        
        # Compute correlation
        z_active_np = z_active.numpy()
        z_pred_np = z_pred.numpy()
        
        if len(z_active_np) > 1:
            corr = np.corrcoef(z_active_np, z_pred_np)[0, 1]
        else:
            corr = float('nan')
        
        correlations[feat_idx] = float(corr) if not np.isnan(corr) else None
    
    return correlations


def main():
    parser = argparse.ArgumentParser(description="Extract feature examples for Figure 8")
    parser.add_argument("--output", type=str, default="results/language/feature_examples.json",
                        help="Output JSON file")
    parser.add_argument("--features", type=str, default="3834,751",
                        help="Comma-separated feature indices to analyze")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of samples to process for examples")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top examples per feature")
    parser.add_argument("--compute-correlation", action="store_true",
                        help="Also compute rank-2 correlations")
    args = parser.parse_args()
    
    # Parse features
    feature_indices = [int(f.strip()) for f in args.features.split(",")]
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Load model and SAE
    print("Loading model and SAE...")
    config = {
        "model": {"pretrained": "tdooms/fw-medium"},
        "sae": {"layer": 7, "expansion": 8, "k": 30},
    }
    ctx = LanguageContext(config, device)
    model = ctx.model
    layer = ctx.layer
    tokenizer = model.tokenizer
    sae_out = ctx.get_sae("mlp-out")
    
    print(f"Analyzing features: {feature_indices}")
    
    # Extract examples
    examples = extract_examples(
        model=model,
        sae_out=sae_out,
        layer=layer,
        tokenizer=tokenizer,
        feature_indices=feature_indices,
        device=device,
        n_samples=args.n_samples,
        top_k_examples=args.top_k,
    )
    
    # Optionally compute correlations
    correlations = {}
    if args.compute_correlation:
        correlations = compute_rank2_correlations(
            model=model,
            sae_out=sae_out,
            layer=layer,
            tokenizer=tokenizer,
            feature_indices=feature_indices,
            device=device,
            n_samples=2000,
        )
    
    # Prepare output
    results = {
        "model": "tdooms/fw-medium",
        "layer": layer,
        "features": {},
    }
    
    for feat_idx in feature_indices:
        results["features"][str(feat_idx)] = {
            "examples": examples.get(feat_idx, []),
            "rank2_correlation": correlations.get(feat_idx),
            "n_examples": len(examples.get(feat_idx, [])),
        }
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE EXAMPLES SUMMARY")
    print("="*60)
    
    for feat_idx in feature_indices:
        feat_data = results["features"][str(feat_idx)]
        print(f"\nFeature {feat_idx}:")
        print(f"  Examples found: {feat_data['n_examples']}")
        if feat_data['rank2_correlation'] is not None:
            print(f"  Rank-2 correlation: {feat_data['rank2_correlation']:.4f}")
        
        if feat_data['examples']:
            print(f"  Top activating contexts:")
            for i, ex in enumerate(feat_data['examples'][:5]):
                print(f"    {i+1}. [{ex['activation']:.3f}] '{ex['token']}' in: {ex['context'][:60]}...")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
