"""
Correlation Verification for Figure 9 (Section 5).

This script verifies that weight-based eigenvectors actually predict SAE activations
on real data. It computes the correlation between:

1. True Activations: z_true = SAE.encode(mlp_out)
2. Predicted Activations: z_pred = sum_{j=1}^k lambda_j * (v_j^T x)^2

Where:
- x is the input to the bilinear layer (mlp_in, in residual stream space)
- v_j are eigenvectors of the interaction matrix Q
- lambda_j are corresponding eigenvalues

Paper claim: Rank-2 approximation captures >75% of variance (Figure 9).

Usage:
    python src/language/verify_correlation.py --config configs/language_interaction.yaml
    python src/language/verify_correlation.py --config configs/language_interaction.yaml --n-features 50 --ranks 1,2,4,8,16
"""

import sys
from pathlib import Path
import argparse
import json
import warnings

# Suppress torchvision image extension warning
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
    load_config,
    track_emissions,
    init_wandb,
    finish_wandb,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs,
    predict_activation_from_eigenpairs,
)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient between two tensors.
    
    Args:
        x: First tensor [n]
        y: Second tensor [n]
    
    Returns:
        Correlation coefficient (float)
    """
    x = x.float()
    y = y.float()
    
    # Handle edge cases
    if len(x) < 2:
        return float('nan')
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = (x_centered * y_centered).sum()
    denominator = (x_centered.pow(2).sum().sqrt() * y_centered.pow(2).sum().sqrt())
    
    if denominator < 1e-10:
        return float('nan')
    
    return (numerator / denominator).item()


def create_validation_batch(tokenizer, config: dict, device: str, n_samples: int = 1000):
    """
    Create a validation batch from TinyStories dataset.
    
    Args:
        tokenizer: Model tokenizer
        config: Experiment config
        device: Device to use
        n_samples: Number of samples to load
    
    Returns:
        Dict with input_ids and attention_mask tensors
    """
    print("Loading TinyStories validation data...")
    
    # Load dataset (use validation split if available, else sample from train)
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
    except Exception:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Sample if dataset is larger than needed
    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    
    n_ctx = config.get("sae", {}).get("n_ctx", 256)
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=n_ctx,
            padding="max_length",
            return_tensors="pt",
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    
    # Create batch
    batch_size = min(32, len(dataset))
    batch = {
        "input_ids": dataset["input_ids"][:batch_size].to(device),
        "attention_mask": dataset["attention_mask"][:batch_size].to(device),
    }
    
    return batch


def find_active_features(sae_activations: torch.Tensor, min_active: int = 10) -> list:
    """
    Find features that are active in the batch.
    
    Args:
        sae_activations: SAE activations [batch*seq, d_features]
        min_active: Minimum number of active positions required
    
    Returns:
        List of feature indices that are sufficiently active
    """
    # Count active positions per feature (activation > 0)
    active_counts = (sae_activations > 0).sum(dim=0)  # [d_features]
    
    # Find features with enough active positions
    active_features = (active_counts >= min_active).nonzero(as_tuple=True)[0]
    
    return active_features.tolist()


def verify_correlation(
    model,
    sae_out: SAE,
    layer: int,
    val_batch: dict,
    feature_indices: list,
    ranks: list,
    device: str,
    min_active: int = 10,
    max_features: int = 50,
):
    """
    Compute correlation between weight-based predictions and actual SAE activations.
    
    The key equation being verified:
        z_c(x) ≈ sum_{j=1}^k lambda_j * (v_j^T x)^2
    
    Args:
        model: Transformer model
        sae_out: Output SAE (for encoding mlp_out)
        layer: Layer index
        val_batch: Validation batch dict
        feature_indices: List of feature indices to analyze (or None for auto-detect)
        ranks: List of ranks to evaluate [1, 2, 4, 8, 16]
        device: Device for computation
        min_active: Minimum active positions for a feature to be analyzed
        max_features: Maximum number of features to analyze
    
    Returns:
        Dict with correlation results per rank
    """
    results = {k: [] for k in ranks}
    feature_results = []
    
    sight = Sight(model)
    
    print(f"\nCapturing activations for layer {layer}...")
    
    with torch.no_grad():
        # Forward pass to capture mlp_in and mlp_out
        with sight.trace(val_batch, validate=False, scan=False):
            # mlp_in: input to bilinear layer (after LayerNorm n2)
            # This is in residual stream space [batch, seq, d_model]
            mlp_in = sight["mlp-in", layer].save()
            
            # mlp_out: output of bilinear layer [batch, seq, d_model]
            mlp_out = sight["mlp-out", layer].save()
        
        print(f"  mlp_in shape: {mlp_in.shape}")
        print(f"  mlp_out shape: {mlp_out.shape}")
        
        # Flatten batch and sequence dimensions: [batch*seq, d_model]
        mlp_in_flat = mlp_in.flatten(0, 1).float()
        mlp_out_flat = mlp_out.flatten(0, 1).float()
        
        # Move SAE to same device and compute activations
        sae_out_device = sae_out.to(device)
        
        # Compute true SAE activations for all features
        print("Computing true SAE activations...")
        z_true_all = sae_out_device.encode(mlp_out_flat)  # [batch*seq, d_features]
        
        # Move to CPU for analysis
        z_true_all = z_true_all.cpu()
        mlp_in_flat = mlp_in_flat.cpu()
        
        print(f"  z_true_all shape: {z_true_all.shape}")
        print(f"  z_true_all nonzero: {(z_true_all > 0).sum().item()}")
        
        # Find features with sufficient activations
        print("Finding active features...")
        all_active_features = find_active_features(z_true_all, min_active=min_active)
        print(f"  Found {len(all_active_features)} features with >= {min_active} active positions")
        
        # If specific features requested, filter to only those that are active
        # Otherwise use the active features found
        if feature_indices:
            # Filter requested features to only those that are active
            active_set = set(all_active_features)
            feature_indices = [f for f in feature_indices if f in active_set]
            if len(feature_indices) == 0:
                # Fall back to using active features
                print(f"  Warning: None of requested features are active, using auto-detected features")
                feature_indices = all_active_features
        else:
            feature_indices = all_active_features
        
        # Limit to max_features
        if len(feature_indices) > max_features:
            feature_indices = feature_indices[:max_features]
        
        n_features = len(feature_indices)
        
        print(f"\nAnalyzing {n_features} features across ranks {ranks}...")
        
        for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
            # Get decoder direction for this feature
            # w_dec.weight is [d_model, d_features], so column feat_idx gives [d_model]
            out_direction = sae_out_device.w_dec.weight[:, feat_idx].cpu()
            
            # Get eigenpairs from weights (unprojected Q in residual stream space)
            eigenpairs = get_interaction_eigenpairs(
                model=model,
                layer=layer,
                feat_idx=feat_idx,
                out_decoder_direction=out_direction,
                device="cpu",
            )
            
            # True activation for this feature
            z_true = z_true_all[:, feat_idx]  # [batch*seq]
            
            # Filter for active positions (z > 0) 
            # We need sufficient active samples for meaningful correlation
            active_mask = z_true > 0
            n_active = active_mask.sum().item()
            
            if n_active < min_active:
                continue
            
            x_active = mlp_in_flat[active_mask]  # [n_active, d_model]
            z_active = z_true[active_mask]       # [n_active]
            
            feature_corrs = {}
            
            # Compute correlation for each rank
            for k in ranks:
                # Get top-k eigenpairs
                k_actual = min(k, len(eigenpairs.eigenvalues))
                eigenvalues_k = eigenpairs.eigenvalues[:k_actual]
                eigenvectors_k = eigenpairs.eigenvectors[:, :k_actual]
                
                # Predicted activation: sum_j lambda_j * (v_j . x)^2
                z_pred = predict_activation_from_eigenpairs(
                    x_active, eigenvalues_k, eigenvectors_k
                )
                
                # Pearson correlation
                corr = pearson_correlation(z_active, z_pred)
                
                if not np.isnan(corr):
                    results[k].append(corr)
                    feature_corrs[k] = corr
            
            feature_results.append({
                "feat_idx": feat_idx,
                "n_active": n_active,
                "correlations": feature_corrs,
            })
    
    return results, feature_results


def main():
    parser = argparse.ArgumentParser(description="Verify weight-correlation predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/correlation_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-features", type=int, default=50, help="Number of features to analyze")
    parser.add_argument("--ranks", type=str, default="1,2,4,8,16", help="Ranks to evaluate (comma-separated)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--min-active", type=int, default=10, help="Minimum active positions per feature")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    
    # Parse ranks
    ranks = [int(r.strip()) for r in args.ranks.split(",")]
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - some operations may use CPU fallback")
    
    # Load config
    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)
    
    # Initialize wandb
    wandb_enabled = init_wandb(
        name=f"correlation_{config_name}",
        config={**config, "ranks": ranks, "n_features": args.n_features},
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "correlation", "figure9"],
    )
    
    # Run with emissions tracking
    with track_emissions("fact-bilinear") as tracker:
        # Load model
        model_name = config.get("model", {}).get("pretrained", "tdooms/fw-medium")
        print(f"\nLoading model: {model_name}")
        model = Transformer.from_pretrained(model_name, device=device)
        
        # Print model config
        print(f"  d_model: {model.config.d_model}")
        print(f"  d_hidden: {model.config.d_hidden}")
        print(f"  n_layer: {model.config.n_layer}")
        
        # Load output SAE
        sae_config = config.get("sae", {})
        layer = sae_config.get("layer", 2)
        out_config = sae_config.get("output", {"name": "mlp-out", "expansion": 4, "k": 30})
        
        repo = f"{model.config.repo}-scope"
        print(f"\nLoading output SAE from {repo}...")
        print(f"  Point: ({out_config['name']}, {layer})")
        print(f"  Expansion: {out_config['expansion']}, k: {out_config['k']}")
        
        sae_out = SAE.from_pretrained(
            repo,
            point=(out_config["name"], layer),
            expansion=out_config["expansion"],
            k=out_config["k"],
        ).to(device)
        
        print(f"  SAE d_model: {sae_out.d_model}")
        print(f"  SAE d_features: {sae_out.d_features}")
        
        # Create validation batch
        val_batch = create_validation_batch(
            model.tokenizer, config, device, n_samples=args.n_samples
        )
        print(f"  Validation batch size: {val_batch['input_ids'].shape[0]}")
        
        # Select features to analyze
        # Pass None to find active features dynamically, then limit to n_features
        feature_indices = None  # Will be populated by find_active_features in verify_correlation
        
        # Run correlation analysis
        results, feature_results = verify_correlation(
            model=model,
            sae_out=sae_out,
            layer=layer,
            val_batch=val_batch,
            feature_indices=feature_indices,
            ranks=ranks,
            device=device,
            min_active=args.min_active,
            max_features=args.n_features,
        )
    
    # Compute summary statistics
    summary = {
        "model_name": model_name,
        "layer": layer,
        "n_features_requested": args.n_features,
        "n_features_analyzed": len(feature_results),
        "ranks": ranks,
        "correlation_by_rank": {},
        "paper_claim": "rank-2 captures >75% variance",
        "wall_time_seconds": tracker.result.wall_time_seconds,
        "co2_kg": tracker.result.emissions_kg,
    }
    
    print(f"\n{'='*60}")
    print("CORRELATION VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {layer}")
    print(f"Features analyzed: {summary['n_features_analyzed']}")
    print()
    
    for k in ranks:
        corrs = results[k]
        if len(corrs) > 0:
            mean_corr = np.mean(corrs)
            std_corr = np.std(corrs)
            median_corr = np.median(corrs)
            summary["correlation_by_rank"][str(k)] = {
                "mean": float(mean_corr),
                "std": float(std_corr),
                "median": float(median_corr),
                "n_features": len(corrs),
            }
            print(f"Rank {k:2d}: mean={mean_corr:.4f}, std={std_corr:.4f}, median={median_corr:.4f} (n={len(corrs)})")
        else:
            print(f"Rank {k:2d}: No valid correlations computed")
    
    print(f"\nPaper claim: {summary['paper_claim']}")
    if "2" in summary["correlation_by_rank"]:
        rank2_corr = summary["correlation_by_rank"]["2"]["mean"]
        print(f"Our rank-2 correlation: {rank2_corr:.4f}")
    print(f"{'='*60}")
    
    # Finalize wandb
    if wandb_enabled:
        extra_summary = {
            "n_features_analyzed": summary["n_features_analyzed"],
        }
        for k, stats in summary["correlation_by_rank"].items():
            extra_summary[f"correlation_rank_{k}_mean"] = stats["mean"]
            extra_summary[f"correlation_rank_{k}_std"] = stats["std"]
        finish_wandb(tracker.result, extra_summary=extra_summary)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "summary": summary,
        "per_feature": [
            {
                "feat_idx": fr["feat_idx"],
                "n_active": fr["n_active"],
                "correlations": {str(k): float(v) for k, v in fr["correlations"].items()},
            }
            for fr in feature_results
        ],
        "config": config,
    }
    
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
