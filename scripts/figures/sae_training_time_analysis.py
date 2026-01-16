#!/usr/bin/env python3
"""
SAE Training Time Analysis for Figure 10.

This script systematically compares SAE checkpoints with different training times
(v0-v4) to demonstrate that correlation quality improves with SAE training duration.

The paper claims:
- "correlation drastically improves with longer SAE training times" (Appendix C)
- "The correlation distribution is strongly bimodal for the 'under-trained' SAEs"

This script provides empirical evidence for these claims by comparing:
- v0: 1x training (under-trained)
- v1: 2x training
- v2: 4x training
- v3: 8x training
- v4: 16x training (well-trained)

All checkpoints are from fw-medium layer 12 with expansion=16.

Usage:
    python scripts/sae_training_time_analysis.py
    python scripts/sae_training_time_analysis.py --device cuda --n-features 300
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from language.transformer import Transformer
from language.utils import Sight
from sae.sae import SAE, SAEConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two tensors."""
    x = x - x.mean()
    y = y - y.mean()
    corr = (x @ y) / (torch.norm(x) * torch.norm(y) + 1e-10)
    return corr.item()


def load_sae_with_tag(repo: str, layer: int, expansion: int, k: int, tag: str) -> SAE:
    """Load SAE with a specific training tag (v0, v1, v2, v3, v4)."""
    point = ('mlp-out', layer)
    config = SAEConfig(point=point, expansion=expansion, k=k, d_model=0)
    base_name = f"{config.name}-{tag}"
    
    config_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/config.json")
    model_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/model.safetensors")
    
    sae = SAE.from_config(**json.load(open(config_path)))
    load_model(sae, model_path)
    return sae


def analyze_sae_version(
    model: Transformer,
    sae: SAE,
    layer: int,
    mlp_in_all: torch.Tensor,
    z_true_all: torch.Tensor,
    ranks: list,
    n_features: int = -1,
    min_active: int = 50,
) -> dict:
    """
    Analyze correlation for a single SAE version.
    
    Returns dict with per-feature and aggregate statistics.
    
    Args:
        n_features: Number of features to analyze. -1 means all features.
    """
    # Pre-extract weights
    w_l = model.w_l[layer].cpu().float()
    w_r = model.w_r[layer].cpu().float()
    w_p = model.w_p[layer].cpu().float()
    
    # Find features with sufficient activations
    activations_per_feature = (z_true_all > 0).sum(dim=0)
    active_features = (activations_per_feature >= min_active).nonzero().squeeze(-1)
    
    results = {rank: [] for rank in ranks}
    per_feature_results = []
    
    # Analyze all features if n_features == -1
    if n_features == -1 or n_features >= len(active_features):
        features_to_analyze = active_features
    else:
        features_to_analyze = active_features[:n_features]
    
    for feat_idx in tqdm(features_to_analyze, desc="Analyzing features", leave=False):
        feat_idx = feat_idx.item()
        z_true_feat = z_true_all[:, feat_idx]
        active_mask = z_true_feat > 0
        n_active = active_mask.sum().item()
        
        x_active = mlp_in_all[active_mask]
        z_active = z_true_feat[active_mask]
        
        # Compute interaction matrix Q
        out_direction = sae.w_enc.weight[feat_idx, :].cpu().float()
        proj = out_direction @ w_p
        scaled_l = proj.unsqueeze(1) * w_l
        Q = scaled_l.T @ w_r
        Q_sym = 0.5 * (Q + Q.T)
        
        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(Q_sym)
        sort_idx = eigvals.abs().argsort(descending=True)
        eigvals_sorted = eigvals[sort_idx]
        eigvecs_sorted = eigvecs[:, sort_idx]
        
        # Compute correlations for each rank
        feature_corrs = {'feat_idx': feat_idx, 'n_active': n_active}
        for rank in ranks:
            projections = x_active @ eigvecs_sorted[:, :rank]
            z_pred = (projections ** 2) @ eigvals_sorted[:rank]
            corr = pearson_corr(z_active, z_pred)
            
            if not np.isnan(corr):
                results[rank].append(corr)
                feature_corrs[f'rank_{rank}'] = corr
        
        per_feature_results.append(feature_corrs)
    
    # Compute summary statistics
    summary = {}
    for rank in ranks:
        if results[rank]:
            corrs = np.array(results[rank])
            summary[f'rank_{rank}'] = {
                'mean': float(np.mean(corrs)),
                'std': float(np.std(corrs)),
                'median': float(np.median(corrs)),
                'above_75_pct': float(np.mean(corrs > 0.75) * 100),
                'n_features': len(corrs),
            }
    
    return {
        'summary': summary,
        'per_feature': per_feature_results,
        'n_active_features': len(active_features),
    }


def main():
    parser = argparse.ArgumentParser(description="SAE Training Time Analysis")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--n-features', type=int, default=-1, help='Number of features to analyze (-1 for all)')
    parser.add_argument('--n-batches', type=int, default=10, help='Number of batches for activation collection')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--min-active', type=int, default=50, help='Minimum active samples per feature')
    parser.add_argument('--output', type=str, default='results/language/sae_training_time_comparison.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SAE Training Time Analysis (Figure 10)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Features per version: {args.n_features}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    model_name = 'tdooms/fw-medium'
    layer = 12
    expansion = 16
    k = 30
    sae_versions = ['v0', 'v1', 'v2', 'v3', 'v4']
    ranks = [1, 2, 4, 8, 16, 30]
    
    # Load model
    print("Loading model...")
    model = Transformer.from_pretrained(model_name, device='cpu')
    repo = f'{model.config.repo}-scope'
    n_ctx = model.config.n_ctx
    print(f"  Model: {model_name}")
    print(f"  Layer: {layer}, Expansion: {expansion}")
    print(f"  n_ctx: {n_ctx}")
    
    # Load tokenizer and dataset
    print("\nLoading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('roneneldan/TinyStories', split='train[:2000]')
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=n_ctx, 
                        padding='max_length', return_tensors='pt')
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Collect activations once (shared across all SAE versions)
    print("\nCollecting activations...")
    sight = Sight(model)
    all_mlp_in = []
    all_mlp_out = []
    
    with torch.no_grad():
        for i in tqdm(range(args.n_batches), desc="Processing batches"):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(dataset))
            if start_idx >= len(dataset):
                break
            batch = {k: v[start_idx:end_idx] for k, v in dataset[:end_idx].items() 
                    if start_idx < end_idx}
            
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight['mlp-in', layer].save()
                mlp_out = sight['mlp-out', layer].save()
            
            all_mlp_in.append(mlp_in.flatten(0, 1).float().cpu())
            all_mlp_out.append(mlp_out.flatten(0, 1).float().cpu())
    
    mlp_in_all = torch.cat(all_mlp_in, dim=0)
    mlp_out_all = torch.cat(all_mlp_out, dim=0)
    print(f"  Collected {mlp_in_all.shape[0]} tokens")
    
    # Analyze each SAE version
    all_results = {
        'metadata': {
            'model': model_name,
            'layer': layer,
            'expansion': expansion,
            'k': k,
            'n_tokens': mlp_in_all.shape[0],
            'ranks': ranks,
            'timestamp': datetime.now().isoformat(),
        },
        'versions': {}
    }
    
    for version in sae_versions:
        print(f"\n{'='*50}")
        print(f"Analyzing SAE version: {version}")
        print(f"{'='*50}")
        
        # Load SAE
        print(f"  Loading SAE {version}...")
        sae = load_sae_with_tag(repo, layer, expansion, k, version)
        
        # Compute SAE activations
        print(f"  Computing SAE activations...")
        z_true_all = sae.encode(mlp_out_all)
        
        # Analyze
        print(f"  Analyzing correlations (n_features={args.n_features}, -1=all)...")
        results = analyze_sae_version(
            model=model,
            sae=sae,
            layer=layer,
            mlp_in_all=mlp_in_all,
            z_true_all=z_true_all,
            ranks=ranks,
            n_features=args.n_features,
            min_active=args.min_active,
        )
        
        all_results['versions'][version] = results
        
        # Print summary
        print(f"\n  Summary for {version}:")
        for rank in [1, 2, 8]:
            if f'rank_{rank}' in results['summary']:
                s = results['summary'][f'rank_{rank}']
                print(f"    Rank-{rank}: mean={s['mean']:.3f}, median={s['median']:.3f}, "
                      f">{0.75*100:.0f}%: {s['above_75_pct']:.1f}%")
        
        # Clean up
        del sae, z_true_all
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("SUMMARY: SAE Training Time Effect")
    print(f"{'='*70}")
    print(f"{'Version':<10} {'Rank-1':>10} {'Rank-2':>10} {'Rank-8':>10} {'%>0.75':>10}")
    print("-" * 50)
    for version in sae_versions:
        v_results = all_results['versions'][version]['summary']
        r1 = v_results.get('rank_1', {}).get('mean', 0)
        r2 = v_results.get('rank_2', {}).get('mean', 0)
        r8 = v_results.get('rank_8', {}).get('mean', 0)
        pct = v_results.get('rank_2', {}).get('above_75_pct', 0)
        print(f"{version:<10} {r1:>10.3f} {r2:>10.3f} {r8:>10.3f} {pct:>9.1f}%")
    print("-" * 50)
    print(f"{'Paper':>10} {'~0.65':>10} {'>0.75':>10} {'--':>10} {'69%':>10}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
