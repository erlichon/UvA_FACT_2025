#!/usr/bin/env python3
"""
Generate Figure 8-equivalent analysis for ts-medium layer 5.

ts-medium is the paper's "ts-tiny" model (6 layers, 29M params, TinyStories).
Layer 5 is the only layer with both mlp-in and mlp-out SAEs available on HuggingFace.

This provides a closer reproduction to the paper's original Figure 8 than fw-medium.

Usage:
    python scripts/figures/generate_ts_medium_layer5.py [--device mps|cuda|cpu]
"""

import sys
from pathlib import Path
import argparse
import json
import gc

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from language.transformer import Transformer
from sae.sae import SAE
from sae.tracer import Tracer
from datasets import load_dataset

from src.utils import get_device, setup_mps_fallbacks, is_mps_device
from src.paths import LANGUAGE_RESULTS, LANGUAGE_FIGURES


def load_ts_medium_with_saes(device: str):
    """Load ts-medium model with layer 5 SAEs."""
    print("Loading ts-medium model...")
    
    # Load model
    if device == "mps":
        model = Transformer.from_pretrained("tdooms/ts-medium", device="cpu").to(device)
    else:
        model = Transformer.from_pretrained("tdooms/ts-medium", device=device)
    
    print(f"Model: {model.config.n_layer} layers, d_model={model.config.d_model}")
    
    # Load SAEs for layer 5 (4x expansion, k=30)
    print("Loading layer 5 SAEs (mlp-in and mlp-out)...")
    sae_in = SAE.from_pretrained(
        "tdooms/ts-medium-scope",
        point=("mlp-in", 5),
        expansion=4,
        k=30
    ).to(device)
    
    sae_out = SAE.from_pretrained(
        "tdooms/ts-medium-scope", 
        point=("mlp-out", 5),
        expansion=4,
        k=30
    ).to(device)
    
    n_features_in = sae_in.config.d_features
    n_features_out = sae_out.config.d_features
    print(f"SAE input features: {n_features_in}")
    print(f"SAE output features: {n_features_out}")
    
    return model, sae_in, sae_out


def create_dataloader(model, n_samples: int = 10000, batch_size: int = 32):
    """Create TinyStories dataloader."""
    print(f"Loading TinyStories dataset (n={n_samples})...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    def tokenize(examples):
        return model.tokenizer(
            examples["text"],
            truncation=True,
            max_length=model.config.n_ctx,
            padding="max_length",
            return_tensors="pt",
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def compute_and_scores(model, sae_in, sae_out, dataloader, device, n_features_to_scan: int = None, layer: int = 5):
    """
    Compute AND-scores for output features to find strongest AND-gate circuits.
    
    AND-score = |λ1| / (|λ1| + |λ2|) * sign(λ1) != sign(λ2)
    """
    print("\nComputing AND-scores for output features...")
    
    n_features_out = sae_out.config.d_features
    if n_features_to_scan is None:
        n_features_to_scan = n_features_out
    n_features_to_scan = min(n_features_to_scan, n_features_out)
    
    # Collect SAE activations using hooks
    print("Collecting SAE activations...")
    z_in_all = []
    z_out_all = []
    
    # Hook to capture MLP input and output
    mlp_cache = {}
    
    def capture_mlp(module, input, output):
        mlp_cache["input"] = input[0]  # Input to MLP
        mlp_cache["output"] = output   # Output from MLP
    
    # Register hook on layer 5 MLP
    target_mlp = model.transformer.h[layer].mlp
    hook = target_mlp.register_forward_hook(capture_mlp)
    
    n_batches = min(50, len(dataloader))  # Limit for memory
    
    model.eval()
    
    try:
        for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc="Collecting activations")):
            if i >= n_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            
            # Forward pass to capture MLP input/output via hook
            _ = model(input_ids)
            
            mlp_in = mlp_cache["input"]   # [batch, seq, d_model]
            mlp_out = mlp_cache["output"]  # [batch, seq, d_model]
            
            # Flatten to [batch*seq, d_model]
            mlp_in_flat = mlp_in.reshape(-1, mlp_in.shape[-1])
            mlp_out_flat = mlp_out.reshape(-1, mlp_out.shape[-1])
            
            # Encode through SAEs
            _, z_in = sae_in(mlp_in_flat)  # [batch*seq, n_features]
            _, z_out = sae_out(mlp_out_flat)  # [batch*seq, n_features]
            
            # Sample to avoid OOM
            if z_in.shape[0] > 1000:
                idx = torch.randperm(z_in.shape[0])[:1000]
                z_in = z_in[idx]
                z_out = z_out[idx]
            
            z_in_all.append(z_in.cpu())
            z_out_all.append(z_out.cpu())
    finally:
        hook.remove()
    
    z_in_all = torch.cat(z_in_all, dim=0)
    z_out_all = torch.cat(z_out_all, dim=0)
    
    print(f"Total samples: {z_in_all.shape[0]}")
    
    # Compute AND-scores for each output feature
    and_scores = []
    
    print(f"Computing AND-scores for {n_features_to_scan} features...")
    for feat_idx in tqdm(range(n_features_to_scan), desc="AND-scores"):
        z_out_feat = z_out_all[:, feat_idx]
        
        # Skip features with low activation
        active_mask = z_out_feat > 0
        n_active = active_mask.sum().item()
        if n_active < 50:
            and_scores.append({"feature": feat_idx, "and_score": 0, "n_active": n_active})
            continue
        
        # Compute weighted interaction: Q_ij = sum_t z_in[t,i] * z_in[t,j] * z_out[t]
        z_in_active = z_in_all[active_mask]
        z_out_active = z_out_feat[active_mask]
        
        # Weighted outer product (simplified)
        weighted = z_in_active * z_out_active.unsqueeze(1)  # [n_active, n_in]
        Q = weighted.T @ z_in_active  # [n_in, n_in]
        Q = (Q + Q.T) / 2  # Symmetrize
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Q.float())
            # Sort by magnitude
            idx_sorted = eigenvalues.abs().argsort(descending=True)
            eigenvalues = eigenvalues[idx_sorted]
            
            λ1, λ2 = eigenvalues[0].item(), eigenvalues[1].item()
            
            # AND-score: requires opposing signs
            if (λ1 > 0 and λ2 < 0) or (λ1 < 0 and λ2 > 0):
                and_score = abs(λ1) / (abs(λ1) + abs(λ2))
            else:
                and_score = 0
            
            and_scores.append({
                "feature": feat_idx,
                "and_score": and_score,
                "lambda1": λ1,
                "lambda2": λ2,
                "n_active": n_active
            })
        except Exception as e:
            and_scores.append({"feature": feat_idx, "and_score": 0, "n_active": n_active, "error": str(e)})
    
    # Sort by AND-score
    and_scores.sort(key=lambda x: x["and_score"], reverse=True)
    
    return and_scores, z_in_all, z_out_all


def analyze_top_feature(
    feat_idx: int,
    model,
    sae_in,
    sae_out,
    z_in_all: torch.Tensor,
    z_out_all: torch.Tensor,
    device: str,
    top_k_input: int = 15
):
    """Analyze a specific output feature for AND-gate structure."""
    print(f"\nAnalyzing feature {feat_idx}...")
    
    z_out_feat = z_out_all[:, feat_idx]
    active_mask = z_out_feat > 0
    
    z_in_active = z_in_all[active_mask]
    z_out_active = z_out_feat[active_mask]
    
    # Compute Q matrix
    weighted = z_in_active * z_out_active.unsqueeze(1)
    Q = weighted.T @ z_in_active
    Q = (Q + Q.T) / 2
    
    # Find top-k input features by interaction strength
    interaction_strength = Q.abs().sum(dim=1)
    top_k_idx = interaction_strength.argsort(descending=True)[:top_k_input]
    top_features = top_k_idx.tolist()
    
    # Extract submatrix
    Q_sub = Q[top_k_idx][:, top_k_idx]
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Q.float())
    idx_sorted = eigenvalues.abs().argsort(descending=True)
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors = eigenvectors[:, idx_sorted]
    
    # Project top features onto top eigenvectors
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    
    projections = {}
    for fid in top_features:
        projections[fid] = (v1[fid].item(), v2[fid].item())
    
    # Cluster by v1 projection sign
    cluster_pos = [fid for fid in top_features if projections[fid][0] > 0]
    cluster_neg = [fid for fid in top_features if projections[fid][0] < 0]
    
    # Compute rank-2 correlation
    z_pred = (z_in_all @ v1) ** 2 * eigenvalues[0].item() + \
             (z_in_all @ v2) ** 2 * eigenvalues[1].item()
    
    # Correlation on active samples
    z_true_active = z_out_feat[active_mask].numpy()
    z_pred_active = z_pred[active_mask].numpy()
    
    corr = np.corrcoef(z_true_active, z_pred_active)[0, 1]
    
    return {
        "feature": feat_idx,
        "Q_submatrix": Q_sub.cpu().numpy().tolist(),
        "top_input_features": top_features,
        "eigenvalues": eigenvalues[:10].tolist(),
        "feature_projections": {str(k): list(v) for k, v in projections.items()},
        "cluster_pos": cluster_pos,
        "cluster_neg": cluster_neg,
        "correlation": float(corr) if not np.isnan(corr) else 0.0,
        "n_active": int(active_mask.sum()),
        "z_true": z_true_active[:2000].tolist(),
        "z_pred": z_pred_active[:2000].tolist(),
    }


def generate_figure(analysis_data: dict, output_path: Path):
    """Generate Figure 8-style visualization for ts-medium layer 5."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    feat_idx = analysis_data["feature"]
    Q = np.array(analysis_data["Q_submatrix"])
    projs = analysis_data["feature_projections"]
    eigenvalues = analysis_data["eigenvalues"]
    cluster_pos = analysis_data["cluster_pos"]
    cluster_neg = analysis_data["cluster_neg"]
    corr = analysis_data["correlation"]
    
    # Panel A: Interaction Submatrix
    ax = axes[0]
    vmax = np.abs(Q).max()
    im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    feature_ids = analysis_data["top_input_features"]
    n_feats = len(feature_ids)
    ax.set_xticks(range(n_feats))
    ax.set_yticks(range(n_feats))
    ax.set_xticklabels(feature_ids, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_ids, fontsize=7)
    
    # Color by cluster
    for i, fid in enumerate(feature_ids):
        color = '#ff7f0e' if fid in cluster_pos else '#1f77b4' if fid in cluster_neg else '#7f7f7f'
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)
    
    ax.set_title(f"ts-medium Layer 5, Feature {feat_idx}\nA) Interaction Submatrix", fontweight='bold')
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Input Feature")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel B: Eigenvector Projections
    ax = axes[1]
    
    for feat_str, (v1, v2) in projs.items():
        fid = int(feat_str)
        if fid in cluster_pos:
            marker, color, size = '^', '#ff7f0e', 100
        elif fid in cluster_neg:
            marker, color, size = 'o', '#1f77b4', 80
        else:
            marker, color, size = 's', '#7f7f7f', 60
        
        ax.scatter(v1, v2, marker=marker, c=color, s=size, alpha=0.85,
                   edgecolors='black', linewidths=0.5, zorder=5)
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Directional annotations
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
    ax.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))
    
    ax.set_xlabel("v1 (dominant eigenvector)", fontsize=9)
    ax.set_ylabel("v2 (second eigenvector)", fontsize=9)
    
    λ1, λ2 = eigenvalues[0], eigenvalues[1]
    ax.set_title(f"B) Eigenvector Projections\nλ1={λ1:.3f}, λ2={λ2:.3f}", fontweight='bold')
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e',
               markersize=10, markeredgecolor='black', label=f'Cluster + ({len(cluster_pos)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=10, markeredgecolor='black', label=f'Cluster - ({len(cluster_neg)})'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    # Panel C: Correlation Scatter
    ax = axes[2]
    
    z_true = np.array(analysis_data["z_true"])
    z_pred = np.array(analysis_data["z_pred"])
    
    if len(z_true) > 0:
        n_plot = min(2000, len(z_true))
        idx = np.random.choice(len(z_true), n_plot, replace=False)
        ax.scatter(z_true[idx], z_pred[idx], alpha=0.25, s=6, c='#2E86AB', edgecolors='none')
        
        max_val = max(z_true.max(), z_pred.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(min(z_pred.min() * 1.1, 0), max_val)
    
    ax.set_xlabel("True SAE Activation", fontsize=9)
    ax.set_ylabel("Rank-2 Prediction", fontsize=9)
    ax.set_title(f"C) Correlation: r = {corr:.3f}", fontweight='bold')
    
    plt.suptitle(f"ts-medium Layer 5 AND-gate Circuit Analysis (Feature {feat_idx})",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ts-medium layer 5 AND-gate analysis")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--n-features", type=int, default=500, help="Features to scan for AND-scores")
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS fallbacks enabled")
    
    # Output paths
    results_dir = LANGUAGE_RESULTS
    figure_dir = LANGUAGE_FIGURES
    report_fig_dir = PROJECT_ROOT / "Report/figures/language"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and SAEs
    model, sae_in, sae_out = load_ts_medium_with_saes(device)
    
    # Create dataloader
    dataloader = create_dataloader(model, n_samples=args.n_samples)
    
    # Compute AND-scores
    and_scores, z_in_all, z_out_all = compute_and_scores(
        model, sae_in, sae_out, dataloader, device,
        n_features_to_scan=args.n_features
    )
    
    # Save AND-scores
    and_scores_file = results_dir / "ts_medium_layer5_and_scores.json"
    with open(and_scores_file, "w") as f:
        json.dump({
            "model": "tdooms/ts-medium",
            "layer": 5,
            "expansion": 4,
            "top_by_and_score": and_scores[:20],
        }, f, indent=2)
    print(f"\nSaved AND-scores: {and_scores_file}")
    
    # Print top AND-gate features
    print("\n" + "="*60)
    print("TOP 10 AND-GATE FEATURES (ts-medium layer 5)")
    print("="*60)
    for i, item in enumerate(and_scores[:10]):
        print(f"{i+1}. Feature {item['feature']}: AND-score={item['and_score']:.3f}, "
              f"λ1={item.get('lambda1', 0):.3f}, λ2={item.get('lambda2', 0):.3f}")
    
    # Analyze top feature
    if and_scores and and_scores[0]["and_score"] > 0:
        best_feat = and_scores[0]["feature"]
        analysis = analyze_top_feature(
            best_feat, model, sae_in, sae_out, z_in_all, z_out_all, device
        )
        
        # Save analysis
        analysis_file = results_dir / f"ts_medium_layer5_circuit_{best_feat}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis: {analysis_file}")
        
        # Generate figure
        fig_path = figure_dir / f"figure_8_ts_medium_layer5_{best_feat}.pdf"
        generate_figure(analysis, fig_path)
        
        # Copy to report
        report_path = report_fig_dir / f"figure_8_ts_medium_layer5_{best_feat}.pdf"
        generate_figure(analysis, report_path)
        
        print("\n" + "="*60)
        print("BEST AND-GATE CIRCUIT SUMMARY")
        print("="*60)
        print(f"Feature: {best_feat}")
        print(f"AND-score: {and_scores[0]['and_score']:.3f}")
        print(f"λ1 = {analysis['eigenvalues'][0]:.4f}")
        print(f"λ2 = {analysis['eigenvalues'][1]:.4f}")
        print(f"Cluster +: {len(analysis['cluster_pos'])} features")
        print(f"Cluster -: {len(analysis['cluster_neg'])} features")
        print(f"Rank-2 correlation: {analysis['correlation']:.3f}")
    else:
        print("\nNo strong AND-gate circuits found!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
