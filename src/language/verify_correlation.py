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
import matplotlib.pyplot as plt

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
from src.language.context import LanguageContext


def plot_correlation_vs_rank(summary: dict, output_path: str):
    """
    Generate a plot of Average Correlation vs Rank.
    
    Args:
        summary: Summary dict containing correlation_by_rank
        output_path: Path to save the PNG plot
    """
    ranks = []
    means = []
    stds = []
    
    for rank_str, stats in sorted(summary["correlation_by_rank"].items(), key=lambda x: int(x[0])):
        ranks.append(int(rank_str))
        means.append(stats["mean"])
        stds.append(stats["std"])
    
    ranks = np.array(ranks)
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot with error bars
    ax.errorbar(ranks, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                linewidth=2, markersize=8, color='#2E86AB', ecolor='#A23B72')
    
    # Reference line at 0.75 (paper claim)
    ax.axhline(y=0.75, color='#F18F01', linestyle='--', linewidth=1.5, 
               label='Paper claim (75%)')
    
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Weight-Based Prediction vs True SAE Activation\n(Figure 9 Reproduction)', fontsize=14)
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add annotation for rank-2
    if 2 in ranks:
        idx = list(ranks).index(2)
        ax.annotate(f'Rank-2: {means[idx]:.3f}', 
                    xy=(2, means[idx]), xytext=(2.5, means[idx] - 0.15),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


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


def create_validation_dataloader(tokenizer, config: dict, device: str, n_samples: int = 2000, batch_size: int = 32):
    """
    Create a DataLoader for validation data from TinyStories dataset.
    
    Args:
        tokenizer: Model tokenizer
        config: Experiment config
        device: Device to use
        n_samples: Number of samples to load
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader yielding batches with input_ids and attention_mask
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
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    print(f"  Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader


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
    dataloader,
    feature_indices: list,
    ranks: list,
    device: str,
    min_active_per_feature: int = 50,
    target_samples_per_feature: int = 500,
    max_batches: int = 50,
    max_features: int = 50,
    save_scatter: bool = False,
    max_scatter_samples: int = 1000,
    scatter_dir: Path = None,
    model_name: str = "unknown",
):
    """
    Compute correlation between weight-based predictions and actual SAE activations.
    
    CRITICAL FIX: Accumulates samples across multiple batches to ensure statistical
    significance. The correlation is computed on the full accumulated data, not per-batch.
    
    Memory-efficient scatter: When save_scatter=True, scatter data is saved to individual
    files immediately and freed from memory. This allows processing many features without OOM.
    
    The key equation being verified:
        z_c(x) ≈ sum_{j=1}^k lambda_j * (v_j^T x)^2
    
    Args:
        model: Transformer model
        sae_out: Output SAE (for encoding mlp_out)
        layer: Layer index
        dataloader: DataLoader yielding validation batches
        feature_indices: List of feature indices to analyze (or None for auto-detect)
        ranks: List of ranks to evaluate [1, 2, 4, 8, 16]
        device: Device for computation
        min_active_per_feature: Minimum active samples required per feature
        target_samples_per_feature: Target number of active samples before computing correlation
        max_batches: Maximum number of batches to process
        max_features: Maximum number of features to analyze
        save_scatter: Whether to save scatter data for Figure 9C
        max_scatter_samples: Maximum scatter samples per feature
        scatter_dir: Directory to save scatter files (created if save_scatter=True)
        model_name: Model name for scatter file naming
    
    Returns:
        Dict with correlation results per rank
    """
    results = {k: [] for k in ranks}
    feature_results = []
    scatter_files_saved = []
    
    # Create scatter directory if saving scatter data
    if save_scatter and scatter_dir is not None:
        scatter_dir = Path(scatter_dir)
        scatter_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Scatter data will be saved to: {scatter_dir}")
    
    sight = Sight(model)
    sae_out_device = sae_out.to(device)
    
    print(f"\nAccumulating activations for layer {layer} across batches...")
    print(f"  Target: {target_samples_per_feature} active samples per feature")
    print(f"  Max batches: {max_batches}")
    
    # Accumulate mlp_in and SAE activations across batches
    all_mlp_in = []
    all_z_true = []
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass to capture mlp_in and mlp_out
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            # Flatten batch and sequence: [batch*seq, d_model]
            mlp_in_flat = mlp_in.flatten(0, 1).float()
            mlp_out_flat = mlp_out.flatten(0, 1).float()
            
            # Compute SAE activations
            z_true = sae_out_device.encode(mlp_out_flat)  # [batch*seq, d_features]
            
            # Move to CPU and accumulate
            all_mlp_in.append(mlp_in_flat.cpu())
            all_z_true.append(z_true.cpu())
            total_tokens += mlp_in_flat.shape[0]
        
        # Concatenate all accumulated data
        print(f"\nConcatenating {len(all_mlp_in)} batches ({total_tokens} tokens)...")
        mlp_in_all = torch.cat(all_mlp_in, dim=0)  # [total_tokens, d_model]
        z_true_all = torch.cat(all_z_true, dim=0)  # [total_tokens, d_features]
        
        del all_mlp_in, all_z_true
        
        print(f"  mlp_in_all shape: {mlp_in_all.shape}")
        print(f"  z_true_all shape: {z_true_all.shape}")
        print(f"  z_true_all nonzero: {(z_true_all > 0).sum().item()}")
        
        # Find features with sufficient activations
        print("Finding active features...")
        all_active_features = find_active_features(z_true_all, min_active=min_active_per_feature)
        print(f"  Found {len(all_active_features)} features with >= {min_active_per_feature} active positions")
        
        # Select features to analyze
        if feature_indices:
            active_set = set(all_active_features)
            feature_indices = [f for f in feature_indices if f in active_set]
            if len(feature_indices) == 0:
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
            # Get encoder direction for this feature (as per paper's Tracer default)
            # The SAE activation is z_c = ReLU(mlp_out · w_enc[c]), so we need encoder direction
            out_direction = sae_out_device.w_enc.weight[feat_idx, :].cpu()
            
            # Get eigenpairs from weights (unprojected Q in residual stream space)
            eigenpairs = get_interaction_eigenpairs(
                model=model,
                layer=layer,
                feat_idx=feat_idx,
                out_direction=out_direction,
                device="cpu",
            )
            
            # True activation for this feature across ALL accumulated tokens
            z_true_feat = z_true_all[:, feat_idx]  # [total_tokens]
            
            # Filter for active positions (z > 0)
            active_mask = z_true_feat > 0
            n_active = active_mask.sum().item()
            
            # Skip features with insufficient active samples
            if n_active < min_active_per_feature:
                continue
            
            x_active = mlp_in_all[active_mask]   # [n_active, d_model]
            z_active = z_true_feat[active_mask]  # [n_active]
            
            feature_corrs = {}
            z_pred_rank2 = None  # Store rank-2 predictions for scatter plot
            
            # Compute correlation for each rank
            for k in ranks:
                k_actual = min(k, len(eigenpairs.eigenvalues))
                eigenvalues_k = eigenpairs.eigenvalues[:k_actual]
                eigenvectors_k = eigenpairs.eigenvectors[:, :k_actual]
                
                # Predicted activation: sum_j lambda_j * (v_j . x)^2
                z_pred = predict_activation_from_eigenpairs(
                    x_active, eigenvalues_k, eigenvectors_k
                )
                
                # Store rank-2 predictions for scatter plot
                if k == 2 and save_scatter:
                    z_pred_rank2 = z_pred
                
                # Pearson correlation on FULL accumulated data
                corr = pearson_correlation(z_active, z_pred)
                
                if not np.isnan(corr):
                    results[k].append(corr)
                    feature_corrs[k] = corr
            
            # Build feature result dict
            result_dict = {
                "feat_idx": feat_idx,
                "n_active": n_active,
                "correlations": feature_corrs,
            }
            
            # Save scatter data to file immediately (memory-efficient streaming)
            if save_scatter and z_pred_rank2 is not None and scatter_dir is not None:
                scatter_limit = max_scatter_samples if max_scatter_samples > 0 else len(z_active)
                scatter_limit = min(scatter_limit, len(z_active))
                
                scatter_data = {
                    "feat_idx": feat_idx,
                    "n_active": n_active,
                    "z_true": z_active[:scatter_limit].cpu().tolist(),
                    "z_pred_rank2": z_pred_rank2[:scatter_limit].cpu().tolist(),
                    "correlation_rank2": feature_corrs.get(2, None),
                }
                
                # Save to individual file immediately
                scatter_file = scatter_dir / f"scatter_{model_name}_feat_{feat_idx}.json"
                with open(scatter_file, "w") as f:
                    json.dump(scatter_data, f)
                scatter_files_saved.append(str(scatter_file))
                
                # Note in result that scatter data is in separate file
                result_dict["scatter_file"] = str(scatter_file)
                
                # Free memory
                del scatter_data
            
            feature_results.append(result_dict)
    
    return results, feature_results, scatter_files_saved


def main():
    parser = argparse.ArgumentParser(description="Verify weight-correlation predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/correlation_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    
    # Model/SAE overrides (optional, overrides config values)
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--layer", type=int, default=None, help="SAE layer (overrides config)")
    parser.add_argument("--expansion", type=int, default=None, help="SAE expansion (overrides config)")
    parser.add_argument("--k", type=int, default=None, help="SAE top-k (overrides config)")
    
    # Analysis parameters
    parser.add_argument("--n-features", type=int, default=50, help="Number of features to analyze (-1 for all)")
    parser.add_argument("--ranks", type=str, default="1,2,4,8,16", help="Ranks to evaluate (comma-separated or range like 1-60)")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of validation samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum batches to process")
    parser.add_argument("--min-active", type=int, default=50, help="Minimum active samples per feature")
    parser.add_argument("--target-samples", type=int, default=500, help="Target active samples per feature")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--plot", type=str, default=None, help="Path to save correlation plot (PNG)")
    
    # Figure 9C scatter data options
    parser.add_argument("--save-scatter", action="store_true", help="Save scatter data (z_true, z_pred) for Figure 9C")
    parser.add_argument("--max-scatter-samples", type=int, default=1000, help="Max scatter samples per feature (-1 for all)")
    
    args = parser.parse_args()
    
    # Parse ranks - support both comma-separated and range syntax
    if "-" in args.ranks and "," not in args.ranks:
        # Range syntax: "1-60"
        start, end = map(int, args.ranks.split("-"))
        ranks = list(range(start, end + 1))
    else:
        # Comma-separated syntax: "1,2,4,8,16"
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
    
    # Apply CLI overrides to config
    if args.model is not None:
        config.setdefault("model", {})["pretrained"] = args.model
    if args.layer is not None:
        config.setdefault("sae", {})["layer"] = args.layer
    if args.expansion is not None:
        # Set both flat and nested expansion values
        config.setdefault("sae", {})["expansion"] = args.expansion
        # Also update nested configs if they exist
        if "input" in config.get("sae", {}):
            config["sae"]["input"]["expansion"] = args.expansion
        if "output" in config.get("sae", {}):
            config["sae"]["output"]["expansion"] = args.expansion
    if args.k is not None:
        # Set both flat and nested k values
        config.setdefault("sae", {})["k"] = args.k
        # Also update nested configs if they exist
        if "input" in config.get("sae", {}):
            config["sae"]["input"]["k"] = args.k
        if "output" in config.get("sae", {}):
            config["sae"]["output"]["k"] = args.k
    
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
        # Use LanguageContext for unified model/SAE loading
        ctx = LanguageContext(config, device)
        model = ctx.model
        layer = ctx.layer
        expansion = ctx.expansion
        k = ctx.k
        model_name = ctx.model_name
        
        # Load output SAE via context
        sae_out = ctx.get_sae("mlp-out")
        
        # Create validation DataLoader via context
        dataloader = ctx.get_dataloader(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )
        
        # Select features to analyze
        # Pass None to find active features dynamically, then limit to n_features
        feature_indices = None  # Will be populated by find_active_features in verify_correlation
        
        # Create scatter directory if saving scatter data
        output_path = Path(args.output)
        scatter_dir = None
        if args.save_scatter:
            scatter_dir = output_path.parent / "scatter_data"
            scatter_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract model short name for scatter file naming
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Run correlation analysis with batch accumulation
        results, feature_results, scatter_files = verify_correlation(
            model=model,
            sae_out=sae_out,
            layer=layer,
            dataloader=dataloader,
            feature_indices=feature_indices,
            ranks=ranks,
            device=device,
            min_active_per_feature=args.min_active,
            target_samples_per_feature=args.target_samples,
            max_batches=args.max_batches,
            max_features=args.n_features,
            save_scatter=args.save_scatter,
            max_scatter_samples=args.max_scatter_samples,
            scatter_dir=scatter_dir,
            model_name=model_short,
        )
    
    # Compute summary statistics
    summary = {
        "model_name": model_name,
        "layer": layer,
        "expansion": expansion,
        "k": k,
        "n_features_requested": args.n_features if args.n_features != -1 else "all",
        "n_features_analyzed": len(feature_results),
        "ranks": ranks,
        "correlation_by_rank": {},
        "scatter_data_saved": args.save_scatter,
        "scatter_data_dir": str(scatter_dir) if scatter_dir else None,
        "scatter_files_count": len(scatter_files) if args.save_scatter else 0,
        "max_scatter_samples": args.max_scatter_samples if args.save_scatter else None,
        "paper_claim": "69% of features have >0.75 rank-2 correlation",
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
        # Root-level emissions for validation script compatibility
        "emissions": {
            "co2_kg": tracker.result.emissions_kg,
            "wall_time_hours": tracker.result.wall_time_hours,
            "gpu_hours": tracker.result.gpu_hours,
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    if args.save_scatter and scatter_dir:
        print(f"Scatter data saved to: {scatter_dir} ({len(scatter_files)} files)")
    
    # Generate plot if requested
    if args.plot and len(summary["correlation_by_rank"]) > 0:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_correlation_vs_rank(summary, str(plot_path))


if __name__ == "__main__":
    main()
