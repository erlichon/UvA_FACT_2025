"""
Interaction matrix analysis using the original paper's Tracer class.

For each SAE output feature, Tracer computes Q matrices representing
how input feature pairs interact to produce the output feature.

Paper claim: 69% of features have >0.75 correlation with rank-2 approximation.

Usage:
    python src/language/interaction_analysis.py --config configs/language_interaction.yaml
"""

import sys
from pathlib import Path
import argparse
import time
import json
import warnings

# Suppress torchvision image extension warning (libjpeg not needed for our use case)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import numpy as np
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from sae.tracer import Tracer
from sae.functions import compute_effective_rank, compute_truncated_eigenvalues

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    setup_mps_fallbacks,
    is_mps_device,
    safe_eigh,
    init_wandb,
    finish_wandb,
)


def rank_k_approximation_correlation(Q: torch.Tensor, k: int = 2) -> float:
    """
    Compute correlation between Q and its rank-k approximation.

    Q_hat_k = sum_{i=1}^k lambda_i * v_i * v_i^T

    Paper claims 69% of features have >0.75 correlation with k=2.
    """
    # Symmetrize Q
    Q_sym = 0.5 * (Q + Q.T)

    # Eigendecompose (MPS-safe)
    try:
        eigenvalues, eigenvectors = safe_eigh(Q_sym)
    except Exception:
        return float('nan')

    # Sort by absolute magnitude (descending)
    order = eigenvalues.abs().argsort(descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Rank-k approximation
    Q_hat = torch.zeros_like(Q)
    for i in range(min(k, len(eigenvalues))):
        v = eigenvectors[:, i:i+1]
        Q_hat = Q_hat + eigenvalues[i] * (v @ v.T)

    # Pearson correlation
    Q_flat = Q_sym.flatten()
    Q_hat_flat = Q_hat.flatten()

    Q_centered = Q_flat - Q_flat.mean()
    Q_hat_centered = Q_hat_flat - Q_hat_flat.mean()

    numerator = (Q_centered * Q_hat_centered).sum()
    denominator = torch.sqrt((Q_centered ** 2).sum() * (Q_hat_centered ** 2).sum())

    if denominator < 1e-10:
        return float('nan')

    return (numerator / denominator).item()


def analyze_single_feature_original(tracer: Tracer, feat_idx: int, rank_k: int = 2, project: bool = False) -> dict:
    """
    Analyze a single feature using the original tracer.q() implementation.

    Processes one feature at a time to manage memory.
    """
    # Use original tracer.q() - returns Q on the model's device
    Q = tracer.q(feat_idx, project=project)

    # Move to CPU immediately
    Q_cpu = Q.float().cpu()
    del Q

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Symmetrize for eigendecomposition
    Q_sym = 0.5 * (Q_cpu + Q_cpu.T)

    # Compute metrics
    corr = rank_k_approximation_correlation(Q_cpu, k=rank_k)
    eff_rank = compute_effective_rank(Q_sym.unsqueeze(0)).item()
    trunc_eig = compute_truncated_eigenvalues(Q_sym.unsqueeze(0), k=rank_k).item()

    del Q_cpu, Q_sym

    return {
        "correlation": corr,
        "effective_rank": eff_rank,
        "truncated_eigenvalue": trunc_eig,
    }


def analyze_single_feature_manual(tracer: Tracer, feat_idx: int, rank_k: int = 2, project: bool = False) -> dict:
    """
    Analyze a single feature's interaction matrix using manual einsum decomposition.

    Memory-efficient CPU-based implementation.
    """
    model, layer = tracer.model, tracer.layer
    out_latent = tracer.out_latents[feat_idx]  # Single feature vector

    # Compute Q matrix using manual decomposition
    # Original: res = einsum("mi,mj,om,...o->ij", w_l, w_r, w_p, out_latent)

    w_l = model.w_l[layer].float().cpu()  # [m, i]
    w_r = model.w_r[layer].float().cpu()  # [m, j]
    w_p = model.w_p[layer].float().cpu()  # [o, m]
    out_vec = out_latent.float().cpu()    # [o]

    # Compute projection: [o] @ [o, m] -> [m]
    proj = out_vec @ w_p  # [m]

    # Scale w_l by projection
    scaled_l = proj.unsqueeze(1) * w_l  # [m, i]

    # Q[i,j] = sum_m scaled_l[m,i] * w_r[m,j]
    Q = scaled_l.T @ w_r  # [i, j]

    # Clean up intermediates
    del w_l, w_r, w_p, out_vec, proj, scaled_l

    if project:
        inp_latents = tracer.inp_latents.float().cpu()
        Q = inp_latents.T @ Q @ inp_latents
        Q = 0.5 * (Q + Q.T)
        del inp_latents

    # Symmetrize for eigendecomposition
    Q_sym = 0.5 * (Q + Q.T)

    # Compute metrics
    corr = rank_k_approximation_correlation(Q, k=rank_k)
    eff_rank = compute_effective_rank(Q_sym.unsqueeze(0)).item()
    trunc_eig = compute_truncated_eigenvalues(Q_sym.unsqueeze(0), k=rank_k).item()

    del Q, Q_sym

    return {
        "correlation": corr,
        "effective_rank": eff_rank,
        "truncated_eigenvalue": trunc_eig,
    }


# Use the original implementation for project=False, manual for project=True
def analyze_single_feature(tracer: Tracer, feat_idx: int, rank_k: int = 2, project: bool = False) -> dict:
    """
    Wrapper that selects the appropriate implementation.

    For project=False: uses original tracer.q() for correctness
    For project=True: uses manual implementation to avoid OOM from einsum
        The original einsum "il,jk,...ij->...lk" creates a ~281TB intermediate tensor
        when inp_latents is [1024, 8192]. Manual implementation uses matrix multiplication
        (inp_latents.T @ Q @ inp_latents) which is memory-efficient.
    """
    if project:
        # Use manual implementation to avoid OOM from original einsum
        return analyze_single_feature_manual(tracer, feat_idx, rank_k, project)
    else:
        return analyze_single_feature_original(tracer, feat_idx, rank_k, project)


def analyze_interactions_batch(tracer: Tracer, feature_indices: list, rank_k: int = 2, project: bool = False, device: str = "cuda", memory_efficient: bool = True):
    """
    Analyze interaction matrices for multiple output features.

    Uses Tracer.q() to compute Q matrices (from original paper code).
    Memory-efficient mode: computes on CPU to avoid GPU OOM.
    """
    results = {
        "correlations": [],
        "effective_ranks": [],
        "truncated_eigenvalues": [],
        "feature_indices": [],
    }

    is_cuda = device.startswith("cuda") or device == "cuda"

    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        try:
            if memory_efficient:
                # CPU-based analysis to avoid GPU OOM
                metrics = analyze_single_feature(tracer, feat_idx, rank_k=rank_k, project=project)
                corr = metrics["correlation"]
                eff_rank = metrics["effective_rank"]
                trunc_eig = metrics["truncated_eigenvalue"]
            else:
                # Original GPU-based analysis
                Q = tracer.q(feat_idx, project=project)

                if Q is None or Q.numel() == 0:
                    continue

                # Move to CPU immediately to free GPU memory
                Q_cpu = Q.cpu()
                del Q  # Explicitly delete GPU tensor

                # Clear GPU cache
                if is_cuda:
                    torch.cuda.empty_cache()

                # Compute rank-k correlation
                corr = rank_k_approximation_correlation(Q_cpu, k=rank_k)

                # Compute effective rank using original paper's function
                Q_sym = 0.5 * (Q_cpu + Q_cpu.T)
                eff_rank = compute_effective_rank(Q_sym.unsqueeze(0)).item()

                # Compute truncated eigenvalues
                trunc_eig = compute_truncated_eigenvalues(Q_sym.unsqueeze(0), k=rank_k).item()

                # Clean up CPU tensors
                del Q_cpu, Q_sym

            if not np.isnan(corr):
                results["correlations"].append(corr)
                results["effective_ranks"].append(eff_rank)
                results["truncated_eigenvalues"].append(trunc_eig)
                results["feature_indices"].append(feat_idx)

        except Exception as e:
            print(f"Error analyzing feature {feat_idx}: {e}")
            # Still try to clear memory on error
            if is_cuda:
                torch.cuda.empty_cache()
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Interaction matrix analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/interaction_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--memory-efficient", action="store_true", default=True,
                        help="Use memory-efficient CPU-based computation (default: True)")
    parser.add_argument("--no-memory-efficient", action="store_false", dest="memory_efficient",
                        help="Use original GPU-based computation (may OOM on large models)")
    parser.add_argument("--project", action="store_true", default=None,
                        help="Project Q onto SAE latents (overrides config)")
    parser.add_argument("--no-project", action="store_false", dest="project",
                        help="Do not project Q onto SAE latents (overrides config)")
    args = parser.parse_args()

    # Auto-detect device (includes MPS support)
    device = get_device(args.device)

    # Force CPU for interaction analysis - MPS has bugs with einsum operations
    if is_mps_device(device):
        print(f"MPS detected but forcing CPU for interaction analysis (MPS einsum bugs)")
        device = "cpu"
    else:
        print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=config_name,
        config=config,
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "interaction"],
    )

    # Run analysis with emissions tracking
    with track_emissions("fact-bilinear") as tracker:
        # Load model
        model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
        print(f"Loading model: {model_name}")
        model = Transformer.from_pretrained(model_name, device=device)

        # Print model config for debugging
        print(f"\nModel config:")
        print(f"  d_model: {model.config.d_model}")
        print(f"  d_hidden: {model.config.d_hidden}")
        print(f"  n_layer: {model.config.n_layer}")

        # SAE configuration for Tracer
        sae_config = config.get("sae", {})
        layer = sae_config.get("layer", 2)
        inp_config = sae_config.get("input", {"name": "mlp-in", "expansion": 4, "k": 30})
        out_config = sae_config.get("output", {"name": "mlp-out", "expansion": 4, "k": 30})

        # Create Tracer using original paper code
        # Tracer auto-loads pretrained SAEs from {model.config.repo}-scope
        print(f"\nCreating Tracer for layer {layer}...")
        print(f"  Input SAE: {inp_config}")
        print(f"  Output SAE: {out_config}")

        tracer = Tracer(model, layer, out=out_config, inp=inp_config, device=device)

        # Print tracer info for debugging
        print(f"\nTracer info:")
        print(f"  out_latents shape: {tracer.out_latents.shape}")
        print(f"  inp_latents shape: {tracer.inp_latents.shape}")
        print(f"  out SAE d_features: {tracer.out.d_features}")
        print(f"  out SAE d_model: {tracer.out.d_model}")

        # Analysis configuration
        analysis_config = config.get("analysis", {})
        n_features = analysis_config.get("n_features", 500)
        rank_k = analysis_config.get("rank_k", 2)

        # Use command line override if provided, otherwise use config
        if args.project is not None:
            project = args.project
        else:
            project = analysis_config.get("project", False)

        # Get number of output features
        n_out_features = tracer.out.d_features
        print(f"\nTotal output features: {n_out_features}")

        # Select features to analyze
        feature_indices = list(range(min(n_features, n_out_features)))
        print(f"Analyzing {len(feature_indices)} features...")
        print(f"Project onto SAE latents: {project}")
        print(f"Memory-efficient mode: {args.memory_efficient}")

        # Debug: analyze first feature and print Q matrix info
        print(f"\n--- Diagnostic: First feature (idx=0) ---")
        # Always compute Q without projection first (to avoid OOM from original einsum)
        Q_test = tracer.q(0, project=False)
        print(f"Q matrix (unprojected) shape: {Q_test.shape}")
        print(f"Q matrix dtype: {Q_test.dtype}")
        print(f"Q matrix stats: min={Q_test.min():.6f}, max={Q_test.max():.6f}, mean={Q_test.mean():.6f}")

        Q_test_cpu = Q_test.float().cpu()
        del Q_test

        # Apply projection manually if requested (memory-efficient)
        if project:
            inp_latents = tracer.inp_latents.float().cpu()
            Q_test_cpu = inp_latents.T @ Q_test_cpu @ inp_latents
            Q_test_cpu = 0.5 * (Q_test_cpu + Q_test_cpu.T)
            del inp_latents
            print(f"Q matrix (projected) shape: {Q_test_cpu.shape}")

        Q_test_sym = 0.5 * (Q_test_cpu + Q_test_cpu.T)
        test_corr = rank_k_approximation_correlation(Q_test_cpu, k=rank_k)
        test_eff_rank = compute_effective_rank(Q_test_sym.unsqueeze(0)).item()
        print(f"Rank-{rank_k} correlation: {test_corr:.6f}")
        print(f"Effective rank: {test_eff_rank:.2f}")
        del Q_test_cpu, Q_test_sym
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--- End diagnostic ---\n")

        # Run analysis
        results = analyze_interactions_batch(
            tracer, feature_indices, rank_k=rank_k, project=project,
            device=device, memory_efficient=args.memory_efficient
        )

    # Compute summary statistics
    correlations = np.array(results["correlations"])
    effective_ranks = np.array(results["effective_ranks"])

    if len(correlations) == 0:
        print("ERROR: No features successfully analyzed")
        if wandb_enabled:
            finish_wandb(tracker.result, extra_summary={"error": "no_features_analyzed"})
        return

    fraction_above_075 = (correlations > 0.75).mean()
    fraction_above_050 = (correlations > 0.50).mean()

    summary = {
        "n_analyzed": len(correlations),
        "n_total_features": n_out_features,
        "rank_k": rank_k,
        "project_onto_sae_latents": project,
        "model_name": model_name,
        "layer": layer,
        "fraction_above_075": float(fraction_above_075),
        "fraction_above_050": float(fraction_above_050),
        "mean_correlation": float(correlations.mean()),
        "std_correlation": float(correlations.std()),
        "median_correlation": float(np.median(correlations)),
        "mean_effective_rank": float(effective_ranks.mean()),
        "std_effective_rank": float(effective_ranks.std()),
        "paper_claim": "69% of features have >0.75 rank-2 correlation",
        "our_result": f"{fraction_above_075*100:.1f}% of features have >0.75 rank-{rank_k} correlation",
        "claim_supported": bool(fraction_above_075 > 0.60),  # Allow 9% margin
        "wall_time_seconds": tracker.result.wall_time_seconds,
        "co2_kg": tracker.result.emissions_kg,
    }

    print(f"\n{'='*60}")
    print(f"INTERACTION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {layer}")
    print(f"Project onto SAE latents: {project}")
    print(f"Features analyzed: {summary['n_analyzed']}")
    print(f"Fraction with >0.75 correlation: {summary['fraction_above_075']*100:.1f}%")
    print(f"Fraction with >0.50 correlation: {summary['fraction_above_050']*100:.1f}%")
    print(f"Mean correlation: {summary['mean_correlation']:.4f}")
    print(f"Mean effective rank: {summary['mean_effective_rank']:.2f}")
    print(f"\nPaper claim: {summary['paper_claim']}")
    print(f"Our result: {summary['our_result']}")
    print(f"Claim supported: {summary['claim_supported']}")
    print(f"{'='*60}")

    # Finalize wandb with results
    if wandb_enabled:
        finish_wandb(tracker.result, extra_summary={
            "n_features": n_features,
            "fraction_above_075": fraction_above_075,
            "fraction_above_050": fraction_above_050,
            "mean_correlation": summary["mean_correlation"],
            "mean_effective_rank": summary["mean_effective_rank"],
            "claim_supported": summary["claim_supported"],
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    full_results = {
        "summary": summary,
        "per_feature": {
            "feature_indices": [int(x) for x in results["feature_indices"]],
            "correlations": [float(x) for x in results["correlations"]],
            "effective_ranks": [float(x) for x in results["effective_ranks"]],
        },
        "config": config,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
