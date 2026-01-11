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
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm

# Add original code to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORIG_PATH = PROJECT_ROOT / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))

from language.transformer import Transformer
from sae.tracer import Tracer
from sae.functions import compute_effective_rank, compute_truncated_eigenvalues
from codecarbon import EmissionsTracker


def rank_k_approximation_correlation(Q: torch.Tensor, k: int = 2) -> float:
    """
    Compute correlation between Q and its rank-k approximation.

    Q_hat_k = sum_{i=1}^k lambda_i * v_i * v_i^T

    Paper claims 69% of features have >0.75 correlation with k=2.
    """
    # Symmetrize Q
    Q_sym = 0.5 * (Q + Q.T)

    # Eigendecompose
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(Q_sym)
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


def analyze_interactions_batch(tracer: Tracer, feature_indices: list, rank_k: int = 2, project: bool = False):
    """
    Analyze interaction matrices for multiple output features.

    Uses Tracer.q() to compute Q matrices (from original paper code).
    """
    results = {
        "correlations": [],
        "effective_ranks": [],
        "truncated_eigenvalues": [],
        "feature_indices": [],
    }

    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        try:
            # Get Q matrix using original Tracer code
            Q = tracer.q(feat_idx, project=project)

            if Q is None or Q.numel() == 0:
                continue

            # Move to CPU for analysis
            Q = Q.cpu()

            # Compute rank-k correlation
            corr = rank_k_approximation_correlation(Q, k=rank_k)

            # Compute effective rank using original paper's function
            Q_sym = 0.5 * (Q + Q.T)
            eff_rank = compute_effective_rank(Q_sym.unsqueeze(0)).item()

            # Compute truncated eigenvalues
            trunc_eig = compute_truncated_eigenvalues(Q_sym.unsqueeze(0), k=rank_k).item()

            if not np.isnan(corr):
                results["correlations"].append(corr)
                results["effective_ranks"].append(eff_rank)
                results["truncated_eigenvalues"].append(trunc_eig)
                results["feature_indices"].append(feat_idx)

        except Exception as e:
            print(f"Error analyzing feature {feat_idx}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Interaction matrix analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/interaction_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Start tracking
    tracker = EmissionsTracker(project_name="fact-bilinear-language", log_level="warning")
    tracker.start()
    start_time = time.time()

    # Load model
    model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
    print(f"Loading model: {model_name}")
    model = Transformer.from_pretrained(model_name, device=device)

    # SAE configuration for Tracer
    sae_config = config.get("sae", {})
    layer = sae_config.get("layer", 2)
    inp_config = sae_config.get("input", {"name": "mlp-in", "expansion": 4, "k": 30})
    out_config = sae_config.get("output", {"name": "mlp-out", "expansion": 4, "k": 30})

    # Create Tracer using original paper code
    # Tracer auto-loads pretrained SAEs from {model.config.repo}-scope
    print(f"Creating Tracer for layer {layer}...")
    print(f"  Input SAE: {inp_config}")
    print(f"  Output SAE: {out_config}")

    tracer = Tracer(model, layer, out=out_config, inp=inp_config, device=device)

    # Analysis configuration
    analysis_config = config.get("analysis", {})
    n_features = analysis_config.get("n_features", 500)
    rank_k = analysis_config.get("rank_k", 2)
    project = analysis_config.get("project", False)  # Project onto SAE latents

    # Get number of output features
    n_out_features = tracer.out.d_features
    print(f"Total output features: {n_out_features}")

    # Select features to analyze
    feature_indices = list(range(min(n_features, n_out_features)))
    print(f"Analyzing {len(feature_indices)} features...")

    # Run analysis
    results = analyze_interactions_batch(tracer, feature_indices, rank_k=rank_k, project=project)

    # Compute summary statistics
    correlations = np.array(results["correlations"])
    effective_ranks = np.array(results["effective_ranks"])

    if len(correlations) == 0:
        print("ERROR: No features successfully analyzed")
        return

    fraction_above_075 = (correlations > 0.75).mean()
    fraction_above_050 = (correlations > 0.50).mean()

    summary = {
        "n_analyzed": len(correlations),
        "n_total_features": n_out_features,
        "rank_k": rank_k,
        "fraction_above_075": float(fraction_above_075),
        "fraction_above_050": float(fraction_above_050),
        "mean_correlation": float(correlations.mean()),
        "std_correlation": float(correlations.std()),
        "median_correlation": float(np.median(correlations)),
        "mean_effective_rank": float(effective_ranks.mean()),
        "std_effective_rank": float(effective_ranks.std()),
        "paper_claim": "69% of features have >0.75 rank-2 correlation",
        "our_result": f"{fraction_above_075*100:.1f}% of features have >0.75 rank-{rank_k} correlation",
        "claim_supported": fraction_above_075 > 0.60,  # Allow 9% margin
    }

    print(f"\n{'='*60}")
    print(f"INTERACTION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Features analyzed: {summary['n_analyzed']}")
    print(f"Fraction with >0.75 correlation: {summary['fraction_above_075']*100:.1f}%")
    print(f"Fraction with >0.50 correlation: {summary['fraction_above_050']*100:.1f}%")
    print(f"Mean correlation: {summary['mean_correlation']:.4f}")
    print(f"Mean effective rank: {summary['mean_effective_rank']:.2f}")
    print(f"\nPaper claim: {summary['paper_claim']}")
    print(f"Our result: {summary['our_result']}")
    print(f"Claim supported: {summary['claim_supported']}")
    print(f"{'='*60}")

    # Stop tracking
    end_time = time.time()
    emissions_kg = tracker.stop()

    summary["wall_time_seconds"] = end_time - start_time
    summary["co2_kg"] = emissions_kg

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "summary": summary,
        "per_feature": {
            "feature_indices": results["feature_indices"],
            "correlations": results["correlations"],
            "effective_ranks": results["effective_ranks"],
        },
        "config": config,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
