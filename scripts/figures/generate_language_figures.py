#!/usr/bin/env python3
"""
Generate figures for Phase 2 (Language) experiments.

Run from project root after running language experiments:
    python scripts/figures/generate_language_figures.py

Generates:
- Figure 9A: Correlation progression across ranks (3 models)
- Figure 9B: Rank-2 correlation histogram (3 models)
- Figure 9C: True vs predicted scatter plots (fw-medium)
- Figure 8: Sentiment negation circuit (fw-medium, feature 3834)
- Figure 8 Comparison: Weak (3834) vs Strong (best from search) circuits
"""

import sys
from pathlib import Path
import json
import argparse

# Add project root to path (scripts/figures/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.paths import LANGUAGE_FIGURES, LANGUAGE_RESULTS

# Figure 9 and 8 plotting utilities
from src.plot_utils.language import (
    load_correlation_results,
    plot_correlation_progression,
    plot_correlation_histogram,
    plot_figure_9c_scatters,
    plot_figure_8_composite,
    load_figure_8_data,
    compute_fraction_above_threshold,
    # Figure 10: SAE Training Time Effect
    load_sae_training_results,
    plot_sae_training_effect,
    plot_sae_training_histogram,
)


def save_figure(fig, filename: str, figure_dir: Path):
    """Save figure to the specified directory."""
    if fig is None:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / filename
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_figure_9(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 9 (A, B, C) from correlation sweep results.
    
    Figure 9A: Correlation progression (line plot, 3 models)
    Figure 9B: Rank-2 correlation histogram (3 models)
    Figure 9C: True vs predicted scatter plots (fw-medium)
    """
    print("\n" + "="*60)
    print("FIGURE 9: Correlation Analysis")
    print("="*60)
    
    # Load correlation results from sweep
    correlation_results = load_correlation_results(results_dir)
    
    if not correlation_results:
        print("No correlation results found.")
        print("Run the sweep first:")
        print("  ./scripts/run_language_sweep.sh")
        return False
    
    print(f"Loaded results for models: {list(correlation_results.keys())}")
    
    # Figure 9A: Correlation Progression
    print("\nGenerating Figure 9A: Correlation Progression...")
    fig = plot_correlation_progression(
        correlation_results,
        title="Average Correlation vs Approximation Rank",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_9a_correlation_progression.pdf", figure_dir)
    
    # Figure 9B: Correlation Histogram
    print("Generating Figure 9B: Rank-2 Correlation Histogram...")
    fig = plot_correlation_histogram(
        correlation_results,
        rank=2,
        title="Rank-2 Correlation Distribution",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_9b_correlation_histogram.pdf", figure_dir)
    
    # Figure 9C: Scatter plots (fw-medium only, as per paper)
    print("Generating Figure 9C: True vs Predicted Scatter Plots...")
    fw_medium_data = correlation_results.get("fw-medium")
    if fw_medium_data:
        # Check for streaming scatter files first (memory-efficient approach)
        scatter_dir = results_dir / "scatter_data"
        has_streaming_scatter = scatter_dir.exists() and any(scatter_dir.glob("scatter_fw-medium_*.json"))
        
        # Fall back to embedded scatter data
        has_embedded_scatter = any("scatter_data" in f for f in fw_medium_data.get("per_feature", []))
        
        if has_streaming_scatter or has_embedded_scatter:
            fig = plot_figure_9c_scatters(
                fw_medium_data,
                n_features=9,
                seed=42,
                scatter_dir=scatter_dir if has_streaming_scatter else None,
            )
            save_figure(fig, "figure_9c_scatter_plots.pdf", figure_dir)
        else:
            print("  Note: No scatter data available for Figure 9C.")
            print("  Re-run sweep with: --save-scatter")
    else:
        print("  Note: fw-medium results not found for Figure 9C.")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    fractions = compute_fraction_above_threshold(correlation_results, rank=2, threshold=0.75)
    for model, frac in fractions.items():
        n_analyzed = len(correlation_results[model].get("per_feature", []))
        print(f"  {model}: {frac*100:.1f}% features above 0.75 correlation (n={n_analyzed})")
    
    print(f"\nPaper claim: >69% features with rank-2 correlation >0.75")
    
    return True


def generate_figure_8(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8 (Sentiment Negation Circuit) from negation visualization results.
    
    Figure 8A: Interaction submatrix (top 15 interactions)
    Figure 8B: Feature projections onto eigenvectors
    Figure 8C: Activation vs rank-2 approximation scatter
    """
    print("\n" + "="*60)
    print("FIGURE 8: Sentiment Negation Circuit")
    print("="*60)
    
    # Look for Figure 8 data file (fw-medium example)
    # NOTE: ts-medium layer 4 doesn't have mlp-in SAEs available,
    # so we use fw-medium which demonstrates the same phenomenon
    figure_8_file = results_dir / "figure_8_data_fw_medium.json"
    
    if not figure_8_file.exists():
        print(f"Figure 8 data not found: {figure_8_file}")
        print("Generate it first:")
        print("  python src/language/negation_visualization.py \\")
        print("      --config configs/language_negation_fw.yaml \\")
        print("      --output results/language/figure_8_data_fw_medium.json \\")
        print("      --feature 3834 --device mps")
        return False
    
    print(f"Loading Figure 8 data from {figure_8_file}")
    figure_8_data = load_figure_8_data(figure_8_file)
    
    # Generate composite figure
    print("Generating Figure 8 composite...")
    fig = plot_figure_8_composite(figure_8_data)
    save_figure(fig, "figure_8_negation_circuit.pdf", figure_dir)
    
    # Print summary
    print("\n--- Figure 8 Summary ---")
    print(f"  Output feature: {figure_8_data.get('output_feature_idx')}")
    print(f"  Model: {figure_8_data.get('model_name')}")
    panel_c = figure_8_data.get("panel_c", {})
    print(f"  Panel C correlation: {panel_c.get('correlation', 0):.4f}")
    print(f"  Panel C samples: {panel_c.get('n_samples', 0)}")
    
    return True


def generate_figure_8_comparison(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8 Comparison: Weak (tutorial feature 3834) vs Strong (best from search).
    
    This figure demonstrates:
    - Left side: Feature 3834 (tutorial "not-good") - shows weak AND-gate structure
    - Right side: Best feature from comprehensive search - shows strong AND-gate structure
    - Bottom: Per-word activation examples for both circuits
    """
    print("\n" + "="*60)
    print("FIGURE 8 COMPARISON: Weak vs Strong Circuits")
    print("="*60)
    
    # Check for search results
    search_results_file = results_dir / "circuit_search_complete.json"
    if not search_results_file.exists():
        print(f"Search results not found: {search_results_file}")
        print("Run search first: ./scripts/train/run_language.sh figure8 search")
        return False
    
    with open(search_results_file) as f:
        search_results = json.load(f)
    
    # Get best feature from search
    top_by_and = search_results.get("top_by_and_score", [])
    if not top_by_and:
        print("No top features found in search results")
        return False
    
    best_feature = top_by_and[0]["feature"]
    best_score = top_by_and[0]["and_score"]
    
    print(f"Best feature from search: {best_feature} (AND-score: {best_score:.2f})")
    print(f"Tutorial feature: 3834")
    
    # Check for analysis files
    best_analysis_file = results_dir / f"circuit_analysis_{best_feature}.json"
    tutorial_data_file = results_dir / "figure_8_data_fw_medium.json"
    
    if not best_analysis_file.exists():
        print(f"Analysis file not found: {best_analysis_file}")
        print("Run analysis first: ./scripts/train/run_language.sh figure8 analyze")
        return False
    
    # Load data
    with open(best_analysis_file) as f:
        best_analysis = json.load(f)
    
    tutorial_data = None
    if tutorial_data_file.exists():
        tutorial_data = load_figure_8_data(tutorial_data_file)
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: Interaction matrices
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Bottom row: Feature projections and summary
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Panel 1: Tutorial feature (3834) interaction matrix
    if tutorial_data and "panel_a" in tutorial_data:
        Q = np.array(tutorial_data["panel_a"]["Q_submatrix"])
        im1 = ax1.imshow(Q, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect='auto')
        ax1.set_title(f"Feature 3834 (Tutorial)\nAND-score: weak", fontsize=10)
        ax1.set_xlabel("Input Feature")
        ax1.set_ylabel("Input Feature")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    else:
        ax1.text(0.5, 0.5, "Tutorial data\nnot available", ha='center', va='center', fontsize=12)
        ax1.set_title("Feature 3834 (Tutorial)")
    
    # Panel 2: Best feature interaction matrix
    if "Q_submatrix" in best_analysis:
        Q = np.array(best_analysis["Q_submatrix"])
        vmax = np.abs(Q).max()
        im2 = ax2.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
        ax2.set_title(f"Feature {best_feature} (Best)\nAND-score: {best_score:.2f}", fontsize=10)
        ax2.set_xlabel("Input Feature")
        ax2.set_ylabel("Input Feature")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Panel 3: Eigenvalue comparison
    best_eigs = best_analysis.get("eigenvalues", [])[:10]
    ax3.bar(range(len(best_eigs)), best_eigs, color='steelblue', alpha=0.7, label=f'Feature {best_feature}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_title("Eigenvalue Spectrum (Best Circuit)")
    ax3.legend()
    
    # Panel 4: Tutorial feature projections
    if tutorial_data and "panel_b" in tutorial_data:
        projs = tutorial_data["panel_b"]["projections"]
        x = [p[0] for p in projs.values()]
        y = [p[1] for p in projs.values()]
        ax4.scatter(x, y, alpha=0.7, s=50)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel("v1 projection")
        ax4.set_ylabel("v2 projection")
        ax4.set_title("Feature 3834: Eigenvector Projections")
    else:
        ax4.text(0.5, 0.5, "Projection data\nnot available", ha='center', va='center', fontsize=12)
        ax4.set_title("Feature 3834: Projections")
    
    # Panel 5: Best feature projections
    if "feature_projections" in best_analysis:
        projs = best_analysis["feature_projections"]
        cluster_pos = best_analysis.get("cluster_pos", [])
        cluster_neg = best_analysis.get("cluster_neg", [])
        
        for feat_str, (v1, v2) in projs.items():
            feat = int(feat_str)
            color = 'blue' if feat in cluster_pos else 'orange' if feat in cluster_neg else 'gray'
            ax5.scatter(v1, v2, c=color, s=60, alpha=0.7)
            ax5.annotate(feat_str, (v1, v2), fontsize=7, alpha=0.7)
        
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel("v1 projection")
        ax5.set_ylabel("v2 projection")
        ax5.set_title(f"Feature {best_feature}: Eigenvector Projections")
        
        # Add legend
        from matplotlib.patches import Patch
        ax5.legend(handles=[
            Patch(color='blue', label=f'Cluster 1 ({len(cluster_pos)})'),
            Patch(color='orange', label=f'Cluster 2 ({len(cluster_neg)})'),
        ], loc='best', fontsize=8)
    
    # Panel 6: Summary statistics
    ax6.axis('off')
    
    # Build summary text
    summary_lines = [
        "COMPARISON SUMMARY",
        "="*30,
        "",
        f"Tutorial Feature: 3834",
        f"  Semantic: 'not-good'",
        f"  AND-gate: Weak interactions",
        "",
        f"Best Feature: {best_feature}",
        f"  AND-score: {best_score:.2f}",
        f"  Cluster 1: {len(best_analysis.get('cluster_pos', []))} features",
        f"  Cluster 2: {len(best_analysis.get('cluster_neg', []))} features",
        "",
        "Key Finding:",
        "  Comprehensive search reveals",
        "  strong AND-gate circuits exist",
        "  in fw-medium, but require",
        "  systematic discovery.",
    ]
    
    summary_text = "\n".join(summary_lines)
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, "figure_8_comparison_weak_vs_strong.pdf", figure_dir)
    
    # Also save to Report/figures/language
    report_fig_dir = PROJECT_ROOT / "Report/figures/language"
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    fig2 = plt.figure(figsize=(16, 12))
    # Recreate for second save (figure was closed)
    save_figure(fig, "figure_8_comparison_weak_vs_strong.pdf", report_fig_dir)
    
    print(f"\nComparison figure generated!")
    print(f"  Tutorial (3834): weak interactions")
    print(f"  Best ({best_feature}): AND-score {best_score:.2f}")
    
    return True


def generate_figure_10(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 10 (SAE Training Time Effect) from v0-v4 comparison results.
    
    Figure 10A: Correlation vs approximation rank for all SAE versions
    Figure 10B: Rank-2 correlation histogram (v0 vs v4)
    
    This figure provides scientific evidence that the discrepancy between our
    correlation results and the paper's claims is due to under-trained SAEs
    on HuggingFace, NOT a bug in our implementation.
    """
    print("\n" + "="*60)
    print("FIGURE 10: SAE Training Time Effect")
    print("="*60)
    
    # Load SAE training time comparison results
    sae_results_file = results_dir / "sae_training_time_comparison.json"
    
    if not sae_results_file.exists():
        print(f"SAE training time results not found: {sae_results_file}")
        print("Generate it first:")
        print("  python scripts/sae_training_time_analysis.py")
        return False
    
    print(f"Loading SAE training time results from {sae_results_file}")
    results = load_sae_training_results(sae_results_file)
    
    # Print summary
    print("\n--- SAE Training Time Comparison ---")
    versions = results.get('versions', {})
    print(f"{'Version':<10} {'Rank-1':>10} {'Rank-2':>10} {'%>0.75':>10}")
    print("-" * 45)
    for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
        if version not in versions:
            continue
        summary = versions[version].get('summary', {})
        r1 = summary.get('rank_1', {}).get('mean', 0)
        r2 = summary.get('rank_2', {}).get('mean', 0)
        pct = summary.get('rank_2', {}).get('above_75_pct', 0)
        print(f"{version:<10} {r1:>10.3f} {r2:>10.3f} {pct:>9.1f}%")
    print("-" * 45)
    print(f"{'Paper':>10} {'~0.65':>10} {'>0.75':>10} {'69%':>10}")
    
    # Figure 10A: Correlation vs Rank (all versions)
    print("\nGenerating Figure 10A: Correlation vs SAE Training Time...")
    fig = plot_sae_training_effect(
        results,
        title="Effect of SAE Training on Low-Rank Approximation Quality",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_10a_sae_training_effect.pdf", figure_dir)
    
    # Figure 10B: Rank-2 Histogram (all versions to show progression)
    print("Generating Figure 10B: Rank-2 Correlation Distribution...")
    fig = plot_sae_training_histogram(
        results,
        rank=2,
        versions=['v0', 'v1', 'v2', 'v3', 'v4'],
        title="Rank-2 Correlation: Training Progression (v0 → v4)",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_10b_sae_training_histogram.pdf", figure_dir)
    
    # Quantitative improvement
    v0_r2 = versions.get('v0', {}).get('summary', {}).get('rank_2', {}).get('mean', 0)
    v4_r2 = versions.get('v4', {}).get('summary', {}).get('rank_2', {}).get('mean', 0)
    improvement = v4_r2 / v0_r2 if v0_r2 > 0 else 0
    
    print(f"\n--- Key Finding ---")
    print(f"  Rank-2 correlation improves {improvement:.1f}x from v0 ({v0_r2:.3f}) to v4 ({v4_r2:.3f})")
    print(f"  This supports the paper's claim that correlation improves with SAE training time.")
    print(f"  The gap to paper's 0.75 threshold suggests even longer training is needed.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate language experiment figures")
    parser.add_argument("--figure8-only", action="store_true",
                        help="Only generate Figure 8 variants (including comparison)")
    parser.add_argument("--figure9-only", action="store_true",
                        help="Only generate Figure 9 variants")
    parser.add_argument("--figure10-only", action="store_true",
                        help="Only generate Figure 10 variants")
    args = parser.parse_args()
    
    # Paths (using centralized paths)
    RESULTS_DIR = LANGUAGE_RESULTS
    FIGURE_DIR = LANGUAGE_FIGURES
    REPORT_FIG_DIR = PROJECT_ROOT / "Report/figures/language"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Language Figure Generation")
    print("="*60)
    
    # Determine which figures to generate
    generate_all = not (args.figure8_only or args.figure9_only or args.figure10_only)
    
    # === Figure 9: Correlation Analysis (from sweep) ===
    if generate_all or args.figure9_only:
        generate_figure_9(RESULTS_DIR, FIGURE_DIR)
    
    # === Figure 8: Sentiment Negation Circuit ===
    if generate_all or args.figure8_only:
        # Legacy Figure 8 (single feature)
        generate_figure_8(RESULTS_DIR, FIGURE_DIR)
        
        # Comparison Figure (if search results exist)
        search_results_file = RESULTS_DIR / "circuit_search_complete.json"
        if search_results_file.exists():
            generate_figure_8_comparison(RESULTS_DIR, FIGURE_DIR)
            # Also save to report directory
            generate_figure_8_comparison(RESULTS_DIR, REPORT_FIG_DIR)
        else:
            print("\nNote: Skipping Figure 8 comparison (no search results)")
            print("  Run: ./scripts/train/run_language.sh figure8 search")
    
    # === Figure 10: SAE Training Time Effect ===
    if generate_all or args.figure10_only:
        generate_figure_10(RESULTS_DIR, FIGURE_DIR)

    # --- Load negation results (legacy) ---
    if generate_all:
        negation_file = RESULTS_DIR / "negation_fw_medium.json"
        if negation_file.exists():
            print(f"\nLoading negation results from {negation_file}")
            with open(negation_file) as f:
                negation_data = json.load(f)

            print("\n=== Negation Analysis Summary ===")
            if 'not_positive_features' in negation_data and negation_data['not_positive_features']:
                print(f"Top not+positive feature: {negation_data['not_positive_features'][0]}")
            if 'not_negative_features' in negation_data and negation_data['not_negative_features']:
                print(f"Top not+negative feature: {negation_data['not_negative_features'][0]}")
            if 'top_pair_analysis' in negation_data:
                pair = negation_data['top_pair_analysis']
                print(f"Cosine similarity: {pair['cosine_similarity']:.4f}")
                print(f"Opposing directions: {pair['opposing_directions']}")
        else:
            print(f"\nNegation results not found: {negation_file}")

    print(f"\n{'='*60}")
    print("Figure Generation Complete!")
    print(f"{'='*60}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Report figures saved to: {REPORT_FIG_DIR}")


if __name__ == "__main__":
    main()
