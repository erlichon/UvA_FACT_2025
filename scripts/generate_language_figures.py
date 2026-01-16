#!/usr/bin/env python3
"""
Generate figures for Phase 2 (Language) experiments.

Run from project root after running language experiments:
    python scripts/generate_language_figures.py

Generates:
- Figure 9A: Correlation progression across ranks (3 models)
- Figure 9B: Rank-2 correlation histogram (3 models)
- Figure 9C: True vs predicted scatter plots (fw-medium)
- Figure 8: Sentiment negation circuit (fw-medium, feature 3834)
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Figure 9 and 8 plotting utilities
from src.plot_utils.language import (
    load_correlation_results,
    plot_correlation_progression,
    plot_correlation_histogram,
    plot_figure_9c_scatters,
    plot_figure_8_composite,
    load_figure_8_data,
    compute_fraction_above_threshold,
)


def save_figure(fig, filename: str, figure_dir: Path, report_dir: Path):
    """Save figure to both results and report directories."""
    if fig is None:
        return
    fig.savefig(figure_dir / filename, bbox_inches='tight', dpi=300)
    fig.savefig(report_dir / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def generate_figure_9(results_dir: Path, figure_dir: Path, report_dir: Path):
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
    save_figure(fig, "figure_9a_correlation_progression.pdf", figure_dir, report_dir)
    
    # Figure 9B: Correlation Histogram
    print("Generating Figure 9B: Rank-2 Correlation Histogram...")
    fig = plot_correlation_histogram(
        correlation_results,
        rank=2,
        title="Rank-2 Correlation Distribution",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_9b_correlation_histogram.pdf", figure_dir, report_dir)
    
    # Figure 9C: Scatter plots (fw-medium only, as per paper)
    print("Generating Figure 9C: True vs Predicted Scatter Plots...")
    fw_medium_data = correlation_results.get("fw-medium")
    if fw_medium_data:
        # Check if scatter data is available
        has_scatter = any("scatter_data" in f for f in fw_medium_data.get("per_feature", []))
        if has_scatter:
            fig = plot_figure_9c_scatters(
                fw_medium_data,
                n_features=9,
                seed=42,
            )
            save_figure(fig, "figure_9c_scatter_plots.pdf", figure_dir, report_dir)
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


def generate_figure_8(results_dir: Path, figure_dir: Path, report_dir: Path):
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
    save_figure(fig, "figure_8_negation_circuit.pdf", figure_dir, report_dir)
    
    # Print summary
    print("\n--- Figure 8 Summary ---")
    print(f"  Output feature: {figure_8_data.get('output_feature_idx')}")
    print(f"  Model: {figure_8_data.get('model_name')}")
    panel_c = figure_8_data.get("panel_c", {})
    print(f"  Panel C correlation: {panel_c.get('correlation', 0):.4f}")
    print(f"  Panel C samples: {panel_c.get('n_samples', 0)}")
    
    return True


def main():
    # Paths
    RESULTS_DIR = PROJECT_ROOT / "results/language"
    FIGURE_DIR = PROJECT_ROOT / "results/language/figures"
    REPORT_FIGURE_DIR = PROJECT_ROOT / "Report/figures"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Language Figure Generation")
    print("="*60)
    
    # === Figure 9: Correlation Analysis (from sweep) ===
    generate_figure_9(RESULTS_DIR, FIGURE_DIR, REPORT_FIGURE_DIR)
    
    # === Figure 8: Sentiment Negation Circuit ===
    generate_figure_8(RESULTS_DIR, FIGURE_DIR, REPORT_FIGURE_DIR)

    # --- Load negation results (legacy) ---
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
    print(f"Figures saved to:")
    print(f"  - {FIGURE_DIR}")
    print(f"  - {REPORT_FIGURE_DIR}")


if __name__ == "__main__":
    main()
