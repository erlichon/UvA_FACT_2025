#!/usr/bin/env python3
"""
Generate figures for Phase 2 (Language) experiments.

Run from project root after running language experiments:
    python scripts/generate_language_figures.py

Generates:
- Variance explained histogram (69% claim verification)
- Effective rank distribution
- Interaction matrix heatmaps for top features
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

# Language visualization utilities
from src.language.interaction_viz import (
    plot_variance_explained_histogram,
    plot_effective_rank_distribution,
    plot_interaction_matrix,
)


def main():
    # Paths
    RESULTS_DIR = PROJECT_ROOT / "results/language"
    FIGURE_DIR = PROJECT_ROOT / "results/language/figures"
    REPORT_FIGURE_DIR = PROJECT_ROOT / "Report/figures"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load interaction analysis results
    interaction_file = RESULTS_DIR / "interaction_analysis_fwmedium.json"
    negation_file = RESULTS_DIR / "negation_analysis_fwmedium.json"

    if not interaction_file.exists():
        print(f"Interaction results not found: {interaction_file}")
        print("Run language experiments first:")
        print("  python src/language/interaction_analysis.py --config configs/language_interaction.yaml")
        return

    print(f"Loading interaction results from {interaction_file}")
    with open(interaction_file) as f:
        interaction_data = json.load(f)

    # Extract data
    correlations = interaction_data['per_feature']['correlations']
    effective_ranks = interaction_data['per_feature']['effective_ranks']
    summary = interaction_data['summary']

    print(f"\n=== Interaction Analysis Summary ===")
    print(f"Features analyzed: {summary['n_analyzed']}")
    print(f"Mean correlation: {summary['mean_correlation']:.4f}")
    print(f"Fraction above 0.75: {summary['fraction_above_075']*100:.1f}%")
    print(f"Paper target: 69%")
    print(f"Claim supported: {summary['claim_supported']}")

    # --- Figure 1: Variance Explained Histogram ---
    print("\nGenerating variance explained histogram...")
    fig = plot_variance_explained_histogram(
        correlations,
        threshold=0.75,
        target_fraction=0.69,
        rank_k=2,
        title="Rank-2 Variance Explained (fw-medium, Layer 7)",
        save_path=str(FIGURE_DIR / "variance_explained_histogram.pdf"),
    )
    if fig:
        fig.savefig(REPORT_FIGURE_DIR / "variance_explained_histogram.pdf", bbox_inches='tight')
        plt.close(fig)

    # --- Figure 2: Effective Rank Distribution ---
    print("Generating effective rank distribution...")
    fig = plot_effective_rank_distribution(
        effective_ranks,
        title="Effective Rank Distribution (fw-medium, Layer 7)",
        save_path=str(FIGURE_DIR / "effective_rank_distribution.pdf"),
    )
    if fig:
        fig.savefig(REPORT_FIGURE_DIR / "effective_rank_distribution.pdf", bbox_inches='tight')
        plt.close(fig)

    # --- Load negation results ---
    if negation_file.exists():
        print(f"\nLoading negation results from {negation_file}")
        with open(negation_file) as f:
            negation_data = json.load(f)

        print("\n=== Negation Analysis Summary ===")
        print(f"Top not+positive feature: {negation_data['not_positive_features'][0]}")
        print(f"Top not+negative feature: {negation_data['not_negative_features'][0]}")
        if 'top_pair_analysis' in negation_data:
            pair = negation_data['top_pair_analysis']
            print(f"Cosine similarity: {pair['cosine_similarity']:.4f}")
            print(f"Opposing directions: {pair['opposing_directions']}")
    else:
        print(f"\nNegation results not found: {negation_file}")

    # --- Summary CSV ---
    print("\nSaving summary CSV...")
    import pandas as pd
    summary_df = pd.DataFrame([{
        'n_features': summary['n_analyzed'],
        'mean_correlation': summary['mean_correlation'],
        'std_correlation': summary['std_correlation'],
        'median_correlation': summary['median_correlation'],
        'fraction_above_075': summary['fraction_above_075'],
        'mean_effective_rank': summary['mean_effective_rank'],
        'paper_claim_supported': summary['claim_supported'],
    }])
    summary_df.to_csv(FIGURE_DIR / "interaction_summary.csv", index=False)

    print(f"\nFigures saved to:")
    print(f"  - {FIGURE_DIR}")
    print(f"  - {REPORT_FIGURE_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
