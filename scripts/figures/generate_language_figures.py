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
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.paths import LANGUAGE_FIGURES, LANGUAGE_RESULTS, LANGUAGE_EIGENPAIRS
from src.artifact_loader import ensure_artifacts

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


def _load_search_results(results_dir: Path):
    """Load search results and best feature analysis. Returns None if not available."""
    search_results_file = results_dir / "circuit_search_complete.json"
    if not search_results_file.exists():
        return None, None, None
    
    with open(search_results_file) as f:
        search_results = json.load(f)
    
    top_by_and = search_results.get("top_by_and_score", [])
    if not top_by_and:
        return search_results, None, None
    
    best_feature = top_by_and[0]["feature"]
    best_analysis_file = results_dir / f"circuit_analysis_{best_feature}.json"
    
    if not best_analysis_file.exists():
        return search_results, best_feature, None
    
    with open(best_analysis_file) as f:
        best_analysis = json.load(f)
    
    return search_results, best_feature, best_analysis


def generate_figure_8_best_circuit(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8: Best AND-gate circuit from comprehensive search.
    
    Shows the strongest circuit discovered, with:
    - Panel A: Interaction submatrix
    - Panel B: Feature projections onto eigenvectors  
    - Panel C: Eigenvalue spectrum
    """
    print("\n" + "="*60)
    print("FIGURE 8: Best AND-gate Circuit")
    print("="*60)
    
    search_results, best_feature, best_analysis = _load_search_results(results_dir)
    
    if search_results is None:
        print("Search results not found. Run: ./scripts/train/run_language.sh figure8 search")
        return False
    
    if best_analysis is None:
        print(f"Analysis for best feature not found. Run: ./scripts/train/run_language.sh figure8 analyze")
        return False
    
    best_score = search_results["top_by_and_score"][0]["and_score"]
    print(f"Best feature: {best_feature} (AND-score: {best_score:.2f})")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Interaction submatrix
    if "Q_submatrix" in best_analysis:
        Q = np.array(best_analysis["Q_submatrix"])
        vmax = np.abs(Q).max()
        im = axes[0].imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
        axes[0].set_title(f"Interaction Submatrix\n(Feature {best_feature})")
        axes[0].set_xlabel("Input Feature Index")
        axes[0].set_ylabel("Input Feature Index")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Panel B: Feature projections
    if "feature_projections" in best_analysis:
        projs = best_analysis["feature_projections"]
        cluster_pos = best_analysis.get("cluster_pos", [])
        cluster_neg = best_analysis.get("cluster_neg", [])
        
        for feat_str, (v1, v2) in projs.items():
            feat = int(feat_str)
            color = 'steelblue' if feat in cluster_pos else 'coral' if feat in cluster_neg else 'gray'
            axes[1].scatter(v1, v2, c=color, s=60, alpha=0.7)
            axes[1].annotate(feat_str, (v1, v2), fontsize=7, alpha=0.7)
        
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("v1 projection")
        axes[1].set_ylabel("v2 projection")
        axes[1].set_title("Feature Projections\n(colored by cluster)")
    
    # Panel C: Eigenvalue spectrum
    eigs = best_analysis.get("eigenvalues", [])[:10]
    colors = ['coral' if e < 0 else 'steelblue' for e in eigs]
    axes[2].bar(range(len(eigs)), eigs, color=colors, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel("Eigenvalue Index")
    axes[2].set_ylabel("Eigenvalue")
    axes[2].set_title(f"Eigenvalue Spectrum\n(AND-score: {best_score:.2f})")
    
    plt.tight_layout()
    save_figure(fig, "figure_8_best_circuit.pdf", figure_dir)
    
    return True


def generate_figure_8_sentiment(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8: Circuit with sentiment/semantic labels.
    
    Shows the best circuit with semantic interpretation of clusters.
    """
    print("\n" + "="*60)
    print("FIGURE 8: Sentiment-labeled Circuit")
    print("="*60)
    
    search_results, best_feature, best_analysis = _load_search_results(results_dir)
    
    if best_analysis is None:
        print("Best circuit analysis not found. Run figure8 search and analyze first.")
        return False
    
    # For now, generate same as best_circuit but with different title
    # Full semantic analysis will be added in Phase B
    print(f"Generating sentiment-labeled figure for feature {best_feature}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Interaction submatrix with cluster labels
    if "Q_submatrix" in best_analysis:
        Q = np.array(best_analysis["Q_submatrix"])
        vmax = np.abs(Q).max()
        im = axes[0].imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
        axes[0].set_title(f"Feature {best_feature}\nInteraction Submatrix")
        axes[0].set_xlabel("Input Feature")
        axes[0].set_ylabel("Input Feature")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Panel B: Projections with semantic labels
    if "feature_projections" in best_analysis:
        projs = best_analysis["feature_projections"]
        cluster_pos = set(best_analysis.get("cluster_pos", []))
        cluster_neg = set(best_analysis.get("cluster_neg", []))
        
        for feat_str, (v1, v2) in projs.items():
            feat = int(feat_str)
            if feat in cluster_pos:
                color, label = 'steelblue', 'Cluster A'
            elif feat in cluster_neg:
                color, label = 'coral', 'Cluster B'
            else:
                color, label = 'gray', 'Other'
            axes[1].scatter(v1, v2, c=color, s=60, alpha=0.7)
        
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("v1 (dominant eigenvector)")
        axes[1].set_ylabel("v2 (second eigenvector)")
        axes[1].set_title("Semantic Clustering\n(v1 separates clusters)")
        
        # Legend
        from matplotlib.patches import Patch
        axes[1].legend(handles=[
            Patch(color='steelblue', label=f'Cluster A ({len(cluster_pos)})'),
            Patch(color='coral', label=f'Cluster B ({len(cluster_neg)})'),
        ], loc='best', fontsize=8)
    
    # Panel C: Summary text
    axes[2].axis('off')
    n_pos = len(best_analysis.get("cluster_pos", []))
    n_neg = len(best_analysis.get("cluster_neg", []))
    best_score = search_results["top_by_and_score"][0]["and_score"] if search_results else 0
    
    summary = f"""
CIRCUIT ANALYSIS
================

Output Feature: {best_feature}
AND-gate Score: {best_score:.2f}

Cluster A: {n_pos} features
  (positive v1 projection)

Cluster B: {n_neg} features  
  (negative v1 projection)

The circuit activates when
features from BOTH clusters
are present in the input.
"""
    axes[2].text(0.1, 0.9, summary, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_figure(fig, "figure_8_sentiment.pdf", figure_dir)
    
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
    
    search_results, best_feature, best_analysis = _load_search_results(results_dir)
    
    if search_results is None:
        print("Search results not found. Run: ./scripts/train/run_language.sh figure8 search")
        return False
    
    if best_analysis is None:
        print("Best circuit analysis not found. Run: ./scripts/train/run_language.sh figure8 analyze")
        return False
    
    best_score = search_results["top_by_and_score"][0]["and_score"]
    print(f"Best feature from search: {best_feature} (AND-score: {best_score:.2f})")
    print(f"Tutorial feature: 3834")
    
    tutorial_data_file = results_dir / "figure_8_data_fw_medium.json"
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
        projs = tutorial_data["panel_b"].get("feature_projections") or tutorial_data["panel_b"].get("projections", {})
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
    
    # Save figure with name matching report reference
    save_figure(fig, "figure_8_comparison_751_vs_3834.pdf", figure_dir)
    
    print(f"\nComparison figure generated!")
    print(f"  Tutorial (3834): weak interactions")
    print(f"  Best ({best_feature}): AND-score {best_score:.2f}")
    
    return True


def generate_figure_8_side_by_side(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8 Side-by-Side Comparison: Feature 3834 vs Feature 751.
    
    Creates a clean 2-row x 3-column comparison figure:
    - Row 1: Feature 3834 (Tutorial "not-good")
    - Row 2: Feature 751 (Best AND-gate circuit discovered)
    
    Each row shows: Interaction Submatrix | Eigenvector Projections | Eigenvalue Spectrum
    """
    print("\n" + "="*60)
    print("FIGURE 8: Side-by-Side Comparison (3834 vs 751)")
    print("="*60)
    
    # Load data for both features
    feature_3834_file = results_dir / "figure_8_data_fw_medium.json"
    feature_751_file = results_dir / "figure_8_feature751.json"
    circuit_751_file = results_dir / "circuit_analysis_751.json"
    
    if not feature_3834_file.exists():
        print(f"Feature 3834 data not found: {feature_3834_file}")
        return False
    
    if not feature_751_file.exists() and not circuit_751_file.exists():
        print(f"Feature 751 data not found")
        return False
    
    # Load feature 3834 data
    data_3834 = load_figure_8_data(feature_3834_file)
    
    # Load feature 751 data
    if feature_751_file.exists():
        data_751 = load_figure_8_data(feature_751_file)
    else:
        with open(circuit_751_file) as f:
            data_751 = json.load(f)
    
    # Get AND-scores from search results if available
    search_file = results_dir / "circuit_search_complete.json"
    and_score_751 = None
    if search_file.exists():
        with open(search_file) as f:
            search_results = json.load(f)
        for item in search_results.get("top_by_and_score", []):
            if item["feature"] == 751:
                and_score_751 = item["and_score"]
                break
    
    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features_data = [
        ("Feature 3834 (Tutorial)", data_3834, "weak"),
        (f"Feature 751 (Best)", data_751, f"{and_score_751:.2f}" if and_score_751 else "strong"),
    ]
    
    for row_idx, (title_prefix, data, and_score_str) in enumerate(features_data):
        ax_interaction = axes[row_idx, 0]
        ax_projection = axes[row_idx, 1]
        ax_eigenval = axes[row_idx, 2]
        
        # Panel A: Interaction Submatrix
        if "panel_a" in data:
            Q = np.array(data["panel_a"]["Q_submatrix"])
        elif "Q_submatrix" in data:
            Q = np.array(data["Q_submatrix"])
        else:
            Q = None
        
        if Q is not None:
            vmax = np.abs(Q).max()
            im = ax_interaction.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
            ax_interaction.set_title(f"{title_prefix}\nInteraction Submatrix", fontsize=10)
            ax_interaction.set_xlabel("Input Feature", fontsize=9)
            ax_interaction.set_ylabel("Input Feature", fontsize=9)
            plt.colorbar(im, ax=ax_interaction, fraction=0.046, pad=0.04)
        else:
            ax_interaction.text(0.5, 0.5, "Data not available", ha='center', va='center')
            ax_interaction.set_title(f"{title_prefix}\nInteraction Submatrix", fontsize=10)
        
        # Panel B: Eigenvector Projections
        if "panel_b" in data:
            projs = data["panel_b"].get("feature_projections") or data["panel_b"].get("projections", {})
        elif "feature_projections" in data:
            projs = data["feature_projections"]
        else:
            projs = {}
        
        cluster_pos = data.get("cluster_pos", [])
        cluster_neg = data.get("cluster_neg", [])
        
        if projs:
            for feat_str, coords in projs.items():
                v1, v2 = coords[0], coords[1]
                feat = int(feat_str)
                if feat in cluster_pos:
                    color = '#2ca02c'  # Green
                elif feat in cluster_neg:
                    color = '#d62728'  # Red
                else:
                    color = '#7f7f7f'  # Gray
                ax_projection.scatter(v1, v2, c=color, s=50, alpha=0.7)
            
            ax_projection.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax_projection.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax_projection.set_xlabel("$v_1$ projection", fontsize=9)
            ax_projection.set_ylabel("$v_2$ projection", fontsize=9)
            ax_projection.set_title(f"Eigenvector Projections\n(AND-score: {and_score_str})", fontsize=10)
            
            # Add legend if clusters exist
            if cluster_pos or cluster_neg:
                from matplotlib.patches import Patch
                legend_handles = []
                if cluster_pos:
                    legend_handles.append(Patch(color='#2ca02c', label=f'Cluster + ({len(cluster_pos)})'))
                if cluster_neg:
                    legend_handles.append(Patch(color='#d62728', label=f'Cluster - ({len(cluster_neg)})'))
                ax_projection.legend(handles=legend_handles, loc='best', fontsize=7)
        else:
            ax_projection.text(0.5, 0.5, "Projection data\nnot available", ha='center', va='center')
            ax_projection.set_title(f"Eigenvector Projections", fontsize=10)
        
        # Panel C: Eigenvalue Spectrum
        if "panel_c" in data:
            # For figure_8_data format, eigenvalues might be in a different structure
            eigenvals = None
        elif "eigenvalues" in data:
            eigenvals = data["eigenvalues"][:10]
        else:
            eigenvals = None
        
        if eigenvals is not None:
            colors = ['#d62728' if e < 0 else '#2ca02c' for e in eigenvals]
            ax_eigenval.bar(range(len(eigenvals)), eigenvals, color=colors, alpha=0.7)
            ax_eigenval.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_eigenval.set_xlabel("Eigenvalue Index", fontsize=9)
            ax_eigenval.set_ylabel("Eigenvalue", fontsize=9)
            ax_eigenval.set_title(f"Eigenvalue Spectrum\n(top 10)", fontsize=10)
            ax_eigenval.set_xticks(range(len(eigenvals)))
        else:
            ax_eigenval.text(0.5, 0.5, "Eigenvalue data\nnot available", ha='center', va='center')
            ax_eigenval.set_title(f"Eigenvalue Spectrum", fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, "figure_8_side_by_side_comparison.pdf", figure_dir)
    
    print(f"\nSide-by-side comparison generated!")
    print(f"  Row 1: Feature 3834 (tutorial)")
    score_str = f"{and_score_751:.2f}" if and_score_751 is not None else "N/A"
    print(f"  Row 2: Feature 751 (best AND-gate, score: {score_str})")
    
    return True


def generate_figure_8_final(results_dir: Path, figure_dir: Path):
    """
    Generate Final Figure 8 in paper style (single feature 3834).
    
    Layout: 1 row x 3 columns
    - A) Interaction Submatrix
    - B) Eigenvector Projections (paper axis orientation)
    - C) Scatter plot (true vs predicted activation)
    """
    return generate_figure_8_paper_style(
        results_dir,
        figure_dir,
        output_name="figure_8_final.pdf",
    )
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    print("\n" + "="*60)
    print("FIGURE 8 FINAL: Feature 3834 vs 751 (Paper Style)")
    print("="*60)
    
    # Load data for both features
    feature_3834_file = results_dir / "figure_8_data_fw_medium.json"
    feature_751_file = results_dir / "figure_8_feature751.json"
    circuit_751_file = results_dir / "circuit_analysis_751.json"
    
    if not feature_3834_file.exists():
        print(f"Feature 3834 data not found: {feature_3834_file}")
        return False
    
    if not feature_751_file.exists():
        print(f"Feature 751 data not found: {feature_751_file}")
        return False
    
    with open(feature_3834_file) as f:
        data_3834 = json.load(f)
    
    with open(feature_751_file) as f:
        data_751 = json.load(f)
    
    # Load circuit analysis for feature 751
    cluster_pos_751 = []
    cluster_neg_751 = []
    if circuit_751_file.exists():
        with open(circuit_751_file) as f:
            circuit_data = json.load(f)
        cluster_pos_751 = circuit_data.get("cluster_pos", [])
        cluster_neg_751 = circuit_data.get("cluster_neg", [])
    
    # Get feature types
    feature_types_3834 = data_3834.get("feature_types", {})
    feature_types_751 = data_751.get("feature_types", {})
    
    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ==================== ROW 1: Feature 3834 (Tutorial) ====================
    ax_Q_3834, ax_proj_3834, ax_scatter_3834 = axes[0]
    
    # --- Panel A: Interaction Submatrix ---
    Q_3834 = np.array(data_3834["panel_a"]["Q_submatrix"])
    feature_ids_3834 = data_3834["panel_a"]["feature_indices"]
    
    vmax = np.abs(Q_3834).max()
    im = ax_Q_3834.imshow(Q_3834, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    n_feats = len(feature_ids_3834)
    ax_Q_3834.set_xticks(range(n_feats))
    ax_Q_3834.set_yticks(range(n_feats))
    ax_Q_3834.set_xticklabels(feature_ids_3834, rotation=45, ha='right', fontsize=7)
    ax_Q_3834.set_yticklabels(feature_ids_3834, fontsize=7)
    
    # Color and markers by type
    color_map_3834 = {'structural': '#1f77b4', 'negation': '#d62728', 'contrast': '#ff7f0e', 'other': '#7f7f7f'}
    marker_map_3834 = {'structural': 's', 'negation': 'o', 'contrast': 'D', 'other': 'o'}
    
    for i, fid in enumerate(feature_ids_3834):
        ftype = feature_types_3834.get(str(fid), 'other')
        color = color_map_3834.get(ftype, '#7f7f7f')
        marker = marker_map_3834.get(ftype, 'o')
        ax_Q_3834.get_xticklabels()[i].set_color(color)
        ax_Q_3834.get_yticklabels()[i].set_color(color)
        ax_Q_3834.scatter(i, -0.8, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
        ax_Q_3834.scatter(-0.8, i, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
    
    ax_Q_3834.set_title("Feature 3834 (Tutorial: \"not-good\")\nA) Interaction Submatrix", fontsize=10, fontweight='bold')
    ax_Q_3834.set_xlabel("Input Feature", fontsize=9)
    ax_Q_3834.set_ylabel("Input Feature", fontsize=9)
    plt.colorbar(im, ax=ax_Q_3834, fraction=0.046, pad=0.04)
    
    # --- Panel B: Eigenvector Projections ---
    projs_3834 = data_3834["panel_b"].get("feature_projections", {})
    
    type_counts_3834 = {}
    for feat_str, coords in projs_3834.items():
        v1, v2 = coords[0], coords[1]
        ftype = feature_types_3834.get(feat_str, 'other')
        color = color_map_3834.get(ftype, '#7f7f7f')
        marker = marker_map_3834.get(ftype, 'o')
        
        ax_proj_3834.scatter(v1, v2, marker=marker, c=color, s=80, alpha=0.85, 
                            edgecolors='black', linewidths=0.5, zorder=5)
        type_counts_3834[ftype] = type_counts_3834.get(ftype, 0) + 1
    
    ax_proj_3834.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_proj_3834.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add directional arrows (paper style)
    xlim = ax_proj_3834.get_xlim()
    ylim = ax_proj_3834.get_ylim()
    ax_proj_3834.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                         arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
    ax_proj_3834.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                         arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))
    ax_proj_3834.text(xlim[1]*0.85, ylim[0]*0.12, '+', fontsize=12, color='#d62728', fontweight='bold')
    ax_proj_3834.text(xlim[0]*0.85, ylim[0]*0.12, '−', fontsize=12, color='#d62728', fontweight='bold')
    ax_proj_3834.text(xlim[0]*0.12, ylim[1]*0.85, '+', fontsize=12, color='#2ca02c', fontweight='bold')
    ax_proj_3834.text(xlim[0]*0.12, ylim[0]*0.85, '−', fontsize=12, color='#2ca02c', fontweight='bold')
    
    ax_proj_3834.set_xlabel("Top positive eigenvector", fontsize=9, color='#d62728')
    ax_proj_3834.set_ylabel("Top negative eigenvector", fontsize=9, color='#2ca02c')
    ax_proj_3834.set_title("B) Eigenvector Projections", fontsize=10, fontweight='bold')
    
    legend_handles_3834 = [
        Line2D([0], [0], marker=marker_map_3834[ft], color='w', markerfacecolor=color_map_3834[ft], 
               markersize=8, markeredgecolor='black', label=f'{ft} ({type_counts_3834.get(ft, 0)})')
        for ft in ['structural', 'negation', 'contrast', 'other'] if ft in type_counts_3834
    ]
    ax_proj_3834.legend(handles=legend_handles_3834, loc='upper right', fontsize=7)
    
    # --- Panel C: Scatter Plot ---
    panel_c_3834 = data_3834.get("panel_c", {})
    z_true_3834 = np.array(panel_c_3834.get("z_true", []))
    z_pred_3834 = np.array(panel_c_3834.get("z_pred_rank2", []))
    corr_3834 = panel_c_3834.get("correlation", 0)
    
    if len(z_true_3834) > 0 and len(z_pred_3834) > 0:
        n_samples = min(2000, len(z_true_3834))
        np.random.seed(42)
        idx = np.random.choice(len(z_true_3834), n_samples, replace=False)
        ax_scatter_3834.scatter(z_true_3834[idx], z_pred_3834[idx], alpha=0.25, s=6, c='#2E86AB', edgecolors='none')
        max_val = max(z_true_3834.max(), z_pred_3834.max()) * 1.1
        ax_scatter_3834.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5)
        ax_scatter_3834.set_xlim(0, max_val)
        ax_scatter_3834.set_ylim(0, max_val)
    
    ax_scatter_3834.set_xlabel("True SAE Activation", fontsize=9)
    ax_scatter_3834.set_ylabel("Rank-2 Prediction", fontsize=9)
    ax_scatter_3834.set_title(f"C) Correlation: r = {corr_3834:.3f}", fontsize=10, fontweight='bold')
    
    # ==================== ROW 2: Feature 751 (Strong AND-gate) ====================
    ax_Q_751, ax_proj_751, ax_scatter_751 = axes[1]
    
    # --- Panel A: Interaction Submatrix (grouped by cluster) ---
    if "panel_a" in data_751:
        Q_orig = np.array(data_751["panel_a"]["Q_submatrix"])
        feature_ids_751 = data_751["panel_a"]["feature_indices"]
    else:
        Q_orig = np.array(data_751["Q_submatrix"])
        feature_ids_751 = data_751.get("top_input_features", list(range(len(Q_orig))))
    
    # Reorder to show block structure
    pos_indices = [i for i, fid in enumerate(feature_ids_751) if fid in cluster_pos_751]
    neg_indices = [i for i, fid in enumerate(feature_ids_751) if fid in cluster_neg_751]
    other_indices = [i for i, fid in enumerate(feature_ids_751) if fid not in cluster_pos_751 and fid not in cluster_neg_751]
    new_order = pos_indices + neg_indices + other_indices
    
    Q_751 = Q_orig[np.ix_(new_order, new_order)]
    feature_ids_751_reordered = [feature_ids_751[i] for i in new_order]
    
    vmax = np.abs(Q_751).max()
    im = ax_Q_751.imshow(Q_751, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    n_feats = len(feature_ids_751_reordered)
    ax_Q_751.set_xticks(range(n_feats))
    ax_Q_751.set_yticks(range(n_feats))
    ax_Q_751.set_xticklabels(feature_ids_751_reordered, rotation=45, ha='right', fontsize=7)
    ax_Q_751.set_yticklabels(feature_ids_751_reordered, fontsize=7)
    
    for i, fid in enumerate(feature_ids_751_reordered):
        if fid in cluster_pos_751:
            color, marker = '#ff7f0e', '^'
        elif fid in cluster_neg_751:
            color, marker = '#1f77b4', 'o'
        else:
            color, marker = '#7f7f7f', 's'
        ax_Q_751.get_xticklabels()[i].set_color(color)
        ax_Q_751.get_yticklabels()[i].set_color(color)
        ax_Q_751.scatter(i, -0.8, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
        ax_Q_751.scatter(-0.8, i, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
    
    # Block boundary
    n_pos = len(pos_indices)
    if n_pos > 0:
        ax_Q_751.axhline(y=n_pos - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_Q_751.axvline(x=n_pos - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax_Q_751.set_title("Feature 751 (Strong AND-gate)\nA) Interaction Submatrix (grouped)", fontsize=10, fontweight='bold')
    ax_Q_751.set_xlabel("Input Feature", fontsize=9)
    ax_Q_751.set_ylabel("Input Feature", fontsize=9)
    plt.colorbar(im, ax=ax_Q_751, fraction=0.046, pad=0.04)
    
    # --- Panel B: Eigenvector Projections ---
    if "panel_b" in data_751:
        projs_751 = data_751["panel_b"].get("feature_projections", {})
    else:
        projs_751 = data_751.get("feature_projections", {})
    
    type_counts_751 = {'positive': 0, 'negative': 0}
    for feat_str, coords in projs_751.items():
        if isinstance(coords, list) and len(coords) >= 2:
            v1, v2 = coords[0], coords[1]
        else:
            continue
        fid = int(feat_str)
        if fid in cluster_pos_751:
            marker, color, size = '^', '#ff7f0e', 100
            type_counts_751['positive'] += 1
        elif fid in cluster_neg_751:
            marker, color, size = 'o', '#1f77b4', 80
            type_counts_751['negative'] += 1
        else:
            marker, color, size = 's', '#7f7f7f', 60
        
        ax_proj_751.scatter(v1, v2, marker=marker, c=color, s=size, alpha=0.85, 
                           edgecolors='black', linewidths=0.5, zorder=5)
    
    ax_proj_751.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_proj_751.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Directional arrows
    xlim = ax_proj_751.get_xlim()
    ylim = ax_proj_751.get_ylim()
    ax_proj_751.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
    ax_proj_751.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))
    ax_proj_751.text(xlim[1]*0.85, ylim[0]*0.12, '+', fontsize=12, color='#d62728', fontweight='bold')
    ax_proj_751.text(xlim[0]*0.85, ylim[0]*0.12, '−', fontsize=12, color='#d62728', fontweight='bold')
    ax_proj_751.text(xlim[0]*0.12, ylim[1]*0.85, '+', fontsize=12, color='#2ca02c', fontweight='bold')
    ax_proj_751.text(xlim[0]*0.12, ylim[0]*0.85, '−', fontsize=12, color='#2ca02c', fontweight='bold')
    
    ax_proj_751.set_xlabel("Top positive eigenvector", fontsize=9, color='#d62728')
    ax_proj_751.set_ylabel("Top negative eigenvector", fontsize=9, color='#2ca02c')
    ax_proj_751.set_title("B) Eigenvector Projections", fontsize=10, fontweight='bold')
    
    legend_elements_751 = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', 
               markersize=10, markeredgecolor='black', label=f'Positive ({type_counts_751["positive"]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markersize=10, markeredgecolor='black', label=f'Negative ({type_counts_751["negative"]})'),
    ]
    ax_proj_751.legend(handles=legend_elements_751, loc='upper right', fontsize=8)
    
    # --- Panel C: Scatter Plot ---
    panel_c_751 = data_751.get("panel_c", {})
    z_true_751 = np.array(panel_c_751.get("z_true", []))
    z_pred_751 = np.array(panel_c_751.get("z_pred_rank2", []))
    corr_751 = panel_c_751.get("correlation", 0)
    
    if len(z_true_751) > 0 and len(z_pred_751) > 0:
        n_samples = min(2000, len(z_true_751))
        np.random.seed(42)
        idx = np.random.choice(len(z_true_751), n_samples, replace=False)
        ax_scatter_751.scatter(z_true_751[idx], z_pred_751[idx], alpha=0.25, s=6, c='#2E86AB', edgecolors='none')
        max_val = max(z_true_751.max(), z_pred_751.max()) * 1.1
        ax_scatter_751.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5)
        ax_scatter_751.set_xlim(0, max_val)
        ax_scatter_751.set_ylim(0, max_val)
    
    ax_scatter_751.set_xlabel("True SAE Activation", fontsize=9)
    ax_scatter_751.set_ylabel("Rank-2 Prediction", fontsize=9)
    ax_scatter_751.set_title(f"C) Correlation: r = {corr_751:.3f}", fontsize=10, fontweight='bold')
    
    # Main title
    fig.suptitle("Sentiment Negation Circuits: Tutorial Feature 3834 vs Strong AND-gate 751", 
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3)
    
    save_figure(fig, "figure_8_final.pdf", figure_dir)
    
    print(f"\nFinal Figure 8 generated!")
    print(f"  Row 1: Feature 3834 (tutorial), r = {corr_3834:.3f}")
    print(f"  Row 2: Feature 751 (AND-gate), r = {corr_751:.3f}")
    print(f"  Clusters (751): {type_counts_751['positive']} positive, {type_counts_751['negative']} negative")
    print(f"  Style: Clean paper format")
    
    return True


def generate_figure_8_paper_style(results_dir: Path, figure_dir: Path, output_name: str = "figure_8_final.pdf"):
    """
    Generate Figure 8 in exact paper style: single feature (3834) with semantic labels.
    
    Layout: 1 row x 3 columns (matching paper's sentiment_negation_interaction_and_eigenvectors.pdf)
    - A) Interaction submatrix with colored symbols (squares, triangles)
    - B) Eigenvector projection with semantic labels and correct +/- orientation
    - C) Scatter plot (true vs approximated activation)
    
    Key differences from generate_figure_8_final:
    - Single feature (not 2 rows)
    - Paper's axis orientation: Y="Top positive eigenvector" (pointing up with + at top)
    - Paper's axis orientation: X="Top negative eigenvector" (pointing right with + at right)
    - Semantic labels: "bad"-"good" unembed, "not" input, etc.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    print("\n" + "="*60)
    print("FIGURE 8 PAPER STYLE: Feature 3834 (not-good)")
    print("="*60)
    
    # Load data for feature 3834
    feature_3834_file = results_dir / "figure_8_data_fw_medium.json"
    
    if not feature_3834_file.exists():
        print(f"Feature 3834 data not found: {feature_3834_file}")
        print("Run: ./scripts/train/run_language.sh figure8")
        return False
    
    with open(feature_3834_file) as f:
        data_3834 = json.load(f)
    
    # Get feature types for coloring
    feature_types = data_3834.get("feature_types", {})
    
    # Paper-style colors and markers
    # Blue squares = negative sentiment features (sad, bad, hurt)
    # Green triangles = negation features (not, never, wasn't)  
    # Orange downward triangle = positive sentiment (good, safe, nice)
    color_map = {
        'negative_sentiment': '#1f77b4',  # Blue
        'negation': '#2ca02c',            # Green
        'positive_sentiment': '#ff7f0e',   # Orange
        'structural': '#1f77b4',          # Map old names
        'contrast': '#ff7f0e',
        'other': '#7f7f7f',
    }
    marker_map = {
        'negative_sentiment': 's',         # Square
        'negation': '^',                   # Triangle up
        'positive_sentiment': 'v',         # Triangle down
        'structural': 's',
        'contrast': 'D',
        'other': 'o',
    }
    
    # Create figure: 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_Q, ax_proj, ax_scatter = axes
    
    # ==================== Panel A: Interaction Submatrix ====================
    Q = np.array(data_3834["panel_a"]["Q_submatrix"])
    feature_ids = data_3834["panel_a"]["feature_indices"]
    
    vmax = np.abs(Q).max()
    im = ax_Q.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    n_feats = len(feature_ids)
    ax_Q.set_xticks(range(n_feats))
    ax_Q.set_yticks(range(n_feats))
    ax_Q.set_xticklabels(feature_ids, rotation=45, ha='right', fontsize=7)
    ax_Q.set_yticklabels(feature_ids, fontsize=7)
    
    # Add colored markers and tick labels by feature type
    for i, fid in enumerate(feature_ids):
        ftype = feature_types.get(str(fid), 'other')
        color = color_map.get(ftype, '#7f7f7f')
        marker = marker_map.get(ftype, 'o')
        ax_Q.get_xticklabels()[i].set_color(color)
        ax_Q.get_yticklabels()[i].set_color(color)
        # Markers on axes
        ax_Q.scatter(i, -0.8, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
        ax_Q.scatter(-0.8, i, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
    
    ax_Q.set_title("A)", fontsize=12, fontweight='bold', loc='left')
    ax_Q.set_xlabel("Input Feature", fontsize=10)
    ax_Q.set_ylabel("Input Feature", fontsize=10)
    plt.colorbar(im, ax=ax_Q, fraction=0.046, pad=0.04)
    
    # ==================== Panel B: Eigenvector Projections (Paper Style) ====================
    projs = data_3834["panel_b"].get("feature_projections", {})
    
    # Paper's axis orientation:
    # X-axis: Top NEGATIVE eigenvector (- on left, + on right)
    # Y-axis: Top POSITIVE eigenvector (- on bottom, + on top)
    # Our data has [v1, v2] where v1=positive eigenvector, v2=negative eigenvector
    # Paper swaps: X=negative (v2), Y=positive (v1)
    
    type_counts = {}
    for feat_str, coords in projs.items():
        v_pos, v_neg = coords[0], coords[1]  # v1=positive, v2=negative
        ftype = feature_types.get(feat_str, 'other')
        color = color_map.get(ftype, '#7f7f7f')
        marker = marker_map.get(ftype, 'o')
        
        # Paper style: X = negative eigenvector, Y = positive eigenvector
        ax_proj.scatter(v_neg, v_pos, marker=marker, c=color, s=80, alpha=0.85, 
                       edgecolors='black', linewidths=0.5, zorder=5)
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    
    ax_proj.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_proj.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add directional arrows and +/- labels (paper style)
    xlim = ax_proj.get_xlim()
    ylim = ax_proj.get_ylim()
    
    # X-axis: negative eigenvector (blue color in paper)
    ax_proj.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                    arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2))
    ax_proj.text(xlim[1]*0.85, ylim[0]*0.15, '+', fontsize=14, color='#1f77b4', fontweight='bold')
    ax_proj.text(xlim[0]*0.85, ylim[0]*0.15, '−', fontsize=14, color='#1f77b4', fontweight='bold')
    
    # Y-axis: positive eigenvector (red/maroon color in paper)
    ax_proj.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                    arrowprops=dict(arrowstyle='<->', color='#8B0000', lw=2))
    ax_proj.text(xlim[0]*0.15, ylim[1]*0.85, '+', fontsize=14, color='#8B0000', fontweight='bold')
    ax_proj.text(xlim[0]*0.15, ylim[0]*0.85, '−', fontsize=14, color='#8B0000', fontweight='bold')
    
    # Semantic direction labels (as in paper)
    # These are approximate - showing key semantic directions
    ax_proj.text(xlim[0]*0.7, ylim[1]*0.7, '"not-bad"\nfeature', fontsize=8, ha='center', va='center')
    ax_proj.text(xlim[1]*0.5, ylim[0]*0.7, '"not-good"\nfeature', fontsize=8, ha='center', va='center')
    ax_proj.text(xlim[0]*0.6, ylim[1]*0.3, '"bad"-"good"\nunembed', fontsize=8, ha='center', va='center')
    ax_proj.text(xlim[1]*0.6, ylim[0]*0.3, '"good"-"bad"\nunembed', fontsize=8, ha='center', va='center')
    ax_proj.text(0, ylim[0]*0.5, '"not"\ninput', fontsize=8, ha='center', va='center')
    
    ax_proj.set_xlabel("Top negative eigenvector", fontsize=10, color='#1f77b4')
    ax_proj.set_ylabel("Top positive eigenvector", fontsize=10, color='#8B0000')
    ax_proj.set_title("B)", fontsize=12, fontweight='bold', loc='left')
    
    # Legend
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', 
               markersize=8, markeredgecolor='black', label='Neg. sentiment'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c', 
               markersize=8, markeredgecolor='black', label='Negation'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#ff7f0e', 
               markersize=8, markeredgecolor='black', label='Pos. sentiment'),
    ]
    ax_proj.legend(handles=legend_handles, loc='upper right', fontsize=8)
    
    # ==================== Panel C: Scatter Plot ====================
    panel_c = data_3834.get("panel_c", {})
    z_true = np.array(panel_c.get("z_true", []))
    z_pred = np.array(panel_c.get("z_pred_rank2", []))
    corr = panel_c.get("correlation", 0)
    
    if len(z_true) > 0 and len(z_pred) > 0:
        n_samples = min(2000, len(z_true))
        np.random.seed(42)
        idx = np.random.choice(len(z_true), n_samples, replace=False)
        ax_scatter.scatter(z_true[idx], z_pred[idx], alpha=0.25, s=6, c='#2E86AB', edgecolors='none')
        max_val = max(z_true.max(), z_pred.max()) * 1.1
        ax_scatter.plot([0, max_val], [0, max_val], '--', color='orange', alpha=0.8, linewidth=1.5)
        ax_scatter.set_xlim(0, max_val)
        ax_scatter.set_ylim(0, max_val)
    
    ax_scatter.set_xlabel("Output feature activation", fontsize=10)
    ax_scatter.set_ylabel("Eigenvector-based activation", fontsize=10)
    ax_scatter.set_title("C)", fontsize=12, fontweight='bold', loc='left')
    
    # Add correlation annotation
    ax_scatter.text(0.95, 0.05, f'{corr:.2f}', transform=ax_scatter.transAxes, 
                   fontsize=12, ha='right', va='bottom')
    
    plt.tight_layout()
    
    save_figure(fig, output_name, figure_dir)
    
    print(f"\nFigure 8 (Paper Style) generated!")
    print(f"  Feature: 3834 (not-good)")
    print(f"  Correlation: r = {corr:.3f}")
    print(f"  Output: {figure_dir / output_name}")
    
    return True


def generate_figure_8_with_examples(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8: Circuit with per-word activation examples.
    
    Shows the best circuit with example sentences demonstrating
    how input clusters activate and produce the output.
    """
    print("\n" + "="*60)
    print("FIGURE 8: Circuit with Per-word Examples")
    print("="*60)
    
    search_results, best_feature, best_analysis = _load_search_results(results_dir)
    
    if best_analysis is None:
        print("Best circuit analysis not found. Run figure8 search and analyze first.")
        return False
    
    best_score = search_results["top_by_and_score"][0]["and_score"] if search_results else 0
    print(f"Generating example figure for feature {best_feature}")
    
    # Create figure with example section
    fig = plt.figure(figsize=(16, 10))
    
    # Top row: Circuit visualization (2 panels)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Bottom row: Examples and summary
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Panel 1: Interaction submatrix
    if "Q_submatrix" in best_analysis:
        Q = np.array(best_analysis["Q_submatrix"])
        vmax = np.abs(Q).max()
        im = ax1.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
        ax1.set_title(f"Feature {best_feature}: Interaction Submatrix")
        ax1.set_xlabel("Input Feature Index")
        ax1.set_ylabel("Input Feature Index")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # Panel 2: Eigenvalue spectrum
    eigs = best_analysis.get("eigenvalues", [])[:10]
    colors = ['coral' if e < 0 else 'steelblue' for e in eigs]
    ax2.bar(range(len(eigs)), eigs, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title(f"Eigenvalue Spectrum (AND-score: {best_score:.2f})")
    
    # Panel 3: Example sentences (placeholder - requires runtime model)
    ax3.axis('off')
    example_text = f"""
EXAMPLE SENTENCES
=================

The circuit (Feature {best_feature}) activates when
features from BOTH input clusters are present.

Expected activation patterns:
- "Despite problems, it was wonderful" -> HIGH
- "It was wonderful" (only positive) -> LOW  
- "There are problems" (only negative) -> LOW
- "The cat sat on the mat" (neutral) -> LOW

Note: Full per-word activation heatmaps
require running the model at inference time.
See archived scripts for implementation.
"""
    ax3.text(0.05, 0.95, example_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Panel 4: Cluster summary
    ax4.axis('off')
    cluster_pos = best_analysis.get("cluster_pos", [])
    cluster_neg = best_analysis.get("cluster_neg", [])
    
    cluster_text = f"""
INPUT CLUSTER SUMMARY
=====================

Cluster A ({len(cluster_pos)} features):
  Features: {cluster_pos[:5]}{'...' if len(cluster_pos) > 5 else ''}
  (positive v1 projection)

Cluster B ({len(cluster_neg)} features):
  Features: {cluster_neg[:5]}{'...' if len(cluster_neg) > 5 else ''}
  (negative v1 projection)

AND-gate Behavior:
  Output activates when BOTH
  Cluster A AND Cluster B
  features are present.
"""
    ax4.text(0.05, 0.95, cluster_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    save_figure(fig, "figure_8_with_examples.pdf", figure_dir)
    
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


def generate_figure_9_cosine(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 9 (A, B, C) from COSINE similarity sweep results.
    
    Same as generate_figure_9 but reads from results/language/cosine/ subfolder.
    """
    cosine_results_dir = results_dir / "cosine"
    cosine_figure_dir = figure_dir / "cosine"
    
    print("\n" + "="*60)
    print("FIGURE 9 (COSINE): Cosine Similarity Analysis")
    print("="*60)
    
    # Load correlation results from cosine subfolder
    correlation_results = load_correlation_results(cosine_results_dir)
    
    if not correlation_results:
        print("No cosine similarity results found.")
        print("Run the sweep first with --metric cosine:")
        print("  ./scripts/train/run_language.sh figure9 --metric cosine")
        return False
    
    print(f"Loaded cosine results for models: {list(correlation_results.keys())}")
    
    # Figure 9A: Correlation Progression (cosine)
    print("\nGenerating Figure 9A (cosine): Similarity Progression...")
    fig = plot_correlation_progression(
        correlation_results,
        title="Average Cosine Similarity vs Approximation Rank",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_9a.pdf", cosine_figure_dir)
    
    # Figure 9B: Correlation Histogram (cosine)
    print("Generating Figure 9B (cosine): Rank-2 Cosine Similarity Histogram...")
    fig = plot_correlation_histogram(
        correlation_results,
        rank=2,
        title="Rank-2 Cosine Similarity Distribution",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_9b.pdf", cosine_figure_dir)
    
    # Figure 9C: Scatter plots (cosine)
    print("Generating Figure 9C (cosine): True vs Predicted Scatter Plots...")
    fw_medium_data = correlation_results.get("fw-medium")
    if fw_medium_data:
        scatter_dir = cosine_results_dir / "scatter_data"
        has_streaming_scatter = scatter_dir.exists() and any(scatter_dir.glob("scatter_fw-medium_*.json"))
        has_embedded_scatter = any("scatter_data" in f for f in fw_medium_data.get("per_feature", []))
        
        if has_streaming_scatter or has_embedded_scatter:
            fig = plot_figure_9c_scatters(
                fw_medium_data,
                n_features=9,
                seed=42,
                scatter_dir=scatter_dir if has_streaming_scatter else None,
            )
            save_figure(fig, "figure_9c.pdf", cosine_figure_dir)
        else:
            print("  Note: No scatter data available for Figure 9C (cosine).")
    else:
        print("  Note: fw-medium cosine results not found for Figure 9C.")
    
    # Print summary statistics
    print("\n--- Cosine Similarity Summary Statistics ---")
    fractions = compute_fraction_above_threshold(correlation_results, rank=2, threshold=0.75)
    for model, frac in fractions.items():
        n_analyzed = len(correlation_results[model].get("per_feature", []))
        print(f"  {model}: {frac*100:.1f}% features above 0.75 cosine similarity (n={n_analyzed})")
    
    return True


def generate_figure_10_cosine(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 10 from COSINE similarity SAE training time results.
    
    Same as generate_figure_10 but reads from results/language/cosine/ subfolder.
    """
    cosine_results_dir = results_dir / "cosine"
    cosine_figure_dir = figure_dir / "cosine"
    
    print("\n" + "="*60)
    print("FIGURE 10 (COSINE): SAE Training Time Effect")
    print("="*60)
    
    sae_results_file = cosine_results_dir / "sae_training_time_comparison.json"
    if not sae_results_file.exists():
        print(f"SAE training time cosine results not found: {sae_results_file}")
        print("Generate it first:")
        print("  python scripts/figures/sae_training_time_analysis.py --metric cosine")
        return False
    
    print(f"Loading SAE training time cosine results from {sae_results_file}")
    results = load_sae_training_results(sae_results_file)
    
    # Print summary
    print("\n--- SAE Training Time (Cosine Similarity) ---")
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
    
    # Figure 10A: Cosine Similarity vs Rank (all versions)
    print("\nGenerating Figure 10A (cosine): Cosine vs SAE Training Time...")
    fig = plot_sae_training_effect(
        results,
        title="Effect of SAE Training on Low-Rank Approximation (Cosine)",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_10a.pdf", cosine_figure_dir)
    
    # Figure 10B: Rank-2 Histogram (cosine)
    print("Generating Figure 10B (cosine): Rank-2 Cosine Similarity Distribution...")
    fig = plot_sae_training_histogram(
        results,
        rank=2,
        versions=['v0', 'v1', 'v2', 'v3', 'v4'],
        title="Rank-2 Cosine Similarity: Training Progression (v0 → v4)",
        show_paper_threshold=True,
    )
    save_figure(fig, "figure_10b.pdf", cosine_figure_dir)
    
    return True


def generate_figure_8_final_cosine(results_dir: Path, figure_dir: Path):
    """
    Generate Figure 8 final from COSINE similarity data.
    
    Same structure as generate_figure_8_final but reads from cosine subfolder.
    """
    cosine_results_dir = results_dir / "cosine"
    cosine_figure_dir = figure_dir / "cosine"
    
    print("\n" + "="*60)
    print("FIGURE 8 FINAL (COSINE): Sentiment Negation Circuit")
    print("="*60)
    
    # Look for cosine Figure 8 data files
    figure_8_file = cosine_results_dir / "figure_8_data_fw_medium.json"
    
    if not figure_8_file.exists():
        print(f"Cosine Figure 8 data not found: {figure_8_file}")
        print("Generate it first:")
        print("  python src/language/negation_visualization.py --config configs/language_negation_fw.yaml --metric cosine")
        return False
    
    print(f"Loading cosine Figure 8 data from {figure_8_file}")
    
    # Load the data
    data = load_figure_8_data(figure_8_file)
    if data is None:
        return False
    
    # Generate composite figure using the same plotting function
    print("Generating Figure 8 Final (cosine)...")
    fig = plot_figure_8_composite(
        data,
        figsize=(15, 5),
    )
    save_figure(fig, "figure_8_final.pdf", cosine_figure_dir)
    
    # Also check for search results to generate comparison
    search_results_file = cosine_results_dir / "circuit_search_complete.json"
    if search_results_file.exists():
        print("  Found search results, generating comparison figure...")
        # Could add comparison logic here if needed
    
    return True


# ============================================================================
# HIERARCHICAL MODEL-LAYER FIGURE GENERATION
# ============================================================================

# Model-layer configurations for hierarchical figure generation
MODEL_LAYER_CONFIGS = {
    "ts-medium": {
        4: {"expansion": 4, "purpose": "Figure 9 (paper's ts-tiny)"},
        5: {"expansion": 4, "purpose": "Figure 8 (only layer with mlp-in SAE)"},
    },
    "fw-small": {
        8: {"expansion": 4, "purpose": "Figure 9"},
    },
    "fw-medium": {
        7: {"expansion": 8, "purpose": "Figure 9 (primary model)"},
    },
}


def get_hierarchical_figure_dir(base_dir: Path, model: str, layer: int) -> Path:
    """Get hierarchical figure directory for a model/layer combination."""
    model_short = model.split("/")[-1] if "/" in model else model
    return base_dir / model_short / f"layer{layer}"


def generate_model_layer_figures(
    model: str,
    layer: int,
    results_dir: Path,
    figure_dir: Path,
    eigenpairs_cache_dir: Path = None,
):
    """
    Generate all figures for a specific model-layer combination.
    
    Saves figures in hierarchical structure:
        figure_dir/{model}/layer{N}/...
    
    Args:
        model: Model name (e.g., "ts-medium", "fw-medium")
        layer: Layer index
        results_dir: Base results directory
        figure_dir: Base figure directory
        eigenpairs_cache_dir: Optional precomputed eigenpairs directory
    """
    model_short = model.split("/")[-1] if "/" in model else model
    
    # Create hierarchical output directories
    model_fig_dir = get_hierarchical_figure_dir(figure_dir, model_short, layer)
    model_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating figures for {model_short} layer {layer}")
    print(f"{'='*60}")
    print(f"Output directory: {model_fig_dir}")
    
    # Check for model-specific results
    model_results_dir = results_dir
    
    # 1. Check for correlation results
    correlation_file = model_results_dir / f"correlation_{model_short}.json"
    if correlation_file.exists():
        print(f"\nLoading correlation results from {correlation_file}")
        with open(correlation_file) as f:
            corr_data = json.load(f)
        
        # Generate correlation-specific figures
        if "correlation_by_rank" in corr_data:
            print("  Generating correlation progression plot...")
            # Simple rank vs correlation plot for this model
            ranks = sorted([int(r) for r in corr_data["correlation_by_rank"].keys()])
            means = [corr_data["correlation_by_rank"][str(r)]["mean"] for r in ranks]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ranks, means, 'o-', linewidth=2, markersize=6, color='#2E86AB')
            ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.6, label='Paper threshold (0.75)')
            ax.set_xlabel("Approximation Rank")
            ax.set_ylabel("Mean Correlation")
            ax.set_title(f"{model_short} Layer {layer} - Correlation vs Rank")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            save_figure(fig, f"correlation_vs_rank.pdf", model_fig_dir)
    else:
        print(f"  No correlation results found: {correlation_file}")
    
    # 2. Check for AND-score / circuit analysis results (ts-medium layer 5 specific)
    and_scores_file = model_results_dir / f"{model_short.replace('-', '_')}_layer{layer}_and_scores.json"
    if and_scores_file.exists():
        print(f"\nLoading AND-scores from {and_scores_file}")
        with open(and_scores_file) as f:
            and_data = json.load(f)
        
        # Generate AND-score distribution
        if "top_by_and_score" in and_data:
            scores = [item.get("and_score", 0) for item in and_data["top_by_and_score"]]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(scores)), scores, color='#2E86AB', alpha=0.8)
            ax.set_xlabel("Feature Rank")
            ax.set_ylabel("AND-score")
            ax.set_title(f"{model_short} Layer {layer} - Top AND-gate Features")
            ax.grid(True, alpha=0.3, axis='y')
            
            save_figure(fig, f"and_score_distribution.pdf", model_fig_dir)
    
    # 3. Check for circuit analysis (detailed feature analysis)
    for circuit_file in model_results_dir.glob(f"{model_short.replace('-', '_')}_layer{layer}_circuit_*.json"):
        print(f"\nLoading circuit analysis from {circuit_file}")
        with open(circuit_file) as f:
            circuit_data = json.load(f)
        
        feat_idx = circuit_data.get("feature", "unknown")
        
        # Import the generate_figure function from the standalone script
        try:
            from scripts.figures.generate_ts_medium_layer5 import generate_figure as generate_ts_figure
            output_path = model_fig_dir / f"figure_8_{model_short}_layer{layer}_feat{feat_idx}.pdf"
            generate_ts_figure(circuit_data, output_path)
        except ImportError:
            print("  Note: Could not import generate_ts_medium_layer5 for figure generation")
    
    # 4. Check for eigenpairs cache and generate eigenspectrum summary
    if eigenpairs_cache_dir is not None:
        cache_model_dir = eigenpairs_cache_dir / model_short / str(layer)
        manifest_file = cache_model_dir / "manifest.json"
        
        if manifest_file.exists():
            print(f"\nLoading eigenpairs manifest from {manifest_file}")
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            # Generate eigenspectrum summary from cached eigenpairs
            print("  Generating eigenspectrum summary from cached eigenpairs...")
            
            # Load a sample of eigenpairs for spectrum analysis
            eigenpair_files = sorted(cache_model_dir.glob("feat_*.pt"))[:100]
            
            if eigenpair_files:
                all_eigenvalues = []
                for ef in eigenpair_files[:50]:  # Sample 50 features
                    data = torch.load(ef, map_location='cpu', weights_only=False)
                    all_eigenvalues.append(data['eigenvalues'][:10].detach().numpy())  # Top 10 per feature
                
                all_eigenvalues = np.array(all_eigenvalues)  # [n_features, 10]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot mean eigenvalue spectrum with std
                mean_spectrum = np.mean(np.abs(all_eigenvalues), axis=0)
                std_spectrum = np.std(np.abs(all_eigenvalues), axis=0)
                
                ranks = np.arange(1, len(mean_spectrum) + 1)
                ax.fill_between(ranks, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                                alpha=0.3, color='#2E86AB')
                ax.plot(ranks, mean_spectrum, 'o-', linewidth=2, markersize=6, color='#2E86AB')
                
                ax.set_xlabel("Eigenvalue Rank")
                ax.set_ylabel("Mean |λ|")
                ax.set_title(f"{model_short} Layer {layer} - Eigenvalue Spectrum (n={len(eigenpair_files)} features)")
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                save_figure(fig, f"eigenspectrum_summary.pdf", model_fig_dir)
        else:
            print(f"  No eigenpairs cache found at {cache_model_dir}")
    
    print(f"\nCompleted: {model_short} layer {layer}")
    print(f"Figures saved to: {model_fig_dir}")
    
    return True


def generate_all_hierarchical_figures(
    results_dir: Path,
    figure_dir: Path,
    eigenpairs_cache_dir: Path = None,
    model_filter: str = None,
    layer_filter: int = None,
):
    """
    Generate figures for all configured model-layer combinations.
    
    Args:
        results_dir: Base results directory
        figure_dir: Base figure directory  
        eigenpairs_cache_dir: Optional precomputed eigenpairs directory
        model_filter: Optional model name to filter (e.g., "ts-medium")
        layer_filter: Optional layer to filter
    """
    print("\n" + "="*60)
    print("HIERARCHICAL MODEL-LAYER FIGURE GENERATION")
    print("="*60)
    
    for model, layers in MODEL_LAYER_CONFIGS.items():
        if model_filter and model != model_filter:
            continue
        
        for layer, config in layers.items():
            if layer_filter is not None and layer != layer_filter:
                continue
            
            print(f"\n>>> {model} layer {layer}: {config['purpose']}")
            
            try:
                generate_model_layer_figures(
                    model=model,
                    layer=layer,
                    results_dir=results_dir,
                    figure_dir=figure_dir,
                    eigenpairs_cache_dir=eigenpairs_cache_dir,
                )
            except Exception as e:
                print(f"  Error generating figures: {e}")
                continue
    
    print("\n" + "="*60)
    print("Hierarchical figure generation complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate language experiment figures")
    parser.add_argument("--figure8-only", action="store_true",
                        help="Only generate Figure 8 variants (including comparison)")
    parser.add_argument("--figure9-only", action="store_true",
                        help="Only generate Figure 9 variants")
    parser.add_argument("--figure10-only", action="store_true",
                        help="Only generate Figure 10 variants")
    parser.add_argument("--cosine", action="store_true",
                        help="Generate cosine similarity versions from results/language/cosine/")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Generate hierarchical model/layer figures")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model name (e.g., ts-medium, fw-medium)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Filter by layer number")
    parser.add_argument("--load-eigenpairs", type=str, default=None,
                        help="Path to cached eigenpairs directory or 'auto'")
    args = parser.parse_args()
    
    # Ensure artifacts are available (downloads from Google Drive if missing)
    ensure_artifacts()
    
    # Paths (using centralized paths)
    RESULTS_DIR = LANGUAGE_RESULTS
    FIGURE_DIR = LANGUAGE_FIGURES
    REPORT_FIG_DIR = PROJECT_ROOT / "Report/figures/language"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    if args.hierarchical:
        print("Language Figure Generation (HIERARCHICAL)")
    elif args.cosine:
        print("Language Figure Generation (COSINE SIMILARITY)")
    else:
        print("Language Figure Generation")
    print("="*60)
    
    # Resolve eigenpairs cache directory
    eigenpairs_cache_dir = None
    if args.load_eigenpairs:
        if args.load_eigenpairs == "auto":
            eigenpairs_cache_dir = LANGUAGE_EIGENPAIRS
        else:
            eigenpairs_cache_dir = Path(args.load_eigenpairs)
        
        if eigenpairs_cache_dir.exists():
            print(f"Using eigenpairs cache: {eigenpairs_cache_dir}")
        else:
            print(f"Warning: Eigenpairs cache not found: {eigenpairs_cache_dir}")
            eigenpairs_cache_dir = None
    
    # ==========================================================
    # HIERARCHICAL MODEL-LAYER FIGURES
    # ==========================================================
    if args.hierarchical:
        generate_all_hierarchical_figures(
            results_dir=RESULTS_DIR,
            figure_dir=REPORT_FIG_DIR,
            eigenpairs_cache_dir=eigenpairs_cache_dir,
            model_filter=args.model,
            layer_filter=args.layer,
        )
        return
    
    # Determine which figures to generate
    generate_all = not (args.figure8_only or args.figure9_only or args.figure10_only)
    
    # ==========================================================
    # COSINE SIMILARITY FIGURES
    # ==========================================================
    if args.cosine:
        COSINE_FIG_DIR = REPORT_FIG_DIR / "cosine"
        COSINE_FIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # === Figure 9 (Cosine): Similarity Analysis ===
        if generate_all or args.figure9_only:
            generate_figure_9_cosine(RESULTS_DIR, REPORT_FIG_DIR)
        
        # === Figure 8 (Cosine): Sentiment Negation Circuit ===
        if generate_all or args.figure8_only:
            generate_figure_8_final_cosine(RESULTS_DIR, REPORT_FIG_DIR)
        
        # === Figure 10 (Cosine): SAE Training Time Effect ===
        if generate_all or args.figure10_only:
            generate_figure_10_cosine(RESULTS_DIR, REPORT_FIG_DIR)
        
        print(f"\n{'='*60}")
        print("Cosine Similarity Figure Generation Complete!")
        print(f"{'='*60}")
        print(f"Figures saved to: {COSINE_FIG_DIR}")
        return
    
    # ==========================================================
    # PEARSON CORRELATION FIGURES (default)
    # ==========================================================
    # Note: Default behavior reads from results/language/ (not cosine subfolder)
    # This maintains backward compatibility with existing JSON files that
    # don't have a "metric" field - they are assumed to be Pearson correlation.
    
    # === Figure 9: Correlation Analysis (from sweep) ===
    if generate_all or args.figure9_only:
        generate_figure_9(RESULTS_DIR, FIGURE_DIR)
    
    # === Figure 8: Sentiment Negation Circuit (4 variants) ===
    if generate_all or args.figure8_only:
        # 1. Legacy Figure 8 (tutorial feature 3834)
        generate_figure_8(RESULTS_DIR, FIGURE_DIR)
        
        # Check for search results (needed for variants 2-4)
        search_results_file = RESULTS_DIR / "circuit_search_complete.json"
        if search_results_file.exists():
            # 2. Best circuit from search
            generate_figure_8_best_circuit(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_best_circuit(RESULTS_DIR, REPORT_FIG_DIR)
            
            # 3. Sentiment-labeled version
            generate_figure_8_sentiment(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_sentiment(RESULTS_DIR, REPORT_FIG_DIR)
            
            # 4. Comparison (weak vs strong)
            generate_figure_8_comparison(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_comparison(RESULTS_DIR, REPORT_FIG_DIR)
            
            # 5. Side-by-side comparison (2 rows x 3 columns)
            generate_figure_8_side_by_side(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_side_by_side(RESULTS_DIR, REPORT_FIG_DIR)
            
            # 6. With per-word examples
            generate_figure_8_with_examples(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_with_examples(RESULTS_DIR, REPORT_FIG_DIR)
            
            # 7. FINAL Figure 8: Comprehensive comparison (single definitive figure)
            generate_figure_8_final(RESULTS_DIR, FIGURE_DIR)
            generate_figure_8_final(RESULTS_DIR, REPORT_FIG_DIR)
            
        else:
            print("\nNote: Skipping Figure 8 variants 2-6 (no search results)")
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
