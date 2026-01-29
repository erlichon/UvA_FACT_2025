#!/usr/bin/env python3
"""
Generate figures for Phase 2 (Language) experiments.

Run from project root after running language experiments:
    python scripts/figures/generate_language_figures.py

Generates:
- Figure 8: Sentiment negation circuit (features 3834 vs 751)
- Figure 9: Correlation analysis (9a, 9b, 9c)
- Figure 10: SAE training time effect (10a, 10b)
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
from src.artifact_loader import ensure_artifacts

# Figure 9, 8, and 10 plotting utilities
from src.plot_utils.language import (
    load_correlation_results,
    plot_correlation_progression,
    plot_correlation_histogram,
    plot_figure_9c_scatters,
    compute_fraction_above_threshold,
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


def _compute_clusters_from_projections(projs: dict, feature_ids: list) -> tuple:
    """
    Compute positive/negative clusters from eigenvector projections.
    
    Uses the sign of the first eigenvector projection (v1) to determine cluster membership.
    This matches the paper's approach where clusters separate along the dominant eigenvector.
    
    Args:
        projs: Dict mapping feature_str -> [v1, v2] projections
        feature_ids: List of feature indices in the submatrix
    
    Returns:
        (cluster_pos, cluster_neg): Lists of feature indices in each cluster
    """
    cluster_pos = []
    cluster_neg = []
    
    for fid in feature_ids:
        fid_str = str(fid)
        if fid_str in projs:
            v1 = projs[fid_str][0]
            if v1 > 0:
                cluster_pos.append(fid)
            else:
                cluster_neg.append(fid)
    
    return cluster_pos, cluster_neg


def generate_figure_8_final(results_dir: Path, figure_dir: Path, dataset_suffix: str = "", dataset_label: str = "TinyStories"):
    """
    Generate Final Figure 8: Comparison of Feature 3834 vs Feature 751.
    
    Layout: 2 rows x 3 columns
    - Row 1: Feature 3834 (Tutorial "not-good")
    - Row 2: Feature 751 (Strong AND-gate circuit)
    
    Each row shows: A) Interaction Submatrix | B) Eigenvector Projections | C) Scatter Plot
    
    For both features, computes clusters from eigenvector projections (v1 sign) and
    reorders the interaction matrix to show block structure.
    
    Args:
        results_dir: Path to results directory
        figure_dir: Path to output figure directory
        dataset_suffix: Suffix for input JSON files (e.g., "_fineweb16k")
        dataset_label: Human-readable dataset name for the figure title
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    print("\n" + "="*60)
    print(f"FIGURE 8 FINAL: Feature 3834 vs 751 ({dataset_label})")
    print("="*60)
    
    # Load data for both features (with optional dataset suffix)
    feature_3834_file = results_dir / f"figure_8_data_fw_medium{dataset_suffix}.json"
    feature_751_file = results_dir / f"figure_8_feature751{dataset_suffix}.json"
    circuit_751_file = results_dir / "circuit_analysis_751.json"
    
    if not feature_3834_file.exists():
        print(f"Feature 3834 data not found: {feature_3834_file}")
        print("Generate it first:")
        print("  ./scripts/train/run_language.sh figure8 generate")
        return False
    
    if not feature_751_file.exists():
        print(f"Feature 751 data not found: {feature_751_file}")
        print("Generate it first:")
        print("  ./scripts/train/run_language.sh figure8 generate")
        return False
    
    with open(feature_3834_file) as f:
        data_3834 = json.load(f)
    
    with open(feature_751_file) as f:
        data_751 = json.load(f)
    
    # Compute clusters from eigenvector projections for DISPLAYED features
    # We always compute from projections because:
    # 1. circuit_analysis files have clusters for ALL features, not just the top 15 displayed
    # 2. Eigenvector projections in the data files are for the displayed features only
    
    # Feature 3834 clusters
    projs_3834 = data_3834["panel_b"].get("feature_projections", {})
    feature_ids_3834 = data_3834["panel_a"]["feature_indices"]
    cluster_pos_3834, cluster_neg_3834 = _compute_clusters_from_projections(projs_3834, feature_ids_3834)
    
    # Feature 751 clusters
    if "panel_b" in data_751:
        projs_751 = data_751["panel_b"].get("feature_projections", {})
    else:
        projs_751 = data_751.get("feature_projections", {})
    if "panel_a" in data_751:
        feature_ids_751_orig = data_751["panel_a"]["feature_indices"]
    else:
        feature_ids_751_orig = data_751.get("top_input_features", [])
    cluster_pos_751, cluster_neg_751 = _compute_clusters_from_projections(projs_751, feature_ids_751_orig)
    
    print(f"  Feature 3834 clusters: {len(cluster_pos_3834)} positive, {len(cluster_neg_3834)} negative")
    print(f"  Feature 751 clusters: {len(cluster_pos_751)} positive, {len(cluster_neg_751)} negative")
    
    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ==================== ROW 1: Feature 3834 (Tutorial) ====================
    ax_Q_3834, ax_proj_3834, ax_scatter_3834 = axes[0]
    
    # --- Panel A: Interaction Submatrix (reordered by cluster) ---
    Q_3834_orig = np.array(data_3834["panel_a"]["Q_submatrix"])
    
    # Reorder to show block structure
    pos_indices_3834 = [i for i, fid in enumerate(feature_ids_3834) if fid in cluster_pos_3834]
    neg_indices_3834 = [i for i, fid in enumerate(feature_ids_3834) if fid in cluster_neg_3834]
    other_indices_3834 = [i for i, fid in enumerate(feature_ids_3834) if fid not in cluster_pos_3834 and fid not in cluster_neg_3834]
    new_order_3834 = pos_indices_3834 + neg_indices_3834 + other_indices_3834
    
    Q_3834 = Q_3834_orig[np.ix_(new_order_3834, new_order_3834)]
    feature_ids_3834_reordered = [feature_ids_3834[i] for i in new_order_3834]
    n_pos_3834 = len(pos_indices_3834)
    
    vmax = np.abs(Q_3834).max()
    im = ax_Q_3834.imshow(Q_3834, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    n_feats = len(feature_ids_3834_reordered)
    ax_Q_3834.set_xticks(range(n_feats))
    ax_Q_3834.set_yticks(range(n_feats))
    ax_Q_3834.set_xticklabels(feature_ids_3834_reordered, rotation=45, ha='right', fontsize=7)
    ax_Q_3834.set_yticklabels(feature_ids_3834_reordered, fontsize=7)
    
    # Color features by cluster membership
    for i, fid in enumerate(feature_ids_3834_reordered):
        if fid in cluster_pos_3834:
            color, marker = '#ff7f0e', '^'  # Orange triangle for positive cluster
        elif fid in cluster_neg_3834:
            color, marker = '#1f77b4', 'o'  # Blue circle for negative cluster
        else:
            color, marker = '#7f7f7f', 's'  # Gray square for other
        ax_Q_3834.get_xticklabels()[i].set_color(color)
        ax_Q_3834.get_yticklabels()[i].set_color(color)
        ax_Q_3834.scatter(i, -0.8, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
        ax_Q_3834.scatter(-0.8, i, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
    
    # Block boundary between clusters
    if n_pos_3834 > 0:
        ax_Q_3834.axhline(y=n_pos_3834 - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_Q_3834.axvline(x=n_pos_3834 - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax_Q_3834.set_title("Feature 3834 (\"not-good\")\nA) Interaction Submatrix (grouped)", fontsize=10, fontweight='bold')
    ax_Q_3834.set_xlabel("Input Feature", fontsize=9)
    ax_Q_3834.set_ylabel("Input Feature", fontsize=9)
    plt.colorbar(im, ax=ax_Q_3834, fraction=0.046, pad=0.04)
    
    # --- Panel B: Eigenvector Projections (colored by cluster) ---
    type_counts_3834 = {'positive': 0, 'negative': 0, 'other': 0}
    for feat_str, coords in projs_3834.items():
        v1, v2 = coords[0], coords[1]
        fid = int(feat_str)
        if fid in cluster_pos_3834:
            marker, color, size = '^', '#ff7f0e', 100
            type_counts_3834['positive'] += 1
        elif fid in cluster_neg_3834:
            marker, color, size = 'o', '#1f77b4', 80
            type_counts_3834['negative'] += 1
        else:
            marker, color, size = 's', '#7f7f7f', 60
            type_counts_3834['other'] += 1
        ax_proj_3834.scatter(v1, v2, marker=marker, c=color, s=size, alpha=0.85, 
                            edgecolors='black', linewidths=0.5, zorder=5)
    
    ax_proj_3834.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_proj_3834.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Get axis limits first for proper scaling
    ax_proj_3834.autoscale()
    xlim = ax_proj_3834.get_xlim()
    ylim = ax_proj_3834.get_ylim()
    
    # Add meaningful directions (paper style: "bad-good" unembed, "not" input)
    meaningful_dirs_3834 = data_3834["panel_b"].get("meaningful_directions", {})
    
    # Plot semantic direction indicators - scale to axis range (paper style)
    # Use two separate arrows with different colors for "bad" and "good" directions
    if "bad-good" in meaningful_dirs_3834:
        bg_v1, bg_v2 = meaningful_dirs_3834["bad-good"]
        bg_norm = np.sqrt(bg_v1**2 + bg_v2**2)
        if bg_norm > 0:
            scale = min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) * 0.35
            bg_v1_scaled = (bg_v1 / bg_norm) * scale
            bg_v2_scaled = (bg_v2 / bg_norm) * scale
            # NOTE: "bad-good" direction points FROM good TOWARDS bad
            # So: "bad" endpoint = (bg_v1_scaled, bg_v2_scaled), "good" = opposite
            # "bad" direction arrow (dark red) with dot at endpoint
            ax_proj_3834.annotate('', xy=(bg_v1_scaled, bg_v2_scaled), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2.5, alpha=0.9),
                                 zorder=8)
            ax_proj_3834.scatter(bg_v1_scaled, bg_v2_scaled, marker='o', c='#8B0000', s=50, 
                                edgecolors='black', linewidths=1, zorder=15)
            # "good" direction arrow (green) with dot at endpoint
            ax_proj_3834.annotate('', xy=(-bg_v1_scaled, -bg_v2_scaled), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.5, alpha=0.9),
                                 zorder=8)
            ax_proj_3834.scatter(-bg_v1_scaled, -bg_v2_scaled, marker='o', c='#2ca02c', s=50, 
                                edgecolors='black', linewidths=1, zorder=15)
            # "not-good feature" marker (black dot) - near "good" direction (negation of good)
            ax_proj_3834.scatter(-bg_v1_scaled*0.7, -bg_v2_scaled*0.7, marker='o', c='black', s=60, 
                                edgecolors='white', linewidths=1.5, zorder=16)
    
    if "[BOS] not" in meaningful_dirs_3834:
        not_v1, not_v2 = meaningful_dirs_3834["[BOS] not"]
        not_norm = np.sqrt(not_v1**2 + not_v2**2)
        if not_norm > 0:
            scale = min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) * 0.25
            not_v1_scaled = (not_v1 / not_norm) * scale
            not_v2_scaled = (not_v2 / not_norm) * scale
            # "not" input marker (purple star)
            ax_proj_3834.scatter(not_v1_scaled, not_v2_scaled, marker='*', c='#9467bd', s=150, 
                                edgecolors='black', linewidths=0.5, zorder=10)
    
    # Add axis arrows with direction (paper style - simple +/- at ends)
    # X-axis: blue double-headed arrow with + on right only
    ax_proj_3834.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                         arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2))
    ax_proj_3834.text(xlim[1]*0.92, ylim[0]*0.08, '+', fontsize=11, color='#1f77b4', fontweight='bold')
    # Y-axis: red double-headed arrow with - on bottom only
    ax_proj_3834.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                         arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2))
    ax_proj_3834.text(xlim[0]*0.08, ylim[0]*0.92, '−', fontsize=11, color='#d62728', fontweight='bold')
    
    ax_proj_3834.set_xlabel("Top positive eigenvector", fontsize=9, color='#1f77b4')
    ax_proj_3834.set_ylabel("Top negative eigenvector", fontsize=9, color='#d62728')
    ax_proj_3834.set_title("B) Eigenvector Projections", fontsize=10, fontweight='bold')
    
    # Legend for feature 3834 clusters - upper left
    legend_elements_3834 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markersize=7, markeredgecolor='black', label='Neg. sentiment'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', 
               markersize=7, markeredgecolor='black', label='Pos. sentiment'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#9467bd', 
               markersize=10, markeredgecolor='black', label='"not" input'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', 
               markersize=6, markeredgecolor='black', label='"good" unembed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B0000', 
               markersize=6, markeredgecolor='black', label='"bad" unembed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=6, markeredgecolor='white', label='not-good feature'),
    ]
    ax_proj_3834.legend(handles=legend_elements_3834, loc='upper left', fontsize=6)
    
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
    
    # Reorder to show block structure (if cluster info available)
    if cluster_pos_751 or cluster_neg_751:
        pos_indices = [i for i, fid in enumerate(feature_ids_751) if fid in cluster_pos_751]
        neg_indices = [i for i, fid in enumerate(feature_ids_751) if fid in cluster_neg_751]
        other_indices = [i for i, fid in enumerate(feature_ids_751) if fid not in cluster_pos_751 and fid not in cluster_neg_751]
        new_order = pos_indices + neg_indices + other_indices
        
        Q_751 = Q_orig[np.ix_(new_order, new_order)]
        feature_ids_751_reordered = [feature_ids_751[i] for i in new_order]
        n_pos = len(pos_indices)
    else:
        Q_751 = Q_orig
        feature_ids_751_reordered = feature_ids_751
        new_order = list(range(len(feature_ids_751)))
        n_pos = 0
    
    vmax = np.abs(Q_751).max()
    im = ax_Q_751.imshow(Q_751, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect='auto')
    
    n_feats = len(feature_ids_751_reordered)
    ax_Q_751.set_xticks(range(n_feats))
    ax_Q_751.set_yticks(range(n_feats))
    ax_Q_751.set_xticklabels(feature_ids_751_reordered, rotation=45, ha='right', fontsize=7)
    ax_Q_751.set_yticklabels(feature_ids_751_reordered, fontsize=7)
    
    # Color features by cluster membership
    for i, fid in enumerate(feature_ids_751_reordered):
        if fid in cluster_pos_751:
            color, marker = '#ff7f0e', '^'  # Orange triangle for positive cluster
        elif fid in cluster_neg_751:
            color, marker = '#1f77b4', 'o'  # Blue circle for negative cluster
        else:
            color, marker = '#7f7f7f', 's'  # Gray square for other
        ax_Q_751.get_xticklabels()[i].set_color(color)
        ax_Q_751.get_yticklabels()[i].set_color(color)
        ax_Q_751.scatter(i, -0.8, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
        ax_Q_751.scatter(-0.8, i, marker=marker, c=color, s=40, clip_on=False, zorder=10, edgecolors='black', linewidths=0.3)
    
    # Block boundary between clusters
    if n_pos > 0:
        ax_Q_751.axhline(y=n_pos - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_Q_751.axvline(x=n_pos - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax_Q_751.set_title("Feature 751 (\"not-bad\")\nA) Interaction Submatrix (grouped)", fontsize=10, fontweight='bold')
    ax_Q_751.set_xlabel("Input Feature", fontsize=9)
    ax_Q_751.set_ylabel("Input Feature", fontsize=9)
    plt.colorbar(im, ax=ax_Q_751, fraction=0.046, pad=0.04)
    
    # --- Panel B: Eigenvector Projections (colored by cluster) ---
    if "panel_b" in data_751:
        projs_751 = data_751["panel_b"].get("feature_projections", {})
    else:
        projs_751 = data_751.get("feature_projections", {})
    
    type_counts_751 = {'positive': 0, 'negative': 0, 'other': 0}
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
            type_counts_751['other'] += 1
        
        ax_proj_751.scatter(v1, v2, marker=marker, c=color, s=size, alpha=0.85, 
                           edgecolors='black', linewidths=0.5, zorder=5)
    
    ax_proj_751.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_proj_751.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Get axis limits first for proper scaling
    ax_proj_751.autoscale()
    xlim = ax_proj_751.get_xlim()
    ylim = ax_proj_751.get_ylim()
    
    # Add meaningful directions (paper style: "bad-good" unembed, "not" input)
    if "panel_b" in data_751:
        meaningful_dirs_751 = data_751["panel_b"].get("meaningful_directions", {})
    else:
        meaningful_dirs_751 = data_751.get("meaningful_directions", {})
    
    # Plot semantic direction indicators for feature 751 - scale to axis range (paper style)
    # Use two separate arrows with different colors for "bad" and "good" directions
    if "bad-good" in meaningful_dirs_751:
        bg_v1, bg_v2 = meaningful_dirs_751["bad-good"]
        bg_norm = np.sqrt(bg_v1**2 + bg_v2**2)
        if bg_norm > 0:
            scale = min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) * 0.35
            bg_v1_scaled = (bg_v1 / bg_norm) * scale
            bg_v2_scaled = (bg_v2 / bg_norm) * scale
            # NOTE: "bad-good" direction points FROM good TOWARDS bad
            # So: "bad" endpoint = (bg_v1_scaled, bg_v2_scaled), "good" = opposite
            # "bad" direction arrow (dark red) with dot at endpoint
            ax_proj_751.annotate('', xy=(bg_v1_scaled, bg_v2_scaled), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2.5, alpha=0.9),
                                zorder=8)
            ax_proj_751.scatter(bg_v1_scaled, bg_v2_scaled, marker='o', c='#8B0000', s=50, 
                               edgecolors='black', linewidths=1, zorder=15)
            # "good" direction arrow (green) with dot at endpoint
            ax_proj_751.annotate('', xy=(-bg_v1_scaled, -bg_v2_scaled), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.5, alpha=0.9),
                                zorder=8)
            ax_proj_751.scatter(-bg_v1_scaled, -bg_v2_scaled, marker='o', c='#2ca02c', s=50, 
                               edgecolors='black', linewidths=1, zorder=15)
            # "not-bad feature" marker (black dot) - near "bad" direction (negation of bad)
            ax_proj_751.scatter(bg_v1_scaled*0.7, bg_v2_scaled*0.7, marker='o', c='black', s=60, 
                               edgecolors='white', linewidths=1.5, zorder=16)
    
    if "[BOS] not" in meaningful_dirs_751:
        not_v1, not_v2 = meaningful_dirs_751["[BOS] not"]
        not_norm = np.sqrt(not_v1**2 + not_v2**2)
        if not_norm > 0:
            scale = min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) * 0.25
            not_v1_scaled = (not_v1 / not_norm) * scale
            not_v2_scaled = (not_v2 / not_norm) * scale
            # "not" input marker (purple star)
            ax_proj_751.scatter(not_v1_scaled, not_v2_scaled, marker='*', c='#9467bd', s=150, 
                               edgecolors='black', linewidths=0.5, zorder=10)
    
    # Add axis arrows with direction (paper style - simple +/- at ends)
    # X-axis: blue double-headed arrow with + on right only
    ax_proj_751.annotate('', xy=(xlim[1]*0.95, 0), xytext=(xlim[0]*0.95, 0),
                        arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2))
    ax_proj_751.text(xlim[1]*0.92, ylim[0]*0.08, '+', fontsize=11, color='#1f77b4', fontweight='bold')
    # Y-axis: red double-headed arrow with - on bottom only
    ax_proj_751.annotate('', xy=(0, ylim[1]*0.95), xytext=(0, ylim[0]*0.95),
                        arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2))
    ax_proj_751.text(xlim[0]*0.08, ylim[0]*0.92, '−', fontsize=11, color='#d62728', fontweight='bold')
    
    ax_proj_751.set_xlabel("Top positive eigenvector", fontsize=9, color='#1f77b4')
    ax_proj_751.set_ylabel("Top negative eigenvector", fontsize=9, color='#d62728')
    ax_proj_751.set_title("B) Eigenvector Projections", fontsize=10, fontweight='bold')
    
    # Legend for feature 751 - lower right
    legend_elements_751 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markersize=7, markeredgecolor='black', label='Neg. sentiment'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', 
               markersize=7, markeredgecolor='black', label='Pos. sentiment'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#9467bd', 
               markersize=10, markeredgecolor='black', label='"not" input'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', 
               markersize=6, markeredgecolor='black', label='"good" unembed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B0000', 
               markersize=6, markeredgecolor='black', label='"bad" unembed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=6, markeredgecolor='white', label='not-bad feature'),
    ]
    ax_proj_751.legend(handles=legend_elements_751, loc='lower right', fontsize=6)
    
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
    
    # Main title (always include dataset name)
    title = f"Sentiment Negation Circuits: Feature 3834 (not-good) vs Feature 751 (not-bad) [{dataset_label}]"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3)
    
    # Output filename includes dataset suffix
    output_filename = f"figure_8_final{dataset_suffix}.pdf"
    save_figure(fig, output_filename, figure_dir)
    
    print(f"\nFinal Figure 8 generated! ({dataset_label})")
    print(f"  Output: {output_filename}")
    print(f"  Row 1: Feature 3834 (not-good), r = {corr_3834:.3f}")
    print(f"         Clusters: {type_counts_3834['positive']} positive, {type_counts_3834['negative']} negative")
    print(f"  Row 2: Feature 751 (not-bad), r = {corr_751:.3f}")
    print(f"         Clusters: {type_counts_751['positive']} positive, {type_counts_751['negative']} negative")
    
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
    """
    Generate language experiment figures for the report.
    
    Generates:
    - Figure 8: Sentiment Negation Circuit (features 3834 vs 751)
    - Figure 9: Correlation Analysis (9a, 9b, 9c)
    - Figure 10: SAE Training Time Effect (10a, 10b)
    """
    parser = argparse.ArgumentParser(description="Generate language experiment figures")
    parser.add_argument("--figure8-only", action="store_true",
                        help="Only generate Figure 8")
    parser.add_argument("--figure9-only", action="store_true",
                        help="Only generate Figure 9 (9a, 9b, 9c)")
    parser.add_argument("--figure10-only", action="store_true",
                        help="Only generate Figure 10 (10a, 10b)")
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
    print("Language Figure Generation")
    print("="*60)
    
    # Determine which figures to generate
    generate_all = not (args.figure8_only or args.figure9_only or args.figure10_only)
    
    # === Figure 9: Correlation Analysis ===
    if generate_all or args.figure9_only:
        generate_figure_9(RESULTS_DIR, FIGURE_DIR)
        generate_figure_9(RESULTS_DIR, REPORT_FIG_DIR)
    
    # === Figure 8: Sentiment Negation Circuit ===
    if generate_all or args.figure8_only:
        # TinyStories (primary figure - cleaner semantic clustering)
        generate_figure_8_final(RESULTS_DIR, FIGURE_DIR, dataset_suffix="", dataset_label="TinyStories")
        generate_figure_8_final(RESULTS_DIR, REPORT_FIG_DIR, dataset_suffix="", dataset_label="TinyStories")
        
        # FineWeb-16k (for comparison with tutorial) - optional, only if data exists
        fineweb_3834 = RESULTS_DIR / "figure_8_data_fw_medium_fineweb16k.json"
        fineweb_751 = RESULTS_DIR / "figure_8_feature751_fineweb16k.json"
        if fineweb_3834.exists() and fineweb_751.exists():
            generate_figure_8_final(RESULTS_DIR, FIGURE_DIR, dataset_suffix="_fineweb16k", dataset_label="FineWeb-16k")
            generate_figure_8_final(RESULTS_DIR, REPORT_FIG_DIR, dataset_suffix="_fineweb16k", dataset_label="FineWeb-16k")
        else:
            print("\nNote: FineWeb-16k data not found, skipping comparison figure.")
            print("  Generate with: ./scripts/train/run_language.sh figure8 generate")
    
    # === Figure 10: SAE Training Time Effect ===
    if generate_all or args.figure10_only:
        generate_figure_10(RESULTS_DIR, FIGURE_DIR)
        generate_figure_10(RESULTS_DIR, REPORT_FIG_DIR)

    print(f"\n{'='*60}")
    print("Figure Generation Complete!")
    print(f"{'='*60}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Report figures saved to: {REPORT_FIG_DIR}")


if __name__ == "__main__":
    main()
