"""
Language experiment plotting utilities for Section 5 reproduction.

Provides reusable functions for generating:
- Figure 9A: Correlation progression across ranks
- Figure 9B: Correlation histogram (rank-2)
- Figure 9C: True vs predicted activation scatter plots
- Figure 7: Negation circuit visualizations
- Figure A7: Correlation vs training time

Usage:
    from src.plot_utils.language import (
        plot_correlation_progression,
        plot_correlation_histogram,
        plot_correlation_scatters,
        load_correlation_results,
    )
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .style import set_publication_style

# Model colors for consistency across figures
# Note: ts-medium is the 6-layer model (paper refers to it as "ts-tiny")
MODEL_COLORS = {
    "ts-medium": "#2E86AB",    # Blue - 6 layer TinyStories
    "ts-tiny": "#2E86AB",      # Blue (alias)
    "fw-small": "#A23B72",     # Magenta - 12 layer FineWeb
    "fw-medium": "#F18F01",    # Orange - 16 layer FineWeb
}

MODEL_LABELS = {
    "ts-medium": "ts-medium (6L)",
    "ts-tiny": "ts-tiny (6L)",  # alias
    "fw-small": "fw-small (12L)",
    "fw-medium": "fw-medium (16L)",
}

MODEL_MARKERS = {
    "ts-medium": "o",
    "ts-tiny": "o",  # alias
    "fw-small": "s",
    "fw-medium": "^",
}


def load_correlation_results(results_dir: Union[str, Path]) -> Dict[str, dict]:
    """
    Load all correlation JSON files from results directory.
    
    Args:
        results_dir: Path to results/language directory
        
    Returns:
        Dict mapping model name to results dict
        e.g., {"ts-medium": {...}, "fw-small": {...}, "fw-medium": {...}}
    """
    results_dir = Path(results_dir)
    results = {}
    
    # ts-medium is the actual model name (paper calls it "ts-tiny")
    for model in ["ts-medium", "ts-tiny", "fw-small", "fw-medium"]:
        json_path = results_dir / f"correlation_{model}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[model] = json.load(f)
    
    return results


def plot_correlation_progression(
    results_dict: Dict[str, dict],
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation vs Rank",
    show_paper_threshold: bool = True,
) -> plt.Figure:
    """
    Plot correlation progression across ranks for multiple models (Figure 9A).
    
    Args:
        results_dict: Dict mapping model name to correlation results
        ax: Optional matplotlib axes (creates new figure if None)
        title: Plot title
        show_paper_threshold: Whether to show 0.75 reference line
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    for model, data in results_dict.items():
        if "summary" not in data or "correlation_by_rank" not in data["summary"]:
            continue
            
        corr_by_rank = data["summary"]["correlation_by_rank"]
        
        ranks = []
        means = []
        stds = []
        
        for rank_str, stats in sorted(corr_by_rank.items(), key=lambda x: int(x[0])):
            ranks.append(int(rank_str))
            means.append(stats["mean"])
            stds.append(stats["std"])
        
        ranks = np.array(ranks)
        means = np.array(means)
        stds = np.array(stds)
        
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        marker = MODEL_MARKERS.get(model, "o")
        
        ax.errorbar(
            ranks, means, yerr=stds,
            fmt=f"{marker}-",
            label=label,
            color=color,
            capsize=3,
            capthick=1.5,
            linewidth=2,
            markersize=7,
        )
    
    if show_paper_threshold:
        ax.axhline(y=0.75, color="#888888", linestyle="--", linewidth=1, 
                   label="Paper threshold (0.75)", alpha=0.7)
    
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to actual rank values
    all_ranks = set()
    for data in results_dict.values():
        if "summary" in data and "correlation_by_rank" in data["summary"]:
            all_ranks.update(int(r) for r in data["summary"]["correlation_by_rank"].keys())
    if all_ranks:
        ax.set_xticks(sorted(all_ranks))
    
    plt.tight_layout()
    return fig


def plot_correlation_histogram(
    results_dict: Dict[str, dict],
    rank: int = 2,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    bins: int = 20,
    show_paper_threshold: bool = True,
) -> plt.Figure:
    """
    Plot histogram of correlations at a specific rank for multiple models (Figure 9B).
    
    Args:
        results_dict: Dict mapping model name to correlation results
        rank: Rank to plot histogram for (default 2)
        ax: Optional matplotlib axes
        title: Plot title (auto-generated if None)
        bins: Number of histogram bins
        show_paper_threshold: Whether to show 0.75 reference line
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    if title is None:
        title = f"Rank-{rank} Correlation Distribution"
    
    for model, data in results_dict.items():
        if "per_feature" not in data:
            continue
        
        # Extract correlations at specified rank
        correlations = []
        for feat in data["per_feature"]:
            corr_dict = feat.get("correlations", {})
            if str(rank) in corr_dict:
                correlations.append(corr_dict[str(rank)])
        
        if not correlations:
            continue
        
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        
        ax.hist(
            correlations,
            bins=bins,
            alpha=0.6,
            label=f"{label} (n={len(correlations)})",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    
    if show_paper_threshold:
        ax.axvline(x=0.75, color="#d62728", linestyle="--", linewidth=2,
                   label="Threshold (0.75)")
    
    ax.set_xlabel(f"Rank-{rank} Correlation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(-0.5, 1.0)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_correlation_scatters(
    model_data: dict,
    n_features: int = 9,
    rank: int = 2,
    title: str = "True vs Predicted Activation",
    seed: int = 42,
) -> plt.Figure:
    """
    Plot scatter plots of true vs predicted activations for random features (Figure 9C).
    
    Args:
        model_data: Single model's correlation results (containing per_feature with scatter_data)
        n_features: Number of features to plot (should be perfect square, e.g., 9)
        rank: Rank of approximation to use
        title: Plot title
        seed: Random seed for feature selection
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    n_cols = int(np.sqrt(n_features))
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()
    
    # Get features with scatter data
    features_with_scatter = []
    for feat in model_data.get("per_feature", []):
        if "scatter_data" in feat:
            features_with_scatter.append(feat)
    
    if not features_with_scatter:
        # If no scatter data, we can only show correlations as text
        for ax in axes:
            ax.text(0.5, 0.5, "No scatter data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{title}\n(Scatter data not available)")
        plt.tight_layout()
        return fig
    
    # Randomly select features
    np.random.seed(seed)
    n_select = min(n_features, len(features_with_scatter))
    selected = np.random.choice(len(features_with_scatter), n_select, replace=False)
    
    for i, ax in enumerate(axes):
        if i >= n_select:
            ax.axis("off")
            continue
        
        feat = features_with_scatter[selected[i]]
        scatter = feat["scatter_data"]
        
        z_true = np.array(scatter.get("z_true", []))
        z_pred = np.array(scatter.get(f"z_pred_rank{rank}", scatter.get("z_pred", [])))
        
        if len(z_true) == 0 or len(z_pred) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Compute correlation
        corr = np.corrcoef(z_true, z_pred)[0, 1] if len(z_true) > 1 else 0
        
        ax.scatter(z_true, z_pred, alpha=0.3, s=10, color="#2E86AB")
        
        # Add identity line
        max_val = max(z_true.max(), z_pred.max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1)
        
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Feature {feat['feat_idx']} (r={corr:.2f})", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
    
    fig.suptitle(f"{title} (Rank-{rank} Approximation)")
    plt.tight_layout()
    return fig


def plot_interaction_submatrix(
    Q: np.ndarray,
    feature_labels: Optional[List[str]] = None,
    top_k: int = 15,
    ax: Optional[plt.Axes] = None,
    title: str = "Top Interactions",
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Plot heatmap of top interactions from Q matrix (Figure 7A).
    
    Args:
        Q: Interaction matrix [n_features, n_features]
        feature_labels: Labels for features
        top_k: Number of top interacting feature pairs to show
        ax: Optional matplotlib axes
        title: Plot title
        cmap: Colormap name
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # Find top-k interactions by absolute value
    n = Q.shape[0]
    flat_indices = np.argsort(np.abs(Q).flatten())[::-1]
    
    # Get unique features involved in top interactions
    selected_features = set()
    for idx in flat_indices:
        i, j = divmod(idx, n)
        selected_features.add(i)
        selected_features.add(j)
        if len(selected_features) >= top_k:
            break
    
    selected = sorted(list(selected_features))[:top_k]
    submatrix = Q[np.ix_(selected, selected)]
    
    # Plot heatmap
    vmax = np.abs(submatrix).max()
    im = ax.imshow(submatrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    
    # Labels
    if feature_labels:
        labels = [feature_labels[i] if i < len(feature_labels) else str(i) for i in selected]
    else:
        labels = [str(i) for i in selected]
    
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(selected)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Interaction Strength")
    
    plt.tight_layout()
    return fig


def plot_eigenvector_projections(
    projections: Dict[str, Tuple[float, float]],
    feature_types: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Projections onto Top Eigenvectors",
) -> plt.Figure:
    """
    Plot features projected onto top 2 eigenvectors (Figure 7B).
    
    Args:
        projections: Dict mapping feature index to (v1_proj, v2_proj)
        feature_types: Dict mapping feature index to type ("negation", "positive", "negative")
        ax: Optional matplotlib axes
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Default colors and markers for feature types
    type_styles = {
        "negation": {"color": "#2ca02c", "marker": "^", "label": "Negation"},
        "positive": {"color": "#1f77b4", "marker": "s", "label": "Positive sentiment"},
        "negative": {"color": "#ff7f0e", "marker": "v", "label": "Negative sentiment"},
        "other": {"color": "#7f7f7f", "marker": "o", "label": "Other"},
    }
    
    # Group by type
    by_type = {}
    for feat_idx, (v1, v2) in projections.items():
        ftype = feature_types.get(feat_idx, "other") if feature_types else "other"
        if ftype not in by_type:
            by_type[ftype] = {"v1": [], "v2": []}
        by_type[ftype]["v1"].append(v1)
        by_type[ftype]["v2"].append(v2)
    
    # Plot each type
    for ftype, coords in by_type.items():
        style = type_styles.get(ftype, type_styles["other"])
        ax.scatter(
            coords["v1"], coords["v2"],
            c=style["color"],
            marker=style["marker"],
            s=80,
            label=style["label"],
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )
    
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel("Projection onto Eigenvector 1")
    ax.set_ylabel("Projection onto Eigenvector 2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    return fig


def plot_activation_vs_approximation(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Activation vs Rank-2 Approximation",
    show_correlation: bool = True,
) -> plt.Figure:
    """
    Plot true activation vs low-rank approximation (Figure 7C).
    
    Args:
        z_true: True activations
        z_pred: Predicted activations from rank-k approximation
        ax: Optional matplotlib axes
        title: Plot title
        show_correlation: Whether to annotate with correlation value
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    
    ax.scatter(z_true, z_pred, alpha=0.4, s=15, color="#2E86AB", edgecolors="none")
    
    # Identity line
    max_val = max(z_true.max(), z_pred.max())
    min_val = min(z_true.min(), z_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1.5)
    
    if show_correlation and len(z_true) > 1:
        corr = np.corrcoef(z_true, z_pred)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.set_xlabel("True Activation")
    ax.set_ylabel("Predicted Activation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_training_progression(
    sae_results: List[dict],
    training_steps: List[int],
    rank: int = 2,
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation vs SAE Training Time",
) -> plt.Figure:
    """
    Plot correlation histograms for SAEs with different training times (Figure A7).
    
    Args:
        sae_results: List of correlation results for each SAE
        training_steps: Training step multipliers (e.g., [1, 2, 4, 8, 16])
        rank: Rank for correlation computation
        ax: Optional matplotlib axes
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    # Color gradient from dark to bright
    cmap = plt.cm.viridis
    n_saes = len(sae_results)
    
    for i, (data, steps) in enumerate(zip(sae_results, training_steps)):
        if "per_feature" not in data:
            continue
        
        # Extract correlations
        correlations = []
        for feat in data["per_feature"]:
            corr_dict = feat.get("correlations", {})
            if str(rank) in corr_dict:
                correlations.append(corr_dict[str(rank)])
        
        if not correlations:
            continue
        
        color = cmap(i / max(n_saes - 1, 1))
        
        ax.hist(
            correlations,
            bins=30,
            alpha=0.5,
            label=f"{steps}x steps (mean={np.mean(correlations):.2f})",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    
    ax.axvline(x=0.75, color="#d62728", linestyle="--", linewidth=2, alpha=0.7)
    
    ax.set_xlabel(f"Rank-{rank} Correlation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def compute_fraction_above_threshold(
    results_dict: Dict[str, dict],
    rank: int = 2,
    threshold: float = 0.75,
) -> Dict[str, float]:
    """
    Compute fraction of features above correlation threshold for each model.
    
    Args:
        results_dict: Dict mapping model name to correlation results
        rank: Rank to evaluate
        threshold: Correlation threshold
        
    Returns:
        Dict mapping model name to fraction above threshold
    """
    fractions = {}
    
    for model, data in results_dict.items():
        if "per_feature" not in data:
            continue
        
        correlations = []
        for feat in data["per_feature"]:
            corr_dict = feat.get("correlations", {})
            if str(rank) in corr_dict:
                correlations.append(corr_dict[str(rank)])
        
        if correlations:
            above = sum(1 for c in correlations if c >= threshold)
            fractions[model] = above / len(correlations)
    
    return fractions


# =============================================================================
# Figure 8: Sentiment Negation Circuit Visualization
# =============================================================================

# Feature type markers (matches paper Figure 8B)
FIGURE_8_MARKERS = {
    "negation": {"color": "#2ca02c", "marker": "^", "label": "Negation", "s": 120},
    "positive": {"color": "#ff7f0e", "marker": "v", "label": "Positive sentiment", "s": 100},
    "negative": {"color": "#1f77b4", "marker": "s", "label": "Negative sentiment", "s": 100},
    "direction": {"color": "#d62728", "marker": "*", "label": "Direction", "s": 200},
    "other": {"color": "#7f7f7f", "marker": "o", "label": "Other", "s": 60},
}


def plot_figure_8a_submatrix(
    Q_submatrix: np.ndarray,
    feature_indices: List[int],
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Plot Figure 8A: Interaction submatrix heatmap.
    
    Args:
        Q_submatrix: The submatrix containing top interactions
        feature_indices: List of feature indices in the submatrix
        ax: Optional matplotlib axes
        cmap: Colormap name
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    Q = np.array(Q_submatrix)
    vmax = np.abs(Q).max()
    
    im = ax.imshow(Q, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    
    # Feature labels
    labels = [str(f) for f in feature_indices]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    
    ax.set_title("A) Top 15 Interactions")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Interaction Strength", fontsize=9)
    
    return fig


def plot_figure_8b_projections(
    feature_projections: Dict[str, Tuple[float, float]],
    meaningful_directions: Dict[str, Tuple[float, float]],
    feature_types: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Figure 8B: Feature projections onto top eigenvectors.
    
    Args:
        feature_projections: Dict mapping feature index to (v1_proj, v2_proj)
        meaningful_directions: Dict with "bad-good" and "[BOS] not" projections
        feature_types: Optional dict mapping feature index to type
        ax: Optional matplotlib axes
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    # Plot SAE feature projections
    for feat_idx, (v1, v2) in feature_projections.items():
        ftype = feature_types.get(str(feat_idx), "other") if feature_types else "other"
        style = FIGURE_8_MARKERS.get(ftype, FIGURE_8_MARKERS["other"])
        ax.scatter(v1, v2, c=style["color"], marker=style["marker"], 
                   s=style["s"], alpha=0.8, edgecolors="white", linewidth=0.5)
    
    # Plot meaningful directions with special markers
    for dir_name, (v1, v2) in meaningful_directions.items():
        style = FIGURE_8_MARKERS["direction"]
        ax.scatter(v1, v2, c=style["color"], marker=style["marker"],
                   s=style["s"], alpha=1.0, edgecolors="black", linewidth=1,
                   label=f'"{dir_name}"', zorder=10)
        # Add text label
        ax.annotate(dir_name, (v1, v2), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold")
    
    # Add reference lines
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel("Projection onto $v_1$")
    ax.set_ylabel("Projection onto $v_2$")
    ax.set_title("B) Feature Projections")
    
    # Create legend with feature type markers
    legend_elements = []
    for ftype, style in FIGURE_8_MARKERS.items():
        if ftype == "direction":
            continue  # Skip, we add these separately
        legend_elements.append(
            plt.scatter([], [], c=style["color"], marker=style["marker"],
                       s=style["s"], label=style["label"], edgecolors="white")
        )
    ax.legend(handles=legend_elements, loc="best", fontsize=8)
    
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_figure_8c_scatter(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    feature_idx: Optional[int] = None,
) -> plt.Figure:
    """
    Plot Figure 8C: True activation vs rank-2 approximation scatter.
    
    Args:
        z_true: True SAE activations
        z_pred: Predicted activations (rank-2 approximation)
        ax: Optional matplotlib axes
        feature_idx: Optional feature index for title
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    
    z_true = np.array(z_true)
    z_pred = np.array(z_pred)
    
    # Compute correlation
    corr = np.corrcoef(z_true, z_pred)[0, 1] if len(z_true) > 1 else 0
    
    ax.scatter(z_true, z_pred, alpha=0.3, s=10, color="#2E86AB", edgecolors="none")
    
    # Identity line
    max_val = max(z_true.max(), z_pred.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1.5, label="y=x")
    
    # Correlation annotation
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))
    
    ax.set_xlabel("True Activation")
    ax.set_ylabel("Rank-2 Approximation")
    
    title = "C) Activation vs Approximation"
    if feature_idx is not None:
        title += f" (Feature {feature_idx})"
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    return fig


def plot_figure_8_composite(
    figure_8_data: dict,
    feature_types: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (14, 4.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate complete Figure 8 with all three panels.
    
    Args:
        figure_8_data: Dict containing panel_a, panel_b, panel_c data
            (as produced by negation_visualization.py)
        feature_types: Optional dict mapping feature index to type for coloring
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel A: Interaction submatrix
    panel_a = figure_8_data.get("panel_a", {})
    plot_figure_8a_submatrix(
        Q_submatrix=panel_a.get("Q_submatrix", []),
        feature_indices=panel_a.get("feature_indices", []),
        ax=axes[0],
    )
    
    # Panel B: Eigenvector projections
    panel_b = figure_8_data.get("panel_b", {})
    plot_figure_8b_projections(
        feature_projections=panel_b.get("feature_projections", {}),
        meaningful_directions=panel_b.get("meaningful_directions", {}),
        feature_types=feature_types,
        ax=axes[1],
    )
    
    # Panel C: Activation scatter
    panel_c = figure_8_data.get("panel_c", {})
    plot_figure_8c_scatter(
        z_true=panel_c.get("z_true", []),
        z_pred=panel_c.get("z_pred_rank2", []),
        ax=axes[2],
        feature_idx=figure_8_data.get("output_feature_idx"),
    )
    
    # Add main title
    model_name = figure_8_data.get("model_name", "Unknown")
    feature_idx = figure_8_data.get("output_feature_idx", "?")
    fig.suptitle(f"Sentiment Negation Circuit (Feature {feature_idx}, {model_name})", 
                 fontsize=12, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure 8 saved to: {save_path}")
    
    return fig


def plot_figure_9c_scatters(
    results: dict,
    n_features: int = 9,
    seed: int = 42,
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Figure 9C: 3x3 grid of scatter plots for random features.
    
    Args:
        results: Correlation results dict (from verify_correlation.py)
        n_features: Number of features to plot (should be perfect square)
        seed: Random seed for feature selection
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    n_cols = int(np.sqrt(n_features))
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Get features with scatter data
    features_with_scatter = []
    for feat in results.get("per_feature", []):
        if "scatter_data" in feat:
            features_with_scatter.append(feat)
    
    if not features_with_scatter:
        for ax in axes:
            ax.text(0.5, 0.5, "No scatter data\n(use --save-scatter)", 
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Figure 9C: Scatter data not available")
        plt.tight_layout()
        return fig
    
    # Randomly select features
    np.random.seed(seed)
    n_select = min(n_features, len(features_with_scatter))
    selected_indices = np.random.choice(len(features_with_scatter), n_select, replace=False)
    
    for i, ax in enumerate(axes):
        if i >= n_select:
            ax.axis("off")
            continue
        
        feat = features_with_scatter[selected_indices[i]]
        scatter = feat["scatter_data"]
        
        z_true = np.array(scatter.get("z_true", []))
        z_pred = np.array(scatter.get("z_pred_rank2", []))
        
        if len(z_true) == 0 or len(z_pred) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Compute correlation
        corr = np.corrcoef(z_true, z_pred)[0, 1] if len(z_true) > 1 else 0
        
        # Scatter plot
        ax.scatter(z_true, z_pred, alpha=0.3, s=8, color="#2E86AB", edgecolors="none")
        
        # Identity line
        max_val = max(z_true.max(), z_pred.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"Feature {feat['feat_idx']}\n(r = {corr:.2f}, n = {len(z_true)})", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    
    model_name = results.get("summary", {}).get("model_name", "Unknown")
    fig.suptitle(f"Figure 9C: True vs Predicted Activation ({model_name})", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure 9C saved to: {save_path}")
    
    return fig


def load_figure_8_data(json_path: Union[str, Path]) -> dict:
    """
    Load Figure 8 data from JSON file.
    
    Args:
        json_path: Path to figure_8_data.json
    
    Returns:
        Dict with panel_a, panel_b, panel_c data
    """
    with open(json_path) as f:
        return json.load(f)
