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

# Model colors for consistency across figures (paper-style)
# Note: ts-medium is the 6-layer model (paper refers to it as "ts-tiny")
MODEL_COLORS = {
    "ts-medium": "#2ca02c",    # Green - 6 layer TinyStories (ts-tiny in paper)
    "ts-tiny": "#2ca02c",      # Green (alias)
    "fw-small": "#1f77b4",     # Blue - 12 layer FineWeb
    "fw-medium": "#d62728",    # Red - 16 layer FineWeb
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
    title: str = "Feature activation approximation",
    show_paper_threshold: bool = False,
    figsize: Tuple[float, float] = (6, 5),
    y_min: float = 0.0,
) -> plt.Figure:
    """
    Plot correlation progression across ranks for multiple models (Figure 9A).
    
    Matches the paper's clean style: smooth lines, no markers, zoomed y-axis.
    
    Args:
        results_dict: Dict mapping model name to correlation results
        ax: Optional matplotlib axes (creates new figure if None)
        title: Plot title
        show_paper_threshold: Whether to show 0.75 reference line
        figsize: Figure size
        y_min: Minimum y-axis value (paper uses ~0.6 to zoom in)
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    for model, data in results_dict.items():
        if "summary" not in data or "correlation_by_rank" not in data["summary"]:
            continue
            
        corr_by_rank = data["summary"]["correlation_by_rank"]
        
        ranks = []
        means = []
        
        for rank_str, stats in sorted(corr_by_rank.items(), key=lambda x: int(x[0])):
            ranks.append(int(rank_str))
            means.append(stats["mean"])
        
        ranks = np.array(ranks)
        means = np.array(means)
        
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        
        # Paper style: smooth lines, no markers
        ax.plot(
            ranks, means,
            label=label,
            color=color,
            linewidth=2.5,
        )
    
    if show_paper_threshold:
        ax.axhline(y=0.75, color="#888888", linestyle="--", linewidth=1.5, 
                   label="Paper threshold (0.75)", alpha=0.7)
    
    ax.set_xlabel("Top eigenvectors", fontsize=11)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_title(title, fontsize=12)
    
    # Zoom in like the paper (y-axis from y_min to 1.0)
    ax.set_ylim(y_min, 1.0)
    
    # Clean x-ticks (every 10 or so, not every single value)
    all_ranks = set()
    for data in results_dict.values():
        if "summary" in data and "correlation_by_rank" in data["summary"]:
            all_ranks.update(int(r) for r in data["summary"]["correlation_by_rank"].keys())
    if all_ranks:
        max_rank = max(all_ranks)
        # Show ticks at 0, 10, 20, 30... or similar
        tick_step = 10 if max_rank > 20 else 5
        ax.set_xticks([0] + list(range(tick_step, max_rank + 1, tick_step)))
        ax.set_xlim(0, max_rank)
    
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def plot_correlation_histogram(
    results_dict: Dict[str, dict],
    rank: int = 2,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    bins: int = 30,
    show_paper_threshold: bool = True,
    figsize: Tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Plot histogram of correlations at a specific rank for multiple models (Figure 9B).
    
    Uses step histogram for better readability when comparing multiple distributions.
    
    Args:
        results_dict: Dict mapping model name to correlation results
        rank: Rank to plot histogram for (default 2)
        ax: Optional matplotlib axes
        title: Plot title (auto-generated if None)
        bins: Number of histogram bins
        show_paper_threshold: Whether to show 0.75 reference line
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if title is None:
        title = f"Rank-{rank} Correlation Distribution"
    
    # Line styles for visual distinction
    line_styles = {
        'ts-medium': '-',
        'ts-tiny': '-',
        'fw-small': '--',
        'fw-medium': '-.',
    }
    
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
        linestyle = line_styles.get(model, '-')
        
        # Compute statistics
        mean_corr = np.mean(correlations)
        pct_above = np.mean(np.array(correlations) > 0.75) * 100
        
        # Use step histogram for cleaner visualization
        counts, bin_edges = np.histogram(correlations, bins=bins, range=(-0.5, 1.0))
        ax.stairs(counts, bin_edges, color=color, linewidth=2.5,
                  linestyle=linestyle,
                  label=f"{label} (n={len(correlations)}, mean={mean_corr:.2f}, {pct_above:.0f}%>0.75)")
    
    if show_paper_threshold:
        ax.axvline(x=0.75, color="#d62728", linestyle="--", linewidth=2.5,
                   label="Paper threshold (0.75)")
    
    ax.set_xlabel(f"Rank-{rank} Correlation", fontsize=11)
    ax.set_ylabel("Number of Features", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.5, 1.0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
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

# Feature type markers (matches paper Figure 8B style)
# Paper uses: blue squares (negative sentiment), green triangles (negation), orange triangles (positive)
# Extended for fw-medium which has additional categories due to different training data
FIGURE_8_MARKERS = {
    # Paper's original categories
    "negation": {"color": "#2ca02c", "marker": "^", "label": "Negation", "s": 120},
    "positive": {"color": "#ff7f0e", "marker": "v", "label": "Positive sentiment", "s": 100},
    "negative": {"color": "#1f77b4", "marker": "s", "label": "Negative sentiment", "s": 100},
    # Additional categories for fw-medium (FineWeb-EDU model)
    "structural": {"color": "#9467bd", "marker": "o", "label": "Structural", "s": 80},
    "contrast": {"color": "#e377c2", "marker": "D", "label": "Contrast", "s": 80},
    # Generic cluster types (for circuits discovered via interaction analysis)
    "cluster1": {"color": "#d62728", "marker": "v", "label": "Opposing (×−)", "s": 120},
    "cluster2": {"color": "#2ca02c", "marker": "^", "label": "Boosting (×+)", "s": 120},
    # Special markers
    "direction": {"color": "#d62728", "marker": "*", "label": "Direction", "s": 200},
    "other": {"color": "#7f7f7f", "marker": "o", "label": "Other", "s": 60},
}


def plot_figure_8a_submatrix(
    Q_submatrix: np.ndarray,
    feature_indices: List[int],
    feature_types: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdBu_r",
    sort_by_type: bool = True,
) -> plt.Figure:
    """
    Plot Figure 8A: Interaction submatrix heatmap with colored markers.
    
    Args:
        Q_submatrix: The submatrix containing top interactions
        feature_indices: List of feature indices in the submatrix
        feature_types: Optional dict mapping feature index to type for coloring/sorting
        ax: Optional matplotlib axes
        cmap: Colormap name
        sort_by_type: Whether to sort features by semantic type
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    Q = np.array(Q_submatrix)
    feature_indices = list(feature_indices)  # Ensure it's a list
    
    # Sort features by type if requested and feature_types provided
    if sort_by_type and feature_types:
        # Sorting order: negative first, then positive, then negation, then other
        # This matches the paper's Figure 8A layout
        type_order = {"negative": 0, "positive": 1, "negation": 2, "structural": 3, "contrast": 4, "other": 5}
        
        # Create (original_idx, feature_id, sort_key) tuples
        indexed_features = [
            (i, f, type_order.get(feature_types.get(str(f), "other"), 5))
            for i, f in enumerate(feature_indices)
        ]
        # Sort by type, then by feature index for stability
        indexed_features.sort(key=lambda x: (x[2], x[1]))
        
        # Get the reordering
        sort_order = [x[0] for x in indexed_features]
        sorted_features = [x[1] for x in indexed_features]
        
        # Reorder the Q matrix
        Q = Q[np.ix_(sort_order, sort_order)]
        feature_indices = sorted_features
    
    vmax = np.abs(Q).max()
    
    im = ax.imshow(Q, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    
    # Set up tick positions
    n = len(feature_indices)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    # Create simple numeric labels
    ax.set_xticklabels([str(f) for f in feature_indices], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([str(f) for f in feature_indices], fontsize=7)
    
    # Add colored markers next to labels using scatter plots on a secondary axis
    if feature_types:
        # Marker offset from the heatmap edge
        marker_offset = -0.8
        
        for i, f in enumerate(feature_indices):
            ftype = feature_types.get(str(f), "other")
            style = FIGURE_8_MARKERS.get(ftype, FIGURE_8_MARKERS["other"])
            
            # Y-axis markers (left side)
            ax.scatter(marker_offset, i, c=style["color"], marker=style["marker"], 
                      s=40, clip_on=False, zorder=10)
            
            # X-axis markers (top)
            ax.scatter(i, marker_offset, c=style["color"], marker=style["marker"], 
                      s=40, clip_on=False, zorder=10)
    
    ax.set_title("A) Top Interactions")
    
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
    # Use different colors for different direction types
    direction_styles = {
        "bad-good": {"color": "#d62728", "marker": "*", "s": 250},  # Red star
        "[BOS] not": {"color": "#d62728", "marker": "*", "s": 250},  # Red star
        "good": {"color": "#ff7f0e", "marker": "P", "s": 150},  # Orange plus
        "bad": {"color": "#1f77b4", "marker": "X", "s": 150},  # Blue X
    }
    default_style = {"color": "#d62728", "marker": "*", "s": 200}
    
    for dir_name, (v1, v2) in meaningful_directions.items():
        style = direction_styles.get(dir_name, default_style)
        ax.scatter(v1, v2, c=style["color"], marker=style["marker"],
                   s=style["s"], alpha=1.0, edgecolors="black", linewidth=1,
                   label=f'"{dir_name}"', zorder=10)
        # Add text label with offset based on position
        offset = (5, 5) if v1 >= 0 else (-40, 5)
        ax.annotate(dir_name, (v1, v2), xytext=offset, textcoords="offset points",
                    fontsize=8, fontweight="bold", color=style["color"])
    
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
    show_legend: bool = True,
) -> plt.Figure:
    """
    Plot Figure 8C: True activation vs rank-2 approximation scatter.
    
    The dotted line (y=x) represents perfect correlation - where the rank-2
    approximation would exactly match the true activation. Points above the
    line indicate overprediction, points below indicate underprediction.
    
    Args:
        z_true: True SAE activations
        z_pred: Predicted activations (rank-2 approximation)
        ax: Optional matplotlib axes
        feature_idx: Optional feature index for title
        show_legend: Whether to show legend explaining the y=x line
    
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
    
    # Scatter plot
    ax.scatter(z_true, z_pred, alpha=0.3, s=10, color="#2E86AB", edgecolors="none",
               label="Active samples")
    
    # Identity line (y=x) - represents perfect prediction
    max_val = max(z_true.max(), z_pred.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.6, linewidth=1.5, 
            label="y = x (perfect correlation)")
    
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
    
    # Show legend explaining the y=x line
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    return fig


def plot_figure_8_composite(
    figure_8_data: dict,
    feature_types: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (14, 4.5),
    save_path: Optional[str] = None,
    sort_by_type: bool = True,
) -> plt.Figure:
    """
    Generate complete Figure 8 with all three panels.
    
    Args:
        figure_8_data: Dict containing panel_a, panel_b, panel_c data
            (as produced by negation_visualization.py)
        feature_types: Optional dict mapping feature index to type for coloring.
            If not provided, tries to read from figure_8_data["feature_types"]
        figsize: Figure size
        save_path: Optional path to save the figure
        sort_by_type: Whether to sort Panel A features by semantic type
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Try to get feature_types from data if not provided
    if feature_types is None:
        feature_types = figure_8_data.get("feature_types", None)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel A: Interaction submatrix (with type sorting and markers)
    panel_a = figure_8_data.get("panel_a", {})
    plot_figure_8a_submatrix(
        Q_submatrix=panel_a.get("Q_submatrix", []),
        feature_indices=panel_a.get("feature_indices", []),
        feature_types=feature_types,
        ax=axes[0],
        sort_by_type=sort_by_type,
    )
    
    # Panel B: Eigenvector projections (with type coloring)
    panel_b = figure_8_data.get("panel_b", {})
    plot_figure_8b_projections(
        feature_projections=panel_b.get("feature_projections", {}),
        meaningful_directions=panel_b.get("meaningful_directions", {}),
        feature_types=feature_types,
        ax=axes[1],
    )
    
    # Panel C: Activation scatter (with legend explaining y=x line)
    panel_c = figure_8_data.get("panel_c", {})
    plot_figure_8c_scatter(
        z_true=panel_c.get("z_true", []),
        z_pred=panel_c.get("z_pred_rank2", []),
        ax=axes[2],
        feature_idx=figure_8_data.get("output_feature_idx"),
        show_legend=True,
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


def load_scatter_from_streaming_dir(
    scatter_dir: Union[str, Path],
    model_name: Optional[str] = None,
) -> List[dict]:
    """
    Load scatter data from streaming files in a directory.
    
    Args:
        scatter_dir: Directory containing scatter_*.json files
        model_name: Optional model name to filter files (e.g., "fw-medium")
    
    Returns:
        List of scatter data dicts, each containing feat_idx, z_true, z_pred_rank2
    """
    scatter_dir = Path(scatter_dir)
    if not scatter_dir.exists():
        return []
    
    scatter_files = list(scatter_dir.glob("scatter_*.json"))
    if model_name:
        # Filter by model name if provided
        scatter_files = [f for f in scatter_files if model_name in f.name]
    
    scatter_data = []
    for f in scatter_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                scatter_data.append(data)
        except (json.JSONDecodeError, IOError):
            continue
    
    return scatter_data


def plot_figure_9c_scatters(
    results: dict,
    n_features: int = 9,
    seed: int = 42,
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None,
    scatter_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot Figure 9C: 3x3 grid of scatter plots for random features.
    
    Supports both embedded scatter data (legacy) and streaming scatter files (preferred).
    
    Args:
        results: Correlation results dict (from verify_correlation.py)
        n_features: Number of features to plot (should be perfect square)
        seed: Random seed for feature selection
        figsize: Figure size
        save_path: Optional path to save the figure
        scatter_dir: Directory containing streaming scatter files (preferred over embedded)
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    n_cols = int(np.sqrt(n_features))
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    model_name = results.get("summary", {}).get("model_name", "Unknown")
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Try to load from streaming files first (memory-efficient approach)
    features_with_scatter = []
    
    if scatter_dir:
        scatter_dir = Path(scatter_dir)
        streaming_data = load_scatter_from_streaming_dir(scatter_dir, model_short)
        if streaming_data:
            print(f"  Loaded {len(streaming_data)} scatter files from {scatter_dir}")
            for data in streaming_data:
                features_with_scatter.append({
                    "feat_idx": data.get("feat_idx"),
                    "scatter_data": {
                        "z_true": data.get("z_true", []),
                        "z_pred_rank2": data.get("z_pred_rank2", []),
                    },
                    "correlation_rank2": data.get("correlation_rank2"),
                })
    
    # Fallback: try embedded scatter data (legacy)
    if not features_with_scatter:
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
    
    # Stratified sampling: select features from different correlation ranges
    # This gives a representative view of the distribution (high, medium, low correlations)
    np.random.seed(seed)
    
    # Sort features by correlation
    features_with_corr = []
    for i, feat in enumerate(features_with_scatter):
        corr = feat.get("correlation_rank2")
        if corr is None:
            # Compute correlation if not stored
            scatter = feat.get("scatter_data", {})
            z_true = np.array(scatter.get("z_true", []))
            z_pred = np.array(scatter.get("z_pred_rank2", []))
            if len(z_true) > 1 and len(z_pred) > 1:
                corr = np.corrcoef(z_true, z_pred)[0, 1]
            else:
                corr = 0.0
        features_with_corr.append((i, corr, feat))
    
    # Sort by correlation (descending)
    features_with_corr.sort(key=lambda x: x[1], reverse=True)
    
    # Stratified selection: 3 high (r > 0.6), 3 medium (0.2 < r < 0.5), 3 low (r < 0.15)
    n_per_group = n_features // 3
    selected_indices = []
    
    # High correlation group (r > 0.6)
    high_corr = [x for x in features_with_corr if x[1] > 0.6]
    if len(high_corr) >= n_per_group:
        high_selected = np.random.choice(len(high_corr), n_per_group, replace=False)
        selected_indices.extend([high_corr[i][0] for i in high_selected])
    else:
        selected_indices.extend([x[0] for x in high_corr[:n_per_group]])
    
    # Medium correlation group (0.2 < r < 0.5)
    medium_corr = [x for x in features_with_corr if 0.2 < x[1] < 0.5]
    if len(medium_corr) >= n_per_group:
        med_selected = np.random.choice(len(medium_corr), n_per_group, replace=False)
        selected_indices.extend([medium_corr[i][0] for i in med_selected])
    else:
        selected_indices.extend([x[0] for x in medium_corr[:n_per_group]])
    
    # Low correlation group (r < 0.15)
    low_corr = [x for x in features_with_corr if x[1] < 0.15]
    if len(low_corr) >= n_per_group:
        low_selected = np.random.choice(len(low_corr), n_per_group, replace=False)
        selected_indices.extend([low_corr[i][0] for i in low_selected])
    else:
        selected_indices.extend([x[0] for x in low_corr[:n_per_group]])
    
    # Fill remaining slots if needed
    n_select = min(n_features, len(selected_indices))
    if n_select < n_features:
        remaining = [x[0] for x in features_with_corr if x[0] not in selected_indices]
        n_remaining = min(n_features - n_select, len(remaining))
        if n_remaining > 0:
            extra = np.random.choice(len(remaining), n_remaining, replace=False)
            selected_indices.extend([remaining[i] for i in extra])
    
    # Create a lookup from original index to correlation
    idx_to_corr = {x[0]: x[1] for x in features_with_corr}
    
    # Sort selected by correlation for visual ordering (high to low)
    selected_with_corr = [(i, idx_to_corr.get(i, 0)) for i in selected_indices[:n_features]]
    selected_with_corr.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in selected_with_corr]
    n_select = len(selected_indices)
    
    for i, ax in enumerate(axes):
        if i >= n_select:
            ax.axis("off")
            continue
        
        feat = features_with_scatter[selected_indices[i]]
        scatter = feat.get("scatter_data", {})
        
        z_true = np.array(scatter.get("z_true", []))
        z_pred = np.array(scatter.get("z_pred_rank2", []))
        
        if len(z_true) == 0 or len(z_pred) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Compute correlation
        corr = np.corrcoef(z_true, z_pred)[0, 1] if len(z_true) > 1 else 0
        
        # Scatter plot
        ax.scatter(z_true, z_pred, alpha=0.3, s=8, color="#2E86AB", edgecolors="none")
        
        # Identity line (y=x)
        max_val = max(z_true.max(), z_pred.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1)
        
        # Labels - match paper style ("Activation" / "Approximation")
        ax.set_xlabel("Activation", fontsize=8)
        ax.set_ylabel("Approximation", fontsize=8)
        # Title with just correlation (paper style)
        ax.set_title(f"r = {corr:.2f}", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    
    fig.suptitle(f"Rank-2 Correlation Scatter Plots ({model_short})", fontsize=12)
    
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


# =============================================================================
# Figure 10: SAE Training Time Effect
# =============================================================================

# Colors for SAE versions (gradient from light to dark blue - more distinct)
SAE_VERSION_COLORS = {
    'v0': '#a6cee3',  # Light blue - under-trained (more visible)
    'v1': '#7eb8da',  # Medium-light blue
    'v2': '#4292c6',  # Medium blue
    'v3': '#2171b5',  # Medium-dark blue
    'v4': '#084594',  # Dark blue - well-trained
}

SAE_VERSION_LABELS = {
    'v0': 'v0 (1× training)',
    'v1': 'v1 (2× training)',
    'v2': 'v2 (4× training)',
    'v3': 'v3 (8× training)',
    'v4': 'v4 (16× training)',
}


def load_sae_training_results(json_path: Union[str, Path]) -> dict:
    """
    Load SAE training time comparison results.
    
    Args:
        json_path: Path to sae_training_time_comparison.json
    
    Returns:
        Dict with metadata and per-version results
    """
    with open(json_path) as f:
        return json.load(f)


def plot_sae_training_effect(
    results: dict,
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation vs SAE Training Time",
    show_paper_threshold: bool = True,
    figsize: Tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Plot Figure 10A: Correlation vs rank for different SAE training durations.
    
    Shows how correlation improves with longer SAE training times (v0 -> v4).
    
    Args:
        results: Dict from sae_training_time_comparison.json
        ax: Optional matplotlib axes
        title: Plot title
        show_paper_threshold: Whether to show 0.75 reference line
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    versions_data = results.get('versions', {})
    
    # Line styles for clarity - v0 more visible with thicker line
    line_styles = {
        'v0': {'linestyle': '-', 'marker': 'o', 'linewidth': 2.0, 'markersize': 7},
        'v1': {'linestyle': '--', 'marker': 's', 'linewidth': 2.0, 'markersize': 7},
        'v2': {'linestyle': '-.', 'marker': '^', 'linewidth': 2.2, 'markersize': 7},
        'v3': {'linestyle': ':', 'marker': 'D', 'linewidth': 2.5, 'markersize': 7},
        'v4': {'linestyle': '-', 'marker': 'v', 'linewidth': 2.8, 'markersize': 8},
    }
    
    for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
        if version not in versions_data:
            continue
        
        v_data = versions_data[version]
        summary = v_data.get('summary', {})
        
        ranks = []
        means = []
        
        # Extract data for each rank, sorting by numeric rank value (not alphabetically)
        rank_items = [(int(key.replace('rank_', '')), stats) 
                      for key, stats in summary.items() 
                      if key.startswith('rank_')]
        rank_items.sort(key=lambda x: x[0])  # Sort by numeric rank
        
        for rank, stats in rank_items:
            ranks.append(rank)
            means.append(stats['mean'])
        
        if not ranks:
            continue
        
        ranks = np.array(ranks)
        means = np.array(means)
        
        color = SAE_VERSION_COLORS.get(version, '#333333')
        label = SAE_VERSION_LABELS.get(version, version)
        style = line_styles.get(version, {})
        
        # Plot clean lines without error bands for readability
        ax.plot(ranks, means, color=color, label=label, **style)
    
    if show_paper_threshold:
        ax.axhline(y=0.75, color='#d62728', linestyle='--', linewidth=2,
                   label='Paper threshold (0.75)', alpha=0.8)
    
    # Get metric label from metadata (default to Pearson if not present)
    metadata = results.get('metadata', {})
    metric_label = metadata.get('metric_label', 'Pearson correlation')
    
    ax.set_xlabel('Approximation Rank (k)', fontsize=11)
    ax.set_ylabel(f'Mean {metric_label}', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9, handlelength=1.5)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks with log-scale spacing to handle [1, 2, 4, 8, 16, 30] properly
    all_ranks = set()
    for v_data in versions_data.values():
        summary = v_data.get('summary', {})
        for key in summary.keys():
            if key.startswith('rank_'):
                all_ranks.add(int(key.replace('rank_', '')))
    if all_ranks:
        sorted_ranks = sorted(all_ranks)
        ax.set_xticks(sorted_ranks)
        ax.set_xticklabels([str(r) for r in sorted_ranks])
        ax.set_xscale('log', base=2)  # Log scale makes spacing more readable
    
    plt.tight_layout()
    return fig


def plot_sae_training_histogram(
    results: dict,
    rank: int = 2,
    versions: List[str] = ['v0', 'v1', 'v2', 'v3', 'v4'],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    bins: int = 30,
    show_paper_threshold: bool = True,
    figsize: Tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Plot Figure 10B: Histogram of correlations at specified rank for selected SAE versions.
    
    Shows the bimodal → unimodal distribution shift with training time.
    Uses step histograms for better readability when comparing versions.
    Shows progression from under-trained (v0) to well-trained (v4) with progressive opacity.
    
    Args:
        results: Dict from sae_training_time_comparison.json
        rank: Rank to plot histogram for
        versions: Which SAE versions to show (default: all v0-v4 for progression)
        ax: Optional matplotlib axes
        title: Plot title
        bins: Number of histogram bins
        show_paper_threshold: Whether to show 0.75 reference line
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        set_publication_style()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if title is None:
        title = f'Rank-{rank} Correlation Distribution by Training Time'
    
    versions_data = results.get('versions', {})
    
    # Progressive line styles and alpha - more visible as training increases
    line_styles = {'v0': '-', 'v1': '--', 'v2': '-.', 'v3': ':', 'v4': '-'}
    # Progressive alpha: v0 more visible now, v4 fully opaque
    alpha_values = {'v0': 0.65, 'v1': 0.72, 'v2': 0.8, 'v3': 0.9, 'v4': 1.0}
    # Progressive line widths: thicker for better-trained
    linewidth_values = {'v0': 1.8, 'v1': 2.0, 'v2': 2.2, 'v3': 2.4, 'v4': 2.6}
    
    for version in versions:
        if version not in versions_data:
            continue
        
        v_data = versions_data[version]
        per_feature = v_data.get('per_feature', [])
        
        # Extract correlations at specified rank
        correlations = []
        rank_key = f'rank_{rank}'
        for feat in per_feature:
            if rank_key in feat:
                correlations.append(feat[rank_key])
        
        if not correlations:
            continue
        
        correlations = np.array(correlations)
        color = SAE_VERSION_COLORS.get(version, '#333333')
        label = SAE_VERSION_LABELS.get(version, version)
        alpha = alpha_values.get(version, 1.0)
        linewidth = linewidth_values.get(version, 2.0)
        
        # Compute statistics
        mean_corr = np.mean(correlations)
        pct_above = np.mean(correlations > 0.75) * 100
        n_features = len(correlations)
        
        # Use step histogram for clearer visualization
        counts, bin_edges = np.histogram(correlations, bins=bins, range=(-0.5, 1.0))
        ax.stairs(counts, bin_edges, color=color, linewidth=linewidth,
                  linestyle=line_styles.get(version, '-'),
                  alpha=alpha,
                  label=f'{label} (n={n_features}, mean={mean_corr:.2f}, {pct_above:.0f}%>0.75)')
    
    if show_paper_threshold:
        ax.axvline(x=0.75, color='#d62728', linestyle='--', linewidth=2.5,
                   label='Paper threshold (0.75)')
    
    ax.set_xlabel(f'Rank-{rank} Correlation', fontsize=11)
    ax.set_ylabel('Number of Features', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.5, 1.0)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9, handlelength=1.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_figure_10_composite(
    results: dict,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate complete Figure 10 with both panels.
    
    Args:
        results: Dict from sae_training_time_comparison.json
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Correlation vs Rank (all versions)
    plot_sae_training_effect(results, ax=axes[0], 
                             title='A) Correlation vs Approximation Rank')
    
    # Panel B: Histogram (v0 vs v4)
    plot_sae_training_histogram(results, rank=2, versions=['v0', 'v4'],
                                ax=axes[1], title='B) Rank-2 Correlation Distribution')
    
    fig.suptitle('Effect of SAE Training Time on Correlation Quality', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure 10 saved to: {save_path}')
    
    return fig
