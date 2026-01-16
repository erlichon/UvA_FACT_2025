"""
Interaction Matrix Visualization for Language Models.

Provides tools for visualizing Q matrices (interaction structure between SAE features)
and histograms for verifying the paper's 69% claim about rank-2 variance explained.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union
import torch
from torch import Tensor
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_interaction_matrix(
    Q: Tensor,
    feature_idx: int = 0,
    top_k: int = 50,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 7),
    save_path: Optional[str] = None,
    symmetric: bool = True,
) -> "plt.Figure":
    """
    Visualize interaction matrix Q as a heatmap.

    Shows which input feature pairs interact strongly to produce the output feature.

    Args:
        Q: Interaction matrix [n_input, n_input] or [batch, n_input, n_input]
        feature_idx: If Q is batched, which feature to show
        top_k: Show only top-k interacting features (by row/col magnitude)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure
        symmetric: Whether to show symmetric version

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return None

    # Handle batched input
    if Q.dim() == 3:
        Q = Q[feature_idx]

    Q = Q.cpu().float()

    # Optionally symmetrize
    if symmetric:
        Q = 0.5 * (Q + Q.T)

    # Select top-k features by row magnitude
    row_mags = Q.abs().sum(dim=1)
    top_indices = row_mags.topk(min(top_k, len(row_mags))).indices.sort().values

    # Extract submatrix
    Q_sub = Q[top_indices][:, top_indices].numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0
    vmax = np.abs(Q_sub).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(Q_sub, cmap='RdBu_r', norm=norm, aspect='auto')

    # Labels
    if title is None:
        title = f"Interaction Matrix Q (Feature {feature_idx}, top {top_k})"
    ax.set_title(title)
    ax.set_xlabel("Input Feature j")
    ax.set_ylabel("Input Feature i")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Interaction Strength")

    # Add grid
    ax.set_xticks(np.arange(0, len(top_indices), max(1, len(top_indices)//10)))
    ax.set_yticks(np.arange(0, len(top_indices), max(1, len(top_indices)//10)))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_interaction_matrix_interactive(
    Q: Tensor,
    feature_idx: int = 0,
    top_k: int = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive Plotly version of interaction matrix heatmap.

    Args:
        Q: Interaction matrix
        feature_idx: Which feature to show
        top_k: Number of top features
        title: Plot title
        save_path: If provided, save as HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available")
        return None

    # Handle batched input
    if Q.dim() == 3:
        Q = Q[feature_idx]

    Q = Q.cpu().float()
    Q = 0.5 * (Q + Q.T)  # Symmetrize

    # Select top-k features
    row_mags = Q.abs().sum(dim=1)
    top_indices = row_mags.topk(min(top_k, len(row_mags))).indices.sort().values

    Q_sub = Q[top_indices][:, top_indices].numpy()

    if title is None:
        title = f"Interaction Matrix Q (Feature {feature_idx})"

    fig = go.Figure(data=go.Heatmap(
        z=Q_sub,
        colorscale='RdBu_r',
        zmid=0,
        hovertemplate='i=%{y}, j=%{x}<br>Q[i,j]=%{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Input Feature j",
        yaxis_title="Input Feature i",
        template='plotly_white',
        width=600,
        height=550,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_variance_explained_histogram(
    variance_explained: Union[List[float], np.ndarray, Tensor],
    threshold: float = 0.75,
    target_fraction: float = 0.69,
    rank_k: int = 2,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot histogram of variance explained to verify the 69% claim.

    Paper claims: 69% of features have >0.75 variance explained by rank-2 approximation.

    Args:
        variance_explained: List/array of variance explained values per feature
        threshold: Threshold for "good" variance explained (default 0.75)
        target_fraction: Expected fraction above threshold (default 0.69 from paper)
        rank_k: Rank used for approximation
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return None

    # Convert to numpy
    if isinstance(variance_explained, Tensor):
        variance_explained = variance_explained.cpu().numpy()
    variance_explained = np.array(variance_explained)

    # Remove NaN values
    variance_explained = variance_explained[~np.isnan(variance_explained)]

    # Compute statistics
    above_threshold = (variance_explained > threshold).mean() * 100
    mean_var = variance_explained.mean()
    median_var = np.median(variance_explained)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n, bins, patches = ax.hist(
        variance_explained, bins=50, edgecolor='black', alpha=0.7,
        color='steelblue'
    )

    # Color bars above threshold differently
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge >= threshold:
            patch.set_facecolor('coral')

    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')

    # Add target line annotation
    ax.axhline(y=n.max() * 0.9, xmin=threshold, xmax=1.0,
               color='green', alpha=0.3, linewidth=0)

    # Labels
    if title is None:
        title = f"Rank-{rank_k} Variance Explained Distribution"
    ax.set_title(title)
    ax.set_xlabel(f"Variance Explained by Top-{rank_k} Eigenvalues")
    ax.set_ylabel("Number of Features")

    # Add statistics text box
    stats_text = (
        f"Features analyzed: {len(variance_explained)}\n"
        f"Mean: {mean_var:.3f}\n"
        f"Median: {median_var:.3f}\n"
        f"Above {threshold}: {above_threshold:.1f}%\n"
        f"Paper target: {target_fraction*100:.0f}%"
    )

    # Color the percentage based on whether we hit target
    result_color = 'green' if above_threshold >= target_fraction * 100 else 'red'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Add result annotation
    result_text = f"Result: {above_threshold:.1f}% > {threshold}"
    ax.text(0.98, 0.98, result_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            color=result_color, fontweight='bold')

    ax.legend(loc='upper center')
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_variance_explained_interactive(
    variance_explained: Union[List[float], np.ndarray, Tensor],
    threshold: float = 0.75,
    target_fraction: float = 0.69,
    rank_k: int = 2,
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive Plotly histogram of variance explained.

    Args:
        variance_explained: Values per feature
        threshold: Threshold for good variance
        target_fraction: Expected fraction above threshold
        rank_k: Rank used
        save_path: If provided, save as HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available")
        return None

    if isinstance(variance_explained, Tensor):
        variance_explained = variance_explained.cpu().numpy()
    variance_explained = np.array(variance_explained)
    variance_explained = variance_explained[~np.isnan(variance_explained)]

    above_threshold = (variance_explained > threshold).mean() * 100

    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=variance_explained,
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.7,
        name='Features',
        hovertemplate='Variance Explained: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))

    # Add threshold line
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="red",
        annotation_text=f"Threshold = {threshold}",
        annotation_position="top"
    )

    # Add annotation
    result_color = 'green' if above_threshold >= target_fraction * 100 else 'red'
    fig.add_annotation(
        x=0.9, y=0.95, xref='paper', yref='paper',
        text=f"<b>{above_threshold:.1f}%</b> above threshold<br>(Target: {target_fraction*100:.0f}%)",
        showarrow=False,
        font=dict(size=14, color=result_color),
        bgcolor='white',
        bordercolor=result_color,
        borderwidth=2,
    )

    fig.update_layout(
        title=f"Rank-{rank_k} Variance Explained Distribution",
        xaxis_title=f"Variance Explained by Top-{rank_k} Eigenvalues",
        yaxis_title="Number of Features",
        template='plotly_white',
        showlegend=False,
        xaxis_range=[0, 1],
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenvalue_spectrum_per_feature(
    eigenvalues_list: List[Tensor],
    feature_indices: Optional[List[int]] = None,
    top_k: int = 20,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot eigenvalue spectra for multiple output features.

    Shows how eigenvalues decay for different interaction matrices.

    Args:
        eigenvalues_list: List of eigenvalue tensors per feature
        feature_indices: Which features these correspond to
        top_k: Number of eigenvalues to show
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    n_features = min(10, len(eigenvalues_list))  # Show at most 10 features
    colors = plt.cm.viridis(np.linspace(0, 1, n_features))

    for i in range(n_features):
        eigs = eigenvalues_list[i].abs().sort(descending=True).values[:top_k].cpu().numpy()
        label = f"Feature {feature_indices[i]}" if feature_indices else f"Feature {i}"
        ax.plot(range(1, len(eigs)+1), eigs, color=colors[i], alpha=0.7, label=label)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue Magnitude")
    ax.set_title("Eigenvalue Spectra of Interaction Matrices")
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_effective_rank_distribution(
    effective_ranks: Union[List[float], np.ndarray, Tensor],
    title: str = "Effective Rank Distribution",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot distribution of effective ranks across features.

    Lower effective rank indicates more interpretable low-rank structure.

    Args:
        effective_ranks: Effective rank values per feature
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return None

    if isinstance(effective_ranks, Tensor):
        effective_ranks = effective_ranks.cpu().numpy()
    effective_ranks = np.array(effective_ranks)
    effective_ranks = effective_ranks[~np.isnan(effective_ranks)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(effective_ranks, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    mean_rank = effective_ranks.mean()
    median_rank = np.median(effective_ranks)

    ax.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_rank:.1f}')
    ax.axvline(median_rank, color='green', linestyle=':', linewidth=2, label=f'Median = {median_rank:.1f}')

    ax.set_xlabel("Effective Rank")
    ax.set_ylabel("Number of Features")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
