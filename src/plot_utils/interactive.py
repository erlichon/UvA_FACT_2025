"""
Interactive Plotly versions of eigenspectrum and comparison plots.

Provides interactive HTML outputs with hover, zoom, and pan capabilities.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Plotly color mapping for configs
PLOTLY_COLORS = {
    "none": "#1f77b4",      # Blue
    "noise": "#ff7f0e",     # Orange
    "wd": "#2ca02c",        # Green
    "full": "#d62728",      # Red
}

CONFIG_NAMES = {
    "none": "No Reg",
    "noise": "Noise",
    "wd": "Weight Decay",
    "full": "Full Reg",
}


def plot_eigenspectrum_interactive(
    eigenvalues_dict: Dict[str, Float[Tensor, "n_classes n"]],
    title: str = "Eigenspectrum Comparison",
    top_k: int = 100,
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive Plotly version of eigenspectrum comparison.

    Features:
    - Hover to see exact eigenvalue values
    - Click legend to toggle traces
    - Zoom and pan

    Args:
        eigenvalues_dict: Dict mapping config name to eigenvalues [n_classes, n]
        title: Plot title
        top_k: Number of top eigenvalues to show
        log_scale: Use log scale for y-axis
        save_path: If provided, save as HTML file

    Returns:
        Plotly Figure object or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    fig = go.Figure()

    for name, eigenvalues in eigenvalues_dict.items():
        # Sort by magnitude (descending) for each class
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
        sorted_vals = sorted_vals[:, :top_k]

        # Compute mean and std across classes
        mean_vals = sorted_vals.mean(dim=0).numpy()
        std_vals = sorted_vals.std(dim=0).numpy()

        x = np.arange(1, len(mean_vals) + 1)
        color = PLOTLY_COLORS.get(name, px.colors.qualitative.Plotly[0])
        label = CONFIG_NAMES.get(name, name)

        # Add mean line
        fig.add_trace(go.Scatter(
            x=x, y=mean_vals,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>Index: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
        ))

        # Add std shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean_vals + std_vals, (mean_vals - std_vals)[::-1]]),
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Eigenvalue Index (sorted by magnitude)",
        yaxis_title="Eigenvalue Magnitude",
        template="plotly_white",
        hovermode='x unified',
        width=800,
        height=500,
    )

    if log_scale:
        fig.update_yaxes(type="log")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenspectrum_per_class_interactive(
    eigenvalues: Float[Tensor, "n_classes n"],
    class_names: Optional[List[str]] = None,
    title: str = "Eigenspectrum by Class",
    top_k: int = 50,
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive subplot grid of eigenspectra by class.

    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor
        class_names: Names for each class (default: digits 0-9)
        title: Overall title
        top_k: Number of top eigenvalues to show
        save_path: If provided, save as HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        return None

    n_classes = eigenvalues.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Sort eigenvalues by magnitude
    abs_vals = eigenvalues.abs()
    sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
    sorted_vals = sorted_vals[:, :top_k].numpy()

    # Create subplot grid
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Class {name}" for name in class_names],
        shared_xaxes=True, shared_yaxes=True
    )

    x = np.arange(1, top_k + 1)
    for i in range(n_classes):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(
            go.Scatter(
                x=x, y=sorted_vals[i],
                mode='lines',
                line=dict(color='steelblue', width=1.5),
                name=f"Class {class_names[i]}",
                showlegend=False,
                hovertemplate=f"Class {class_names[i]}<br>Index: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
            ),
            row=row, col=col
        )

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=150 * n_rows,
        width=180 * n_cols,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_accuracy_vs_rank_interactive(
    data: Dict[str, Dict[str, List[float]]],
    title: str = "Accuracy vs Effective Rank Trade-off",
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive scatter plot of accuracy vs effective rank.

    Args:
        data: Dict[config_name] -> {"accuracy": [...], "effective_rank": [...]}
        title: Plot title
        save_path: If provided, save as HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    for name, metrics in data.items():
        accs = np.array(metrics["accuracy"])
        ranks = np.array(metrics["effective_rank"])

        color = PLOTLY_COLORS.get(name, px.colors.qualitative.Plotly[0])
        label = CONFIG_NAMES.get(name, name)

        # Individual points
        fig.add_trace(go.Scatter(
            x=ranks, y=accs,
            mode='markers',
            name=label,
            marker=dict(color=color, size=10, opacity=0.7),
            hovertemplate=f"{label}<br>Rank: %{{x:.2f}}<br>Acc: %{{y:.2f}}%<extra></extra>"
        ))

        # Mean point with error bars
        fig.add_trace(go.Scatter(
            x=[ranks.mean()], y=[accs.mean()],
            mode='markers',
            name=f"{label} (mean)",
            marker=dict(color=color, size=15, symbol='x'),
            error_x=dict(type='data', array=[ranks.std()], visible=True, color=color),
            error_y=dict(type='data', array=[accs.std()], visible=True, color=color),
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Effective Rank",
        yaxis_title="Validation Accuracy (%)",
        template="plotly_white",
        hovermode='closest',
        width=700,
        height=500,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenvectors_interactive(
    eigenvectors: Float[Tensor, "n_classes n d_input"],
    eigenvalues: Float[Tensor, "n_classes n"],
    top_k: int = 5,
    class_names: Optional[List[str]] = None,
    image_shape: Tuple[int, int] = (28, 28),
    title: str = "Top Eigenvectors by Class",
    save_path: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive grid of top eigenvectors as heatmaps.

    Args:
        eigenvectors: [n_classes, n_eigenvectors, d_input]
        eigenvalues: [n_classes, n_eigenvectors] for sorting
        top_k: Number of top eigenvectors per class
        class_names: Names for classes
        image_shape: Shape for reshaping eigenvectors
        title: Plot title
        save_path: If provided, save as HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        return None

    n_classes = eigenvectors.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Sort by eigenvalue magnitude
    sorted_idxs = eigenvalues.abs().argsort(dim=-1, descending=True)

    # Create subplot grid
    fig = make_subplots(
        rows=n_classes, cols=top_k,
        subplot_titles=[f"" for _ in range(n_classes * top_k)],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for i in range(n_classes):
        for j in range(top_k):
            idx = sorted_idxs[i, j].item()
            vec = eigenvectors[i, idx].reshape(image_shape).cpu().numpy()

            fig.add_trace(
                go.Heatmap(
                    z=np.flipud(vec),
                    colorscale='RdBu_r',
                    zmid=0,
                    showscale=False,
                    hovertemplate=f"Class {class_names[i]}, Rank {j+1}<br>λ={eigenvalues[i, idx]:.3f}<extra></extra>"
                ),
                row=i+1, col=j+1
            )

    # Add class labels on left
    for i, name in enumerate(class_names):
        fig.add_annotation(
            x=-0.05, y=1 - (i + 0.5) / n_classes,
            xref="paper", yref="paper",
            text=f"Class {name}",
            showarrow=False,
            font=dict(size=10),
            xanchor='right'
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=80 * n_classes,
        width=100 * top_k + 50,
        margin=dict(l=60, r=20, t=40, b=20),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig
