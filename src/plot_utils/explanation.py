"""
Sample explanation visualization for bilinear MLPs.

Shows how eigenvectors contribute to classifying a single input sample.
Provides both matplotlib (static PDF) and Plotly (interactive HTML) versions.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.plot_utils.style import set_publication_style


def compute_eigenvector_activations(
    sample: Float[Tensor, "h w"],
    eigenvalues: Float[Tensor, "n_classes n"],
    eigenvectors: Float[Tensor, "n_classes n d_input"],
) -> Float[Tensor, "n_classes n"]:
    """
    Compute activation contribution of each eigenvector for a given sample.

    Formula: acts[c, i] = (dot(eigenvector[c, i], sample))^2 * eigenvalue[c, i]

    Args:
        sample: Input image [height, width]
        eigenvalues: [n_classes, n_eigenvectors]
        eigenvectors: [n_classes, n_eigenvectors, d_input]

    Returns:
        Activations [n_classes, n_eigenvectors]
    """
    sample_flat = sample.flatten().cpu()
    eigenvalues = eigenvalues.cpu()
    eigenvectors = eigenvectors.cpu()

    # Compute dot products: [n_classes, n_eigenvectors]
    dots = torch.einsum("i, cni -> cn", sample_flat, eigenvectors)

    # Activations = dot^2 * eigenvalue
    activations = dots.pow(2) * eigenvalues

    return activations


def plot_sample_explanation(
    sample: Float[Tensor, "h w"],
    eigenvalues: Float[Tensor, "n_classes n"],
    eigenvectors: Float[Tensor, "n_classes n d_input"],
    logits: Optional[Float[Tensor, "n_classes"]] = None,
    top_k_eigenvectors: int = 5,
    top_k_classes: int = 3,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    image_shape: Tuple[int, int] = (28, 28),
) -> plt.Figure:
    """
    Visualize how eigenvectors explain a single sample's classification.

    Shows:
    - Input image
    - Top-k eigenvector contributions per predicted class
    - Top-k eigenvectors as heatmaps
    - Final class predictions (logits)

    Args:
        sample: Input image [height, width]
        eigenvalues: [n_classes, n_eigenvectors]
        eigenvectors: [n_classes, n_eigenvectors, d_input]
        logits: Optional class logits [n_classes]. If None, uses activation sums.
        top_k_eigenvectors: Number of top eigenvectors to show per class
        top_k_classes: Number of top predicted classes to analyze
        figsize: Figure size
        save_path: If provided, save figure to this path
        image_shape: Shape of input image for reshaping eigenvectors

    Returns:
        matplotlib Figure object
    """
    set_publication_style()

    # Compute activations
    activations = compute_eigenvector_activations(sample, eigenvalues, eigenvectors)

    # Determine top classes from logits or activation sums
    if logits is None:
        class_scores = activations.sum(dim=-1)
    else:
        class_scores = logits.cpu()

    top_classes = class_scores.topk(top_k_classes).indices.sort().values

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, top_k_classes + 2, figure=fig, wspace=0.3, hspace=0.4)

    # Input image (top-left, spans 2 rows)
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(sample.cpu().numpy(), cmap='gray')
    ax_input.set_title("Input", fontsize=10)
    ax_input.axis('off')

    # For each top class, show eigenvalue contributions and top eigenvector
    for i, cls in enumerate(top_classes):
        cls = cls.item()

        # Get activations for this class, sorted by magnitude
        class_acts = activations[cls]
        sorted_acts, sorted_idxs = class_acts.abs().sort(descending=True)

        # Top row: positive contributions (constructive)
        ax_pos = fig.add_subplot(gs[0, i + 1])
        top_pos_idx = sorted_idxs[0].item()
        eigenvec = eigenvectors[cls, top_pos_idx].reshape(image_shape)
        ax_pos.imshow(eigenvec.cpu().numpy(), cmap='RdBu_r', vmin=-eigenvec.abs().max(), vmax=eigenvec.abs().max())
        ax_pos.set_title(f"Class {cls}\n({class_acts[top_pos_idx]:.2f})", fontsize=9)
        ax_pos.axis('off')

        # Bottom row: contribution bar chart
        ax_bar = fig.add_subplot(gs[1, i + 1])
        top_k_acts = class_acts[sorted_idxs[:top_k_eigenvectors]].cpu().numpy()
        colors = ['green' if a > 0 else 'red' for a in top_k_acts]
        ax_bar.barh(range(top_k_eigenvectors), top_k_acts, color=colors, alpha=0.7)
        ax_bar.set_yticks(range(top_k_eigenvectors))
        ax_bar.set_yticklabels([f"#{sorted_idxs[j].item()}" for j in range(top_k_eigenvectors)], fontsize=7)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Contribution", fontsize=8)
        ax_bar.axvline(0, color='black', linewidth=0.5)

    # Final logits bar chart (right column)
    ax_logits = fig.add_subplot(gs[:, -1])
    all_classes = range(eigenvalues.shape[0])
    logit_values = class_scores.cpu().numpy()
    colors = ['steelblue'] * len(logit_values)
    for cls in top_classes:
        colors[cls.item()] = 'coral'
    ax_logits.barh(all_classes, logit_values, color=colors)
    ax_logits.set_yticks(all_classes)
    ax_logits.set_yticklabels([str(c) for c in all_classes], fontsize=8)
    ax_logits.set_xlabel("Score", fontsize=9)
    ax_logits.set_title("Class Scores", fontsize=10)
    ax_logits.invert_yaxis()

    plt.suptitle("Eigenvector Explanation", fontsize=12, y=1.02)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_sample_explanation_interactive(
    sample: Float[Tensor, "h w"],
    eigenvalues: Float[Tensor, "n_classes n"],
    eigenvectors: Float[Tensor, "n_classes n d_input"],
    logits: Optional[Float[Tensor, "n_classes"]] = None,
    top_k_eigenvectors: int = 10,
    top_k_classes: int = 3,
    save_path: Optional[str] = None,
    image_shape: Tuple[int, int] = (28, 28),
):
    """
    Interactive Plotly visualization of eigenvector explanations.

    Similar to plot_sample_explanation but with interactive features:
    - Hover to see exact values
    - Zoom and pan
    - Toggle traces

    Args:
        sample: Input image [height, width]
        eigenvalues: [n_classes, n_eigenvectors]
        eigenvectors: [n_classes, n_eigenvectors, d_input]
        logits: Optional class logits
        top_k_eigenvectors: Number of eigenvalues to show in spectrum
        top_k_classes: Number of top classes to analyze
        save_path: If provided, save as HTML
        image_shape: Shape for reshaping eigenvectors

    Returns:
        Plotly figure or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    colors = px.colors.qualitative.Plotly

    # Compute activations
    activations = compute_eigenvector_activations(sample, eigenvalues, eigenvectors)

    # Determine top classes
    if logits is None:
        class_scores = activations.sum(dim=-1)
    else:
        class_scores = logits.cpu()

    top_classes = class_scores.topk(top_k_classes).indices.sort().values

    # Get contributions for top classes
    contrib_list = []
    idx_list = []
    for cls in top_classes:
        cls = cls.item()
        sorted_acts, sorted_idxs = activations[cls].sort()
        contrib_list.append(sorted_acts)
        idx_list.append(sorted_idxs)

    # Create subplot layout
    titles = [''] + [f"Class {c.item()}" for c in top_classes] + ['Input', ''] + [f"Class {c.item()}" for c in top_classes] + ['Logits']
    fig = make_subplots(rows=2, cols=top_k_classes + 2, subplot_titles=titles, vertical_spacing=0.15)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)

    # Add eigenvalue contribution lines for each class
    for i, (cls, contrib, idxs) in enumerate(zip(top_classes, contrib_list, idx_list)):
        params = dict(showlegend=False, marker=dict(color=colors[i % len(colors)]))

        # Top row: positive contributions (top eigenvalues)
        fig.add_scatter(y=contrib[-top_k_eigenvectors-2:].flip(0).cpu().numpy(), mode="lines", **params, row=1, col=1)
        fig.add_scatter(y=contrib[-1:].flip(0).cpu().numpy(), mode="markers", **params, row=1, col=1)

        # Bottom row: negative contributions
        fig.add_scatter(y=contrib[:top_k_eigenvectors+2].cpu().numpy(), mode="lines", **params, row=2, col=1)
        fig.add_scatter(y=contrib[:1].cpu().numpy(), mode="markers", **params, row=2, col=1)

    # Add eigenvector heatmaps
    for i, (cls, idxs) in enumerate(zip(top_classes, idx_list)):
        cls_val = cls.item()
        params = dict(showscale=False, colorscale="RdBu", zmid=0)

        # Top eigenvector (positive)
        vec_pos = eigenvectors[cls_val, idxs[-1]].reshape(image_shape).cpu().numpy()
        fig.add_heatmap(z=np.flipud(vec_pos), **params, row=1, col=i+2)

        # Bottom eigenvector (negative)
        vec_neg = eigenvectors[cls_val, idxs[0]].reshape(image_shape).cpu().numpy()
        fig.add_heatmap(z=np.flipud(vec_neg), **params, row=2, col=i+2)

    # Add input image
    fig.add_heatmap(z=np.flipud(sample.cpu().numpy()), colorscale="RdBu", zmid=0, showscale=False, row=1, col=top_k_classes+2)

    # Add logits bar chart
    logit_values = class_scores.cpu().numpy()
    bar_colors = ["gray"] * len(logit_values)
    text_labels = [""] * len(logit_values)
    for j, c in enumerate(top_classes):
        bar_colors[c.item()] = colors[j % len(colors)]
        text_labels[c.item()] = str(c.item())

    fig.add_bar(y=logit_values, marker_color=bar_colors, text=text_labels, showlegend=False,
                textposition='outside', textfont=dict(size=10), row=2, col=top_k_classes+2)
    fig.update_yaxes(range=[logit_values.min(), logit_values.max() * 1.3], row=2, col=top_k_classes+2)

    # Update axis labels
    fig.update_xaxes(visible=True, tickvals=[top_k_eigenvectors], ticktext=[f'{top_k_eigenvectors}'], zeroline=False, col=1)

    # Layout
    fig.update_layout(
        width=800, height=400,
        margin=dict(l=20, r=20, b=20, t=40),
        template="plotly_white",
        title="Eigenvector Explanation"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenspectrum_with_signs(
    eigenvalues: Float[Tensor, "n_classes n"],
    eigenvectors: Float[Tensor, "n_classes n d_input"],
    digit: int = 0,
    n_eigenvectors: int = 5,
    n_eigenvalues: int = 20,
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None,
):
    """
    Plot eigenspectrum with positive/negative separation (Plotly version).

    Shows:
    - Top row: Positive eigenvalues with corresponding eigenvectors
    - Bottom row: Negative eigenvalues with corresponding eigenvectors

    Args:
        eigenvalues: [n_classes, n_eigenvectors]
        eigenvectors: [n_classes, n_eigenvectors, d_input]
        digit: Which class/digit to analyze
        n_eigenvectors: Number of eigenvector images to show (default: 5)
        n_eigenvalues: Number of eigenvalues in spectrum plot
        image_shape: Shape for reshaping eigenvectors
        save_path: If provided, save as HTML

    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1 + n_eigenvectors)

    vals = eigenvalues[digit].cpu()
    vecs = eigenvectors[digit].cpu()

    # Indices for top positive and top negative eigenvalues
    negative_idx = torch.arange(n_eigenvectors)
    positive_idx = -1 - negative_idx

    # Top row: positive eigenvalues spectrum
    fig.add_trace(go.Scatter(y=vals[-n_eigenvalues-2:].flip(0).numpy(), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=negative_idx.flip(0).numpy(), y=vals[positive_idx].flip(0).numpy(),
                             mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    # Bottom row: negative eigenvalues spectrum
    fig.add_trace(go.Scatter(y=vals[:n_eigenvalues+2].numpy(), mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative_idx.numpy(), y=vals[negative_idx].numpy(),
                             mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    # Add eigenvector heatmaps
    for i, idx in enumerate(positive_idx):
        vec = vecs[idx].reshape(image_shape).numpy()
        fig.add_trace(go.Heatmap(z=np.flipud(vec), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative_idx):
        vec = vecs[idx].reshape(image_shape).numpy()
        fig.add_trace(go.Heatmap(z=np.flipud(vec), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[n_eigenvalues], ticktext=[f'{n_eigenvalues}'], zeroline=False, col=1)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1)

    # Add eigenvalue tick labels
    pos_tickvals = [0] + [vals[idx].item() for idx in positive_idx]
    pos_ticktext = [f'{v:.2f}' for v in pos_tickvals]
    fig.update_yaxes(visible=True, tickvals=pos_tickvals, ticktext=pos_ticktext, col=1, row=1)

    neg_tickvals = [0] + [vals[idx].item() for idx in negative_idx]
    neg_ticktext = [f'{v:.2f}' for v in neg_tickvals]
    fig.update_yaxes(visible=True, tickvals=neg_tickvals, ticktext=neg_ticktext, col=1, row=2)

    fig.update_layout(
        autosize=False,
        width=170 * (n_eigenvectors + 1),
        height=300,
        margin=dict(l=20, r=20, b=20, t=40),
        template="plotly_white",
        showlegend=False,
        title=f"Eigenspectrum for Class {digit}"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def generate_all_digit_eigenspectra(
    eigenvalues: Float[Tensor, "n_classes n"],
    eigenvectors: Float[Tensor, "n_classes n d_input"],
    output_dir: str,
    n_eigenvectors: int = 5,
    n_eigenvalues: int = 20,
    image_shape: Tuple[int, int] = (28, 28),
) -> List[str]:
    """
    Generate interactive eigenspectrum displays for all 10 digit classes.

    Creates one HTML file per digit showing:
    - Top row: Positive eigenvalue spectrum + top-k positive eigenvectors
    - Bottom row: Negative eigenvalue spectrum + top-k negative eigenvectors

    Args:
        eigenvalues: [n_classes, n_eigenvectors] tensor
        eigenvectors: [n_classes, n_eigenvectors, d_input] tensor
        output_dir: Directory to save HTML files
        n_eigenvectors: Number of eigenvector images per sign (default: 5)
        n_eigenvalues: Number of eigenvalues to show in spectrum
        image_shape: Shape for reshaping eigenvectors

    Returns:
        List of saved file paths
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    n_classes = min(eigenvalues.shape[0], 10)  # Typically 10 digits

    for digit in range(n_classes):
        save_path = output_path / f"eigenspectrum_digit_{digit}.html"
        plot_eigenspectrum_with_signs(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            digit=digit,
            n_eigenvectors=n_eigenvectors,
            n_eigenvalues=n_eigenvalues,
            image_shape=image_shape,
            save_path=str(save_path),
        )
        saved_files.append(str(save_path))

    print(f"Generated {len(saved_files)} eigenspectrum HTML files in {output_dir}")
    return saved_files
