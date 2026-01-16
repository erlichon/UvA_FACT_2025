"""
Eigenvector visualization functions.

Functions for visualizing eigenvectors as images (for MNIST/Fashion-MNIST).
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


def plot_eigenvectors_grid(
    eigenvectors: Float[Tensor, "n_classes n_components d_input"],
    eigenvalues: Float[Tensor, "n_classes n_components"],
    n_top: int = 5,
    classes: Optional[List[int]] = None,
    img_shape: Tuple[int, int] = (28, 28),
    title: str = "Top Eigenvectors",
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    group_by_sign: bool = True,
    show_both_signs: bool = False,
) -> plt.Figure:
    """
    Plot grid of top eigenvectors as images.

    Shows top eigenvectors (by absolute eigenvalue magnitude) for each class.
    Eigenvectors are reshaped to image dimensions.

    Args:
        eigenvectors: [n_classes, n_components, d_input] - eigenvector matrix
        eigenvalues: [n_classes, n_components] - for sorting by magnitude
        n_top: Number of top eigenvectors to show per class (per sign if show_both_signs)
        classes: Which classes to plot (default: all)
        img_shape: Shape to reshape eigenvectors (28, 28 for MNIST)
        title: Plot title
        save_path: If provided, save figure
        class_names: Optional names for classes
        group_by_sign: If True, show positive eigenvectors first, then negative
        show_both_signs: If True, show n_top positive AND n_top negative (2*n_top columns)

    Returns:
        matplotlib Figure with grid of eigenvector images

    Layout:
        Rows: Classes (0-9 or selected)
        Columns: Top eigenvectors (sorted by |eigenvalue|, optionally grouped by sign)
    """
    n_classes_total = eigenvectors.shape[0]
    if classes is None:
        classes = list(range(n_classes_total))

    if class_names is None:
        class_names = [str(i) for i in range(n_classes_total)]

    n_rows = len(classes)
    
    # If showing both signs, double the columns
    n_cols = n_top * 2 if show_both_signs else n_top

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1 + 0.6)
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, cls in enumerate(classes):
        vals = eigenvalues[cls]
        abs_vals = vals.abs()
        
        # Get indices for positive and negative eigenvalues, each sorted by magnitude
        pos_mask = vals > 0
        neg_mask = vals < 0
        
        # Sort positive eigenvalues by magnitude (descending)
        pos_indices = torch.where(pos_mask)[0]
        if len(pos_indices) > 0:
            pos_magnitudes = abs_vals[pos_indices]
            pos_sorted = pos_indices[pos_magnitudes.argsort(descending=True)]
        else:
            pos_sorted = torch.tensor([], dtype=torch.long)
        
        # Sort negative eigenvalues by magnitude (descending)
        neg_indices = torch.where(neg_mask)[0]
        if len(neg_indices) > 0:
            neg_magnitudes = abs_vals[neg_indices]
            neg_sorted = neg_indices[neg_magnitudes.argsort(descending=True)]
        else:
            neg_sorted = torch.tensor([], dtype=torch.long)
        
        if show_both_signs:
            # Show n_top positive, then n_top negative
            sorted_indices = torch.cat([
                pos_sorted[:n_top], 
                neg_sorted[:n_top]
            ])
        elif group_by_sign:
            # Combine: positive first, then negative
            sorted_indices = torch.cat([pos_sorted, neg_sorted])
        else:
            # Original behavior: sort by absolute magnitude
            sorted_indices = abs_vals.argsort(descending=True)

        for col in range(min(n_cols, len(sorted_indices))):
            idx = sorted_indices[col]
            vec = eigenvectors[cls, idx].numpy()
            val = eigenvalues[cls, idx].item()

            # Reshape to image
            img = vec.reshape(img_shape)

            # Normalize for visualization (symmetric around zero)
            vmax = np.abs(img).max()
            if vmax < 1e-10:
                vmax = 1.0

            axes[row, col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].axis("off")

            if row == 0:
                if show_both_signs:
                    # Clear column header distinction for positive vs negative
                    if col < n_top:
                        axes[row, col].set_title(f"+{col + 1}", fontsize=8)
                    else:
                        axes[row, col].set_title(f"-{col - n_top + 1}", fontsize=8)
                else:
                    # Show sign indicator in column header
                    sign = "+" if val > 0 else "-"
                    axes[row, col].set_title(f"#{col + 1} ({sign})", fontsize=9)

        # Add class label on left
        axes[row, 0].annotate(
            class_names[cls],
            xy=(-0.15, 0.5),
            xycoords="axes fraction",
            fontsize=10,
            ha="right",
            va="center",
        )

    # Add super-titles for positive/negative sections if showing both signs
    if show_both_signs:
        # Add "Positive" and "Negative" labels
        fig.text(0.25, 0.99, "Positive Eigenvalues", ha='center', fontsize=11, fontweight='bold')
        fig.text(0.75, 0.99, "Negative Eigenvalues", ha='center', fontsize=11, fontweight='bold')

    fig.suptitle(title, y=1.03)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_single_eigenvector(
    eigenvector: Float[Tensor, "d_input"],
    eigenvalue: float,
    class_idx: int,
    rank: int,
    img_shape: Tuple[int, int] = (28, 28),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (3, 3),
) -> plt.Figure:
    """
    Plot a single eigenvector as an image with colorbar.

    Args:
        eigenvector: [d_input] eigenvector
        eigenvalue: Corresponding eigenvalue
        class_idx: Which class this eigenvector belongs to
        rank: Rank of this eigenvector (1-indexed)
        img_shape: Shape to reshape eigenvector
        title: Optional custom title
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    vec = eigenvector.numpy()
    img = vec.reshape(img_shape)

    vmax = np.abs(img).max()
    if vmax < 1e-10:
        vmax = 1.0

    im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axis("off")

    if title is None:
        title = f"Class {class_idx}, Rank {rank}\n(eigenvalue = {eigenvalue:.2e})"
    ax.set_title(title, fontsize=10)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig
