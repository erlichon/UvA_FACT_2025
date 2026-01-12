"""
Eigenspectrum visualization functions.

Functions for plotting eigenvalue distributions and decay patterns.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.plot_utils.style import COLORS, CONFIG_NAMES


def plot_eigenspectrum_comparison(
    eigenvalues_dict: Dict[str, Float[Tensor, "n_classes n"]],
    title: str = "Eigenspectrum Comparison",
    figsize: Tuple[float, float] = (6, 4),
    log_scale: bool = True,
    top_k: int = 100,
    save_path: Optional[str] = None,
    show_std: bool = True,
) -> plt.Figure:
    """
    Plot eigenvalue spectra for multiple configurations.

    Shows mean across classes with optional std shading.

    Args:
        eigenvalues_dict: Dict mapping config name to eigenvalues [n_classes, n]
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis
        top_k: Only plot top-k eigenvalues (by magnitude)
        save_path: If provided, save figure to this path
        show_std: Whether to show std shading

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, eigenvalues in eigenvalues_dict.items():
        # Sort by magnitude (descending) for each class
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)

        # Take top k
        sorted_vals = sorted_vals[:, :top_k]

        # Compute mean and std across classes
        mean_vals = sorted_vals.mean(dim=0).numpy()
        std_vals = sorted_vals.std(dim=0).numpy()

        x = np.arange(1, len(mean_vals) + 1)

        # Get color from palette or use default
        color = COLORS.get(name, None)
        label = CONFIG_NAMES.get(name, name)

        # Plot mean line
        line = ax.plot(x, mean_vals, label=label, color=color, linewidth=1.5)[0]

        # Plot std shading
        if show_std:
            ax.fill_between(
                x,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xlabel("Eigenvalue Index (sorted by magnitude)")
    ax.set_ylabel("Eigenvalue Magnitude")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if log_scale:
        ax.set_yscale("log")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenspectrum_per_class(
    eigenvalues: Float[Tensor, "n_classes n"],
    class_names: Optional[List[str]] = None,
    title: str = "Eigenspectrum by Class",
    figsize: Tuple[float, float] = (10, 6),
    top_k: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenspectrum for each class in a subplot grid.

    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor
        class_names: Names for each class (default: digits 0-9)
        title: Overall title
        figsize: Figure size
        top_k: Number of top eigenvalues to show
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    x = np.arange(1, top_k + 1)
    for i in range(n_classes):
        axes[i].plot(x, sorted_vals[i], linewidth=1.2, color=COLORS.get("full", "steelblue"))
        axes[i].set_title(f"Class {class_names[i]}", fontsize=9)
        axes[i].set_yscale("log")

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.supxlabel("Eigenvalue Index")
    fig.supylabel("Magnitude")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_eigenvalue_decay(
    eigenvalues_dict: Dict[str, Float[Tensor, "n_classes n"]],
    top_k: int = 20,
    title: str = "Eigenvalue Decay Rate",
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot normalized eigenvalue decay to compare decay rates across configs.

    Args:
        eigenvalues_dict: Dict mapping config name to eigenvalues
        top_k: Number of top eigenvalues to show
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, eigenvalues in eigenvalues_dict.items():
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
        sorted_vals = sorted_vals[:, :top_k]

        # Normalize by first eigenvalue
        normalized = sorted_vals / sorted_vals[:, 0:1]
        mean_norm = normalized.mean(dim=0).numpy()

        x = np.arange(1, len(mean_norm) + 1)
        color = COLORS.get(name, None)
        label = CONFIG_NAMES.get(name, name)
        ax.plot(x, mean_norm, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Normalized Magnitude (relative to 1st)")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig
