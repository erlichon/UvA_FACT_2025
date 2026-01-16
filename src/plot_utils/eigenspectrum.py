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

# T-distribution critical values for common sample sizes (df = n-1)
# For 90% CI: two-tailed, alpha=0.10, so we need t_{0.95}
T_CRIT_90 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
    6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
    14: 1.761, 19: 1.729, 29: 1.699, 49: 1.677, 99: 1.660,
    float('inf'): 1.645,  # z-value for large n
}


def get_t_critical(df: int, confidence: float = 0.90) -> float:
    """Get t-critical value for given degrees of freedom."""
    if confidence != 0.90:
        # Fallback to approximate z-value for other confidence levels
        # For 95% CI: 1.96, for 99% CI: 2.576
        if confidence == 0.95:
            return 1.96 if df > 30 else 2.0
        elif confidence == 0.99:
            return 2.576 if df > 30 else 2.75
        else:
            return 1.645  # Default to 90% z-value
    
    # For 90% CI, use lookup table
    if df in T_CRIT_90:
        return T_CRIT_90[df]
    # Find closest value
    for key in sorted(T_CRIT_90.keys()):
        if key >= df:
            return T_CRIT_90[key]
    return T_CRIT_90[float('inf')]


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence interval for mean across rows.
    
    Args:
        data: [n_samples, n_points] array
        confidence: Confidence level (default 0.90 for 90% CI)
        
    Returns:
        (lower_bound, upper_bound) arrays of shape [n_points]
    """
    n = data.shape[0]
    mean = data.mean(axis=0)
    std_err = data.std(axis=0, ddof=1) / np.sqrt(n)
    
    # Get t-critical value
    t_crit = get_t_critical(n - 1, confidence)
    
    margin = t_crit * std_err
    return mean - margin, mean + margin


def plot_eigenspectrum_comparison(
    eigenvalues_dict: Dict[str, Float[Tensor, "n_classes n"]],
    title: str = "Eigenspectrum Comparison",
    figsize: Tuple[float, float] = (6, 4),
    log_scale: bool = True,
    top_k: int = 100,
    save_path: Optional[str] = None,
    show_envelope: bool = True,
    confidence: float = 0.90,
) -> plt.Figure:
    """
    Plot eigenvalue spectra for multiple configurations.

    Shows mean across classes with 90% confidence interval envelope.

    Args:
        eigenvalues_dict: Dict mapping config name to eigenvalues [n_classes, n]
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis
        top_k: Only plot top-k eigenvalues (by magnitude)
        save_path: If provided, save figure to this path
        show_envelope: Whether to show confidence interval envelope
        confidence: Confidence level for envelope (default 0.90)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, eigenvalues in eigenvalues_dict.items():
        # Sort by magnitude (descending) for each class
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)

        # Take top k
        sorted_vals = sorted_vals[:, :top_k].numpy()

        # Compute mean across classes
        mean_vals = sorted_vals.mean(axis=0)

        x = np.arange(1, len(mean_vals) + 1)

        # Get color from palette or use default
        color = COLORS.get(name, None)
        label = CONFIG_NAMES.get(name, name)

        # Plot mean line
        line = ax.plot(x, mean_vals, label=label, color=color, linewidth=1.5)[0]

        # Plot 90% CI envelope
        if show_envelope and sorted_vals.shape[0] > 1:
            ci_low, ci_high = compute_confidence_interval(sorted_vals, confidence)
            ax.fill_between(
                x,
                ci_low,
                ci_high,
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
    figsize: Tuple[float, float] = (6, 4),
    top_k: int = 50,
    save_path: Optional[str] = None,
    combined: bool = True,
    confidence: float = 0.90,
) -> plt.Figure:
    """
    Plot eigenspectrum for each class.

    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor
        class_names: Names for each class (default: digits 0-9)
        title: Overall title
        figsize: Figure size
        top_k: Number of top eigenvalues to show
        save_path: If provided, save figure
        combined: If True, plot all classes on single graph with envelope.
                  If False, use subplot grid.
        confidence: Confidence level for CI (default 0.90)

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

    if combined:
        # Single plot with mean line and 90% CI envelope
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(1, top_k + 1)
        mean_vals = sorted_vals.mean(axis=0)
        min_vals = sorted_vals.min(axis=0)
        max_vals = sorted_vals.max(axis=0)
        
        # Compute 90% CI
        ci_low, ci_high = compute_confidence_interval(sorted_vals, confidence)
        
        # Plot mean line
        ax.plot(x, mean_vals, linewidth=2, color=COLORS.get("full", "steelblue"), 
                label="Mean across classes")
        
        # Plot min-max envelope (light)
        ax.fill_between(x, min_vals, max_vals, alpha=0.15, 
                        color=COLORS.get("full", "steelblue"),
                        label="Min-max range")
        
        # Plot 90% CI envelope (darker)
        ax.fill_between(x, ci_low, ci_high, 
                        alpha=0.3, color=COLORS.get("full", "steelblue"),
                        label=f"{int(confidence*100)}% CI")
        
        ax.set_xlabel("Eigenvalue Index (sorted by magnitude)")
        ax.set_ylabel("Eigenvalue Magnitude")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
    else:
        # Original subplot grid
        n_cols = 5
        n_rows = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), sharex=True, sharey=True)
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


def plot_truncation_accuracy(
    truncation_results: Dict[str, np.ndarray],
    max_eigenvalues: int = 50,
    title: str = "Accuracy vs Truncation Level",
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None,
    confidence: float = 0.90,
) -> plt.Figure:
    """
    Plot accuracy as function of eigenvalue truncation (Figure 5B from paper).
    
    Shows how accuracy changes when only retaining top-k eigenvalues.
    Averaged over seeds with 90% CI.
    
    Args:
        truncation_results: Dict mapping config name to [n_seeds, n_truncation_levels] accuracy array
        max_eigenvalues: Maximum number of eigenvalues on x-axis
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure
        confidence: Confidence level for CI
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, accuracies in truncation_results.items():
        # accuracies shape: [n_seeds, n_truncation_levels]
        if accuracies.ndim == 1:
            accuracies = accuracies[np.newaxis, :]
            
        n_points = min(accuracies.shape[1], max_eigenvalues)
        x = np.arange(1, n_points + 1)
        
        mean_acc = accuracies[:, :n_points].mean(axis=0)
        
        color = COLORS.get(name, None)
        label = CONFIG_NAMES.get(name, name)
        
        line = ax.plot(x, mean_acc * 100, label=label, color=color, linewidth=1.5)[0]
        
        # Add 90% CI if multiple seeds
        if accuracies.shape[0] > 1:
            ci_low, ci_high = compute_confidence_interval(accuracies[:, :n_points] * 100, confidence)
            ax.fill_between(x, ci_low, ci_high, alpha=0.2, color=line.get_color())
    
    ax.set_xlabel("Number of Eigenvalues Retained")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
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
