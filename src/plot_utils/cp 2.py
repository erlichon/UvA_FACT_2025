"""
Visualization utilities for CP decomposition analysis (Extension CP).

Functions for visualizing eigenvectors, eigenvalues, and CP model comparisons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List

from .style import set_publication_style


def get_top_eigenvectors(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    n_top: int = 5,
    n_bottom: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract top and bottom eigenvectors for each class.
    
    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor of eigenvalues
        eigenvectors: [n_classes, n_eigenvalues, n_eigenvalues] tensor of eigenvectors
        n_top: Number of top (largest) eigenvectors to extract
        n_bottom: Number of bottom (smallest) eigenvectors to extract
    
    Returns:
        Tuple of (top_indices, top_vecs, bottom_indices, bottom_vecs)
        - top_indices: [n_classes, n_top] indices of top eigenvectors
        - top_vecs: [n_classes, n_top, n_eigenvalues] top eigenvectors
        - bottom_indices: [n_classes, n_bottom] indices of bottom eigenvectors
        - bottom_vecs: [n_classes, n_bottom, n_eigenvalues] bottom eigenvectors
    """
    n_classes = eigenvalues.shape[0]
    
    top_indices_list = []
    top_vecs_list = []
    bottom_indices_list = []
    bottom_vecs_list = []
    
    for class_idx in range(n_classes):
        class_vals = eigenvalues[class_idx]  # [n_eigenvalues]
        class_vecs = eigenvectors[class_idx]  # [n_eigenvalues, n_eigenvalues]
        
        # Top n_top (largest eigenvalues)
        top_idx = class_vals.argsort(descending=True)[:n_top]
        top_vecs = class_vecs[top_idx]  # [n_top, n_eigenvalues]
        
        # Bottom n_bottom (smallest eigenvalues, most negative)
        bottom_idx = class_vals.argsort(descending=False)[:n_bottom]
        bottom_vecs = class_vecs[bottom_idx]  # [n_bottom, n_eigenvalues]
        
        top_indices_list.append(top_idx)
        top_vecs_list.append(top_vecs)
        bottom_indices_list.append(bottom_idx)
        bottom_vecs_list.append(bottom_vecs)
    
    top_indices = torch.stack(top_indices_list)  # [n_classes, n_top]
    top_vecs = torch.stack(top_vecs_list)  # [n_classes, n_top, n_eigenvalues]
    bottom_indices = torch.stack(bottom_indices_list)  # [n_classes, n_bottom]
    bottom_vecs = torch.stack(bottom_vecs_list)  # [n_classes, n_bottom, n_eigenvalues]
    
    return top_indices, top_vecs, bottom_indices, bottom_vecs


def plot_eigenvector_grid(
    eigenvectors: torch.Tensor,
    labels: list,
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[Path] = None,
    title: str = "Eigenvectors",
    cmap: str = 'RdBu'
) -> plt.Figure:
    """
    Create a grid plot of eigenvectors reshaped to 28×28 images.
    
    Args:
        eigenvectors: [n_rows, n_cols, 784] tensor of eigenvectors to plot
        labels: List of labels for rows and columns (e.g., class names)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        title: Figure title
        cmap: Colormap for visualization
        
    Returns:
        matplotlib Figure object
    """
    n_rows, n_cols, _ = eigenvectors.shape
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Find global vmin/vmax for consistent scaling
    vmax = eigenvectors.abs().max().item()
    vmin = -vmax
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            vec = eigenvectors[i, j].detach().cpu().numpy().reshape(28, 28)
            ax.imshow(vec, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # Add row labels on first column
            if j == 0:
                ax.set_ylabel(labels[i] if isinstance(labels, list) else f"Row {i}", 
                             fontsize=10, rotation=0, ha='right', va='center')
            
            # Add column headers on first row
            if i == 0:
                ax.set_title(labels[j] if isinstance(labels, list) else f"Col {j}", 
                            fontsize=10, pad=5)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def visualize_top_eigenvectors(
    checkpoint_path: Path,
    n_top: int = 5,
    n_bottom: int = 5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize top and bottom eigenvectors from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_top: Number of top eigenvectors to show
        n_bottom: Number of bottom eigenvectors to show
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    eigenvalues = checkpoint['eigenvalues']  # [10, 784]
    eigenvectors = checkpoint['eigenvectors']  # [10, 784, 784]
    
    # Extract top and bottom eigenvectors
    top_indices, top_vecs, bottom_indices, bottom_vecs = get_top_eigenvectors(
        eigenvalues, eigenvectors, n_top=n_top, n_bottom=n_bottom
    )
    
    # Create figure with two subplots: top and bottom
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    
    # Find global vmin/vmax
    vmax = max(top_vecs.abs().max().item(), bottom_vecs.abs().max().item())
    vmin = -vmax
    
    # Plot top eigenvectors (row 0)
    for class_idx in range(10):
        ax = axes[0, class_idx]
        # Show the first (largest) top eigenvector
        vec = top_vecs[class_idx, 0].detach().cpu().numpy().reshape(28, 28)
        ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(f'Class {class_idx}', fontsize=10)
        if class_idx == 0:
            ax.set_ylabel('Top 5\n(Largest)', fontsize=11, rotation=0, ha='right', va='center')
    
    # Plot bottom eigenvectors (row 1)
    for class_idx in range(10):
        ax = axes[1, class_idx]
        # Show the first (most negative) bottom eigenvector
        vec = bottom_vecs[class_idx, 0].detach().cpu().numpy().reshape(28, 28)
        ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if class_idx == 0:
            ax.set_ylabel('Bottom 5\n(Smallest)', fontsize=11, rotation=0, ha='right', va='center')
    
    # Extract checkpoint info for title
    checkpoint_name = checkpoint_path.stem
    plt.suptitle(f'Top and Bottom Eigenvectors: {checkpoint_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_cp_rank_comparison(
    results_df,
    metric_x: str = 'effective_rank',
    metric_y: str = 'val_acc',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot CP rank comparison scatter plot.
    
    Args:
        results_df: DataFrame with columns [rank, init_mode, seed, effective_rank, val_acc, ...]
        metric_x: X-axis metric name
        metric_y: Y-axis metric name
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Group by rank and compute mean/std
    rank_stats = results_df.groupby('rank').agg({
        metric_x: ['mean', 'std'],
        metric_y: ['mean', 'std'],
    }).reset_index()
    rank_stats.columns = ['rank', 'x_mean', 'x_std', 'y_mean', 'y_std']
    
    # Color by log2(rank)
    scatter = ax.scatter(
        rank_stats['x_mean'],
        rank_stats['y_mean'] * 100 if 'acc' in metric_y else rank_stats['y_mean'],
        c=np.log2(rank_stats['rank']),
        cmap='viridis',
        s=150,
        edgecolors='black',
        linewidths=1.5,
        zorder=3,
    )
    
    # Add error bars
    ax.errorbar(
        rank_stats['x_mean'],
        rank_stats['y_mean'] * 100 if 'acc' in metric_y else rank_stats['y_mean'],
        xerr=rank_stats['x_std'],
        yerr=rank_stats['y_std'] * 100 if 'acc' in metric_y else rank_stats['y_std'],
        fmt='none',
        color='gray',
        alpha=0.5,
        capsize=3,
        zorder=2,
    )
    
    # Add rank labels
    for _, row in rank_stats.iterrows():
        y_val = row['y_mean'] * 100 if 'acc' in metric_y else row['y_mean']
        ax.annotate(
            f"R={int(row['rank'])}",
            (row['x_mean'], y_val),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    
    ax.set_xlabel(metric_x.replace('_', ' ').title(), fontsize=12)
    ylabel = metric_y.replace('_', ' ').title()
    if 'acc' in metric_y:
        ylabel += ' (%)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('CP Rank vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₂(CP Rank)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig
