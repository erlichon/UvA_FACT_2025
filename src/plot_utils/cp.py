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


def visualize_eigenvectors_for_config(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    n_top: int = 5,
    n_bottom: int = 5,
    title: str = "Eigenvectors",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize top and bottom eigenvectors for a checkpoint's eigenvalues/eigenvectors.
    
    Creates a grid with 10 rows (one per class) and (n_top + n_bottom) columns
    showing the top positive and top negative eigenvectors with eigenvalue labels.
    
    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor
        eigenvectors: [n_classes, n_eigenvalues, d_input] tensor
        n_top: Number of top (largest) eigenvectors to show
        n_bottom: Number of bottom (smallest) eigenvectors to show
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    # Extract top and bottom eigenvectors
    top_indices, top_vecs, bottom_indices, bottom_vecs = get_top_eigenvectors(
        eigenvalues, eigenvectors, n_top=n_top, n_bottom=n_bottom
    )
    
    n_classes = eigenvalues.shape[0]
    
    # Create figure: n_classes rows × (n_top + n_bottom) columns
    fig, axes = plt.subplots(n_classes, n_top + n_bottom, figsize=(2 * (n_top + n_bottom), 2 * n_classes))
    
    # Find global vmin/vmax for consistent scaling
    vmax = max(top_vecs.abs().max().item(), bottom_vecs.abs().max().item())
    vmin = -vmax
    
    for class_idx in range(n_classes):
        # Plot top eigenvectors (positive)
        for i in range(n_top):
            ax = axes[class_idx, i]
            vec = top_vecs[class_idx, i].detach().cpu().numpy().reshape(28, 28)
            ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # Add eigenvalue label
            eig_val = eigenvalues[class_idx, top_indices[class_idx, i]].item()
            ax.set_title(f'{eig_val:.2f}', fontsize=8, pad=2)
            
            # Add class label on first column
            if i == 0:
                ax.set_ylabel(f'Class {class_idx}', fontsize=10, rotation=0, ha='right', va='center')
        
        # Plot bottom eigenvectors (negative)
        for i in range(n_bottom):
            ax = axes[class_idx, n_top + i]
            vec = bottom_vecs[class_idx, i].detach().cpu().numpy().reshape(28, 28)
            ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # Add eigenvalue label
            eig_val = eigenvalues[class_idx, bottom_indices[class_idx, i]].item()
            ax.set_title(f'{eig_val:.2f}', fontsize=8, pad=2)
    
    # Column headers
    for i in range(n_top):
        axes[0, i].annotate(f'Top {i+1}', xy=(0.5, 1.15), xycoords='axes fraction',
                           ha='center', fontsize=9, fontweight='bold', color='green')
    for i in range(n_bottom):
        axes[0, n_top + i].annotate(f'Bot {i+1}', xy=(0.5, 1.15), xycoords='axes fraction',
                                    ha='center', fontsize=9, fontweight='bold', color='red')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def compare_modes_for_rank(
    checkpoint_data: dict,
    rank: int,
    modes: List[str] = ['fixed', 'lambda', 'gated'],
    n_top: int = 5,
    n_bottom: int = 5,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Compare eigenvectors across all modes for a fixed CP rank.
    
    Args:
        checkpoint_data: Dict mapping (rank, mode) tuples to checkpoint dicts
        rank: CP rank to compare
        modes: List of mode names to compare
        n_top: Number of top eigenvectors per mode
        n_bottom: Number of bottom eigenvectors per mode
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object (or None if no checkpoints found)
    """
    set_publication_style()
    
    # Collect all eigenvectors for global scaling
    all_vecs = []
    mode_data = {}
    for mode in modes:
        key = (rank, mode)
        if key in checkpoint_data:
            checkpoint = checkpoint_data[key]
            eigenvalues = checkpoint['eigenvalues']
            eigenvectors = checkpoint['eigenvectors']
            top_indices, top_vecs, bottom_indices, bottom_vecs = get_top_eigenvectors(
                eigenvalues, eigenvectors, n_top=n_top, n_bottom=n_bottom
            )
            mode_data[mode] = {
                'eigenvalues': eigenvalues,
                'top_indices': top_indices,
                'top_vecs': top_vecs,
                'bottom_indices': bottom_indices,
                'bottom_vecs': bottom_vecs,
            }
            all_vecs.extend([top_vecs, bottom_vecs])
    
    if not mode_data:
        print(f"No checkpoints found for rank {rank}")
        return None
    
    n_modes = len(mode_data)
    vmax = max(v.abs().max().item() for v in all_vecs)
    vmin = -vmax
    
    # Create figure
    n_cols = n_modes * (n_top + n_bottom)
    fig, axes = plt.subplots(10, n_cols, figsize=(1.5 * n_cols, 20))
    
    for mode_idx, mode in enumerate(modes):
        if mode not in mode_data:
            continue
        
        data = mode_data[mode]
        col_start = mode_idx * (n_top + n_bottom)
        
        for class_idx in range(10):
            # Top eigenvectors
            for i in range(n_top):
                ax = axes[class_idx, col_start + i]
                vec = data['top_vecs'][class_idx, i].detach().cpu().numpy().reshape(28, 28)
                ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
                ax.axis('off')
                
                if class_idx == 0:
                    ax.set_title(f'{mode}\nTop{i+1}', fontsize=7)
                if col_start + i == 0:
                    ax.set_ylabel(f'{class_idx}', fontsize=9, rotation=0, ha='right', va='center')
            
            # Bottom eigenvectors
            for i in range(n_bottom):
                ax = axes[class_idx, col_start + n_top + i]
                vec = data['bottom_vecs'][class_idx, i].detach().cpu().numpy().reshape(28, 28)
                ax.imshow(vec, cmap='RdBu', vmin=vmin, vmax=vmax)
                ax.axis('off')
                
                if class_idx == 0:
                    ax.set_title(f'Bot{i+1}', fontsize=7)
    
    plt.suptitle(f'Eigenvector Comparison: CP Rank {rank}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_cp_accuracy_vs_rank(
    cp_df,
    baseline_df=None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot CP accuracy vs rank.
    
    Args:
        cp_df: DataFrame with CP results (columns: rank, accuracy, effective_rank)
        baseline_df: Optional DataFrame with dense baseline results
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(cp_df) > 0:
        # Aggregate CP results by rank
        cp_agg = cp_df.groupby('rank').agg({
            'accuracy': ['mean', 'std'],
        }).reset_index()
        cp_agg.columns = ['rank', 'acc_mean', 'acc_std']
        
        ax.errorbar(cp_agg['rank'], cp_agg['acc_mean'] * 100, 
                    yerr=cp_agg['acc_std'] * 100,
                    fmt='o-', label='CP', markersize=10, linewidth=2, capsize=5)
    
    # Add dense baselines as horizontal lines
    if baseline_df is not None and len(baseline_df) > 0:
        mode_col = 'mode' if 'mode' in baseline_df.columns else 'config'
        for mode in ['none', 'full']:
            subset = baseline_df[baseline_df[mode_col].str.contains(mode, case=False)]
            if len(subset) > 0:
                mean_acc = subset['accuracy'].mean() * 100
                ax.axhline(y=mean_acc, linestyle='--', alpha=0.7, 
                          label=f'Dense ({mode}): {mean_acc:.1f}%')
    
    ax.set_xlabel('CP Rank', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CP Accuracy vs Rank', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cp_effective_rank_vs_cp_rank(
    cp_df,
    baseline_df=None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show_ideal_line: bool = False,
) -> plt.Figure:
    """
    Plot CP effective rank vs CP rank.
    
    Args:
        cp_df: DataFrame with CP results (columns: rank, effective_rank)
        baseline_df: Optional DataFrame with dense baseline results (columns: mode, effective_rank)
        figsize: Figure size
        save_path: Optional path to save figure
        show_ideal_line: If True, show diagonal line where eff_rank = cp_rank
        
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(cp_df) > 0:
        cp_agg = cp_df.groupby('rank').agg({
            'effective_rank': ['mean', 'std'],
        }).reset_index()
        cp_agg.columns = ['rank', 'eff_mean', 'eff_std']
        
        ax.errorbar(cp_agg['rank'], cp_agg['eff_mean'],
                    yerr=cp_agg['eff_std'],
                    fmt='s-', label='CP effective rank', markersize=10, linewidth=2, capsize=5)
        
        # Ideal line: effective_rank = cp_rank (optional)
        if show_ideal_line:
            max_rank = cp_agg['rank'].max()
            ax.plot([8, max_rank], [8, max_rank], 'k--', alpha=0.5, linewidth=2, 
                    label='Ideal (eff_rank = cp_rank)')
        
        # Add dense baselines as horizontal lines
        if baseline_df is not None and len(baseline_df) > 0:
            mode_col = 'mode' if 'mode' in baseline_df.columns else 'config'
            min_rank = cp_agg['rank'].min()
            max_rank = cp_agg['rank'].max()
            
            # Dense (no reg) - none config
            none_df = baseline_df[baseline_df[mode_col].str.contains('none', case=False)]
            if len(none_df) > 0:
                none_eff = none_df['effective_rank'].mean()
                ax.axhline(y=none_eff, color='red', linestyle='--', linewidth=2, 
                          label=f'Dense (no reg): {none_eff:.1f}')
            
            # Dense (full reg) - full config
            full_df = baseline_df[baseline_df[mode_col].str.contains('full', case=False)]
            if len(full_df) > 0:
                full_eff = full_df['effective_rank'].mean()
                ax.axhline(y=full_eff, color='green', linestyle='--', linewidth=2, 
                          label=f'Dense (full reg): {full_eff:.1f}')
    
    ax.set_xlabel('CP Rank', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Effective Rank vs CP Rank', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cp_pareto_frontier(
    cp_df,
    baseline_df=None,
    ranks=None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot CP accuracy vs effective rank (Pareto frontier).
    
    Matches the style of cp_rank_accuracy_tradeoff.pdf from generate_extension_cp_figures.py.
    
    Args:
        cp_df: DataFrame with CP results (columns: rank, accuracy, effective_rank)
        baseline_df: Optional DataFrame with dense baseline results
        ranks: List of ranks to include (default: all)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(cp_df) > 0:
        if ranks is None:
            ranks = sorted(cp_df['rank'].unique())
        
        # Group by rank and compute mean/std
        rank_stats = cp_df.groupby('rank').agg({
            'accuracy': ['mean', 'std'],
            'effective_rank': ['mean', 'std'],
        }).reset_index()
        rank_stats.columns = ['rank', 'acc_mean', 'acc_std', 'eff_rank_mean', 'eff_rank_std']
        
        # Plot CP models with color by log(rank)
        scatter = ax.scatter(
            rank_stats['eff_rank_mean'], 
            rank_stats['acc_mean'] * 100,
            c=np.log2(rank_stats['rank']),
            cmap='viridis',
            s=150,
            edgecolors='black',
            linewidths=1.5,
            zorder=3,
            label='CP Models'
        )
        
        # Add error bars
        ax.errorbar(
            rank_stats['eff_rank_mean'],
            rank_stats['acc_mean'] * 100,
            xerr=rank_stats['eff_rank_std'],
            yerr=rank_stats['acc_std'] * 100,
            fmt='none',
            color='gray',
            alpha=0.5,
            capsize=3,
            zorder=2,
        )
        
        # Add rank labels
        for _, row in rank_stats.iterrows():
            ax.annotate(
                f"R={int(row['rank'])}",
                (row['eff_rank_mean'], row['acc_mean'] * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='log₂(CP Rank)')
    
    # Dense baselines (all 4 configs)
    if baseline_df is not None and len(baseline_df) > 0:
        mode_col = 'mode' if 'mode' in baseline_df.columns else 'config'
        markers = {'none': 's', 'noise': '^', 'wd': 'v', 'full': 'D'}
        labels = {
            'none': 'Dense (no reg)',
            'noise': 'Dense (noise)',
            'wd': 'Dense (WD)',
            'full': 'Dense (full reg)',
        }
        
        # Group by config and compute mean
        dense_stats = baseline_df.groupby(mode_col).agg({
            'accuracy': 'mean',
            'effective_rank': 'mean',
        }).reset_index()
        
        for _, row in dense_stats.iterrows():
            config = row[mode_col]
            # Extract config name (e.g., 'dense_none' -> 'none')
            config_key = config.replace('dense_', '') if 'dense_' in config else config
            marker = markers.get(config_key, 'o')
            label = labels.get(config_key, f'Dense ({config_key})')
            
            ax.scatter(
                row['effective_rank'],
                row['accuracy'] * 100,
                marker=marker,
                s=100,
                color='red',
                edgecolors='black',
                linewidths=1,
                label=label,
                zorder=4,
            )
    
    ax.set_xlabel('Effective Rank', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('CP Rank vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, title='Dense Baselines')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
