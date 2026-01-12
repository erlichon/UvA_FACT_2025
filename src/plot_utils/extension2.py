"""
Extension 2: Cross-Dataset Structural Robustness plotting functions.

Functions for visualizing USPS transfer, semantic confusion, subspace overlap,
and eigenspectra comparisons for Extension 2 analysis.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.plot_utils.style import COLORS
from src.analysis.spectral import effective_rank


def plot_usps_transfer_comparison(
    usps_results: Dict,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot USPS transfer test results comparing baseline vs regularized models.
    
    Shows per-digit accuracy on USPS dataset for MNIST-trained models.
    
    Args:
        usps_results: Dict with keys 'baseline' and 'regularized', each containing
                      'per_class_accuracy' dict mapping digit -> {'accuracy': float}
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    digits = list(range(10))
    baseline_accs = [
        usps_results['baseline']['per_class_accuracy'].get(str(d), {}).get('accuracy', 0) 
        for d in digits
    ]
    reg_accs = [
        usps_results['regularized']['per_class_accuracy'].get(str(d), {}).get('accuracy', 0) 
        for d in digits
    ]
    
    x = np.arange(10)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_accs, width, 
                   label='Baseline (no noise)', color='#d62728')
    bars2 = ax.bar(x + width/2, reg_accs, width, 
                   label='Regularized (σ=0.15)', color='#1f77b4')
    
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Accuracy on USPS', fontsize=12)
    ax.set_title('USPS Transfer Test: MNIST-Trained Models on USPS Digits', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_semantic_confusion_distributions(
    semantic_results: Dict,
    letter_digit_similarity: Dict[str, int],
    figsize: Tuple[float, float] = (15, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot semantic confusion test results showing letter classification distributions.
    
    Creates 2x5 grid showing how baseline vs regularized models classify
    EMNIST letters that look like digits.
    
    Args:
        semantic_results: Dict with keys 'baseline' and 'regularized', each containing
                         letter -> {'prediction_distribution': [10 floats]}
        letter_digit_similarity: Dict mapping letter -> expected digit (e.g. 'O' -> 0)
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    letters = ['O', 'I', 'Z', 'S', 'B']
    
    for i, letter in enumerate(letters):
        baseline = semantic_results['baseline'][letter]
        regularized = semantic_results['regularized'][letter]
        expected = letter_digit_similarity[letter]
        
        # Baseline
        ax = axes[0, i]
        dist = baseline['prediction_distribution']
        colors = ['green' if j == expected else 'steelblue' for j in range(10)]
        ax.bar(range(10), dist, color=colors)
        ax.set_xlabel('Predicted Digit')
        ax.set_title(f"Baseline: '{letter}'")
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Proportion')
        
        # Regularized
        ax = axes[1, i]
        dist = regularized['prediction_distribution']
        colors = ['green' if j == expected else 'steelblue' for j in range(10)]
        ax.bar(range(10), dist, color=colors)
        ax.set_xlabel('Predicted Digit')
        ax.set_title(f"Regularized: '{letter}'")
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Proportion')
    
    plt.suptitle('Semantic Confusion Test: EMNIST Letter Classification', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_subspace_overlap_comparison(
    subspace_results: Dict,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot subspace overlap comparison for expected shape-similar pairs vs random baseline.
    
    Shows eigenvector subspace similarity between MNIST digits and EMNIST letters.
    
    Args:
        subspace_results: Dict with keys:
                         - 'expected_pairs': {pair_name -> {'mean_cos': float, ...}}
                         - 'random_baseline': {'mean': float, 'std': float}
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    pairs = list(subspace_results['expected_pairs'].keys())
    mean_cos = [subspace_results['expected_pairs'][p]['mean_cos'] for p in pairs]
    random_mean = subspace_results['random_baseline']['mean']
    random_std = subspace_results['random_baseline']['std']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar chart
    x = np.arange(len(pairs))
    bars = ax.bar(x, mean_cos, color='steelblue', label='Expected pairs')
    
    # Random baseline
    ax.axhline(random_mean, color='red', linestyle='--', 
               label=f'Random baseline ({random_mean:.3f})')
    ax.fill_between([-0.5, len(pairs)-0.5], 
                    random_mean-random_std, random_mean+random_std, 
                    color='red', alpha=0.1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=11)
    ax.set_ylabel('Subspace Overlap (mean cosine)', fontsize=12)
    ax.set_xlabel('MNIST Digit - EMNIST Letter Pair', fontsize=12)
    ax.set_title('Eigenvector Subspace Similarity: Expected Shape Pairs', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_eigenspectra_side_by_side(
    vals_baseline: Float[Tensor, "n_classes n_components"],
    vals_regularized: Float[Tensor, "n_classes n_components"],
    figsize: Tuple[float, float] = (14, 5),
    top_k: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra side-by-side for baseline vs regularized models.
    
    Shows eigenvalue decay patterns with effective rank in title.
    
    Args:
        vals_baseline: [n_classes, n_components] eigenvalues for baseline model
        vals_regularized: [n_classes, n_components] eigenvalues for regularized model
        figsize: Figure size
        top_k: Number of top eigenvalues to show
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute effective ranks
    eff_rank_baseline = effective_rank(vals_baseline)
    eff_rank_regularized = effective_rank(vals_regularized)
    
    # Plot eigenspectra (sorted by magnitude)
    for cls in range(10):
        # Baseline
        vals_sorted = vals_baseline[cls].abs().sort(descending=True).values
        axes[0].semilogy(vals_sorted[:top_k].numpy(), alpha=0.6)
        
        # Regularized
        vals_sorted = vals_regularized[cls].abs().sort(descending=True).values
        axes[1].semilogy(vals_sorted[:top_k].numpy(), alpha=0.6)
    
    axes[0].set_title(
        f'Baseline (No Noise)\nMean Eff. Rank: {eff_rank_baseline.mean():.1f}', 
        fontsize=12
    )
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('|Eigenvalue| (log scale)')
    
    axes[1].set_title(
        f'Regularized (σ=0.15)\nMean Eff. Rank: {eff_rank_regularized.mean():.1f}', 
        fontsize=12
    )
    axes[1].set_xlabel('Eigenvalue Index')
    
    plt.suptitle('Eigenvalue Spectra: Effect of Regularization', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
