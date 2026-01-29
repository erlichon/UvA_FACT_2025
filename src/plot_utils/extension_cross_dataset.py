"""
Extension 2: Cross-Dataset Structural Robustness plotting functions.

Functions for visualizing USPS transfer, semantic confusion, subspace overlap,
eigenspectra comparisons, eigenvector comparisons, cosine similarity heatmaps,
principal angles, and t-SNE/PCA embeddings for Extension 2 analysis.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.plot_utils.style import COLORS
from src.vision.spectral import effective_rank


# Digit-letter pairs for Extension 2 analysis
# Format: (MNIST digit index, EMNIST letter index, label)
DIGIT_LETTER_PAIRS = [
    (0, 14, "0-O"),  # Zero vs letter O (circularity)
    (1, 8, "1-I"),   # One vs letter I (vertical stroke)
    (2, 25, "2-Z"),  # Two vs letter Z (zigzag)
    (5, 18, "5-S"),  # Five vs letter S (curves)
]


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
                   label='Regularized (sigma=0.15)', color='#1f77b4')
    
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
    vals_left: Float[Tensor, "n_classes n_components"],
    vals_right: Float[Tensor, "n_classes n_components"],
    labels: Optional[Tuple[str, str]] = None,
    figsize: Tuple[float, float] = (14, 5),
    top_k: int = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra side-by-side for two models.
    
    Shows eigenvalue decay patterns with effective rank in title.
    
    Args:
        vals_left: [n_classes, n_components] eigenvalues for left panel
        vals_right: [n_classes, n_components] eigenvalues for right panel
        labels: Optional tuple of (left_label, right_label) for titles
        figsize: Figure size
        top_k: Number of top eigenvalues to show
        title: Optional overall figure title
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    if labels is None:
        labels = ("Model A", "Model B")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute effective ranks
    eff_rank_left = effective_rank(vals_left)
    eff_rank_right = effective_rank(vals_right)
    
    # Plot eigenspectra (sorted by magnitude)
    n_classes = min(vals_left.shape[0], vals_right.shape[0])
    for cls in range(n_classes):
        # Left panel
        vals_sorted = vals_left[cls].abs().sort(descending=True).values
        axes[0].semilogy(vals_sorted[:top_k].numpy(), alpha=0.6)
        
        # Right panel
        vals_sorted = vals_right[cls].abs().sort(descending=True).values
        axes[1].semilogy(vals_sorted[:top_k].numpy(), alpha=0.6)
    
    axes[0].set_title(
        f'{labels[0]}\nMean Eff. Rank: {eff_rank_left.mean():.1f}', 
        fontsize=12
    )
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('|Eigenvalue| (log scale)')
    
    axes[1].set_title(
        f'{labels[1]}\nMean Eff. Rank: {eff_rank_right.mean():.1f}', 
        fontsize=12
    )
    axes[1].set_xlabel('Eigenvalue Index')
    
    if title:
        plt.suptitle(title, fontsize=14, y=1.02)
    else:
        plt.suptitle('Eigenvalue Spectra Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_eigenspectra_comparison_with_ci(
    mnist_eigenvalues_list: List[Float[Tensor, "n_classes n_components"]],
    emnist_eigenvalues_list: List[Float[Tensor, "n_classes n_components"]],
    top_k: int = 50,
    ci_level: float = 0.90,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra with 90% confidence intervals aggregated across seeds.
    
    Each class is plotted as a line with shaded CI region.
    
    Args:
        mnist_eigenvalues_list: List of MNIST eigenvalue tensors from different seeds
                               Each tensor has shape [n_classes, n_components]
        emnist_eigenvalues_list: List of EMNIST eigenvalue tensors from different seeds
        top_k: Number of top eigenvalues to show
        ci_level: Confidence interval level (default: 0.90 for 90% CI)
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    from scipy import stats
    
    n_seeds = len(mnist_eigenvalues_list)
    n_classes = mnist_eigenvalues_list[0].shape[0]
    
    # Colors for each digit class (0-9)
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate the t-value for the CI
    alpha = 1 - ci_level
    t_val = stats.t.ppf(1 - alpha/2, df=n_seeds - 1)
    
    # Helper function to compute mean and CI
    def compute_mean_and_ci(eigenvalues_list, class_idx, top_k):
        # Collect sorted eigenvalues from all seeds
        all_sorted = []
        for vals in eigenvalues_list:
            sorted_vals = vals[class_idx].abs().sort(descending=True).values[:top_k].numpy()
            all_sorted.append(sorted_vals)
        
        all_sorted = np.array(all_sorted)  # [n_seeds, top_k]
        mean = all_sorted.mean(axis=0)
        std = all_sorted.std(axis=0, ddof=1)
        se = std / np.sqrt(n_seeds)
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se
        
        return mean, ci_lower, ci_upper
    
    # Compute effective ranks
    mnist_eff_ranks = []
    emnist_eff_ranks = []
    for vals in mnist_eigenvalues_list:
        mnist_eff_ranks.append(effective_rank(vals).mean().item())
    for vals in emnist_eigenvalues_list:
        emnist_eff_ranks.append(effective_rank(vals).mean().item())
    
    mnist_eff_rank_mean = np.mean(mnist_eff_ranks)
    mnist_eff_rank_std = np.std(mnist_eff_ranks)
    emnist_eff_rank_mean = np.mean(emnist_eff_ranks)
    emnist_eff_rank_std = np.std(emnist_eff_ranks)
    
    x = np.arange(top_k)
    
    # Plot MNIST eigenspectra
    ax = axes[0]
    for cls in range(min(n_classes, 10)):
        mean, ci_lower, ci_upper = compute_mean_and_ci(mnist_eigenvalues_list, cls, top_k)
        
        # Use log scale for y-axis
        line, = ax.semilogy(x, mean, color=class_colors[cls], linewidth=1.5, 
                           label=f'Class {cls}', alpha=0.9)
        ax.fill_between(x, ci_lower, ci_upper, color=class_colors[cls], alpha=0.2)
    
    ax.set_title(f'MNIST (Full Reg)\nEff. Rank: {mnist_eff_rank_mean:.1f} ± {mnist_eff_rank_std:.1f}', 
                 fontsize=12)
    ax.set_xlabel('Eigenvalue Index', fontsize=11)
    ax.set_ylabel('|Eigenvalue| (log scale)', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, top_k-1)
    
    # Plot EMNIST-Digits eigenspectra
    ax = axes[1]
    emnist_n_classes = min(emnist_eigenvalues_list[0].shape[0], 10)
    for cls in range(emnist_n_classes):
        mean, ci_lower, ci_upper = compute_mean_and_ci(emnist_eigenvalues_list, cls, top_k)
        
        line, = ax.semilogy(x, mean, color=class_colors[cls], linewidth=1.5,
                           label=f'Class {cls}', alpha=0.9)
        ax.fill_between(x, ci_lower, ci_upper, color=class_colors[cls], alpha=0.2)
    
    ax.set_title(f'EMNIST-Digits (Regularized)\nEff. Rank: {emnist_eff_rank_mean:.1f} ± {emnist_eff_rank_std:.1f}', 
                 fontsize=12)
    ax.set_xlabel('Eigenvalue Index', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, top_k-1)
    
    plt.suptitle(f'Eigenvalue Spectra: MNIST vs EMNIST-Digits ({n_seeds} seeds, {int(ci_level*100)}% CI)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# NEW FUNCTIONS FOR EXTENSION 2 VISUALIZATION IMPROVEMENTS
# =============================================================================


def _get_top_k_eigenvectors_by_magnitude(
    eigenvalues: Float[Tensor, "n_components"],
    eigenvectors: Float[Tensor, "n_components d_input"],
    k: int,
) -> Tuple[Float[Tensor, "k"], Float[Tensor, "k d_input"]]:
    """
    Select top-k eigenvectors sorted by absolute eigenvalue magnitude.
    
    Args:
        eigenvalues: [n_components] eigenvalues for one class
        eigenvectors: [n_components, d_input] eigenvectors for one class
        k: Number of top eigenvectors to select
    
    Returns:
        (top_k_eigenvalues, top_k_eigenvectors)
    """
    _, indices = eigenvalues.abs().sort(descending=True)
    top_k_indices = indices[:k]
    return eigenvalues[top_k_indices], eigenvectors[top_k_indices]


def plot_digit_letter_eigenvector_comparison(
    mnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    mnist_vals: Float[Tensor, "n_classes n_components"],
    emnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    emnist_vals: Float[Tensor, "n_classes n_components"],
    digit_class: int,
    letter_class: int,
    pair_label: str,
    n_top: int = 5,
    img_shape: Tuple[int, int] = (28, 28),
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot top eigenvectors comparison between a MNIST digit and EMNIST letter.
    
    Shows top-k positive and top-k negative eigenvectors for both digit and letter
    in a 2-row grid (row 1: MNIST digit, row 2: EMNIST letter).
    
    Args:
        mnist_vecs: MNIST eigenvectors [10, n_components, 784]
        mnist_vals: MNIST eigenvalues [10, n_components]
        emnist_vecs: EMNIST eigenvectors [26, n_components, 784]
        emnist_vals: EMNIST eigenvalues [26, n_components]
        digit_class: MNIST digit class index (e.g., 0)
        letter_class: EMNIST letter class index (e.g., 14 for 'O')
        pair_label: Label for the pair (e.g., "0-O")
        n_top: Number of top eigenvectors per sign (positive/negative)
        img_shape: Shape to reshape eigenvectors for visualization
        figsize: Figure size (auto-calculated if None)
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    from src.plot_utils.eigenvectors import plot_eigenvectors_grid
    
    # Extract digit label and letter label from pair_label
    digit_label = pair_label.split("-")[0]
    letter_label = pair_label.split("-")[1]
    
    # Create combined tensors for comparison (2 classes: MNIST and EMNIST)
    combined_vecs = torch.stack([
        mnist_vecs[digit_class].cpu(),
        emnist_vecs[letter_class].cpu()
    ])  # [2, n_components, 784]
    
    combined_vals = torch.stack([
        mnist_vals[digit_class].cpu(),
        emnist_vals[letter_class].cpu()
    ])  # [2, n_components]
    
    title = f"Eigenvector Comparison: MNIST '{digit_label}' vs EMNIST '{letter_label}'"
    
    fig = plot_eigenvectors_grid(
        combined_vecs,
        combined_vals,
        n_top=n_top,
        title=title,
        show_both_signs=True,
        classes=[0, 1],
        class_names=[f"MNIST '{digit_label}'", f"EMNIST '{letter_label}'"],
        img_shape=img_shape,
        save_path=save_path,
    )
    
    return fig


def plot_subspace_overlap_by_rank(
    mnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    emnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    mnist_vals: Float[Tensor, "n_classes n_components"],
    emnist_vals: Float[Tensor, "n_classes n_components"],
    pairs: Optional[List[Tuple[int, int, str]]] = None,
    k_values: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot subspace overlap vs number of eigenvectors used (low-rank approximation).
    
    Shows how subspace similarity changes as more eigenvectors are included,
    from k=1 to k=100. Higher overlap at low k indicates aligned top features.
    
    Args:
        mnist_vecs: MNIST eigenvectors [10, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [26, n_components, 784]
        mnist_vals: MNIST eigenvalues [10, n_components] (for sorting)
        emnist_vals: EMNIST eigenvalues [26, n_components] (for sorting)
        pairs: List of (digit_idx, letter_idx, label) tuples. Defaults to DIGIT_LETTER_PAIRS.
        k_values: List of k values to evaluate. Defaults to [1, 2, ..., 100].
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    from src.vision.subspace import compute_subspace_overlap
    
    if pairs is None:
        pairs = DIGIT_LETTER_PAIRS
    
    if k_values is None:
        max_k = min(100, mnist_vecs.shape[1], emnist_vecs.shape[1])
        k_values = list(range(1, max_k + 1))
    
    # Colors for different pairs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute overlap for each pair at each k value
    for idx, (digit_idx, letter_idx, label) in enumerate(pairs):
        overlaps = []
        
        # Get sorted eigenvectors by magnitude
        _, digit_sorted_idx = mnist_vals[digit_idx].abs().sort(descending=True)
        _, letter_sorted_idx = emnist_vals[letter_idx].abs().sort(descending=True)
        
        digit_vecs_sorted = mnist_vecs[digit_idx][digit_sorted_idx].cpu()
        letter_vecs_sorted = emnist_vecs[letter_idx][letter_sorted_idx].cpu()
        
        for k in k_values:
            overlap = compute_subspace_overlap(
                digit_vecs_sorted[:k],
                letter_vecs_sorted[:k],
                k=k,
                method='mean_cos'
            )
            overlaps.append(overlap)
        
        ax.plot(k_values, overlaps, label=label, color=colors[idx % len(colors)], linewidth=2)
    
    # Compute random baseline (average across many random pairs)
    random_overlaps_by_k = {k: [] for k in k_values}
    for digit_idx in range(10):
        for letter_idx in range(min(26, emnist_vecs.shape[0])):
            # Skip expected pairs
            is_expected = any(digit_idx == p[0] and letter_idx == p[1] for p in pairs)
            if is_expected:
                continue
            
            _, digit_sorted_idx = mnist_vals[digit_idx].abs().sort(descending=True)
            _, letter_sorted_idx = emnist_vals[letter_idx].abs().sort(descending=True)
            
            digit_vecs_sorted = mnist_vecs[digit_idx][digit_sorted_idx].cpu()
            letter_vecs_sorted = emnist_vecs[letter_idx][letter_sorted_idx].cpu()
            
            # Sample a subset of k values for efficiency
            for k in [1, 5, 10, 20, 50, 100]:
                if k in k_values and k <= min(digit_vecs_sorted.shape[0], letter_vecs_sorted.shape[0]):
                    overlap = compute_subspace_overlap(
                        digit_vecs_sorted[:k],
                        letter_vecs_sorted[:k],
                        k=k,
                        method='mean_cos'
                    )
                    random_overlaps_by_k[k].append(overlap)
    
    # Plot random baseline as shaded region
    random_k = sorted([k for k in random_overlaps_by_k.keys() if len(random_overlaps_by_k[k]) > 0])
    random_means = [np.mean(random_overlaps_by_k[k]) for k in random_k]
    random_stds = [np.std(random_overlaps_by_k[k]) for k in random_k]
    
    ax.plot(random_k, random_means, 'k--', label='Random baseline', linewidth=1.5, alpha=0.7)
    ax.fill_between(random_k, 
                    np.array(random_means) - np.array(random_stds),
                    np.array(random_means) + np.array(random_stds),
                    color='gray', alpha=0.2)
    
    ax.set_xlabel('Number of Eigenvectors (k)', fontsize=12)
    ax.set_ylabel('Subspace Overlap (mean cosine)', fontsize=12)
    ax.set_title('Subspace Overlap vs Low-Rank Approximation', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim(1, max(k_values))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_cosine_similarity_heatmap(
    digit_vecs: Float[Tensor, "n_components d_input"],
    letter_vecs: Float[Tensor, "n_components d_input"],
    digit_vals: Float[Tensor, "n_components"],
    letter_vals: Float[Tensor, "n_components"],
    k: int = 10,
    digit_label: str = "0",
    letter_label: str = "O",
    figsize: Tuple[float, float] = (8, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot k×k cosine similarity heatmap between top-k eigenvectors of two classes.
    
    Shows pairwise absolute cosine similarity between eigenvectors,
    helping visualize which eigenvectors are aligned.
    
    Args:
        digit_vecs: [n_components, d_input] eigenvectors for MNIST digit
        letter_vecs: [n_components, d_input] eigenvectors for EMNIST letter
        digit_vals: [n_components] eigenvalues for MNIST digit
        letter_vals: [n_components] eigenvalues for EMNIST letter
        k: Number of top eigenvectors to compare
        digit_label: Label for the digit (e.g., "0")
        letter_label: Label for the letter (e.g., "O")
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    # Get top-k eigenvectors by eigenvalue magnitude
    _, digit_sorted = digit_vals.abs().sort(descending=True)
    _, letter_sorted = letter_vals.abs().sort(descending=True)
    
    digit_top_k = digit_vecs[digit_sorted[:k]].cpu()
    letter_top_k = letter_vecs[letter_sorted[:k]].cpu()
    
    # Normalize eigenvectors
    digit_norm = digit_top_k / (digit_top_k.norm(dim=1, keepdim=True) + 1e-10)
    letter_norm = letter_top_k / (letter_top_k.norm(dim=1, keepdim=True) + 1e-10)
    
    # Compute absolute cosine similarity matrix
    cos_matrix = (digit_norm @ letter_norm.T).abs().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cos_matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(k):
        for j in range(k):
            color = 'white' if cos_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cos_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=8, color=color)
    
    ax.set_xticks(range(k))
    ax.set_xticklabels([f'{i+1}' for i in range(k)], fontsize=10)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f'{i+1}' for i in range(k)], fontsize=10)
    ax.set_xlabel(f"EMNIST '{letter_label}' Eigenvector Rank", fontsize=12)
    ax.set_ylabel(f"MNIST '{digit_label}' Eigenvector Rank", fontsize=12)
    ax.set_title(f"Cosine Similarity: '{digit_label}' vs '{letter_label}' (top {k})", fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, label='|Cosine Similarity|')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_eigenvalue_distribution_overlay(
    digit_vals: Float[Tensor, "n_components"],
    letter_vals: Float[Tensor, "n_components"],
    digit_label: str = "0",
    letter_label: str = "O",
    n_bins: int = 30,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlapping histograms of eigenvalue magnitude distributions.
    
    Shows if MNIST digit and EMNIST letter have similar eigenvalue spectra,
    which would indicate similar representational structure.
    
    Args:
        digit_vals: [n_components] eigenvalues for MNIST digit
        letter_vals: [n_components] eigenvalues for EMNIST letter
        digit_label: Label for the digit (e.g., "0")
        letter_label: Label for the letter (e.g., "O")
        n_bins: Number of histogram bins
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get absolute eigenvalues
    digit_abs = digit_vals.abs().cpu().numpy()
    letter_abs = letter_vals.abs().cpu().numpy()
    
    # Compute shared bin edges
    all_vals = np.concatenate([digit_abs, letter_abs])
    bins = np.logspace(np.log10(all_vals.min() + 1e-10), np.log10(all_vals.max()), n_bins)
    
    # Plot histograms
    ax.hist(digit_abs, bins=bins, alpha=0.6, label=f"MNIST '{digit_label}'", color='#1f77b4')
    ax.hist(letter_abs, bins=bins, alpha=0.6, label=f"EMNIST '{letter_label}'", color='#ff7f0e')
    
    # Add vertical lines for means
    digit_mean = digit_abs.mean()
    letter_mean = letter_abs.mean()
    ax.axvline(digit_mean, color='#1f77b4', linestyle='--', linewidth=2, 
               label=f"MNIST mean: {digit_mean:.4f}")
    ax.axvline(letter_mean, color='#ff7f0e', linestyle='--', linewidth=2,
               label=f"EMNIST mean: {letter_mean:.4f}")
    
    ax.set_xscale('log')
    ax.set_xlabel('|Eigenvalue| (log scale)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f"Eigenvalue Distribution: '{digit_label}' vs '{letter_label}'", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_principal_angles(
    mnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    emnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    mnist_vals: Float[Tensor, "n_classes n_components"],
    emnist_vals: Float[Tensor, "n_classes n_components"],
    pairs: Optional[List[Tuple[int, int, str]]] = None,
    k: int = 20,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot principal angles between digit and letter subspaces for all pairs.
    
    Principal angles measure how aligned two subspaces are:
    - θ = 0° means subspaces share a direction
    - θ = 90° means orthogonal (no similarity)
    
    If first few angles are near 0°, the subspaces share aligned directions.
    
    Args:
        mnist_vecs: MNIST eigenvectors [10, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [26, n_components, 784]
        mnist_vals: MNIST eigenvalues [10, n_components]
        emnist_vals: EMNIST eigenvalues [26, n_components]
        pairs: List of (digit_idx, letter_idx, label) tuples
        k: Number of principal angles to compute
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    from src.vision.subspace import principal_angles
    
    if pairs is None:
        pairs = DIGIT_LETTER_PAIRS
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bar_width = 0.18
    x = np.arange(k)
    
    for idx, (digit_idx, letter_idx, label) in enumerate(pairs):
        # Get sorted eigenvectors
        _, digit_sorted_idx = mnist_vals[digit_idx].abs().sort(descending=True)
        _, letter_sorted_idx = emnist_vals[letter_idx].abs().sort(descending=True)
        
        digit_vecs_sorted = mnist_vecs[digit_idx][digit_sorted_idx[:k]].cpu()
        letter_vecs_sorted = emnist_vecs[letter_idx][letter_sorted_idx[:k]].cpu()
        
        # Compute principal angles
        angles = principal_angles(digit_vecs_sorted, letter_vecs_sorted)
        angles_deg = np.degrees(angles.numpy())
        
        # Plot as grouped bars
        offset = (idx - len(pairs)/2 + 0.5) * bar_width
        ax.bar(x + offset, angles_deg[:k], bar_width, label=label, color=colors[idx % len(colors)], alpha=0.8)
    
    # Reference lines
    ax.axhline(0, color='green', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(45, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='45° (moderate)')
    ax.axhline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90° (orthogonal)')
    
    ax.set_xlabel('Principal Angle Index', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Principal Angles Between Digit-Letter Subspaces', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(k)])
    ax.set_ylim(0, 95)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_eigenvector_embedding(
    mnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    emnist_vecs: Float[Tensor, "n_classes n_components d_input"],
    mnist_vals: Float[Tensor, "n_classes n_components"],
    emnist_vals: Float[Tensor, "n_classes n_components"],
    pairs: Optional[List[Tuple[int, int, str]]] = None,
    k: int = 5,
    method: str = "pca",
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot 2D embedding (t-SNE or PCA) of top eigenvectors from digit-letter pairs.
    
    If universal features are learned, MNIST and EMNIST eigenvectors for the
    same shape should cluster together.
    
    Args:
        mnist_vecs: MNIST eigenvectors [10, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [26, n_components, 784]
        mnist_vals: MNIST eigenvalues [10, n_components]
        emnist_vals: EMNIST eigenvalues [26, n_components]
        pairs: List of (digit_idx, letter_idx, label) tuples
        k: Number of top eigenvectors per class to include
        method: "pca" or "tsne"
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object, or None if sklearn is not available
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Warning: sklearn not available, skipping eigenvector embedding")
        return None
    
    try:
        from sklearn.manifold import TSNE
        has_tsne = True
    except ImportError:
        has_tsne = False
        if method == "tsne":
            print("Warning: TSNE not available, falling back to PCA")
            method = "pca"
    
    if pairs is None:
        pairs = DIGIT_LETTER_PAIRS
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Collect eigenvectors from all pairs
    all_vecs = []
    labels = []
    sources = []  # 'mnist' or 'emnist'
    pair_indices = []
    
    for idx, (digit_idx, letter_idx, label) in enumerate(pairs):
        # Get sorted MNIST eigenvectors
        _, digit_sorted_idx = mnist_vals[digit_idx].abs().sort(descending=True)
        digit_top_k = mnist_vecs[digit_idx][digit_sorted_idx[:k]].cpu().numpy()
        
        for vec in digit_top_k:
            all_vecs.append(vec)
            labels.append(label)
            sources.append('mnist')
            pair_indices.append(idx)
        
        # Get sorted EMNIST eigenvectors
        _, letter_sorted_idx = emnist_vals[letter_idx].abs().sort(descending=True)
        letter_top_k = emnist_vecs[letter_idx][letter_sorted_idx[:k]].cpu().numpy()
        
        for vec in letter_top_k:
            all_vecs.append(vec)
            labels.append(label)
            sources.append('emnist')
            pair_indices.append(idx)
    
    all_vecs = np.array(all_vecs)  # [n_samples, 784]
    
    # Compute 2D embedding
    if method == "tsne" and has_tsne:
        n_samples = all_vecs.shape[0]
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(all_vecs)
    else:
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(all_vecs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    for idx, (digit_idx, letter_idx, label) in enumerate(pairs):
        # MNIST points (circles)
        mask_mnist = np.array([(p == idx and s == 'mnist') for p, s in zip(pair_indices, sources)])
        ax.scatter(embedding[mask_mnist, 0], embedding[mask_mnist, 1],
                  c=colors[idx], marker='o', s=100, alpha=0.8, 
                  label=f'{label} (MNIST)', edgecolors='black', linewidths=0.5)
        
        # EMNIST points (triangles)
        mask_emnist = np.array([(p == idx and s == 'emnist') for p, s in zip(pair_indices, sources)])
        ax.scatter(embedding[mask_emnist, 0], embedding[mask_emnist, 1],
                  c=colors[idx], marker='^', s=100, alpha=0.8,
                  label=f'{label} (EMNIST)', edgecolors='black', linewidths=0.5)
    
    method_name = "t-SNE" if method == "tsne" else "PCA"
    ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
    ax.set_title(f'{method_name} of Top-{k} Eigenvectors by Digit-Letter Pair', fontsize=14)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
