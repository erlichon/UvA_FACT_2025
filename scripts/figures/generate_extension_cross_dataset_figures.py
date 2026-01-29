#!/usr/bin/env python3
"""
Generate Cross-Dataset Robustness figures (Extension 1).

This script is the single entrypoint for all cross-dataset figures. It only uses
existing checkpoints (no new training).

Figures generated:

Original (subspace-based metrics):
1. Eigenvector comparison: MNIST digit vs EMNIST letter (per pair)
2. Eigenvalue distribution overlays (per pair)
3. Subspace overlap vs k (mean_cos method)
4. Similarity heatmap (10x26 matrix, mean_cos)
5. Principal angles between subspaces
6. 3-way comparison: MNIST digit 0, EMNIST digit 0, EMNIST letter O
7. Selection method comparison: magnitude vs balanced

New (eigenvalue-aware metrics):
8. Similarity vs k for eigenvalue_weighted metric
9. Similarity vs k for quadratic_form metric
10. Heatmaps for eigenvalue_weighted and quadratic_form metrics
11. Quadratic form heatmap (MNIST digits vs EMNIST digits, 10x10 matrix)
12. 3-way metric comparison (mean_cos + 2 weighted)
13. Ranking analysis: where expected pairs rank among all letters
14. Statistical comparison: t-test similar vs dissimilar pairs

Usage:
    python scripts/figures/generate_extension_cross_dataset_figures.py
    python scripts/figures/generate_extension_cross_dataset_figures.py --sections eigenvectors
    python scripts/figures/generate_extension_cross_dataset_figures.py --sections similarity_weighted ranking
    ./scripts/train/run_extension_cross_dataset.sh figures  # Preferred wrapper
"""

import sys
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import matplotlib
from scipy import stats

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Add project + original code paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.paths import (
    MNIST_CHECKPOINTS,
    EXTENSION2_CHECKPOINTS,
    EXTENSION2_FIGURES,
)
from src.vision.spectral import load_checkpoint_eigenvalues
from src.vision.subspace import (
    compute_subspace_overlap,
    select_balanced_eigenvectors,
    compute_weighted_similarity,
)
from src.plot_utils.style import set_publication_style
from src.artifact_loader import ensure_artifacts
from src.plot_utils.extension2 import (
    DIGIT_LETTER_PAIRS,
    plot_digit_letter_eigenvector_comparison,
    plot_eigenvalue_distribution_overlay,
    plot_subspace_overlap_by_rank,
    plot_principal_angles,
    plot_eigenvector_embedding,
)
from src.plot_utils.eigenvectors import plot_eigenvectors_grid

warnings.filterwarnings("ignore", message="Failed to load image Python extension:*")


@dataclass(frozen=True)
class Dirs:
    """Directory paths for Extension 2."""
    mnist_ckpt: Path
    emnist_digits_ckpt: Path
    emnist_letters_ckpt: Path
    figure_out: Path


def get_dirs() -> Dirs:
    """Get directory paths, with fallbacks for different checkpoint names.
    
    For cross-dataset robustness, ALL models should be trained with CoM normalization.
    MNIST CoM checkpoints are in checkpoints/extension2/ (trained via run_extension_cross_dataset.sh).
    """
    # MNIST: Prefer extension2 CoM checkpoint, fall back to vision/mnist (non-CoM, but warn)
    mnist_candidates = [
        EXTENSION2_CHECKPOINTS / "mnist_dense_full_com_seed42.pt",  # Extension 2 CoM (preferred)
        MNIST_CHECKPOINTS / "mnist_dense_full_seed42.pt",  # Vision (fallback)
    ]
    emnist_digits_candidates = [
        EXTENSION2_CHECKPOINTS / "emnist_digits_regularized_seed42.pt",
    ]
    emnist_letters_candidates = [
        EXTENSION2_CHECKPOINTS / "emnist_letters_regularized_seed42.pt",
    ]
    
    mnist_ckpt = next((p for p in mnist_candidates if p.exists()), mnist_candidates[0])
    emnist_digits_ckpt = next((p for p in emnist_digits_candidates if p.exists()), emnist_digits_candidates[0])
    emnist_letters_ckpt = next((p for p in emnist_letters_candidates if p.exists()), emnist_letters_candidates[0])
    
    # Warn if using non-CoM MNIST checkpoint
    if "vision" in str(mnist_ckpt) or "dense_full_seed" in str(mnist_ckpt):
        print("WARNING: Using vision MNIST checkpoint (no CoM).")
        print("   For fair cross-dataset comparison, train MNIST with CoM:")
        print("   ./scripts/train/run_extension_cross_dataset.sh train mnist")
    
    return Dirs(
        mnist_ckpt=mnist_ckpt,
        emnist_digits_ckpt=emnist_digits_ckpt,
        emnist_letters_ckpt=emnist_letters_ckpt,
        figure_out=EXTENSION2_FIGURES,
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    """Save figure to the specified path."""
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"  Saved: {out_path}")


def load_checkpoints(d: Dirs) -> Tuple[
    torch.Tensor, torch.Tensor,  # MNIST vals, vecs
    Optional[torch.Tensor], Optional[torch.Tensor],  # EMNIST digits vals, vecs
    torch.Tensor, torch.Tensor,  # EMNIST letters vals, vecs
]:
    """Load all required checkpoints."""
    print("Loading checkpoints...")
    
    # MNIST
    if not d.mnist_ckpt.exists():
        raise FileNotFoundError(f"MNIST checkpoint not found: {d.mnist_ckpt}")
    mnist_vals, mnist_vecs = load_checkpoint_eigenvalues(str(d.mnist_ckpt))
    print(f"  MNIST: {mnist_vecs.shape}")
    
    # EMNIST Digits (optional)
    digits_vals, digits_vecs = None, None
    if d.emnist_digits_ckpt.exists():
        digits_vals, digits_vecs = load_checkpoint_eigenvalues(str(d.emnist_digits_ckpt))
        print(f"  EMNIST Digits: {digits_vecs.shape}")
    else:
        print(f"  EMNIST Digits: not found (skipping 3-way comparison)")
    
    # EMNIST Letters
    if not d.emnist_letters_ckpt.exists():
        raise FileNotFoundError(f"EMNIST Letters checkpoint not found: {d.emnist_letters_ckpt}")
    letters_vals, letters_vecs = load_checkpoint_eigenvalues(str(d.emnist_letters_ckpt))
    print(f"  EMNIST Letters: {letters_vecs.shape}")
    
    return mnist_vals, mnist_vecs, digits_vals, digits_vecs, letters_vals, letters_vecs


def generate_eigenvector_comparisons(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate eigenvector comparison figures for each digit-letter pair."""
    print("\n=== Eigenvector Comparisons ===")
    
    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
        fig = plot_digit_letter_eigenvector_comparison(
            mnist_vecs, mnist_vals,
            letters_vecs, letters_vals,
            digit_class=digit_idx,
            letter_class=letter_idx,
            pair_label=label,
            n_top=5,
        )
        
        digit, letter = label.split("-")
        out_path = d.figure_out / f"extension2_eigenvec_{digit}_{letter}.pdf"
        _save_figure(fig, out_path)
        plt.close(fig)


def generate_eigenvalue_distributions(
    d: Dirs,
    mnist_vals: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate eigenvalue distribution overlays for each digit-letter pair."""
    print("\n=== Eigenvalue Distributions ===")
    
    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
        digit, letter = label.split("-")
        
        fig = plot_eigenvalue_distribution_overlay(
            mnist_vals[digit_idx],
            letters_vals[letter_idx],
            digit_label=digit,
            letter_label=letter,
        )
        
        out_path = d.figure_out / f"extension2_eigenval_{digit}_{letter}.pdf"
        _save_figure(fig, out_path)
        plt.close(fig)


def generate_similarity_vs_k(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate subspace overlap vs k plot using mean cosine of principal angles."""
    print("\n=== Similarity vs k ===")
    
    k_values = list(range(2, 101))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute for expected similar pairs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (digit_idx, letter_idx, label) in enumerate(DIGIT_LETTER_PAIRS):
        overlaps = []
        
        _, d_sorted = mnist_vals[digit_idx].abs().sort(descending=True)
        _, l_sorted = letters_vals[letter_idx].abs().sort(descending=True)
        
        d_vecs = mnist_vecs[digit_idx][d_sorted].cpu()
        l_vecs = letters_vecs[letter_idx][l_sorted].cpu()
        
        for k in k_values:
            overlap = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
            overlaps.append(overlap)
        
        ax.plot(k_values, overlaps, label=f'{label} (similar)', color=colors[idx], linewidth=2)
    
    # Control baseline: dissimilar cross-model pairs
    control_pairs = [
        (0, 23, "0-X"),  # Round digit vs crossed letter
        (1, 22, "1-W"),  # Vertical stroke vs wide angular
        (3, 7, "3-H"),   # Curved digit vs rectangular letter
        (7, 14, "7-O"),  # Angular stroke vs round letter
    ]
    
    control_overlaps_by_k = {k: [] for k in k_values}
    
    for d_idx, l_idx, _ in control_pairs:
        _, d_sorted = mnist_vals[d_idx].abs().sort(descending=True)
        _, l_sorted = letters_vals[l_idx].abs().sort(descending=True)
        d_vecs = mnist_vecs[d_idx][d_sorted].cpu()
        l_vecs = letters_vecs[l_idx][l_sorted].cpu()
        
        for k in k_values:
            overlap = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
            control_overlaps_by_k[k].append(overlap)
    
    control_means = [np.mean(control_overlaps_by_k[k]) for k in k_values]
    control_stds = [np.std(control_overlaps_by_k[k]) for k in k_values]
    
    ax.plot(k_values, control_means, 'k--', label='Dissimilar (0-X, 1-W, 3-H, 7-O)', linewidth=1.5, alpha=0.7)
    ax.fill_between(k_values, 
                    np.array(control_means) - np.array(control_stds),
                    np.array(control_means) + np.array(control_stds),
                    color='gray', alpha=0.2)
    
    ax.set_xlabel('k (number of eigenvectors)', fontsize=12)
    ax.set_ylabel('Subspace Cosine Similarity', fontsize=12)
    ax.set_title('Subspace Overlap vs Number of Eigenvectors', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_similarity_vs_k.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_similarity_heatmap(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate heatmap of subspace cosine similarity for all digit-letter pairs."""
    print("\n=== Similarity Heatmap ===")
    
    # Helper to get sorted eigenvectors
    def get_sorted_vecs(vals, vecs, class_idx):
        _, sorted_idx = vals[class_idx].abs().sort(descending=True)
        return vecs[class_idx][sorted_idx].cpu()
    
    # Compute full 10x26 matrix using mean_cos (original metric)
    matrix = np.zeros((10, 26))
    for digit in range(10):
        d_vecs = get_sorted_vecs(mnist_vals, mnist_vecs, digit)
        for letter in range(26):
            l_vecs = get_sorted_vecs(letters_vals, letters_vecs, letter)
            matrix[digit, letter] = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    letter_labels = [chr(65+i) for i in range(26)]
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0.35, vmax=0.65)
    
    # Add text annotations
    for digit in range(10):
        for letter in range(26):
            color = 'white' if matrix[digit, letter] > 0.55 else 'black'
            ax.text(letter, digit, f'{matrix[digit, letter]:.2f}', 
                   ha='center', va='center', fontsize=7, color=color)
    
    ax.set_xticks(range(26))
    ax.set_xticklabels(letter_labels, fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10), fontsize=10)
    ax.set_xlabel('EMNIST Letter', fontsize=12)
    ax.set_ylabel('MNIST Digit', fontsize=12)
    ax.set_title(f'Subspace Cosine Similarity (k={k})', fontsize=14)
    
    # Mark expected similar pairs with blue boxes
    for digit_idx, letter_idx, _ in DIGIT_LETTER_PAIRS:
        rect = plt.Rectangle((letter_idx-0.5, digit_idx-0.5), 1, 1, 
                             fill=False, edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
    
    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_similarity_heatmap.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)
    
    # Print statistics
    similar_vals = [matrix[d, l] for d, l, _ in DIGIT_LETTER_PAIRS]
    print(f"  Expected similar pairs mean: {np.mean(similar_vals):.4f}")
    print(f"  Overall mean: {matrix.mean():.4f}")
    
    # Print rank of expected match
    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
        rank = (matrix[digit_idx] >= matrix[digit_idx, letter_idx]).sum()
        print(f"  {label}: rank {rank}/26")


def generate_3way_comparison(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    digits_vecs: Optional[torch.Tensor],
    digits_vals: Optional[torch.Tensor],
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate 3-way eigenvector comparison: MNIST digit 0, EMNIST digit 0, EMNIST letter O."""
    print("\n=== 3-Way Comparison ===")
    
    if digits_vecs is None:
        print("  Skipping: EMNIST Digits checkpoint not available")
        return
    
    # Create combined tensors for 3-way comparison
    combined_vecs = torch.stack([
        mnist_vecs[0].cpu(),    # MNIST digit 0
        digits_vecs[0].cpu(),   # EMNIST digit 0
        letters_vecs[14].cpu()  # EMNIST letter O
    ])
    
    combined_vals = torch.stack([
        mnist_vals[0].cpu(),
        digits_vals[0].cpu(),
        letters_vals[14].cpu()
    ])
    
    fig = plot_eigenvectors_grid(
        combined_vecs,
        combined_vals,
        n_top=5,
        title="Eigenvector Comparison: Digit '0' vs Letter 'O'",
        show_both_signs=True,
        classes=[0, 1, 2],
        class_names=["MNIST Digit '0'", "EMNIST Digit '0'", "EMNIST Letter 'O'"],
    )
    
    out_path = d.figure_out / "extension2_3way_comparison.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_3way_comparison_0_O_X(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate 3-way comparison: MNIST digit 0, EMNIST letter O, EMNIST letter X.

    This is the figure used in the report for cosine-similarity failure analysis.
    """
    print("\n=== 3-Way Comparison (0 vs O vs X) ===")

    # Class indices
    mnist_class = 0      # digit 0
    letter_o_idx = 14    # letter O (A=0, ..., O=14)
    letter_x_idx = 23    # letter X (A=0, ..., X=23)

    # Stack eigenvectors and eigenvalues for the three classes
    combined_vecs = torch.stack([
        mnist_vecs[mnist_class].cpu(),       # MNIST digit 0
        letters_vecs[letter_o_idx].cpu(),    # EMNIST letter O
        letters_vecs[letter_x_idx].cpu(),    # EMNIST letter X
    ])

    combined_vals = torch.stack([
        mnist_vals[mnist_class].cpu(),
        letters_vals[letter_o_idx].cpu(),
        letters_vals[letter_x_idx].cpu(),
    ])

    fig = plot_eigenvectors_grid(
        combined_vecs,
        combined_vals,
        n_top=5,
        title="Eigenvector Comparison: Digit '0' vs Letters 'O' and 'X'",
        show_both_signs=True,
        classes=[0, 1, 2],
        class_names=["MNIST Digit '0'", "EMNIST Letter 'O'", "EMNIST Letter 'X'"],
    )

    out_path = d.figure_out / "extension2_3way_comparison_0_O_X.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_eigenvector_comparison_table(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    digits_vecs: Optional[torch.Tensor],
    digits_vals: Optional[torch.Tensor],
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 10,
) -> None:
    """Generate LaTeX table comparing top-k eigenvectors using absolute cosine similarity.
    
    Uses balanced selection: k//2 positive and k-k//2 negative eigenvectors.
    Computes mean results over 5 seeds (42, 43, 44, 45, 46).
    
    Compares:
    - MNIST digit 0 vs EMNIST digit 0
    - MNIST digit 0 vs EMNIST letter O
    
    Args:
        d: Directory paths
        mnist_vecs: MNIST eigenvectors [10, d_hidden, d_input] (seed 42, used for reference)
        mnist_vals: MNIST eigenvalues [10, d_hidden] (seed 42, used for reference)
        digits_vecs: EMNIST digits eigenvectors [10, d_hidden, d_input] or None (seed 42, used for reference)
        digits_vals: EMNIST digits eigenvalues [10, d_hidden] or None (seed 42, used for reference)
        letters_vecs: EMNIST letters eigenvectors [26, d_hidden, d_input] (seed 42, used for reference)
        letters_vals: EMNIST letters eigenvalues [26, d_hidden] (seed 42, used for reference)
        k: Number of eigenvectors to compare (default: 10, selects 5 positive + 5 negative)
    """
    print("\n=== Eigenvector Comparison Table (Balanced Selection, Mean over 5 seeds) ===")
    
    # Load checkpoints for all 5 seeds
    seeds = [42, 43, 44, 45, 46]
    ext2_ckpt_dir = EXTENSION2_CHECKPOINTS
    vision_ckpt_dir = MNIST_CHECKPOINTS
    
    # Helper function to get balanced eigenvectors and eigenvalues
    def get_balanced_eigenvectors_and_vals(vals, vecs, class_idx, k):
        """Get balanced eigenvectors and corresponding eigenvalues."""
        class_vals = vals[class_idx].cpu()
        class_vecs = vecs[class_idx].cpu()
        
        pos_idx = torch.nonzero(class_vals > 0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(class_vals < 0, as_tuple=False).squeeze(-1)
        
        pos_sorted = pos_idx[class_vals[pos_idx].abs().argsort(descending=True)] if pos_idx.numel() > 0 else pos_idx
        neg_sorted = neg_idx[class_vals[neg_idx].abs().argsort(descending=True)] if neg_idx.numel() > 0 else neg_idx
        
        k_pos = k // 2
        k_neg = k - k_pos
        
        selected_idx = []
        if k_pos > 0 and pos_sorted.numel() > 0:
            selected_idx.extend(pos_sorted[:k_pos].tolist())
        if k_neg > 0 and neg_sorted.numel() > 0:
            selected_idx.extend(neg_sorted[:k_neg].tolist())
        
        # Fill remaining if needed
        if len(selected_idx) < k:
            selected_mask = torch.zeros_like(class_vals, dtype=torch.bool)
            if len(selected_idx) > 0:
                selected_mask[selected_idx] = True
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            if remaining.numel() > 0:
                remaining_sorted = remaining[class_vals[remaining].abs().argsort(descending=True)]
                needed = k - len(selected_idx)
                selected_idx.extend(remaining_sorted[:needed].tolist())
        
        return class_vecs[selected_idx], class_vals[selected_idx]
    
    # Compute absolute cosine similarity
    def abs_cosine_similarity(vecs_A, vecs_B):
        """Compute absolute cosine similarity between two sets of eigenvectors."""
        vecs_A_norm = vecs_A / (vecs_A.norm(dim=1, keepdim=True) + 1e-10)
        vecs_B_norm = vecs_B / (vecs_B.norm(dim=1, keepdim=True) + 1e-10)
        cos_matrix = vecs_A_norm @ vecs_B_norm.T
        return cos_matrix.abs()
    
    # Accumulate similarities across seeds
    sim_mnist_emnist_digit_by_seed = []
    sim_mnist_emnist_letter_by_seed = []
    loaded_seeds = []
    
    for seed in seeds:
        # Try to load checkpoints for this seed
        mnist_candidates = [
            ext2_ckpt_dir / f"mnist_dense_full_com_seed{seed}.pt",
            ext2_ckpt_dir / f"mnist_dense_full_seed{seed}.pt",
            vision_ckpt_dir / f"mnist_dense_full_seed{seed}.pt",
        ]
        digits_candidates = [
            ext2_ckpt_dir / f"emnist_digits_regularized_seed{seed}.pt",
        ]
        letters_candidates = [
            ext2_ckpt_dir / f"emnist_letters_regularized_seed{seed}.pt",
        ]
        
        mnist_ckpt = next((p for p in mnist_candidates if p.exists()), None)
        digits_ckpt = next((p for p in digits_candidates if p.exists()), None)
        letters_ckpt = next((p for p in letters_candidates if p.exists()), None)
        
        if mnist_ckpt is None or digits_ckpt is None or letters_ckpt is None:
            print(f"  Seed {seed}: Missing checkpoints, skipping...")
            continue
        
        try:
            mnist_vals_seed, mnist_vecs_seed = load_checkpoint_eigenvalues(str(mnist_ckpt))
            digits_vals_seed, digits_vecs_seed = load_checkpoint_eigenvalues(str(digits_ckpt))
            letters_vals_seed, letters_vecs_seed = load_checkpoint_eigenvalues(str(letters_ckpt))
            
            # Get balanced eigenvectors
            mnist_0_vecs, mnist_0_vals = get_balanced_eigenvectors_and_vals(mnist_vals_seed, mnist_vecs_seed, 0, k)
            digits_0_vecs, _ = get_balanced_eigenvectors_and_vals(digits_vals_seed, digits_vecs_seed, 0, k)
            letters_o_vecs, _ = get_balanced_eigenvectors_and_vals(letters_vals_seed, letters_vecs_seed, 14, k)
            
            # Compute similarities
            sim_mnist_emnist_digit = abs_cosine_similarity(mnist_0_vecs, digits_0_vecs)
            sim_mnist_emnist_letter = abs_cosine_similarity(mnist_0_vecs, letters_o_vecs)
            
            sim_mnist_emnist_digit_by_seed.append(sim_mnist_emnist_digit)
            sim_mnist_emnist_letter_by_seed.append(sim_mnist_emnist_letter)
            loaded_seeds.append(seed)
            
        except Exception as e:
            print(f"  Seed {seed}: Error loading checkpoint: {e}")
            continue
    
    if len(loaded_seeds) == 0:
        print("  ⚠️  No multi-seed checkpoints found, using single checkpoint...")
        # Fallback to single seed
        if digits_vecs is None or digits_vals is None:
            print("  Skipping: EMNIST Digits checkpoint not available")
            return
        
        mnist_0_vecs, mnist_0_vals = get_balanced_eigenvectors_and_vals(mnist_vals, mnist_vecs, 0, k)
        digits_0_vecs, _ = get_balanced_eigenvectors_and_vals(digits_vals, digits_vecs, 0, k)
        letters_o_vecs, _ = get_balanced_eigenvectors_and_vals(letters_vals, letters_vecs, 14, k)
        
        sim_mnist_emnist_digit = abs_cosine_similarity(mnist_0_vecs, digits_0_vecs)
        sim_mnist_emnist_letter = abs_cosine_similarity(mnist_0_vecs, letters_o_vecs)
        
        sim_mnist_emnist_digit_mean = sim_mnist_emnist_digit
        sim_mnist_emnist_letter_mean = sim_mnist_emnist_letter
        loaded_seeds = [42]  # Mark as single seed for caption
    else:
        print(f"  Loaded {len(loaded_seeds)} seeds: {loaded_seeds}")
        # Compute mean across seeds
        sim_mnist_emnist_digit_stack = torch.stack(sim_mnist_emnist_digit_by_seed)  # [n_seeds, k, k]
        sim_mnist_emnist_letter_stack = torch.stack(sim_mnist_emnist_letter_by_seed)  # [n_seeds, k, k]
        
        sim_mnist_emnist_digit_mean = sim_mnist_emnist_digit_stack.mean(dim=0)  # [k, k]
        sim_mnist_emnist_letter_mean = sim_mnist_emnist_letter_stack.mean(dim=0)  # [k, k]
        
        # Get eigenvalue signs from first seed (seed 42)
        mnist_0_vecs, mnist_0_vals = get_balanced_eigenvectors_and_vals(mnist_vals, mnist_vecs, 0, k)
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    caption = f"Absolute Cosine Similarity between Top 10 Eigenvectors (Balanced: 5 Positive + 5 Negative)"
    if len(loaded_seeds) > 1:
        caption += f", Mean over {len(loaded_seeds)} seeds"
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append("\\label{tab:eigenvector_comparison}")
    latex_lines.append("\\begin{tabular}{cc|cc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Eigenvector & Eigenvalue & \\multicolumn{2}{c}{Maximum Absolute Cosine Similarity} \\\\")
    latex_lines.append("\\cmidrule(lr){3-4}")
    latex_lines.append("Rank & Sign & MNIST-0 vs & MNIST-0 vs \\\\")
    latex_lines.append(" & & EMNIST-0 & EMNIST-O \\\\")
    latex_lines.append("\\midrule")
    
    # For each eigenvector rank, compute mean and std of max similarity across seeds
    max_sims_digit_by_rank = []  # List of lists: [rank][seed] = max similarity
    max_sims_letter_by_rank = []
    
    if len(loaded_seeds) > 1:
        for rank in range(k):
            max_digit_per_seed = [sim[rank].max().item() for sim in sim_mnist_emnist_digit_by_seed]
            max_letter_per_seed = [sim[rank].max().item() for sim in sim_mnist_emnist_letter_by_seed]
            max_sims_digit_by_rank.append(max_digit_per_seed)
            max_sims_letter_by_rank.append(max_letter_per_seed)
    
    # For each eigenvector rank, find the best match in the other set
    for rank in range(k):
        max_sim_mnist_emnist_digit_mean = sim_mnist_emnist_digit_mean[rank].max().item()
        max_sim_mnist_emnist_letter_mean = sim_mnist_emnist_letter_mean[rank].max().item()
        
        # Get eigenvalue sign for MNIST (as reference)
        val_sign = "+" if mnist_0_vals[rank].item() > 0 else "-"
        
        if len(loaded_seeds) > 1:
            # Compute std across seeds for this rank's max similarity
            max_sim_mnist_emnist_digit_std = np.std(max_sims_digit_by_rank[rank], ddof=1) if len(max_sims_digit_by_rank[rank]) > 1 else 0.0
            max_sim_mnist_emnist_letter_std = np.std(max_sims_letter_by_rank[rank], ddof=1) if len(max_sims_letter_by_rank[rank]) > 1 else 0.0
            
            latex_lines.append(
                f"{rank+1} & {val_sign} & ${max_sim_mnist_emnist_digit_mean:.3f} \\pm {max_sim_mnist_emnist_digit_std:.3f}$ & "
                f"${max_sim_mnist_emnist_letter_mean:.3f} \\pm {max_sim_mnist_emnist_letter_std:.3f}$ \\\\"
            )
        else:
            latex_lines.append(
                f"{rank+1} & {val_sign} & {max_sim_mnist_emnist_digit_mean:.3f} & "
                f"{max_sim_mnist_emnist_letter_mean:.3f} \\\\"
            )
    
    # Add mean row with std
    mean_mnist_emnist_digit = sim_mnist_emnist_digit_mean.max(dim=1)[0].mean().item()
    mean_mnist_emnist_letter = sim_mnist_emnist_letter_mean.max(dim=1)[0].mean().item()
    
    # Compute std of the max similarities across ranks
    if len(loaded_seeds) > 1:
        # For each seed, get the max similarity per rank, then compute std across seeds
        max_sims_digit_per_seed = [sim.max(dim=1)[0].mean().item() for sim in sim_mnist_emnist_digit_by_seed]
        max_sims_letter_per_seed = [sim.max(dim=1)[0].mean().item() for sim in sim_mnist_emnist_letter_by_seed]
        
        std_mnist_emnist_digit = np.std(max_sims_digit_per_seed, ddof=1) if len(max_sims_digit_per_seed) > 1 else 0.0
        std_mnist_emnist_letter = np.std(max_sims_letter_per_seed, ddof=1) if len(max_sims_letter_per_seed) > 1 else 0.0
    else:
        std_mnist_emnist_digit = 0.0
        std_mnist_emnist_letter = 0.0
    
    latex_lines.append("\\midrule")
    if len(loaded_seeds) > 1:
        latex_lines.append(
            f"\\textbf{{Mean}} & & \\textbf{{{mean_mnist_emnist_digit:.3f} $\\pm$ {std_mnist_emnist_digit:.3f}}} & "
            f"\\textbf{{{mean_mnist_emnist_letter:.3f} $\\pm$ {std_mnist_emnist_letter:.3f}}} \\\\"
        )
    else:
        mean_mnist_emnist_digit_str = f"{mean_mnist_emnist_digit:.3f}"
        mean_mnist_emnist_letter_str = f"{mean_mnist_emnist_letter:.3f}"
        latex_lines.append(
            f"\\textbf{{Mean}} & & \\textbf{{{mean_mnist_emnist_digit_str}}} & "
            f"\\textbf{{{mean_mnist_emnist_letter_str}}} \\\\"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    # Save to file
    table_path = d.figure_out / "eigenvector_comparison_table.tex"
    # Save to Report directory (same level as figures)
    report_table_path = PROJECT_ROOT / "Report" / "eigenvector_comparison_table.tex"
    
    _ensure_dir(table_path.parent)
    _ensure_dir(report_table_path.parent)
    
    with open(table_path, 'w') as f:
        f.write(latex_table)
    
    with open(report_table_path, 'w') as f:
        f.write(latex_table)
    
    print(f"  Saved: {table_path.name}")
    print(f"  Saved: {report_table_path.name}")
    
    # Print summary statistics
    print(f"\n  Summary Statistics (mean over {len(loaded_seeds)} seeds):")
    if len(loaded_seeds) > 1:
        print(f"    MNIST-0 vs EMNIST-0: mean = {mean_mnist_emnist_digit:.3f} ± {std_mnist_emnist_digit:.3f}")
        print(f"    MNIST-0 vs EMNIST-O: mean = {mean_mnist_emnist_letter:.3f} ± {std_mnist_emnist_letter:.3f}")
    else:
        print(f"    MNIST-0 vs EMNIST-0: mean = {mean_mnist_emnist_digit:.3f}")
        print(f"    MNIST-0 vs EMNIST-O: mean = {mean_mnist_emnist_letter:.3f}")


def generate_eigenvector_comparison_table_0_O_X(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 10,
) -> None:
    """Generate LaTeX table for MNIST-0 vs EMNIST-O and MNIST-0 vs EMNIST-X.

    This corresponds to the table used in the report to show cosine similarity failing
    to distinguish similar (0–O) from dissimilar (0–X) pairs.
    """
    print("\n=== Eigenvector Comparison Table (MNIST-0 vs O/X, Balanced, Mean over 5 seeds) ===")

    from src.paths import EXTENSION2_CHECKPOINTS, MNIST_CHECKPOINTS

    seeds = [42, 43, 44, 45, 46]
    ext2_ckpt_dir = EXTENSION2_CHECKPOINTS
    vision_ckpt_dir = MNIST_CHECKPOINTS

    mnist_class = 0
    letter_o_idx = 14
    letter_x_idx = 23

    def get_balanced_eigenvectors_and_vals(vals, vecs, class_idx, k_local):
        class_vals = vals[class_idx].cpu()
        class_vecs = vecs[class_idx].cpu()

        pos_idx = torch.nonzero(class_vals > 0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(class_vals < 0, as_tuple=False).squeeze(-1)

        pos_sorted = pos_idx[class_vals[pos_idx].abs().argsort(descending=True)] if pos_idx.numel() > 0 else pos_idx
        neg_sorted = neg_idx[class_vals[neg_idx].abs().argsort(descending=True)] if neg_idx.numel() > 0 else neg_idx

        k_pos = k_local // 2
        k_neg = k_local - k_pos

        selected_idx = []
        if k_pos > 0 and pos_sorted.numel() > 0:
            selected_idx.extend(pos_sorted[:k_pos].tolist())
        if k_neg > 0 and neg_sorted.numel() > 0:
            selected_idx.extend(neg_sorted[:k_neg].tolist())

        if len(selected_idx) < k_local:
            selected_mask = torch.zeros_like(class_vals, dtype=torch.bool)
            if len(selected_idx) > 0:
                selected_mask[selected_idx] = True
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            if remaining.numel() > 0:
                remaining_sorted = remaining[class_vals[remaining].abs().argsort(descending=True)]
                needed = k_local - len(selected_idx)
                selected_idx.extend(remaining_sorted[:needed].tolist())

        return class_vecs[selected_idx], class_vals[selected_idx]

    def abs_cosine_similarity(vecs_A, vecs_B):
        vecs_A_norm = vecs_A / (vecs_A.norm(dim=1, keepdim=True) + 1e-10)
        vecs_B_norm = vecs_B / (vecs_B.norm(dim=1, keepdim=True) + 1e-10)
        cos_matrix = vecs_A_norm @ vecs_B_norm.T
        return cos_matrix.abs()

    sim_mnist_o_by_seed = []
    sim_mnist_x_by_seed = []
    loaded_seeds = []

    for seed in seeds:
        mnist_candidates = [
            ext2_ckpt_dir / f"mnist_dense_full_com_seed{seed}.pt",
            ext2_ckpt_dir / f"mnist_dense_full_seed{seed}.pt",
            vision_ckpt_dir / f"mnist_dense_full_seed{seed}.pt",
        ]
        letters_candidates = [
            ext2_ckpt_dir / f"emnist_letters_regularized_seed{seed}.pt",
        ]

        mnist_ckpt = next((p for p in mnist_candidates if p.exists()), None)
        letters_ckpt = next((p for p in letters_candidates if p.exists()), None)

        if mnist_ckpt is None or letters_ckpt is None:
            print(f"  Seed {seed}: Missing checkpoints, skipping...")
            continue

        try:
            mnist_vals_seed, mnist_vecs_seed = load_checkpoint_eigenvalues(str(mnist_ckpt))
            letters_vals_seed, letters_vecs_seed = load_checkpoint_eigenvalues(str(letters_ckpt))

            mnist_0_vecs, mnist_0_vals = get_balanced_eigenvectors_and_vals(mnist_vals_seed, mnist_vecs_seed, mnist_class, k)
            letters_o_vecs, _ = get_balanced_eigenvectors_and_vals(letters_vals_seed, letters_vecs_seed, letter_o_idx, k)
            letters_x_vecs, _ = get_balanced_eigenvectors_and_vals(letters_vals_seed, letters_vecs_seed, letter_x_idx, k)

            sim_mnist_o = abs_cosine_similarity(mnist_0_vecs, letters_o_vecs)
            sim_mnist_x = abs_cosine_similarity(mnist_0_vecs, letters_x_vecs)

            sim_mnist_o_by_seed.append(sim_mnist_o)
            sim_mnist_x_by_seed.append(sim_mnist_x)
            loaded_seeds.append(seed)
        except Exception as e:
            print(f"  Seed {seed}: Error loading checkpoint: {e}")
            continue

    if len(loaded_seeds) == 0:
        print("  ⚠️  No multi-seed checkpoints found, aborting 0/O/X table generation.")
        return

    print(f"  Loaded {len(loaded_seeds)} seeds: {loaded_seeds}")

    sim_mnist_o_stack = torch.stack(sim_mnist_o_by_seed)  # [n_seeds, k, k]
    sim_mnist_x_stack = torch.stack(sim_mnist_x_by_seed)  # [n_seeds, k, k]

    sim_mnist_o_mean = sim_mnist_o_stack.mean(dim=0)
    sim_mnist_x_mean = sim_mnist_x_stack.mean(dim=0)

    # Eigenvalue signs from reference MNIST eigenvalues
    mnist_ckpt_ref = ext2_ckpt_dir / f"mnist_dense_full_com_seed{loaded_seeds[0]}.pt"
    mnist_vals_ref, mnist_vecs_ref = load_checkpoint_eigenvalues(str(mnist_ckpt_ref))
    _, mnist_0_vals_ref = get_balanced_eigenvectors_and_vals(mnist_vals_ref, mnist_vecs_ref, mnist_class, k)

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(
        f"\\caption{{Absolute Cosine Similarity between Top 10 Eigenvectors (Balanced: 5 Positive + 5 Negative), Mean over {len(loaded_seeds)} seeds}}"
    )
    latex_lines.append("\\label{tab:eigenvector_comparison_0_O_X}")
    latex_lines.append("\\begin{tabular}{cc|cc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Eigenvector & Eigenvalue & \\multicolumn{2}{c}{Maximum Absolute Cosine Similarity} \\\\")
    latex_lines.append("\\cmidrule(lr){3-4}")
    latex_lines.append("Rank & Sign & MNIST-0 vs & MNIST-0 vs \\\\")
    latex_lines.append(" & & EMNIST-O & EMNIST-X \\\\")
    latex_lines.append("\\midrule")

    # Per-rank std across seeds
    max_sims_o_by_rank = []
    max_sims_x_by_rank = []
    for rank in range(k):
        max_o_per_seed = [sim[rank].max().item() for sim in sim_mnist_o_by_seed]
        max_x_per_seed = [sim[rank].max().item() for sim in sim_mnist_x_by_seed]
        max_sims_o_by_rank.append(max_o_per_seed)
        max_sims_x_by_rank.append(max_x_per_seed)

    for rank in range(k):
        max_o_mean = sim_mnist_o_mean[rank].max().item()
        max_x_mean = sim_mnist_x_mean[rank].max().item()

        val_sign = "+" if mnist_0_vals_ref[rank].item() > 0 else "-"

        max_o_std = np.std(max_sims_o_by_rank[rank], ddof=1) if len(max_sims_o_by_rank[rank]) > 1 else 0.0
        max_x_std = np.std(max_sims_x_by_rank[rank], ddof=1) if len(max_sims_x_by_rank[rank]) > 1 else 0.0

        latex_lines.append(
            f"{rank+1} & {val_sign} & "
            f"${max_o_mean:.3f} \\pm {max_o_std:.3f}$ & "
            f"${max_x_mean:.3f} \\pm {max_x_std:.3f}$ \\\\"
        )

    # Mean row
    mean_o = sim_mnist_o_mean.max(dim=1)[0].mean().item()
    mean_x = sim_mnist_x_mean.max(dim=1)[0].mean().item()

    max_sims_o_per_seed = [sim.max(dim=1)[0].mean().item() for sim in sim_mnist_o_by_seed]
    max_sims_x_per_seed = [sim.max(dim=1)[0].mean().item() for sim in sim_mnist_x_by_seed]

    std_o = np.std(max_sims_o_per_seed, ddof=1) if len(max_sims_o_per_seed) > 1 else 0.0
    std_x = np.std(max_sims_x_per_seed, ddof=1) if len(max_sims_x_per_seed) > 1 else 0.0

    latex_lines.append("\\midrule")
    latex_lines.append(
        f"\\textbf{{Mean}} & & "
        f"\\textbf{{{mean_o:.3f} $\\pm$ {std_o:.3f}}} & "
        f"\\textbf{{{mean_x:.3f} $\\pm$ {std_x:.3f}}} \\\\"
    )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    report_table_path = PROJECT_ROOT / "Report" / "eigenvector_comparison_table_0_O_X.tex"
    _ensure_dir(report_table_path.parent)
    with open(report_table_path, "w") as f:
        f.write(latex_table)

    print(f"  Saved: {report_table_path.name}")


def generate_selection_method_comparison(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Compare magnitude-based vs balanced eigenvector selection methods."""
    print("\n=== Selection Method Comparison ===")
    
    k_values = list(range(2, 101))
    
    def get_top_k_by_magnitude(vals, vecs, k):
        """Select top-k by absolute eigenvalue magnitude."""
        n_classes = vals.shape[0]
        selected = []
        for c in range(n_classes):
            _, idx = vals[c].abs().sort(descending=True)
            selected.append(vecs[c, idx[:k]])
        return torch.stack(selected)
    
    # Expected shape-similar pairs: (MNIST digit idx, EMNIST letter idx)
    # Letter indices: I=8, O=14, S=18, Z=25
    expected_pairs = [(0, 14), (1, 8), (2, 25), (5, 18)]  # 0-O, 1-I, 2-Z, 5-S
    
    # Control pairs: CROSS-MODEL dissimilar (MNIST digit vs EMNIST letter with different shapes)
    # Must be cross-model since same-model classes share identical eigenvector subspace
    # Letter indices: H=7, O=14, W=22, X=23
    control_pairs = [(0, 23), (1, 22), (3, 7), (7, 14)]  # 0-X, 1-W, 3-H, 7-O
    
    mag_expected = []
    bal_expected = []
    mag_control = []
    bal_control = []
    
    for k in k_values:
        # Magnitude-based selection
        mnist_mag = get_top_k_by_magnitude(mnist_vals, mnist_vecs, k)
        letters_mag = get_top_k_by_magnitude(letters_vals, letters_vecs, k)
        
        # Expected pairs (magnitude) - MNIST digit vs EMNIST letter (similar shapes)
        mag_exp_overlaps = []
        for d_idx, l_idx in expected_pairs:
            overlap = compute_subspace_overlap(mnist_mag[d_idx], letters_mag[l_idx], k=k, method='mean_cos')
            mag_exp_overlaps.append(overlap)
        mag_expected.append(np.mean(mag_exp_overlaps))
        
        # Control pairs (magnitude) - MNIST digit vs EMNIST letter (dissimilar shapes)
        mag_ctrl_overlaps = []
        for d_idx, l_idx in control_pairs:
            overlap = compute_subspace_overlap(mnist_mag[d_idx], letters_mag[l_idx], k=k, method='mean_cos')
            mag_ctrl_overlaps.append(overlap)
        mag_control.append(np.mean(mag_ctrl_overlaps))
        
        # Balanced selection
        mnist_bal = select_balanced_eigenvectors(mnist_vals, mnist_vecs, k=k)
        letters_bal = select_balanced_eigenvectors(letters_vals, letters_vecs, k=k)
        
        # Expected pairs (balanced) - MNIST digit vs EMNIST letter (similar shapes)
        bal_exp_overlaps = []
        for d_idx, l_idx in expected_pairs:
            overlap = compute_subspace_overlap(mnist_bal[d_idx], letters_bal[l_idx], k=k, method='mean_cos')
            bal_exp_overlaps.append(overlap)
        bal_expected.append(np.mean(bal_exp_overlaps))
        
        # Control pairs (balanced) - MNIST digit vs EMNIST letter (dissimilar shapes)
        bal_ctrl_overlaps = []
        for d_idx, l_idx in control_pairs:
            overlap = compute_subspace_overlap(mnist_bal[d_idx], letters_bal[l_idx], k=k, method='mean_cos')
            bal_ctrl_overlaps.append(overlap)
        bal_control.append(np.mean(bal_ctrl_overlaps))
    
    # Find best k (max gap between expected and control)
    mag_gaps = np.array(mag_expected) - np.array(mag_control)
    bal_gaps = np.array(bal_expected) - np.array(bal_control)
    best_mag_k = k_values[np.argmax(mag_gaps)]
    best_bal_k = k_values[np.argmax(bal_gaps)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Expected pairs (solid lines) - MNIST digit vs EMNIST letter (shape-similar)
    ax.plot(k_values, mag_expected, 'b-', linewidth=2, label=f'Magnitude - Similar (0-O, 1-I, ...)')
    ax.plot(k_values, bal_expected, 'r-', linewidth=2, label=f'Balanced - Similar (0-O, 1-I, ...)')
    
    # Control pairs (dashed lines) - MNIST digit vs EMNIST letter (dissimilar shapes)
    ax.plot(k_values, mag_control, 'b--', linewidth=1.5, alpha=0.6, label=f'Magnitude - Dissimilar (0-X, 1-W, ...)')
    ax.plot(k_values, bal_control, 'r--', linewidth=1.5, alpha=0.6, label=f'Balanced - Dissimilar (0-X, 1-W, ...)')
    
    ax.set_xlabel('k (number of eigenvectors)', fontsize=12)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax.set_title(f'Selection Method Comparison\n(Best discrimination: Magnitude k={best_mag_k}, Balanced k={best_bal_k})', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_selection_comparison.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_principal_angles(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate principal angles plot."""
    print("\n=== Principal Angles ===")
    
    fig = plot_principal_angles(
        mnist_vecs, letters_vecs,
        mnist_vals, letters_vals,
        pairs=DIGIT_LETTER_PAIRS,
        k=20,
    )
    
    out_path = d.figure_out / "extension2_principal_angles.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


# =============================================================================
# EIGENVALUE-AWARE METRICS: Similarity vs k plots and heatmaps
# =============================================================================


def _compute_similarity_vs_k_for_metric(
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    method: str,
    k_values: List[int],
) -> Tuple[Dict[str, List[float]], List[float], List[float]]:
    """
    Compute similarity vs k for a given metric.
    
    Returns:
        similar_overlaps: Dict mapping pair label to list of overlaps per k
        control_means: List of mean control overlaps per k
        control_stds: List of std of control overlaps per k
    """
    similar_overlaps = {}
    
    # Control pairs: dissimilar cross-model pairs
    control_pairs = [
        (0, 23, "0-X"),
        (1, 22, "1-W"),
        (3, 7, "3-H"),
        (7, 14, "7-O"),
    ]
    
    # Compute for expected similar pairs
    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
        overlaps = []
        for k in k_values:
            overlap = compute_weighted_similarity(
                mnist_vecs[digit_idx].cpu(),
                letters_vecs[letter_idx].cpu(),
                mnist_vals[digit_idx].cpu(),
                letters_vals[letter_idx].cpu(),
                k=k,
                method=method,
            )
            overlaps.append(overlap)
        similar_overlaps[label] = overlaps
    
    # Compute control baseline
    control_overlaps_by_k = {k: [] for k in k_values}
    for d_idx, l_idx, _ in control_pairs:
        for k in k_values:
            overlap = compute_weighted_similarity(
                mnist_vecs[d_idx].cpu(),
                letters_vecs[l_idx].cpu(),
                mnist_vals[d_idx].cpu(),
                letters_vals[l_idx].cpu(),
                k=k,
                method=method,
            )
            control_overlaps_by_k[k].append(overlap)
    
    control_means = [np.mean(control_overlaps_by_k[k]) for k in k_values]
    control_stds = [np.std(control_overlaps_by_k[k]) for k in k_values]
    
    return similar_overlaps, control_means, control_stds


def generate_similarity_vs_k_weighted(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate similarity vs k plots for all three weighted metrics.
    
    For quadratic_form, loads all 5 seeds and computes 90% confidence intervals.
    """
    print("\n=== Similarity vs k (Weighted Metrics) ===")
    
    k_values = list(range(2, 101))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    metrics = [
        ('eigenvalue_weighted', 'Eigenvalue-Weighted Cosine Similarity', (0, 1)),
        ('quadratic_form', 'Quadratic Form Similarity', (-0.5, 1)),
    ]
    
    for method, title, ylim in metrics:
        print(f"  Computing {method}...")
        
        # For quadratic_form, load multiple seeds and compute CI
        if method == 'quadratic_form':
            # Load checkpoints for all 5 seeds
            seeds = [42, 43, 44, 45, 46]
            ext2_ckpt_dir = EXTENSION2_CHECKPOINTS
            
            similar_overlaps_by_seed = {label: [] for _, _, label in DIGIT_LETTER_PAIRS}
            control_overlaps_by_seed = []
            loaded_seeds = []
            
            for seed in seeds:
                # Try different checkpoint naming patterns
                mnist_candidates = [
                    ext2_ckpt_dir / f"mnist_dense_full_com_seed{seed}.pt",
                    ext2_ckpt_dir / f"mnist_dense_full_seed{seed}.pt",
                ]
                letters_candidates = [
                    ext2_ckpt_dir / f"emnist_letters_regularized_seed{seed}.pt",
                ]
                
                mnist_ckpt = next((p for p in mnist_candidates if p.exists()), None)
                letters_ckpt = next((p for p in letters_candidates if p.exists()), None)
                
                if mnist_ckpt is None or letters_ckpt is None:
                    print(f"    Seed {seed}: Missing checkpoints, skipping...")
                    continue
                
                try:
                    mnist_vals_seed, mnist_vecs_seed = load_checkpoint_eigenvalues(str(mnist_ckpt))
                    letters_vals_seed, letters_vecs_seed = load_checkpoint_eigenvalues(str(letters_ckpt))
                    
                    # Compute similarity vs k for this seed
                    seed_similar_overlaps = {}
                    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
                        overlaps = []
                        for k in k_values:
                            overlap = compute_weighted_similarity(
                                mnist_vecs_seed[digit_idx].cpu(),
                                letters_vecs_seed[letter_idx].cpu(),
                                mnist_vals_seed[digit_idx].cpu(),
                                letters_vals_seed[letter_idx].cpu(),
                                k=k,
                                method=method,
                            )
                            overlaps.append(overlap)
                        seed_similar_overlaps[label] = overlaps
                        similar_overlaps_by_seed[label].append(overlaps)
                    
                    # Control pairs
                    control_pairs = [(0, 23), (1, 22), (3, 7), (7, 14)]
                    seed_control_overlaps_by_k = {k: [] for k in k_values}
                    for d_idx, l_idx in control_pairs:
                        for k in k_values:
                            overlap = compute_weighted_similarity(
                                mnist_vecs_seed[d_idx].cpu(),
                                letters_vecs_seed[l_idx].cpu(),
                                mnist_vals_seed[d_idx].cpu(),
                                letters_vals_seed[l_idx].cpu(),
                                k=k,
                                method=method,
                            )
                            seed_control_overlaps_by_k[k].append(overlap)
                    # Store as list of lists: [mean_over_pairs for each k]
                    seed_control_means = [np.mean(seed_control_overlaps_by_k[k]) for k in k_values]
                    control_overlaps_by_seed.append(seed_control_means)
                    loaded_seeds.append(seed)
                except Exception as e:
                    print(f"    Seed {seed}: Error loading checkpoint: {e}")
                    continue
            
            if len(loaded_seeds) == 0:
                print("    ⚠️  No multi-seed checkpoints found, using single checkpoint...")
                similar_overlaps, control_means, control_stds = _compute_similarity_vs_k_for_metric(
                    mnist_vecs, mnist_vals, letters_vecs, letters_vals, method, k_values
                )
                use_ci = False
            else:
                print(f"    Loaded {len(loaded_seeds)} seeds: {loaded_seeds}")
                # Compute mean and CI across seeds
                similar_overlaps = {}
                similar_overlaps_ci_low = {}
                similar_overlaps_ci_high = {}
                
                for label in similar_overlaps_by_seed:
                    overlaps_stack = np.array(similar_overlaps_by_seed[label])  # [n_seeds, n_k]
                    overlaps_mean = np.mean(overlaps_stack, axis=0)
                    overlaps_std = np.std(overlaps_stack, axis=0, ddof=1)
                    sem = overlaps_std / np.sqrt(len(loaded_seeds))
                    t_val = stats.t.ppf(0.95, len(loaded_seeds) - 1)  # 90% CI
                    
                    similar_overlaps[label] = overlaps_mean.tolist()
                    similar_overlaps_ci_low[label] = (overlaps_mean - t_val * sem).tolist()
                    similar_overlaps_ci_high[label] = (overlaps_mean + t_val * sem).tolist()
                
                # Control baseline
                control_stack = np.array(control_overlaps_by_seed)  # [n_seeds, n_k]
                control_mean_by_k = np.mean(control_stack, axis=0)  # Mean across seeds
                control_std_by_k = np.std(control_stack, axis=0, ddof=1)  # Std across seeds
                control_sem = control_std_by_k / np.sqrt(len(loaded_seeds))
                t_val = stats.t.ppf(0.95, len(loaded_seeds) - 1)
                
                control_means = control_mean_by_k.tolist()
                control_ci_low = (control_mean_by_k - t_val * control_sem).tolist()
                control_ci_high = (control_mean_by_k + t_val * control_sem).tolist()
                control_stds = control_std_by_k.tolist()
                use_ci = True
        else:
            # For other metrics, use single checkpoint
            similar_overlaps, control_means, control_stds = _compute_similarity_vs_k_for_metric(
                mnist_vecs, mnist_vals, letters_vecs, letters_vals, method, k_values
            )
            use_ci = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot similar pairs
        for idx, (digit_idx, letter_idx, label) in enumerate(DIGIT_LETTER_PAIRS):
            ax.plot(k_values, similar_overlaps[label], 
                   label=f'{label} (similar)', color=colors[idx], linewidth=2)
            
            # Add CI bands for quadratic_form
            if use_ci and method == 'quadratic_form':
                ax.fill_between(k_values,
                               similar_overlaps_ci_low[label],
                               similar_overlaps_ci_high[label],
                               color=colors[idx], alpha=0.2)
        
        # Plot control baseline
        ax.plot(k_values, control_means, 'k--', 
               label='Dissimilar (0-X, 1-W, 3-H, 7-O)', linewidth=1.5, alpha=0.7)
        
        if use_ci and method == 'quadratic_form':
            ax.fill_between(k_values, control_ci_low, control_ci_high,
                           color='gray', alpha=0.2)
        else:
            ax.fill_between(k_values,
                           np.array(control_means) - np.array(control_stds),
                           np.array(control_means) + np.array(control_stds),
                           color='gray', alpha=0.2)
        
        ax.set_xlabel('k (number of eigenvectors)', fontsize=12)
        ax.set_ylabel('Similarity', fontsize=12)
        plot_title = f'{title} vs Number of Eigenvectors'
        if use_ci and method == 'quadratic_form':
            plot_title += f'\nMean across {len(loaded_seeds)} seeds (90% CI)'
        ax.set_title(plot_title, fontsize=14)
        ax.legend(loc='best')
        ax.set_xlim(0, 100)
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        out_path = d.figure_out / f"extension2_similarity_vs_k_{method}.pdf"
        _save_figure(fig, out_path)
        plt.close(fig)


def _compute_similarity_heatmap_for_metric(
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    method: str,
    k: int,
) -> np.ndarray:
    """Compute 10x26 similarity matrix for a given metric."""
    matrix = np.zeros((10, 26))
    for digit in range(10):
        for letter in range(26):
            matrix[digit, letter] = compute_weighted_similarity(
                mnist_vecs[digit].cpu(),
                letters_vecs[letter].cpu(),
                mnist_vals[digit].cpu(),
                letters_vals[letter].cpu(),
                k=k,
                method=method,
            )
    return matrix


def generate_similarity_heatmap_weighted(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate heatmaps for all three weighted metrics."""
    print("\n=== Similarity Heatmaps (Weighted Metrics) ===")
    
    metrics = [
        ('eigenvalue_weighted', 'Eigenvalue-Weighted Cosine', (0.0, 0.3)),
        ('quadratic_form', 'Quadratic Form', (-0.2, 0.4)),
    ]
    
    for method, title, vrange in metrics:
        print(f"  Computing {method} heatmap...")
        
        matrix = _compute_similarity_heatmap_for_metric(
            mnist_vecs, mnist_vals, letters_vecs, letters_vals, method, k
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        letter_labels = [chr(65+i) for i in range(26)]
        im = ax.imshow(matrix, cmap='RdBu_r' if method == 'quadratic_form' else 'YlOrRd',
                      aspect='auto', vmin=vrange[0], vmax=vrange[1])
        
        # Add text annotations
        mid_val = (vrange[0] + vrange[1]) / 2
        for digit in range(10):
            for letter in range(26):
                color = 'white' if matrix[digit, letter] > mid_val + 0.1 * (vrange[1] - vrange[0]) else 'black'
                ax.text(letter, digit, f'{matrix[digit, letter]:.2f}',
                       ha='center', va='center', fontsize=6, color=color)
        
        ax.set_xticks(range(26))
        ax.set_xticklabels(letter_labels, fontsize=10)
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(10), fontsize=10)
        ax.set_xlabel('EMNIST Letter', fontsize=12)
        ax.set_ylabel('MNIST Digit', fontsize=12)
        ax.set_title(f'{title} Similarity (k={k})', fontsize=14)
        
        # Mark expected similar pairs with blue boxes
        for digit_idx, letter_idx, _ in DIGIT_LETTER_PAIRS:
            rect = plt.Rectangle((letter_idx-0.5, digit_idx-0.5), 1, 1,
                                 fill=False, edgecolor='blue', linewidth=3)
            ax.add_patch(rect)
        
        plt.colorbar(im, ax=ax, label='Similarity')
        plt.tight_layout()
        
        out_path = d.figure_out / f"extension2_heatmap_{method}.pdf"
        _save_figure(fig, out_path)
        plt.close(fig)
        
        # Print statistics
        similar_vals = [matrix[dig, let] for dig, let, _ in DIGIT_LETTER_PAIRS]
        dissimilar_vals = []
        for dig in range(10):
            for let in range(26):
                if (dig, let, None) not in DIGIT_LETTER_PAIRS:
                    dissimilar_vals.append(matrix[dig, let])
        
        print(f"    Similar pairs: {np.mean(similar_vals):.4f} ± {np.std(similar_vals):.4f}")
        print(f"    Dissimilar pairs: {np.mean(dissimilar_vals):.4f} ± {np.std(dissimilar_vals):.4f}")


def generate_cosine_similarity_heatmap_abs(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate absolute cosine similarity heatmap (MNIST digits vs EMNIST letters).

    This is the cosine-based counterpart to the quadratic-form heatmap, used in the report
    to show that cosine similarity fails to separate similar from dissimilar pairs.
    """
    print("\n=== Absolute Cosine Similarity Heatmap (MNIST Digits vs EMNIST Letters) ===")

    def get_sorted_eigenvectors(vals, vecs, class_idx, k_local):
        class_vals = vals[class_idx].cpu()
        class_vecs = vecs[class_idx].cpu()
        _, sorted_idx = class_vals.abs().sort(descending=True)
        top_k_idx = sorted_idx[:k_local]
        return class_vecs[top_k_idx], class_vals[top_k_idx]

    def abs_cosine_similarity(vecs_A, vecs_B):
        vecs_A_norm = vecs_A / (vecs_A.norm(dim=1, keepdim=True) + 1e-10)
        vecs_B_norm = vecs_B / (vecs_B.norm(dim=1, keepdim=True) + 1e-10)
        cos_matrix = vecs_A_norm @ vecs_B_norm.T
        return cos_matrix.abs()

    def compute_mean_abs_cosine_similarity(vecs_A, vecs_B):
        sim_matrix = abs_cosine_similarity(vecs_A, vecs_B)
        max_sims_per_rank = sim_matrix.max(dim=1)[0]
        return max_sims_per_rank.mean().item()

    matrix = np.zeros((10, 26))
    for digit in range(10):
        d_vecs, _ = get_sorted_eigenvectors(mnist_vals, mnist_vecs, digit, k)
        for letter in range(26):
            l_vecs, _ = get_sorted_eigenvectors(letters_vals, letters_vecs, letter, k)
            matrix[digit, letter] = compute_mean_abs_cosine_similarity(d_vecs, l_vecs)

    print(f"  Matrix computed: {matrix.shape}")
    print(f"  Range: [{matrix.min():.3f}, {matrix.max():.3f}], mean={matrix.mean():.3f}")

    fig, ax = plt.subplots(figsize=(14, 6))

    letter_labels = [chr(65 + i) for i in range(26)]
    vmin, vmax = 0.2, 0.6
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)

    mid_val = (vmin + vmax) / 2
    for digit in range(10):
        for letter in range(26):
            color = "white" if matrix[digit, letter] > mid_val + 0.1 * (vmax - vmin) else "black"
            ax.text(
                letter,
                digit,
                f"{matrix[digit, letter]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    ax.set_xticks(range(26))
    ax.set_xticklabels(letter_labels, fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10), fontsize=10)
    ax.set_xlabel("EMNIST Letter", fontsize=12, fontweight="bold")
    ax.set_ylabel("MNIST Digit", fontsize=12, fontweight="bold")
    ax.set_title(f"Absolute Cosine Similarity (k={k})", fontsize=14, fontweight="bold")

    # Expected similar pairs: (0,O), (1,I), (2,Z), (5,S)
    expected_pairs = [
        (0, 14, "0-O"),  # O is index 14
        (1, 8, "1-I"),   # I is index 8
        (2, 25, "2-Z"),  # Z is index 25
        (5, 18, "5-S"),  # S is index 18
    ]
    for digit_idx, letter_idx, _ in expected_pairs:
        rect = plt.Rectangle((letter_idx - 0.5, digit_idx - 0.5), 1, 1,
                             fill=False, edgecolor="blue", linewidth=3)
        ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, label="Absolute Cosine Similarity", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    out_path = d.figure_out / "extension2_heatmap_abs_cosine_similarity.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)

    # Print summary statistics
    similar_vals = [matrix[d, l] for d, l, _ in expected_pairs]
    dissimilar_vals = []
    for d in range(10):
        for l in range(26):
            if (d, l) not in [(dp, lp) for dp, lp, _ in expected_pairs]:
                dissimilar_vals.append(matrix[d, l])

    print("  Statistics:")
    print(f"    Expected similar pairs: {np.mean(similar_vals):.4f} ± {np.std(similar_vals):.4f}")
    print(f"    Dissimilar pairs: {np.mean(dissimilar_vals):.4f} ± {np.std(dissimilar_vals):.4f}")
    print(f"    Gap: {np.mean(similar_vals) - np.mean(dissimilar_vals):.4f}")


def generate_quadratic_form_heatmap_digits(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    digits_vecs: torch.Tensor,
    digits_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate quadratic form similarity heatmap for MNIST digits vs EMNIST digits (10x10 matrix)."""
    print("\n=== Quadratic Form Heatmap: MNIST Digits vs EMNIST Digits ===")
    
    if digits_vecs is None or digits_vals is None:
        print("  ⚠️  EMNIST Digits checkpoint not found, skipping...")
        return
    
    # Compute 10x10 similarity matrix using quadratic_form metric
    matrix = np.zeros((10, 10))
    for mnist_digit in range(10):
        for emnist_digit in range(10):
            matrix[mnist_digit, emnist_digit] = compute_weighted_similarity(
                mnist_vecs[mnist_digit].cpu(),
                digits_vecs[emnist_digit].cpu(),
                mnist_vals[mnist_digit].cpu(),
                digits_vals[emnist_digit].cpu(),
                k=k,
                method='quadratic_form',
            )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use RdBu_r colormap (red-blue reversed) for quadratic form
    # Range typically [-0.2, 0.4] for quadratic form
    vmin, vmax = -0.2, 0.4
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add text annotations
    mid_val = (vmin + vmax) / 2
    for mnist_digit in range(10):
        for emnist_digit in range(10):
            # Color text based on value (white for high, black for low)
            color = 'white' if matrix[mnist_digit, emnist_digit] > mid_val + 0.1 * (vmax - vmin) else 'black'
            ax.text(emnist_digit, mnist_digit, f'{matrix[mnist_digit, emnist_digit]:.2f}',
                   ha='center', va='center', fontsize=9, color=color, weight='bold')
    
    # Set labels
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10), fontsize=11)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10), fontsize=11)
    ax.set_xlabel('EMNIST Digit', fontsize=13, fontweight='bold')
    ax.set_ylabel('MNIST Digit', fontsize=13, fontweight='bold')
    ax.set_title(f'Quadratic Form Similarity: MNIST ↔ EMNIST-Digits (k={k})', fontsize=14, fontweight='bold')
    
    # Mark diagonal (same digit) with blue boxes
    for digit in range(10):
        rect = plt.Rectangle((digit-0.5, digit-0.5), 1, 1,
                             fill=False, edgecolor='blue', linewidth=2.5)
        ax.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Quadratic Form Similarity', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save
    out_path = d.figure_out / "extension2_quadratic_form_heatmap_digits.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)
    
    # Print statistics
    diagonal_vals = [matrix[i, i] for i in range(10)]  # Same digit pairs
    off_diagonal_vals = []
    for i in range(10):
        for j in range(10):
            if i != j:
                off_diagonal_vals.append(matrix[i, j])
    
    print(f"  Diagonal (same digit): {np.mean(diagonal_vals):.4f} ± {np.std(diagonal_vals):.4f}")
    print(f"  Off-diagonal (different digits): {np.mean(off_diagonal_vals):.4f} ± {np.std(off_diagonal_vals):.4f}")
    print(f"  Ratio (diagonal/off-diagonal): {np.mean(diagonal_vals) / (np.mean(off_diagonal_vals) + 1e-10):.2f}x")


def generate_metric_comparison(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate 4-way comparison of all metrics (mean_cos + 3 weighted)."""
    print("\n=== Metric Comparison (4-way) ===")
    
    k_values = list(range(2, 51))  # Shorter range for clearer visualization
    
    all_metrics = [
        ('mean_cos', 'Mean Cosine (Principal Angles)'),
        ('eigenvalue_weighted', 'Eigenvalue-Weighted Cosine'),
        ('quadratic_form', 'Quadratic Form'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for ax_idx, (method, title) in enumerate(all_metrics):
        ax = axes[ax_idx]
        
        if method == 'mean_cos':
            # Use original compute_subspace_overlap
            similar_overlaps = {}
            for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
                overlaps = []
                _, d_sorted = mnist_vals[digit_idx].abs().sort(descending=True)
                _, l_sorted = letters_vals[letter_idx].abs().sort(descending=True)
                d_vecs = mnist_vecs[digit_idx][d_sorted].cpu()
                l_vecs = letters_vecs[letter_idx][l_sorted].cpu()
                
                for k in k_values:
                    overlap = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
                    overlaps.append(overlap)
                similar_overlaps[label] = overlaps
            
            # Control
            control_pairs = [(0, 23), (1, 22), (3, 7), (7, 14)]
            control_overlaps_by_k = {k: [] for k in k_values}
            for d_idx, l_idx in control_pairs:
                _, d_sorted = mnist_vals[d_idx].abs().sort(descending=True)
                _, l_sorted = letters_vals[l_idx].abs().sort(descending=True)
                d_vecs = mnist_vecs[d_idx][d_sorted].cpu()
                l_vecs = letters_vecs[l_idx][l_sorted].cpu()
                for k in k_values:
                    overlap = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
                    control_overlaps_by_k[k].append(overlap)
            control_means = [np.mean(control_overlaps_by_k[k]) for k in k_values]
            control_stds = [np.std(control_overlaps_by_k[k]) for k in k_values]
        else:
            similar_overlaps, control_means, control_stds = _compute_similarity_vs_k_for_metric(
                mnist_vecs, mnist_vals, letters_vecs, letters_vals, method, k_values
            )
        
        # Plot
        for idx, (digit_idx, letter_idx, label) in enumerate(DIGIT_LETTER_PAIRS):
            ax.plot(k_values, similar_overlaps[label], color=colors[idx], linewidth=2, label=label)
        
        ax.plot(k_values, control_means, 'k--', linewidth=1.5, alpha=0.7, label='Dissimilar')
        ax.fill_between(k_values,
                       np.array(control_means) - np.array(control_stds),
                       np.array(control_means) + np.array(control_stds),
                       color='gray', alpha=0.2)
        
        # Compute discrimination at k=10
        sim_at_k10 = np.mean([similar_overlaps[label][8] for _, _, label in DIGIT_LETTER_PAIRS])
        ctrl_at_k10 = control_means[8]
        gap = sim_at_k10 - ctrl_at_k10
        
        ax.set_xlabel('k (number of eigenvectors)', fontsize=11)
        ax.set_ylabel('Similarity', fontsize=11)
        ax.set_title(f'{title}\n(Gap@k=10: {gap:.3f})', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 50)
        if method == 'quadratic_form':
            ax.set_ylim(-0.3, 0.8)
        else:
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Similarity Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_metric_comparison_4way.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_ranking_analysis(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate ranking analysis: where expected pairs rank among all 26 letters."""
    print("\n=== Ranking Analysis ===")
    
    all_metrics = [
        ('mean_cos', 'Mean Cosine'),
        ('eigenvalue_weighted', 'Eigenvalue-Weighted'),
        ('quadratic_form', 'Quadratic Form'),
    ]
    
    # Compute matrices for all metrics
    matrices = {}
    for method, _ in all_metrics:
        if method == 'mean_cos':
            matrix = np.zeros((10, 26))
            for digit in range(10):
                _, d_sorted = mnist_vals[digit].abs().sort(descending=True)
                d_vecs = mnist_vecs[digit][d_sorted].cpu()
                for letter in range(26):
                    _, l_sorted = letters_vals[letter].abs().sort(descending=True)
                    l_vecs = letters_vecs[letter][l_sorted].cpu()
                    matrix[digit, letter] = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
        else:
            matrix = _compute_similarity_heatmap_for_metric(
                mnist_vecs, mnist_vals, letters_vecs, letters_vals, method, k
            )
        matrices[method] = matrix
    
    # Create ranking figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DIGIT_LETTER_PAIRS))
    width = 0.2
    
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for m_idx, (method, label) in enumerate(all_metrics):
        matrix = matrices[method]
        ranks = []
        for digit_idx, letter_idx, _ in DIGIT_LETTER_PAIRS:
            # Rank 1 = highest similarity
            rank = (matrix[digit_idx] >= matrix[digit_idx, letter_idx]).sum()
            ranks.append(rank)
        
        offset = (m_idx - 1.5) * width
        bars = ax.bar(x + offset, ranks, width, label=label, color=metric_colors[m_idx], alpha=0.8)
        
        # Add rank values on top of bars
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(ranks[i]), ha='center', va='bottom', fontsize=9)
    
    ax.axhline(5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Top 5 threshold')
    ax.axhline(13, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Random (13)')
    
    ax.set_xlabel('Digit-Letter Pair', fontsize=12)
    ax.set_ylabel('Rank of Expected Match (1=best)', fontsize=12)
    ax.set_title(f'Ranking Analysis: Where Expected Pairs Rank (k={k})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in DIGIT_LETTER_PAIRS], fontsize=11)
    ax.set_ylim(0, 28)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_ranking_analysis.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)
    
    # Print summary
    print("  Ranking Summary (lower is better):")
    for method, label in all_metrics:
        matrix = matrices[method]
        ranks = []
        for digit_idx, letter_idx, _ in DIGIT_LETTER_PAIRS:
            rank = (matrix[digit_idx] >= matrix[digit_idx, letter_idx]).sum()
            ranks.append(rank)
        print(f"    {label}: mean rank = {np.mean(ranks):.1f}, ranks = {ranks}")


def generate_statistical_comparison(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 20,
) -> None:
    """Generate statistical comparison: similar vs dissimilar with significance tests."""
    print("\n=== Statistical Comparison ===")
    
    from scipy import stats
    
    all_metrics = [
        ('mean_cos', 'Mean Cosine'),
        ('eigenvalue_weighted', 'Eigenvalue-Weighted'),
        ('quadratic_form', 'Quadratic Form'),
    ]
    
    control_pairs = [
        (0, 23), (1, 22), (3, 7), (7, 14),
        (4, 0), (6, 5), (8, 11), (9, 17),  # Additional dissimilar pairs
    ]
    
    results = {}
    
    for method, label in all_metrics:
        # Compute similar pair similarities
        similar_sims = []
        for digit_idx, letter_idx, _ in DIGIT_LETTER_PAIRS:
            if method == 'mean_cos':
                _, d_sorted = mnist_vals[digit_idx].abs().sort(descending=True)
                _, l_sorted = letters_vals[letter_idx].abs().sort(descending=True)
                d_vecs = mnist_vecs[digit_idx][d_sorted].cpu()
                l_vecs = letters_vecs[letter_idx][l_sorted].cpu()
                sim = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
            else:
                sim = compute_weighted_similarity(
                    mnist_vecs[digit_idx].cpu(), letters_vecs[letter_idx].cpu(),
                    mnist_vals[digit_idx].cpu(), letters_vals[letter_idx].cpu(),
                    k=k, method=method
                )
            similar_sims.append(sim)
        
        # Compute dissimilar pair similarities
        dissimilar_sims = []
        for d_idx, l_idx in control_pairs:
            if method == 'mean_cos':
                _, d_sorted = mnist_vals[d_idx].abs().sort(descending=True)
                _, l_sorted = letters_vals[l_idx].abs().sort(descending=True)
                d_vecs = mnist_vecs[d_idx][d_sorted].cpu()
                l_vecs = letters_vecs[l_idx][l_sorted].cpu()
                sim = compute_subspace_overlap(d_vecs[:k], l_vecs[:k], k=k, method='mean_cos')
            else:
                sim = compute_weighted_similarity(
                    mnist_vecs[d_idx].cpu(), letters_vecs[l_idx].cpu(),
                    mnist_vals[d_idx].cpu(), letters_vals[l_idx].cpu(),
                    k=k, method=method
                )
            dissimilar_sims.append(sim)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(similar_sims, dissimilar_sims)
        
        results[method] = {
            'similar_mean': np.mean(similar_sims),
            'similar_std': np.std(similar_sims),
            'dissimilar_mean': np.mean(dissimilar_sims),
            'dissimilar_std': np.std(dissimilar_sims),
            'gap': np.mean(similar_sims) - np.mean(dissimilar_sims),
            't_stat': t_stat,
            'p_value': p_value,
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(all_metrics))
    width = 0.35
    
    similar_means = [results[m]['similar_mean'] for m, _ in all_metrics]
    similar_stds = [results[m]['similar_std'] for m, _ in all_metrics]
    dissimilar_means = [results[m]['dissimilar_mean'] for m, _ in all_metrics]
    dissimilar_stds = [results[m]['dissimilar_std'] for m, _ in all_metrics]
    
    bars1 = ax.bar(x - width/2, similar_means, width, yerr=similar_stds,
                  label='Similar pairs (0-O, 1-I, 2-Z, 5-S)', color='#2ecc71', capsize=5)
    bars2 = ax.bar(x + width/2, dissimilar_means, width, yerr=dissimilar_stds,
                  label='Dissimilar pairs (control)', color='#e74c3c', capsize=5)
    
    # Add significance stars
    for i, (method, _) in enumerate(all_metrics):
        p = results[method]['p_value']
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        
        y_max = max(similar_means[i] + similar_stds[i], dissimilar_means[i] + dissimilar_stds[i])
        ax.text(i, y_max + 0.05, stars, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Similarity Metric', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title(f'Statistical Comparison: Similar vs Dissimilar Pairs (k={k})\n(*** p<0.001, ** p<0.01, * p<0.05)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in all_metrics], fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    out_path = d.figure_out / "extension2_statistical_comparison_all_metrics.pdf"
    _save_figure(fig, out_path)
    plt.close(fig)
    
    # Print results
    print("  Statistical Comparison Results:")
    for method, label in all_metrics:
        r = results[method]
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else ''))
        print(f"    {label}: similar={r['similar_mean']:.4f}±{r['similar_std']:.4f}, "
              f"dissimilar={r['dissimilar_mean']:.4f}±{r['dissimilar_std']:.4f}, "
              f"gap={r['gap']:.4f}, p={r['p_value']:.4f} {sig}")


SECTION_MAP = {
    # Original sections
    "eigenvectors": generate_eigenvector_comparisons,
    "distributions": generate_eigenvalue_distributions,
    "similarity": generate_similarity_vs_k,
    "similarity_heatmap": generate_similarity_heatmap,
    "3way": generate_3way_comparison,
    "eigenvector_table": generate_eigenvector_comparison_table,
    "selection": generate_selection_method_comparison,
    "angles": generate_principal_angles,
    # Eigenvalue-aware metrics sections
    "similarity_weighted": generate_similarity_vs_k_weighted,
    "heatmap_weighted": generate_similarity_heatmap_weighted,
    "heatmap_digits": generate_quadratic_form_heatmap_digits,
    "metric_comparison": generate_metric_comparison,
    "ranking": generate_ranking_analysis,
    "statistical": generate_statistical_comparison,
    # New cosine-based diagnostics used in the report
    "3way_0_O_X": generate_3way_comparison_0_O_X,
    "eigenvector_table_0_O_X": generate_eigenvector_comparison_table_0_O_X,
    "cosine_heatmap_abs": generate_cosine_similarity_heatmap_abs,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Extension 2 figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=list(SECTION_MAP.keys()) + ["all"],
        default=["all"],
        help="Which sections to generate",
    )
    args = parser.parse_args()
    
    set_publication_style()
    
    # Ensure artifacts are available (downloads from Google Drive if missing)
    ensure_artifacts()
    
    d = get_dirs()
    print(f"Output directory: {d.figure_out}")
    
    # Load checkpoints
    mnist_vals, mnist_vecs, digits_vals, digits_vecs, letters_vals, letters_vecs = load_checkpoints(d)
    
    # Determine which sections to run
    sections = args.sections
    if "all" in sections:
        sections = list(SECTION_MAP.keys())
    
    for section in sections:
        func = SECTION_MAP[section]
        
        # Handle functions with different signatures
        if section in ["3way", "eigenvector_table"]:
            func(d, mnist_vecs, mnist_vals, digits_vecs, digits_vals, letters_vecs, letters_vals)
        elif section == "heatmap_digits":
            func(d, mnist_vecs, mnist_vals, digits_vecs, digits_vals)
        elif section in ["eigenvectors", "similarity", "similarity_heatmap", "selection", "angles",
                         "similarity_weighted", "heatmap_weighted", "metric_comparison", "ranking", "statistical",
                         "3way_0_O_X", "eigenvector_table_0_O_X", "cosine_heatmap_abs"]:
            func(d, mnist_vecs, mnist_vals, letters_vecs, letters_vals)
        elif section == "distributions":
            func(d, mnist_vals, letters_vals)
    
    print("\n" + "=" * 60)
    print("Extension 2 figure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
