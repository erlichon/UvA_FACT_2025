#!/usr/bin/env python3
"""
Generate Extension 2 figures (Cross-Dataset Structural Robustness).

This script is the single entrypoint for all Extension 2 figures. It only uses
existing checkpoints (no new training).

Figures generated:
1. Eigenvector comparison: MNIST digit vs EMNIST letter (per pair)
2. Cosine similarity heatmaps (per pair)
3. Eigenvalue distribution overlays (per pair)
4. Subspace overlap vs k (similarity over eigenvector count)
5. Principal angles between subspaces
6. 3-way comparison: MNIST digit 0, EMNIST digit 0, EMNIST letter O
7. Selection method comparison: magnitude vs balanced

Usage:
    python scripts/figures/generate_extension2_figures.py
    python scripts/figures/generate_extension2_figures.py --sections eigenvectors
    ./scripts/train/run_extension2.sh figures  # Preferred wrapper
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

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Add project + original code paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.vision.spectral import load_checkpoint_eigenvalues
from src.vision.subspace import compute_subspace_overlap, select_balanced_eigenvectors
from src.plot_utils.style import set_publication_style
from src.plot_utils.extension2 import (
    DIGIT_LETTER_PAIRS,
    plot_digit_letter_eigenvector_comparison,
    plot_cosine_similarity_heatmap,
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
    report_figures: Path


def get_dirs() -> Dirs:
    """Get directory paths, with fallbacks for different checkpoint names.
    
    For Extension 2, ALL models should be trained with CoM normalization.
    MNIST CoM checkpoints are in results/extension2/checkpoints/ (trained via run_extension2.sh).
    """
    ext2_ckpt_dir = PROJECT_ROOT / "results/extension2/checkpoints"
    phase1_ckpt_dir = PROJECT_ROOT / "results/phase1/checkpoints"
    
    # MNIST: Prefer extension2 CoM checkpoint, fall back to phase1 (non-CoM, but warn)
    mnist_candidates = [
        ext2_ckpt_dir / "mnist_dense_full_com_seed42.pt",  # Extension 2 CoM (preferred)
        phase1_ckpt_dir / "mnist_dense_full_seed42.pt",     # Phase 1 non-CoM (fallback)
    ]
    emnist_digits_candidates = [
        ext2_ckpt_dir / "emnist_digits_regularized_seed42.pt",
    ]
    emnist_letters_candidates = [
        ext2_ckpt_dir / "emnist_letters_regularized_seed42.pt",
    ]
    
    mnist_ckpt = next((p for p in mnist_candidates if p.exists()), mnist_candidates[0])
    emnist_digits_ckpt = next((p for p in emnist_digits_candidates if p.exists()), emnist_digits_candidates[0])
    emnist_letters_ckpt = next((p for p in emnist_letters_candidates if p.exists()), emnist_letters_candidates[0])
    
    # Warn if using non-CoM MNIST checkpoint
    if "phase1" in str(mnist_ckpt) or "dense_full_seed" in str(mnist_ckpt):
        print("⚠️  WARNING: Using Phase 1 MNIST checkpoint (no CoM).")
        print("   For fair Extension 2 comparison, train MNIST with CoM:")
        print("   ./scripts/train/run_extension2.sh train mnist")
    
    return Dirs(
        mnist_ckpt=mnist_ckpt,
        emnist_digits_ckpt=emnist_digits_ckpt,
        emnist_letters_ckpt=emnist_letters_ckpt,
        figure_out=PROJECT_ROOT / "results/extension2/figures",
        report_figures=PROJECT_ROOT / "Report/figures",
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_and_copy(fig: plt.Figure, out_path: Path, report_path: Path, dpi: int = 300) -> None:
    _ensure_dir(out_path.parent)
    _ensure_dir(report_path.parent)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(report_path, bbox_inches="tight", dpi=dpi)
    print(f"  Saved: {out_path.name}")


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
        out_path = d.figure_out / f"eigenvec_comparison_{digit}_{letter}.pdf"
        report_path = d.report_figures / f"extension2_eigenvec_{digit}_{letter}.pdf"
        _save_and_copy(fig, out_path, report_path)
        plt.close(fig)


def generate_cosine_heatmaps(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
) -> None:
    """Generate cosine similarity heatmaps for each digit-letter pair."""
    print("\n=== Cosine Similarity Heatmaps ===")
    
    for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
        digit, letter = label.split("-")
        
        fig = plot_cosine_similarity_heatmap(
            mnist_vecs[digit_idx],
            letters_vecs[letter_idx],
            mnist_vals[digit_idx],
            letters_vals[letter_idx],
            k=10,
            digit_label=digit,
            letter_label=letter,
        )
        
        out_path = d.figure_out / f"cosine_heatmap_{digit}_{letter}.pdf"
        report_path = d.report_figures / f"extension2_cosine_{digit}_{letter}.pdf"
        _save_and_copy(fig, out_path, report_path)
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
        
        out_path = d.figure_out / f"eigenval_dist_{digit}_{letter}.pdf"
        report_path = d.report_figures / f"extension2_eigenval_{digit}_{letter}.pdf"
        _save_and_copy(fig, out_path, report_path)
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
    
    out_path = d.figure_out / "similarity_vs_k.pdf"
    report_path = d.report_figures / "extension2_similarity_vs_k.pdf"
    _save_and_copy(fig, out_path, report_path)
    plt.close(fig)


def generate_similarity_heatmap(
    d: Dirs,
    mnist_vecs: torch.Tensor,
    mnist_vals: torch.Tensor,
    letters_vecs: torch.Tensor,
    letters_vals: torch.Tensor,
    k: int = 10,
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
    
    out_path = d.figure_out / "similarity_heatmap.pdf"
    report_path = d.report_figures / "extension2_similarity_heatmap.pdf"
    _save_and_copy(fig, out_path, report_path)
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
    
    out_path = d.figure_out / "3way_comparison.pdf"
    report_path = d.report_figures / "extension2_3way_comparison.pdf"
    _save_and_copy(fig, out_path, report_path)
    plt.close(fig)


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
    
    out_path = d.figure_out / "selection_method_comparison.pdf"
    report_path = d.report_figures / "extension2_selection_comparison.pdf"
    _save_and_copy(fig, out_path, report_path)
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
    
    out_path = d.figure_out / "principal_angles.pdf"
    report_path = d.report_figures / "extension2_principal_angles.pdf"
    _save_and_copy(fig, out_path, report_path)
    plt.close(fig)


SECTION_MAP = {
    "eigenvectors": generate_eigenvector_comparisons,
    "heatmaps": generate_cosine_heatmaps,
    "distributions": generate_eigenvalue_distributions,
    "similarity": generate_similarity_vs_k,
    "similarity_heatmap": generate_similarity_heatmap,
    "3way": generate_3way_comparison,
    "selection": generate_selection_method_comparison,
    "angles": generate_principal_angles,
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
    
    d = get_dirs()
    print(f"Output directory: {d.figure_out}")
    print(f"Report directory: {d.report_figures}")
    
    # Load checkpoints
    mnist_vals, mnist_vecs, digits_vals, digits_vecs, letters_vals, letters_vecs = load_checkpoints(d)
    
    # Determine which sections to run
    sections = args.sections
    if "all" in sections:
        sections = list(SECTION_MAP.keys())
    
    for section in sections:
        func = SECTION_MAP[section]
        
        # Handle functions with different signatures
        if section == "3way":
            func(d, mnist_vecs, mnist_vals, digits_vecs, digits_vals, letters_vecs, letters_vals)
        elif section in ["eigenvectors", "heatmaps", "similarity", "similarity_heatmap", "selection", "angles"]:
            func(d, mnist_vecs, mnist_vals, letters_vecs, letters_vals)
        elif section == "distributions":
            func(d, mnist_vals, letters_vals)
    
    print("\n" + "=" * 60)
    print("Extension 2 figure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
