#!/usr/bin/env python3
"""
Check subspace similarity between different datasets/classes.

Computes subspace overlap using top-k eigenvectors with multiple metrics.

Usage:
    python scripts/extension2/check_similarity.py --k 30
    python scripts/extension2/check_similarity.py --with-com
"""

import sys
from pathlib import Path
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.spectral import load_checkpoint_eigenvalues
from src.vision.subspace import compute_subspace_overlap
from src.utils import get_device, setup_mps_fallbacks, is_mps_device


def compute_subspace_comparison(vecs_A, vals_A, vecs_B, vals_B, 
                                class_A, class_B, k, label_A, label_B):
    """
    Compute subspace similarity between two classes.
    
    Args:
        vecs_A: Eigenvectors for dataset A [n_classes, n_components, d_input]
        vals_A: Eigenvalues for dataset A [n_classes, n_components]
        vecs_B: Eigenvectors for dataset B [n_classes, n_components, d_input]
        vals_B: Eigenvalues for dataset B [n_classes, n_components]
        class_A: Class index in dataset A
        class_B: Class index in dataset B
        k: Number of top eigenvectors to compare
        label_A: Label string for dataset A
        label_B: Label string for dataset B
    """
    # Get eigenvectors for these classes
    vals_A_class = vals_A[class_A]  # [n_components]
    vecs_A_class = vecs_A[class_A]  # [n_components, 784]
    vals_B_class = vals_B[class_B]  # [n_components]
    vecs_B_class = vecs_B[class_B]  # [n_components, 784]
    
    # Sort by eigenvalue magnitude (descending)
    A_indices = vals_A_class.abs().argsort(descending=True)
    B_indices = vals_B_class.abs().argsort(descending=True)
    
    # Get top-k eigenvectors (sorted by magnitude)
    A_topk_vecs = vecs_A_class[A_indices[:k]]  # [k, 784]
    B_topk_vecs = vecs_B_class[B_indices[:k]]  # [k, 784]
    
    # Get corresponding eigenvalues for reference
    A_topk_vals = vals_A_class[A_indices[:k]]
    B_topk_vals = vals_B_class[B_indices[:k]]
    
    print(f"\nTop-{k} Eigenvalues (by magnitude):")
    print(f"  {label_A}: {A_topk_vals.abs().cpu().numpy()[:5]} ... (showing first 5)")
    print(f"  {label_B}: {B_topk_vals.abs().cpu().numpy()[:5]} ... (showing first 5)")
    
    # Compute subspace overlap using all three methods
    print("\n" + "=" * 80)
    print("SUBSPACE OVERLAP METRICS")
    print("=" * 80)
    
    methods = ['mean_cos', 'grassmann', 'projection']
    results = {}
    
    for method in methods:
        similarity = compute_subspace_overlap(
            A_topk_vecs,
            B_topk_vecs,
            k=k,
            method=method
        )
        results[method] = similarity
        
        method_names = {
            'mean_cos': 'Mean Cosine (Principal Angles)',
            'grassmann': 'Grassmann Distance',
            'projection': 'Projection-Based (Frobenius)'
        }
        
        print(f"\n{method_names[method]}:")
        print(f"  Similarity: {similarity:.6f}")
        print(f"  Interpretation: ", end="")
        if similarity > 0.7:
            print("High similarity - subspaces are well-aligned")
        elif similarity > 0.4:
            print("Moderate similarity - some alignment")
        elif similarity > 0.2:
            print("Low similarity - limited alignment")
        else:
            print("Very low similarity - subspaces are nearly orthogonal")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check subspace similarity between datasets")
    parser.add_argument("--k", type=int, default=30, help="Number of top eigenvectors to compare")
    parser.add_argument("--reg-config", type=str, default="phase1", choices=["phase1", "current"],
                        help="Regularization config: 'phase1' (noise=0.5, wd=1.0) or 'current' (noise=0.15, wd=0.5)")
    parser.add_argument("--no-com", action="store_true",
                        help="Use checkpoints trained without CoM normalization")
    parser.add_argument("--with-com", action="store_true",
                        help="Use checkpoints trained WITH CoM normalization (both MNIST and EMNIST)")
    args = parser.parse_args()
    
    # Configuration
    k = args.k
    reg_config = args.reg_config
    use_nocom = args.no_com
    use_com = args.with_com
    
    # Device setup
    device = get_device()
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Load checkpoints
    checkpoint_dir = PROJECT_ROOT / "results/phase1/checkpoints"
    
    if reg_config == "phase1":
        if use_com:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_com_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_phase1_reg_com_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_phase1_reg_com_seed42.pt"
            reg_label = "Phase 1 (noise=0.5, wd=1.0) - WITH CoM normalization"
        elif use_nocom:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_phase1_reg_nocom_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_phase1_reg_nocom_seed42.pt"
            reg_label = "Phase 1 (noise=0.5, wd=1.0) - NO CoM normalization"
        else:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_phase1_reg_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_phase1_reg_seed42.pt"
            reg_label = "Phase 1 (noise=0.5, wd=1.0) - Default (MNIST no CoM, EMNIST unknown)"
    else:
        if use_com:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_com_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_regularized_com_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_regularized_com_seed42.pt"
            reg_label = "Current (noise=0.15, wd=0.5) - WITH CoM normalization"
        elif use_nocom:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_regularized_nocom_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_regularized_nocom_seed42.pt"
            reg_label = "Current (noise=0.15, wd=0.5) - NO CoM normalization"
        else:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            digits_checkpoint = checkpoint_dir / "emnist_digits_regularized_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_regularized_seed42.pt"
            reg_label = "Current (noise=0.15, wd=0.5) - Default (MNIST no CoM, EMNIST unknown)"
    
    print(f"\nRegularization: {reg_label}")
    print(f"Comparing top-{k} eigenvectors")
    print("=" * 80)
    
    # Load eigenvalues and eigenvectors
    print("\nLoading checkpoints...")
    
    # Check which checkpoints exist
    mnist_exists = mnist_checkpoint.exists()
    digits_exists = digits_checkpoint.exists()
    letters_exists = letters_checkpoint.exists()
    
    if not mnist_exists:
        print(f"ERROR: MNIST checkpoint not found: {mnist_checkpoint}")
        return
    
    if not letters_exists:
        print(f"ERROR: EMNIST Letters checkpoint not found: {letters_checkpoint}")
        return
    
    # Load available checkpoints
    mnist_vals, mnist_vecs = load_checkpoint_eigenvalues(str(mnist_checkpoint))
    print(f"MNIST: {mnist_vecs.shape}")
    
    if digits_exists:
        digits_vals, digits_vecs = load_checkpoint_eigenvalues(str(digits_checkpoint))
        print(f"EMNIST Digits: {digits_vecs.shape}")
    
    letters_vals, letters_vecs = load_checkpoint_eigenvalues(str(letters_checkpoint))
    print(f"EMNIST Letters: {letters_vecs.shape}")
    
    # Class indices
    mnist_digit_0 = 0
    emnist_digit_0 = 0
    emnist_letter_O = 14  # 'O' is the 15th letter (A=0, B=1, ..., O=14)
    
    results1 = None
    results2 = None
    
    # Comparison 1: MNIST digit 0 vs EMNIST digit 0 (if available)
    if digits_exists:
        print("\n" + "=" * 80)
        print("COMPARISON 1: MNIST Digit 0 vs EMNIST Digit 0")
        print("=" * 80)
        
        results1 = compute_subspace_comparison(
            mnist_vecs, mnist_vals,
            digits_vecs, digits_vals,
            mnist_digit_0, emnist_digit_0, k,
            "MNIST Digit 0", "EMNIST Digit 0"
        )
    else:
        print(f"\nSkipping MNIST 0 vs EMNIST 0: EMNIST Digits checkpoint not found")
    
    # Comparison 2: MNIST digit 0 vs EMNIST letter O
    print("\n" + "=" * 80)
    print("COMPARISON 2: MNIST Digit 0 vs EMNIST Letter O")
    print("=" * 80)
    
    results2 = compute_subspace_comparison(
        mnist_vecs, mnist_vals,
        letters_vecs, letters_vals,
        mnist_digit_0, emnist_letter_O, k,
        "MNIST Digit 0", "EMNIST Letter O"
    )
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"Top-{k} eigenvectors, Regularization: {reg_label}")
    print("\n" + "-" * 80)
    if results1 is not None:
        print(f"{'Metric':<20} {'MNIST 0 vs EMNIST 0':<25} {'MNIST 0 vs EMNIST O':<25}")
        print("-" * 80)
        print(f"{'Mean Cosine':<20} {results1['mean_cos']:<25.6f} {results2['mean_cos']:<25.6f}")
        print(f"{'Grassmann':<20} {results1['grassmann']:<25.6f} {results2['grassmann']:<25.6f}")
        print(f"{'Projection':<20} {results1['projection']:<25.6f} {results2['projection']:<25.6f}")
        print("-" * 80)
        
        # Interpretation
        print("\nInterpretation:")
        if results1['mean_cos'] > results2['mean_cos']:
            diff = results1['mean_cos'] - results2['mean_cos']
            print(f"  MNIST 0 vs EMNIST 0 has {diff:.4f} higher similarity than MNIST 0 vs EMNIST O")
            print("  -> Digit-to-digit similarity is stronger than digit-to-letter similarity")
        else:
            diff = results2['mean_cos'] - results1['mean_cos']
            print(f"  MNIST 0 vs EMNIST O has {diff:.4f} higher similarity than MNIST 0 vs EMNIST 0")
            print("  -> Shape similarity (circular) may be more important than dataset match")
    else:
        print(f"{'Metric':<20} {'MNIST 0 vs EMNIST O':<25}")
        print("-" * 80)
        print(f"{'Mean Cosine':<20} {results2['mean_cos']:<25.6f}")
        print(f"{'Grassmann':<20} {results2['grassmann']:<25.6f}")
        print(f"{'Projection':<20} {results2['projection']:<25.6f}")
        print("-" * 80)
        print("\nNote: EMNIST Digits checkpoint not available, only showing MNIST 0 vs EMNIST O comparison")


if __name__ == "__main__":
    main()
