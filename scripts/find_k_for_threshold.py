#!/usr/bin/env python3
"""
Find the minimum k (number of eigenvectors) needed to reach a target similarity threshold.
"""

import sys
from pathlib import Path
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.spectral import load_checkpoint_eigenvalues
from src.vision.subspace import compute_subspace_overlap
from src.utils import get_device, setup_mps_fallbacks, is_mps_device


def main():
    # Configuration
    target_threshold = 0.8  # Target mean cosine similarity
    reg_config = "phase1"
    use_com = True
    
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
            letters_checkpoint = checkpoint_dir / "emnist_letters_phase1_reg_com_seed42.pt"
            reg_label = "Phase 1 (noise=0.5, wd=1.0) - WITH CoM normalization"
        else:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_phase1_reg_seed42.pt"
            reg_label = "Phase 1 (noise=0.5, wd=1.0)"
    else:
        if use_com:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_com_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_regularized_com_seed42.pt"
            reg_label = "Current (noise=0.15, wd=0.5) - WITH CoM normalization"
        else:
            mnist_checkpoint = checkpoint_dir / "mnist_dense_full_seed42.pt"
            letters_checkpoint = checkpoint_dir / "emnist_letters_regularized_seed42.pt"
            reg_label = "Current (noise=0.15, wd=0.5)"
    
    print(f"\nRegularization: {reg_label}")
    print(f"Target: Mean Cosine >= {target_threshold}")
    print("=" * 80)
    
    # Load checkpoints
    print("\nLoading checkpoints...")
    mnist_vals, mnist_vecs = load_checkpoint_eigenvalues(str(mnist_checkpoint))
    letters_vals, letters_vecs = load_checkpoint_eigenvalues(str(letters_checkpoint))
    
    print(f"MNIST: {mnist_vecs.shape}")
    print(f"EMNIST Letters: {letters_vecs.shape}")
    
    # Class indices
    mnist_digit_0 = 0
    emnist_letter_O = 14  # 'O' is the 15th letter
    
    # Get eigenvectors for these classes
    mnist_vals_class = mnist_vals[mnist_digit_0]  # [n_components]
    mnist_vecs_class = mnist_vecs[mnist_digit_0]  # [n_components, 784]
    letters_vals_class = letters_vals[emnist_letter_O]  # [n_components]
    letters_vecs_class = letters_vecs[emnist_letter_O]  # [n_components, 784]
    
    # Sort by eigenvalue magnitude (descending)
    mnist_indices = mnist_vals_class.abs().argsort(descending=True)
    letters_indices = letters_vals_class.abs().argsort(descending=True)
    
    # Iterate over k values
    print("\n" + "=" * 80)
    print("ITERATING OVER k VALUES")
    print("=" * 80)
    print(f"{'k':<6} {'Mean Cosine':<15} {'Grassmann':<15} {'Projection':<15} {'Status':<15}")
    print("-" * 80)
    
    max_k = min(256, len(mnist_vecs_class), len(letters_vecs_class))  # Don't exceed available eigenvectors
    found_k = None
    max_cos = 0
    best_k = None
    
    # Test k values: fine-grained around peak (30-45), then coarser
    k_values = []
    k_values.extend(range(10, 30, 5))  # 10, 15, 20, 25
    k_values.extend(range(30, 46, 1))  # Fine-grained: 30-45
    k_values.extend(range(50, max_k + 1, 10))  # Coarser: 50, 60, 70, ...
    if k_values[-1] != max_k:
        k_values.append(max_k)  # Include max_k if not already there
    k_values = sorted(set(k_values))  # Remove duplicates and sort
    
    for k in k_values:
        # Get top-k eigenvectors
        mnist_topk = mnist_vecs_class[mnist_indices[:k]]  # [k, 784]
        letters_topk = letters_vecs_class[letters_indices[:k]]  # [k, 784]
        
        # Compute similarity
        mean_cos = compute_subspace_overlap(mnist_topk, letters_topk, k=k, method='mean_cos')
        grassmann = compute_subspace_overlap(mnist_topk, letters_topk, k=k, method='grassmann')
        projection = compute_subspace_overlap(mnist_topk, letters_topk, k=k, method='projection')
        
        if mean_cos > max_cos:
            max_cos = mean_cos
            best_k = k
        
        status = "✓ TARGET REACHED" if mean_cos >= target_threshold else ""
        if mean_cos >= target_threshold and found_k is None:
            found_k = k
        
        print(f"{k:<6} {mean_cos:<15.6f} {grassmann:<15.6f} {projection:<15.6f} {status:<15}")
        
        # If we've found the target and it's been stable, we can stop
        if found_k is not None and k >= found_k + 10:
            break
    
    print("-" * 80)
    
    # Also try balanced selection
    print("\n" + "=" * 80)
    print("TESTING BALANCED SELECTION (k/2 positive + k/2 negative)")
    print("=" * 80)
    print(f"{'k':<6} {'Mean Cosine':<15} {'Grassmann':<15} {'Projection':<15} {'Status':<15}")
    print("-" * 80)
    
    from src.vision.subspace import select_balanced_eigenvectors
    
    # Get balanced selection for all classes
    mnist_balanced = select_balanced_eigenvectors(
        mnist_vals, mnist_vecs, k=max_k
    )
    letters_balanced = select_balanced_eigenvectors(
        letters_vals, letters_vecs, k=max_k
    )
    
    mnist_balanced_class = mnist_balanced[mnist_digit_0]  # [max_k, 784]
    letters_balanced_class = letters_balanced[emnist_letter_O]  # [max_k, 784]
    
    # Test balanced selection with same k values
    found_k_balanced = None
    max_balanced_cos = 0
    best_k_balanced = None
    
    for k in k_values[:20]:  # Test first 20 k values
        if k > max_k:
            break
        
        mnist_bal_k = mnist_balanced_class[:k]
        letters_bal_k = letters_balanced_class[:k]
        
        mean_cos_bal = compute_subspace_overlap(mnist_bal_k, letters_bal_k, k=k, method='mean_cos')
        grassmann_bal = compute_subspace_overlap(mnist_bal_k, letters_bal_k, k=k, method='grassmann')
        projection_bal = compute_subspace_overlap(mnist_bal_k, letters_bal_k, k=k, method='projection')
        
        if mean_cos_bal > max_balanced_cos:
            max_balanced_cos = mean_cos_bal
            best_k_balanced = k
        
        status_bal = "✓ TARGET REACHED" if mean_cos_bal >= target_threshold else ""
        if mean_cos_bal >= target_threshold and found_k_balanced is None:
            found_k_balanced = k
        
        print(f"{k:<6} {mean_cos_bal:<15.6f} {grassmann_bal:<15.6f} {projection_bal:<15.6f} {status_bal:<15}")
        
        if found_k_balanced is not None:
            break
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Top-k by magnitude:")
    if found_k is not None:
        print(f"  ✓ Target reached at k = {found_k}")
    else:
        print(f"  ✗ Target not reached (max: {max_cos:.6f} at k={best_k})")
        print(f"    Final Mean Cosine similarity: {max_cos:.6f} < {target_threshold}")
    
    print("\nBalanced selection:")
    if found_k_balanced is not None:
        print(f"  ✓ Target reached at k = {found_k_balanced}")
    else:
        print(f"  ✗ Target not reached (max: {max_balanced_cos:.6f} at k={best_k_balanced})")
        print(f"    Final Mean Cosine similarity: {max_balanced_cos:.6f} < {target_threshold}")
        print(f"    Try different selection methods or check if threshold is achievable")


if __name__ == "__main__":
    main()
