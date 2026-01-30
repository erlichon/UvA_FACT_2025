#!/usr/bin/env python3
"""
Calculate absolute cosine similarity between top 10 eigenvectors (balanced 5 positive + 5 negative)
for:
- MNIST 3 vs EMNIST 7
- MNIST 3 vs EMNIST 2
- MNIST 3 vs EMNIST E
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EXTENSION2_CHECKPOINTS
from src.vision.spectral import load_checkpoint_eigenvalues
from scripts.extension_cross_dataset.utils import (
    get_balanced_eigenvectors_and_vals,
    abs_cosine_similarity,
)


def compute_similarity_stats(vecs_A, vecs_B, vals_A):
    """Compute similarity statistics: per-rank max and overall mean."""
    sim_matrix = abs_cosine_similarity(vecs_A, vecs_B)  # [k, k]
    
    # For each eigenvector in A, find max similarity with any eigenvector in B
    max_sims_per_rank = sim_matrix.max(dim=1)[0]  # [k]
    
    # Overall mean
    mean_sim = max_sims_per_rank.mean().item()
    
    # Per-rank similarities
    per_rank = max_sims_per_rank.tolist()
    
    # Get eigenvalue signs for reference
    signs = ["+" if v > 0 else "-" for v in vals_A]
    
    return {
        'mean': mean_sim,
        'per_rank': per_rank,
        'signs': signs,
        'sim_matrix': sim_matrix
    }


def main():
    print("=" * 80)
    print("Eigenvector Similarity Calculation")
    print("=" * 80)
    print("\nComparing top 10 eigenvectors (balanced: 5 positive + 5 negative)")
    print("  - MNIST class 3 (digit 3)")
    print("  - EMNIST digits class 7 (digit 7)")
    print("  - EMNIST digits class 2 (digit 2)")
    print("  - EMNIST letters class E (index 4)")
    print()
    
    # Load checkpoints (using multiple seeds)
    seeds = [42, 43, 44, 45, 46]
    k = 10
    
    # Class indices
    mnist_class = 3  # digit 3
    emnist_digit_class_7 = 7  # digit 7
    emnist_digit_class_2 = 2  # digit 2
    emnist_letter_class = 4  # letter E (A=0, B=1, C=2, D=3, E=4)
    
    # Accumulate results across seeds
    results_mnist_emnist_digit_7 = []
    results_mnist_emnist_digit_2 = []
    results_mnist_emnist_letter = []
    loaded_seeds = []
    
    for seed in seeds:
        # Try to load checkpoints
        mnist_ckpt = EXTENSION2_CHECKPOINTS / f"mnist_dense_full_com_seed{seed}.pt"
        digits_ckpt = EXTENSION2_CHECKPOINTS / f"emnist_digits_regularized_seed{seed}.pt"
        letters_ckpt = EXTENSION2_CHECKPOINTS / f"emnist_letters_regularized_seed{seed}.pt"
        
        if not all(p.exists() for p in [mnist_ckpt, digits_ckpt, letters_ckpt]):
            print(f"  Seed {seed}: Missing checkpoints, skipping...")
            continue
        
        try:
            # Load eigenvalues and eigenvectors
            mnist_vals, mnist_vecs = load_checkpoint_eigenvalues(str(mnist_ckpt))
            digits_vals, digits_vecs = load_checkpoint_eigenvalues(str(digits_ckpt))
            letters_vals, letters_vecs = load_checkpoint_eigenvalues(str(letters_ckpt))
            
            # Get balanced eigenvectors
            mnist_3_vecs, mnist_3_vals = get_balanced_eigenvectors_and_vals(
                mnist_vals, mnist_vecs, mnist_class, k
            )
            digits_7_vecs, _ = get_balanced_eigenvectors_and_vals(
                digits_vals, digits_vecs, emnist_digit_class_7, k
            )
            digits_2_vecs, _ = get_balanced_eigenvectors_and_vals(
                digits_vals, digits_vecs, emnist_digit_class_2, k
            )
            letters_e_vecs, _ = get_balanced_eigenvectors_and_vals(
                letters_vals, letters_vecs, emnist_letter_class, k
            )
            
            # Compute similarities
            sim_mnist_digits_7 = abs_cosine_similarity(mnist_3_vecs, digits_7_vecs)
            sim_mnist_digits_2 = abs_cosine_similarity(mnist_3_vecs, digits_2_vecs)
            sim_mnist_letters = abs_cosine_similarity(mnist_3_vecs, letters_e_vecs)
            
            results_mnist_emnist_digit_7.append(sim_mnist_digits_7)
            results_mnist_emnist_digit_2.append(sim_mnist_digits_2)
            results_mnist_emnist_letter.append(sim_mnist_letters)
            loaded_seeds.append(seed)
            
        except Exception as e:
            print(f"  Seed {seed}: Error loading checkpoint: {e}")
            continue
    
    if len(loaded_seeds) == 0:
        print("ERROR: No checkpoints found!")
        return
    
    print(f"Loaded {len(loaded_seeds)} seeds: {loaded_seeds}\n")
    
    # Compute statistics across seeds
    sim_digit_7_stack = torch.stack(results_mnist_emnist_digit_7)  # [n_seeds, k, k]
    sim_digit_2_stack = torch.stack(results_mnist_emnist_digit_2)  # [n_seeds, k, k]
    sim_letter_stack = torch.stack(results_mnist_emnist_letter)  # [n_seeds, k, k]
    
    sim_digit_7_mean = sim_digit_7_stack.mean(dim=0)  # [k, k]
    sim_digit_2_mean = sim_digit_2_stack.mean(dim=0)  # [k, k]
    sim_letter_mean = sim_letter_stack.mean(dim=0)  # [k, k]
    
    # Get eigenvalue signs from first seed
    mnist_ckpt = EXTENSION2_CHECKPOINTS / f"mnist_dense_full_com_seed{loaded_seeds[0]}.pt"
    mnist_vals, mnist_vecs = load_checkpoint_eigenvalues(str(mnist_ckpt))
    _, mnist_3_vals = get_balanced_eigenvectors_and_vals(mnist_vals, mnist_vecs, mnist_class, k)
    
    # Compute per-rank max similarities
    max_sims_digit_7_per_rank = sim_digit_7_mean.max(dim=1)[0]  # [k]
    max_sims_digit_2_per_rank = sim_digit_2_mean.max(dim=1)[0]  # [k]
    max_sims_letter_per_rank = sim_letter_mean.max(dim=1)[0]  # [k]
    
    # Compute std across seeds for each rank
    max_sims_digit_7_by_rank = []
    max_sims_digit_2_by_rank = []
    max_sims_letter_by_rank = []
    for rank in range(k):
        max_digit_7_per_seed = [sim[rank].max().item() for sim in results_mnist_emnist_digit_7]
        max_digit_2_per_seed = [sim[rank].max().item() for sim in results_mnist_emnist_digit_2]
        max_letter_per_seed = [sim[rank].max().item() for sim in results_mnist_emnist_letter]
        max_sims_digit_7_by_rank.append(max_digit_7_per_seed)
        max_sims_digit_2_by_rank.append(max_digit_2_per_seed)
        max_sims_letter_by_rank.append(max_letter_per_seed)
    
    # Print results
    print("=" * 80)
    print("RESULTS: Absolute Cosine Similarity")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Sign':<6} {'MNIST-3 vs EMNIST-7':<25} {'MNIST-3 vs EMNIST-2':<25} {'MNIST-3 vs EMNIST-E':<25}")
    print("-" * 100)
    
    for rank in range(k):
        val_sign = "+" if mnist_3_vals[rank].item() > 0 else "-"
        digit_7_mean = max_sims_digit_7_per_rank[rank].item()
        digit_2_mean = max_sims_digit_2_per_rank[rank].item()
        letter_mean = max_sims_letter_per_rank[rank].item()
        
        digit_7_std = np.std(max_sims_digit_7_by_rank[rank], ddof=1) if len(max_sims_digit_7_by_rank[rank]) > 1 else 0.0
        digit_2_std = np.std(max_sims_digit_2_by_rank[rank], ddof=1) if len(max_sims_digit_2_by_rank[rank]) > 1 else 0.0
        letter_std = np.std(max_sims_letter_by_rank[rank], ddof=1) if len(max_sims_letter_by_rank[rank]) > 1 else 0.0
        
        print(f"{rank+1:<6} {val_sign:<6} {digit_7_mean:.3f} ± {digit_7_std:.3f} {'':<5} {digit_2_mean:.3f} ± {digit_2_std:.3f} {'':<5} {letter_mean:.3f} ± {letter_std:.3f}")
    
    # Overall mean
    mean_digit_7 = max_sims_digit_7_per_rank.mean().item()
    mean_digit_2 = max_sims_digit_2_per_rank.mean().item()
    mean_letter = max_sims_letter_per_rank.mean().item()
    
    # Std of means across seeds
    mean_digit_7_per_seed = [sim.max(dim=1)[0].mean().item() for sim in results_mnist_emnist_digit_7]
    mean_digit_2_per_seed = [sim.max(dim=1)[0].mean().item() for sim in results_mnist_emnist_digit_2]
    mean_letter_per_seed = [sim.max(dim=1)[0].mean().item() for sim in results_mnist_emnist_letter]
    
    std_digit_7 = np.std(mean_digit_7_per_seed, ddof=1) if len(mean_digit_7_per_seed) > 1 else 0.0
    std_digit_2 = np.std(mean_digit_2_per_seed, ddof=1) if len(mean_digit_2_per_seed) > 1 else 0.0
    std_letter = np.std(mean_letter_per_seed, ddof=1) if len(mean_letter_per_seed) > 1 else 0.0
    
    print("-" * 100)
    print(f"{'Mean':<6} {'':<6} {mean_digit_7:.3f} ± {std_digit_7:.3f} {'':<5} {mean_digit_2:.3f} ± {std_digit_2:.3f} {'':<5} {mean_letter:.3f} ± {std_letter:.3f}")
    print("=" * 100)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  MNIST-3 vs EMNIST-7 (digit): {mean_digit_7:.3f} ± {std_digit_7:.3f}")
    print(f"  MNIST-3 vs EMNIST-2 (digit): {mean_digit_2:.3f} ± {std_digit_2:.3f}")
    print(f"  MNIST-3 vs EMNIST-E (letter): {mean_letter:.3f} ± {std_letter:.3f}")


if __name__ == "__main__":
    main()

