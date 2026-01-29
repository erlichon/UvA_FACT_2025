#!/usr/bin/env python3
"""
Calculate absolute cosine similarity between MNIST 0 and all EMNIST classes.

Compares:
- MNIST 0 vs all EMNIST digits (0-9)
- MNIST 0 vs all EMNIST letters (A-Z, 26 letters)
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EXTENSION2_CHECKPOINTS
from src.vision.spectral import load_checkpoint_eigenvalues
from scripts.extension2.utils import (
    get_balanced_eigenvectors_and_vals,
    compute_mean_abs_cosine_similarity,
)


def main():
    print("=" * 80)
    print("MNIST-0 vs All EMNIST Classes: Absolute Cosine Similarity")
    print("=" * 80)
    print("\nComparing top 10 eigenvectors (balanced: 5 positive + 5 negative)")
    print("  - MNIST class 0 (digit 0)")
    print("  - All EMNIST digits (0-9)")
    print("  - All EMNIST letters (A-Z)")
    print()
    
    # Load checkpoints (using all seeds)
    seeds = [42, 43, 44, 45, 46]
    k = 10
    mnist_class = 0  # digit 0
    
    # Accumulate results across seeds
    # Structure: results[class_idx] = [sim_seed1, sim_seed2, ...]
    digit_results = {i: [] for i in range(10)}  # EMNIST digits 0-9
    letter_results = {i: [] for i in range(26)}  # EMNIST letters A-Z
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
            
            # Get balanced eigenvectors for MNIST 0
            mnist_0_vecs, _ = get_balanced_eigenvectors_and_vals(
                mnist_vals, mnist_vecs, mnist_class, k
            )
            
            # Compare with all EMNIST digits
            for digit_class in range(10):
                digit_vecs, _ = get_balanced_eigenvectors_and_vals(
                    digits_vals, digits_vecs, digit_class, k
                )
                sim = compute_mean_abs_cosine_similarity(mnist_0_vecs, digit_vecs)
                digit_results[digit_class].append(sim)
            
            # Compare with all EMNIST letters
            for letter_class in range(26):
                letter_vecs, _ = get_balanced_eigenvectors_and_vals(
                    letters_vals, letters_vecs, letter_class, k
                )
                sim = compute_mean_abs_cosine_similarity(mnist_0_vecs, letter_vecs)
                letter_results[letter_class].append(sim)
            
            loaded_seeds.append(seed)
            
        except Exception as e:
            print(f"  Seed {seed}: Error loading checkpoint: {e}")
            continue
    
    if len(loaded_seeds) == 0:
        print("ERROR: No checkpoints found!")
        return
    
    print(f"Loaded {len(loaded_seeds)} seeds: {loaded_seeds}\n")
    
    # Compute statistics across seeds
    def compute_stats(similarities):
        """Compute mean and std across seeds."""
        if len(similarities) == 0:
            return 0.0, 0.0
        return np.mean(similarities), np.std(similarities, ddof=1) if len(similarities) > 1 else 0.0
    
    # Process digit results
    digit_stats = {}
    for digit_class in range(10):
        mean_sim, std_sim = compute_stats(digit_results[digit_class])
        digit_stats[digit_class] = (mean_sim, std_sim)
    
    # Process letter results
    letter_stats = {}
    for letter_class in range(26):
        mean_sim, std_sim = compute_stats(letter_results[letter_class])
        letter_stats[letter_class] = (mean_sim, std_sim)
    
    # Print results
    print("=" * 80)
    print("MNIST-0 vs EMNIST DIGITS (0-9)")
    print("=" * 80)
    print(f"\n{'Digit':<8} {'Similarity':<20} {'Rank':<8}")
    print("-" * 40)
    
    # Sort by similarity
    sorted_digits = sorted(digit_stats.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (digit, (mean, std)) in enumerate(sorted_digits, 1):
        print(f"{digit:<8} {mean:.3f} ± {std:.3f} {'':<5} #{rank}")
    
    print("\n" + "=" * 80)
    print("MNIST-0 vs EMNIST LETTERS (A-Z)")
    print("=" * 80)
    print(f"\n{'Letter':<8} {'Similarity':<20} {'Rank':<8}")
    print("-" * 40)
    
    # Sort by similarity
    sorted_letters = sorted(letter_stats.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (letter_idx, (mean, std)) in enumerate(sorted_letters, 1):
        letter = chr(ord('A') + letter_idx)
        print(f"{letter:<8} {mean:.3f} ± {std:.3f} {'':<5} #{rank}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    digit_means = [mean for mean, _ in digit_stats.values()]
    letter_means = [mean for mean, _ in letter_stats.values()]
    
    print(f"\nMNIST-0 vs EMNIST Digits:")
    print(f"  Mean similarity: {np.mean(digit_means):.3f}")
    print(f"  Std similarity: {np.std(digit_means):.3f}")
    print(f"  Min similarity: {np.min(digit_means):.3f} (digit {np.argmin(digit_means)})")
    print(f"  Max similarity: {np.max(digit_means):.3f} (digit {np.argmax(digit_means)})")
    
    print(f"\nMNIST-0 vs EMNIST Letters:")
    print(f"  Mean similarity: {np.mean(letter_means):.3f}")
    print(f"  Std similarity: {np.std(letter_means):.3f}")
    min_letter_idx = np.argmin(letter_means)
    max_letter_idx = np.argmax(letter_means)
    print(f"  Min similarity: {np.min(letter_means):.3f} (letter {chr(ord('A') + min_letter_idx)})")
    print(f"  Max similarity: {np.max(letter_means):.3f} (letter {chr(ord('A') + max_letter_idx)})")
    
    # Expected pair: 0 vs O (letter O is index 14)
    o_idx = 14
    o_sim, o_std = letter_stats[o_idx]
    print(f"\nExpected pair (0 vs O): {o_sim:.3f} ± {o_std:.3f}")
    o_rank = next(rank for rank, (idx, _) in enumerate(sorted_letters, 1) if idx == o_idx)
    print(f"  Rank among all letters: #{o_rank} / 26")
    
    # Save to CSV for further analysis
    output_dir = PROJECT_ROOT / "results" / "extension2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for digits
    digit_df = pd.DataFrame([
        {"digit": d, "similarity_mean": mean, "similarity_std": std}
        for d, (mean, std) in digit_stats.items()
    ])
    digit_df = digit_df.sort_values("similarity_mean", ascending=False)
    digit_df["rank"] = range(1, len(digit_df) + 1)
    digit_df.to_csv(output_dir / "mnist0_vs_emnist_digits.csv", index=False)
    print(f"\nSaved digit results to: {output_dir / 'mnist0_vs_emnist_digits.csv'}")
    
    # Create DataFrame for letters
    letter_df = pd.DataFrame([
        {"letter": chr(ord('A') + idx), "letter_idx": idx, "similarity_mean": mean, "similarity_std": std}
        for idx, (mean, std) in letter_stats.items()
    ])
    letter_df = letter_df.sort_values("similarity_mean", ascending=False)
    letter_df["rank"] = range(1, len(letter_df) + 1)
    letter_df.to_csv(output_dir / "mnist0_vs_emnist_letters.csv", index=False)
    print(f"Saved letter results to: {output_dir / 'mnist0_vs_emnist_letters.csv'}")


if __name__ == "__main__":
    main()

