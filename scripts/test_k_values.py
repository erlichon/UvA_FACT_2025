"""
Test different k values for eigenvector subspace comparison.

This script helps determine the optimal number of top eigenvectors (k)
to use for cross-dataset subspace comparison in Extension 2.

Usage:
    python scripts/test_k_values.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.subspace import compute_subspace_overlap, sort_eigenvectors_by_magnitude
from src.utils import get_device, setup_mps_fallbacks, is_mps_device


def load_model_eigenvectors(checkpoint_path, device):
    """Load model and extract eigenvectors."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get eigenvalues and eigenvectors from checkpoint
    if 'eigenvalues' in ckpt and 'eigenvectors' in ckpt:
        eigenvalues = ckpt['eigenvalues'].to(device)
        eigenvectors = ckpt['eigenvectors'].to(device)
    else:
        raise ValueError(f"Checkpoint missing eigenvalues/eigenvectors: {checkpoint_path}")
    
    # Sort by magnitude
    sorted_vecs = sort_eigenvectors_by_magnitude(eigenvalues, eigenvectors)
    
    return sorted_vecs


def compute_overlap_for_k(mnist_vecs, emnist_vecs, k, method='mean_cos'):
    """Compute overlap for expected and random pairs with given k."""
    
    # Expected pairs: (MNIST digit, EMNIST letter, mnist_idx, emnist_idx)
    expected_pairs = [
        ('0', 'O', 0, 14),
        ('1', 'I', 1, 8),
        ('2', 'Z', 2, 25),
        ('5', 'S', 5, 18),
        ('6', 'B', 6, 1),
    ]
    
    # Compute overlaps for expected pairs
    expected_overlaps = []
    for _, _, mnist_idx, emnist_idx in expected_pairs:
        overlap = compute_subspace_overlap(
            mnist_vecs[mnist_idx],
            emnist_vecs[emnist_idx],
            k=k,
            method=method
        )
        expected_overlaps.append(overlap)
    
    # Compute overlaps for random pairs (all non-expected combinations)
    random_overlaps = []
    for mnist_idx in range(10):
        for emnist_idx in range(26):
            # Skip expected pairs
            is_expected = any(
                m_idx == mnist_idx and e_idx == emnist_idx
                for _, _, m_idx, e_idx in expected_pairs
            )
            if not is_expected:
                overlap = compute_subspace_overlap(
                    mnist_vecs[mnist_idx],
                    emnist_vecs[emnist_idx],
                    k=k,
                    method=method
                )
                random_overlaps.append(overlap)
    
    expected_mean = np.mean(expected_overlaps)
    random_mean = np.mean(random_overlaps)
    random_std = np.std(random_overlaps)
    ratio = expected_mean / (random_mean + 1e-10)
    
    return {
        'expected_mean': expected_mean,
        'expected_overlaps': expected_overlaps,
        'random_mean': random_mean,
        'random_std': random_std,
        'random_overlaps': random_overlaps,
        'ratio': ratio,
    }


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Load checkpoints
    checkpoint_dir = PROJECT_ROOT / "results/extension2/checkpoints"
    mnist_path = checkpoint_dir / "mnist_regularized_seed42.pt"
    emnist_path = checkpoint_dir / "emnist_letters_regularized_seed42.pt"
    
    if not mnist_path.exists() or not emnist_path.exists():
        print("ERROR: Checkpoints not found!")
        print(f"  Looking for: {mnist_path}")
        print(f"  Looking for: {emnist_path}")
        return
    
    # Load eigenvectors
    print("\nLoading models...")
    mnist_vecs = load_model_eigenvectors(mnist_path, device)
    emnist_vecs = load_model_eigenvectors(emnist_path, device)
    
    print(f"MNIST eigenvectors shape: {mnist_vecs.shape}")
    print(f"EMNIST eigenvectors shape: {emnist_vecs.shape}")
    
    # Test different k values
    k_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    methods = ['mean_cos', 'grassmann', 'projection']
    
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT K VALUES")
    print("=" * 80)
    
    results = {}
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        print(f"{'k':<5} {'Expected':<12} {'Random':<12} {'Ratio':<8} {'Separation'}")
        print("-" * 60)
        
        method_results = []
        
        for k in k_values:
            result = compute_overlap_for_k(mnist_vecs, emnist_vecs, k, method=method)
            
            # Calculate separation (how many std devs above random mean)
            separation = (result['expected_mean'] - result['random_mean']) / (result['random_std'] + 1e-10)
            
            method_results.append({
                'k': k,
                'expected_mean': result['expected_mean'],
                'random_mean': result['random_mean'],
                'random_std': result['random_std'],
                'ratio': result['ratio'],
                'separation': separation,
            })
            
            print(f"{k:<5} {result['expected_mean']:<12.4f} "
                  f"{result['random_mean']:<12.4f} "
                  f"{result['ratio']:<8.2f} {separation:+.2f}σ")
        
        results[method] = method_results
    
    # Find best k for each method
    print("\n" + "=" * 80)
    print("BEST K VALUES BY METHOD")
    print("=" * 80)
    
    for method in methods:
        method_results = results[method]
        
        # Find k with highest ratio
        best_by_ratio = max(method_results, key=lambda x: x['ratio'])
        
        # Find k with highest separation
        best_by_separation = max(method_results, key=lambda x: x['separation'])
        
        print(f"\n{method}:")
        print(f"  Best ratio:      k={best_by_ratio['k']:<3} (ratio={best_by_ratio['ratio']:.3f})")
        print(f"  Best separation: k={best_by_separation['k']:<3} (sep={best_by_separation['separation']:+.2f}σ)")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        method_results = results[method]
        
        k_vals = [r['k'] for r in method_results]
        ratios = [r['ratio'] for r in method_results]
        separations = [r['separation'] for r in method_results]
        
        # Plot ratio
        ax.plot(k_vals, ratios, 'o-', linewidth=2, markersize=8, label='Expected/Random Ratio')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0)')
        
        # Highlight best k
        best_k = max(method_results, key=lambda x: x['ratio'])
        ax.plot(best_k['k'], best_k['ratio'], 'r*', markersize=20, 
                label=f"Best k={best_k['k']}")
        
        ax.set_xlabel('k (number of eigenvectors)', fontsize=12)
        ax.set_ylabel('Expected/Random Ratio', fontsize=12)
        ax.set_title(f'{method.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add separation as secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(k_vals, separations, 's--', color='green', alpha=0.5, 
                 linewidth=1.5, markersize=6, label='Separation (σ)')
        ax2.set_ylabel('Separation (std devs)', fontsize=11, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = PROJECT_ROOT / "results/extension2"
    output_path = output_dir / "k_value_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save to Report/figures
    report_path = PROJECT_ROOT / "Report/figures/extension2_k_analysis.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"Also saved to: {report_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nRecommendations:")
    print("1. Choose k with highest ratio for discriminative power")
    print("2. Consider k with highest separation for statistical significance")
    print("3. Balance between interpretability (lower k) and robustness (higher k)")


if __name__ == "__main__":
    main()
