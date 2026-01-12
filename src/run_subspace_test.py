#!/usr/bin/env python3
"""
Run subspace geometry test for Extension 2.

Compares eigenvector subspaces between MNIST and EMNIST models
to test whether learned mechanisms are universal across datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add original code to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "bilinear-decomposition-main"))

from src.evaluate_extension2 import run_subspace_geometry_test


def main():
    parser = argparse.ArgumentParser(
        description="Run subspace geometry test between MNIST and EMNIST models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input checkpoints
    parser.add_argument("--mnist-checkpoint", type=str, required=True,
                       help="Path to MNIST checkpoint")
    parser.add_argument("--emnist-checkpoint", type=str, required=True,
                       help="Path to EMNIST checkpoint")
    
    # Parameters
    parser.add_argument("--k", type=int, default=10,
                       help="Number of top eigenvectors to compare")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Load checkpoints
    print(f"Loading MNIST checkpoint: {args.mnist_checkpoint}")
    mnist_ckpt = torch.load(args.mnist_checkpoint, map_location='cpu', weights_only=False)
    
    print(f"Loading EMNIST checkpoint: {args.emnist_checkpoint}")
    emnist_ckpt = torch.load(args.emnist_checkpoint, map_location='cpu', weights_only=False)
    
    # Extract eigenvalues and eigenvectors
    mnist_vals = mnist_ckpt['eigenvalues']
    mnist_vecs = mnist_ckpt['eigenvectors']
    emnist_vals = emnist_ckpt['eigenvalues']
    emnist_vecs = emnist_ckpt['eigenvectors']
    
    print(f"MNIST eigenvalues shape: {mnist_vals.shape}")
    print(f"EMNIST eigenvalues shape: {emnist_vals.shape}")
    print()
    
    # Run subspace geometry test
    results = run_subspace_geometry_test(
        mnist_eigenvalues=mnist_vals,
        mnist_eigenvectors=mnist_vecs,
        emnist_eigenvalues=emnist_vals,
        emnist_eigenvectors=emnist_vecs,
        k=args.k,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Expected pairs mean overlap: {results['summary']['expected_mean']:.4f}")
    print(f"  Random baseline: {results['random_baseline']['mean']:.4f}")
    print(f"  Ratio: {results['summary']['ratio']:.2f}x")
    print(f"  {results['summary']['conclusion']}")


if __name__ == "__main__":
    main()
