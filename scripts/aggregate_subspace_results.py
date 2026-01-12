#!/usr/bin/env python3
"""
Aggregate subspace geometry results across all 5 seeds.

Computes mean and std for subspace overlap metrics and generates
a summary report showing whether universal geometric features are learned.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all subspace test results."""
    results = []
    for seed in [42, 43, 44, 45, 46]:
        result_path = results_dir / f"subspace_seed{seed}.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            print(f"Warning: Missing results for seed {seed}")
    return results


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregate metrics across seeds."""
    
    # Extract expected pairs metrics
    pairs = ['0-O', '1-I', '2-Z', '5-S', '6-B']
    pair_metrics = {pair: {'mean_cos': [], 'grassmann': [], 'projection': []} 
                   for pair in pairs}
    
    for result in results:
        for pair in pairs:
            if pair in result['expected_pairs']:
                metrics = result['expected_pairs'][pair]
                pair_metrics[pair]['mean_cos'].append(metrics['mean_cos'])
                pair_metrics[pair]['grassmann'].append(metrics['grassmann'])
                pair_metrics[pair]['projection'].append(metrics['projection'])
    
    # Compute means and stds
    aggregated = {}
    for pair in pairs:
        if pair_metrics[pair]['mean_cos']:
            aggregated[pair] = {
                'mean_cos': {
                    'mean': np.mean(pair_metrics[pair]['mean_cos']),
                    'std': np.std(pair_metrics[pair]['mean_cos']),
                },
                'grassmann': {
                    'mean': np.mean(pair_metrics[pair]['grassmann']),
                    'std': np.std(pair_metrics[pair]['grassmann']),
                },
                'projection': {
                    'mean': np.mean(pair_metrics[pair]['projection']),
                    'std': np.std(pair_metrics[pair]['projection']),
                }
            }
    
    # Random baseline
    random_means = [r['random_baseline']['mean'] for r in results]
    random_stds = [r['random_baseline']['std'] for r in results]
    
    aggregated['random_baseline'] = {
        'mean': {
            'mean': np.mean(random_means),
            'std': np.std(random_means),
        },
        'std': {
            'mean': np.mean(random_stds),
            'std': np.std(random_stds),
        }
    }
    
    # Overall summary
    expected_means = [aggregated[pair]['mean_cos']['mean'] for pair in pairs if pair in aggregated]
    aggregated['summary'] = {
        'expected_pairs_mean_cos': {
            'mean': np.mean(expected_means),
            'std': np.std(expected_means),
        },
        'random_baseline_mean': aggregated['random_baseline']['mean']['mean'],
        'ratio': np.mean(expected_means) / aggregated['random_baseline']['mean']['mean'],
        'n_seeds': len(results),
    }
    
    return aggregated


def print_report(aggregated: Dict):
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("SUBSPACE GEOMETRY TEST: AGGREGATED RESULTS (5 SEEDS)")
    print("=" * 70)
    
    print("\nExpected Shape-Similar Pairs (Mean ± Std):")
    print("-" * 70)
    pairs = ['0-O', '1-I', '2-Z', '5-S', '6-B']
    pair_names = {
        '0-O': 'Digit 0 ↔ Letter O',
        '1-I': 'Digit 1 ↔ Letter I',
        '2-Z': 'Digit 2 ↔ Letter Z',
        '5-S': 'Digit 5 ↔ Letter S',
        '6-B': 'Digit 6 ↔ Letter B',
    }
    
    for pair in pairs:
        if pair in aggregated:
            metrics = aggregated[pair]
            print(f"\n{pair_names[pair]:20s}")
            print(f"  Mean Cosine:    {metrics['mean_cos']['mean']:.4f} ± {metrics['mean_cos']['std']:.4f}")
            print(f"  Grassmann:      {metrics['grassmann']['mean']:.4f} ± {metrics['grassmann']['std']:.4f}")
            print(f"  Projection:     {metrics['projection']['mean']:.4f} ± {metrics['projection']['std']:.4f}")
    
    print("\n" + "-" * 70)
    print(f"Random Baseline (control):")
    rb = aggregated['random_baseline']
    print(f"  Mean: {rb['mean']['mean']:.4f} ± {rb['mean']['std']:.4f}")
    
    summary = aggregated['summary']
    print("\n" + "-" * 70)
    print(f"Overall Summary:")
    print(f"  Expected pairs mean:   {summary['expected_pairs_mean_cos']['mean']:.4f} ± {summary['expected_pairs_mean_cos']['std']:.4f}")
    print(f"  Random baseline:       {summary['random_baseline_mean']:.4f}")
    print(f"  Ratio (expected/random): {summary['ratio']:.2f}x")
    print(f"  Number of seeds:       {summary['n_seeds']}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if summary['ratio'] > 1.5:
        print("✓ Strong evidence for universal geometric feature learning!")
        print("  Eigenvector subspaces for similar shapes overlap significantly")
        print("  more than random pairs, proving learned mechanisms are universal.")
    elif summary['ratio'] > 1.2:
        print("○ Moderate evidence for universal geometric feature learning.")
        print("  Some overlap detected, but effect is modest.")
    else:
        print("✗ No strong evidence for universal geometric feature learning.")
        print("  Expected pairs overlap is similar to random baseline.")
    print("=" * 70)


def main():
    results_dir = Path("results/extension2/subspace")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run: bash scripts/run_subspace_test_all_seeds.sh first")
        return
    
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    
    if len(results) < 5:
        print(f"Warning: Only {len(results)}/5 seed results found")
        if len(results) == 0:
            print("No results to aggregate. Exiting.")
            return
    
    aggregated = aggregate_metrics(results)
    
    # Save aggregated results
    output_path = results_dir / "aggregated_results.json"
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {output_path}")
    
    # Print report
    print_report(aggregated)


if __name__ == "__main__":
    main()
