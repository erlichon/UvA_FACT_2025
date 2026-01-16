#!/usr/bin/env python3
"""
Aggregate subspace geometry results across all 5 seeds.

Computes mean and std for subspace overlap metrics and generates
a summary report showing whether universal geometric features are learned.

Usage:
    python scripts/extension2/aggregate_results.py
    
Or via runner:
    ./scripts/train/run_extension2.sh aggregate
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_subspace_results(
    results_dir: Path,
    seeds: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Load subspace test results from JSON files.
    
    Args:
        results_dir: Directory containing subspace_seed{N}.json files
        seeds: List of seeds to load (default: [42, 43, 44, 45, 46])
        
    Returns:
        List of result dictionaries, one per seed
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    
    results = []
    for seed in seeds:
        result_path = results_dir / f"subspace_seed{seed}.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            print(f"Warning: Missing results for seed {seed}")
    return results


def aggregate_subspace_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate subspace overlap metrics across seeds.
    
    Computes mean and std for each metric across all seeds.
    
    Args:
        results: List of result dictionaries from load_subspace_results()
        
    Returns:
        Aggregated metrics dictionary with mean/std for each pair
    """
    # Expected digit-letter pairs based on visual similarity
    pairs = ['0-O', '1-I', '2-Z', '5-S', '6-B']
    pair_metrics = {pair: {'mean_cos': [], 'grassmann': [], 'projection': []} 
                   for pair in pairs}
    
    for result in results:
        for pair in pairs:
            if pair in result.get('expected_pairs', {}):
                metrics = result['expected_pairs'][pair]
                pair_metrics[pair]['mean_cos'].append(metrics['mean_cos'])
                pair_metrics[pair]['grassmann'].append(metrics['grassmann'])
                pair_metrics[pair]['projection'].append(metrics['projection'])
    
    # Compute means and stds for each pair
    aggregated = {}
    for pair in pairs:
        if pair_metrics[pair]['mean_cos']:
            aggregated[pair] = {
                'mean_cos': {
                    'mean': float(np.mean(pair_metrics[pair]['mean_cos'])),
                    'std': float(np.std(pair_metrics[pair]['mean_cos'])),
                },
                'grassmann': {
                    'mean': float(np.mean(pair_metrics[pair]['grassmann'])),
                    'std': float(np.std(pair_metrics[pair]['grassmann'])),
                },
                'projection': {
                    'mean': float(np.mean(pair_metrics[pair]['projection'])),
                    'std': float(np.std(pair_metrics[pair]['projection'])),
                }
            }
    
    # Random baseline (control group)
    random_means = [r['random_baseline']['mean'] for r in results if 'random_baseline' in r]
    random_stds = [r['random_baseline']['std'] for r in results if 'random_baseline' in r]
    
    if random_means:
        aggregated['random_baseline'] = {
            'mean': {
                'mean': float(np.mean(random_means)),
                'std': float(np.std(random_means)),
            },
            'std': {
                'mean': float(np.mean(random_stds)),
                'std': float(np.std(random_stds)),
            }
        }
    
    # Overall summary
    expected_means = [aggregated[pair]['mean_cos']['mean'] for pair in pairs if pair in aggregated]
    if expected_means and 'random_baseline' in aggregated:
        aggregated['summary'] = {
            'expected_pairs_mean_cos': {
                'mean': float(np.mean(expected_means)),
                'std': float(np.std(expected_means)),
            },
            'random_baseline_mean': aggregated['random_baseline']['mean']['mean'],
            'ratio': float(np.mean(expected_means) / aggregated['random_baseline']['mean']['mean']),
            'n_seeds': len(results),
        }
    
    return aggregated


def print_report(aggregated: Dict) -> None:
    """Print human-readable report of aggregated results."""
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
    if 'random_baseline' in aggregated:
        print(f"Random Baseline (control):")
        rb = aggregated['random_baseline']
        print(f"  Mean: {rb['mean']['mean']:.4f} ± {rb['mean']['std']:.4f}")
    
    if 'summary' in aggregated:
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
            print("  Strong evidence for universal geometric feature learning!")
            print("  Eigenvector subspaces for similar shapes overlap significantly")
            print("  more than random pairs, proving learned mechanisms are universal.")
        elif summary['ratio'] > 1.2:
            print("  Moderate evidence for universal geometric feature learning.")
            print("  Some overlap detected, but effect is modest.")
        else:
            print("  No strong evidence for universal geometric feature learning.")
            print("  Expected pairs overlap is similar to random baseline.")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate subspace geometry results")
    parser.add_argument("--results-dir", type=str, default="results/extension2/subspace",
                       help="Directory containing subspace_seed{N}.json files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for aggregated results (default: {results-dir}/aggregated_results.json)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run: ./scripts/train/run_extension2.sh eval subspace first")
        return
    
    print(f"Loading results from: {results_dir}")
    results = load_subspace_results(results_dir)
    
    if len(results) < 5:
        print(f"Warning: Only {len(results)}/5 seed results found")
        if len(results) == 0:
            print("No results to aggregate. Exiting.")
            return
    
    aggregated = aggregate_subspace_metrics(results)
    
    # Save aggregated results
    output_path = Path(args.output) if args.output else results_dir / "aggregated_results.json"
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {output_path}")
    
    # Print report
    print_report(aggregated)
    
    # Return aggregated dict for notebook compatibility
    return aggregated


if __name__ == "__main__":
    main()
