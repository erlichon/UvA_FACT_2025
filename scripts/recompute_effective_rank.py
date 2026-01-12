#!/usr/bin/env python3
"""
Recompute effective rank for all checkpoints using the corrected formula.

This script:
1. Loads all checkpoints from results/phase1/ and results/phase1_fashion/
2. Recomputes effective rank using the corrected (L1/L2)^2 formula
3. Updates checkpoints with new metrics
4. Prints comparison table (old vs new values)

Usage:
    python scripts/recompute_effective_rank.py
    python scripts/recompute_effective_rank.py --dry-run  # Don't save changes
"""

import sys
from pathlib import Path
import argparse
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spectral import effective_rank, effective_rank_entropy


def recompute_checkpoint(checkpoint_path: Path, dry_run: bool = False) -> dict:
    """Recompute effective rank for a single checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    eigenvalues = checkpoint.get('eigenvalues')
    if eigenvalues is None:
        return None

    # Compute both formulas for comparison
    old_eff_rank = effective_rank_entropy(eigenvalues).mean().item()
    new_eff_rank = effective_rank(eigenvalues).mean().item()

    # Get original stored value (may differ from recomputed old)
    stored_eff_rank = checkpoint.get('metrics', {}).get('effective_rank', old_eff_rank)

    result = {
        'path': checkpoint_path.name,
        'config': checkpoint.get('config', {}).get('name', 'unknown'),
        'stored_eff_rank': stored_eff_rank,
        'old_formula': old_eff_rank,
        'new_formula': new_eff_rank,
        'ratio': new_eff_rank / old_eff_rank if old_eff_rank > 0 else 0,
    }

    # Update checkpoint with new effective rank
    if not dry_run:
        if 'metrics' not in checkpoint:
            checkpoint['metrics'] = {}
        checkpoint['metrics']['effective_rank'] = new_eff_rank
        checkpoint['metrics']['effective_rank_entropy'] = old_eff_rank  # Keep old for reference
        torch.save(checkpoint, checkpoint_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Recompute effective rank for checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes")
    args = parser.parse_args()

    # Find all checkpoint directories
    checkpoint_dirs = [
        PROJECT_ROOT / "results" / "phase1" / "checkpoints",
        PROJECT_ROOT / "results" / "phase1_fashion" / "checkpoints",
    ]

    results = []

    for ckpt_dir in checkpoint_dirs:
        if not ckpt_dir.exists():
            print(f"Skipping {ckpt_dir} (not found)")
            continue

        print(f"\nProcessing {ckpt_dir}...")

        for ckpt_path in sorted(ckpt_dir.glob("*.pt")):
            result = recompute_checkpoint(ckpt_path, dry_run=args.dry_run)
            if result:
                results.append(result)
                print(f"  {result['path']}: {result['old_formula']:.2f} -> {result['new_formula']:.2f}")

    if not results:
        print("No checkpoints found!")
        return

    # Print summary table
    print("\n" + "=" * 80)
    print("EFFECTIVE RANK COMPARISON (Old Entropy-based vs New Ratio-based)")
    print("=" * 80)
    print(f"{'Checkpoint':<45} {'Old':<10} {'New':<10} {'Ratio':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['path']:<45} {r['old_formula']:<10.2f} {r['new_formula']:<10.2f} {r['ratio']:<10.2f}")

    # Aggregate by config type
    print("\n" + "=" * 80)
    print("AGGREGATED BY CONFIG")
    print("=" * 80)

    from collections import defaultdict
    by_config = defaultdict(list)

    for r in results:
        # Extract config name from filename
        name = r['path'].replace('.pt', '')
        # Remove seed suffix
        parts = name.rsplit('_seed', 1)
        config_name = parts[0] if len(parts) > 1 else name
        by_config[config_name].append(r)

    print(f"{'Config':<30} {'Old Mean':<12} {'New Mean':<12} {'New Std':<12}")
    print("-" * 70)

    for config_name, config_results in sorted(by_config.items()):
        old_mean = sum(r['old_formula'] for r in config_results) / len(config_results)
        new_mean = sum(r['new_formula'] for r in config_results) / len(config_results)
        new_vals = [r['new_formula'] for r in config_results]
        new_std = (sum((v - new_mean)**2 for v in new_vals) / len(new_vals)) ** 0.5
        print(f"{config_name:<30} {old_mean:<12.2f} {new_mean:<12.2f} {new_std:<12.2f}")

    if args.dry_run:
        print("\n[DRY RUN] No changes saved. Run without --dry-run to update checkpoints.")
    else:
        print(f"\nUpdated {len(results)} checkpoints with new effective rank values.")


if __name__ == "__main__":
    main()
