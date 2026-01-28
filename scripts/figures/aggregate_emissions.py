#!/usr/bin/env python3
"""
Aggregate emissions data from all experiments for the report.

This script collects emissions data from:
- Vision checkpoints (checkpoints/vision/**/*.pt)
- Extension 2 checkpoints (checkpoints/extension2/*.pt)
- Extension CP checkpoints (checkpoints/extension_cp/*.pt)
- Language JSON results (results/language/*.json)

And produces a summary JSON file at results/emissions_summary.json with:
- Total emissions across all experiments
- Per-experiment-type breakdowns
- Data formatted for report Table 1

Usage:
    python scripts/figures/aggregate_emissions.py
    python scripts/figures/aggregate_emissions.py --output custom_path.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def extract_checkpoint_emissions(path: Path) -> dict | None:
    """Extract emissions data from a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'emissions' in ckpt and ckpt['emissions'] is not None:
            return {
                'file': str(path.relative_to(PROJECT_ROOT)),
                'co2_kg': ckpt['emissions'].get('co2_kg', 0),
                'wall_time_hours': ckpt['emissions'].get('wall_time_hours', 0),
                'gpu_hours': ckpt['emissions'].get('gpu_hours', 0),
                'config': ckpt.get('config', {}),
            }
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
    return None


def extract_json_emissions(path: Path) -> dict | None:
    """Extract emissions data from a JSON result file."""
    try:
        with open(path) as f:
            data = json.load(f)
        
        # Check top-level
        if 'co2_kg' in data:
            return {
                'file': str(path.relative_to(PROJECT_ROOT)),
                'co2_kg': data.get('co2_kg', 0),
                'wall_time_hours': data.get('wall_time_hours', 0),
                'wall_time_seconds': data.get('wall_time_seconds', 0),
                'gpu_hours': data.get('gpu_hours', 0),
            }
        
        # Check in 'emissions' dict
        if 'emissions' in data and isinstance(data['emissions'], dict):
            return {
                'file': str(path.relative_to(PROJECT_ROOT)),
                'co2_kg': data['emissions'].get('co2_kg', 0),
                'wall_time_hours': data['emissions'].get('wall_time_hours', 0),
                'gpu_hours': data['emissions'].get('gpu_hours', 0),
            }
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
    return None


def aggregate_category(emissions_list: list) -> dict:
    """Aggregate emissions from a list of emission dicts."""
    if not emissions_list:
        return {
            'total_co2_kg': 0,
            'total_wall_time_hours': 0,
            'total_gpu_hours': 0,
            'n_experiments': 0,
        }
    
    return {
        'total_co2_kg': sum(e.get('co2_kg', 0) or 0 for e in emissions_list),
        'total_wall_time_hours': sum(e.get('wall_time_hours', 0) or 0 for e in emissions_list),
        'total_gpu_hours': sum(e.get('gpu_hours', 0) or 0 for e in emissions_list),
        'n_experiments': len(emissions_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate emissions data for report")
    parser.add_argument('--output', type=str, default='results/emissions_summary.json',
                        help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    args = parser.parse_args()
    
    print("=" * 70)
    print("EMISSIONS DATA AGGREGATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Collect all emissions data
    emissions_by_category = defaultdict(list)
    
    # ========================================================================
    # VISION CHECKPOINTS
    # ========================================================================
    print(">>> Collecting vision checkpoint emissions...")
    
    vision_categories = {
        'vision_mnist_base': PROJECT_ROOT / "checkpoints/vision/mnist",
        'vision_fashion': PROJECT_ROOT / "checkpoints/vision/fashion",
        'vision_noise_sweep': PROJECT_ROOT / "checkpoints/vision/noise_sweep",
        'vision_size_sweep': PROJECT_ROOT / "checkpoints/vision/size_sweep",
        'vision_challenge': PROJECT_ROOT / "checkpoints/vision/challenge",
        'vision_adversarial': PROJECT_ROOT / "checkpoints/vision/adversarial",
    }
    
    for category, ckpt_dir in vision_categories.items():
        if ckpt_dir.exists():
            for pt_file in ckpt_dir.glob('*.pt'):
                emissions = extract_checkpoint_emissions(pt_file)
                if emissions:
                    emissions_by_category[category].append(emissions)
                    if args.verbose:
                        print(f"  {pt_file.name}: {emissions['co2_kg']:.6f} kg CO2")
    
    # ========================================================================
    # EXTENSION 2 CHECKPOINTS
    # ========================================================================
    print("\n>>> Collecting extension 2 checkpoint emissions...")
    
    ext2_dir = PROJECT_ROOT / "checkpoints/extension2"
    if ext2_dir.exists():
        for pt_file in ext2_dir.glob('*.pt'):
            emissions = extract_checkpoint_emissions(pt_file)
            if emissions:
                emissions_by_category['extension2'].append(emissions)
                if args.verbose:
                    print(f"  {pt_file.name}: {emissions['co2_kg']:.6f} kg CO2")
    
    # ========================================================================
    # EXTENSION CP CHECKPOINTS
    # ========================================================================
    print("\n>>> Collecting extension CP checkpoint emissions...")
    
    ext_cp_dir = PROJECT_ROOT / "checkpoints/extension_cp"
    if ext_cp_dir.exists():
        for pt_file in ext_cp_dir.glob('*.pt'):
            emissions = extract_checkpoint_emissions(pt_file)
            if emissions:
                emissions_by_category['extension_cp'].append(emissions)
                if args.verbose:
                    print(f"  {pt_file.name}: {emissions['co2_kg']:.6f} kg CO2")
    
    # ========================================================================
    # LANGUAGE JSON RESULTS
    # ========================================================================
    print("\n>>> Collecting language experiment emissions...")
    
    lang_dir = PROJECT_ROOT / "results/language"
    if lang_dir.exists():
        # Figure 9 - correlation results
        for pattern in ['correlation_*.json']:
            for json_file in lang_dir.glob(pattern):
                emissions = extract_json_emissions(json_file)
                if emissions:
                    emissions_by_category['language_figure9'].append(emissions)
                    if args.verbose:
                        print(f"  {json_file.name}: {emissions['co2_kg']:.6f} kg CO2")
        
        # Figure 8 - circuit search
        circuit_file = lang_dir / "circuit_search_complete.json"
        if circuit_file.exists():
            emissions = extract_json_emissions(circuit_file)
            if emissions:
                emissions_by_category['language_figure8'].append(emissions)
                if args.verbose:
                    print(f"  {circuit_file.name}: {emissions['co2_kg']:.6f} kg CO2")
        
        # Figure 10 - SAE training time
        sae_file = lang_dir / "sae_training_time_comparison.json"
        if sae_file.exists():
            emissions = extract_json_emissions(sae_file)
            if emissions:
                emissions_by_category['language_figure10'].append(emissions)
                if args.verbose:
                    print(f"  {sae_file.name}: {emissions['co2_kg']:.6f} kg CO2")
        
        # Negation analysis
        neg_file = lang_dir / "negation_analysis.json"
        if neg_file.exists():
            emissions = extract_json_emissions(neg_file)
            if emissions:
                emissions_by_category['language_negation'].append(emissions)
                if args.verbose:
                    print(f"  {neg_file.name}: {emissions['co2_kg']:.6f} kg CO2")
        
        # Interaction analysis
        int_file = lang_dir / "interaction_analysis.json"
        if int_file.exists():
            emissions = extract_json_emissions(int_file)
            if emissions:
                emissions_by_category['language_interaction'].append(emissions)
                if args.verbose:
                    print(f"  {int_file.name}: {emissions['co2_kg']:.6f} kg CO2")
    
    # ========================================================================
    # LANGUAGE EIGENPAIRS PRECOMPUTATION
    # ========================================================================
    print("\n>>> Collecting eigenpairs precomputation emissions...")
    
    eigenpairs_dir = PROJECT_ROOT / "results/language/eigenpairs"
    if eigenpairs_dir.exists():
        for model_dir in eigenpairs_dir.iterdir():
            if model_dir.is_dir():
                # Only count eigenpairs precompute for fw-medium (used in Figure 9)
                if model_dir.name != "fw-medium":
                    continue
                for layer_dir in model_dir.iterdir():
                    if layer_dir.is_dir():
                        manifest_file = layer_dir / "manifest.json"
                        if manifest_file.exists():
                            emissions = extract_json_emissions(manifest_file)
                            if emissions:
                                emissions_by_category['language_eigenpairs'].append(emissions)
                                if args.verbose:
                                    print(f"  {model_dir.name}/layer{layer_dir.name}: {emissions['co2_kg']:.6f} kg CO2")
    
    # ========================================================================
    # AGGREGATE BY CATEGORY
    # ========================================================================
    print("\n>>> Aggregating by category...")
    
    aggregated = {}
    for category, emissions_list in emissions_by_category.items():
        aggregated[category] = aggregate_category(emissions_list)
    
    # ========================================================================
    # COMPUTE TOTALS BY SECTION
    # ========================================================================
    
    # Vision total (all vision experiments)
    vision_categories_list = [k for k in aggregated.keys() if k.startswith('vision_')]
    vision_total = {
        'total_co2_kg': sum(aggregated[k]['total_co2_kg'] for k in vision_categories_list),
        'total_wall_time_hours': sum(aggregated[k]['total_wall_time_hours'] for k in vision_categories_list),
        'total_gpu_hours': sum(aggregated[k]['total_gpu_hours'] for k in vision_categories_list),
        'n_experiments': sum(aggregated[k]['n_experiments'] for k in vision_categories_list),
    }
    
    # Language total
    language_categories_list = [k for k in aggregated.keys() if k.startswith('language_')]
    language_total = {
        'total_co2_kg': sum(aggregated[k]['total_co2_kg'] for k in language_categories_list),
        'total_wall_time_hours': sum(aggregated[k]['total_wall_time_hours'] for k in language_categories_list),
        'total_gpu_hours': sum(aggregated[k]['total_gpu_hours'] for k in language_categories_list),
        'n_experiments': sum(aggregated[k]['n_experiments'] for k in language_categories_list),
    }
    
    # Extension 2 total
    ext2_total = aggregated.get('extension2', aggregate_category([]))
    
    # Extension CP total
    ext_cp_total = aggregated.get('extension_cp', aggregate_category([]))
    
    # Grand total
    grand_total = {
        'total_co2_kg': vision_total['total_co2_kg'] + language_total['total_co2_kg'] + 
                        ext2_total['total_co2_kg'] + ext_cp_total['total_co2_kg'],
        'total_wall_time_hours': vision_total['total_wall_time_hours'] + language_total['total_wall_time_hours'] +
                                  ext2_total['total_wall_time_hours'] + ext_cp_total['total_wall_time_hours'],
        'total_gpu_hours': vision_total['total_gpu_hours'] + language_total['total_gpu_hours'] +
                           ext2_total['total_gpu_hours'] + ext_cp_total['total_gpu_hours'],
        'n_experiments': vision_total['n_experiments'] + language_total['n_experiments'] +
                         ext2_total['n_experiments'] + ext_cp_total['n_experiments'],
    }
    
    # ========================================================================
    # BUILD SUMMARY
    # ========================================================================
    
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(PROJECT_ROOT),
        },
        'by_category': aggregated,
        'by_section': {
            'vision': vision_total,
            'language': language_total,
            'extension2': ext2_total,
            'extension_cp': ext_cp_total,
        },
        'grand_total': grand_total,
        # For report Table 1 (formatted for easy copy-paste)
        'report_table': {
            'Vision (Section 4)': {
                'experiments': vision_total['n_experiments'],
                'wall_time_hours': round(vision_total['total_wall_time_hours'], 2),
                'gpu_hours': round(vision_total['total_gpu_hours'], 3),
                'co2_kg': round(vision_total['total_co2_kg'], 6),
            },
            'Language (Section 5)': {
                'experiments': language_total['n_experiments'],
                'wall_time_hours': round(language_total['total_wall_time_hours'], 2),
                'gpu_hours': round(language_total['total_gpu_hours'], 3),
                'co2_kg': round(language_total['total_co2_kg'], 6),
            },
            'Extension 2 (Cross-Dataset)': {
                'experiments': ext2_total['n_experiments'],
                'wall_time_hours': round(ext2_total['total_wall_time_hours'], 2),
                'gpu_hours': round(ext2_total['total_gpu_hours'], 3),
                'co2_kg': round(ext2_total['total_co2_kg'], 6),
            },
            'Extension 3 (CP-Decomposition)': {
                'experiments': ext_cp_total['n_experiments'],
                'wall_time_hours': round(ext_cp_total['total_wall_time_hours'], 2),
                'gpu_hours': round(ext_cp_total['total_gpu_hours'], 3),
                'co2_kg': round(ext_cp_total['total_co2_kg'], 6),
            },
            'TOTAL': {
                'experiments': grand_total['n_experiments'],
                'wall_time_hours': round(grand_total['total_wall_time_hours'], 2),
                'gpu_hours': round(grand_total['total_gpu_hours'], 3),
                'co2_kg': round(grand_total['total_co2_kg'], 6),
            },
        },
    }
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EMISSIONS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Section':<35} {'Runs':>8} {'Hours':>10} {'GPU-h':>10} {'CO2 (kg)':>12}")
    print("-" * 75)
    
    for section, data in summary['report_table'].items():
        print(f"{section:<35} {data['experiments']:>8} {data['wall_time_hours']:>10.2f} "
              f"{data['gpu_hours']:>10.3f} {data['co2_kg']:>12.6f}")
    
    print("-" * 75)
    
    print(f"\nResults saved to: {output_path}")
    print("\nTo update the report:")
    print("  1. Copy values from results/emissions_summary.json")
    print("  2. Update Report/sections/06_fact_discussion.tex Table 1")


if __name__ == "__main__":
    main()
