#!/usr/bin/env python3
"""
Validate all experiment outputs have required emissions data.

This script checks:
- Vision checkpoints (checkpoints/vision/**/*.pt)
- Extension 2 checkpoints (checkpoints/extension_cross_dataset/*.pt)
- Extension CP checkpoints (checkpoints/extension_cp/*.pt)
- Language JSON results (results/language/*.json)

Each output must contain emissions data (co2_kg, wall_time_hours) for the report.

Usage:
    python scripts/figures/validate_emissions.py
    python scripts/figures/validate_emissions.py --strict  # Fail on any missing file
    python scripts/figures/validate_emissions.py --verbose  # Show all files checked
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def check_checkpoint(path: Path) -> tuple[bool, str | None]:
    """
    Check that a vision/extension checkpoint has emissions data.
    
    Args:
        path: Path to .pt checkpoint file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'emissions' not in ckpt:
            return False, "Missing 'emissions' field"
        
        if ckpt['emissions'] is None:
            return False, "'emissions' field is None"
        
        required_keys = ['co2_kg', 'wall_time_hours']
        for key in required_keys:
            if key not in ckpt['emissions']:
                return False, f"Missing 'emissions.{key}'"
            if ckpt['emissions'][key] is None:
                return False, f"'emissions.{key}' is None"
        
        return True, None
        
    except Exception as e:
        return False, f"Failed to load: {e}"


def check_json(path: Path, required_keys: list[str] = None) -> tuple[bool, str | None]:
    """
    Check that a language JSON result has emissions data.
    
    Args:
        path: Path to JSON file
        required_keys: Keys to check for (default: ['co2_kg'])
        
    Returns:
        Tuple of (success, error_message)
    """
    if required_keys is None:
        required_keys = ['co2_kg']
    
    try:
        with open(path) as f:
            data = json.load(f)
        
        for key in required_keys:
            # Check top-level
            if key in data:
                continue
            # Check in 'emissions' dict
            if 'emissions' in data and isinstance(data['emissions'], dict):
                if key in data['emissions']:
                    continue
            # Check in 'summary' dict (for interaction_analysis.json)
            if 'summary' in data and isinstance(data['summary'], dict):
                if key in data['summary']:
                    continue
            return False, f"Missing '{key}'"
        
        return True, None
        
    except Exception as e:
        return False, f"Failed to load: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate emissions data in experiment outputs")
    parser.add_argument('--strict', action='store_true', 
                        help='Fail if any expected file is missing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all files checked')
    parser.add_argument('--sample', type=int, default=0,
                        help='Only check N files per directory (0 = all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("EMISSIONS DATA VALIDATION")
    print("=" * 70)
    
    errors = []
    warnings = []
    checked = 0
    passed = 0
    
    # ========================================================================
    # CHECK VISION CHECKPOINTS
    # ========================================================================
    print("\n>>> Checking vision checkpoints...")
    
    vision_dirs = [
        PROJECT_ROOT / "checkpoints/vision/mnist",
        PROJECT_ROOT / "checkpoints/vision/fashion",
        PROJECT_ROOT / "checkpoints/vision/noise_sweep",
        PROJECT_ROOT / "checkpoints/vision/size_sweep",
        PROJECT_ROOT / "checkpoints/vision/challenge",
        PROJECT_ROOT / "checkpoints/vision/adversarial",
    ]
    
    for ckpt_dir in vision_dirs:
        if not ckpt_dir.exists():
            if args.strict:
                warnings.append(f"Directory not found: {ckpt_dir}")
            continue
        
        pt_files = list(ckpt_dir.glob('*.pt'))
        if args.sample > 0:
            pt_files = pt_files[:args.sample]
        
        for pt_file in pt_files:
            checked += 1
            ok, err = check_checkpoint(pt_file)
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  [OK] {pt_file.relative_to(PROJECT_ROOT)}")
            else:
                errors.append(f"{pt_file.relative_to(PROJECT_ROOT)}: {err}")
                if args.verbose:
                    print(f"  [FAIL] {pt_file.relative_to(PROJECT_ROOT)}: {err}")
    
    # ========================================================================
    # CHECK EXTENSION 2 CHECKPOINTS
    # ========================================================================
    print("\n>>> Checking extension 2 checkpoints...")
    
    extension_cross_dataset_dir = PROJECT_ROOT / "checkpoints/extension_cross_dataset"
    if extension_cross_dataset_dir.exists():
        pt_files = list(extension_cross_dataset_dir.glob('*.pt'))
        if args.sample > 0:
            pt_files = pt_files[:args.sample]
        
        for pt_file in pt_files:
            checked += 1
            ok, err = check_checkpoint(pt_file)
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  [OK] {pt_file.relative_to(PROJECT_ROOT)}")
            else:
                errors.append(f"{pt_file.relative_to(PROJECT_ROOT)}: {err}")
                if args.verbose:
                    print(f"  [FAIL] {pt_file.relative_to(PROJECT_ROOT)}: {err}")
    else:
        if args.strict:
            warnings.append(f"Directory not found: {extension_cross_dataset_dir}")
    
    # ========================================================================
    # CHECK EXTENSION CP CHECKPOINTS
    # ========================================================================
    print("\n>>> Checking extension CP checkpoints...")
    
    ext_cp_dir = PROJECT_ROOT / "checkpoints/extension_cp"
    if ext_cp_dir.exists():
        pt_files = list(ext_cp_dir.glob('*.pt'))
        if args.sample > 0:
            pt_files = pt_files[:args.sample]
        
        for pt_file in pt_files:
            checked += 1
            ok, err = check_checkpoint(pt_file)
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  [OK] {pt_file.relative_to(PROJECT_ROOT)}")
            else:
                errors.append(f"{pt_file.relative_to(PROJECT_ROOT)}: {err}")
                if args.verbose:
                    print(f"  [FAIL] {pt_file.relative_to(PROJECT_ROOT)}: {err}")
    else:
        if args.strict:
            warnings.append(f"Directory not found: {ext_cp_dir}")
    
    # ========================================================================
    # CHECK LANGUAGE JSON RESULTS
    # ========================================================================
    print("\n>>> Checking language JSON results...")
    
    # Define expected language result files and their required keys
    language_files = {
        # Figure 9 correlation results (one per model)
        'correlation_fw-medium.json': ['co2_kg'],
        'correlation_fw-small.json': ['co2_kg'],
        'correlation_ts-medium.json': ['co2_kg'],
        # Figure 8 circuit search
        'circuit_search_complete.json': ['co2_kg'],
        # Figure 8 legacy data
        'figure_8_data_fw_medium.json': ['co2_kg'],
        # Figure 10 SAE training time
        'sae_training_time_comparison.json': ['co2_kg'],
        # Negation discovery
        'negation_analysis.json': ['co2_kg'],
        # Interaction analysis  
        'interaction_analysis.json': ['co2_kg'],
    }
    
    lang_dir = PROJECT_ROOT / "results/language"
    if lang_dir.exists():
        for fname, required_keys in language_files.items():
            fpath = lang_dir / fname
            if fpath.exists():
                checked += 1
                ok, err = check_json(fpath, required_keys)
                if ok:
                    passed += 1
                    if args.verbose:
                        print(f"  [OK] results/language/{fname}")
                else:
                    errors.append(f"results/language/{fname}: {err}")
                    if args.verbose:
                        print(f"  [FAIL] results/language/{fname}: {err}")
            else:
                if args.strict:
                    warnings.append(f"File not found: results/language/{fname}")
                elif args.verbose:
                    print(f"  [SKIP] results/language/{fname} (not found)")
    else:
        if args.strict:
            warnings.append(f"Directory not found: {lang_dir}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Files checked: {checked}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(errors)}")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print("\nVALIDATION FAILED")
        print("Some outputs are missing emissions data.")
        print("Re-run experiments with updated scripts to add emissions tracking.")
        sys.exit(1)
    else:
        if checked == 0:
            print("\nWARNING: No files found to validate!")
            print("Run experiments first with ./scripts/train/run_all.sh")
            if args.strict:
                sys.exit(1)
        else:
            print("\nVALIDATION PASSED")
            print("All checked outputs have emissions data.")


if __name__ == "__main__":
    main()
