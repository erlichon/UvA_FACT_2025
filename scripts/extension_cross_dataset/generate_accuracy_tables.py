#!/usr/bin/env python3
"""
Generate Extension 2 accuracy tables with mean ± std over 5 seeds.

Computes cross-dataset accuracies:
1. MNIST ↔ EMNIST-Digits (bidirectional)
2. MNIST → USPS
3. MNIST → EMNIST-Letters (semantic confusion)

Outputs LaTeX tables with mean ± std for each metric.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add original code to path
_ORIG_PATH = PROJECT_ROOT / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from image.model import Model, Config
from src.data.mnist import MNIST
from src.data.emnist import EMNISTDigits, EMNISTLetters
from src.data.usps import USPS


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load checkpoint and return model + config."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt


def evaluate_model_on_dataset(
    checkpoint_path: Path,
    test_dataset,
    device: str = "cpu",
) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate a model checkpoint on a test dataset.
    
    Returns:
        (overall_accuracy, per_class_accuracy_dict)
    """
    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path)
    
    # Extract config parameters (with defaults)
    config_dict = ckpt.get('config', {})
    config = Config(
        d_hidden=config_dict.get('d_hidden', 256),
        d_input=config_dict.get('d_input', 784),
        d_output=config_dict.get('d_output', 10),
        n_layer=config_dict.get('n_layer', 1),
        bias=config_dict.get('bias', False),
        residual=config_dict.get('residual', False),
        epochs=1,  # Not used for evaluation
        seed=ckpt.get('seed', 42),
    )
    model = Model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Get test data
    # Model.forward expects [batch, 28, 28] and flattens internally
    x = test_dataset.x  # [N, 1, 28, 28] or [N, 28, 28]
    y = test_dataset.y  # [N]
    
    # Evaluate
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1).cpu()
    
    # Overall accuracy
    overall_acc = (preds == y).float().mean().item()
    
    # Per-class accuracy
    per_class_acc = {}
    for class_idx in range(10):  # All datasets have 10 classes for digits
        mask = (y == class_idx)
        if mask.sum() > 0:
            class_acc = (preds[mask] == y[mask]).float().mean().item()
            per_class_acc[class_idx] = class_acc
    
    return overall_acc, per_class_acc


def evaluate_semantic_confusion(
    checkpoint_path: Path,
    test_dataset: EMNISTLetters,
    letter_to_digit: Dict[str, int],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate semantic confusion: which digit does each letter map to?
    
    Args:
        checkpoint_path: Path to MNIST-trained checkpoint
        test_dataset: EMNIST Letters test set
        letter_to_digit: Mapping from letter to expected digit (e.g., {'O': 0, 'I': 1})
        
    Returns:
        Dict mapping letter to accuracy (e.g., {'O': 0.95, 'I': 0.78})
    """
    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path)
    
    # Extract config parameters (with defaults)
    config_dict = ckpt.get('config', {})
    config = Config(
        d_hidden=config_dict.get('d_hidden', 256),
        d_input=config_dict.get('d_input', 784),
        d_output=config_dict.get('d_output', 10),
        n_layer=config_dict.get('n_layer', 1),
        bias=config_dict.get('bias', False),
        residual=config_dict.get('residual', False),
        epochs=1,  # Not used for evaluation
        seed=ckpt.get('seed', 42),
    )
    model = Model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Get test data
    # Model.forward expects [batch, 28, 28] and flattens internally
    x = test_dataset.x  # [N, 1, 28, 28] or [N, 28, 28]
    y_letters = test_dataset.y  # [N] - letter class indices (0-25)
    
    # Evaluate
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1).cpu()  # Predicted digit (0-9)
    
    # Map letter indices to letter names
    letter_names = [chr(ord('A') + i) for i in range(26)]
    
    results = {}
    for letter, expected_digit in letter_to_digit.items():
        # Find letter index
        letter_idx = ord(letter) - ord('A')
        
        # Get samples of this letter
        mask = (y_letters == letter_idx)
        if mask.sum() == 0:
            results[letter] = 0.0
            continue
        
        # Check if predictions match expected digit
        correct = (preds[mask] == expected_digit)
        acc = correct.float().mean().item()
        results[letter] = acc
    
    return results


def compute_cross_dataset_accuracies(
    checkpoint_dir: Path,
    seeds: List[int] = [42, 43, 44, 45, 46],
    device: str = "cpu",
) -> Dict:
    """
    Compute all cross-dataset accuracies across seeds.
    
    Returns:
        Dict with structure:
        {
            'mnist_on_emnist_digits': {
                'mean': float,
                'std': float,
                'per_seed': [float, ...]
            },
            'emnist_digits_on_mnist': {...},
            'mnist_on_usps': {...},
            'mnist_on_emnist_letters': {
                'O': {'mean': float, 'std': float, ...},
                'I': {...},
                ...
            }
        }
    """
    results = {
        'mnist_on_emnist_digits': {'per_seed': []},
        'emnist_digits_on_mnist': {'per_seed': []},
        'mnist_on_usps': {'per_seed': []},
        'mnist_on_usps_baseline': {'per_seed': []},  # Baseline (no regularization)
        'mnist_on_emnist_letters': {
            'O': {'per_seed': []},
            'I': {'per_seed': []},
            'Z': {'per_seed': []},
            'S': {'per_seed': []},
            # 'B' removed per user request
        }
    }
    
    # Load test datasets (with CoM normalization for fair comparison)
    print("Loading test datasets...")
    emnist_digits_test = EMNISTDigits(train=False, device=device, apply_com=True)
    mnist_test = MNIST(train=False, device=device, apply_com=True)
    usps_test = USPS(train=False, device=device, apply_com=True)
    emnist_letters_test = EMNISTLetters(train=False, device=device, apply_com=True)
    
    letter_to_digit = {'O': 0, 'I': 1, 'Z': 2, 'S': 5}  # B removed per user request
    
    # Check if baseline checkpoints exist (in vision checkpoints directory)
    # checkpoint_dir is checkpoints/extension_cross_dataset, so parent is checkpoints
    baseline_checkpoint_dir = checkpoint_dir.parent / "vision" / "mnist"
    
    # Evaluate for each seed
    for seed in seeds:
        print(f"\nProcessing seed {seed}...")
        
        # Paths to checkpoints
        mnist_ckpt = checkpoint_dir / f"mnist_dense_full_com_seed{seed}.pt"
        emnist_digits_ckpt = checkpoint_dir / f"emnist_digits_regularized_seed{seed}.pt"
        mnist_baseline_ckpt = baseline_checkpoint_dir / f"mnist_dense_none_seed{seed}.pt"
        
        if not mnist_ckpt.exists():
            print(f"  Warning: {mnist_ckpt} not found, skipping seed {seed}")
            continue
        if not emnist_digits_ckpt.exists():
            print(f"  Warning: {emnist_digits_ckpt} not found, skipping seed {seed}")
            continue
        
        # 1. MNIST → EMNIST-Digits
        print(f"  Evaluating MNIST → EMNIST-Digits...")
        acc, _ = evaluate_model_on_dataset(mnist_ckpt, emnist_digits_test, device)
        results['mnist_on_emnist_digits']['per_seed'].append(acc)
        
        # 2. EMNIST-Digits → MNIST
        print(f"  Evaluating EMNIST-Digits → MNIST...")
        acc, _ = evaluate_model_on_dataset(emnist_digits_ckpt, mnist_test, device)
        results['emnist_digits_on_mnist']['per_seed'].append(acc)
        
        # 3. MNIST → USPS (regularized)
        print(f"  Evaluating MNIST → USPS (regularized)...")
        acc, _ = evaluate_model_on_dataset(mnist_ckpt, usps_test, device)
        results['mnist_on_usps']['per_seed'].append(acc)
        
        # 3b. MNIST → USPS (baseline)
        if mnist_baseline_ckpt.exists():
            print(f"  Evaluating MNIST → USPS (baseline)...")
            acc, _ = evaluate_model_on_dataset(mnist_baseline_ckpt, usps_test, device)
            results['mnist_on_usps_baseline']['per_seed'].append(acc)
        else:
            print(f"  Warning: {mnist_baseline_ckpt} not found, skipping baseline for seed {seed}")
        
        # 4. MNIST → EMNIST-Letters (semantic confusion)
        print(f"  Evaluating MNIST → EMNIST-Letters (semantic confusion)...")
        letter_accs = evaluate_semantic_confusion(
            mnist_ckpt, emnist_letters_test, letter_to_digit, device
        )
        for letter, acc in letter_accs.items():
            results['mnist_on_emnist_letters'][letter]['per_seed'].append(acc)
    
    # Compute mean and std for each metric
    for key in ['mnist_on_emnist_digits', 'emnist_digits_on_mnist', 'mnist_on_usps', 'mnist_on_usps_baseline']:
        if results[key]['per_seed']:
            results[key]['mean'] = np.mean(results[key]['per_seed'])
            results[key]['std'] = np.std(results[key]['per_seed'])
    
    # Compute bidirectional average
    if results['mnist_on_emnist_digits']['per_seed'] and results['emnist_digits_on_mnist']['per_seed']:
        bidirectional_means = [
            (m + e) / 2
            for m, e in zip(
                results['mnist_on_emnist_digits']['per_seed'],
                results['emnist_digits_on_mnist']['per_seed']
            )
        ]
        results['bidirectional_avg'] = {
            'mean': np.mean(bidirectional_means),
            'std': np.std(bidirectional_means),
            'per_seed': bidirectional_means
        }
    
    # Compute mean and std for semantic confusion
    for letter in ['O', 'I', 'Z', 'S']:  # B removed per user request
        if letter in results['mnist_on_emnist_letters'] and results['mnist_on_emnist_letters'][letter]['per_seed']:
            results['mnist_on_emnist_letters'][letter]['mean'] = np.mean(
                results['mnist_on_emnist_letters'][letter]['per_seed']
            )
            results['mnist_on_emnist_letters'][letter]['std'] = np.std(
                results['mnist_on_emnist_letters'][letter]['per_seed']
            )
    
    return results


def determine_std_precision(std: float) -> int:
    """
    Determine number of decimal places needed to show leading non-zero digit of std.
    
    Args:
        std: Standard deviation (as fraction, e.g., 0.00024)
        
    Returns:
        Number of decimal places needed to show at least the leading non-zero digit
    """
    if std == 0.0:
        return 1
    
    std_pct = std * 100  # Convert to percentage
    
    # Find position of first non-zero digit after decimal point
    # e.g., 0.024 -> first non-zero at position 2 -> need 3 decimal places (0.024)
    # e.g., 0.834 -> first non-zero at position 1 -> need 1 decimal place (0.8)
    # e.g., 0.0012 -> first non-zero at position 3 -> need 4 decimal places (0.0012)
    
    if std_pct >= 1.0:
        return 1  # e.g., 1.2 -> 1.2%
    elif std_pct >= 0.1:
        return 2  # e.g., 0.83 -> 0.83% (showing leading digit 8 with some precision)
    else:
        # For values < 0.1, find how many decimal places until first non-zero
        # Convert to string to find first non-zero digit position
        std_str = f"{std_pct:.10f}"
        # Find position of first non-zero digit after decimal point
        decimal_pos = std_str.find('.')
        if decimal_pos == -1:
            return 1
        
        for i in range(decimal_pos + 1, len(std_str)):
            if std_str[i] != '0':
                # Found first non-zero at position i, need i - decimal_pos decimal places
                # Add 1 to show the digit itself
                return i - decimal_pos + 1
        
        return 3  # Fallback


def format_percentage(value: float, std: float = None, precision: int = 1) -> str:
    """Format percentage with optional std, showing enough precision for std's leading digit."""
    if std is not None:
        # Determine precision needed for std
        std_precision = determine_std_precision(std)
        # Use max of requested precision and std precision
        final_precision = max(precision, std_precision)
        return f"{value*100:.{final_precision}f} ± {std*100:.{final_precision}f}\\%"
    return f"{value*100:.{precision}f}\\%"


def generate_latex_table_mnist_emnist_digits(results: Dict, output_path: Path):
    """Generate LaTeX table for MNIST ↔ EMNIST-Digits."""
    # Sanity-check that required statistics exist
    required_keys = ["mnist_on_emnist_digits", "emnist_digits_on_mnist", "bidirectional_avg"]
    missing = [k for k in required_keys if k not in results or "mean" not in results[k]]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing cross-dataset accuracy statistics for: {missing_str}. "
            "This usually means no valid checkpoints were found for these runs. "
            "Make sure cross-dataset models are trained (run `run_extension_cross_dataset.sh train all` "
            "or `run_extension_cross_dataset.sh all`) before generating accuracy tables."
        )

    content = """% Extension 2: MNIST ↔ EMNIST-Digits Cross-Dataset Accuracy
% Mean ± std over 5 seeds

\\begin{table}[h]
\\centering
\\caption{Cross-Dataset Accuracy: MNIST $\\leftrightarrow$ EMNIST-Digits (mean over 5 seeds)}
\\label{tab:extension_cross_dataset_mnist_emnist_digits}
\\begin{tabular}{lc}
\\toprule
\\textbf{Direction} & \\textbf{Accuracy} \\\\
\\midrule
MNIST $\\rightarrow$ EMNIST-Digits & """ + format_percentage(
        results['mnist_on_emnist_digits']['mean'],
        results['mnist_on_emnist_digits']['std']
    ) + """ \\\\
EMNIST-Digits $\\rightarrow$ MNIST & """ + format_percentage(
        results['emnist_digits_on_mnist']['mean'],
        results['emnist_digits_on_mnist']['std']
    ) + """ \\\\
\\midrule
Bidirectional Average & """ + format_percentage(
        results['bidirectional_avg']['mean'],
        results['bidirectional_avg']['std']
    ) + """ \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    output_path.write_text(content)
    print(f"Generated: {output_path}")


def generate_latex_table_usps(results: Dict, output_path: Path):
    """Generate LaTeX table for MNIST → USPS."""
    # Check if baseline results exist
    baseline_row = ""
    if 'mnist_on_usps_baseline' in results and results['mnist_on_usps_baseline'].get('per_seed'):
        baseline_row = "Baseline & " + format_percentage(
            results['mnist_on_usps_baseline']['mean'],
            results['mnist_on_usps_baseline']['std']
        ) + " \\\\\n"
    
    content = """% Extension 2: MNIST → USPS Cross-Dataset Accuracy
% Mean ± std over 5 seeds

\\begin{table}[h]
\\centering
\\caption{Cross-Dataset Accuracy: MNIST $\\rightarrow$ USPS (mean over 5 seeds)}
\\label{tab:extension_cross_dataset_usps}
\\begin{tabular}{lc}
\\toprule
\\textbf{Model} & \\textbf{Accuracy} \\\\
\\midrule
""" + baseline_row + "Regularized & " + format_percentage(
        results['mnist_on_usps']['mean'],
        results['mnist_on_usps']['std']
    ) + """ \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    output_path.write_text(content)
    print(f"Generated: {output_path}")


def generate_latex_table_emnist_letters(results: Dict, output_path: Path):
    """Generate LaTeX table for MNIST → EMNIST-Letters (semantic confusion)."""
    letter_mappings = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        # 'B': '6' removed per user request
    }
    
    rows = []
    for letter, digit in letter_mappings.items():
        if letter in results['mnist_on_emnist_letters']:
            mean = results['mnist_on_emnist_letters'][letter]['mean']
            std = results['mnist_on_emnist_letters'][letter]['std']
            rows.append(
                f"{letter} $\\rightarrow$ {digit} & " +
                format_percentage(mean, std) + " \\\\"
            )
    
    content = """% Extension 2: MNIST → EMNIST-Letters Semantic Confusion
% Mean ± std over 5 seeds

\\begin{table}[h]
\\centering
\\caption{Semantic Confusion: MNIST on EMNIST Letters (mean over 5 seeds)}
\\label{tab:extension_cross_dataset_emnist_letters}
\\begin{tabular}{lc}
\\toprule
\\textbf{Letter $\\rightarrow$ Digit} & \\textbf{Accuracy} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    output_path.write_text(content)
    print(f"Generated: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Extension 2 accuracy tables with mean ± std"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "extension_cross_dataset",
        help="Directory containing Extension 2 checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Report" / "figures" / "extension_cross_dataset",
        help="Directory to save output tables"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated list of seeds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation (cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Extension 2: Cross-Dataset Accuracy Table Generation")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device}")
    print()
    
    # Compute accuracies
    results = compute_cross_dataset_accuracies(
        args.checkpoint_dir,
        seeds=seeds,
        device=args.device
    )
    
    # Save raw results as JSON
    json_path = args.output_dir / "accuracy_results_all_seeds.json"
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    json_path.write_text(json.dumps(serializable_results, indent=2))
    print(f"\nSaved raw results: {json_path}")
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table_mnist_emnist_digits(
        results,
        args.output_dir / "accuracy_table_mnist_emnist_digits.tex"
    )
    generate_latex_table_usps(
        results,
        args.output_dir / "accuracy_table_usps.tex"
    )
    generate_latex_table_emnist_letters(
        results,
        args.output_dir / "accuracy_table_emnist_letters.tex"
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
