"""
Debug Extension 2: Visual Comparison of Eigenvectors

Objective: Visually compare top-3 eigenvectors of MNIST '0' vs EMNIST 'O'
to understand why subspace similarity is random.

Goal: Check if the "shapes" are forming at the same pixel coordinates.

Usage:
    python src/debug_extension2_visuals.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_device, setup_mps_fallbacks, is_mps_device


def compute_center_of_mass(image_2d):
    """
    Compute center of mass of a 2D image.
    
    Args:
        image_2d: 2D numpy array (28x28)
    
    Returns:
        (y_com, x_com): Center of mass coordinates
    """
    # Normalize to positive values
    img = image_2d - image_2d.min()
    img = img / (img.max() + 1e-10)
    
    # Compute center of mass
    total_mass = img.sum()
    if total_mass == 0:
        return (14, 14)  # Default to center
    
    y_indices, x_indices = np.indices(img.shape)
    y_com = (y_indices * img).sum() / total_mass
    x_com = (x_indices * img).sum() / total_mass
    
    return (y_com, x_com)


def load_eigenvectors_from_checkpoint(checkpoint_path, device):
    """
    Load eigenvalues and eigenvectors from checkpoint (same as Phase 1).
    
    Uses the same approach as vision_analysis.py: load directly without pre-sorting,
    since plot_eigenvectors_grid handles sorting internally.
    """
    from src.analysis.spectral import load_checkpoint_eigenvalues
    
    print(f"Loading: {checkpoint_path.name}")
    eigenvalues, eigenvectors = load_checkpoint_eigenvalues(str(checkpoint_path))
    
    print(f"  Shape: eigenvalues={eigenvalues.shape}, eigenvectors={eigenvectors.shape}")
    
    # Return raw (unsorted) - plot_eigenvectors_grid will sort internally
    return eigenvectors, eigenvalues


def plot_eigenvector_comparison(mnist_vecs, emnist_vecs, mnist_vals, emnist_vals, 
                                 mnist_class=0, emnist_class=14, k=5, output_path=None,
                                 emnist_type="letters"):
    """
    Plot top-k positive and top-k negative eigenvectors for MNIST digit and EMNIST (letter or digit).
    
    Uses the same visualization style as Phase 1 (plot_eigenvectors_grid with show_both_signs=True).
    
    Args:
        mnist_vecs: MNIST eigenvectors [n_classes, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [n_classes, n_components, 784]
        mnist_vals: MNIST eigenvalues [n_classes, n_components]
        emnist_vals: EMNIST eigenvalues [n_classes, n_components]
        mnist_class: MNIST class index (0 for digit '0')
        emnist_class: EMNIST class index (14 for letter 'O', or 0 for digit '0')
        k: Number of top eigenvectors to plot per sign (positive/negative)
        output_path: Path to save figure (optional)
        emnist_type: "letters" or "digits" to determine label format
    """
    import torch
    from src.plot_utils.eigenvectors import plot_eigenvectors_grid
    
    # Determine labels
    mnist_label = str(mnist_class)
    if emnist_type == "digits":
        emnist_label = str(emnist_class)
        emnist_type_label = "Digit"
    else:
        emnist_label = chr(ord('A') + emnist_class)
        emnist_type_label = "Letter"
    
    # Create combined tensors for comparison (2 classes: MNIST and EMNIST)
    # Move to CPU before stacking (visualization requires CPU tensors)
    combined_vecs = torch.stack([
        mnist_vecs[mnist_class].cpu(),  # [n_components, 784]
        emnist_vecs[emnist_class].cpu()  # [n_components, 784]
    ])  # [2, n_components, 784]
    
    combined_vals = torch.stack([
        mnist_vals[mnist_class].cpu(),  # [n_components]
        emnist_vals[emnist_class].cpu()  # [n_components]
    ])  # [2, n_components]
    
    # Use the exact same pattern as vision_analysis.py (lines 163-169)
    title = f"Eigenvector Comparison: MNIST Digit '{mnist_label}' vs EMNIST {emnist_type_label} '{emnist_label}'"
    
    fig = plot_eigenvectors_grid(
        combined_vecs,
        combined_vals,
        n_top=k,
        title=title,
        show_both_signs=True,
        classes=[0, 1],  # Show only the two classes we're comparing
        class_names=[f"MNIST '{mnist_label}'", f"EMNIST '{emnist_label}'"],
        save_path=str(output_path) if output_path else None,
    )
    
    # Print summary information
    print("\n" + "=" * 80)
    print(f"EIGENVECTOR COMPARISON: MNIST '{mnist_label}' vs EMNIST '{emnist_label}'")
    print("=" * 80)
    
    # Get eigenvalues for printing
    mnist_class_vals = mnist_vals[mnist_class].cpu().numpy()
    emnist_class_vals = emnist_vals[emnist_class].cpu().numpy()
    
    # Positive eigenvalues
    mnist_pos = mnist_class_vals[mnist_class_vals > 0]
    emnist_pos = emnist_class_vals[emnist_class_vals > 0]
    if len(mnist_pos) > 0:
        mnist_pos_sorted = np.sort(mnist_pos)[::-1][:k]
    else:
        mnist_pos_sorted = np.array([])
    if len(emnist_pos) > 0:
        emnist_pos_sorted = np.sort(emnist_pos)[::-1][:k]
    else:
        emnist_pos_sorted = np.array([])
    
    # Negative eigenvalues
    mnist_neg = mnist_class_vals[mnist_class_vals < 0]
    emnist_neg = emnist_class_vals[emnist_class_vals < 0]
    if len(mnist_neg) > 0:
        mnist_neg_sorted = np.sort(mnist_neg)[:k]  # Most negative first
    else:
        mnist_neg_sorted = np.array([])
    if len(emnist_neg) > 0:
        emnist_neg_sorted = np.sort(emnist_neg)[:k]
    else:
        emnist_neg_sorted = np.array([])
    
    print(f"\nMNIST Digit '{mnist_label}' - Top {len(mnist_pos_sorted)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(mnist_pos_sorted):
        print(f"  Pos #{i+1}: λ={val:.6f}")
    
    print(f"\nEMNIST {emnist_type_label} '{emnist_label}' - Top {len(emnist_pos_sorted)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(emnist_pos_sorted):
        print(f"  Pos #{i+1}: λ={val:.6f}")
    
    print(f"\nMNIST Digit '{mnist_label}' - Top {len(mnist_neg_sorted)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(mnist_neg_sorted):
        print(f"  Neg #{i+1}: λ={val:.6f}")
    
    print(f"\nEMNIST {emnist_type_label} '{emnist_label}' - Top {len(emnist_neg_sorted)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(emnist_neg_sorted):
        print(f"  Neg #{i+1}: λ={val:.6f}")
    
    if output_path:
        print(f"\nFigure saved to: {output_path}")
    
    return fig


def plot_emnist_to_emnist_comparison(emnist_digits_vecs, emnist_letters_vecs, 
                                      emnist_digits_vals, emnist_letters_vals,
                                      digit_class=0, letter_class=14, k=5, output_path=None):
    """
    Plot top-k positive and top-k negative eigenvectors for EMNIST digit vs EMNIST letter.
    
    Uses the same visualization style as Phase 1 (plot_eigenvectors_grid with show_both_signs=True).
    
    Args:
        emnist_digits_vecs: EMNIST digits eigenvectors [10, n_components, 784]
        emnist_letters_vecs: EMNIST letters eigenvectors [26, n_components, 784]
        emnist_digits_vals: EMNIST digits eigenvalues [10, n_components]
        emnist_letters_vals: EMNIST letters eigenvalues [26, n_components]
        digit_class: EMNIST digit class index (0 for digit '0')
        letter_class: EMNIST letter class index (14 for letter 'O')
        k: Number of top eigenvectors to plot per sign (positive/negative)
        output_path: Path to save figure (optional)
    """
    import torch
    from src.plot_utils.eigenvectors import plot_eigenvectors_grid
    
    # Determine labels
    digit_label = str(digit_class)
    letter_label = chr(ord('A') + letter_class)
    
    # Create combined tensors for comparison (2 classes: EMNIST digit and EMNIST letter)
    combined_vecs = torch.stack([
        emnist_digits_vecs[digit_class].cpu(),  # [n_components, 784]
        emnist_letters_vecs[letter_class].cpu()  # [n_components, 784]
    ])  # [2, n_components, 784]
    
    combined_vals = torch.stack([
        emnist_digits_vals[digit_class].cpu(),  # [n_components]
        emnist_letters_vals[letter_class].cpu()  # [n_components]
    ])  # [2, n_components]
    
    title = f"Eigenvector Comparison: EMNIST Digit '{digit_label}' vs EMNIST Letter '{letter_label}'"
    
    fig = plot_eigenvectors_grid(
        combined_vecs,
        combined_vals,
        n_top=k,
        title=title,
        show_both_signs=True,
        classes=[0, 1],  # Show only the two classes we're comparing
        class_names=[f"EMNIST Digit '{digit_label}'", f"EMNIST Letter '{letter_label}'"],
        save_path=str(output_path) if output_path else None,
    )
    
    # Print summary information
    print("\n" + "=" * 80)
    print(f"EIGENVECTOR COMPARISON: EMNIST Digit '{digit_label}' vs EMNIST Letter '{letter_label}'")
    print("=" * 80)
    
    # Get eigenvalues for printing
    digit_class_vals = emnist_digits_vals[digit_class].cpu().numpy()
    letter_class_vals = emnist_letters_vals[letter_class].cpu().numpy()
    
    # Positive eigenvalues
    digit_pos = digit_class_vals[digit_class_vals > 0]
    letter_pos = letter_class_vals[letter_class_vals > 0]
    if len(digit_pos) > 0:
        digit_pos_sorted = np.sort(digit_pos)[::-1][:k]
    else:
        digit_pos_sorted = np.array([])
    if len(letter_pos) > 0:
        letter_pos_sorted = np.sort(letter_pos)[::-1][:k]
    else:
        letter_pos_sorted = np.array([])
    
    # Negative eigenvalues
    digit_neg = digit_class_vals[digit_class_vals < 0]
    letter_neg = letter_class_vals[letter_class_vals < 0]
    if len(digit_neg) > 0:
        digit_neg_sorted = np.sort(digit_neg)[:k]  # Most negative first
    else:
        digit_neg_sorted = np.array([])
    if len(letter_neg) > 0:
        letter_neg_sorted = np.sort(letter_neg)[:k]
    else:
        letter_neg_sorted = np.array([])
    
    print(f"\nEMNIST Digit '{digit_label}' - Top {len(digit_pos_sorted)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(digit_pos_sorted):
        print(f"  Pos #{i+1}: λ={val:.6f}")
    
    print(f"\nEMNIST Letter '{letter_label}' - Top {len(letter_pos_sorted)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(letter_pos_sorted):
        print(f"  Pos #{i+1}: λ={val:.6f}")
    
    print(f"\nEMNIST Digit '{digit_label}' - Top {len(digit_neg_sorted)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(digit_neg_sorted):
        print(f"  Neg #{i+1}: λ={val:.6f}")
    
    print(f"\nEMNIST Letter '{letter_label}' - Top {len(letter_neg_sorted)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i, val in enumerate(letter_neg_sorted):
        print(f"  Neg #{i+1}: λ={val:.6f}")
    
    if output_path:
        print(f"\nFigure saved to: {output_path}")
    
    return fig


def compute_pairwise_cosine_similarity(mnist_vecs, emnist_vecs, mnist_vals, emnist_vals, 
                                        mnist_class=0, emnist_class=14, k=5):
    """
    Compute pairwise cosine similarities between MNIST and EMNIST eigenvectors,
    separated by positive and negative eigenvalues.
    """
    import torch
    
    # Get eigenvectors and eigenvalues for this class
    mnist_class_vals = mnist_vals[mnist_class].cpu().numpy()
    mnist_class_vecs = mnist_vecs[mnist_class].cpu()
    emnist_class_vals = emnist_vals[emnist_class].cpu().numpy()
    emnist_class_vecs = emnist_vecs[emnist_class].cpu()
    
    # Separate by sign
    def get_pos_neg(vals, vecs, k):
        pos_mask = vals > 0
        neg_mask = vals < 0
        
        pos_vals = vals[pos_mask]
        pos_vecs = vecs[pos_mask]
        neg_vals = vals[neg_mask]
        neg_vecs = vecs[neg_mask]
        
        # Sort and take top k (use list indexing to avoid negative strides)
        if len(pos_vals) > 0:
            pos_sorted = np.argsort(pos_vals)[::-1][:k].copy()  # Make a copy to avoid negative strides
            pos_vecs = pos_vecs[torch.from_numpy(pos_sorted)]
        if len(neg_vals) > 0:
            neg_sorted = np.argsort(neg_vals)[:k].copy()
            neg_vecs = neg_vecs[torch.from_numpy(neg_sorted)]
        
        return pos_vecs, neg_vecs
    
    mnist_pos, mnist_neg = get_pos_neg(mnist_class_vals, mnist_class_vecs, k)
    emnist_pos, emnist_neg = get_pos_neg(emnist_class_vals, emnist_class_vecs, k)
    
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY (Positive Eigenvectors)")
    print("=" * 80)
    
    if len(mnist_pos) > 0 and len(emnist_pos) > 0:
        # Normalize
        mnist_pos_norm = mnist_pos / (mnist_pos.norm(dim=1, keepdim=True) + 1e-10)
        emnist_pos_norm = emnist_pos / (emnist_pos.norm(dim=1, keepdim=True) + 1e-10)
        
        cos_sim_pos = mnist_pos_norm @ emnist_pos_norm.T
        
        print(f"\nPositive Cosine Similarity Matrix ({len(mnist_pos)}x{len(emnist_pos)}):")
        print("         " + "  ".join([f"E+{i+1}" for i in range(len(emnist_pos))]))
        for i in range(len(mnist_pos)):
            row_str = f"M+{i+1}  "
            for j in range(len(emnist_pos)):
                row_str += f"{cos_sim_pos[i, j].item():7.3f}  "
            print(row_str)
        
        print(f"\nMean similarity (positive): {cos_sim_pos.mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY (Negative Eigenvectors)")
    print("=" * 80)
    
    if len(mnist_neg) > 0 and len(emnist_neg) > 0:
        # Normalize
        mnist_neg_norm = mnist_neg / (mnist_neg.norm(dim=1, keepdim=True) + 1e-10)
        emnist_neg_norm = emnist_neg / (emnist_neg.norm(dim=1, keepdim=True) + 1e-10)
        
        cos_sim_neg = mnist_neg_norm @ emnist_neg_norm.T
        
        print(f"\nNegative Cosine Similarity Matrix ({len(mnist_neg)}x{len(emnist_neg)}):")
        print("         " + "  ".join([f"E-{i+1}" for i in range(len(emnist_neg))]))
        for i in range(len(mnist_neg)):
            row_str = f"M-{i+1}  "
            for j in range(len(emnist_neg)):
                row_str += f"{cos_sim_neg[i, j].item():7.3f}  "
            print(row_str)
        
        print(f"\nMean similarity (negative): {cos_sim_neg.mean().item():.4f}")


def compute_emnist_to_emnist_cosine_similarity(emnist_digits_vecs, emnist_letters_vecs,
                                               emnist_digits_vals, emnist_letters_vals,
                                               digit_class=0, letter_class=14, k=5):
    """
    Compute pairwise cosine similarities between EMNIST digits and EMNIST letters eigenvectors,
    separated by positive and negative eigenvalues.
    """
    import torch
    
    # Get eigenvectors and eigenvalues for this class
    digit_class_vals = emnist_digits_vals[digit_class].cpu().numpy()
    digit_class_vecs = emnist_digits_vecs[digit_class].cpu()
    letter_class_vals = emnist_letters_vals[letter_class].cpu().numpy()
    letter_class_vecs = emnist_letters_vecs[letter_class].cpu()
    
    # Separate by sign
    def get_pos_neg(vals, vecs, k):
        pos_mask = vals > 0
        neg_mask = vals < 0
        
        pos_vals = vals[pos_mask]
        pos_vecs = vecs[pos_mask]
        neg_vals = vals[neg_mask]
        neg_vecs = vecs[neg_mask]
        
        # Sort and take top k (use list indexing to avoid negative strides)
        if len(pos_vals) > 0:
            pos_sorted = np.argsort(pos_vals)[::-1][:k].copy()  # Make a copy to avoid negative strides
            pos_vecs = pos_vecs[torch.from_numpy(pos_sorted)]
        if len(neg_vals) > 0:
            neg_sorted = np.argsort(neg_vals)[:k].copy()
            neg_vecs = neg_vecs[torch.from_numpy(neg_sorted)]
        
        return pos_vecs, neg_vecs
    
    digit_pos, digit_neg = get_pos_neg(digit_class_vals, digit_class_vecs, k)
    letter_pos, letter_neg = get_pos_neg(letter_class_vals, letter_class_vecs, k)
    
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY (Positive Eigenvectors)")
    print("=" * 80)
    
    if len(digit_pos) > 0 and len(letter_pos) > 0:
        # Normalize
        digit_pos_norm = digit_pos / (digit_pos.norm(dim=1, keepdim=True) + 1e-10)
        letter_pos_norm = letter_pos / (letter_pos.norm(dim=1, keepdim=True) + 1e-10)
        
        cos_sim_pos = digit_pos_norm @ letter_pos_norm.T
        
        print(f"\nPositive Cosine Similarity Matrix ({len(digit_pos)}x{len(letter_pos)}):")
        print("         " + "  ".join([f"L+{i+1}" for i in range(len(letter_pos))]))
        for i in range(len(digit_pos)):
            row_str = f"D+{i+1}  "
            for j in range(len(letter_pos)):
                row_str += f"{cos_sim_pos[i, j].item():7.3f}  "
            print(row_str)
        
        print(f"\nMean similarity (positive): {cos_sim_pos.mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY (Negative Eigenvectors)")
    print("=" * 80)
    
    if len(digit_neg) > 0 and len(letter_neg) > 0:
        # Normalize
        digit_neg_norm = digit_neg / (digit_neg.norm(dim=1, keepdim=True) + 1e-10)
        letter_neg_norm = letter_neg / (letter_neg.norm(dim=1, keepdim=True) + 1e-10)
        
        cos_sim_neg = digit_neg_norm @ letter_neg_norm.T
        
        print(f"\nNegative Cosine Similarity Matrix ({len(digit_neg)}x{len(letter_neg)}):")
        print("         " + "  ".join([f"L-{i+1}" for i in range(len(letter_neg))]))
        for i in range(len(digit_neg)):
            row_str = f"D-{i+1}  "
            for j in range(len(letter_neg)):
                row_str += f"{cos_sim_neg[i, j].item():7.3f}  "
            print(row_str)
        
        print(f"\nMean similarity (negative): {cos_sim_neg.mean().item():.4f}")


def main():
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Debug Extension 2: Visual Eigenvector Comparison")
    parser.add_argument("--compare-emnist-to-emnist", action="store_true",
                        help="Compare EMNIST digits to EMNIST letters (instead of MNIST to EMNIST)")
    parser.add_argument("--mnist-digit", type=str, default="0", 
                        help="MNIST digit to compare (0-9)")
    parser.add_argument("--emnist-letter", type=str, default=None, 
                        help="EMNIST letter to compare (A-Z) - use with --emnist-type letters")
    parser.add_argument("--emnist-digit", type=str, default=None,
                        help="EMNIST digit to compare (0-9) - use with --emnist-type digits")
    parser.add_argument("--emnist-type", type=str, default="letters", choices=["letters", "digits"],
                        help="EMNIST dataset type: 'letters' (A-Z) or 'digits' (0-9)")
    parser.add_argument("--reg-config", type=str, default="current", choices=["current", "phase1"],
                        help="Regularization config: 'current' (noise=0.15, wd=0.5) or 'phase1' (noise=0.5, wd=1.0)")
    parser.add_argument("--with-com", action="store_true",
                        help="Use checkpoints trained WITH CoM normalization (both MNIST and EMNIST)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top eigenvectors to compare per sign (positive/negative)")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Checkpoint paths (using Phase 1 infrastructure)
    checkpoint_dir = PROJECT_ROOT / "results/phase1/checkpoints"
    
    if args.compare_emnist_to_emnist:
        # EMNIST digits vs EMNIST letters comparison
        # Determine digit and letter classes
        if args.emnist_digit is None:
            emnist_digit = "0"  # Default
        else:
            emnist_digit = args.emnist_digit
        digit_class = int(emnist_digit)
        
        if args.emnist_letter is None:
            emnist_letter = "O"  # Default
        else:
            emnist_letter = args.emnist_letter.upper()
        letter_class = ord(emnist_letter) - ord('A')
        
        # Select checkpoints based on reg config and CoM
        if args.reg_config == "phase1":
            if args.with_com:
                digits_checkpoint = "emnist_digits_phase1_reg_com_seed42.pt"
                letters_checkpoint = "emnist_letters_phase1_reg_com_seed42.pt"
                reg_label = "Phase 1 reg (σ=0.5, λ=1.0) - WITH CoM"
            else:
                digits_checkpoint = "emnist_digits_phase1_reg_seed42.pt"
                letters_checkpoint = "emnist_letters_phase1_reg_seed42.pt"
                reg_label = "Phase 1 reg (σ=0.5, λ=1.0)"
        else:
            if args.with_com:
                digits_checkpoint = "emnist_digits_regularized_com_seed42.pt"
                letters_checkpoint = "emnist_letters_regularized_com_seed42.pt"
                reg_label = "Current reg (σ=0.15, λ=0.5) - WITH CoM"
            else:
                digits_checkpoint = "emnist_digits_regularized_seed42.pt"
                letters_checkpoint = "emnist_letters_regularized_seed42.pt"
                reg_label = "Current reg (σ=0.15, λ=0.5)"
        
        digits_path = checkpoint_dir / digits_checkpoint
        letters_path = checkpoint_dir / letters_checkpoint
        
        print(f"\nComparing: EMNIST digit '{emnist_digit}' (class {digit_class}) vs "
              f"EMNIST letter '{emnist_letter}' (class {letter_class})")
        print(f"Regularization config: {reg_label}")
        
        # Check if files exist
        if not digits_path.exists():
            print(f"ERROR: EMNIST digits checkpoint not found: {digits_path}")
            return
        
        if not letters_path.exists():
            print(f"ERROR: EMNIST letters checkpoint not found: {letters_path}")
            return
        
        print("\n" + "=" * 80)
        print("LOADING CHECKPOINTS")
        print("=" * 80)
        
        # Load eigenvectors
        digits_vecs, digits_vals = load_eigenvectors_from_checkpoint(digits_path, device)
        letters_vecs, letters_vals = load_eigenvectors_from_checkpoint(letters_path, device)
        
        print(f"\nEMNIST Digits: {digits_vecs.shape[0]} classes, {digits_vecs.shape[1]} eigenvectors each")
        print(f"EMNIST Letters: {letters_vecs.shape[0]} classes, {letters_vecs.shape[1]} eigenvectors each")
        
        # Determine comparison label for file naming
        comparison_label = f"EMNIST{emnist_digit}_vs_EMNIST{emnist_letter}"
        reg_suffix = "phase1reg" if args.reg_config == "phase1" else "currentreg"
        if args.with_com:
            reg_suffix = f"{reg_suffix}_com"
        comparison_label = f"{comparison_label}_{reg_suffix}"
        
        # Plot comparison
        output_dir = PROJECT_ROOT / "results/extension2"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eigenvector_comparison_{comparison_label}.pdf"
        
        fig = plot_emnist_to_emnist_comparison(
            digits_vecs, letters_vecs, digits_vals, letters_vals,
            digit_class=digit_class, letter_class=letter_class, k=args.k,
            output_path=output_path
        )
        
        # Compute pairwise cosine similarities
        compute_emnist_to_emnist_cosine_similarity(
            digits_vecs, letters_vecs, digits_vals, letters_vals,
            digit_class=digit_class, letter_class=letter_class, k=args.k
        )
        
        # Also save to Report/figures
        report_path = PROJECT_ROOT / f"Report/figures/extension2_eigenvec_comparison_{comparison_label}.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Also saved to: {report_path}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nKey Questions to Answer:")
        print("1. Are the Centers of Mass aligned between EMNIST digits and letters?")
        print("2. Do the eigenvectors show similar visual patterns?")
        print("3. Are the L2 norms of differences small?")
        print("4. Are diagonal cosine similarities high?")
        print("\nIf NO to any: The models are learning different features!")
        
    else:
        # MNIST vs EMNIST comparison (original mode)
        # Map digit/letter to class indices
        mnist_digit = args.mnist_digit
        mnist_class = int(mnist_digit)
        
        # Determine EMNIST class based on type
        if args.emnist_type == "digits":
            if args.emnist_digit is None:
                # Default to same digit as MNIST
                emnist_digit = mnist_digit
            else:
                emnist_digit = args.emnist_digit
            emnist_class = int(emnist_digit)
            emnist_label = f"digit '{emnist_digit}'"
            if args.reg_config == "phase1":
                if args.with_com:
                    emnist_checkpoint = "emnist_digits_phase1_reg_com_seed42.pt"
                else:
                    emnist_checkpoint = "emnist_digits_phase1_reg_seed42.pt"
            else:
                if args.with_com:
                    emnist_checkpoint = "emnist_digits_regularized_com_seed42.pt"
                else:
                    emnist_checkpoint = "emnist_digits_regularized_seed42.pt"
        else:  # letters
            if args.emnist_letter is None:
                emnist_letter = "O"  # Default
            else:
                emnist_letter = args.emnist_letter.upper()
            emnist_class = ord(emnist_letter) - ord('A')
            emnist_label = f"letter '{emnist_letter}'"
            if args.reg_config == "phase1":
                if args.with_com:
                    emnist_checkpoint = "emnist_letters_phase1_reg_com_seed42.pt"
                else:
                    emnist_checkpoint = "emnist_letters_phase1_reg_seed42.pt"
            else:
                if args.with_com:
                    emnist_checkpoint = "emnist_letters_regularized_com_seed42.pt"
                else:
                    emnist_checkpoint = "emnist_letters_regularized_seed42.pt"
        
        # Also select MNIST checkpoint based on reg config and CoM
        if args.reg_config == "phase1":
            if args.with_com:
                mnist_checkpoint = "mnist_dense_full_com_seed42.pt"
                reg_label = "Phase 1 reg (σ=0.5, λ=1.0) - WITH CoM"
            else:
                mnist_checkpoint = "mnist_dense_full_seed42.pt"
                reg_label = "Phase 1 reg (σ=0.5, λ=1.0)"
        else:
            if args.with_com:
                mnist_checkpoint = "mnist_dense_full_com_seed42.pt"
                reg_label = "Current reg (σ=0.15, λ=0.5) - WITH CoM"
            else:
                mnist_checkpoint = "mnist_dense_full_seed42.pt"
                reg_label = "Current reg (σ=0.15, λ=0.5)"
        
        print(f"\nComparing: MNIST digit '{mnist_digit}' (class {mnist_class}) vs "
              f"EMNIST {emnist_label} (class {emnist_class})")
        print(f"Regularization config: {reg_label}")
        
        mnist_path = checkpoint_dir / mnist_checkpoint
        emnist_path = checkpoint_dir / emnist_checkpoint
        
        # Check if files exist
        if not mnist_path.exists():
            print(f"ERROR: MNIST checkpoint not found: {mnist_path}")
            return
        
        if not emnist_path.exists():
            print(f"ERROR: EMNIST checkpoint not found: {emnist_path}")
            return
        
        print("\n" + "=" * 80)
        print("LOADING CHECKPOINTS")
        print("=" * 80)
        
        # Load eigenvectors
        mnist_vecs, mnist_vals = load_eigenvectors_from_checkpoint(mnist_path, device)
        emnist_vecs, emnist_vals = load_eigenvectors_from_checkpoint(emnist_path, device)
        
        print(f"\nMNIST: {mnist_vecs.shape[0]} classes, {mnist_vecs.shape[1]} eigenvectors each")
        print(f"EMNIST: {emnist_vecs.shape[0]} classes, {emnist_vecs.shape[1]} eigenvectors each")
        
        # Determine comparison label for file naming
        if args.emnist_type == "digits":
            comparison_label = f"{mnist_digit}_vs_{emnist_digit}"
            emnist_display_label = emnist_digit
        else:
            comparison_label = f"{mnist_digit}_vs_{emnist_letter}"
            emnist_display_label = emnist_letter
        
        # Add reg config to filename
        reg_suffix = "phase1reg" if args.reg_config == "phase1" else "currentreg"
        if args.with_com:
            reg_suffix = f"{reg_suffix}_com"
        comparison_label = f"{comparison_label}_{reg_suffix}"
        
        # Plot comparison
        output_dir = PROJECT_ROOT / "results/extension2"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eigenvector_comparison_{comparison_label}.pdf"
        
        fig = plot_eigenvector_comparison(
            mnist_vecs, emnist_vecs, mnist_vals, emnist_vals,
            mnist_class=mnist_class, emnist_class=emnist_class, k=args.k,
            output_path=output_path, emnist_type=args.emnist_type
        )
        
        # Compute pairwise cosine similarities
        compute_pairwise_cosine_similarity(
            mnist_vecs, emnist_vecs, mnist_vals, emnist_vals,
            mnist_class=mnist_class, emnist_class=emnist_class, k=args.k
        )
        
        # Also save to Report/figures
        report_path = PROJECT_ROOT / f"Report/figures/extension2_eigenvec_comparison_{comparison_label}.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Also saved to: {report_path}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nKey Questions to Answer:")
        print("1. Are the Centers of Mass aligned between MNIST and EMNIST?")
        print("2. Do the eigenvectors show similar visual patterns?")
        print("3. Are the L2 norms of differences small?")
        print("4. Are diagonal cosine similarities high?")
        print("\nIf NO to any: The models are learning different features!")
    
    plt.show()


if __name__ == "__main__":
    main()
