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

from src.analysis.subspace import sort_eigenvectors_by_magnitude
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
    """Load and sort eigenvectors from checkpoint."""
    print(f"Loading: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'eigenvalues' not in ckpt or 'eigenvectors' not in ckpt:
        raise ValueError(f"Checkpoint missing eigenvalues/eigenvectors")
    
    eigenvalues = ckpt['eigenvalues'].to(device)
    eigenvectors = ckpt['eigenvectors'].to(device)
    
    print(f"  Shape: eigenvalues={eigenvalues.shape}, eigenvectors={eigenvectors.shape}")
    
    # Sort by magnitude
    sorted_vecs = sort_eigenvectors_by_magnitude(eigenvalues, eigenvectors)
    
    return sorted_vecs, eigenvalues


def plot_eigenvector_comparison(mnist_vecs, emnist_vecs, mnist_vals, emnist_vals, 
                                 mnist_class=0, emnist_class=14, k=5, output_path=None):
    """
    Plot top-k positive and top-k negative eigenvectors for MNIST digit and EMNIST letter.
    
    Args:
        mnist_vecs: MNIST eigenvectors [n_classes, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [n_classes, n_components, 784]
        mnist_vals: MNIST eigenvalues [n_classes, n_components]
        emnist_vals: EMNIST eigenvalues [n_classes, n_components]
        mnist_class: MNIST class index (0 for digit '0')
        emnist_class: EMNIST class index (14 for letter 'O')
        k: Number of top eigenvectors to plot per sign (positive/negative)
        output_path: Path to save figure (optional)
    """
    # Get all eigenvectors and eigenvalues for this class
    mnist_class_vals = mnist_vals[mnist_class].cpu().numpy()  # [n_components]
    mnist_class_vecs = mnist_vecs[mnist_class].cpu().numpy()  # [n_components, 784]
    emnist_class_vals = emnist_vals[emnist_class].cpu().numpy()
    emnist_class_vecs = emnist_vecs[emnist_class].cpu().numpy()
    
    # Separate by sign and sort
    def get_top_k_by_sign(vals, vecs, k):
        # Positive eigenvalues (sorted descending)
        pos_mask = vals > 0
        if pos_mask.sum() > 0:
            pos_vals = vals[pos_mask]
            pos_vecs = vecs[pos_mask]
            pos_sorted = np.argsort(pos_vals)[::-1]  # Descending
            pos_vals = pos_vals[pos_sorted][:k]
            pos_vecs = pos_vecs[pos_sorted][:k]
        else:
            pos_vals = np.array([])
            pos_vecs = np.array([]).reshape(0, 784)
        
        # Negative eigenvalues (sorted by most negative)
        neg_mask = vals < 0
        if neg_mask.sum() > 0:
            neg_vals = vals[neg_mask]
            neg_vecs = vecs[neg_mask]
            neg_sorted = np.argsort(neg_vals)  # Ascending (most negative first)
            neg_vals = neg_vals[neg_sorted][:k]
            neg_vecs = neg_vecs[neg_sorted][:k]
        else:
            neg_vals = np.array([])
            neg_vecs = np.array([]).reshape(0, 784)
        
        return pos_vals, pos_vecs, neg_vals, neg_vecs
    
    mnist_pos_vals, mnist_pos_vecs, mnist_neg_vals, mnist_neg_vecs = get_top_k_by_sign(
        mnist_class_vals, mnist_class_vecs, k
    )
    emnist_pos_vals, emnist_pos_vecs, emnist_neg_vals, emnist_neg_vecs = get_top_k_by_sign(
        emnist_class_vals, emnist_class_vecs, k
    )
    
    # Determine labels
    mnist_label = str(mnist_class)
    emnist_label = chr(ord('A') + emnist_class)
    
    # Create figure with 4 rows: MNIST+, EMNIST+, MNIST-, EMNIST-
    fig = plt.figure(figsize=(k*3, 16))
    gs = GridSpec(4, k, figure=fig, hspace=0.5, wspace=0.3)
    
    # Title
    fig.suptitle(f"Eigenvector Comparison: MNIST Digit '{mnist_label}' vs EMNIST Letter '{emnist_label}'\n"
                 f"Top-{k} Positive and Top-{k} Negative Eigenvectors",
                 fontsize=16, fontweight='bold', y=0.99)
    
    print("\n" + "=" * 80)
    print(f"EIGENVECTOR COMPARISON: MNIST '{mnist_label}' vs EMNIST '{emnist_label}'")
    print("=" * 80)
    
    # ---- Row 0: MNIST Positive Eigenvectors ----
    print(f"\nMNIST Digit '{mnist_label}' - Top {len(mnist_pos_vals)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i in range(len(mnist_pos_vals)):
        ax = fig.add_subplot(gs[0, i])
        img = mnist_pos_vecs[i].reshape(28, 28)
        y_com, x_com = compute_center_of_mass(img)
        
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.plot(x_com, y_com, 'g*', markersize=12, markeredgecolor='yellow', markeredgewidth=1.5)
        ax.set_title(f"MNIST+ #{i+1}\nλ={mnist_pos_vals[i]:.3f}\nCoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        
        print(f"  Pos #{i+1}: λ={mnist_pos_vals[i]:.6f}, CoM=({x_com:.2f}, {y_com:.2f})")
    
    # ---- Row 1: EMNIST Positive Eigenvectors ----
    print(f"\nEMNIST Letter '{emnist_label}' - Top {len(emnist_pos_vals)} POSITIVE Eigenvectors:")
    print("-" * 80)
    for i in range(len(emnist_pos_vals)):
        ax = fig.add_subplot(gs[1, i])
        img = emnist_pos_vecs[i].reshape(28, 28)
        y_com, x_com = compute_center_of_mass(img)
        
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.plot(x_com, y_com, 'g*', markersize=12, markeredgecolor='yellow', markeredgewidth=1.5)
        ax.set_title(f"EMNIST+ #{i+1}\nλ={emnist_pos_vals[i]:.3f}\nCoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        
        print(f"  Pos #{i+1}: λ={emnist_pos_vals[i]:.6f}, CoM=({x_com:.2f}, {y_com:.2f})")
    
    # ---- Row 2: MNIST Negative Eigenvectors ----
    print(f"\nMNIST Digit '{mnist_label}' - Top {len(mnist_neg_vals)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i in range(len(mnist_neg_vals)):
        ax = fig.add_subplot(gs[2, i])
        img = mnist_neg_vecs[i].reshape(28, 28)
        y_com, x_com = compute_center_of_mass(img)
        
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.plot(x_com, y_com, 'r*', markersize=12, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.set_title(f"MNIST- #{i+1}\nλ={mnist_neg_vals[i]:.3f}\nCoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        
        print(f"  Neg #{i+1}: λ={mnist_neg_vals[i]:.6f}, CoM=({x_com:.2f}, {y_com:.2f})")
    
    # ---- Row 3: EMNIST Negative Eigenvectors ----
    print(f"\nEMNIST Letter '{emnist_label}' - Top {len(emnist_neg_vals)} NEGATIVE Eigenvectors:")
    print("-" * 80)
    for i in range(len(emnist_neg_vals)):
        ax = fig.add_subplot(gs[3, i])
        img = emnist_neg_vecs[i].reshape(28, 28)
        y_com, x_com = compute_center_of_mass(img)
        
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.plot(x_com, y_com, 'r*', markersize=12, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.set_title(f"EMNIST- #{i+1}\nλ={emnist_neg_vals[i]:.3f}\nCoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        
        print(f"  Neg #{i+1}: λ={emnist_neg_vals[i]:.6f}, CoM=({x_com:.2f}, {y_com:.2f})")
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    plt.colorbar(im, cax=cbar_ax, label='Eigenvector Value')
    
    # Add legend for CoM markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='g', 
               markeredgecolor='yellow', markersize=12, label='CoM (Positive)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='r', 
               markeredgecolor='cyan', markersize=12, label='CoM (Negative)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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


def main():
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Debug Extension 2: Visual Eigenvector Comparison")
    parser.add_argument("--mnist-digit", type=str, default="0", 
                        help="MNIST digit to compare (0-9)")
    parser.add_argument("--emnist-letter", type=str, default="O", 
                        help="EMNIST letter to compare (A-Z)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top eigenvectors to compare per sign (positive/negative)")
    args = parser.parse_args()
    
    # Map digit/letter to class indices
    mnist_digit = args.mnist_digit
    emnist_letter = args.emnist_letter.upper()
    
    # MNIST: 0-9 are classes 0-9
    mnist_class = int(mnist_digit)
    
    # EMNIST Letters: A-Z are classes 0-25
    emnist_class = ord(emnist_letter) - ord('A')
    
    print(f"\nComparing: MNIST digit '{mnist_digit}' (class {mnist_class}) vs "
          f"EMNIST letter '{emnist_letter}' (class {emnist_class})")
    
    device = get_device()
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Checkpoint paths
    checkpoint_dir = PROJECT_ROOT / "results/extension2/checkpoints"
    mnist_path = checkpoint_dir / "mnist_regularized_seed42.pt"
    emnist_path = checkpoint_dir / "emnist_letters_regularized_seed42.pt"
    
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
    
    # Plot comparison
    output_dir = PROJECT_ROOT / "results/extension2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eigenvector_comparison_{mnist_digit}_vs_{emnist_letter}.pdf"
    
    fig = plot_eigenvector_comparison(
        mnist_vecs, emnist_vecs, mnist_vals, emnist_vals,
        mnist_class=mnist_class, emnist_class=emnist_class, k=args.k,
        output_path=output_path
    )
    
    # Compute pairwise cosine similarities
    compute_pairwise_cosine_similarity(
        mnist_vecs, emnist_vecs, mnist_vals, emnist_vals,
        mnist_class=mnist_class, emnist_class=emnist_class, k=args.k
    )
    
    # Also save to Report/figures
    report_path = PROJECT_ROOT / f"Report/figures/extension2_eigenvec_comparison_{mnist_digit}_vs_{emnist_letter}.pdf"
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
