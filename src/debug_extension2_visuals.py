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
                                 mnist_class=0, emnist_class=14, k=3, output_path=None):
    """
    Plot top-k eigenvectors for MNIST digit and EMNIST letter side by side.
    
    Args:
        mnist_vecs: MNIST eigenvectors [n_classes, n_components, 784]
        emnist_vecs: EMNIST eigenvectors [n_classes, n_components, 784]
        mnist_vals: MNIST eigenvalues [n_classes, n_components]
        emnist_vals: EMNIST eigenvalues [n_classes, n_components]
        mnist_class: MNIST class index (0 for digit '0')
        emnist_class: EMNIST class index (14 for letter 'O')
        k: Number of top eigenvectors to plot
        output_path: Path to save figure (optional)
    """
    # Extract top-k eigenvectors
    mnist_k_vecs = mnist_vecs[mnist_class, :k].cpu().numpy()  # [k, 784]
    emnist_k_vecs = emnist_vecs[emnist_class, :k].cpu().numpy()  # [k, 784]
    
    # Get corresponding eigenvalues
    mnist_k_vals = mnist_vals[mnist_class, :k].cpu().numpy()
    emnist_k_vals = emnist_vals[emnist_class, :k].cpu().numpy()
    
    # Determine labels
    mnist_label = str(mnist_class)
    emnist_label = chr(ord('A') + emnist_class)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, k, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle(f"Eigenvector Comparison: MNIST Digit '{mnist_label}' vs EMNIST Letter '{emnist_label}'\n"
                 f"Top-{k} Eigenvectors (Sorted by Magnitude)",
                 fontsize=16, fontweight='bold', y=0.98)
    
    print("\n" + "=" * 80)
    print(f"EIGENVECTOR COMPARISON: MNIST '{mnist_label}' vs EMNIST '{emnist_label}'")
    print("=" * 80)
    
    # Plot MNIST eigenvectors (Row 0)
    print(f"\nMNIST Digit '{mnist_label}' (Class {mnist_class}):")
    print("-" * 80)
    for i in range(k):
        ax = fig.add_subplot(gs[0, i])
        
        # Reshape to 28x28
        img = mnist_k_vecs[i].reshape(28, 28)
        
        # Compute center of mass
        y_com, x_com = compute_center_of_mass(img)
        
        # Plot
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        
        # Mark center of mass
        ax.plot(x_com, y_com, 'g*', markersize=15, markeredgecolor='yellow', markeredgewidth=1.5)
        
        ax.set_title(f"MNIST '{mnist_label}' Eigenvec #{i+1}\nλ={mnist_k_vals[i]:.3f}\n"
                     f"CoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        
        print(f"  Eigenvec #{i+1}: λ={mnist_k_vals[i]:.4f}, "
              f"CoM=({x_com:.2f}, {y_com:.2f}), "
              f"Range=[{img.min():.3f}, {img.max():.3f}]")
    
    # Plot EMNIST eigenvectors (Row 1)
    print(f"\nEMNIST Letter '{emnist_label}' (Class {emnist_class}):")
    print("-" * 80)
    for i in range(k):
        ax = fig.add_subplot(gs[1, i])
        
        # Reshape to 28x28
        img = emnist_k_vecs[i].reshape(28, 28)
        
        # Compute center of mass
        y_com, x_com = compute_center_of_mass(img)
        
        # Plot
        im = ax.imshow(img, cmap='RdBu_r', vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        
        # Mark center of mass
        ax.plot(x_com, y_com, 'g*', markersize=15, markeredgecolor='yellow', markeredgewidth=1.5)
        
        ax.set_title(f"EMNIST '{emnist_label}' Eigenvec #{i+1}\nλ={emnist_k_vals[i]:.3f}\n"
                     f"CoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        
        print(f"  Eigenvec #{i+1}: λ={emnist_k_vals[i]:.4f}, "
              f"CoM=({x_com:.2f}, {y_com:.2f}), "
              f"Range=[{img.min():.3f}, {img.max():.3f}]")
    
    # Plot differences (Row 2)
    print("\nDifferences (MNIST - EMNIST):")
    print("-" * 80)
    for i in range(k):
        ax = fig.add_subplot(gs[2, i])
        
        # Compute difference
        diff = mnist_k_vecs[i] - emnist_k_vecs[i]
        diff_img = diff.reshape(28, 28)
        
        # Compute CoM for difference
        y_com, x_com = compute_center_of_mass(np.abs(diff_img))
        
        # Plot
        max_abs = np.abs(diff_img).max()
        im = ax.imshow(diff_img, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        
        # Mark center of mass of difference
        ax.plot(x_com, y_com, 'm*', markersize=15, markeredgecolor='cyan', markeredgewidth=1.5)
        
        # Compute L2 norm of difference
        l2_norm = np.linalg.norm(diff)
        
        ax.set_title(f"Difference #{i+1}\nL2={l2_norm:.3f}\n"
                     f"CoM=({x_com:.1f}, {y_com:.1f})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        
        print(f"  Difference #{i+1}: L2_norm={l2_norm:.4f}, "
              f"Max_abs={max_abs:.3f}, "
              f"CoM_of_diff=({x_com:.2f}, {y_com:.2f})")
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Eigenvector Value')
    
    # Add legend for CoM markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='g', 
               markeredgecolor='yellow', markersize=12, label='Center of Mass'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='m', 
               markeredgecolor='cyan', markersize=12, label='CoM of Difference'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    return fig


def compute_pairwise_cosine_similarity(mnist_vecs, emnist_vecs, mnist_class=0, emnist_class=14, k=3):
    """
    Compute pairwise cosine similarities between MNIST and EMNIST eigenvectors.
    """
    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY")
    print("=" * 80)
    
    mnist_k = mnist_vecs[mnist_class, :k].cpu()  # [k, 784]
    emnist_k = emnist_vecs[emnist_class, :k].cpu()  # [k, 784]
    
    # Normalize
    mnist_k_norm = mnist_k / (mnist_k.norm(dim=1, keepdim=True) + 1e-10)
    emnist_k_norm = emnist_k / (emnist_k.norm(dim=1, keepdim=True) + 1e-10)
    
    # Compute pairwise cosine similarity
    cos_sim_matrix = mnist_k_norm @ emnist_k_norm.T
    
    print(f"\nCosine Similarity Matrix ({k}x{k}):")
    print("         " + "  ".join([f"EMNIST_{i+1}" for i in range(k)]))
    for i in range(k):
        row_str = f"MNIST_{i+1}  "
        for j in range(k):
            row_str += f"{cos_sim_matrix[i, j].item():8.4f}  "
        print(row_str)
    
    # Best matches
    print(f"\nBest matches:")
    for i in range(k):
        best_j = cos_sim_matrix[i].argmax().item()
        best_sim = cos_sim_matrix[i, best_j].item()
        print(f"  MNIST_{i+1} ↔ EMNIST_{best_j+1}: {best_sim:.4f}")
    
    # Diagonal (corresponding eigenvectors)
    print(f"\nDiagonal similarities (corresponding eigenvectors):")
    for i in range(k):
        print(f"  MNIST_{i+1} ↔ EMNIST_{i+1}: {cos_sim_matrix[i, i].item():.4f}")
    
    print(f"\nMean diagonal similarity: {cos_sim_matrix.diag().mean().item():.4f}")
    print(f"Mean off-diagonal similarity: {(cos_sim_matrix.sum() - cos_sim_matrix.diag().sum()).item() / (k*k - k):.4f}")


def main():
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Debug Extension 2: Visual Eigenvector Comparison")
    parser.add_argument("--mnist-digit", type=str, default="0", 
                        help="MNIST digit to compare (0-9)")
    parser.add_argument("--emnist-letter", type=str, default="O", 
                        help="EMNIST letter to compare (A-Z)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of top eigenvectors to compare")
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
        mnist_vecs, emnist_vecs,
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
