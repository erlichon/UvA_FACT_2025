"""
Adversarial robustness visualization (Figure 7).

Paper-style visualization showing how eigenvector-derived adversarial masks
affect model accuracy and misclassification rates.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.plot_utils.style import set_publication_style


def compute_adversarial_curves(
    model: torch.nn.Module,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target_class: int,
    alphas: List[float],
    target_ranks: List[int],
    top_k_pinv: int,
    restrict_mask: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """
    Evaluate accuracy and misclassification curves across alpha values.
    
    Args:
        model: The trained model
        eigenvalues: Eigenvalues tensor [n_classes, d_hidden]
        eigenvectors: Eigenvectors tensor [n_classes, d_hidden, d_input]
        x: Input data flattened [n_samples, d_input]
        y: Labels [n_samples]
        target_class: Target digit for adversarial attack
        alphas: List of perturbation strengths
        target_ranks: List of eigenvector ranks to average over
        top_k_pinv: Number of top eigenvectors for pseudoinverse
        restrict_mask: Optional mask to restrict perturbation (e.g., rare-edge pixels)
        
    Returns:
        Dictionary with accuracy and misclassification curves
    """
    from src.vision.adversarial import (
        compute_adversarial_mask,
        apply_adversarial_perturbation,
        compute_random_baseline_mask,
    )
    
    adv_acc, rand_acc, adv_mis, rand_mis = [], [], [], []
    rand_seed = 42
    
    for alpha in alphas:
        accs_adv, accs_rand, mis_adv, mis_rand = [], [], [], []
        for rank in target_ranks:
            adv_mask = compute_adversarial_mask(
                eigenvectors[target_class],
                eigenvalues[target_class],
                target_rank=rank,
                top_k=top_k_pinv,
                use_positive_only=True,
            )
            rand_mask = compute_random_baseline_mask(adv_mask, seed=rand_seed)
            rand_seed += 1

            if restrict_mask is not None:
                adv_mask = adv_mask * restrict_mask
                rand_mask = rand_mask * restrict_mask

            x_adv = apply_adversarial_perturbation(x, adv_mask, alpha=float(alpha))
            x_rand = apply_adversarial_perturbation(x, rand_mask, alpha=float(alpha))

            with torch.no_grad():
                preds_adv = model(x_adv.reshape(-1, 28, 28)).argmax(dim=-1)
                preds_rand = model(x_rand.reshape(-1, 28, 28)).argmax(dim=-1)

            accs_adv.append((preds_adv == y).float().mean().item())
            accs_rand.append((preds_rand == y).float().mean().item())
            mis_adv.append((preds_adv == target_class).float().mean().item())
            mis_rand.append((preds_rand == target_class).float().mean().item())

        adv_acc.append(float(np.mean(accs_adv)))
        rand_acc.append(float(np.mean(accs_rand)))
        adv_mis.append(float(np.mean(mis_adv)))
        rand_mis.append(float(np.mean(mis_rand)))

    return {
        "adv_acc": adv_acc,
        "rand_acc": rand_acc,
        "adv_misclass": adv_mis,
        "rand_misclass": rand_mis,
    }


def aggregate_curves(seed_curves: List[Dict[str, List[float]]]) -> Dict[str, np.ndarray]:
    """Aggregate curves across seeds to get mean and std."""
    keys = list(seed_curves[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        arr = np.array([c[k] for c in seed_curves], dtype=np.float32)
        out[f"{k}_mean"] = arr.mean(axis=0)
        out[f"{k}_std"] = arr.std(axis=0)
    return out


def find_misclassified_example(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    target_class: int,
) -> torch.Tensor:
    """Find an example that gets misclassified to target_class after perturbation."""
    from src.vision.adversarial import apply_adversarial_perturbation
    
    x_adv = apply_adversarial_perturbation(x, mask, alpha=alpha)
    with torch.no_grad():
        preds_orig = model(x.reshape(-1, 28, 28)).argmax(dim=-1)
        preds_adv = model(x_adv.reshape(-1, 28, 28)).argmax(dim=-1)
    candidates = (preds_orig == y) & (preds_adv == target_class) & (y != target_class)
    idx = int(candidates.nonzero()[0].item()) if candidates.any() else 0
    return x_adv[idx].reshape(28, 28).detach().cpu()


def get_visualization_artifacts(
    model: torch.nn.Module,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target_class: int,
    top_k_pinv: int,
    alpha_annotate: float,
    restrict_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get visualization artifacts for a model.
    
    Returns:
        Tuple of (eigenvector, adversarial_mask, random_mask, misclassified_example)
    """
    from src.vision.adversarial import (
        compute_adversarial_mask,
        compute_random_baseline_mask,
    )
    
    # Choose top positive eigenvector for the target digit
    vals_t = eigenvalues[target_class]
    pos = torch.where(vals_t > 0)[0]
    idx = int(pos[vals_t[pos].argmax()].item()) if len(pos) else int(vals_t.abs().argmax().item())
    
    viz_eig = eigenvectors[target_class, idx].detach().cpu()
    viz_adv = compute_adversarial_mask(
        eigenvectors[target_class], eigenvalues[target_class],
        target_rank=0, top_k=top_k_pinv, use_positive_only=True,
    ).detach().cpu()
    
    if restrict_mask is not None:
        viz_adv = viz_adv * restrict_mask.detach().cpu()
    
    viz_rand = compute_random_baseline_mask(viz_adv, seed=42).detach().cpu()
    if restrict_mask is not None:
        viz_rand = viz_rand * restrict_mask.detach().cpu()
    
    viz_mis = find_misclassified_example(
        model, x, y, viz_adv.to(eigenvalues.device), alpha_annotate, target_class
    )
    
    return viz_eig, viz_adv, viz_rand, viz_mis


def plot_adversarial_figure(
    agg_noise: Dict[str, np.ndarray],
    agg_noreg: Dict[str, np.ndarray],
    viz_noise: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    viz_noreg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    alphas: List[float],
    target_class: int,
    alpha_annotate: float = 2.0,
) -> plt.Figure:
    """
    Create paper-style Figure 7 with adversarial robustness comparison.
    
    Args:
        agg_noise: Aggregated curves for noise-regularized model
        agg_noreg: Aggregated curves for no-regularization model
        viz_noise: Visualization artifacts (eig, adv_mask, rand_mask, mis_example) for noise model
        viz_noreg: Visualization artifacts for no-reg model
        alphas: List of perturbation strengths
        target_class: Target digit
        alpha_annotate: Alpha value to highlight in annotations
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    fig = plt.figure(figsize=(15, 5.5))
    gs = GridSpec(
        2, 7,
        figure=fig,
        width_ratios=[0.35, 1.2, 1.2, 1.2, 1.2, 1.6, 1.6],
        hspace=0.35,
        wspace=0.25,
        left=0.04,
        right=0.99,
        top=0.88,
        bottom=0.08,
    )

    # Column titles
    col_titles = ["Eigenvector", "Adv. mask", "Misclassified", "Rand. mask", "Accuracy", "Misclassification"]
    for col, title in enumerate(col_titles):
        ax_t = fig.add_subplot(gs[0, col + 1])
        ax_t.set_title(title, fontsize=10, pad=3)
        ax_t.axis("off")

    def im_signed(ax, v: torch.Tensor) -> None:
        img = v.reshape(28, 28).numpy()
        vmax = float(np.abs(img).max() or 1.0)
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    def im_gray(ax, v: torch.Tensor) -> None:
        ax.imshow(v.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    alpha_idx = int(np.argmin(np.abs(np.array(alphas) - alpha_annotate)))

    def annotate_point(ax, label: str, mean: float, std: float) -> None:
        ax.text(
            0.02, 0.02,
            f"{label}@α={alphas[alpha_idx]:.1f}:\n{mean:.3f} ± {std:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
        )

    viz_eig_noise, viz_adv_noise, viz_rand_noise, viz_mis_noise = viz_noise
    viz_eig_noreg, viz_adv_noreg, viz_rand_noreg, viz_mis_noreg = viz_noreg

    rows = [
        ("A) Noise\n(sigma=0.15)", agg_noise, viz_eig_noise, viz_adv_noise, viz_rand_noise, viz_mis_noise),
        ("B) No reg", agg_noreg, viz_eig_noreg, viz_adv_noreg, viz_rand_noreg, viz_mis_noreg),
    ]

    for r, (row_label, agg, eig, adv_mask, rand_mask, mis_ex) in enumerate(rows):
        # Label cell
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.5, row_label, fontsize=10, fontweight="bold", va="center", ha="center")

        # Eigenvector
        ax = fig.add_subplot(gs[r, 1])
        im_signed(ax, eig)

        # Adversarial mask
        ax = fig.add_subplot(gs[r, 2])
        im_signed(ax, adv_mask)

        # Misclassified example
        ax = fig.add_subplot(gs[r, 3])
        im_gray(ax, mis_ex)

        # Random mask
        ax = fig.add_subplot(gs[r, 4])
        im_signed(ax, rand_mask)

        # Accuracy curve
        ax = fig.add_subplot(gs[r, 5])
        ax.errorbar(alphas, agg["adv_acc_mean"], yerr=agg["adv_acc_std"], fmt="o-", color="C1",
                    label="Adversarial", linewidth=2, markersize=4, capsize=3)
        ax.errorbar(alphas, agg["rand_acc_mean"], yerr=agg["rand_acc_std"], fmt="s-", color="C2",
                    label="Random", linewidth=2, markersize=4, capsize=3)
        ax.axhline(y=float(agg["adv_acc_mean"][0]), color="C0", linewidth=2, label="Original")
        ax.set_xlabel("Mask strength α", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        annotate_point(ax, "Acc", float(agg["adv_acc_mean"][alpha_idx]), float(agg["adv_acc_std"][alpha_idx]))

        # Misclassification curve
        ax = fig.add_subplot(gs[r, 6])
        ax.errorbar(alphas, agg["adv_misclass_mean"], yerr=agg["adv_misclass_std"], fmt="o-", color="C1",
                    label="Adversarial", linewidth=2, markersize=4, capsize=3)
        ax.errorbar(alphas, agg["rand_misclass_mean"], yerr=agg["rand_misclass_std"], fmt="s-", color="C2",
                    label="Random", linewidth=2, markersize=4, capsize=3)
        ax.set_xlabel("Mask strength α", fontsize=10)
        ax.set_ylabel(f"P(pred={target_class})", fontsize=10)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        annotate_point(ax, "Mis", float(agg["adv_misclass_mean"][alpha_idx]), float(agg["adv_misclass_std"][alpha_idx]))

    return fig


def generate_figure_7(
    mnist_checkpoints_dir: Path,
    seeds: List[int],
    device: str = "cpu",
    target_class: int = 3,
    alphas: Optional[List[float]] = None,
    top_k_pinv: int = 10,
    target_ranks: Optional[List[int]] = None,
    alpha_annotate: float = 2.0,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Generate paper-style Figure 7 (adversarial robustness).
    
    Args:
        mnist_checkpoints_dir: Directory containing MNIST checkpoints
        seeds: List of random seeds to use
        device: Device for computation
        target_class: Target digit for adversarial attack
        alphas: List of perturbation strengths (default: paper values)
        top_k_pinv: Number of top eigenvectors for pseudoinverse
        target_ranks: Eigenvector ranks to average over (default: [0, 1, 2])
        alpha_annotate: Alpha value to highlight in annotations
        
    Returns:
        Tuple of (figure, summary_dict)
    """
    import sys
    from pathlib import Path
    
    # Add original code path
    project_root = Path(__file__).parent.parent.parent
    orig_path = project_root / "bilinear-decomposition-main"
    if str(orig_path) not in sys.path:
        sys.path.insert(0, str(orig_path))
    
    from image.datasets import MNIST
    from image.model import Model, Config
    from src.vision.adversarial import compute_rare_edge_pixel_mask
    
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    if target_ranks is None:
        target_ranks = [0, 1, 2]
    
    # Load test data
    test_data = MNIST(train=False, device=device)
    x = test_data.x.flatten(start_dim=1)
    y = test_data.y
    
    # Rare-edge constraint mask for no-reg model
    rare_edge_mask = compute_rare_edge_pixel_mask(x.detach().cpu()).to(device)
    
    # Run across seeds
    noise_curves_all: List[Dict[str, List[float]]] = []
    noreg_curves_all: List[Dict[str, List[float]]] = []
    
    viz_noise = None
    viz_noreg = None
    
    for seed in seeds:
        # A) Noise regularization (sigma=0.15)
        ckpt_noise_path = mnist_checkpoints_dir / f"mnist_dense_noise015_seed{seed}.pt"
        ckpt_noreg_path = mnist_checkpoints_dir / f"mnist_dense_none_seed{seed}.pt"
        
        if not ckpt_noise_path.exists() or not ckpt_noreg_path.exists():
            continue
        
        # Load noise-reg model
        ckpt_noise = torch.load(ckpt_noise_path, map_location=device, weights_only=False)
        eigenvalues_noise = ckpt_noise["eigenvalues"].to(device)
        eigenvectors_noise = ckpt_noise["eigenvectors"].to(device)
        cfg_noise = Config(d_hidden=int(ckpt_noise["config"]["d_hidden"]), epochs=1, seed=seed)
        model_noise = Model(cfg_noise).to(device)
        model_noise.load_state_dict(ckpt_noise["model_state_dict"])
        model_noise.eval()
        
        noise_curves_all.append(compute_adversarial_curves(
            model_noise, eigenvalues_noise, eigenvectors_noise,
            x, y, target_class, alphas, target_ranks, top_k_pinv,
            restrict_mask=None
        ))
        
        # Load no-reg model
        ckpt_noreg = torch.load(ckpt_noreg_path, map_location=device, weights_only=False)
        eigenvalues_noreg = ckpt_noreg["eigenvalues"].to(device)
        eigenvectors_noreg = ckpt_noreg["eigenvectors"].to(device)
        cfg_noreg = Config(d_hidden=int(ckpt_noreg["config"]["d_hidden"]), epochs=1, seed=seed)
        model_noreg = Model(cfg_noreg).to(device)
        model_noreg.load_state_dict(ckpt_noreg["model_state_dict"])
        model_noreg.eval()
        
        noreg_curves_all.append(compute_adversarial_curves(
            model_noreg, eigenvalues_noreg, eigenvectors_noreg,
            x, y, target_class, alphas, target_ranks, top_k_pinv,
            restrict_mask=rare_edge_mask
        ))
        
        # Save visualization artifacts from first seed
        if viz_noise is None:
            viz_noise = get_visualization_artifacts(
                model_noise, eigenvalues_noise, eigenvectors_noise,
                x, y, target_class, top_k_pinv, alpha_annotate,
                restrict_mask=None
            )
            viz_noreg = get_visualization_artifacts(
                model_noreg, eigenvalues_noreg, eigenvectors_noreg,
                x, y, target_class, top_k_pinv, alpha_annotate,
                restrict_mask=rare_edge_mask
            )
    
    if len(noise_curves_all) == 0 or len(noreg_curves_all) == 0:
        raise RuntimeError("No adversarial results computed (missing checkpoints?)")
    
    agg_noise = aggregate_curves(noise_curves_all)
    agg_noreg = aggregate_curves(noreg_curves_all)
    
    # Create figure
    fig = plot_adversarial_figure(
        agg_noise, agg_noreg,
        viz_noise, viz_noreg,
        alphas, target_class, alpha_annotate
    )
    
    # Summary info
    alpha_idx = int(np.argmin(np.abs(np.array(alphas) - alpha_annotate)))
    summary = {
        "noise_acc_at_alpha": float(agg_noise["adv_acc_mean"][alpha_idx]),
        "noise_acc_std": float(agg_noise["adv_acc_std"][alpha_idx]),
        "noreg_acc_at_alpha": float(agg_noreg["adv_acc_mean"][alpha_idx]),
        "noreg_acc_std": float(agg_noreg["adv_acc_std"][alpha_idx]),
        "alpha_annotate": alpha_annotate,
        "n_seeds_noise": len(noise_curves_all),
        "n_seeds_noreg": len(noreg_curves_all),
    }
    
    return fig, summary
