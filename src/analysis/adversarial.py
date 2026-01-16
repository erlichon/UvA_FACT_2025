"""
Adversarial mask construction for Figure 7.

This module implements adversarial attacks using eigenvector-based masks.
The key insight is that the pseudoinverse of the top eigenvectors gives
masks that maximally activate specific eigenvalue components.
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


def compute_adversarial_mask(
    eigenvectors: Float[Tensor, "d_hidden d_input"],
    eigenvalues: Float[Tensor, "d_hidden"],
    target_rank: int = 0,
    top_k: int = 10,
    use_positive_only: bool = True,
) -> Float[Tensor, "d_input"]:
    """
    Compute adversarial mask using pseudoinverse of top eigenvectors.
    
    Following the paper's approach (Section 4.4):
    - Use top-k POSITIVE eigenvectors (not by magnitude)
    - Compute pseudoinverse V^+
    - Mask_i = (V^+)_{i:} activates eigenvector i specifically
    
    Args:
        eigenvectors: Eigenvectors for a single class, shape [d_hidden, d_input]
        eigenvalues: Eigenvalues for the same class, shape [d_hidden]
        target_rank: Which eigenvector to target (0 = top positive eigenvector)
        top_k: Number of top eigenvectors to consider in pseudoinverse
        use_positive_only: If True, use only positive eigenvalues (paper's method)
        
    Returns:
        Adversarial mask, shape [d_input]
    """
    if use_positive_only:
        # Get top-k POSITIVE eigenvectors (paper's method)
        positive_mask = eigenvalues > 0
        positive_vals = eigenvalues.clone()
        positive_vals[~positive_mask] = -float('inf')
        _, top_indices = torch.topk(positive_vals, k=min(top_k, positive_mask.sum().item()))
    else:
        # Get top-k eigenvectors by eigenvalue magnitude
        _, top_indices = torch.topk(eigenvalues.abs(), k=top_k)
    
    top_vecs = eigenvectors[top_indices, :]  # [k, d_input]
    
    # Compute pseudoinverse
    # V^+ has shape [d_input, k]
    V_pinv = torch.linalg.pinv(top_vecs)  # [d_input, k]
    
    # The mask that activates eigenvector at target_rank
    mask = V_pinv[:, target_rank]
    
    return mask


def compute_class_adversarial_masks(
    eigenvectors: Float[Tensor, "classes d_hidden d_input"],
    eigenvalues: Float[Tensor, "classes d_hidden"],
    top_k: int = 10,
) -> Float[Tensor, "classes d_input"]:
    """
    Compute adversarial masks for all classes (targeting top eigenvector).
    
    Args:
        eigenvectors: Per-class eigenvectors
        eigenvalues: Per-class eigenvalues
        top_k: Number of eigenvectors to consider
        
    Returns:
        Adversarial masks, shape [classes, d_input]
    """
    n_classes = eigenvectors.shape[0]
    d_input = eigenvectors.shape[2]
    
    masks = torch.zeros(n_classes, d_input)
    
    for c in range(n_classes):
        masks[c] = compute_adversarial_mask(
            eigenvectors[c], eigenvalues[c],
            target_rank=0, top_k=top_k,
        )
    
    return masks


def apply_adversarial_perturbation(
    x: Float[Tensor, "batch d_input"],
    mask: Float[Tensor, "d_input"],
    alpha: float = 0.1,
) -> Float[Tensor, "batch d_input"]:
    """
    Apply adversarial perturbation to input images.
    
    Args:
        x: Input images, shape [batch, d_input]
        mask: Adversarial mask, shape [d_input]
        alpha: Perturbation strength
        
    Returns:
        Perturbed images, shape [batch, d_input]
    """
    # Normalize mask to unit norm
    mask_normalized = mask / mask.norm().clamp(min=1e-10)
    
    # Apply perturbation
    return x + alpha * mask_normalized


def evaluate_adversarial_attack(
    x: Float[Tensor, "batch d_input"],
    y: Float[Tensor, "batch"],
    model,
    masks: Float[Tensor, "classes d_input"],
    target_class: int,
    alphas: List[float] = None,
) -> pd.DataFrame:
    """
    Evaluate adversarial attack effectiveness.
    
    Args:
        x: Test images
        y: True labels
        model: Trained model
        masks: Adversarial masks per class
        target_class: Target class for misclassification
        alphas: List of perturbation strengths
        
    Returns:
        DataFrame with accuracy and misclassification rates per alpha
    """
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    mask = masks[target_class]
    results = []
    
    for alpha in alphas:
        # Apply perturbation
        x_perturbed = apply_adversarial_perturbation(x, mask, alpha)
        
        # Get predictions
        with torch.no_grad():
            logits = model(x_perturbed.reshape(-1, 28, 28))
            preds = logits.argmax(dim=-1)
        
        # Compute metrics
        accuracy = (preds == y).float().mean().item()
        misclass_as_target = (preds == target_class).float().mean().item()
        
        results.append({
            'alpha': alpha,
            'accuracy': accuracy,
            'misclass_as_target': misclass_as_target,
        })
    
    return pd.DataFrame(results)


def compute_random_baseline_mask(
    mask: Float[Tensor, "d_input"],
    seed: int = 42,
) -> Float[Tensor, "d_input"]:
    """
    Create a random baseline mask by permuting the adversarial mask.
    
    This preserves the norm and distribution of values but removes
    the spatial structure.
    
    Args:
        mask: Original adversarial mask
        seed: Random seed for reproducibility
        
    Returns:
        Randomly permuted mask
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(mask))
    return mask[perm]


def compute_rare_edge_pixel_mask(
    x: Float[Tensor, "batch d_input"],
    image_shape: Tuple[int, int] = (28, 28),
    border: int = 3,
    active_threshold: float = 0.0,
    rare_threshold: float = 0.01,
) -> Float[Tensor, "d_input"]:
    """
    Compute the paper-style constraint mask for Figure 7B:
    keep only outer-edge pixels that are active on <1% of samples.

    Paper text (Figure 7 caption): \"mask is only applied to the outer edge of pixels
    that are active on less than 1% of samples.\"

    Args:
        x: Dataset images flattened [batch, d_input] in [0,1].
        image_shape: (H, W), default MNIST 28x28.
        border: Edge thickness in pixels.
        active_threshold: Pixel is considered \"active\" if value > threshold.
        rare_threshold: Pixel is \"rare\" if active frequency < threshold.

    Returns:
        Boolean-like float mask in {0,1} with shape [d_input].
    """
    h, w = image_shape
    d_input = h * w
    if x.shape[-1] != d_input:
        raise ValueError(f"Expected x.shape[-1]=={d_input}, got {x.shape[-1]}")

    # Frequency of activation per pixel.
    active = (x > active_threshold).float()
    freq = active.mean(dim=0)  # [d_input]
    rare = freq < rare_threshold

    # Outer edge mask.
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    edge = (yy < border) | (yy >= h - border) | (xx < border) | (xx >= w - border)
    edge = edge.reshape(-1).to(rare.device)

    keep = rare & edge
    return keep.float()


def compare_adversarial_vs_random(
    x: Float[Tensor, "batch d_input"],
    y: Float[Tensor, "batch"],
    model,
    eigenvectors: Float[Tensor, "classes d_hidden d_input"],
    eigenvalues: Float[Tensor, "classes d_hidden"],
    target_class: int = 0,
    alphas: List[float] = None,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare adversarial attack vs random baseline.
    
    Args:
        x: Test images
        y: True labels
        model: Trained model
        eigenvectors: Per-class eigenvectors
        eigenvalues: Per-class eigenvalues
        target_class: Target class for attack
        alphas: Perturbation strengths
        top_k: Number of eigenvectors for mask construction
        
    Returns:
        Tuple of (adversarial_results, random_results) DataFrames
    """
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    # Compute adversarial mask
    adv_mask = compute_adversarial_mask(
        eigenvectors[target_class], eigenvalues[target_class],
        target_rank=0, top_k=top_k,
    )
    
    # Compute random baseline
    random_mask = compute_random_baseline_mask(adv_mask)
    
    # Evaluate both
    adv_results = []
    random_results = []
    
    for alpha in alphas:
        # Adversarial
        x_adv = apply_adversarial_perturbation(x, adv_mask, alpha)
        with torch.no_grad():
            logits_adv = model(x_adv.reshape(-1, 28, 28))
            preds_adv = logits_adv.argmax(dim=-1)
        
        adv_results.append({
            'alpha': alpha,
            'accuracy': (preds_adv == y).float().mean().item(),
            'misclass_as_target': (preds_adv == target_class).float().mean().item(),
        })
        
        # Random baseline
        x_random = apply_adversarial_perturbation(x, random_mask, alpha)
        with torch.no_grad():
            logits_random = model(x_random.reshape(-1, 28, 28))
            preds_random = logits_random.argmax(dim=-1)
        
        random_results.append({
            'alpha': alpha,
            'accuracy': (preds_random == y).float().mean().item(),
            'misclass_as_target': (preds_random == target_class).float().mean().item(),
        })
    
    return pd.DataFrame(adv_results), pd.DataFrame(random_results)


def run_adversarial_experiment(
    checkpoint_path: str,
    device: str = "cpu",
    target_class: int = 0,
    top_k: int = 10,
) -> Dict:
    """
    Run full adversarial experiment on a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for computation
        target_class: Target class for attack
        top_k: Number of eigenvectors for mask
        
    Returns:
        Dict with results and masks
    """
    import sys
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))
    
    from image.model import Model, Config
    from image.datasets import MNIST
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eigenvalues = ckpt['eigenvalues']
    eigenvectors = ckpt['eigenvectors']
    
    # Recreate model
    config = Config(
        d_hidden=ckpt['config']['d_hidden'],
        epochs=1,
        seed=ckpt.get('seed', 42),
    )
    model = Model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load test data
    test_data = MNIST(train=False, device=device)
    x = test_data.x.flatten(start_dim=1)
    y = test_data.y
    
    # Run comparison
    adv_df, random_df = compare_adversarial_vs_random(
        x, y, model,
        eigenvectors, eigenvalues,
        target_class=target_class,
        top_k=top_k,
    )
    
    # Compute mask for visualization
    adv_mask = compute_adversarial_mask(
        eigenvectors[target_class], eigenvalues[target_class],
        target_rank=0, top_k=top_k,
    )
    
    return {
        'adversarial_results': adv_df,
        'random_results': random_df,
        'mask': adv_mask,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
    }
