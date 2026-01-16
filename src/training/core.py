"""
Core training utilities shared across vision experiments.

This module provides reusable training functions for both MNIST and EMNIST
to avoid code duplication between src/train.py and src/train_emnist.py.
"""

import torch
from einops import einsum
import kornia
from typing import Optional, Tuple

from src.utils import set_seed, setup_mps_fallbacks, is_mps_device


def decompose_model_mps_safe(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose model into eigenvalues and eigenvectors with MPS compatibility.

    This is a reimplementation of model.decompose() that handles MPS devices
    by moving tensors to CPU for eigendecomposition.

    Args:
        model: Trained bilinear Model instance

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    device = next(model.parameters()).device

    # Get model weights
    w_u = model.w_u  # [cls, out]
    w_lr = model.w_lr[0]  # [2, out, hidden]
    w_e = model.w_e  # [hidden, input]

    l, r = w_lr.unbind(0)  # Each: [out, hidden]

    # Compute third-order tensor: b[cls, in1, in2]
    b = einsum(w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")

    # Symmetrize
    b = 0.5 * (b + b.mT)

    # Eigendecomposition - move to CPU for MPS compatibility
    if device.type == "mps":
        b_cpu = b.cpu()
        vals, vecs = torch.linalg.eigh(b_cpu)
        vals = vals.to(device)
        vecs = vecs.to(device)
    else:
        vals, vecs = torch.linalg.eigh(b)

    # Project eigenvectors back to input space
    vecs = einsum(vecs, w_e, "cls emb comp, emb inp -> cls comp inp")

    return vals, vecs


def apply_variance_corrected_init(model, enabled: bool = True):
    """
    Apply variance-corrected initialization to push model into Rich Training regime.

    Computes per-layer scaling based on input dimension: scale = d_in^0.25
    This prevents the "Lazy Training" pathology where the baseline collapses to low rank.

    Args:
        model: Model instance with w_lr (bilinear weights) and w_e (embedding)
        enabled: If False, skip scaling (equivalent to init_scale=1.0)
    """
    if not enabled:
        return

    with torch.no_grad():
        # Scale embedding layer: w_e has shape [d_hidden, d_input]
        if hasattr(model, 'w_e') and model.w_e is not None:
            d_in = model.w_e.shape[-1]  # Input dimension (784)
            scale = d_in ** 0.25
            model.w_e.data *= scale
            print(f"  w_e: d_in={d_in}, scale={scale:.2f}")

        # Scale bilinear layer: w_lr has shape [n_layers, 2, d_hidden, d_hidden]
        if hasattr(model, 'w_lr') and model.w_lr is not None:
            d_in = model.w_lr.shape[-1]  # Input dimension (256)
            scale = d_in ** 0.25
            model.w_lr.data *= scale
            print(f"  w_lr: d_in={d_in}, scale={scale:.2f}")


def create_noise_transform(noise_std: float) -> Optional[kornia.augmentation.RandomGaussianNoise]:
    """
    Create noise augmentation transform.
    
    Args:
        noise_std: Standard deviation of Gaussian noise (0 = no noise)
        
    Returns:
        Kornia transform or None if noise_std is 0
    """
    if noise_std > 0:
        return kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    return None


def train_model(
    model,
    train_data,
    test_data,
    device: str,
    seed: int,
    epochs: int,
    noise_std: float = 0.0,
    variance_corrected_init: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """
    Train a bilinear model with optional noise augmentation.
    
    Args:
        model: Model instance
        train_data: Training dataset
        test_data: Test dataset
        device: Device to train on
        seed: Random seed
        epochs: Number of training epochs
        noise_std: Standard deviation for Gaussian noise augmentation
        variance_corrected_init: Whether to apply variance correction
        
    Returns:
        Tuple of (eigenvalues, eigenvectors, history)
    """
    set_seed(seed)
    
    # Setup MPS fallbacks if needed
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - using CPU fallback for eigendecomposition")
    
    # Apply variance-corrected initialization
    if variance_corrected_init:
        print("Applying variance-corrected initialization (Rich Training regime):")
        apply_variance_corrected_init(model, enabled=True)
    
    # Create transform (noise augmentation)
    transform = create_noise_transform(noise_std)
    
    # Train
    print(f"Training for {epochs} epochs...")
    history = model.fit(train_data, test_data, transform=transform)
    
    # Compute eigendecomposition (original paper's method, now MPS-safe)
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = model.decompose()
    
    return eigenvalues, eigenvectors, history
