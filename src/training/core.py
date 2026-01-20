"""
Core training utilities shared across vision experiments.

This module provides reusable training functions for MNIST, EMNIST, and USPS
to avoid code duplication. All training scripts should use these utilities.
"""

import torch
import kornia
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import wandb

from src.utils import set_seed, setup_mps_fallbacks, is_mps_device, get_history_column
from src.vision.spectral import effective_rank, spectral_summary


# NOTE: decompose_model_mps_safe() was removed - use model.decompose() directly.
# The original paper code (bilinear-decomposition-main/image/model.py) already
# handles MPS by moving to CPU for torch.linalg.eigh() operations.


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


def log_training_history(history, wandb_enabled: bool) -> None:
    """
    Log per-epoch training metrics to wandb.
    
    Args:
        history: Training history DataFrame from model.fit()
        wandb_enabled: Whether wandb logging is active
    """
    if not wandb_enabled or wandb.run is None:
        return

    # Get column names (handle both naming conventions)
    train_acc_col = 'train/acc' if 'train/acc' in history.columns else 'train_acc'
    val_acc_col = 'val/acc' if 'val/acc' in history.columns else ('test/acc' if 'test/acc' in history.columns else 'val_acc')
    train_loss_col = 'train/loss' if 'train/loss' in history.columns else 'train_loss'
    val_loss_col = 'val/loss' if 'val/loss' in history.columns else ('test/loss' if 'test/loss' in history.columns else 'val_loss')

    # Log each epoch
    for epoch in range(len(history)):
        metrics = {"epoch": epoch}
        if train_acc_col in history.columns:
            metrics["train/acc"] = history[train_acc_col].iloc[epoch]
        if val_acc_col in history.columns:
            metrics["val/acc"] = history[val_acc_col].iloc[epoch]
        if train_loss_col in history.columns:
            metrics["train/loss"] = history[train_loss_col].iloc[epoch]
        if val_loss_col in history.columns:
            metrics["val/loss"] = history[val_loss_col].iloc[epoch]
        wandb.log(metrics, step=epoch)


def log_spectral_metrics(eigenvalues: torch.Tensor, wandb_enabled: bool) -> Dict[str, float]:
    """
    Log comprehensive spectral metrics to wandb summary.
    
    Args:
        eigenvalues: Eigenvalues tensor [n_classes, d_hidden]
        wandb_enabled: Whether wandb logging is active
        
    Returns:
        Dictionary of spectral metrics (for use in checkpoint)
    """
    if not wandb_enabled or wandb.run is None:
        return {}

    # Compute full spectral summary
    summary = spectral_summary(eigenvalues)

    # Compute per-class effective rank
    per_class_eff_rank = effective_rank(eigenvalues)

    # Add per-class metrics
    for cls_idx in range(len(per_class_eff_rank)):
        summary[f"effective_rank_class_{cls_idx}"] = per_class_eff_rank[cls_idx].item()

    # Log eigenvalue histogram for each class (first 3 classes to avoid clutter)
    for cls_idx in range(min(3, eigenvalues.shape[0])):
        cls_eigenvalues = eigenvalues[cls_idx].abs().detach().cpu().numpy()
        wandb.run.summary[f"eigenvalues_class_{cls_idx}"] = wandb.Histogram(cls_eigenvalues)

    return summary


def save_checkpoint(
    path: Path,
    config: dict,
    model,
    history,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    seed: int,
    epochs: int,
    emissions: dict = None,
) -> Dict[str, Any]:
    """
    Save model checkpoint with eigenspectrum data.
    
    Supports both dense and CP models with appropriate config fields.
    
    Args:
        path: Output path for checkpoint file
        config: Experiment configuration dict
        model: Trained model instance
        history: Training history DataFrame
        eigenvalues: Eigenvalues tensor [n_classes, d_hidden] or [n_classes, d_input] for CP
        eigenvectors: Eigenvectors tensor [n_classes, d_hidden, d_input] or [n_classes, d_input, d_input] for CP
        seed: Random seed used
        epochs: Number of training epochs
        emissions: Optional dict with CodeCarbon emissions data (co2_kg, wall_time_hours, gpu_hours)
        
    Returns:
        Checkpoint dictionary (also saved to disk)
    """
    # Extract final metrics from history
    final_train_acc = get_history_column(history, 'train_acc', 'train/acc')
    final_val_acc = get_history_column(history, 'val_acc', 'val/acc', 'test_acc', 'test/acc')
    final_train_loss = get_history_column(history, 'train_loss', 'train/loss')
    final_val_loss = get_history_column(history, 'val_loss', 'val/loss', 'test_loss', 'test/loss')

    # Compute effective rank
    eff_rank = effective_rank(eigenvalues).mean().item()

    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    mode = config.get('model', {}).get('mode', 'dense')

    # Build config dict with common fields
    checkpoint_config = {
        'mode': mode,
        'd_hidden': config['model']['d_hidden'],
        'epochs': epochs,
        'lr': config['training'].get('lr', 1e-3),
        'noise_std': config['regularization'].get('noise_std', 0.0),
        'weight_decay': config['regularization']['weight_decay'],
        'dataset': dataset_name,
        'variance_corrected_init': config.get('model', {}).get('variance_corrected_init', False),
        'apply_com': config.get('data', {}).get('apply_com', False),
    }
    
    # Add CP-specific fields if in CP mode
    if mode == 'cp':
        checkpoint_config.update({
            'rank': config['model']['rank'],
            'cp_init_mode': config['model'].get('cp_init_mode', 'lambda'),
            'l1_coeff': config['regularization'].get('l1_coeff', 0.0),
            'lambda_l1_coeff': config['regularization'].get('lambda_l1_coeff', 0.0),
            'lambda_l0_coeff': config['regularization'].get('lambda_l0_coeff', 0.0),
        })

    checkpoint = {
        'config': checkpoint_config,
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': float(final_train_acc),
            'val_acc': float(final_val_acc),
            'train_loss': float(final_train_loss),
            'val_loss': float(final_val_loss),
            'effective_rank': float(eff_rank),
        },
        'seed': seed,
        'eigenvalues': eigenvalues.detach().cpu(),
        'eigenvectors': eigenvectors.detach().cpu(),
        'emissions': emissions,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint
