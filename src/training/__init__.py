"""
Training utilities for bilinear MLP experiments.

NOTE: decompose_model_mps_safe() was removed - use model.decompose() directly.
The original paper code already handles MPS compatibility.
"""

from .core import (
    apply_variance_corrected_init,
    create_noise_transform,
    train_model,
    log_training_history,
    log_spectral_metrics,
    save_checkpoint,
)

__all__ = [
    "apply_variance_corrected_init",
    "create_noise_transform",
    "train_model",
    "log_training_history",
    "log_spectral_metrics",
    "save_checkpoint",
]
