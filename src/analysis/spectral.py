"""
Spectral analysis utilities for bilinear layer interpretability.

This module provides functions for computing effective rank and other
spectral metrics used to evaluate the interpretability of bilinear models.
"""

import torch
from torch import Tensor
from jaxtyping import Float
from typing import Union


def effective_rank(eigenvalues: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """
    Compute entropy-based effective rank (Roy & Bhattacharyya 2007).

    The effective rank measures how "spread out" the eigenvalue distribution is.
    Lower values indicate a sharper spectrum (more interpretable).

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]

    Returns:
        Effective rank tensor of shape [...]
    """
    # Use absolute values (eigenvalues can be negative)
    abs_vals = eigenvalues.abs()

    # Normalize to probability distribution
    p = abs_vals / abs_vals.sum(dim=-1, keepdim=True)

    # Avoid log(0)
    p = p.clamp(min=1e-10)

    # Compute entropy
    entropy = -(p * p.log()).sum(dim=-1)

    # Effective rank is exp(entropy)
    return entropy.exp()


def top_k_coverage(eigenvalues: Float[Tensor, "... n"], k: int = 5) -> Float[Tensor, "..."]:
    """
    Compute fraction of variance explained by top-k eigenvalues.

    Higher coverage with smaller k indicates more interpretable structure.

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]
        k: Number of top eigenvalues to consider

    Returns:
        Coverage ratio tensor of shape [...]
    """
    abs_vals = eigenvalues.abs()

    # Sort in descending order
    sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)

    # Sum of top-k
    top_k_sum = sorted_vals[..., :k].sum(dim=-1)

    # Total sum
    total_sum = sorted_vals.sum(dim=-1)

    return top_k_sum / total_sum.clamp(min=1e-10)


def eigenvalue_decay_rate(eigenvalues: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """
    Compute the decay rate of the eigenvalue spectrum.

    Faster decay indicates more low-rank structure.

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]

    Returns:
        Decay rate (ratio of 2nd to 1st eigenvalue)
    """
    abs_vals = eigenvalues.abs()
    sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)

    # Ratio of second to first (lower = faster decay)
    return sorted_vals[..., 1] / sorted_vals[..., 0].clamp(min=1e-10)


def spectral_summary(eigenvalues: Float[Tensor, "classes n"]) -> dict:
    """
    Compute a summary of spectral metrics across all classes.

    Args:
        eigenvalues: Eigenvalue tensor of shape [classes, n]

    Returns:
        Dictionary with mean and std of various metrics
    """
    eff_rank = effective_rank(eigenvalues)
    top5_cov = top_k_coverage(eigenvalues, k=5)
    top10_cov = top_k_coverage(eigenvalues, k=10)
    decay = eigenvalue_decay_rate(eigenvalues)

    return {
        'effective_rank_mean': eff_rank.mean().item(),
        'effective_rank_std': eff_rank.std().item(),
        'top5_coverage_mean': top5_cov.mean().item(),
        'top5_coverage_std': top5_cov.std().item(),
        'top10_coverage_mean': top10_cov.mean().item(),
        'top10_coverage_std': top10_cov.std().item(),
        'decay_rate_mean': decay.mean().item(),
        'decay_rate_std': decay.std().item(),
    }


def load_checkpoint_eigenvalues(checkpoint_path: str) -> tuple[Tensor, Tensor]:
    """
    Load eigenvalues and eigenvectors from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint .pt file

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['eigenvalues'], checkpoint['eigenvectors']
