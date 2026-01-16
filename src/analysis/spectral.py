"""
Spectral analysis utilities for bilinear layer interpretability.

This module provides functions for computing effective rank and other
spectral metrics used to evaluate the interpretability of bilinear models.
"""

import torch
from torch import Tensor
from jaxtyping import Float


def effective_rank(eigenvalues: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """
    Compute ratio-based effective rank: (||lambda||_1 / ||lambda||_2)^2

    This matches the original paper implementation in sae/functions.py.
    The effective rank measures how "spread out" the eigenvalue distribution is.
    Lower values indicate a sharper spectrum (more interpretable).

    Formula: (L1 / L2)^2 where L1 = sum(|lambda|), L2 = sqrt(sum(lambda^2))

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]

    Returns:
        Effective rank tensor of shape [...]
    """
    abs_vals = eigenvalues.abs()

    # L1 norm: sum of absolute values
    l1 = abs_vals.sum(dim=-1)

    # L2 norm: sqrt of sum of squares
    l2 = abs_vals.pow(2).sum(dim=-1).sqrt()

    # Effective rank = (L1/L2)^2
    return (l1 / l2.clamp(min=1e-10)).pow(2)


def effective_rank_entropy(eigenvalues: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """
    Compute entropy-based effective rank (Roy & Bhattacharyya 2007).

    NOTE: This is NOT the formula used in the original paper. It is kept for
    reference and comparison. Use effective_rank() for paper-compatible results.

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


def kurtosis(eigenvalues: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """
    Compute excess kurtosis of the eigenvalue distribution.

    Kurtosis measures the "tailedness" of the distribution.
    Higher kurtosis indicates more outlier eigenvalues.

    Formula: E[(X - mu)^4] / sigma^4 - 3 (excess kurtosis)

    This matches the original paper implementation in sae/functions.py.

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]

    Returns:
        Excess kurtosis tensor of shape [...]
    """
    # Use absolute values for consistency with other metrics
    vals = eigenvalues.abs()

    # Compute mean and std
    mean = vals.mean(dim=-1, keepdim=True)
    std = vals.std(dim=-1, keepdim=True).clamp(min=1e-10)

    # Standardize
    standardized = (vals - mean) / std

    # Fourth moment
    fourth_moment = standardized.pow(4).mean(dim=-1)

    # Excess kurtosis (subtract 3 for normal distribution baseline)
    return fourth_moment - 3


def truncated_eigenvalue_sum(
    eigenvalues: Float[Tensor, "... n"],
    k: int = 2
) -> Float[Tensor, "..."]:
    """
    Compute sum of top-k eigenvalues by magnitude.

    This is used to measure the "concentration" of eigenvalue mass.

    Args:
        eigenvalues: Eigenvalue tensor of shape [..., n]
        k: Number of top eigenvalues to sum

    Returns:
        Sum of top-k absolute eigenvalues
    """
    abs_vals = eigenvalues.abs()
    sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
    return sorted_vals[..., :k].sum(dim=-1)


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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['eigenvalues'], checkpoint['eigenvectors']


def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load full checkpoint dict from file.

    Args:
        checkpoint_path: Path to the checkpoint .pt file

    Returns:
        Full checkpoint dictionary
    """
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def load_all_checkpoints(
    checkpoint_dir: str,
    dataset: str = "mnist",
    configs: list = None,
    seeds: list = None,
) -> "pd.DataFrame":
    """
    Load metrics from all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        dataset: 'mnist' or 'fashion'
        configs: List of config names (default: ['none', 'noise', 'wd', 'full'])
        seeds: List of seeds (default: [42, 43, 44, 45, 46])

    Returns:
        DataFrame with columns: config, seed, accuracy, effective_rank, top5_coverage, etc.
    """
    import pandas as pd
    from pathlib import Path

    if configs is None:
        configs = ["none", "noise", "wd", "full"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    results = []
    checkpoint_dir = Path(checkpoint_dir)

    for config in configs:
        for seed in seeds:
            filename = f"{dataset}_dense_{config}_seed{seed}.pt"
            path = checkpoint_dir / filename

            if not path.exists():
                print(f"Warning: {path} not found")
                continue

            checkpoint = load_checkpoint(str(path))
            eigenvalues = checkpoint["eigenvalues"]

            # Compute spectral metrics
            summary = spectral_summary(eigenvalues)

            results.append({
                "config": config,
                "seed": seed,
                "accuracy": checkpoint["metrics"]["val_acc"],
                "train_accuracy": checkpoint["metrics"].get("train_acc", None),
                "effective_rank": summary["effective_rank_mean"],
                "effective_rank_std": summary["effective_rank_std"],
                "top5_coverage": summary["top5_coverage_mean"],
                "top10_coverage": summary["top10_coverage_mean"],
                "decay_rate": summary["decay_rate_mean"],
            })

    return pd.DataFrame(results)


def aggregate_by_config(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Aggregate results by configuration (mean +/- std across seeds).

    Args:
        df: DataFrame from load_all_checkpoints()

    Returns:
        Aggregated DataFrame with mean/std for each config
    """
    metrics = ["accuracy", "effective_rank", "top5_coverage", "top10_coverage", "decay_rate"]

    agg_dict = {}
    for m in metrics:
        if m in df.columns:
            agg_dict[m] = ["mean", "std"]

    aggregated = df.groupby("config").agg(agg_dict)
    aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
    aggregated = aggregated.reset_index()

    return aggregated


def compute_rank_ratio(df: "pd.DataFrame", baseline: str = "none", target: str = "full") -> float:
    """
    Compute effective rank ratio between two configurations.

    Args:
        df: DataFrame from load_all_checkpoints()
        baseline: Baseline config name (typically 'none')
        target: Target config name (typically 'full' or 'wd')

    Returns:
        Ratio of mean effective ranks (target / baseline)
    """
    baseline_mean = df[df["config"] == baseline]["effective_rank"].mean()
    target_mean = df[df["config"] == target]["effective_rank"].mean()
    return target_mean / baseline_mean
