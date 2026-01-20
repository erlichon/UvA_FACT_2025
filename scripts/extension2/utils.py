#!/usr/bin/env python3
"""
Shared utilities for Extension 2 analysis scripts.

Provides helpers for:
- Balanced eigenvector selection (top-k with positive/negative balance)
- Absolute cosine similarity between eigenvector sets
- Mean absolute cosine similarity (max per rank, then mean)
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def get_balanced_eigenvectors_and_vals(
    vals: Tensor,
    vecs: Tensor,
    class_idx: int,
    k: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Get balanced eigenvectors and corresponding eigenvalues for a given class.

    Selects k//2 positive and k - k//2 negative eigenvectors, each by highest
    absolute eigenvalue magnitude, then fills any remaining slots with the next
    largest-magnitude eigenvectors regardless of sign.
    """
    class_vals = vals[class_idx].cpu()
    class_vecs = vecs[class_idx].cpu()

    pos_idx = torch.nonzero(class_vals > 0, as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(class_vals < 0, as_tuple=False).squeeze(-1)

    pos_sorted = (
        pos_idx[class_vals[pos_idx].abs().argsort(descending=True)]
        if pos_idx.numel() > 0
        else pos_idx
    )
    neg_sorted = (
        neg_idx[class_vals[neg_idx].abs().argsort(descending=True)]
        if neg_idx.numel() > 0
        else neg_idx
    )

    k_pos = k // 2
    k_neg = k - k_pos

    selected_idx = []
    if k_pos > 0 and pos_sorted.numel() > 0:
        selected_idx.extend(pos_sorted[:k_pos].tolist())
    if k_neg > 0 and neg_sorted.numel() > 0:
        selected_idx.extend(neg_sorted[:k_neg].tolist())

    # Fill remaining with largest-magnitude eigenvectors of any sign
    if len(selected_idx) < k:
        selected_mask = torch.zeros_like(class_vals, dtype=torch.bool)
        if selected_idx:
            selected_mask[selected_idx] = True
        remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
        if remaining.numel() > 0:
            remaining_sorted = remaining[
                class_vals[remaining].abs().argsort(descending=True)
            ]
            needed = k - len(selected_idx)
            selected_idx.extend(remaining_sorted[:needed].tolist())

    return class_vecs[selected_idx], class_vals[selected_idx]


def abs_cosine_similarity(vecs_A: Tensor, vecs_B: Tensor) -> Tensor:
    """Compute absolute cosine similarity matrix between two sets of eigenvectors.

    Args:
        vecs_A: Tensor of shape [k_A, d]
        vecs_B: Tensor of shape [k_B, d]

    Returns:
        Tensor of shape [k_A, k_B] with absolute cosine similarities.
    """
    vecs_A_norm = vecs_A / (vecs_A.norm(dim=1, keepdim=True) + 1e-10)
    vecs_B_norm = vecs_B / (vecs_B.norm(dim=1, keepdim=True) + 1e-10)
    cos_matrix = vecs_A_norm @ vecs_B_norm.T
    return cos_matrix.abs()


def compute_mean_abs_cosine_similarity(vecs_A: Tensor, vecs_B: Tensor) -> float:
    """Mean absolute cosine similarity using max-per-rank then mean.

    For each eigenvector in A, finds the maximum similarity with any eigenvector in B,
    then averages those maxima across all ranks.
    """
    sim_matrix = abs_cosine_similarity(vecs_A, vecs_B)
    max_sims_per_rank = sim_matrix.max(dim=1)[0]
    return max_sims_per_rank.mean().item()

