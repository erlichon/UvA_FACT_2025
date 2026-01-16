"""
Truncation analysis utilities for Figure 5.

This module provides functions for:
1. Computing truncated predictions using only top-k eigenvalues
2. Computing eigenvector similarity across model sizes
3. Cross-seed and cross-size eigenvector comparisons
"""

import torch
from torch import Tensor
from jaxtyping import Float
from pathlib import Path
from itertools import combinations
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats


def truncated_prediction(
    x: Float[Tensor, "batch d_input"],
    eigenvalues: Float[Tensor, "classes d_hidden"],
    eigenvectors: Float[Tensor, "classes d_hidden d_input"],
    k: int,
) -> Float[Tensor, "batch"]:
    """
    Compute predictions using only top-k eigenvalues per class.
    
    Uses the formula: pred = argmax_c [ sum_{i=1}^{k} lambda_i^(c) * (v_i^(c)^T x)^2 ]
    
    Args:
        x: Input images, shape [batch, d_input]
        eigenvalues: Per-class eigenvalues, shape [classes, d_hidden]
        eigenvectors: Per-class eigenvectors, shape [classes, d_hidden, d_input]
        k: Number of top eigenvalues to retain (by magnitude)
        
    Returns:
        Predicted class labels, shape [batch]
    """
    n_classes = eigenvalues.shape[0]
    batch_size = x.shape[0]
    
    # For each class, get top-k eigenvalues by magnitude
    scores = torch.zeros(batch_size, n_classes, device=x.device)
    
    for c in range(n_classes):
        # Get top-k indices by magnitude
        abs_vals = eigenvalues[c].abs()
        _, top_indices = torch.topk(abs_vals, k=min(k, len(abs_vals)))
        
        # Get corresponding eigenvalues and eigenvectors
        top_lambdas = eigenvalues[c, top_indices]  # [k]
        top_vecs = eigenvectors[c, top_indices, :]  # [k, d_input]
        
        # Compute projections: (v^T x)^2
        projections = torch.einsum("ki,bi->bk", top_vecs, x)  # [batch, k]
        proj_squared = projections ** 2
        
        # Score = sum_i lambda_i * (v_i^T x)^2
        scores[:, c] = torch.einsum("bk,k->b", proj_squared, top_lambdas)
    
    return scores.argmax(dim=-1)


def truncated_accuracy(
    x: Float[Tensor, "batch d_input"],
    y: Float[Tensor, "batch"],
    eigenvalues: Float[Tensor, "classes d_hidden"],
    eigenvectors: Float[Tensor, "classes d_hidden d_input"],
    k: int,
) -> float:
    """
    Compute accuracy using truncated predictions with top-k eigenvalues.
    
    Args:
        x: Input images
        y: True labels
        eigenvalues: Per-class eigenvalues
        eigenvectors: Per-class eigenvectors
        k: Number of top eigenvalues to retain
        
    Returns:
        Accuracy as float
    """
    preds = truncated_prediction(x, eigenvalues, eigenvectors, k)
    return (preds == y).float().mean().item()


def truncated_error(
    x: Float[Tensor, "batch d_input"],
    y: Float[Tensor, "batch"],
    eigenvalues: Float[Tensor, "classes d_hidden"],
    eigenvectors: Float[Tensor, "classes d_hidden d_input"],
    k: int,
) -> float:
    """
    Compute classification error (1 - accuracy) using truncated predictions.
    
    For Paper Figure 5B: Error plotted on log scale.
    
    Args:
        x: Input images
        y: True labels
        eigenvalues: Per-class eigenvalues
        eigenvectors: Per-class eigenvectors
        k: Number of top eigenvalues to retain
        
    Returns:
        Classification error as float (1 - accuracy)
    """
    acc = truncated_accuracy(x, y, eigenvalues, eigenvectors, k)
    return 1.0 - acc


def compute_eigenvector_similarity(
    vecs1: Float[Tensor, "classes d_hidden d_input"],
    vecs2: Float[Tensor, "classes d_hidden d_input"],
    eigenvalues1: Float[Tensor, "classes d_hidden"],
    eigenvalues2: Float[Tensor, "classes d_hidden"],
    top_k: int = 10,
) -> float:
    """
    Compute mean cosine similarity between top eigenvectors of two models.
    
    Eigenvectors are ordered by eigenvalue magnitude, and we compare
    the top-k eigenvectors for each class.
    
    Args:
        vecs1, vecs2: Eigenvectors from two models
        eigenvalues1, eigenvalues2: Corresponding eigenvalues
        top_k: Number of top eigenvectors to compare
        
    Returns:
        Mean absolute cosine similarity (averaged over classes and top-k vectors)
    """
    n_classes = vecs1.shape[0]
    similarities = []
    
    for c in range(n_classes):
        # Sort eigenvectors by eigenvalue magnitude
        _, idx1 = torch.topk(eigenvalues1[c].abs(), k=top_k)
        _, idx2 = torch.topk(eigenvalues2[c].abs(), k=top_k)
        
        top_vecs1 = vecs1[c, idx1, :]  # [top_k, d_input]
        top_vecs2 = vecs2[c, idx2, :]  # [top_k, d_input]
        
        # Normalize
        top_vecs1 = top_vecs1 / top_vecs1.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        top_vecs2 = top_vecs2 / top_vecs2.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Compute cosine similarity for each pair (take absolute since sign is arbitrary)
        for i in range(top_k):
            sim = torch.abs(torch.dot(top_vecs1[i], top_vecs2[i]))
            similarities.append(sim.item())
    
    return np.mean(similarities)


def compute_similarity_by_rank(
    checkpoints: List[dict],
    max_rank: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity between eigenvectors at each rank across seeds.
    
    For Paper Figure 5A: Shows how similarity degrades for lower-ranked eigenvectors.
    
    Args:
        checkpoints: List of checkpoint dicts (different seeds, same size)
        max_rank: Maximum eigenvector rank to compare
        
    Returns:
        Tuple of (similarities_mean, similarities_std) arrays of shape [max_rank]
    """
    if len(checkpoints) < 2:
        return np.ones(max_rank), np.zeros(max_rank)
    
    # For each rank, collect similarities across all seed pairs
    similarities_by_rank = [[] for _ in range(max_rank)]
    
    for ckpt1, ckpt2 in combinations(checkpoints, 2):
        vecs1 = ckpt1['eigenvectors']
        vecs2 = ckpt2['eigenvectors']
        vals1 = ckpt1['eigenvalues']
        vals2 = ckpt2['eigenvalues']
        
        n_classes = vecs1.shape[0]
        
        for c in range(n_classes):
            # Sort by eigenvalue magnitude
            _, idx1 = torch.topk(vals1[c].abs(), k=min(max_rank, len(vals1[c])))
            _, idx2 = torch.topk(vals2[c].abs(), k=min(max_rank, len(vals2[c])))
            
            for rank in range(min(max_rank, len(idx1))):
                vec1 = vecs1[c, idx1[rank], :]
                vec2 = vecs2[c, idx2[rank], :]
                
                # Normalize
                vec1 = vec1 / vec1.norm().clamp(min=1e-10)
                vec2 = vec2 / vec2.norm().clamp(min=1e-10)
                
                # Absolute cosine similarity
                sim = torch.abs(torch.dot(vec1, vec2)).item()
                similarities_by_rank[rank].append(sim)
    
    # Aggregate
    means = np.array([np.mean(sims) if sims else 0.0 for sims in similarities_by_rank])
    stds = np.array([np.std(sims) if sims else 0.0 for sims in similarities_by_rank])
    
    return means, stds


def compute_within_size_similarity(
    checkpoints: List[dict],
    top_k: int = 10,
) -> Tuple[float, float]:
    """
    Compute mean eigenvector similarity across seeds for same model size.
    
    Args:
        checkpoints: List of checkpoint dicts with 'eigenvalues' and 'eigenvectors'
        top_k: Number of top eigenvectors to compare
        
    Returns:
        Tuple of (mean_similarity, std_similarity)
    """
    if len(checkpoints) < 2:
        return 1.0, 0.0
    
    similarities = []
    for ckpt1, ckpt2 in combinations(checkpoints, 2):
        sim = compute_eigenvector_similarity(
            ckpt1['eigenvectors'], ckpt2['eigenvectors'],
            ckpt1['eigenvalues'], ckpt2['eigenvalues'],
            top_k=top_k,
        )
        similarities.append(sim)
    
    return np.mean(similarities), np.std(similarities)


def load_size_sweep_checkpoints(
    checkpoint_dir: str,
    sizes: List[int] = None,
    seeds: List[int] = None,
) -> Dict[int, List[dict]]:
    """
    Load checkpoints from model size sweep.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        sizes: List of model sizes (d_hidden values)
        seeds: List of seeds
        
    Returns:
        Dict mapping size -> list of checkpoint dicts
    """
    if sizes is None:
        sizes = [30, 50, 100, 300, 500, 1000]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    
    checkpoint_dir = Path(checkpoint_dir)
    results = {}
    
    for size in sizes:
        results[size] = []
        for seed in seeds:
            filename = f"mnist_size_{size}_seed{seed}.pt"
            path = checkpoint_dir / filename
            
            if path.exists():
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                results[size].append(ckpt)
            else:
                print(f"Warning: {path} not found")
    
    return results


def compute_all_truncation_curves(
    test_x: Tensor,
    test_y: Tensor,
    checkpoints_by_size: Dict[int, List[dict]],
    truncation_levels: List[int] = None,
    return_error: bool = True,
) -> pd.DataFrame:
    """
    Compute truncation error curves for all model sizes.
    
    For Paper Figure 5B: Classification error vs truncation level.
    
    Args:
        test_x: Test images
        test_y: Test labels
        checkpoints_by_size: Dict mapping size -> list of checkpoints
        truncation_levels: List of k values for truncation
        return_error: If True, return classification error; if False, return accuracy
        
    Returns:
        DataFrame with columns: size, seed, k, error (or accuracy)
    """
    if truncation_levels is None:
        truncation_levels = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30]
    
    results = []
    
    for size, ckpts in checkpoints_by_size.items():
        for i, ckpt in enumerate(ckpts):
            seed = ckpt.get('seed', 42 + i)
            eigenvalues = ckpt['eigenvalues']
            eigenvectors = ckpt['eigenvectors']
            
            # Limit truncation levels to available eigenvalues per class
            max_k = min(eigenvalues.shape[1], 30)  # Paper goes to ~30
            
            for k in truncation_levels:
                if k > max_k:
                    continue
                    
                acc = truncated_accuracy(
                    test_x, test_y,
                    eigenvalues, eigenvectors,
                    k=k,
                )
                
                metric_value = (1.0 - acc) if return_error else acc
                metric_name = 'error' if return_error else 'accuracy'
                
                results.append({
                    'size': size,
                    'seed': seed,
                    'k': k,
                    metric_name: metric_value,
                })
    
    return pd.DataFrame(results)


def compute_similarity_matrix(
    checkpoints_by_size: Dict[int, List[dict]],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute within-size eigenvector similarity for all model sizes.
    
    Args:
        checkpoints_by_size: Dict mapping size -> list of checkpoints
        top_k: Number of top eigenvectors to compare
        
    Returns:
        DataFrame with columns: size, mean_similarity, std_similarity, n_comparisons
    """
    results = []
    
    for size, ckpts in checkpoints_by_size.items():
        mean_sim, std_sim = compute_within_size_similarity(ckpts, top_k=top_k)
        n_comparisons = len(list(combinations(range(len(ckpts)), 2)))
        
        results.append({
            'size': size,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'n_comparisons': n_comparisons,
        })
    
    return pd.DataFrame(results)


def aggregate_truncation_curves(
    df: pd.DataFrame,
    confidence: float = 0.90,
    metric_name: str = 'error',
) -> pd.DataFrame:
    """
    Aggregate truncation curves with confidence intervals.
    
    For Paper Figure 5B: Aggregates classification error with 90% CI.
    
    Args:
        df: DataFrame from compute_all_truncation_curves()
        confidence: Confidence level for intervals
        metric_name: Name of metric column ('error' or 'accuracy')
        
    Returns:
        DataFrame with mean metric and CI bounds per size and k
    """
    results = []
    
    for (size, k), group in df.groupby(['size', 'k']):
        values = group[metric_name].values
        n = len(values)
        mean = values.mean()
        
        if n > 1:
            sem = values.std() / np.sqrt(n)
            # Use t-distribution for small samples
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            ci_low = mean - t_val * sem
            ci_high = mean + t_val * sem
        else:
            ci_low = ci_high = mean
        
        results.append({
            'size': size,
            'k': k,
            f'{metric_name}_mean': mean,
            f'{metric_name}_ci_low': ci_low,
            f'{metric_name}_ci_high': ci_high,
            'n_seeds': n,
        })
    
    return pd.DataFrame(results)
