"""
Interaction Matrix Utilities for Bilinear MLP Analysis.

This module provides the core computation for extracting interaction matrices
from bilinear transformer layers and performing eigendecomposition.

The key equation (Section 3 of the paper):
    z_c(x) = x^T Q_c x + b

Where:
- x: Input to the bilinear layer (residual stream, d_model)
- Q_c: Interaction matrix for feature c (d_model x d_model)
- z_c: Activation of SAE feature c

The interaction matrix Q_c is computed by contracting the bilinear layer weights
with the SAE output direction for feature c.

IMPORTANT: The paper's Tracer uses ENCODER directions by default (use_encoder=True).
The SAE activation is z_c = ReLU(mlp_out · w_enc[c]), so to predict how a feature
activates, we need the encoder direction w_enc.weight[c, :], NOT decoder.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from jaxtyping import Float

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.utils import safe_eigh


@dataclass
class InteractionEigenpairs:
    """
    Results from eigendecomposition of an interaction matrix.
    
    Attributes:
        eigenvalues: Eigenvalues sorted by descending magnitude [d_model]
        eigenvectors: Corresponding eigenvectors as columns [d_model, d_model]
        Q_matrix: The symmetrized interaction matrix [d_model, d_model]
        sort_indices: Indices used for sorting (for debugging)
    """
    eigenvalues: Float[Tensor, "d_model"]
    eigenvectors: Float[Tensor, "d_model d_model"]
    Q_matrix: Float[Tensor, "d_model d_model"]
    sort_indices: Optional[Tensor] = None


def compute_interaction_matrix(
    w_l: Float[Tensor, "d_hidden d_model"],
    w_r: Float[Tensor, "d_hidden d_model"],
    w_p: Float[Tensor, "d_model d_hidden"],
    out_direction: Float[Tensor, "d_model"],
    symmetrize: bool = True,
) -> Float[Tensor, "d_model d_model"]:
    """
    Compute the interaction matrix Q for a given output feature direction.
    
    The bilinear layer computes:
        mlp_out = w_p @ (w_l @ x * w_r @ x)
    
    For output feature c with direction e_c (encoder direction for SAE):
        z_c = e_c^T @ mlp_out = x^T Q_c x
    
    Where Q_c is derived from contracting the weights with e_c:
        Q[i,j] = sum_m sum_o e_c[o] * w_p[o,m] * w_l[m,i] * w_r[m,j]
    
    Args:
        w_l: Left projection weights [d_hidden, d_model]
        w_r: Right projection weights [d_hidden, d_model]
        w_p: Output projection weights [d_model, d_hidden]
        out_direction: SAE direction for the output feature [d_model]
            NOTE: For SAE features, use ENCODER direction w_enc.weight[feat_idx, :]
            as per paper's Tracer (use_encoder=True by default).
        symmetrize: If True, return 0.5 * (Q + Q^T) for numerical stability
    
    Returns:
        Q: Interaction matrix [d_model, d_model]
    """
    # Step 1: Contract output direction with output projection
    # proj[m] = sum_o e_c[o] * w_p[o,m] = e_c^T @ w_p
    proj = out_direction @ w_p  # [d_hidden]
    
    # Step 2: Scale left weights by projection
    # scaled_l[m,i] = proj[m] * w_l[m,i]
    scaled_l = proj.unsqueeze(1) * w_l  # [d_hidden, d_model]
    
    # Step 3: Contract to form Q matrix
    # Q[i,j] = sum_m scaled_l[m,i] * w_r[m,j] = scaled_l^T @ w_r
    Q = scaled_l.T @ w_r  # [d_model, d_model]
    
    # Step 4: Symmetrize for numerical stability
    # The quadratic form x^T Q x only sees the symmetric part anyway,
    # but symmetrization ensures real eigenvalues and orthogonal eigenvectors
    if symmetrize:
        Q = 0.5 * (Q + Q.T)
    
    return Q


def get_interaction_eigenpairs(
    model,
    layer: int,
    feat_idx: int,
    out_direction: Float[Tensor, "d_model"],
    device: str = "cpu",
) -> InteractionEigenpairs:
    """
    Compute the interaction matrix Q and its eigendecomposition for a given feature.
    
    This is the core utility for analyzing how input dimensions interact to
    produce a specific SAE output feature.
    
    Args:
        model: Transformer model with w_l, w_r, w_p properties
        layer: Layer index
        feat_idx: Feature index (used for logging only, direction is explicit)
        out_direction: SAE direction for the output feature [d_model]
            NOTE: For SAE features, use ENCODER direction w_enc.weight[feat_idx, :]
            as per paper's Tracer (use_encoder=True by default).
        device: Device for computation (eigendecomposition done on CPU for MPS safety)
    
    Returns:
        InteractionEigenpairs containing eigenvalues, eigenvectors, and Q matrix,
        all sorted by descending eigenvalue magnitude.
    
    Example:
        >>> # Use ENCODER direction (paper's default)
        >>> eigenpairs = get_interaction_eigenpairs(model, layer=2, feat_idx=0, 
        ...                                          out_direction=sae.w_enc.weight[0, :])
        >>> top_eigenvalue = eigenpairs.eigenvalues[0]
        >>> top_eigenvector = eigenpairs.eigenvectors[:, 0]
    """
    # Extract weights (move to CPU for numerical stability)
    w_l = model.w_l[layer].float().cpu()  # [d_hidden, d_model]
    w_r = model.w_r[layer].float().cpu()  # [d_hidden, d_model]
    w_p = model.w_p[layer].float().cpu()  # [d_model, d_hidden]
    out_vec = out_direction.float().cpu()  # [d_model]
    
    # Compute interaction matrix (symmetrized)
    Q = compute_interaction_matrix(w_l, w_r, w_p, out_vec, symmetrize=True)
    
    # Eigendecomposition (MPS-safe via safe_eigh)
    # Note: torch.linalg.eigh returns eigenvalues in ascending order (by value, not magnitude)
    eigenvalues, eigenvectors = safe_eigh(Q)
    
    # CRITICAL: Sort by MAGNITUDE (absolute value), not by signed value!
    # The paper relies on cancellation between large positive and negative eigenvalues.
    # The largest magnitude eigenvalues (regardless of sign) capture the strongest interactions.
    sort_indices = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues_sorted = eigenvalues[sort_indices]
    eigenvectors_sorted = eigenvectors[:, sort_indices]
    
    return InteractionEigenpairs(
        eigenvalues=eigenvalues_sorted,
        eigenvectors=eigenvectors_sorted,
        Q_matrix=Q,
        sort_indices=sort_indices,
    )


def get_interaction_eigenpairs_from_tracer(
    tracer,
    feat_idx: int,
    device: str = "cpu",
) -> InteractionEigenpairs:
    """
    Convenience function to get eigenpairs using an existing Tracer object.
    
    This extracts the output direction from the tracer's out_latents and
    calls get_interaction_eigenpairs. Note that tracer.out_latents uses
    encoder directions by default (use_encoder=True in Tracer.__init__).
    
    Args:
        tracer: Tracer object with model, layer, and out_latents
        feat_idx: Feature index in the output SAE
        device: Device for computation
    
    Returns:
        InteractionEigenpairs for the specified feature
    """
    out_direction = tracer.out_latents[feat_idx]
    return get_interaction_eigenpairs(
        model=tracer.model,
        layer=tracer.layer,
        feat_idx=feat_idx,
        out_direction=out_direction,
        device=device,
    )


def rank_k_approximation(
    eigenpairs: InteractionEigenpairs,
    k: int,
) -> tuple[Float[Tensor, "k"], Float[Tensor, "d_model k"]]:
    """
    Get the top-k eigenvalues and eigenvectors for low-rank approximation.
    
    The rank-k approximation of z = x^T Q x is:
        z_hat = sum_{j=1}^k lambda_j * (v_j^T x)^2
    
    Args:
        eigenpairs: Result from get_interaction_eigenpairs
        k: Rank of approximation
    
    Returns:
        Tuple of (eigenvalues[k], eigenvectors[d_model, k])
    """
    k = min(k, len(eigenpairs.eigenvalues))
    return eigenpairs.eigenvalues[:k], eigenpairs.eigenvectors[:, :k]


def predict_activation_from_eigenpairs(
    x: Float[Tensor, "batch d_model"],
    eigenvalues: Float[Tensor, "k"],
    eigenvectors: Float[Tensor, "d_model k"],
) -> Float[Tensor, "batch"]:
    """
    Compute the predicted activation using the low-rank approximation.
    
    z_hat = sum_{j=1}^k lambda_j * (v_j^T x)^2
    
    Args:
        x: Input activations [batch, d_model]
        eigenvalues: Top-k eigenvalues [k]
        eigenvectors: Top-k eigenvectors as columns [d_model, k]
    
    Returns:
        Predicted activations [batch]
    """
    # Project input onto eigenvectors: [batch, k]
    projections = x @ eigenvectors
    
    # Square and weight by eigenvalues: [batch]
    z_pred = (projections ** 2) @ eigenvalues
    
    return z_pred


def compute_interaction_matrix_streaming(
    model,
    layer: int,
    out_direction: Float[Tensor, "d_model"],
    chunk_size: int = 256,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Float[Tensor, "d_model d_model"]:
    """
    Memory-efficient Q matrix computation using streaming across d_hidden.
    
    This reduces peak GPU memory from ~17GB to ~1GB by chunking the computation.
    
    The key insight is that the einsum:
        Q[i,j] = Σ_m Σ_o w_l[m,i] * w_r[m,j] * w_p[o,m] * out_direction[o]
    
    Can be decomposed into:
        z[m] = w_p.T @ out_direction  (small: [d_hidden])
        Q[i,j] = Σ_m w_l[m,i] * w_r[m,j] * z[m]  (streamable across m)
    
    Args:
        model: Transformer model with w_l, w_r, w_p properties
        layer: Layer index
        out_direction: SAE encoder direction for the output feature [d_model]
        chunk_size: Number of hidden dimensions to process at once (default 256)
        dtype: Computation dtype (default float32)
        device: Device for computation (supports cuda, mps, cpu)
    
    Returns:
        Q: Symmetrized interaction matrix [d_model, d_model]
    """
    # Get model weights
    w_l = model.w_l[layer]  # [d_hidden, d_model]
    w_r = model.w_r[layer]  # [d_hidden, d_model]
    w_p = model.w_p[layer]  # [d_model, d_hidden]
    
    d_hidden = w_l.shape[0]
    d_model = w_l.shape[1]
    
    # Move out_direction to same device as model weights
    out_vec = out_direction.to(w_p.device).to(dtype)
    
    # Step 1: Compute z = w_p.T @ out_direction
    # This is small: [d_hidden] = [4096] = 16KB
    z = torch.mv(w_p.T.to(dtype), out_vec)  # [d_hidden]
    
    # Step 2: Stream Q computation across d_hidden
    Q = torch.zeros(d_model, d_model, device=device, dtype=dtype)
    
    for start in range(0, d_hidden, chunk_size):
        end = min(start + chunk_size, d_hidden)
        
        # Get chunks and move to compute device
        w_l_chunk = w_l[start:end].to(device).to(dtype)
        w_r_chunk = w_r[start:end].to(device).to(dtype)
        z_chunk = z[start:end].to(device)
        
        # Compute chunk contribution to Q
        # Q += einsum("cm,c,cn->mn", w_l_chunk, z_chunk, w_r_chunk)
        Q += torch.einsum('cm,c,cn->mn', w_l_chunk, z_chunk, w_r_chunk)
        
        # Clear intermediate tensors
        del w_l_chunk, w_r_chunk, z_chunk
    
    # Clear GPU cache if on CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Symmetrize for numerical stability
    Q = 0.5 * (Q + Q.T)
    
    return Q


def get_interaction_eigenpairs_streaming(
    model,
    layer: int,
    feat_idx: int,
    out_direction: Float[Tensor, "d_model"],
    chunk_size: int = 256,
    compute_device: str = "cpu",
) -> InteractionEigenpairs:
    """
    Compute interaction matrix Q using streaming (GPU-accelerated) and eigendecomposition.
    
    This is a faster version of get_interaction_eigenpairs that:
    1. Computes Q matrix on GPU using memory-efficient streaming
    2. Performs eigendecomposition on CPU (required for MPS numerical stability)
    
    Args:
        model: Transformer model with w_l, w_r, w_p properties
        layer: Layer index
        feat_idx: Feature index (used for logging only)
        out_direction: SAE encoder direction for the output feature [d_model]
        chunk_size: Chunk size for streaming Q computation (default 256)
        compute_device: Device for Q matrix computation (cuda, mps, cpu)
    
    Returns:
        InteractionEigenpairs containing eigenvalues, eigenvectors, and Q matrix,
        all sorted by descending eigenvalue magnitude.
    
    Example:
        >>> eigenpairs = get_interaction_eigenpairs_streaming(
        ...     model, layer=7, feat_idx=0, 
        ...     out_direction=sae.w_enc.weight[0, :],
        ...     compute_device="cuda"
        ... )
    """
    # Compute Q matrix on GPU using streaming
    Q = compute_interaction_matrix_streaming(
        model=model,
        layer=layer,
        out_direction=out_direction,
        chunk_size=chunk_size,
        dtype=torch.float32,
        device=compute_device,
    )
    
    # Move Q to CPU for eigendecomposition (required for MPS numerical stability)
    Q_cpu = Q.cpu()
    
    # Free GPU memory
    del Q
    if compute_device == "cuda":
        torch.cuda.empty_cache()
    
    # Eigendecomposition on CPU (MPS-safe via safe_eigh)
    eigenvalues, eigenvectors = safe_eigh(Q_cpu)
    
    # Sort by MAGNITUDE (absolute value), not by signed value
    sort_indices = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues_sorted = eigenvalues[sort_indices]
    eigenvectors_sorted = eigenvectors[:, sort_indices]
    
    return InteractionEigenpairs(
        eigenvalues=eigenvalues_sorted,
        eigenvectors=eigenvectors_sorted,
        Q_matrix=Q_cpu,
        sort_indices=sort_indices,
    )
