"""
Memory-efficient streaming Q computation for bilinear interaction matrices.

This module provides a chunked computation approach that reduces peak GPU memory
from ~17GB to ~1GB, enabling Figure 8 to run on CUDA GPUs with 40GB memory.

The key insight is that the einsum:
    Q[i,j] = sigma_m sigma_o w_l[m,i] * w_r[m,j] * w_p[o,m] * out_latents[o]

Can be decomposed into:
    z[m] = w_p.T @ out_latents  (small: [d_hidden])
    Q[i,j] = sigma_m w_l[m,i] * w_r[m,j] * z[m]  (streamable across m)

For projection onto SAE latents:
    The original tracer.q(project=True) uses an inefficient einsum that
    allocates O(n_features^2 * d_model^2) intermediate memory (~281TB!).
    
    We use sequential matrix multiplication instead:
    Q_projected = inp_latents.T @ Q @ inp_latents
    
    Which only needs O(n_features * d_model) intermediate memory (~32MB).
"""

import torch
from typing import Optional


def q_streaming(
    tracer,
    idx: int,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Memory-efficient Q matrix computation by streaming across d_hidden.
    
    This replaces the original tracer.q() einsum which requires ~17GB intermediate
    memory with a chunked computation that uses ~1GB per chunk.
    
    Args:
        tracer: Tracer instance with loaded model and SAEs
        idx: Output feature index
        chunk_size: Number of hidden dimensions to process at once (default 256)
                   - 256: ~1GB peak memory (for 40GB GPU)
                   - 128: ~0.5GB peak memory (for 24GB GPU)
        dtype: Computation dtype (default float32)
        device: Device to use (default: tracer.device)
    
    Returns:
        Q: [d_model, d_model] interaction matrix
    """
    model = tracer.model
    layer = tracer.layer
    if device is None:
        device = tracer.device
    
    # Get model weights
    # w_l, w_r: [d_hidden, d_model] (e.g., [4096, 1024] for fw-medium)
    # w_p: [d_model, d_hidden] (e.g., [1024, 4096])
    w_l = model.w_l[layer]
    w_r = model.w_r[layer]
    w_p = model.w_p[layer]
    out_lat = tracer.out_latents[idx]  # [d_model]
    
    d_hidden = w_l.shape[0]
    d_model = w_l.shape[1]
    
    # Step 1: Compute z = w_p.T @ out_lat
    # This is small: [d_hidden] = [4096] = 16KB
    z = torch.mv(w_p.T, out_lat)  # [d_hidden]
    
    # Step 2: Stream Q computation across d_hidden
    # Instead of: Q = einsum("mi,mj,m->ij", w_l, w_r, z)  which needs [d_hidden, d_model, d_model]
    # We compute in chunks to reduce peak memory
    Q = torch.zeros(d_model, d_model, device=device, dtype=dtype)
    
    for start in range(0, d_hidden, chunk_size):
        end = min(start + chunk_size, d_hidden)
        
        # Get chunks: [chunk_size, d_model]
        w_l_chunk = w_l[start:end].to(dtype)
        w_r_chunk = w_r[start:end].to(dtype)
        z_chunk = z[start:end].to(dtype)
        
        # Compute chunk contribution to Q
        # Q += einsum("cm,c,cn->mn", w_l_chunk, z_chunk, w_r_chunk)
        # This is: Q[m,n] += sigma_c w_l[c,m] * z[c] * w_r[c,n]
        # Equivalent to: (w_l_chunk.T * z_chunk) @ w_r_chunk
        # Memory: chunk_size * d_model = 256 * 1024 * 4 bytes = 1MB intermediate
        Q += torch.einsum('cm,c,cn->mn', w_l_chunk, z_chunk, w_r_chunk)
        
        # Clear intermediate tensors
        del w_l_chunk, w_r_chunk, z_chunk
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return Q


def q_streaming_projected(
    tracer,
    idx: int,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
    projection_chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Memory-efficient Q matrix computation WITH projection onto SAE latents.
    
    This replaces tracer.q(idx, project=True) which uses an inefficient einsum
    that allocates O(n_features^2 * d_model^2) intermediate memory (~281TB!).
    
    We use sequential matrix multiplication instead:
        Q_projected = inp_latents.T @ Q @ inp_latents
    
    Which only needs O(n_features * d_model) intermediate memory (~32MB).
    
    Args:
        tracer: Tracer instance with loaded model and SAEs
        idx: Output feature index
        chunk_size: Chunk size for Q computation (default 256)
        dtype: Computation dtype (default float32)
        device: Device to use (default: tracer.device)
        projection_chunk_size: Chunk size for projection (default 1024)
    
    Returns:
        Q_projected: [n_features, n_features] interaction matrix in SAE latent space
    """
    # Step 1: Compute Q in d_model space using streaming
    Q = q_streaming(tracer, idx, chunk_size=chunk_size, dtype=dtype, device=device)
    
    # Step 2: Project onto SAE latents using memory-efficient matrix multiplication
    # Original einsum: "il,jk,...ij->...lk" means inp_latents has shape [d_model, n_features]
    # Q_projected = inp_latents.T @ Q @ inp_latents
    # inp_latents: [d_model, n_features] (note: d_model first!)
    # Q: [d_model, d_model]
    # Q_projected: [n_features, n_features]
    
    inp_lat = tracer.inp_latents.to(dtype)  # [d_model, n_features]
    n_features = inp_lat.shape[1]  # Second dimension is n_features
    
    # Compute in two steps to avoid large intermediates:
    # temp = Q @ inp_latents  -> [d_model, n_features]
    # Q_projected = inp_latents.T @ temp  -> [n_features, n_features]
    
    # For very large n_features, chunk the final multiplication
    temp = Q @ inp_lat  # [d_model, d_model] @ [d_model, n_features] = [d_model, n_features]
    
    if n_features <= projection_chunk_size:
        # Small enough to do in one go
        Q_projected = inp_lat.T @ temp  # [n_features, d_model] @ [d_model, n_features] = [n_features, n_features]
    else:
        # Chunk the projection to reduce peak memory
        Q_projected = torch.zeros(n_features, n_features, device=device, dtype=dtype)
        
        for start in range(0, n_features, projection_chunk_size):
            end = min(start + projection_chunk_size, n_features)
            # Q_projected[start:end, :] = inp_lat.T[start:end, :] @ temp
            Q_projected[start:end, :] = inp_lat[:, start:end].T @ temp
    
    # Symmetrize for numerical stability
    Q_projected = 0.5 * (Q_projected + Q_projected.T)
    
    return Q_projected


def q_streaming_batch(
    tracer,
    indices: list[int],
    chunk_size: int = 256,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> list[torch.Tensor]:
    """
    Compute Q matrices for multiple features efficiently.
    
    Shares the model weight loading across features for better cache utilization.
    
    Args:
        tracer: Tracer instance
        indices: List of output feature indices
        chunk_size: Hidden dimension chunk size
        dtype: Computation dtype
        device: Device to use
    
    Returns:
        List of Q matrices, one per index
    """
    model = tracer.model
    layer = tracer.layer
    if device is None:
        device = tracer.device
    
    w_l = model.w_l[layer]
    w_r = model.w_r[layer]
    w_p = model.w_p[layer]
    
    d_hidden = w_l.shape[0]
    d_model = w_l.shape[1]
    
    results = []
    
    for idx in indices:
        out_lat = tracer.out_latents[idx]
        z = torch.mv(w_p.T, out_lat)
        
        Q = torch.zeros(d_model, d_model, device=device, dtype=dtype)
        
        for start in range(0, d_hidden, chunk_size):
            end = min(start + chunk_size, d_hidden)
            w_l_chunk = w_l[start:end].to(dtype)
            w_r_chunk = w_r[start:end].to(dtype)
            z_chunk = z[start:end].to(dtype)
            
            Q += torch.einsum('cm,c,cn->mn', w_l_chunk, z_chunk, w_r_chunk)
            
            del w_l_chunk, w_r_chunk, z_chunk
        
        results.append(Q)
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results
