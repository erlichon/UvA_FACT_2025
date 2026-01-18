"""
Memory-Efficient Eigendecomposition for Large Projected Matrices.

This module provides iterative methods to compute top-k eigenvectors of
Q_projected = V^T @ Q @ V without materializing the full n_features x n_features matrix.

Key insight: We only need the top k eigenvectors (sorted by magnitude), not all n_features.
For fw-medium with expansion=8, n_features=8192, so the full matrix would be ~268MB.
This module keeps memory at O(d_model x n_features) ≈ 32MB.

Mathematical equivalence:
- The paper sorts eigenvectors by eigenvalue MAGNITUDE (absolute value)
- The sentiment negation circuit has both large positive (+0.62) and negative (-0.66) eigenvalues
- We run lobpcg twice (largest=True and largest=False) then sort by magnitude

Usage:
    from src.language.memory_efficient_eigen import top_k_eigenvectors_by_magnitude
    
    eigenvalues, eigenvectors = top_k_eigenvectors_by_magnitude(
        Q=Q_unprojected,           # [d_model, d_model]
        inp_latents=sae_decoder,   # [d_model, n_features]
        k=2,
    )
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Callable
import warnings


class ProjectedMatrixOperator:
    """
    Linear operator for Q_projected = V^T @ Q @ V without materializing it.
    
    This allows iterative eigensolvers to work with the implicit matrix.
    """
    
    def __init__(self, Q: Tensor, V: Tensor):
        """
        Args:
            Q: Unprojected interaction matrix [d_model, d_model]
            V: SAE decoder directions [d_model, n_features]
        """
        self.Q = Q
        self.V = V
        self.n_features = V.shape[1]
        self.d_model = V.shape[0]
        
    def matvec(self, v: Tensor) -> Tensor:
        """
        Compute Q_projected @ v = V^T @ Q @ V @ v without materializing Q_projected.
        
        Args:
            v: Vector [n_features] or batch [batch, n_features]
            
        Returns:
            Result [n_features] or [batch, n_features]
        """
        # Handle batched input
        if v.dim() == 1:
            # v: [n_features]
            # Step 1: V @ v -> [d_model]
            Vv = self.V @ v
            # Step 2: Q @ (V @ v) -> [d_model]
            QVv = self.Q @ Vv
            # Step 3: V^T @ (Q @ V @ v) -> [n_features]
            return self.V.T @ QVv
        else:
            # v: [batch, n_features]
            # Step 1: V @ v^T -> [d_model, batch], then transpose
            Vv = (self.V @ v.T).T  # [batch, d_model]
            # Step 2: Q @ Vv^T -> [d_model, batch], then transpose
            QVv = (self.Q @ Vv.T).T  # [batch, d_model]
            # Step 3: V^T @ QVv^T -> [n_features, batch], then transpose
            return (self.V.T @ QVv.T).T  # [batch, n_features]
    
    def __matmul__(self, v: Tensor) -> Tensor:
        """Allow using @ operator."""
        return self.matvec(v)


def _lobpcg_safe(
    A_matvec: Callable[[Tensor], Tensor],
    n: int,
    k: int,
    largest: bool = True,
    tol: float = 1e-6,
    max_iter: int = 1000,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """
    Safe wrapper around torch.lobpcg that handles edge cases.
    
    Args:
        A_matvec: Function that computes A @ v
        n: Matrix dimension
        k: Number of eigenpairs to find
        largest: If True, find largest eigenvalues; if False, find smallest
        tol: Convergence tolerance
        max_iter: Maximum iterations
        device: Device for computation
        
    Returns:
        (eigenvalues, eigenvectors) tuple, eigenvalues shape [k], eigenvectors shape [n, k]
    """
    # Create random initial vectors
    torch.manual_seed(42)  # For reproducibility
    X = torch.randn(n, k, device=device, dtype=torch.float32)
    X, _ = torch.linalg.qr(X)  # Orthonormalize
    
    # Create the matrix as a LinearOperator-like callable
    # torch.lobpcg expects a matrix or a function
    # We'll construct the full matrix only if n is small enough, otherwise use power iteration
    
    if n <= 4096:
        # For smaller matrices, we can afford to materialize for lobpcg
        # Build matrix column by column
        A = torch.zeros(n, n, device=device, dtype=torch.float32)
        eye = torch.eye(n, device=device, dtype=torch.float32)
        for i in range(n):
            A[:, i] = A_matvec(eye[:, i])
        A = 0.5 * (A + A.T)  # Ensure symmetry
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eigenvalues, eigenvectors = torch.lobpcg(
                    A, k=k, X=X, largest=largest, tol=tol, niter=max_iter
                )
            return eigenvalues, eigenvectors
        except Exception as e:
            # Fallback to full eigendecomposition if lobpcg fails
            warnings.warn(f"lobpcg failed ({e}), falling back to eigh")
            eigenvalues, eigenvectors = torch.linalg.eigh(A)
            if largest:
                idx = torch.argsort(eigenvalues, descending=True)[:k]
            else:
                idx = torch.argsort(eigenvalues, descending=False)[:k]
            return eigenvalues[idx], eigenvectors[:, idx]
    else:
        # For large matrices, use power iteration / subspace iteration
        return _subspace_iteration(A_matvec, n, k, largest, tol, max_iter, device)


def _subspace_iteration(
    A_matvec: Callable[[Tensor], Tensor],
    n: int,
    k: int,
    largest: bool = True,
    tol: float = 1e-6,
    max_iter: int = 1000,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """
    Subspace iteration method for finding top-k eigenvectors.
    
    This is a simpler alternative to lobpcg that only needs matrix-vector products.
    
    Args:
        A_matvec: Function that computes A @ v
        n: Matrix dimension
        k: Number of eigenpairs to find
        largest: If True, find largest eigenvalues; if False, find smallest (uses -A)
        tol: Convergence tolerance  
        max_iter: Maximum iterations
        device: Device for computation
        
    Returns:
        (eigenvalues, eigenvectors) tuple
    """
    # For smallest eigenvalues, we negate the matrix
    if largest:
        matvec = A_matvec
    else:
        matvec = lambda v: -A_matvec(v)
    
    # Initialize random orthonormal vectors
    torch.manual_seed(42)
    V = torch.randn(n, k, device=device, dtype=torch.float32)
    V, _ = torch.linalg.qr(V)
    
    prev_eigenvalues = torch.zeros(k, device=device)
    
    for iteration in range(max_iter):
        # Apply A to all vectors: AV = A @ V
        AV = torch.zeros_like(V)
        for j in range(k):
            AV[:, j] = matvec(V[:, j])
        
        # Rayleigh quotient: eigenvalues = diag(V^T A V)
        eigenvalues = torch.sum(V * AV, dim=0)
        
        # Check convergence
        if iteration > 0:
            diff = torch.abs(eigenvalues - prev_eigenvalues).max()
            if diff < tol:
                break
        prev_eigenvalues = eigenvalues.clone()
        
        # Orthonormalize AV for next iteration
        V, _ = torch.linalg.qr(AV)
    
    # If we were finding smallest eigenvalues, negate back
    if not largest:
        eigenvalues = -eigenvalues
    
    return eigenvalues, V


def top_k_eigenvectors_by_magnitude(
    Q: Tensor,
    inp_latents: Tensor,
    k: int = 2,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[Tensor, Tensor]:
    """
    Compute top-k eigenvectors of Q_projected sorted by eigenvalue MAGNITUDE.
    
    This is mathematically equivalent to:
        Q_projected = inp_latents.T @ Q @ inp_latents
        eigenvalues, eigenvectors = torch.linalg.eigh(Q_projected)
        sort_idx = torch.argsort(eigenvalues.abs(), descending=True)[:k]
        return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
    
    But uses O(d_model * n_features) memory instead of O(n_features^2).
    
    Args:
        Q: Unprojected interaction matrix [d_model, d_model]
        inp_latents: SAE decoder directions [d_model, n_features]
        k: Number of top eigenvectors to return (sorted by magnitude)
        tol: Convergence tolerance for iterative solver
        max_iter: Maximum iterations for iterative solver
        
    Returns:
        (eigenvalues, eigenvectors) tuple where:
        - eigenvalues: [k] tensor of eigenvalues sorted by descending magnitude
        - eigenvectors: [n_features, k] tensor of corresponding eigenvectors
    """
    device = Q.device
    n_features = inp_latents.shape[1]
    
    # Ensure float32 and on same device
    Q = Q.float().to(device)
    inp_latents = inp_latents.float().to(device)
    
    # Create operator for Q_projected
    op = ProjectedMatrixOperator(Q, inp_latents)
    
    # Find k largest positive eigenvalues
    print(f"  Finding {k} largest positive eigenvalues...")
    pos_eigenvalues, pos_eigenvectors = _lobpcg_safe(
        op.matvec, n_features, k, largest=True, tol=tol, max_iter=max_iter, device=device
    )
    
    # Find k most negative eigenvalues (smallest, i.e., most negative)
    print(f"  Finding {k} most negative eigenvalues...")
    neg_eigenvalues, neg_eigenvectors = _lobpcg_safe(
        op.matvec, n_features, k, largest=False, tol=tol, max_iter=max_iter, device=device
    )
    
    # Combine all 2k eigenpairs
    all_eigenvalues = torch.cat([pos_eigenvalues, neg_eigenvalues])
    all_eigenvectors = torch.cat([pos_eigenvectors, neg_eigenvectors], dim=1)
    
    # Sort by MAGNITUDE (absolute value) and take top k
    sort_indices = torch.argsort(all_eigenvalues.abs(), descending=True)[:k]
    
    eigenvalues_sorted = all_eigenvalues[sort_indices]
    eigenvectors_sorted = all_eigenvectors[:, sort_indices]
    
    print(f"  Top {k} eigenvalues by magnitude: {eigenvalues_sorted.tolist()}")
    
    return eigenvalues_sorted, eigenvectors_sorted


def top_k_eigenvectors_direct(
    Q: Tensor,
    inp_latents: Tensor,
    k: int = 2,
) -> Tuple[Tensor, Tensor]:
    """
    Direct computation of top-k eigenvectors (materializes full matrix).
    
    This is the original approach - provided for comparison and validation.
    Use top_k_eigenvectors_by_magnitude() for memory efficiency.
    
    Args:
        Q: Unprojected interaction matrix [d_model, d_model]
        inp_latents: SAE decoder directions [d_model, n_features]
        k: Number of top eigenvectors to return
        
    Returns:
        (eigenvalues, eigenvectors) tuple sorted by descending magnitude
    """
    from src.utils import safe_eigh
    
    # Compute full projected matrix (memory-intensive!)
    n_features = inp_latents.shape[1]
    print(f"  Computing full projected Q matrix ({n_features}×{n_features})...")
    
    Q_V = Q @ inp_latents  # [d_model, n_features]
    Q_projected = inp_latents.T @ Q_V  # [n_features, n_features]
    Q_projected = 0.5 * (Q_projected + Q_projected.T)  # Symmetrize
    
    # Full eigendecomposition
    print(f"  Computing full eigendecomposition...")
    eigenvalues, eigenvectors = safe_eigh(Q_projected)
    
    # Sort by magnitude
    sort_indices = torch.argsort(eigenvalues.abs(), descending=True)[:k]
    
    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]
