"""
Subspace Geometry Analysis for Cross-Dataset Robustness.

Extension 2: Subspace Geometry Metric
Implements Principal Angles analysis to quantify the similarity between
eigenvector subspaces from different models/datasets.

Key Concept:
If MNIST '0' and EMNIST 'O' learned the same "circularity" mechanism,
their top eigenvector subspaces should have high overlap despite
being trained on different datasets.

The principal angles between two subspaces measure how "aligned" they are:
- Principal angle θ₁ = 0° means subspaces share a direction
- All angles = 0° means identical subspaces
- θ = 90° means orthogonal (no similarity)

Usage:
    from src.vision.subspace import compute_subspace_overlap, principal_angles
    
    # Compare top-10 eigenvectors of MNIST '0' and EMNIST 'O'
    overlap = compute_subspace_overlap(mnist_vecs_0, emnist_vecs_O, k=10)
    print(f"Subspace similarity: {overlap:.3f}")  # 0 to 1
"""

import torch
from torch import Tensor
from jaxtyping import Float
from typing import Tuple, Optional
import numpy as np


def principal_angles(
    A: Float[Tensor, "k1 d"],
    B: Float[Tensor, "k2 d"],
) -> Float[Tensor, "min_k"]:
    """
    Compute principal angles between two subspaces.
    
    The principal angles θ₁ ≤ θ₂ ≤ ... ≤ θₖ between subspaces span(A) and span(B)
    are defined as the angles between the most aligned pairs of directions.
    
    Method: SVD of A^T @ B gives cos(θᵢ) as singular values.
    
    Args:
        A: Matrix with columns spanning first subspace [k1, d]
           k1 vectors of dimension d
        B: Matrix with columns spanning second subspace [k2, d]
           k2 vectors of dimension d
    
    Returns:
        Principal angles in radians, shape [min(k1, k2)]
    
    Reference:
        Golub & Van Loan, "Matrix Computations", Section 6.4.3
    """
    # Ensure row vectors (each row is an eigenvector)
    # We want to compute angles between column spaces, so transpose if needed
    
    # Orthonormalize both subspaces using QR decomposition
    # A: [k1, d] -> Q_A: [k1, d] orthonormal rows
    Q_A, _ = torch.linalg.qr(A.T)  # [d, k1]
    Q_B, _ = torch.linalg.qr(B.T)  # [d, k2]
    
    # Compute SVD of Q_A^T @ Q_B
    # Singular values are cos(principal_angles)
    _, S, _ = torch.linalg.svd(Q_A.T @ Q_B, full_matrices=False)
    
    # Clamp to valid range for arccos (numerical stability)
    S = S.clamp(-1.0, 1.0)
    
    # Principal angles
    angles = torch.arccos(S)
    
    return angles


def compute_subspace_overlap(
    vecs_A: Float[Tensor, "n_components d_input"],
    vecs_B: Float[Tensor, "n_components d_input"],
    k: int = 10,
    method: str = 'mean_cos',
) -> float:
    """
    Compute similarity score between two eigenvector subspaces.
    
    This is the core metric for Extension 2: proving that regularized
    models learn universal geometric features that transfer across datasets.
    
    Args:
        vecs_A: First set of eigenvectors [n, d_input]
                (e.g., MNIST digit '0' eigenvectors)
        vecs_B: Second set of eigenvectors [n, d_input]
                (e.g., EMNIST letter 'O' eigenvectors)
        k: Number of top eigenvectors to compare (default: 10)
        method: Similarity metric:
            - 'mean_cos': Mean of cos(principal_angles) ∈ [0, 1]
            - 'grassmann': Grassmann distance-based similarity ∈ [0, 1]
            - 'projection': Projection-based Frobenius norm ∈ [0, 1]
    
    Returns:
        Similarity score in [0, 1]:
        - 1.0 = identical subspaces
        - 0.0 = orthogonal subspaces
    
    Example:
        # High overlap expected: MNIST '0' vs EMNIST 'O' (both circular)
        overlap = compute_subspace_overlap(mnist_vecs[0], emnist_vecs[14], k=10)
        
        # Low overlap expected: MNIST '0' vs EMNIST 'X' (different shapes)
        overlap = compute_subspace_overlap(mnist_vecs[0], emnist_vecs[23], k=10)
    """
    # Take top-k eigenvectors
    A = vecs_A[:k]  # [k, d_input]
    B = vecs_B[:k]  # [k, d_input]
    
    # Normalize eigenvectors
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-10)
    B = B / (B.norm(dim=-1, keepdim=True) + 1e-10)
    
    if method == 'mean_cos':
        # Compute principal angles
        angles = principal_angles(A, B)
        # Mean cosine of principal angles
        return torch.cos(angles).mean().item()
    
    elif method == 'grassmann':
        # Grassmann distance-based similarity
        # d_G = sqrt(sum(θᵢ²)), similarity = 1 - d_G / (k * π/2)
        angles = principal_angles(A, B)
        grassmann_dist = torch.sqrt((angles ** 2).sum())
        max_dist = np.sqrt(k) * (np.pi / 2)
        return 1.0 - (grassmann_dist.item() / max_dist)
    
    elif method == 'projection':
        # Projection-based similarity using Frobenius norm
        # Measures how much of subspace A is captured by projection onto B
        
        # Orthonormalize
        Q_A, _ = torch.linalg.qr(A.T)  # [d, k]
        Q_B, _ = torch.linalg.qr(B.T)  # [d, k]
        
        # Projection matrix of A onto B: P = Q_B @ Q_B^T @ Q_A
        # Overlap = ||P||_F² / k
        P = Q_B @ (Q_B.T @ Q_A)
        overlap = (P ** 2).sum() / k
        return overlap.item()
    
    elif method == 'hungarian':
        # Hungarian-matched cosine similarity
        # Finds optimal 1-to-1 matching between eigenvectors
        # This handles the case where eigenvectors capture similar features
        # but are ordered differently by eigenvalue magnitude
        from scipy.optimize import linear_sum_assignment
        
        # Compute all pairwise cosines
        cos_matrix = (A @ B.T).abs().cpu().numpy()  # [k, k]
        
        # Cost matrix for Hungarian algorithm (we want to maximize similarity)
        cost_matrix = 1 - cos_matrix
        
        # Find optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Return mean of matched cosine similarities
        matched_cosines = cos_matrix[row_ind, col_ind]
        return matched_cosines.mean()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_class_eigenvectors(
    model,
    class_idx: int,
    k: int = 10,
    device: str = 'cpu',
) -> Float[Tensor, "k d_input"]:
    """
    Extract top-k mechanism eigenvectors for a specific class.
    
    This performs eigendecomposition of the symmetrized bilinear tensor
    for the specified output class.
    
    Args:
        model: Trained bilinear model with decompose() method
        class_idx: Output class index (0-9 for MNIST, 0-25 for EMNIST)
        k: Number of top eigenvectors to return
        device: Device for computation
    
    Returns:
        Top-k eigenvectors sorted by eigenvalue magnitude [k, d_input]
    """
    # Get full eigendecomposition
    eigenvalues, eigenvectors = model.decompose()
    
    # Extract for specific class
    vals = eigenvalues[class_idx]  # [n_components]
    vecs = eigenvectors[class_idx]  # [n_components, d_input]
    
    # Sort by magnitude and take top-k
    _, indices = vals.abs().sort(descending=True)
    top_k_indices = indices[:k]
    
    return vecs[top_k_indices].to(device)


def compare_mechanisms(
    model_A,
    model_B,
    class_A: int,
    class_B: int,
    k: int = 10,
    methods: Optional[list] = None,
) -> dict:
    """
    Compare mechanisms between two models for specified classes.
    
    Use case: Compare MNIST '0' mechanism with EMNIST 'O' mechanism
    to verify they learned the same "circularity" concept.
    
    Args:
        model_A: First model (e.g., MNIST-trained)
        model_B: Second model (e.g., EMNIST-trained)
        class_A: Class index in model A (e.g., 0 for digit '0')
        class_B: Class index in model B (e.g., 14 for letter 'O')
        k: Number of top eigenvectors to compare
        methods: List of similarity methods (default: all three)
    
    Returns:
        Dictionary with similarity scores for each method
    """
    if methods is None:
        methods = ['mean_cos', 'grassmann', 'projection']
    
    # Extract eigenvectors
    vecs_A = extract_class_eigenvectors(model_A, class_A, k=k)
    vecs_B = extract_class_eigenvectors(model_B, class_B, k=k)
    
    results = {}
    for method in methods:
        results[method] = compute_subspace_overlap(vecs_A, vecs_B, k=k, method=method)
    
    return results


def pairwise_class_similarity(
    model_A,
    model_B,
    n_classes_A: int,
    n_classes_B: int,
    k: int = 10,
    method: str = 'mean_cos',
) -> Float[Tensor, "n_A n_B"]:
    """
    Compute pairwise similarity matrix between all classes of two models.
    
    Useful for visualizing which MNIST digits match which EMNIST letters.
    
    Args:
        model_A: First model
        model_B: Second model
        n_classes_A: Number of classes in model A
        n_classes_B: Number of classes in model B
        k: Number of eigenvectors to compare
        method: Similarity method
    
    Returns:
        Similarity matrix [n_classes_A, n_classes_B]
    """
    # Get all eigenvectors
    _, vecs_A = model_A.decompose()  # [n_A, n_components, d_input]
    _, vecs_B = model_B.decompose()  # [n_B, n_components, d_input]
    
    similarity_matrix = torch.zeros(n_classes_A, n_classes_B)
    
    for i in range(n_classes_A):
        for j in range(n_classes_B):
            # Sort by eigenvalue magnitude and take top-k
            sim = compute_subspace_overlap(vecs_A[i], vecs_B[j], k=k, method=method)
            similarity_matrix[i, j] = sim
    
    return similarity_matrix


def semantic_similarity_score(
    similarity_matrix: Float[Tensor, "10 26"],
    expected_pairs: dict,
) -> dict:
    """
    Compute semantic similarity scores based on expected letter-digit pairs.
    
    Tests whether the learned mechanisms match human intuition about
    shape similarity (e.g., 'O' should match '0').
    
    Args:
        similarity_matrix: Pairwise similarity [10 digits, 26 letters]
        expected_pairs: Dict mapping letters to expected digits
                        e.g., {'O': 0, 'I': 1, 'Z': 2, 'S': 5, 'B': 6}
    
    Returns:
        Dictionary with:
        - 'expected_mean': Mean similarity for expected pairs
        - 'random_mean': Mean similarity for non-expected pairs
        - 'ratio': Ratio of expected to random (higher = better)
        - 'per_pair': Individual pair similarities
    """
    expected_sims = []
    per_pair = {}
    
    for letter, expected_digit in expected_pairs.items():
        letter_idx = ord(letter.upper()) - ord('A')
        sim = similarity_matrix[expected_digit, letter_idx].item()
        expected_sims.append(sim)
        per_pair[f"{letter}->{expected_digit}"] = sim
    
    # Random baseline: all pairs except expected
    mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    for letter, digit in expected_pairs.items():
        letter_idx = ord(letter.upper()) - ord('A')
        mask[digit, letter_idx] = False
    
    random_sims = similarity_matrix[mask]
    
    return {
        'expected_mean': np.mean(expected_sims),
        'random_mean': random_sims.mean().item(),
        'ratio': np.mean(expected_sims) / (random_sims.mean().item() + 1e-10),
        'per_pair': per_pair,
    }


def sort_eigenvectors_by_magnitude(
    eigenvalues: Float[Tensor, "n_classes n_components"],
    eigenvectors: Float[Tensor, "n_classes n_components n_features"],
) -> Float[Tensor, "n_classes n_components n_features"]:
    """
    Sort eigenvectors by their corresponding eigenvalue magnitude.
    
    For each class, sorts eigenvectors in descending order of eigenvalue
    magnitude, so the first eigenvector has the largest eigenvalue.
    
    This function is used to ensure consistent eigenvector ordering when
    comparing subspaces between different models, since eigendecomposition
    can return eigenvectors in arbitrary order.
    
    Args:
        eigenvalues: Eigenvalues tensor [n_classes, n_components]
        eigenvectors: Eigenvectors tensor [n_classes, n_components, n_features]
    
    Returns:
        Sorted eigenvectors tensor [n_classes, n_components, n_features]
        where eigenvectors[:, i, :] corresponds to eigenvalues[:, i] in
        descending order of magnitude.
    
    Example:
        >>> vals = torch.tensor([[0.5, 0.8, 0.2], [0.3, 0.9, 0.1]])
        >>> vecs = torch.randn(2, 3, 784)
        >>> sorted_vecs = sort_eigenvectors_by_magnitude(vals, vecs)
        >>> # sorted_vecs[0] now has vecs corresponding to [0.8, 0.5, 0.2]
        >>> # sorted_vecs[1] now has vecs corresponding to [0.9, 0.3, 0.1]
    
    Note:
        This function uses absolute value for sorting to handle potentially
        negative eigenvalues correctly (magnitude is what matters for
        importance, not sign).
    """
    n_classes = eigenvalues.shape[0]
    sorted_vecs = []
    
    for c in range(n_classes):
        _, indices = eigenvalues[c].abs().sort(descending=True)
        sorted_vecs.append(eigenvectors[c, indices])
    
    return torch.stack(sorted_vecs)


def select_balanced_eigenvectors(
    eigenvalues: Float[Tensor, "n_classes n_components"],
    eigenvectors: Float[Tensor, "n_classes n_components n_features"],
    k: int = 10,
) -> Float[Tensor, "n_classes k n_features"]:
    """
    Select top-k eigenvectors per class with balanced sign.

    We take k//2 eigenvectors with positive eigenvalues and k-k//2 with
    negative eigenvalues, each chosen by highest magnitude. If a class
    lacks enough positive or negative eigenvalues, we fill the remaining
    slots with the highest-magnitude eigenvectors regardless of sign.
    """
    n_classes, n_components = eigenvalues.shape
    k = min(k, n_components)
    selected_vecs = []

    for c in range(n_classes):
        vals = eigenvalues[c]
        vecs = eigenvectors[c]

        pos_idx = torch.nonzero(vals > 0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(vals < 0, as_tuple=False).squeeze(-1)

        pos_sorted = pos_idx[vals[pos_idx].abs().argsort(descending=True)] if pos_idx.numel() > 0 else pos_idx
        neg_sorted = neg_idx[vals[neg_idx].abs().argsort(descending=True)] if neg_idx.numel() > 0 else neg_idx

        k_pos = k // 2
        k_neg = k - k_pos

        chunks = []
        if k_pos > 0 and pos_sorted.numel() > 0:
            chunks.append(pos_sorted[:k_pos])
        if k_neg > 0 and neg_sorted.numel() > 0:
            chunks.append(neg_sorted[:k_neg])

        if len(chunks) > 0:
            selected_idx = torch.cat(chunks)
        else:
            selected_idx = torch.tensor([], dtype=torch.long, device=vals.device)

        if selected_idx.numel() < k:
            selected_mask = torch.zeros_like(vals, dtype=torch.bool)
            if selected_idx.numel() > 0:
                selected_mask[selected_idx] = True
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            if remaining.numel() > 0:
                remaining_sorted = remaining[vals[remaining].abs().argsort(descending=True)]
                needed = k - selected_idx.numel()
                selected_idx = torch.cat([selected_idx, remaining_sorted[:needed]])

        selected_vecs.append(vecs[selected_idx])

    return torch.stack(selected_vecs)
