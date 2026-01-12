"""
Verification script to debug the interaction analysis implementation.

Tests the einsum decomposition used in analyze_single_feature() against
the original tracer.q() implementation using synthetic data first,
then optionally with the real model.
"""

import sys
from pathlib import Path

import torch


def test_einsum_decomposition():
    """
    Test that my manual decomposition matches the original einsum.
    Uses small synthetic tensors for quick verification.
    """
    print("=== Testing einsum decomposition with synthetic data ===\n")

    # Small dimensions for testing
    d_hidden = 8   # m
    d_model = 4    # i, j, o

    # Create random tensors
    torch.manual_seed(42)
    w_l = torch.randn(d_hidden, d_model)  # [m, i]
    w_r = torch.randn(d_hidden, d_model)  # [m, j]
    w_p = torch.randn(d_model, d_hidden)  # [o, m]
    out_vec = torch.randn(d_model)        # [o]

    print(f"w_l shape: {w_l.shape}")
    print(f"w_r shape: {w_r.shape}")
    print(f"w_p shape: {w_p.shape}")
    print(f"out_vec shape: {out_vec.shape}")

    # Original einsum (from tracer.q)
    Q_einsum = torch.einsum("mi,mj,om,o->ij", w_l, w_r, w_p, out_vec)
    print(f"\nQ_einsum shape: {Q_einsum.shape}")

    # My decomposition
    proj = out_vec @ w_p  # [o] @ [o, m] -> [m]
    print(f"proj shape: {proj.shape}")

    scaled_l = proj.unsqueeze(1) * w_l  # [m, 1] * [m, i] -> [m, i]
    print(f"scaled_l shape: {scaled_l.shape}")

    scaled_r = w_r  # [m, j]
    Q_mine = scaled_l.T @ scaled_r  # [i, m] @ [m, j] -> [i, j]
    print(f"Q_mine shape: {Q_mine.shape}")

    # Compare
    diff = (Q_einsum - Q_mine).abs()
    print(f"\nMax absolute difference: {diff.max():.10e}")
    print(f"Mean absolute difference: {diff.mean():.10e}")
    print(f"Are they close? {torch.allclose(Q_einsum, Q_mine, atol=1e-6)}")

    if torch.allclose(Q_einsum, Q_mine, atol=1e-6):
        print("\n[PASS] Decomposition is mathematically correct!")
    else:
        print("\n[FAIL] Decomposition does NOT match!")
        print(f"Q_einsum:\n{Q_einsum}")
        print(f"Q_mine:\n{Q_mine}")

    return torch.allclose(Q_einsum, Q_mine, atol=1e-6)


def test_effective_rank():
    """
    Test effective rank computation on matrices with known properties.
    """
    print("\n=== Testing effective rank computation ===\n")

    # Rank-1 matrix: should have effective rank ~1
    v = torch.randn(100)
    Q_rank1 = torch.outer(v, v)
    Q_rank1_sym = 0.5 * (Q_rank1 + Q_rank1.T)

    vals = torch.linalg.eigvalsh(Q_rank1_sym)
    l1 = vals.abs().sum()
    l2 = vals.pow(2).sum().sqrt()
    eff_rank_1 = (l1/l2).pow(2).item()
    print(f"Rank-1 matrix (100x100): effective_rank = {eff_rank_1:.4f} (expected ~1)")

    # Full-rank random matrix: should have effective rank close to dim
    Q_full = torch.randn(100, 100)
    Q_full_sym = 0.5 * (Q_full + Q_full.T)

    vals = torch.linalg.eigvalsh(Q_full_sym)
    l1 = vals.abs().sum()
    l2 = vals.pow(2).sum().sqrt()
    eff_rank_full = (l1/l2).pow(2).item()
    print(f"Random full-rank matrix (100x100): effective_rank = {eff_rank_full:.4f} (expected ~100)")

    # Rank-2 matrix
    v1 = torch.randn(100)
    v2 = torch.randn(100)
    Q_rank2 = 3 * torch.outer(v1, v1) + 2 * torch.outer(v2, v2)
    Q_rank2_sym = 0.5 * (Q_rank2 + Q_rank2.T)

    vals = torch.linalg.eigvalsh(Q_rank2_sym)
    l1 = vals.abs().sum()
    l2 = vals.pow(2).sum().sqrt()
    eff_rank_2 = (l1/l2).pow(2).item()
    print(f"Rank-2 matrix (100x100): effective_rank = {eff_rank_2:.4f} (expected ~2)")


def test_rank2_correlation():
    """
    Test rank-2 correlation computation.
    """
    print("\n=== Testing rank-2 correlation ===\n")

    # Create a rank-2 matrix - should have perfect correlation
    torch.manual_seed(42)
    v1 = torch.randn(100)
    v2 = torch.randn(100)
    Q_rank2 = 5 * torch.outer(v1, v1) + 3 * torch.outer(v2, v2)
    Q_sym = 0.5 * (Q_rank2 + Q_rank2.T)

    # Compute rank-2 approximation
    eigenvalues, eigenvectors = torch.linalg.eigh(Q_sym)
    order = eigenvalues.abs().argsort(descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    Q_hat = torch.zeros_like(Q_sym)
    for i in range(2):
        v = eigenvectors[:, i:i+1]
        Q_hat = Q_hat + eigenvalues[i] * (v @ v.T)

    # Correlation
    Q_flat = Q_sym.flatten()
    Q_hat_flat = Q_hat.flatten()
    Q_centered = Q_flat - Q_flat.mean()
    Q_hat_centered = Q_hat_flat - Q_hat_flat.mean()

    numerator = (Q_centered * Q_hat_centered).sum()
    denominator = torch.sqrt((Q_centered ** 2).sum() * (Q_hat_centered ** 2).sum())
    corr = (numerator / denominator).item()

    print(f"Rank-2 matrix correlation with rank-2 approx: {corr:.6f} (expected ~1.0)")

    # Full rank matrix should have lower correlation
    Q_full = torch.randn(100, 100)
    Q_full_sym = 0.5 * (Q_full + Q_full.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(Q_full_sym)
    order = eigenvalues.abs().argsort(descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    Q_hat = torch.zeros_like(Q_full_sym)
    for i in range(2):
        v = eigenvectors[:, i:i+1]
        Q_hat = Q_hat + eigenvalues[i] * (v @ v.T)

    Q_flat = Q_full_sym.flatten()
    Q_hat_flat = Q_hat.flatten()
    Q_centered = Q_flat - Q_flat.mean()
    Q_hat_centered = Q_hat_flat - Q_hat_flat.mean()

    numerator = (Q_centered * Q_hat_centered).sum()
    denominator = torch.sqrt((Q_centered ** 2).sum() * (Q_hat_centered ** 2).sum())
    corr_full = (numerator / denominator).item()

    print(f"Full-rank matrix correlation with rank-2 approx: {corr_full:.6f} (expected <<1.0)")


def main():
    # Test 1: Verify einsum decomposition
    decomp_ok = test_einsum_decomposition()

    # Test 2: Effective rank
    test_effective_rank()

    # Test 3: Rank-2 correlation
    test_rank2_correlation()

    print("\n" + "="*60)
    if decomp_ok:
        print("Einsum decomposition is CORRECT.")
        print("If results are still wrong, the issue may be elsewhere")
        print("(e.g., model weight access, dtype, device issues)")
    else:
        print("Einsum decomposition is WRONG - needs fixing!")
    print("="*60)


if __name__ == "__main__":
    main()
