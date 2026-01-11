"""
Bilinear Layer: Dense (Original) + CP-Decomposition (Extension)

Dense Mode:
    Wraps original implementation from bilinear-decomposition-main/shared/components.py
    y = gate(W_l @ x) * (W_r @ x)

CP Mode (Phase 2 Extension):
    Explicit rank-R decomposition of the interaction tensor.
    y = ((x @ A) * (x @ B) * lambdas) @ C.T
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from shared.components import Bilinear as OriginalBilinear


class BilinearDense(nn.Module):
    """
    Wrapper around original Bilinear layer for exact reproduction.

    This ensures we use the exact same implementation as the paper.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str = None):
        super().__init__()
        self._original = OriginalBilinear(d_in, d_out, bias=bias, gate=gate)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self._original(x)

    @property
    def w_l(self) -> Float[Tensor, "d_out d_in"]:
        return self._original.w_l

    @property
    def w_r(self) -> Float[Tensor, "d_out d_in"]:
        return self._original.w_r

    @property
    def weight(self):
        return self._original.weight


class BilinearCP(nn.Module):
    """
    CP-Decomposed Bilinear Layer (Phase 2 Extension).

    Explicitly parameterizes the interaction tensor with rank R:
    B[i,j,k] = sum_r lambda[r] * A[i,r] * B[j,r] * C[k,r]

    Args:
        d_in: Input dimension
        d_out: Output dimension
        rank: CP decomposition rank
        bias: Include bias term (not implemented for CP)
    """

    def __init__(self, d_in: int, d_out: int, rank: int, bias: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("Bias not supported for CP mode")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # CP factors: A, B for input projections, C for output
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.C = nn.Parameter(torch.randn(d_out, rank) * 0.02)
        self.lambdas = nn.Parameter(torch.ones(rank))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # x: [..., d_in]
        left = x @ self.A          # [..., rank]
        right = x @ self.B         # [..., rank]
        hidden = left * right * self.lambdas  # [..., rank]
        return hidden @ self.C.T   # [..., d_out]

    @property
    def w_l(self) -> Float[Tensor, "d_out d_in"]:
        """Reconstruct W_l for compatibility with analysis code."""
        # W_l[o, i] = sum_r lambda[r] * C[o,r] * A[i,r]
        return (self.C * self.lambdas) @ self.A.T

    @property
    def w_r(self) -> Float[Tensor, "d_out d_in"]:
        """Reconstruct W_r for compatibility with analysis code."""
        # W_r[o, i] = sum_r C[o,r] * B[i,r]
        return self.C @ self.B.T


def create_bilinear(d_in: int, d_out: int, mode: str = 'dense',
                    rank: int = None, bias: bool = False, gate: str = None):
    """
    Factory function to create appropriate bilinear layer.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        mode: 'dense' (original) or 'cp' (extension)
        rank: CP rank (required if mode='cp')
        bias: Include bias term
        gate: Gating function for dense mode

    Returns:
        BilinearDense or BilinearCP instance
    """
    if mode == 'dense':
        return BilinearDense(d_in, d_out, bias=bias, gate=gate)
    elif mode == 'cp':
        if rank is None:
            raise ValueError("rank required for CP mode")
        return BilinearCP(d_in, d_out, rank=rank, bias=bias)
    else:
        raise ValueError(f"Unknown mode: {mode}")
