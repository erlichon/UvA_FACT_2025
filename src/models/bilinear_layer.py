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

    Args:
        d_in: Input dimension
        d_out: Output dimension
        bias: Include bias term
        gate: Gating function name
        variance_corrected_init: If True, apply d_in^0.25 scaling for Rich Training regime.
                                 This prevents "Lazy Training" pathology.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str = None,
                 variance_corrected_init: bool = False):
        super().__init__()
        self._original = OriginalBilinear(d_in, d_out, bias=bias, gate=gate)

        # Apply variance-corrected initialization for Rich Training regime
        # Scale = d_in^0.25, similar to Kaiming init but adapted for bilinear
        if variance_corrected_init:
            scale = d_in ** 0.25
            with torch.no_grad():
                # Scale the weight matrix (contains both w_l and w_r)
                # Original Bilinear inherits from nn.Linear with weight [2*d_out, d_in]
                self._original.weight.data *= scale
                # Scale bias if present
                if hasattr(self._original, 'bias') and self._original.bias is not None:
                    self._original.bias.data *= scale

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
        variance_corrected_init: If True, apply d_in^0.25 scaling for Rich Training regime.
    """

    def __init__(self, d_in: int, d_out: int, rank: int, bias: bool = False,
                 variance_corrected_init: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("Bias not supported for CP mode")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # CP factors: A, B for input projections, C for output
        # Base initialization: Xavier-style with bilinear correction
        # For bilinear: output variance ~ Var(x)^2 * Var(A) * Var(B) * Var(C) * rank
        # To maintain unit variance: std = (d_in * rank)^(-1/4) for A, B and (rank)^(-1/2) for C
        std_ab = (d_in * rank) ** (-0.25)
        std_c = rank ** (-0.5)

        # Apply variance correction for Rich Training regime
        if variance_corrected_init:
            scale = d_in ** 0.25
            std_ab *= scale
            std_c *= scale

        self.A = nn.Parameter(torch.randn(d_in, rank) * std_ab)
        self.B = nn.Parameter(torch.randn(d_in, rank) * std_ab)
        self.C = nn.Parameter(torch.randn(d_out, rank) * std_c)
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
                    rank: int = None, bias: bool = False, gate: str = None,
                    variance_corrected_init: bool = False):
    """
    Factory function to create appropriate bilinear layer.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        mode: 'dense' (original) or 'cp' (extension)
        rank: CP rank (required if mode='cp')
        bias: Include bias term
        gate: Gating function for dense mode
        variance_corrected_init: If True, apply d_in^0.25 scaling to push model
                                 into "Rich Training" regime. Default False for
                                 exact reproduction; enabled via config in train.py.

    Returns:
        BilinearDense or BilinearCP instance
    """
    if mode == 'dense':
        return BilinearDense(d_in, d_out, bias=bias, gate=gate,
                            variance_corrected_init=variance_corrected_init)
    elif mode == 'cp':
        if rank is None:
            raise ValueError("rank required for CP mode")
        return BilinearCP(d_in, d_out, rank=rank, bias=bias,
                         variance_corrected_init=variance_corrected_init)
    else:
        raise ValueError(f"Unknown mode: {mode}")
