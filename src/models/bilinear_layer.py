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
        cp_init_mode: Initialization mode - "fixed", "lambda", or "gated" (default: "lambda")
    """
        
    def __init__(self, d_in: int, d_out: int, rank: int, bias: bool = False, cp_init_mode: str = "lambda"):
        super().__init__()
        if bias:
            raise NotImplementedError("Bias not supported for CP mode")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.cp_init_mode = cp_init_mode

        if cp_init_mode == "fixed":
            # Fixed mode: Small scale initialization (0.02) to prevent activation explosion
            self.A = nn.Parameter(torch.randn(d_in, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(d_in, rank) * 0.02)
            self.C = nn.Parameter(torch.randn(d_out, rank) * 0.02)
            
            self.lambdas = nn.Parameter(torch.ones(rank))
        elif cp_init_mode == "lambda":
            # Lambda mode: Current implementation with sigma-calculated scale
            # Calculate specific std to preserve variance through the triple-product
            # Derived from: Var(out) = Rank * (d_in * sigma^2)^2 * sigma^2
            sigma = (1 / (rank * (d_in ** 2))) ** (1/6)
            
            # Initialize with this calculated scale
            self.A = nn.Parameter(torch.randn(d_in, rank) * sigma)
            self.B = nn.Parameter(torch.randn(d_in, rank) * sigma)
            self.C = nn.Parameter(torch.randn(d_out, rank) * sigma)
            
            # Lambda initialization with small Gaussian noise around 1.0
            # This breaks symmetry, ensuring each rank-1 component starts with slightly
            # different importance, helping optimizer prioritize which ranks to prune
            self.lambdas = nn.Parameter(torch.ones(rank) + torch.randn(rank) * 0.01)
        elif cp_init_mode == "gated":
            # Gated mode: Probabilistic gated approach with gate logits
            # Calculate specific std to preserve variance through the triple-product
            sigma = (1 / (rank * (d_in ** 2))) ** (1/6)
            
            # Initialize factors with calculated scale
            self.A = nn.Parameter(torch.randn(d_in, rank) * sigma)
            self.B = nn.Parameter(torch.randn(d_in, rank) * sigma)
            self.C = nn.Parameter(torch.randn(d_out, rank) * sigma)
            
            # Gate logits: Start with value 5.0 (puts initial Sigmoid probability near 1.0)
            # This keeps all ranks active at the start
            self.gate_logits = nn.Parameter(torch.ones(rank) * 5.0)
            
            # Learnable scaling factor
            self.scaling_factor = nn.Parameter(torch.ones(1))
        else:
            raise ValueError(f"Unknown cp_init_mode: {cp_init_mode}. Must be 'fixed', 'lambda', or 'gated'")

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Forward pass using vectorized matrix multiplications.
        
        Args:
            x: Input tensor [..., d_in]
            
        Returns:
            Output tensor [..., d_out]
        """
        # Step 1: Column normalization (Canonical Constraint)
        eps = 1e-8
        A_norm = self.A / (self.A.norm(dim=0, keepdim=True) + eps)
        B_norm = self.B / (self.B.norm(dim=0, keepdim=True) + eps)
        
        # Compute projections
        left = x @ A_norm          # [..., rank]
        right = x @ B_norm         # [..., rank]
        
        # Step 2: Mode-specific interaction scaling
        if self.cp_init_mode == "fixed":
            # Fixed mode: Simple lambda scaling
            # bypass canonical normalization step for fixed mode
            left = x @ self.A
            right = x @ self.B
            hidden = left * right * self.lambdas
        elif self.cp_init_mode == "lambda":
            # Lambda mode: Current gating logic with hard soft-threshold
            # We find the max importance and define a 'cutoff' zone.
            # Ranks with lambda below 5% of the max are likely noise.
            #test
            #uncomment below to bypass cononical normalization step
            # left = x @ self.A
            # right = x @ self.B
            with torch.no_grad():
                max_val = self.lambdas.abs().max()
                threshold = max_val * 0.05  # 5% threshold; tune this to 0.1 for more force
                
            # Differentiable masking: 
            # Forces small values to 0 while leaving important ones alone.
            # We use a 'Soft' approach to avoid abrupt gradient spikes.
            mask = (self.lambdas.abs() > threshold).float()
            gated_lambdas = self.lambdas * mask
            hidden = left * right * gated_lambdas
        elif self.cp_init_mode == "gated":
            # Gated mode: Concrete (Hard-Sigmoid) gate
            # Stretched Sigmoid that can be pushed past 0 and 1 boundaries, then clamped
            # Formula: gate_r = clamp(sigmoid(phi_r) * 1.1 - 0.05, 0, 1)
            gates = torch.clamp(torch.sigmoid(self.gate_logits) * 1.1 - 0.05, 0, 1)
            hidden = left * right * gates * self.scaling_factor
        else:
            raise ValueError(f"Unknown cp_init_mode: {self.cp_init_mode}")
        
        return hidden @ self.C.T

    @property
    def w_l(self) -> Float[Tensor, "d_out d_in"]:
        """
        Reconstructed left weight matrix for eigendecomposition compatibility.
        
        For CP decomposition: w_l = C @ diag(lambdas) @ A.T
        This absorbs the lambda scaling into w_l.
        
        Returns:
            Tensor of shape [d_out, d_in]
        """
        if self.cp_init_mode == "gated":
            # For gated mode, use gates instead of lambdas
            gates = torch.clamp(torch.sigmoid(self.gate_logits) * 1.1 - 0.05, 0, 1)
            return (self.C * gates.unsqueeze(0) * self.scaling_factor) @ self.A.T
        else:
            return (self.C * self.lambdas.unsqueeze(0)) @ self.A.T

    @property
    def w_r(self) -> Float[Tensor, "d_out d_in"]:
        """
        Reconstructed right weight matrix for eigendecomposition compatibility.
        
        For CP decomposition: w_r = C @ B.T
        Lambda scaling is absorbed into w_l, not w_r.
        
        Returns:
            Tensor of shape [d_out, d_in]
        """
        return self.C @ self.B.T


def create_bilinear(d_in: int, d_out: int, mode: str = 'dense',
                    rank: int = None, bias: bool = False, gate: str = None,
                    cp_init_mode: str = "lambda", variance_corrected_init: bool = False):
    """
    Factory function to create appropriate bilinear layer.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        mode: 'dense' (original) or 'cp' (extension)
        rank: CP rank (required if mode='cp')
        bias: Include bias term
        gate: Gating function for dense mode
        cp_init_mode: CP initialization mode - "fixed", "lambda", or "gated" (default: "lambda")
        variance_corrected_init: Apply variance correction for Rich Training regime (dense mode only)

    Returns:
        BilinearDense or BilinearCP instance
    """
    if mode == 'dense':
        return BilinearDense(d_in, d_out, bias=bias, gate=gate,
                            variance_corrected_init=variance_corrected_init)
    elif mode == 'cp':
        if rank is None:
            raise ValueError("rank required for CP mode")
        return BilinearCP(d_in, d_out, rank=rank, bias=bias, cp_init_mode=cp_init_mode)
    else:
        raise ValueError(f"Unknown mode: {mode}")
