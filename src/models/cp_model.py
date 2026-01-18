"""
CP-Bilinear Image Model

Image classifier using CP-decomposed bilinear layer.
Compatible with original model's .decompose() interface.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from jaxtyping import Float
from torch import Tensor
import pandas as pd
from einops import einsum

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from shared.components import Linear
from src.models.bilinear_layer import BilinearCP
from src.utils import safe_eigh


class CPImageModel(nn.Module):
    """
    MNIST classifier with CP-decomposed bilinear layer.

    Architecture:
        Input (784) -> Embed (d_hidden) -> BilinearCP (d_hidden) -> Head (10)

    Args:
        d_hidden: Hidden dimension
        rank: CP decomposition rank
        n_classes: Number of output classes
    """

    def __init__(self, d_hidden: int = 256, rank: int = 32, n_classes: int = 10, cp_init_mode: str = "lambda"):
        super().__init__()
        self.d_hidden = d_hidden
        self.rank = rank
        self.n_classes = n_classes
        self.cp_init_mode = cp_init_mode

        # Architecture matches original
        self.embed = Linear(784, d_hidden, bias=False)
        self.bilinear = BilinearCP(d_hidden, d_hidden, rank=rank, cp_init_mode=cp_init_mode)
        self.head = Linear(d_hidden, n_classes, bias=False)

    def forward(self, x: Float[Tensor, "batch 784"]) -> Float[Tensor, "batch n_classes"]:
        # Flatten input if needed (handles [batch, 1, 28, 28] or [batch, 28, 28] -> [batch, 784])
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        h = self.embed(x)
        h = self.bilinear(h)
        return self.head(h)

    def fit(self, train_data, test_data, epochs: int = 100, lr: float = 1e-3,
            weight_decay: float = 0.1, l1_coeff: float = 0.0, lambda_l1_coeff: float = 0.0,
            lambda_l0_coeff: float = 0.0, transform=None, verbose: bool = True):
        """
        Training loop matching original model interface.

        Args:
            train_data: Training dataset with .x and .y attributes
            test_data: Test dataset
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: AdamW weight decay
            l1_coeff: L1 regularization coefficient for factors B and C (encourages sparse features)
            lambda_l1_coeff: L1 regularization coefficient for lambda vector (encourages rank pruning, lambda/gated modes)
            lambda_l0_coeff: L0 proxy penalty coefficient for gate logits (gated mode only)
            transform: Optional augmentation transform
            verbose: Print progress

        Returns:
            DataFrame with training history
        """
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        history = []
        for epoch in range(epochs):
            # Training
            self.train()
            x, y = train_data.x, train_data.y
            if transform is not None:
                x = transform(x.view(-1, 1, 28, 28)).view(-1, 784)

            logits = self(x)
            loss = criterion(logits, y)
            
            # Add L1 penalty to factors B and C for sparse feature selection
            if l1_coeff > 0:
                l1_penalty = l1_coeff * (
                    self.bilinear.B.abs().sum() + 
                    self.bilinear.C.abs().sum()
                )
                loss = loss + l1_penalty
            
            # Add L1 penalty to lambda vector for rank pruning (lambda/fixed modes)
            # This provides "downward pressure" to drive effective rank toward minimal set
            if lambda_l1_coeff > 0 and hasattr(self.bilinear, 'lambdas'):
                lambda_l1_penalty = lambda_l1_coeff * self.bilinear.lambdas.abs().sum()
                #lambda_l1_penalty = lambda_l1_coeff * torch.sqrt(self.bilinear.lambdas.abs() + 1e-8).sum()
                loss = loss + lambda_l1_penalty
            
            # Add L0 proxy penalty for gate logits (gated mode)
            # This is a direct proxy for the L0 norm (count of active ranks)
            # Penalty: gamma * sum(sigmoid(gate_logits))
            if lambda_l0_coeff > 0 and hasattr(self.bilinear, 'gate_logits'):
                lambda_l0_penalty = lambda_l0_coeff * torch.sigmoid(self.bilinear.gate_logits).sum()
                loss = loss + lambda_l0_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_acc = (logits.argmax(-1) == y).float().mean().item()

            # Validation
            self.eval()
            with torch.no_grad():
                val_logits = self(test_data.x)
                val_loss = criterion(val_logits, test_data.y).item()
                val_acc = (val_logits.argmax(-1) == test_data.y).float().mean().item()

            history.append({
                'epoch': epoch,
                'train_loss': loss.item(),
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        return pd.DataFrame(history)

    def decompose(self):
        """
        Eigendecomposition of interaction tensor for interpretability analysis.

        Reconstructs the interaction tensor Q directly from CP factors (A, B, C, lambdas)
        using einsum to sum rank-1 outer products. Then projects to input space and
        performs eigendecomposition.

        Returns:
            (eigenvalues, eigenvectors) tuple
            - eigenvalues: [n_classes, 784] (input space dimension)
            - eigenvectors: [n_classes, 784, 784] (each eigenvector is 784-dim)

        Note: Shape matches dense models for compatibility with analysis code.
        """
        # Get CP factors
        A = self.bilinear.A  # [d_hidden, rank]
        B = self.bilinear.B  # [d_hidden, rank]
        C = self.bilinear.C  # [d_hidden, rank]
        
        # Get scaling factors based on mode
        if hasattr(self.bilinear, 'lambdas'):
            # Fixed or lambda mode: use lambdas directly
            lambdas = self.bilinear.lambdas  # [rank]
        elif hasattr(self.bilinear, 'gate_logits'):
            # Gated mode: use gate probabilities * scaling factor as effective lambdas
            gates = torch.clamp(torch.sigmoid(self.bilinear.gate_logits) * 1.1 - 0.05, 0, 1)
            lambdas = gates * self.bilinear.scaling_factor  # [rank]
        else:
            raise ValueError("BilinearCP must have either 'lambdas' or 'gate_logits'")
        
        # Get embedding and head weights
        w_e = self.embed.weight  # [d_hidden, 784]
        w_h = self.head.weight   # [n_classes, d_hidden]

        eigenvalues_list = []
        eigenvectors_list = []

        for c in range(self.n_classes):
            # Per-class interaction matrix in INPUT space
            # Reconstruct directly from CP factors using einsum
            
            # Get head weight for this class
            w_h_c = w_h[c]  # [d_hidden]
            
            # Project CP factors A and B to input space via embedding
            # A_proj[r, i] = sum_h A[h, r] * w_e[h, i]
            # B_proj[r, j] = sum_h B[h, r] * w_e[h, j]
            A_proj = einsum(A, w_e, "h r, h i -> r i")  # [rank, 784]
            B_proj = einsum(B, w_e, "h r, h j -> r j")  # [rank, 784]
            
            # Weight C factors by head weight for this class
            # C_weighted[r] = sum_h w_h[c, h] * C[h, r]
            C_weighted = einsum(w_h_c, C, "h, h r -> r")  # [rank]
            
            # Reconstruct interaction matrix in input space directly from CP factors
            # M[i, j] = sum_r lambda[r] * C_weighted[r] * A_proj[r, i] * B_proj[r, j]
            # This sums rank-1 outer products: lambda[r] * C_weighted[r] * (A_proj[r, :] ⊗ B_proj[r, :])
            M = einsum(lambdas, C_weighted, A_proj, B_proj, "r, r, r i, r j -> i j")

            # Symmetrize for real eigenvalues
            M_sym = 0.5 * (M + M.T)

            # Eigendecomposition in input space (MPS-safe)
            vals, vecs = safe_eigh(M_sym)

            # Sort by magnitude (descending)
            sorted_idx = vals.abs().argsort(descending=True)
            vals = vals[sorted_idx]
            vecs = vecs[:, sorted_idx]

            eigenvalues_list.append(vals)
            eigenvectors_list.append(vecs.T)  # [784, 784]

        eigenvalues = torch.stack(eigenvalues_list)    # [n_classes, 784]
        eigenvectors = torch.stack(eigenvectors_list)  # [n_classes, 784, 784]

        return eigenvalues, eigenvectors

