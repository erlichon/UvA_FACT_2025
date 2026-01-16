# Claude Code Prompt: Person F (CP Implementation Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 2 Extension 2** of a FACT-AI course project as **Person F (CP Implementation Lead)**.

### Project Summary
We are reproducing and extending "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417). Phase 1 (reproduction) is complete. You are now working on:
- **Extension 2**: CP-Decomposition implementation with XOR sanity check

### Your Role
- Implement the `BilinearCP` class (structural low-rank)
- Create the `CPImageModel` for MNIST classification
- Validate CP implementation with XOR sanity check
- Hand off working CP code to Person D for rank sweep

### Timeline
- **Days 9-11** (Jan 16-18): Extension 2 - CP implementation + XOR gate
- **Day 11**: Handoff to Person D

### Dependencies
- Person A's infrastructure: `src/models/bilinear_layer.py` (has skeleton)
- Analysis utilities from Person B: `src/analysis/spectral.py`

---

## CRITICAL CONSTRAINTS

1. **CP IMPLEMENTATION MUST PASS XOR GATE**: Before handing off to Person D:
   - R=1 must FAIL on XOR (accuracy ~50%)
   - R>=2 must SUCCEED on XOR (accuracy >95%)
   - This validates the rank constraint is working correctly

2. **MATCH DENSE API**: The `BilinearCP` class must have the same interface as `BilinearDense`:
   - Same `forward(x)` signature
   - Same `w_l` and `w_r` properties (for eigendecomposition compatibility)

3. **FACTOR NAMING CONVENTION**: MUST match Person A's convention:
   - A: left input factors [d_in, rank]
   - B: right input factors [d_in, rank]
   - C: output factors [d_out, rank]
   - lambdas: scaling factors [rank]

4. **5 SEEDS**: All experiments use seeds `[42, 43, 44, 45, 46]`

---

## VERIFICATION & BUDGET REQUIREMENTS

> **IMPORTANT**: The team has a total budget of **25,000 SBUs** on Snellius. XOR gate uses NO budget.

### Local Verification (MANDATORY)

The XOR sanity check runs LOCALLY - no Snellius needed:

```bash
# Test XOR sanity check locally (should complete in <30 sec on CPU)
python src/test_xor_sanity.py

# Test CPImageModel with 2 epochs locally
python -c "
import torch
from src.models.cp_model import CPImageModel

model = CPImageModel(d_hidden=64, rank=8, n_classes=10)
x = torch.randn(32, 784)
y = model(x)
print(f'Output shape: {y.shape}')  # Should be [32, 10]
"
```

**Verification Checklist**:
- [ ] BilinearCP forward pass works
- [ ] w_l and w_r properties return correct shapes
- [ ] XOR gate check passes (R=1 fails, R>=2 succeeds)
- [ ] CPImageModel trains for 2 epochs without error
- [ ] CPImageModel.decompose() returns correct shapes

### Job Submission Protocol

1. **XOR gate is LOCAL ONLY** - No GPU needed, run on local machine
2. **Claude outputs job files ONLY** - Human submits if needed
3. **E2 uses ZERO Snellius budget** - All verification is local

---

## CP-DECOMPOSITION THEORY

### Standard Bilinear
```
y = (W_l @ x) * (W_r @ x)
```
This implicitly creates a third-order tensor with rank up to d_hidden.

### CP-Decomposed Bilinear
```
B[i,j,k] = sum_{r=1}^R lambda_r * C[i,r] * A[j,r] * B[k,r]
```
This explicitly constrains the tensor to rank R.

**Key Insight**: By setting R < d_hidden, we enforce structural low-rank WITHOUT needing regularization. This is our core hypothesis.

---

## FILE SPECIFICATIONS

### 1. BilinearCP Class

**Update `src/models/bilinear_layer.py`**:

Person A already created a skeleton. You must verify/complete it:

```python
class BilinearCP(nn.Module):
    """
    CP-Decomposed Bilinear Layer.

    The interaction tensor is explicitly parameterized as rank-R:
    B[out, in1, in2] = sum_r lambda[r] * C[out,r] * A[in1,r] * B[in2,r]

    This provides structural low-rank constraint (vs emergent low-rank from regularization).

    IMPORTANT: Factor naming convention (MUST MATCH Person A):
        - A: left input factors [d_in, rank]
        - B: right input factors [d_in, rank]
        - C: output factors [d_out, rank]
        - lambdas: scaling factors [rank]

    Args:
        d_in: Input dimension
        d_out: Output dimension
        rank: CP decomposition rank (REQUIRED)
        bias: Include bias (not implemented)
    """

    def __init__(self, d_in: int, d_out: int, rank: int, bias: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("Bias not supported for CP mode")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # CP factors - STANDARDIZED naming (matches Person A and CONTEXT.md):
        # A: left input factors [d_in, rank]
        # B: right input factors [d_in, rank]
        # C: output factors [d_out, rank]
        # lambdas: scaling factors [rank]
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.C = nn.Parameter(torch.randn(d_out, rank) * 0.02)
        self.lambdas = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [..., d_in]

        Returns:
            Output tensor [..., d_out]

        Math:
            y[..., k] = sum_r lambda[r] * (sum_i x[...,i] * A[i,r])
                                        * (sum_j x[...,j] * B[j,r])
                                        * C[k, r]
        Efficient form:
            left = x @ A           # [..., rank]
            right = x @ B          # [..., rank]
            hidden = left * right * lambdas  # [..., rank]
            y = hidden @ C.T       # [..., d_out]
        """
        left = x @ self.A           # [..., rank]
        right = x @ self.B          # [..., rank]
        hidden = left * right * self.lambdas  # [..., rank]
        return hidden @ self.C.T    # [..., d_out]

    @property
    def w_l(self):
        """
        Reconstruct left weight matrix for eigendecomposition compatibility.

        W_l[out, in] = sum_r sqrt(lambda[r]) * C[out,r] * A[in,r]

        Note: We split lambda equally between w_l and w_r for symmetry.
        """
        sqrt_lambda = self.lambdas.abs().sqrt()
        return (self.C * sqrt_lambda) @ self.A.T  # [d_out, d_in]

    @property
    def w_r(self):
        """
        Reconstruct right weight matrix for eigendecomposition compatibility.

        W_r[out, in] = sum_r sqrt(lambda[r]) * C[out,r] * B[in,r]
        """
        sqrt_lambda = self.lambdas.abs().sqrt()
        return (self.C * sqrt_lambda) @ self.B.T  # [d_out, d_in]

    def get_interaction_tensor(self):
        """
        Explicitly construct the interaction tensor for analysis.

        B[out, in1, in2] = sum_r lambda[r] * C[out,r] * A[in1,r] * B[in2,r]

        Returns:
            Tensor of shape [d_out, d_in, d_in]
        """
        # Use einsum for clarity
        return torch.einsum('or,ir,jr,r->oij', self.C, self.A, self.B, self.lambdas)
```

### 2. CP-Compatible Image Model

**`src/models/cp_model.py`**:
```python
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bilinear-decomposition-main"))

from shared.components import Linear
from src.models.bilinear_layer import BilinearCP


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

    def __init__(self, d_hidden: int = 256, rank: int = 32, n_classes: int = 10):
        super().__init__()
        self.d_hidden = d_hidden
        self.rank = rank
        self.n_classes = n_classes

        # Architecture matches original
        self.embed = Linear(784, d_hidden)
        self.bilinear = BilinearCP(d_hidden, d_hidden, rank=rank)
        self.head = Linear(d_hidden, n_classes)

    def forward(self, x: Float[Tensor, "batch 784"]) -> Float[Tensor, "batch n_classes"]:
        h = self.embed(x)
        h = self.bilinear(h)
        return self.head(h)

    def fit(self, train_data, test_data, epochs: int = 100, lr: float = 1e-3,
            weight_decay: float = 0.1, transform=None, verbose: bool = True):
        """
        Training loop matching original model interface.

        Args:
            train_data: Training dataset with .x and .y attributes
            test_data: Test dataset
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: AdamW weight decay
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

        NOTE: For CP models, eigendecomposition is done in input space (784-dim for MNIST),
        not hidden space like dense models. This is because the CP factorization directly
        constrains the interaction tensor rank.

        Returns:
            (eigenvalues, eigenvectors) tuple
            - eigenvalues: [n_classes, 784] (input space dimension)
            - eigenvectors: [n_classes, 784, 784] (each eigenvector is 784-dim)

        IMPORTANT: Shape differs from dense models!
            - Dense: [n_classes, d_hidden, 784] - limited by hidden dim
            - CP: [n_classes, 784, 784] - full input space eigendecomposition
        """
        # Get weight matrices - w_l and w_r are [d_hidden, d_hidden]
        w_l = self.bilinear.w_l  # [d_hidden, d_hidden]
        w_r = self.bilinear.w_r  # [d_hidden, d_hidden]
        w_e = self.embed.weight  # [d_hidden, 784]
        w_h = self.head.weight   # [n_classes, d_hidden]

        eigenvalues_list = []
        eigenvectors_list = []

        for c in range(self.n_classes):
            # Per-class interaction matrix in INPUT space
            w_h_c = w_h[c]  # [d_hidden]

            # Project to input space
            # M[i,j] = sum_h w_h[c,h] * (w_l[h,:] @ w_e)[i] * (w_r[h,:] @ w_e)[j]
            left_proj = w_l @ w_e    # [d_hidden, 784]
            right_proj = w_r @ w_e   # [d_hidden, 784]

            # Weighted sum over hidden dim
            M = torch.einsum('h,hi,hj->ij', w_h_c, left_proj, right_proj)

            # Symmetrize for real eigenvalues
            M_sym = 0.5 * (M + M.T)

            # Eigendecomposition in input space
            vals, vecs = torch.linalg.eigh(M_sym)

            # Sort by magnitude (descending)
            sorted_idx = vals.abs().argsort(descending=True)
            vals = vals[sorted_idx]
            vecs = vecs[:, sorted_idx]

            eigenvalues_list.append(vals)
            eigenvectors_list.append(vecs.T)  # [784, 784]

        eigenvalues = torch.stack(eigenvalues_list)    # [n_classes, 784]
        eigenvectors = torch.stack(eigenvectors_list)  # [n_classes, 784, 784]

        return eigenvalues, eigenvectors
```

### 3. XOR Sanity Check

**`src/test_xor_sanity.py`**:
```python
"""
XOR Sanity Check for CP-Bilinear Implementation

XOR is a rank-2 function: XOR(a,b) = a + b - 2*a*b
This requires at least rank-2 to represent.

This script validates:
- CP R=1 fails (accuracy ~50%)
- CP R>=2 succeeds (accuracy >95%)
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np


class XORModel(nn.Module):
    """Simple model for XOR task using CP bilinear."""

    def __init__(self, rank: int):
        super().__init__()
        from src.models.bilinear_layer import BilinearCP

        self.bilinear = BilinearCP(d_in=2, d_out=1, rank=rank)

    def forward(self, x):
        return self.bilinear(x).squeeze(-1)


def generate_xor_data(n_samples: int = 1000, device: str = 'cpu'):
    """Generate XOR dataset."""
    x = torch.randint(0, 2, (n_samples, 2), dtype=torch.float32, device=device)
    y = (x[:, 0] != x[:, 1]).float()  # XOR
    return x, y


def train_xor(rank: int, epochs: int = 500, lr: float = 0.1, device: str = 'cpu'):
    """Train model on XOR and return final accuracy."""
    model = XORModel(rank=rank).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train, y_train = generate_xor_data(1000, device)
    x_test, y_test = generate_xor_data(500, device)

    for epoch in range(epochs):
        model.train()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        preds = (test_logits > 0).float()
        accuracy = (preds == y_test).float().mean().item()

    return accuracy


def run_sanity_check(seeds: list = [42, 43, 44, 45, 46]):
    """Run XOR sanity check for different ranks."""
    print("=" * 60)
    print("XOR SANITY CHECK FOR CP-BILINEAR")
    print("=" * 60)
    print("\nExpected: R=1 fails (~50%), R>=2 succeeds (>95%)")
    print("-" * 60)

    results = {}
    for rank in [1, 2, 4, 8]:
        accuracies = []
        for seed in seeds:
            torch.manual_seed(seed)
            acc = train_xor(rank=rank, device='cpu')
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[rank] = (mean_acc, std_acc)

        status = "PASS" if (rank == 1 and mean_acc < 0.6) or (rank > 1 and mean_acc > 0.9) else "FAIL"
        print(f"Rank {rank}: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}% [{status}]")

    print("-" * 60)

    # Gate check
    r1_pass = results[1][0] < 0.6
    r2_pass = results[2][0] > 0.9

    print("\nGATE CHECK:")
    print(f"  [{'X' if r1_pass else ' '}] R=1 fails (accuracy < 60%)")
    print(f"  [{'X' if r2_pass else ' '}] R=2 succeeds (accuracy > 90%)")

    if r1_pass and r2_pass:
        print("\n*** XOR GATE PASSED - CP implementation is correct ***")
        return True
    else:
        print("\n*** XOR GATE FAILED - Debug CP implementation ***")
        return False


if __name__ == '__main__':
    run_sanity_check()
```

---

## EXPECTED OUTPUTS

### Files to Create/Update
```
src/
├── models/
│   ├── bilinear_layer.py      # Update BilinearCP class
│   └── cp_model.py            # NEW: CP-compatible image model
└── test_xor_sanity.py         # NEW: XOR gate check script
```

### Handoff to Person D (Day 11)
Provide:
1. Working `BilinearCP` class in `src/models/bilinear_layer.py`
2. XOR sanity check passes (screenshot or log)
3. `CPImageModel` in `src/models/cp_model.py`
4. Quick demo showing CPImageModel trains successfully

---

## EXECUTION CHECKLIST

### Day 9-10: Implementation
- [ ] Verify Person A's BilinearCP skeleton
- [ ] Complete/fix BilinearCP implementation
- [ ] Create `src/models/cp_model.py`
- [ ] Implement `CPImageModel.decompose()`

### Day 10-11: Validation
- [ ] Create `src/test_xor_sanity.py`
- [ ] Run XOR sanity check
- [ ] **GATE CHECK**: R=1 fails, R>=2 succeeds
- [ ] Test CPImageModel with 2 epochs

### Day 11: Handoff
- [ ] XOR gate passed
- [ ] Document any issues/notes for Person D
- [ ] Notify Person D that CP implementation is ready

---

## XOR GATE IS CRITICAL

The XOR gate validates the fundamental correctness of the CP implementation:

| Rank | Expected Accuracy | Reason |
|------|------------------|--------|
| R=1 | ~50% (random) | XOR is rank-2, cannot be represented |
| R=2 | >95% | Exactly sufficient rank |
| R>2 | >95% | More than sufficient rank |

If R=1 achieves >60% accuracy, the implementation is WRONG.
If R=2 achieves <90% accuracy, the implementation may be WRONG or undertrained.

**Do NOT hand off to Person D until XOR gate passes!**

---

**Begin by examining Person A's BilinearCP skeleton in `src/models/bilinear_layer.py`, then verify/complete the implementation. Run the XOR sanity check to validate.**

## PROMPT END
