# DEPRECATED - DO NOT USE

> **Status**: DEPRECATED as of Jan 9, 2026
> **Reason**: CIFAR-10 and ViT targets are CANCELED. This document contains outdated strategy.
> **Canonical Source**: Use `WORKPLAN.md` exclusively for project planning.
> **New Strategy**: MNIST -> Rotated MNIST / EMNIST (robustness focus)

---

# ~~Extension Strategy Analysis: Low-Rank Bilinear ViT~~

## ~~Executive Summary~~

~~**Verdict**: High-reward but high-risk. Recommend a **tiered implementation** that secures baseline grades first, then progressively adds ambitious extensions.~~

---

## 1. Critical Analysis of the Proposal

### 1.1 What's Strong

| Aspect | Assessment |
|--------|------------|
| **Novelty** | Genuine research contribution - paper ignores scalability |
| **Rigor** | Tucker/CP decomposition is mathematically principled |
| **Relevance** | ViTs + LoRA-style methods are highly topical |
| **Impact** | Could be publication-worthy if executed well |

### 1.2 What's Risky

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Scope creep** | HIGH | Tiered implementation plan |
| **Compute requirements** | HIGH | Start with CIFAR-10, not ImageNet |
| **Technical complexity** | MEDIUM | Validate on MNIST first |
| **Time pressure** | HIGH | Prioritize reproduction baseline |
| **ViT integration** | MEDIUM | Can fallback to simple MLP |

---

## 2. Mathematical Deep Dive

### 2.1 The Parameter Problem

**Standard Bilinear Layer:**
```
output = (W_l @ x) * (W_r @ x)      # W_l, W_r: [d_out, d_in]
output = W_p @ hidden               # W_p: [d_model, d_out]
```

Parameter count: `2 * d_out * d_in + d_model * d_out`

For d_in = d_out = d_model = 1024 (ViT-Base scale):
- Standard: ~3M parameters per MLP
- Bilinear: ~4M parameters per MLP (2x W matrices + projection)

**Interaction Tensor (implicit):**
```
B[cls, i, j] = sum_h W_p[cls, h] * W_l[h, i] * W_r[h, j]
```
This is O(d³) to store explicitly, but **never materialized** in practice.

### 2.2 Key Insight: Bilinear is Already Low-Rank

The interaction tensor B is implicitly rank-d_hidden due to its factored form:
```
B = W_p ⊗₁ W_l ⊗₂ W_r   (mode products)
```

**The real question**: Can we reduce d_hidden while preserving interpretability?

### 2.3 Proposed Decomposition Strategies

#### Strategy A: LoRA-Style Factorization
```python
# Instead of W_l: [d_out, d_in]
# Use W_l = U_l @ V_l where U_l: [d_out, r], V_l: [r, d_in]

class LowRankBilinear(nn.Module):
    def __init__(self, d_in, d_out, rank):
        self.U_l = nn.Linear(rank, d_out, bias=False)
        self.V_l = nn.Linear(d_in, rank, bias=False)
        self.U_r = nn.Linear(rank, d_out, bias=False)
        self.V_r = nn.Linear(d_in, rank, bias=False)

    def forward(self, x):
        left = self.U_l(self.V_l(x))   # [batch, d_out]
        right = self.U_r(self.V_r(x))  # [batch, d_out]
        return left * right
```

**Parameter reduction**: From `2 * d_out * d_in` to `2 * (d_out + d_in) * r`

For d_in = d_out = 1024, r = 64: **32x reduction**

#### Strategy B: CP Decomposition (Direct Tensor)
```python
# Directly parameterize interaction as rank-R tensor
# B[o, i, j] ≈ sum_r lambda[r] * a[o, r] * b[i, r] * c[j, r]

class CPBilinear(nn.Module):
    def __init__(self, d_in, d_out, rank):
        self.A = nn.Parameter(torch.randn(d_out, rank))  # output factors
        self.B = nn.Parameter(torch.randn(d_in, rank))   # input factors (left)
        self.C = nn.Parameter(torch.randn(d_in, rank))   # input factors (right)
        self.lambdas = nn.Parameter(torch.ones(rank))    # scaling

    def forward(self, x):
        # x: [batch, d_in]
        left = x @ self.B      # [batch, rank]
        right = x @ self.C     # [batch, rank]
        hidden = left * right * self.lambdas  # [batch, rank]
        return hidden @ self.A.T  # [batch, d_out]
```

**Advantage**: Explicit rank control, direct spectral interpretation

#### Strategy C: Tucker Decomposition
```python
# B ≈ G ×₁ U_out ×₂ U_in1 ×₃ U_in2
# G: [r_out, r_in, r_in] core tensor

class TuckerBilinear(nn.Module):
    def __init__(self, d_in, d_out, ranks):
        r_out, r_in = ranks
        self.core = nn.Parameter(torch.randn(r_out, r_in, r_in))
        self.U_out = nn.Linear(r_out, d_out, bias=False)
        self.U_in = nn.Linear(d_in, r_in, bias=False)

    def forward(self, x):
        # x: [batch, d_in]
        z = self.U_in(x)  # [batch, r_in]
        # Compute z^T @ core @ z for each output
        interaction = torch.einsum('bi,oij,bj->bo', z, self.core, z)
        return self.U_out.weight @ interaction.T  # Hmm, dimensions need care
```

**Note**: Tucker is more expressive but harder to implement correctly.

### 2.4 Recommended: Start with CP Decomposition

**Why CP over Tucker:**
1. Simpler to implement and debug
2. Direct eigenvalue interpretation (lambdas = singular values)
3. Easier gradient flow
4. Well-studied convergence properties

**Why CP over LoRA-style:**
LoRA (`W = U @ V`) does **not** preserve the symmetric interaction structure required for eigendecomposition interpretability. The paper's core claim relies on:
```
B_sym = 0.5 * (B + B.T)  →  eigendecomposition  →  interpretable eigenvectors
```

LoRA destroys this symmetry. CP decomposition preserves it because:
```
B[o,i,j] = Σ_r λ_r * a[o,r] * b[i,r] * c[j,r]
```
can be symmetrized as:
```
B_sym[o,i,j] = Σ_r λ_r * a[o,r] * 0.5*(b[i,r]*c[j,r] + b[j,r]*c[i,r])
```

This is the mathematically correct choice for this specific transparency task.

---

## 3. Spectral Analysis Framework

**This is your "safety net" contribution.**

Even if the Low-Rank ViT fails to train or achieves low accuracy, the Effective Rank metric allows you to write a high-scoring report about *why* it failed:
- "The effective rank of the interaction tensor collapsed to X, indicating feature degeneracy"
- "Low-rank constraint forced premature convergence to local minima"
- "Spectral analysis reveals that rank R=64 is insufficient for CIFAR-10 complexity"

This directly addresses the "Transparency" topic by **quantifying model internal complexity** - a contribution the original paper lacks.

### 3.1 Effective Rank Metric

```python
def effective_rank(singular_values):
    """
    Compute effective rank via entropy of normalized singular values.
    Lower effective rank = more interpretable (sharper structure).
    """
    p = singular_values / singular_values.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy)
```

### 3.2 Interpretability Hypothesis

**Claim**: Models with lower effective rank learn more interpretable features.

**Test**:
1. Train bilinear models with varying rank constraints
2. Measure effective rank of learned interaction tensors
3. Correlate with qualitative interpretability (eigenspectrum sharpness)

### 3.3 Eigenvalue Decay Analysis

```python
def analyze_eigenspectrum(model):
    """
    Compute eigenvalue decay for bilinear layers.
    Sharp decay = interpretable; flat = memorization.
    """
    vals, vecs = model.decompose()

    # Normalize eigenvalues per class
    vals_norm = vals.abs() / vals.abs().sum(dim=-1, keepdim=True)

    # Compute metrics
    top1_ratio = vals_norm[:, -1].mean()  # Top eigenvalue dominance
    top5_ratio = vals_norm[:, -5:].sum(dim=-1).mean()  # Top-5 coverage
    eff_rank = effective_rank(vals.abs().mean(dim=0))

    return {
        'top1_ratio': top1_ratio,
        'top5_ratio': top5_ratio,
        'effective_rank': eff_rank
    }
```

---

## 4. Tiered Implementation Plan

### Tier 1: Baseline (MUST COMPLETE - Days 1-7)
**Goal**: Solid reproduction to secure passing grade

- [ ] Reproduce toy task interaction matrices
- [ ] Reproduce MNIST eigendecomposition
- [ ] Verify regularization findings
- [ ] Document any discrepancies

**Deliverable**: Working reproduction notebook

### Tier 2: Safe Extensions (SHOULD COMPLETE - Days 8-12)
**Goal**: Satisfy "beyond reproduction" requirement

- [ ] Fashion-MNIST with same methodology
- [ ] Systematic regularization ablation
- [ ] Spectral analysis metrics (effective rank, eigenvalue decay)

**Deliverable**: Extension results with quantitative interpretability metrics

### Tier 3: Low-Rank Bilinear (COULD COMPLETE - Days 8-11)
**Goal**: Novel architectural contribution

- [ ] Implement `CPBilinear` layer
- [ ] **GATE CHECK (Day 9): Test on toy XOR task first**
  - If CPBilinear cannot learn XOR with rank < d_hidden → abort Tiers 4-5
  - Pivot to spectral analysis as primary contribution
- [ ] Test on MNIST: accuracy vs. rank trade-off
- [ ] Compare eigenspectrum: full-rank vs. low-rank
- [ ] **PRIMARY TARGET: Scale to CIFAR-10**

**Deliverable**: Low-rank bilinear layer with empirical analysis on CIFAR-10

**Why CIFAR-10 is the primary target (not Tiny-ImageNet):**
- CIFAR-10 allows rapid iteration (hours vs. days)
- Going from 28x28 grayscale to 32x32 RGB is the mathematical hurdle
- Proves "scalability" claim without excessive compute
- Only attempt Tiny-ImageNet if CIFAR-10 runs are clean by Day 11

### Tier 4: Vision Transformer (STRETCH - only if ahead by Day 11)
**Goal**: High-impact result

- [ ] Replace ViT-Tiny MLP with CPBilinear
- [ ] Train on CIFAR-10 (not ImageNet)
- [ ] Compare "concept atoms" vs. attention maps

**Deliverable**: Bilinear ViT proof-of-concept

### Tier 5: BiDoRA / Tiny-ImageNet (REACH - only if Tier 4 succeeds)
**Goal**: Maximum impact

- [ ] Magnitude-direction decoupling for stability (if training unstable)
- [ ] Tiny-ImageNet only if CIFAR-10 ViT works cleanly

---

## 5. Pragmatic Recommendations

### 5.1 Don't Skip Reproduction

The course manual is clear: reproduction must be solid. Graders will check if basic claims are verified before looking at extensions.

### 5.2 Start with CIFAR-10, Not ImageNet

| Dataset | Resolution | Classes | Training Time | GPU Memory |
|---------|------------|---------|---------------|------------|
| MNIST | 28x28 | 10 | Minutes | <1GB |
| CIFAR-10 | 32x32 | 10 | Hours | ~4GB |
| Tiny-ImageNet | 64x64 | 200 | Days | ~16GB |
| ImageNet-100 | 224x224 | 100 | Weeks | ~32GB |

**Recommendation**: CIFAR-10 is the sweet spot for novelty vs. feasibility.

### 5.3 Focus on Spectral Analysis

Even if low-rank bilinear doesn't outperform, **quantitative interpretability metrics** are a solid contribution:
- Effective rank
- Eigenvalue decay rate
- Top-k eigenvalue coverage

This provides a framework for evaluating interpretability that the original paper lacks.

### 5.4 ViT is Optional

A simpler "Bilinear ResNet" or "Bilinear MLP-Mixer" might be more tractable:
- Fewer moving parts than attention
- Still novel (paper only does simple MLPs)
- Easier to debug

---

## 6. Concrete Next Steps

### Immediate (Today)
1. Set up development environment
2. Run original tutorials to verify code works
3. Start Tier 1 reproduction

### This Week
1. Complete toy task reproduction
2. Complete MNIST reproduction
3. Begin Fashion-MNIST extension
4. Implement spectral analysis utilities

### Next Week
1. Implement `CPBilinear` layer
2. Test on MNIST with varying ranks
3. Scale to CIFAR-10
4. Write report draft

### Final Week
1. Polish results
2. (Optional) ViT integration
3. Finalize report and presentation

---

## 7. Risk Mitigation Checklist

- [ ] **Backup plan if low-rank fails**: Spectral analysis alone is publishable
- [ ] **Backup plan if CIFAR-10 fails**: Fashion-MNIST + ablations are sufficient
- [ ] **Backup plan if ViT fails**: Simple MLP on CIFAR-10 is still novel
- [ ] **Compute fallback**: Google Colab Pro ($10/month) if local GPU insufficient
- [ ] **Time buffer**: Leave 3 days before deadline for unexpected issues

---

## 8. Summary (REVISED - Jan 8, 2026)

### 8.1 The Kill List (Scope Management)

| Extension | Decision | Rationale |
|-----------|----------|-----------|
| Fashion-MNIST | **KEEP** | Low-hanging fruit, proves not overfitted to MNIST digits |
| Regularization Study | **DE-PRIORITIZE** | Paper already claims noise is crucial; confirmation ≠ extension |
| CIFAR-10 | **PRIMARY TARGET** | Proves method works on "real images" (color/texture) |
| ViT Integration | **KILL** | Too risky for timeline; CIFAR-10 is sufficient |
| Tiny-ImageNet | **KILL** | Massive compute overhead; not needed to prove scalability |
| BiDoRA | **KILL** | Nice-to-have but not core contribution |

### 8.2 Final Priority Matrix

| Component | Priority | Risk | Reward | Deadline |
|-----------|----------|------|--------|----------|
| Reproduction (Toy + MNIST) | **MUST** | Low | Required | Jan 11 |
| CPBilinear + XOR Check | **MUST** | Medium | High | Jan 13 |
| Fashion-MNIST | HIGH | Low | Medium | Jan 14 |
| Spectral Analysis Metrics | HIGH | Low | High | Jan 14 |
| CPBilinear on MNIST | HIGH | Medium | High | Jan 15 |
| **CPBilinear on CIFAR-10** | **PRIMARY** | Medium | High | **Jan 16** |
| ~~ViT Integration~~ | ~~KILLED~~ | - | - | - |
| ~~Tiny-ImageNet~~ | ~~KILLED~~ | - | - | - |

### 8.3 Critical Gates

| Gate | Date | Condition | Action if FAIL |
|------|------|-----------|----------------|
| XOR Sanity | **Jan 13** | CPBilinear learns XOR with R>=2 | Pivot to spectral analysis only |
| CIFAR-10 | **Jan 16** | Training converges | Report negative result + analysis |
| **RESULT FREEZE** | **Jan 17** | All experiments done | Write with available results |

### 8.4 The "Transparency" Narrative

Report introduction must connect to FACT-AI Topic 1.4:

> "Current mechanistic interpretability methods (like SAEs) are post-hoc and expensive. Weight-based methods are intrinsic but unscalable. Our Low-Rank CP formulation makes intrinsic interpretability scalable by treating rank R as a hyperparameter for the **Interpretability-Efficiency Trade-off**."

**Bottom line**: Result Freeze is Jan 17. CIFAR-10 is the primary deliverable. ViT/ImageNet are killed.
