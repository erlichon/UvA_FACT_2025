# Extension 2: Cross-Dataset Eigenvector Similarity Metrics Analysis

## Executive Summary

**Problem**: Standard subspace overlap metrics (mean cosine of principal angles, Grassmann distance) fail to discriminate between visually similar digit-letter pairs (e.g., "0" and "O") and dissimilar pairs (e.g., "0" and "X") when comparing bilinear MLP eigenvectors across MNIST and EMNIST datasets.

**Root Cause**: In bilinear models, class-specific information is encoded in **eigenvalue weightings**, not eigenvector directions. Within a model, the eigenvector matrix V is shared across all classes; only the eigenvalues λ_c differ per class.

**Solution**: We developed **Quadratic Form Similarity**, a metric that directly compares the class-specific weight matrices by incorporating both eigenvector alignment and eigenvalue patterns.

**Key Result**: Quadratic Form Similarity achieves:
- **Mean rank 1.2** for similar pairs (3/4 pairs rank #1 among all 26 letters)
- **Gap of 0.459** between similar (0.400) and dissimilar (-0.060) pairs
- **p < 0.0001** statistical significance

---

## Background: Why Subspace Metrics Fail

### The Bilinear Model Structure

A bilinear MLP computes class outputs as:

```
y_c = x^T A_c x
```

Where the class-specific weight matrix has eigendecomposition:

```
A_c = V @ diag(λ_c) @ V^T
```

**Critical insight**: Within a single trained model:
- **V (eigenvector matrix)** is SHARED across all classes
- **λ_c (eigenvalues)** are CLASS-SPECIFIC

This means two classes from the same model will have **identical** eigenvector subspaces but different eigenvalue patterns.

### Why Subspace Metrics Converge

When comparing MNIST digits to EMNIST letters:
1. Both models learn similar low-level features (edges, strokes, curves)
2. At high k, the eigenvector subspaces capture these shared "handwritten character features"
3. Subspace overlap converges to similar values for ALL pairs, destroying discrimination

**Evidence**: Mean Cosine similarity at k=20:
- Similar pairs: 0.679
- Dissimilar pairs: 0.657
- p-value: 0.065 (NOT significant!)

---

## Metrics Investigated

### 1. Mean Cosine of Principal Angles (Original)

**What it measures**: Alignment between eigenvector subspaces, ignoring eigenvalue magnitudes.

**Formula**:
```
sim = mean(cos(θ_i))
```
where θ_i are the principal angles between subspaces span(V_A[:k]) and span(V_B[:k]).

**Range**: [0, 1]

**Limitation**: Ignores which eigenvector directions are important (high eigenvalue) vs. unimportant (low eigenvalue).

### 2. Eigenvalue-Weighted Cosine Similarity

**What it measures**: Weighted average of cosine similarities, where high-eigenvalue directions contribute more.

**Formula**:
```
sim = Σᵢ Σⱼ |λᵢᴬ| · |λⱼᴮ| · cos²(vᵢᴬ, vⱼᴮ) / Z
```
where Z = (Σᵢ|λᵢᴬ|) · (Σⱼ|λⱼᴮ|) normalizes the result.

**Range**: [0, 1]

**Properties**:
- Only counts similarity when eigenvectors align AND both have high eigenvalues
- Uses absolute eigenvalues (sign doesn't matter)

### 3. Quadratic Form Similarity (RECOMMENDED)

**What it measures**: Direct similarity between the class-specific weight matrices.

**Formula**:
```
sim = trace(A · B) / (||A||_F · ||B||_F)
```

For low-rank approximations (top-k eigenvectors), this simplifies to:
```
sim = Σᵢ Σⱼ λᵢᴬ · λⱼᴮ · cos²(vᵢᴬ, vⱼᴮ) / (||λᴬ||₂ · ||λᴮ||₂)
```

**Range**: [-1, 1]

**Key Properties**:
- **Directly compares what the model computes** (x^T A x)
- **Eigenvalue signs matter**: same signs contribute positively, opposite signs cancel
- Can be negative if eigenvalue patterns are opposing

### 4. CKA (Centered Kernel Alignment)

**What it measures**: Structural similarity between eigenvalue-weighted representations.

**Formula**:
```
CKA = HSIC(K, L) / √(HSIC(K, K) · HSIC(L, L))
```
where K and L are Gram matrices of √|λ|-weighted eigenvectors.

**Range**: [0, 1]

**Properties**:
- Invariant to orthogonal transformations and isotropic scaling
- Captures whether representations encode similar structure

**Limitation**: Too invariant for our use case - fails to discriminate similar from dissimilar pairs.

---

## Results at k=20

### Metric Comparison Table

| Metric | Similar Mean | Dissimilar Mean | Gap | p-value | Significance |
|--------|-------------|-----------------|-----|---------|--------------|
| **Quadratic Form** | **0.400** | **-0.060** | **0.459** | **<0.0001** | **\*\*\*** |
| Eigenvalue-Weighted | 0.032 | 0.029 | 0.004 | 0.010 | * |
| Mean Cosine | 0.679 | 0.657 | 0.023 | 0.065 | n.s. |
| CKA | 0.957 | 0.957 | 0.000 | 0.986 | n.s. |

### Ranking Analysis

For each similar pair, we compute where the expected letter ranks among all 26 letters when sorted by similarity to the digit.

| Metric | 0-O | 1-I | 2-Z | 5-S | Mean Rank |
|--------|-----|-----|-----|-----|-----------|
| **Quadratic Form** | **1** | **2** | **1** | **1** | **1.2** |
| Eigenvalue-Weighted | 10 | 2 | 3 | 1 | 4.0 |
| Mean Cosine | 11 | 1 | 11 | 1 | 6.0 |
| CKA | 5 | 25 | 5 | 23 | 14.5 |

**Interpretation**: 
- Quadratic Form: 3/4 pairs rank #1, the 4th ranks #2
- Random baseline would be rank 13

---

## Why Quadratic Form Works

### 1. Direct Comparison of Model Computation

Quadratic Form similarity directly compares:
```
sim(A_digit, A_letter) ∝ trace(A_digit · A_letter)
```

This measures: "Would the digit and letter models produce similar outputs for the same input?"

### 2. Eigenvalue Signs Encode Class Information

In bilinear models:
- **Positive eigenvalues**: activate when input aligns with eigenvector
- **Negative eigenvalues**: suppress when input aligns with eigenvector

For visually similar classes (0 and O):
- Both should have similar patterns of positive/negative eigenvalues
- Quadratic Form captures this because λᵢᴬ · λⱼᴮ > 0 when signs match

For dissimilar classes:
- Eigenvalue patterns may oppose, leading to cancellation
- This explains the **negative** dissimilar mean (-0.060)

### 3. Mathematical Interpretation

The quadratic form inner product:
```
⟨A, B⟩_F = Σᵢ Σⱼ λᵢᴬ λⱼᴮ (vᵢᴬ · vⱼᴮ)²
```

Only contributes positively when:
1. Eigenvectors align (high cos²)
2. Eigenvalues have the same sign

This is exactly what we want: similarity when models use the same features in the same way.

---

## Conclusion

### Recommendation

**Use Quadratic Form Similarity** for comparing bilinear MLP eigenvectors across datasets or models when class-specific discrimination is required.

### Key Takeaways

1. **Subspace metrics fail** because they ignore eigenvalue information, which encodes class-specific structure in bilinear models.

2. **Quadratic Form Similarity** succeeds because it:
   - Directly compares weight matrices (what the model computes)
   - Respects eigenvalue signs (class-specific information)
   - Requires both eigenvector alignment AND similar eigenvalue patterns

3. **CKA fails** for this task because its invariances wash out the discriminative signal we need.

### Support for Extension 2 Hypothesis

These results support our hypothesis that regularized bilinear MLPs learn **transferable geometric features**:

- Similar digit-letter pairs (0-O, 1-I, 2-Z, 5-S) have high Quadratic Form similarity
- This indicates the models learned similar weight structures for visually similar shapes
- The discrimination is strong (p < 0.0001) and consistent (mean rank 1.2)

The Quadratic Form metric provides **quantitative evidence** that the learned eigenvector representations capture meaningful, transferable visual features rather than dataset-specific artifacts.

---

## Implementation Reference

The metrics are implemented in [`src/vision/subspace.py`](../src/vision/subspace.py):

- `compute_subspace_overlap()` - Original subspace metrics (mean_cos, grassmann, projection, hungarian)
- `compute_quadratic_form_similarity()` - Quadratic Form similarity
- `compute_eigenvalue_weighted_cosine()` - Eigenvalue-weighted cosine
- `compute_cka_similarity()` - CKA on weighted representations
- `compute_weighted_similarity()` - Unified wrapper function

Figures are generated by [`scripts/figures/generate_extension2_figures.py`](../scripts/figures/generate_extension2_figures.py).
