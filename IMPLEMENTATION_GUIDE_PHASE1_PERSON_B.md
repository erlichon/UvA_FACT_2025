# Phase 1 Implementation Guide: Person B (Analysis Lead)

> **Purpose**: This document provides complete specifications for implementing Phase 1 (Reproduction) analysis and visualization. Load this into a Claude instance along with the prompt at the bottom.

---

## 1. Project Context

**Goal**: Reproduce Section 4 (Vision) of "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., 2024).

**Status**: Person A infrastructure is **COMPLETE**. Training code, configs, and spectral utilities are ready.

**Your Role**: Person B - Analysis Lead
- Deep understanding of paper's eigendecomposition theory
- Create visualization functions (Person A created spectral metrics)
- Generate Figure 3/4 equivalents (eigenspectrum, eigenvectors)
- Ablation analysis: which regularization contributes to interpretability
- Complete reproduction notebook with statistical analysis

**What Person A Delivered**:
- `src/train.py` - Training script with wandb + codecarbon
- `src/models/bilinear_layer.py` - BilinearDense (wrapper), BilinearCP (extension)
- `src/vision/spectral.py` - effective_rank, top_k_coverage, spectral_summary
- `configs/mnist_dense_*.yaml` - 4 ablation configs (none, noise, wd, full)
- `jobs/train_array.job` - SLURM job for 20 runs (4 configs x 5 seeds)

**What You Need to Create**:
- `src/vision/visualization.py` - Plotting functions
- `notebooks/01_reproduction.ipynb` - Analysis notebook
- `docs/paper_baselines.md` - Expected values from paper
- Generated figures in `results/phase1/figures/`

---

## 2. Paper Deep-Dive: Section 4 (Vision)

### 2.1 Core Theory

**Bilinear Layer Decomposition**:
The bilinear MLP computes: `y = (W_l @ x) ⊙ (W_r @ x)`

This can be expressed as a third-order interaction tensor:
```
B[c, i, j] = Σ_h W_head[c, h] * W_l[h, i] * W_r[h, j]
```

Where:
- `c` = output class (0-9 for MNIST)
- `i, j` = input pixel indices
- `h` = hidden dimension

**Symmetrization** (crucial for eigendecomposition):
```
B_sym = 0.5 * (B + B.transpose(-1, -2))
```

This removes arbitrary asymmetry from the factorization.

**Eigendecomposition**:
```
B_sym[c] = V[c] @ diag(λ[c]) @ V[c].T
```

- `λ[c]` = eigenvalues for class c (sorted ascending)
- `V[c]` = eigenvectors for class c (columns)

**Interpretation**:
- Large positive eigenvalues → features that INCREASE class probability
- Large negative eigenvalues → features that DECREASE class probability
- Near-zero eigenvalues → irrelevant directions
- **Low-rank structure** = few large eigenvalues = interpretable

### 2.2 Key Findings from Paper (Section 4)

| Finding | Details |
|---------|---------|
| **Regularization enables interpretability** | Without noise/WD, model overfits → high-rank interaction tensor |
| **Noise is crucial** | Gaussian noise (std=0.4) forces model to learn robust features |
| **Weight decay helps** | WD=0.5 encourages low-rank solutions |
| **Eigenvectors are digit-like** | Top eigenvectors resemble digit templates (with regularization) |
| **~10 significant eigenvalues per class** | Paper claims ~10 dimensions capture digit structure |

### 2.3 Figures to Reproduce

**Figure 3: Eigenspectrum Analysis**
- X-axis: Eigenvalue index (sorted by magnitude)
- Y-axis: Eigenvalue magnitude
- Multiple lines: one per class (0-9) or comparison (reg vs no-reg)
- Shows: Sharp decay with regularization, flat without

**Figure 4: Eigenvector Visualization**
- Grid of 28×28 images
- Rows: Different classes
- Columns: Top-k eigenvectors (positive and negative)
- Shows: Digit-like patterns with regularization, noise patterns without

---

## 3. Directory Structure

```
UvA_FACT_2025/
├── src/
│   ├── models/
│   │   └── bilinear_layer.py    # [PERSON A] BilinearDense, BilinearCP
│   ├── analysis/
│   │   ├── __init__.py          # [PERSON A] Created
│   │   ├── spectral.py          # [PERSON A] effective_rank, top_k_coverage, etc.
│   │   └── visualization.py     # [PERSON B] YOUR CODE
│   └── train.py                 # [PERSON A] Training script
├── configs/
│   └── mnist_dense_*.yaml       # [PERSON A] 4 config files
├── notebooks/
│   └── 01_reproduction.ipynb    # [PERSON B] YOUR CODE
├── results/
│   └── phase1/
│       ├── checkpoints/         # [PERSON A] Model checkpoints
│       └── figures/             # [PERSON B] YOUR OUTPUT
└── docs/
    └── paper_baselines.md       # [PERSON B] YOUR CODE
```

---

## 4. File Specifications

### 4.1 `src/vision/spectral.py` [ALREADY IMPLEMENTED BY PERSON A]

Person A has already implemented these functions. You can use them directly:

```python
from src.vision.spectral import (
    effective_rank,           # Entropy-based effective rank (Roy & Bhattacharyya 2007)
    top_k_coverage,           # Fraction of variance in top-k eigenvalues
    eigenvalue_decay_rate,    # Ratio of 2nd to 1st eigenvalue (lower = faster decay)
    spectral_summary,         # Full summary dict with mean/std across classes
    load_checkpoint_eigenvalues,  # Load pre-computed eigenvalues/eigenvectors
)
```

**Usage Examples**:
```python
# Load eigenvalues from checkpoint
vals, vecs = load_checkpoint_eigenvalues('results/phase1/checkpoints/mnist_dense_full_seed42.pt')
print(vals.shape)  # [10, 256] - 10 classes, 256 eigenvalues each

# Compute metrics
eff_rank = effective_rank(vals)  # [10] - one per class
top5 = top_k_coverage(vals, k=5)  # [10]
summary = spectral_summary(vals)  # dict with mean/std of all metrics
```

**Checkpoint Format** (what Person A saves):
```python
checkpoint = {
    'eigenvalues': tensor([10, 256]),     # Pre-computed eigenvalues
    'eigenvectors': tensor([10, 256, 784]),  # Pre-computed eigenvectors
    'config': {...},                       # Experiment config
    'metrics': {'train_acc': ..., 'val_acc': ..., 'effective_rank': ...},
    'seed': 42,
}
```

---

### 4.2 `src/vision/visualization.py`

**Purpose**: Publication-quality plotting functions.

```python
"""
Visualization Utilities for Bilinear MLP Interpretability

Functions for reproducing paper figures:
- Figure 3: Eigenspectrum plots
- Figure 4: Eigenvector grids
- Custom: Accuracy vs Effective Rank trade-off
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
```

**Functions to Implement**:

#### `plot_eigenspectrum(eigenvalues, ...) -> Figure`
```python
def plot_eigenspectrum(
    eigenvalues: Dict[str, Tensor],
    title: str = "Eigenspectrum Comparison",
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True,
    top_k: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot eigenvalue spectra for multiple models (Figure 3 equivalent).

    Args:
        eigenvalues: Dict mapping model name to eigenvalues [n_classes, n]
                    Will plot mean across classes with std shading
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis
        top_k: Only plot top-k eigenvalues (by magnitude)
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object

    Example:
        >>> vals = {"No Reg": model1_vals, "Full Reg": model2_vals}
        >>> fig = plot_eigenspectrum(vals, save_path="fig3.pdf")
    """
```

#### `plot_eigenspectrum_per_class(eigenvalues, ...) -> Figure`
```python
def plot_eigenspectrum_per_class(
    eigenvalues: Tensor,
    class_names: List[str] = None,
    title: str = "Eigenspectrum by Class",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot eigenspectrum for each class separately.

    Args:
        eigenvalues: Tensor [n_classes, n_eigenvalues]
        class_names: Names for each class (default: 0-9 for MNIST)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure with subplot per class or overlaid lines
    """
```

#### `plot_eigenvectors_grid(eigenvectors, ...) -> Figure`
```python
def plot_eigenvectors_grid(
    eigenvectors: Tensor,
    eigenvalues: Tensor,
    n_pos: int = 3,
    n_neg: int = 3,
    classes: List[int] = None,
    img_shape: Tuple[int, int] = (28, 28),
    title: str = "Top Eigenvectors",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot grid of top eigenvectors (Figure 4 equivalent).

    Shows top positive and negative eigenvectors for each class.
    Eigenvectors are reshaped to image dimensions.

    Args:
        eigenvectors: [n_classes, n_components, d_input]
        eigenvalues: [n_classes, n_components] for sorting
        n_pos: Number of top positive eigenvectors to show
        n_neg: Number of top negative eigenvectors to show
        classes: Which classes to plot (default: all)
        img_shape: Shape to reshape eigenvectors (28, 28 for MNIST)
        title: Plot title
        save_path: If provided, save figure

    Returns:
        matplotlib Figure with grid of eigenvector images

    Layout:
        Rows: Classes (0-9 or selected)
        Columns: [neg_n, ..., neg_1, pos_1, ..., pos_n]
    """
```

#### `plot_accuracy_vs_effective_rank(results, ...) -> Figure`
```python
def plot_accuracy_vs_effective_rank(
    results: "pd.DataFrame",
    title: str = "Accuracy vs Interpretability Trade-off",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of accuracy vs effective rank for all experiments.

    This visualizes the trade-off between performance and interpretability.

    Args:
        results: DataFrame with columns:
                - 'config': experiment config name
                - 'accuracy': test accuracy (mean across seeds)
                - 'accuracy_std': std of accuracy
                - 'effective_rank': mean effective rank
                - 'effective_rank_std': std of effective rank
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure with scatter plot + error bars
    """
```

---

### 4.3 `notebooks/01_reproduction.ipynb`

**Purpose**: Complete reproduction notebook with all Phase 1 results.

**Structure**:

```markdown
# Phase 1: Reproducing Bilinear MLP Vision Experiments

## 1. Setup
- Import libraries
- Load trained models (from Person A's checkpoints)
- Verify models loaded correctly

## 2. Eigenspectrum Analysis
### 2.1 Extract Eigenspectra
- Call model.decompose() for all 4 configs
- Compute metrics (effective rank, coverage)

### 2.2 Figure 3: Eigenspectrum Comparison
- Plot no_reg vs full_reg eigenvalue decay
- Show low-rank structure emerges with regularization

### 2.3 Per-Class Analysis
- Eigenspectrum for each digit class
- Verify ~10 significant eigenvalues per class

## 3. Eigenvector Visualization
### 3.1 Figure 4: Without Regularization
- Show overfitting patterns (noise-like)

### 3.2 Figure 4: With Regularization
- Show interpretable digit-like patterns

### 3.3 Comparison
- Side-by-side for specific classes

## 4. Ablation Study
### 4.1 Regularization Components
- Compare: none, noise_only, wd_only, full
- Which contributes more to interpretability?

### 4.2 Statistical Analysis
- t-test: effective_rank(reg) < effective_rank(no_reg)
- Effect sizes

## 5. Accuracy vs Interpretability
### 5.1 Trade-off Plot
- Scatter: accuracy vs effective rank
- Pareto frontier

### 5.2 Discussion
- Is there a trade-off?
- Optimal regularization?

## 6. Comparison with Paper
### 6.1 Metrics Table
| Metric | Paper | Ours (mean ± std) |
|--------|-------|-------------------|
| Accuracy (no reg) | ~97% | ? |
| Accuracy (full reg) | ~95% | ? |
| Effective Rank (no reg) | ~150-200 | ? |
| Effective Rank (full reg) | ~20-40 | ? |

### 6.2 Discrepancy Analysis
- Any significant differences?
- Possible explanations?

## 7. Gate Check Verification
- [ ] Effective rank ratio < 0.5
- [ ] Eigenvectors resemble digits
- [ ] Accuracy within 2% of paper

## 8. Conclusions
- Summary of reproduction findings
- Ready for Phase 2 extensions
```

---

### 4.4 `docs/paper_baselines.md`

**Purpose**: Document expected values from paper for comparison.

```markdown
# Paper Baseline Values: Section 4 (Vision)

Source: Pearce et al. (2024) "Bilinear MLPs enable weight-based mechanistic interpretability"

## Experimental Setup (Paper)
- Dataset: MNIST
- Architecture: Single bilinear layer, d_hidden=256
- Training: 100 epochs, AdamW, lr=1e-3
- Regularization: Gaussian noise (std=0.4), weight decay (0.5)

## Reported Results

### Accuracy
| Configuration | Test Accuracy |
|---------------|---------------|
| No regularization | ~97-98% |
| With regularization | ~94-95% |

Note: Paper prioritizes interpretability over accuracy.

### Eigenspectrum Characteristics
| Configuration | Effective Rank (approx) | Top-10 Coverage |
|---------------|-------------------------|-----------------|
| No regularization | ~150-200 (high) | ~20-30% (low) |
| With regularization | ~20-40 (low) | ~70-90% (high) |

### Eigenvector Quality (Qualitative)
- **No regularization**: Overfitting patterns, noise-like, class-specific but not interpretable
- **With regularization**: Digit-like templates, clear positive/negative features

### Key Claims to Verify
1. "Regularization induces low-rank structure" (Section 4.2)
2. "~10 eigenvalues per class capture digit structure" (Section 4.3)
3. "Eigenvectors resemble digit templates" (Figure 4)
4. "Noise augmentation is crucial" (Section 4.4)

## Our Success Criteria
- Effective rank ratio (reg/no-reg) < 0.5
- Top eigenvectors visually resemble digits
- Accuracy within 2% of paper values
```

---

## 5. Analysis Workflow

### Day 1: Paper Study + Setup
```
1. Read Section 4 thoroughly
2. Understand eigendecomposition theory (Section 2-3)
3. Study original code: model.decompose(), plotting.py
4. Create docs/paper_baselines.md
5. Verify Person A's spectral.py functions work:
   - Test load_checkpoint_eigenvalues()
   - Test effective_rank(), top_k_coverage()
```

### Day 2: Visualization Implementation
```
1. Implement src/vision/visualization.py
   - plot_eigenspectrum_comparison() - Figure 3
   - plot_eigenvectors_grid() - Figure 4
   - plot_accuracy_vs_effective_rank()
   - plot_ablation_summary()

2. Test with one checkpoint (if available)
```

### Day 3: Notebook Setup
```
1. Create notebooks/01_reproduction.ipynb structure
2. Implement all plotting cells
3. Generate draft figures from test checkpoint
```

### Day 4-5: Full Analysis (after cluster runs complete)
```
1. Load all 20 checkpoints (eigenvalues pre-computed)
2. Generate all figures
3. Compute statistics (mean, std across seeds)
4. Ablation analysis
5. Compare with paper baselines
```

### Day 6: Polish
```
1. Publication-quality figures (PDF export)
2. Complete notebook narrative
3. Verify gate check criteria
4. Draft Phase 1 summary
```

### Day 7: Gate Check
```
1. Review with Person A
2. Final verification
3. Plan Phase 2
```

---

## 6. Code Snippets

### Loading Checkpoints (Simple - Eigenvalues Pre-computed)
```python
import torch
from pathlib import Path
from src.vision.spectral import load_checkpoint_eigenvalues, spectral_summary

# Load a single checkpoint
vals, vecs = load_checkpoint_eigenvalues('results/phase1/checkpoints/mnist_dense_full_seed42.pt')

# Load full checkpoint for all data
checkpoint = torch.load('results/phase1/checkpoints/mnist_dense_full_seed42.pt', map_location='cpu')
config = checkpoint['config']
metrics = checkpoint['metrics']
seed = checkpoint['seed']
```

### Loading All Checkpoints for Comparison
```python
from pathlib import Path
import pandas as pd

def load_all_results(checkpoint_dir: str) -> pd.DataFrame:
    """Load metrics from all checkpoints."""
    results = []
    for config in ['none', 'noise', 'wd', 'full']:
        for seed in [42, 43, 44, 45, 46]:
            path = Path(checkpoint_dir) / f"mnist_dense_{config}_seed{seed}.pt"
            if not path.exists():
                continue

            vals, _ = load_checkpoint_eigenvalues(str(path))
            checkpoint = torch.load(path, map_location='cpu')
            summary = spectral_summary(vals)

            results.append({
                'config': config,
                'seed': seed,
                'accuracy': checkpoint['metrics']['val_acc'],
                **summary,
            })

    return pd.DataFrame(results)

# Usage
results_df = load_all_results('results/phase1/checkpoints')
print(results_df.groupby('config').mean())
```

### Statistical Testing
```python
from scipy import stats
import numpy as np

def compare_effective_ranks(ranks_noreg: np.ndarray, ranks_reg: np.ndarray):
    """Statistical comparison of effective ranks."""
    # Independent t-test (different configurations)
    t_stat, p_value = stats.ttest_ind(ranks_noreg, ranks_reg)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((ranks_noreg.var() + ranks_reg.var()) / 2)
    cohens_d = (ranks_noreg.mean() - ranks_reg.mean()) / pooled_std

    # Ratio
    ratio = ranks_reg.mean() / ranks_noreg.mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ratio': ratio,
        'gate_check_pass': ratio < 0.5
    }

# Usage
noreg_ranks = results_df[results_df['config'] == 'none']['effective_rank_mean'].values
full_ranks = results_df[results_df['config'] == 'full']['effective_rank_mean'].values
stats_result = compare_effective_ranks(noreg_ranks, full_ranks)
print(f"Ratio: {stats_result['ratio']:.3f}, p-value: {stats_result['p_value']:.6f}")
```

---

## 7. Integration with Person A

### Checkpoint Format (IMPLEMENTED)
Person A saves checkpoints in this format:

```python
checkpoint = {
    'config': {
        'mode': 'dense',
        'd_hidden': 256,
        'd_input': 784,
        'n_classes': 10,
        'noise_std': 0.4,      # or 0.0
        'weight_decay': 0.5,   # or 0.0
        'epochs': 100,
        'lr': 0.001,
        'batch_size': 64,
    },
    'model_state_dict': model.state_dict(),
    'metrics': {
        'train_acc': 0.98,
        'val_acc': 0.95,
        'effective_rank': 45.2,
    },
    'seed': 42,
    # PRE-COMPUTED EIGENVALUES (no need to call model.decompose())
    'eigenvalues': tensor([10, 256]),     # [n_classes, d_hidden]
    'eigenvectors': tensor([10, 256, 784]),  # [n_classes, d_hidden, d_input]
}
```

### Checkpoint Naming Convention
```
results/phase1/checkpoints/mnist_dense_{config}_seed{seed}.pt
```
Where `config` is one of: `none`, `noise`, `wd`, `full`

### Sync Points
- **Person A COMPLETE**: Infrastructure ready, waiting for cluster runs
- **After cluster runs**: 20 checkpoints will be available
- **Gate check**: Verify effective rank ratio < 0.5 between full and none configs

---

## 8. Prompt for New Claude Instance

Copy this prompt to start implementation:

---

**PROMPT START**

```
I am implementing Phase 1 (Reproduction) of the FACT-AI bilinear interpretability project as Person B (Analysis Lead).

Please read the following context files:
- @CLAUDE.md - Project setup and conventions
- @CONTEXT.md - Current project state
- @IMPLEMENTATION_GUIDE_PHASE1_PERSON_B.md - Detailed analysis specifications

INFRASTRUCTURE STATUS: Person A has COMPLETED the infrastructure:
- src/vision/spectral.py exists with effective_rank, top_k_coverage, spectral_summary, load_checkpoint_eigenvalues
- Checkpoints include pre-computed eigenvalues/eigenvectors
- Training configs are ready in configs/mnist_dense_*.yaml

My tasks:
1. Create docs/paper_baselines.md with expected values from paper Section 4
2. Implement src/vision/visualization.py with:
   - plot_eigenspectrum_comparison() - Figure 3 equivalent
   - plot_eigenvectors_grid() - Figure 4 equivalent
   - plot_accuracy_vs_effective_rank()
   - plot_ablation_summary()
3. Create notebooks/01_reproduction.ipynb with full analysis

Key requirements:
- Use matplotlib for all plots (not plotly - needs to work on cluster)
- Use jaxtyping for tensor type hints
- Figures should be publication-quality (PDF export, proper labels, legends)
- Use existing spectral.py functions (don't reimplement)
- Checkpoints have pre-computed eigenvalues - no need to call model.decompose()

Start by creating docs/paper_baselines.md, then implement visualization.py. Show me each file and ask for confirmation before moving to the next.
```

**PROMPT END**

---

## 9. Expected Outputs

After Phase 1 completion:

```
results/phase1/figures/
├── eigenspectrum_comparison.pdf     # Main result: reg vs no-reg
├── eigenspectrum_per_class.pdf      # Per-digit breakdown
├── eigenvectors_noreg.pdf           # Overfitting patterns
├── eigenvectors_reg.pdf             # Interpretable patterns
├── eigenvectors_comparison.pdf      # Side-by-side
├── accuracy_vs_effrank.pdf          # Trade-off plot
└── ablation_summary.pdf             # Noise vs WD contribution

notebooks/01_reproduction.ipynb      # Complete analysis notebook
docs/paper_baselines.md              # Expected values
docs/phase1_results.md               # Our findings summary
```
