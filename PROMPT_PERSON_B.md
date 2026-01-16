# Claude Code Prompt: Person B (Analysis Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 1 (Reproduction)** of a FACT-AI course project as **Person B (Analysis Lead)**.

### Project Summary
We are reproducing "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417) for a UvA MSc AI course. Person A (Infrastructure Lead) has **COMPLETED** the training infrastructure and saves checkpoints with pre-computed eigenvalues/eigenvectors. You handle all analysis, visualization, and figure generation.

### Infrastructure Status: COMPLETE
Person A has delivered:
- Training script (`src/train.py`) - verified working
- Bilinear layer (`src/models/bilinear_layer.py`) - BilinearDense wrapper
- Spectral utilities (`src/vision/spectral.py`) - effective_rank, top_k_coverage
- Config files (`configs/mnist_dense_*.yaml`) - 4 ablation configurations
- SLURM jobs (`jobs/train_array.job`) - array job for 20 runs

### Your Role
- Deep understanding of paper's eigendecomposition theory
- Extend spectral analysis utilities if needed
- Generate publication-quality figures (matplotlib, NOT plotly)
- Create the analysis notebook with statistical comparisons
- Verify reproduction matches paper claims

### Your Tasks
1. Create `src/vision/visualization.py` with plotting functions
2. Create `notebooks/01_reproduction.ipynb` for analysis
3. Generate all figures once checkpoints are ready
4. Write statistical comparison with paper baselines

---

## CRITICAL REQUIREMENTS

1. **MATPLOTLIB ONLY**: Use matplotlib for all plots. Plotly doesn't work well on the cluster and figures need PDF export.

2. **JAXTYPING**: Use jaxtyping for all tensor type hints.

3. **PUBLICATION QUALITY**: Figures must be suitable for the TMLR report (PDF export, proper labels, legends, font sizes).

4. **STATISTICAL RIGOR**: Report mean +/- std across 5 seeds, include t-tests for key comparisons.

---

## CHECKPOINT FORMAT (FROM PERSON A)

Person A's checkpoints contain pre-computed eigenvalues/eigenvectors:

```python
checkpoint = {
    'config': {
        'mode': 'dense',
        'd_hidden': 256,
        'd_input': 784,
        'n_classes': 10,
        'noise_std': 0.4,  # or 0.0 for none/wd configs
        'weight_decay': 0.5,  # or 0.0 for none/noise configs
    },
    'model_state_dict': model.state_dict(),
    'metrics': {
        'train_acc': 0.98,
        'val_acc': 0.95,
        'effective_rank': 45.2,  # mean across classes
    },
    'seed': 42,
    'eigenvalues': tensor([10, 256]),   # [n_classes, d_hidden]
    'eigenvectors': tensor([10, 256, 784]),  # [n_classes, d_hidden, d_input]
}
```

### Loading Checkpoints

```python
from src.vision.spectral import load_checkpoint_eigenvalues

# Load eigenvalues/eigenvectors directly
eigenvalues, eigenvectors = load_checkpoint_eigenvalues(
    'results/phase1/checkpoints/mnist_dense_full_seed42.pt'
)
print(f"Eigenvalues shape: {eigenvalues.shape}")  # [10, 256]
print(f"Eigenvectors shape: {eigenvectors.shape}")  # [10, 256, 784]
```

### Using Existing Spectral Functions

```python
from src.vision.spectral import effective_rank, top_k_coverage, spectral_summary

# Compute effective rank per class
eff_rank = effective_rank(eigenvalues)  # [10]
print(f"Mean effective rank: {eff_rank.mean():.2f}")

# Top-k coverage
top5 = top_k_coverage(eigenvalues, k=5)  # [10]
top10 = top_k_coverage(eigenvalues, k=10)  # [10]

# Full summary across all classes
summary = spectral_summary(eigenvalues)
# Returns dict with mean/std of effective_rank, top5_coverage, top10_coverage, decay_rate
```

### Analysis Protocol

1. **Test with one checkpoint first** - Load mnist_dense_full_seed42.pt and verify shapes
2. **Check shapes explicitly** - Print tensor shapes at each step
3. **Save intermediate results** - CSV files for reproducibility

---

## PAPER THEORY: EIGENDECOMPOSITION OF BILINEAR LAYERS

### The Core Insight

A bilinear layer computes:
```
y = (W_l @ x) * (W_r @ x)
```

This can be expressed as a **third-order interaction tensor**:
```
y_c = sum_{i,j} B[c,i,j] * x_i * x_j
```
Where `B[c,i,j] = sum_h W_head[c,h] * W_l[h,i] * W_r[h,j]`

For each output class `c`, the matrix `B[c,:,:]` captures how pairs of input pixels interact to produce that class.

### Symmetrization (Crucial!)

The tensor `B` is generally asymmetric due to the arbitrary factorization into W_l and W_r. We symmetrize:
```python
B_sym = 0.5 * (B + B.transpose(-1, -2))
```

This removes the arbitrary asymmetry and allows meaningful eigendecomposition.

### Eigendecomposition

For each class c, decompose the symmetric matrix:
```
B_sym[c] = V[c] @ diag(lambda[c]) @ V[c].T
```

Where:
- `lambda[c]`: eigenvalues (sorted by magnitude)
- `V[c]`: eigenvectors (columns are 784-dimensional for MNIST)

### Interpretation

| Eigenvalue | Meaning |
|------------|---------|
| Large positive | Input patterns that INCREASE class probability |
| Large negative | Input patterns that DECREASE class probability |
| Near zero | Irrelevant input directions |

**KEY PAPER CLAIM**: With regularization, only ~10 eigenvalues per class are significant, and the corresponding eigenvectors look like digit templates.

### Effective Rank

Measures "intrinsic dimensionality" via entropy of normalized eigenvalues:
```
p_i = |lambda_i| / sum(|lambda|)
H = -sum(p_i * log(p_i))
EffRank = exp(H)
```

- **Low effective rank** (~20-40): Concentrated spectrum, interpretable
- **High effective rank** (~150-200): Flat spectrum, overfitting

---

## DIRECTORY STRUCTURE

```
UvA_FACT_2025/
├── src/
│   ├── models/
│   │   └── bilinear_layer.py    # [PERSON A] BilinearDense, BilinearCP
│   ├── analysis/
│   │   ├── __init__.py          # [PERSON A] Created
│   │   ├── spectral.py          # [PERSON A] effective_rank, top_k_coverage
│   │   └── visualization.py     # [PERSON B] YOUR CODE - plotting functions
│   └── train.py                 # [PERSON A] Training script
├── configs/
│   ├── mnist_dense_none.yaml    # [PERSON A] No regularization
│   ├── mnist_dense_noise.yaml   # [PERSON A] Noise only
│   ├── mnist_dense_wd.yaml      # [PERSON A] Weight decay only
│   └── mnist_dense_full.yaml    # [PERSON A] Full regularization
├── notebooks/
│   └── 01_reproduction.ipynb    # [PERSON B] YOUR CODE - analysis notebook
├── results/
│   └── phase1/
│       ├── checkpoints/         # [PERSON A] Trained model checkpoints
│       └── figures/             # [PERSON B] YOUR OUTPUT - generated figures
├── Report/
│   └── figures/                 # [PERSON B] Copy final figures here
└── docs/
    └── paper_baselines.md       # [PERSON B] Expected values from paper
```

---

## FILE SPECIFICATIONS

### 1. `src/vision/spectral.py` [ALREADY IMPLEMENTED BY PERSON A]

The following functions are already available:

```python
from src.vision.spectral import (
    effective_rank,           # Entropy-based effective rank
    top_k_coverage,           # Fraction of variance in top-k eigenvalues
    eigenvalue_decay_rate,    # Ratio of 2nd to 1st eigenvalue
    spectral_summary,         # Full summary dict (mean/std of all metrics)
    load_checkpoint_eigenvalues,  # Load eigenvalues/eigenvectors from checkpoint
)

# Example usage:
eigenvalues, eigenvectors = load_checkpoint_eigenvalues('checkpoint.pt')
eff_rank = effective_rank(eigenvalues)  # [n_classes]
summary = spectral_summary(eigenvalues)  # dict with all metrics
```

### OPTIONAL: Additional Functions for Person B

If you need additional analysis functions, add them to spectral.py:

```python
def compare_configurations(
    checkpoint_dir: str,
    configs: list = ['none', 'noise', 'wd', 'full'],
    seeds: list = [42, 43, 44, 45, 46]
) -> pd.DataFrame:
    """Compare metrics across all configurations and seeds."""
    from pathlib import Path
    import pandas as pd

    results = []
    for config in configs:
        for seed in seeds:
            path = Path(checkpoint_dir) / f"mnist_dense_{config}_seed{seed}.pt"
            if not path.exists():
                print(f"Warning: {path} not found")
                continue

            vals, _ = load_checkpoint_eigenvalues(str(path))
            checkpoint = torch.load(path, map_location='cpu')

            results.append({
                'config': config,
                'seed': seed,
                'accuracy': checkpoint['metrics']['val_acc'],
                **spectral_summary(vals),
            })

    return pd.DataFrame(results)
```

---

### 2. `src/vision/visualization.py`

```python
"""
Visualization Utilities for Bilinear MLP Interpretability

Functions for reproducing paper figures:
- Figure 3: Eigenspectrum plots
- Figure 4: Eigenvector grids
- Custom: Accuracy vs Effective Rank trade-off
"""

import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_eigenspectrum_comparison(
    eigenvalues_dict: Dict[str, Float[Tensor, "n_classes n"]],
    title: str = "Eigenspectrum Comparison",
    figsize: Tuple[float, float] = (10, 6),
    log_scale: bool = True,
    top_k: int = 100,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra for multiple models (Figure 3 equivalent).

    Shows mean across classes with std shading.

    Args:
        eigenvalues_dict: Dict mapping model name to eigenvalues [n_classes, n]
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis
        top_k: Only plot top-k eigenvalues (by magnitude)
        save_path: If provided, save figure to this path (should be .pdf)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors
    for i, (name, eigenvalues) in enumerate(eigenvalues_dict.items()):
        # Sort by magnitude (descending) for each class
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)

        # Take top k
        sorted_vals = sorted_vals[:, :top_k]

        # Compute mean and std across classes
        mean_vals = sorted_vals.mean(dim=0).numpy()
        std_vals = sorted_vals.std(dim=0).numpy()

        x = np.arange(1, len(mean_vals) + 1)

        # Plot mean line
        ax.plot(x, mean_vals, label=name, color=colors[i], linewidth=2)

        # Plot std shading
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.2, color=colors[i])

    ax.set_xlabel("Eigenvalue Index (sorted by magnitude)")
    ax.set_ylabel("Eigenvalue Magnitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_eigenspectrum_per_class(
    eigenvalues: Float[Tensor, "n_classes n"],
    class_names: Optional[List[str]] = None,
    title: str = "Eigenspectrum by Class",
    figsize: Tuple[float, float] = (12, 8),
    top_k: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenspectrum for each class separately (subplots).

    Args:
        eigenvalues: [n_classes, n_eigenvalues] tensor
        class_names: Names for each class (default: digits 0-9)
        title: Overall title
        figsize: Figure size
        top_k: Number of top eigenvalues to show
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    n_classes = eigenvalues.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Sort eigenvalues by magnitude
    abs_vals = eigenvalues.abs()
    sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
    sorted_vals = sorted_vals[:, :top_k].numpy()

    # Create subplot grid
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    x = np.arange(1, top_k + 1)
    for i in range(n_classes):
        axes[i].plot(x, sorted_vals[i], linewidth=1.5)
        axes[i].set_title(f"Class {class_names[i]}")
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.supxlabel("Eigenvalue Index")
    fig.supylabel("Magnitude")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_eigenvectors_grid(
    eigenvectors: Float[Tensor, "n_classes n_components d_input"],
    eigenvalues: Float[Tensor, "n_classes n_components"],
    n_top: int = 5,
    classes: Optional[List[int]] = None,
    img_shape: Tuple[int, int] = (28, 28),
    title: str = "Top Eigenvectors",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot grid of top eigenvectors (Figure 4 equivalent).

    Shows top eigenvectors (by absolute eigenvalue magnitude) for each class.
    Eigenvectors are reshaped to image dimensions.

    Args:
        eigenvectors: [n_classes, n_components, d_input] - eigenvector matrix
        eigenvalues: [n_classes, n_components] - for sorting by magnitude
        n_top: Number of top eigenvectors to show per class
        classes: Which classes to plot (default: all)
        img_shape: Shape to reshape eigenvectors (28, 28 for MNIST)
        title: Plot title
        save_path: If provided, save figure

    Returns:
        matplotlib Figure with grid of eigenvector images

    Layout:
        Rows: Classes (0-9 or selected)
        Columns: Top eigenvectors (sorted by |eigenvalue|)
    """
    n_classes_total = eigenvectors.shape[0]
    if classes is None:
        classes = list(range(n_classes_total))

    n_rows = len(classes)
    n_cols = n_top

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, cls in enumerate(classes):
        # Sort eigenvectors by eigenvalue magnitude
        abs_vals = eigenvalues[cls].abs()
        sorted_indices = abs_vals.argsort(descending=True)

        for col in range(n_top):
            idx = sorted_indices[col]
            vec = eigenvectors[cls, idx].numpy()
            val = eigenvalues[cls, idx].item()

            # Reshape to image
            img = vec.reshape(img_shape)

            # Normalize for visualization
            vmax = np.abs(img).max()
            axes[row, col].imshow(img, cmap='RdBu', vmin=-vmax, vmax=vmax)
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(f"#{col+1}", fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f"Class {cls}", fontsize=10, rotation=0,
                                          labelpad=25, va='center')

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_accuracy_vs_effective_rank(
    results_df: "pd.DataFrame",
    title: str = "Accuracy vs Interpretability Trade-off",
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of accuracy vs effective rank for all experiments.

    This visualizes the trade-off between performance and interpretability.

    Args:
        results_df: DataFrame with columns:
                    - 'config': experiment config name
                    - 'accuracy': test accuracy
                    - 'effective_rank': mean effective rank
                    (can have multiple rows per config for different seeds)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure with scatter plot
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=figsize)

    # Group by config
    configs = results_df['config'].unique()
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, config in enumerate(configs):
        subset = results_df[results_df['config'] == config]

        # Plot individual seeds as small points
        ax.scatter(subset['effective_rank'], subset['accuracy'] * 100,
                   c=[colors[i]], marker=markers[i % len(markers)],
                   s=50, alpha=0.5, label=None)

        # Plot mean as larger point with error bars
        mean_rank = subset['effective_rank'].mean()
        std_rank = subset['effective_rank'].std()
        mean_acc = subset['accuracy'].mean() * 100
        std_acc = subset['accuracy'].std() * 100

        ax.errorbar(mean_rank, mean_acc, xerr=std_rank, yerr=std_acc,
                    fmt=markers[i % len(markers)], color=colors[i],
                    markersize=12, capsize=5, capthick=2, linewidth=2,
                    label=config)

    ax.set_xlabel("Effective Rank (lower = more interpretable)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(title="Configuration")
    ax.grid(True, alpha=0.3)

    # Annotate trade-off direction
    ax.annotate("Better\nInterpretability", xy=(0.15, 0.15), xycoords='axes fraction',
                fontsize=10, ha='center', style='italic', color='gray')
    ax.annotate("", xy=(0.05, 0.05), xycoords='axes fraction',
                xytext=(0.25, 0.25), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray'))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_ablation_summary(
    aggregated_df: "pd.DataFrame",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot summarizing ablation results.

    Shows accuracy and effective rank for each configuration.

    Args:
        aggregated_df: DataFrame from aggregate_by_config() with columns:
                       config, accuracy_mean, accuracy_std, effective_rank_mean, etc.
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = aggregated_df['config']
    x = np.arange(len(configs))
    width = 0.6

    # Accuracy plot
    ax1.bar(x, aggregated_df['accuracy_mean'] * 100, width,
            yerr=aggregated_df['accuracy_std'] * 100, capsize=5,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy by Configuration")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Effective rank plot
    ax2.bar(x, aggregated_df['effective_rank_mean'], width,
            yerr=aggregated_df['effective_rank_std'], capsize=5,
            color='coral', alpha=0.8)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Effective Rank")
    ax2.set_title("Effective Rank by Configuration (lower = better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig
```

---

### 3. `notebooks/01_reproduction.ipynb` Structure

Create a Jupyter notebook with this structure:

```markdown
# Phase 1: Reproducing Bilinear MLP Vision Experiments

## 1. Setup

```python
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.vision.spectral import (
    extract_from_checkpoint,
    compute_all_metrics,
    compare_configurations,
    aggregate_by_config,
    effective_rank,
    top_k_coverage,
)
from src.vision.visualization import (
    plot_eigenspectrum_comparison,
    plot_eigenspectrum_per_class,
    plot_eigenvectors_grid,
    plot_accuracy_vs_effective_rank,
    plot_ablation_summary,
)

CHECKPOINT_DIR = "../results/phase1/checkpoints"
FIGURE_DIR = "../results/phase1/figures"
REPORT_FIGURE_DIR = "../Report/figures"

Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORT_FIGURE_DIR).mkdir(parents=True, exist_ok=True)
```

## 2. Load All Checkpoints

```python
# Compare all configurations
results_df = compare_configurations(CHECKPOINT_DIR)
print(f"Loaded {len(results_df)} experiments")
results_df.head(10)
```

## 3. Eigenspectrum Analysis

### 3.1 Figure 3: Eigenspectrum Comparison (Main Result)

```python
# Load representative checkpoints for comparison
noreg_data = extract_from_checkpoint(f"{CHECKPOINT_DIR}/mnist_dense_none_seed42.pt")
full_data = extract_from_checkpoint(f"{CHECKPOINT_DIR}/mnist_dense_full_seed42.pt")

eigenvalues_dict = {
    "No Regularization": noreg_data['eigenvalues'],
    "Full Regularization (noise + WD)": full_data['eigenvalues'],
}

fig = plot_eigenspectrum_comparison(
    eigenvalues_dict,
    title="Eigenspectrum: Regularization Induces Low-Rank Structure",
    save_path=f"{FIGURE_DIR}/eigenspectrum_comparison.pdf"
)

# Also save to Report/figures
fig.savefig(f"{REPORT_FIGURE_DIR}/eigenspectrum_comparison.pdf")
plt.show()
```

### 3.2 Per-Class Eigenspectrum

```python
fig = plot_eigenspectrum_per_class(
    full_data['eigenvalues'],
    title="Eigenspectrum by Digit Class (with regularization)",
    save_path=f"{FIGURE_DIR}/eigenspectrum_per_class.pdf"
)
plt.show()
```

## 4. Eigenvector Visualization

### 4.1 Figure 4: Without Regularization

```python
fig = plot_eigenvectors_grid(
    noreg_data['eigenvectors'],
    noreg_data['eigenvalues'],
    n_top=5,
    title="Top Eigenvectors: No Regularization (Overfitting Patterns)",
    save_path=f"{FIGURE_DIR}/eigenvectors_noreg.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/eigenvectors_noreg.pdf")
plt.show()
```

### 4.2 Figure 4: With Regularization

```python
fig = plot_eigenvectors_grid(
    full_data['eigenvectors'],
    full_data['eigenvalues'],
    n_top=5,
    title="Top Eigenvectors: Full Regularization (Interpretable Patterns)",
    save_path=f"{FIGURE_DIR}/eigenvectors_reg.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/eigenvectors_reg.pdf")
plt.show()
```

## 5. Ablation Study

### 5.1 Aggregate Results

```python
aggregated = aggregate_by_config(results_df)
print(aggregated.to_markdown(index=False))
```

### 5.2 Ablation Bar Plot

```python
fig = plot_ablation_summary(
    aggregated,
    save_path=f"{FIGURE_DIR}/ablation_summary.pdf"
)
plt.show()
```

## 6. Accuracy vs Interpretability Trade-off

```python
fig = plot_accuracy_vs_effective_rank(
    results_df,
    title="Accuracy vs Interpretability Trade-off",
    save_path=f"{FIGURE_DIR}/accuracy_vs_effrank.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/accuracy_vs_effrank.pdf")
plt.show()
```

## 7. Statistical Analysis

### 7.1 t-test: Effective Rank Comparison

```python
from scipy import stats

noreg_ranks = results_df[results_df['config'] == 'none']['effective_rank']
full_ranks = results_df[results_df['config'] == 'full']['effective_rank']

t_stat, p_value = stats.ttest_ind(noreg_ranks, full_ranks)
cohens_d = (noreg_ranks.mean() - full_ranks.mean()) / np.sqrt(
    (noreg_ranks.var() + full_ranks.var()) / 2
)

print(f"No Reg Effective Rank: {noreg_ranks.mean():.2f} +/- {noreg_ranks.std():.2f}")
print(f"Full Reg Effective Rank: {full_ranks.mean():.2f} +/- {full_ranks.std():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Ratio (full/none): {full_ranks.mean() / noreg_ranks.mean():.4f}")
```

## 8. Comparison with Paper

### 8.1 Metrics Table

```python
paper_values = {
    'Configuration': ['No regularization', 'Full regularization'],
    'Paper Accuracy': ['~97-98%', '~94-95%'],
    'Ours Accuracy': [
        f"{results_df[results_df['config']=='none']['accuracy'].mean()*100:.1f}%",
        f"{results_df[results_df['config']=='full']['accuracy'].mean()*100:.1f}%",
    ],
    'Paper Eff. Rank': ['~150-200', '~20-40'],
    'Ours Eff. Rank': [
        f"{results_df[results_df['config']=='none']['effective_rank'].mean():.1f}",
        f"{results_df[results_df['config']=='full']['effective_rank'].mean():.1f}",
    ],
}
comparison_df = pd.DataFrame(paper_values)
print(comparison_df.to_markdown(index=False))
```

## 9. Gate Check Verification

```python
ratio = full_ranks.mean() / noreg_ranks.mean()
print("GATE CHECK CRITERIA:")
print(f"  [{'X' if ratio < 0.5 else ' '}] Effective rank ratio < 0.5: {ratio:.4f}")
print(f"  [ ] Eigenvectors resemble digits: (visual inspection above)")
print(f"  [{'X' if 0.93 < results_df[results_df['config']=='full']['accuracy'].mean() < 0.97 else ' '}] Accuracy within 2% of paper")
```

## 10. Conclusions

- Summarize findings
- Note any discrepancies with paper
- Ready for Phase 2?
```

---

### 4. `docs/paper_baselines.md`

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

## EXECUTION CHECKLIST

### Step 1: Verify Infrastructure (Person A Complete)
```bash
# Verify directories exist
ls src/vision/spectral.py  # Should exist
ls configs/mnist_dense_*.yaml  # 4 config files
ls jobs/train_array.job  # SLURM array job
```

### Step 2: Create Your Directories
```bash
mkdir -p notebooks results/phase1/figures docs
```

### Step 3: Test Loading Checkpoints
```python
# Test with existing checkpoint (after Person A runs training)
import torch
from src.vision.spectral import load_checkpoint_eigenvalues, spectral_summary

vals, vecs = load_checkpoint_eigenvalues("results/phase1/checkpoints/mnist_dense_full_seed42.pt")
print(f"Eigenvalues shape: {vals.shape}")  # [10, 256]
print(f"Eigenvectors shape: {vecs.shape}")  # [10, 256, 784]
print(spectral_summary(vals))
```

### Step 4: Create visualization.py
Implement plotting functions (see specification below).

### Step 5: Create Analysis Notebook
Create `notebooks/01_reproduction.ipynb` with full analysis.

### Step 6: Generate All Figures
Run notebook once all 20 checkpoints exist.

### Step 7: Copy to Report
```bash
cp results/phase1/figures/*.pdf Report/figures/
```

---

## EXPECTED OUTPUTS

```
results/phase1/figures/
├── eigenspectrum_comparison.pdf   # Main result: reg vs no-reg
├── eigenspectrum_per_class.pdf    # Per-digit breakdown
├── eigenvectors_noreg.pdf         # Overfitting patterns
├── eigenvectors_reg.pdf           # Interpretable patterns
├── accuracy_vs_effrank.pdf        # Trade-off plot
└── ablation_summary.pdf           # Bar chart comparison

Report/figures/
├── eigenspectrum_comparison.pdf   # Copied for report
├── eigenvectors_noreg.pdf
├── eigenvectors_reg.pdf
└── accuracy_vs_effrank.pdf

notebooks/01_reproduction.ipynb    # Complete analysis
docs/paper_baselines.md            # Reference values
```

---

**Now begin by creating `docs/paper_baselines.md`, then implement `src/vision/spectral.py`. Show me each file and ask for confirmation before proceeding.**

## PROMPT END
