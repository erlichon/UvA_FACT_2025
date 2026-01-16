# Claude Code Prompt: Person E (Synthesis + ViT Formulation Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 2 Extensions 4 + 5** of a FACT-AI course project as **Person E (Synthesis + ViT Formulation Lead)**.

### Project Summary
We are reproducing and extending "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417). All experiments are complete. You are now working on:
- **Extension 4**: Structure vs. Regularization Synthesis - The final comparison
- **Extension 5**: ViT Formulation - Conceptual extension (no code)

### Your Role
- Aggregate ALL results from Phase 1 and Phase 2
- Generate publication-quality comparison figures
- Perform statistical significance tests
- Write the ViT formulation section (conceptual, no implementation)
- Create the final combined notebook `03_all_results.ipynb`
- Ensure all figures are copied to `Report/figures/`

### Timeline
- **Days 14-16** (Jan 21-23): Extension 4 - Synthesis and comparison
- **Day 16-17** (Jan 23-24): Extension 5 - ViT formulation
- **Day 17** (Jan 24): **RESULT FREEZE** - No new experiments after this

### Dependencies
- **Phase 1** (Person A+B): `results/phase1/checkpoints/` - Dense model checkpoints
- **Person C**: `results/phase2/robustness/` - Robustness analysis
- **Person D**: `results/phase2/checkpoints/` - CP model checkpoints

---

## VERIFICATION & BUDGET REQUIREMENTS

> **IMPORTANT**: The team has a total budget of **25,000 SBUs** on Snellius.

### Local Verification (MANDATORY)

Person E does **NO training** - only analysis. But you must verify analysis code works:

```python
# Test with mock data before using real checkpoints
import torch
import pandas as pd

# Create mock results DataFrame
mock_results = pd.DataFrame({
    'mode': ['dense', 'dense', 'cp', 'cp'],
    'config': ['none', 'full', 'r32', 'r64'],
    'rank': [None, None, 32, 64],
    'seed': [42, 42, 42, 42],
    'accuracy': [0.97, 0.94, 0.93, 0.95],
    'effective_rank': [180, 35, 30, 55],
})

# Test synthesis functions
from src.synthesize_results import aggregate_results, statistical_comparison
agg = aggregate_results(mock_results)
print(agg)
```

**Verification Checklist**:
- [ ] `load_all_results()` finds all checkpoints
- [ ] `aggregate_results()` produces correct format
- [ ] `statistical_comparison()` runs without errors
- [ ] All visualization functions produce valid figures
- [ ] PDF export works correctly

### Budget Summary

| Phase | Person | Est. SBUs | Running Total |
|-------|--------|-----------|---------------|
| P1 | A+B | 260 | 260 |
| E1 | C | 20 | 280 |
| E2 | F | 0 (local) | 280 |
| E3 | D | 360 | 640 |
| E4+E5 | E | 0 (analysis) | 640 |
| Buffer | All | 360 | 1000 |

*Total estimated: ~1,000 SBUs (well under 25,000 limit)*

---

## CRITICAL DELIVERABLES

### Figures for Report
All figures must be saved to BOTH:
- `results/phase2/figures/` (for reference)
- `Report/figures/` (for LaTeX inclusion)

| Figure | Description | Section |
|--------|-------------|---------|
| `structure_vs_reg.pdf` | Main comparison: Pareto frontier | 05_extension.tex |
| `pareto_frontier.pdf` | Accuracy vs Effective Rank scatter | 05_extension.tex |
| `eigenspectrum_overlay.pdf` | Dense(reg) vs CP eigenspectra | 05_extension.tex |
| `eigenvector_comparison.pdf` | Side-by-side eigenvector grids | 05_extension.tex |

### Final Notebook
`notebooks/03_all_results.ipynb` - Required deliverable containing ALL results from both phases.

### ViT Formulation
Write mathematical formulation for Section 5 of report (conceptual only, NO code implementation).

---

## EXTENSION 4: STRUCTURE VS. REGULARIZATION SYNTHESIS

### 4.1 Core Question

> Can **structural low-rank** (CP-Decomposition) achieve the same interpretability as **emergent low-rank** (regularization) WITHOUT requiring noise augmentation?

### 4.2 Data to Aggregate

**From Phase 1 (Dense Models)**:
```
results/phase1/checkpoints/
├── mnist_dense_none_seed{42-46}.pt    # Baseline (no reg)
├── mnist_dense_noise_seed{42-46}.pt   # Noise only
├── mnist_dense_wd_seed{42-46}.pt      # Weight decay only
└── mnist_dense_full_seed{42-46}.pt    # Full regularization
```

**From Person C (Robustness)**:
```
results/phase2/robustness/
├── rotated_mnist_results.csv
└── emnist_analysis.json
```

**From Person D (CP Sweep)**:
```
results/phase2/checkpoints/
├── mnist_cp_r8_seed{42-46}.pt
├── mnist_cp_r16_seed{42-46}.pt
├── mnist_cp_r32_seed{42-46}.pt
├── mnist_cp_r64_seed{42-46}.pt
├── mnist_cp_r128_seed{42-46}.pt
└── mnist_cp_r256_seed{42-46}.pt
```

### 4.3 Key Comparisons

| Comparison | Question | Figure |
|------------|----------|--------|
| Dense(none) vs Dense(full) | Does regularization improve interpretability? | Phase 1 |
| Dense(full) vs CP(R=32) | Can structure match regularization? | Pareto frontier |
| CP eigenspectrum vs Dense eigenspectrum | Are the spectra similar? | Overlay plot |
| CP eigenvectors vs Dense eigenvectors | Are features similar? | Side-by-side grid |

### 4.4 Statistical Tests

For key comparisons, perform:
1. **t-test**: Compare means across seeds
2. **Effect size** (Cohen's d): Magnitude of difference
3. **95% CI**: Confidence interval for mean difference

---

## FILE SPECIFICATIONS

### 1. Main Synthesis Script

**`src/synthesize_results.py`**:
```python
"""
Synthesize all experimental results for final analysis.

Aggregates Phase 1 and Phase 2 results into unified DataFrames.
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "bilinear-decomposition-main"))

from src.vision.spectral import extract_from_checkpoint, compute_all_metrics


def load_all_phase1_results(checkpoint_dir: str = "results/phase1/checkpoints") -> pd.DataFrame:
    """Load all Phase 1 (dense model) results."""
    checkpoint_dir = Path(checkpoint_dir)
    configs = ['none', 'noise', 'wd', 'full']
    seeds = [42, 43, 44, 45, 46]

    results = []
    for config in configs:
        for seed in seeds:
            path = checkpoint_dir / f"mnist_dense_{config}_seed{seed}.pt"
            if not path.exists():
                print(f"Warning: {path} not found")
                continue

            data = extract_from_checkpoint(str(path))
            metrics = compute_all_metrics(data['eigenvalues'])

            results.append({
                'phase': 'P1',
                'mode': 'dense',
                'config': config,
                'rank': None,
                'noise_std': 0.4 if config in ['noise', 'full'] else 0.0,
                'weight_decay': 0.5 if config in ['wd', 'full'] else 0.0,
                'seed': seed,
                'accuracy': data['metrics']['val_acc'],
                'effective_rank': metrics['effective_rank_mean'],
                'top5_coverage': metrics['top5_coverage_mean'],
                'top10_coverage': metrics['top10_coverage_mean'],
            })

    return pd.DataFrame(results)


def load_all_phase2_cp_results(checkpoint_dir: str = "results/phase2/checkpoints") -> pd.DataFrame:
    """Load all Phase 2 (CP model) results."""
    checkpoint_dir = Path(checkpoint_dir)
    ranks = [8, 16, 32, 64, 128, 256]
    seeds = [42, 43, 44, 45, 46]

    results = []
    for rank in ranks:
        for seed in seeds:
            path = checkpoint_dir / f"mnist_cp_r{rank}_seed{seed}.pt"
            if not path.exists():
                print(f"Warning: {path} not found")
                continue

            data = extract_from_checkpoint(str(path))

            results.append({
                'phase': 'P2',
                'mode': 'cp',
                'config': f'r{rank}',
                'rank': rank,
                'noise_std': 0.0,
                'weight_decay': 0.1,
                'seed': seed,
                'accuracy': data['metrics']['val_acc'],
                'effective_rank': data['metrics'].get('effective_rank',
                    compute_all_metrics(data['eigenvalues'])['effective_rank_mean']),
                'top5_coverage': data['metrics'].get('top5_coverage', 0),
                'top10_coverage': data['metrics'].get('top10_coverage', 0),
            })

    return pd.DataFrame(results)


def load_all_results() -> pd.DataFrame:
    """Load and combine all results."""
    p1_df = load_all_phase1_results()
    p2_df = load_all_phase2_cp_results()
    return pd.concat([p1_df, p2_df], ignore_index=True)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by configuration (mean +/- std across seeds)."""
    agg = df.groupby(['mode', 'config', 'rank']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'effective_rank': ['mean', 'std'],
        'top5_coverage': ['mean', 'std'],
    }).round(4)

    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def statistical_comparison(
    group1: pd.Series,
    group2: pd.Series,
    name1: str = "Group 1",
    name2: str = "Group 2"
) -> Dict:
    """
    Perform statistical comparison between two groups.

    Returns dict with t-test results, effect size, and confidence interval.
    """
    from scipy import stats

    # t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
    cohens_d = (group1.mean() - group2.mean()) / pooled_std

    # 95% CI for mean difference
    diff = group1.mean() - group2.mean()
    se_diff = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
    ci_95 = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    return {
        'comparison': f"{name1} vs {name2}",
        'mean_1': group1.mean(),
        'mean_2': group2.mean(),
        'diff': diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_95_low': ci_95[0],
        'ci_95_high': ci_95[1],
        'significant': p_value < 0.05,
    }


if __name__ == "__main__":
    # Load and display all results
    all_results = load_all_results()
    print(f"Total experiments: {len(all_results)}")
    print("\nAggregated results:")
    print(aggregate_results(all_results).to_markdown())
```

### 2. Final Visualization Functions

**Add to `src/vision/visualization.py`**:
```python
def plot_pareto_frontier(
    results_df: pd.DataFrame,
    highlight_models: List[str] = None,
    title: str = "Accuracy vs Interpretability: Pareto Frontier",
    figsize: Tuple[float, float] = (12, 8),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot accuracy vs effective rank with Pareto frontier highlighted.

    The Pareto frontier shows models that are not dominated by any other model
    (i.e., no model is both more accurate AND more interpretable).

    Args:
        results_df: DataFrame with columns: mode, config, accuracy, effective_rank
        highlight_models: List of (mode, config) tuples to highlight
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by mode
    mode_colors = {'dense': 'tab:blue', 'cp': 'tab:orange'}
    mode_markers = {'dense': 'o', 'cp': 's'}

    # Group and plot
    for (mode, config), group in results_df.groupby(['mode', 'config']):
        color = mode_colors.get(mode, 'gray')
        marker = mode_markers.get(mode, 'o')

        # Individual points (small, transparent)
        ax.scatter(group['effective_rank'], group['accuracy'] * 100,
                   c=color, marker=marker, s=30, alpha=0.3)

        # Mean point (large)
        mean_eff = group['effective_rank'].mean()
        mean_acc = group['accuracy'].mean() * 100
        std_eff = group['effective_rank'].std()
        std_acc = group['accuracy'].std() * 100

        ax.errorbar(mean_eff, mean_acc, xerr=std_eff, yerr=std_acc,
                    fmt=marker, color=color, markersize=12, capsize=5,
                    label=f"{mode} ({config})")

    # Compute and plot Pareto frontier
    agg = results_df.groupby(['mode', 'config']).agg({
        'accuracy': 'mean',
        'effective_rank': 'mean'
    }).reset_index()

    # Pareto: lower effective rank AND higher accuracy is better
    pareto_points = []
    for _, row in agg.iterrows():
        dominated = False
        for _, other in agg.iterrows():
            # other dominates row if: other.acc >= row.acc AND other.eff <= row.eff
            # with at least one strict inequality
            if (other['accuracy'] >= row['accuracy'] and
                other['effective_rank'] <= row['effective_rank'] and
                (other['accuracy'] > row['accuracy'] or
                 other['effective_rank'] < row['effective_rank'])):
                dominated = True
                break
        if not dominated:
            pareto_points.append(row)

    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points).sort_values('effective_rank')
        ax.plot(pareto_df['effective_rank'], pareto_df['accuracy'] * 100,
                'k--', linewidth=2, alpha=0.5, label='Pareto frontier')

    ax.set_xlabel('Effective Rank (lower = more interpretable)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Better\n(high acc, low rank)', xy=(0.05, 0.95),
                xycoords='axes fraction', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_eigenspectrum_overlay(
    models_dict: Dict[str, Tuple[torch.Tensor, str]],
    title: str = "Eigenspectrum Comparison",
    top_k: int = 100,
    figsize: Tuple[float, float] = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Overlay eigenspectra from multiple models.

    Args:
        models_dict: Dict mapping label to (eigenvalues tensor, color)
        title: Plot title
        top_k: Number of eigenvalues to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, (eigenvalues, color) in models_dict.items():
        # Sort by magnitude
        abs_vals = eigenvalues.abs()
        sorted_vals, _ = abs_vals.sort(dim=-1, descending=True)
        sorted_vals = sorted_vals[:, :top_k]

        # Mean across classes
        mean_vals = sorted_vals.mean(dim=0).numpy()
        std_vals = sorted_vals.std(dim=0).numpy()

        x = np.arange(1, len(mean_vals) + 1)
        ax.plot(x, mean_vals, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.2, color=color)

    ax.set_xlabel('Eigenvalue Index (sorted by magnitude)')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_eigenvector_comparison_grid(
    models_data: Dict[str, Dict],
    n_top: int = 3,
    classes: List[int] = [0, 1, 2, 3, 4],
    figsize: Tuple[float, float] = (15, 8),
    save_path: str = None,
) -> plt.Figure:
    """
    Compare top eigenvectors across multiple models.

    Args:
        models_data: Dict mapping model name to dict with 'eigenvalues' and 'eigenvectors'
        n_top: Number of top eigenvectors per class
        classes: Which classes to show
        figsize: Figure size
        save_path: Path to save figure
    """
    n_models = len(models_data)
    n_classes = len(classes)

    fig, axes = plt.subplots(n_models, n_classes * n_top,
                             figsize=figsize)

    model_names = list(models_data.keys())

    for row, model_name in enumerate(model_names):
        data = models_data[model_name]
        eigenvalues = data['eigenvalues']
        eigenvectors = data['eigenvectors']

        col = 0
        for cls in classes:
            # Sort by eigenvalue magnitude
            abs_vals = eigenvalues[cls].abs()
            sorted_idx = abs_vals.argsort(descending=True)

            for k in range(n_top):
                idx = sorted_idx[k]
                vec = eigenvectors[cls, idx].numpy().reshape(28, 28)

                vmax = np.abs(vec).max()
                axes[row, col].imshow(vec, cmap='RdBu', vmin=-vmax, vmax=vmax)
                axes[row, col].axis('off')

                if row == 0:
                    if k == 0:
                        axes[row, col].set_title(f'Class {cls}', fontsize=10)
                if col == 0:
                    axes[row, col].set_ylabel(model_name, fontsize=10,
                                              rotation=0, labelpad=50, va='center')
                col += 1

    plt.suptitle('Top Eigenvectors: Model Comparison', y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
```

### 3. Final Notebook Structure

**`notebooks/03_all_results.ipynb`**:
```markdown
# Complete Results: Bilinear MLP Interpretability Study

## 1. Setup and Data Loading

```python
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.synthesize_results import (
    load_all_results,
    aggregate_results,
    statistical_comparison,
)
from src.vision.spectral import extract_from_checkpoint
from src.vision.visualization import (
    plot_pareto_frontier,
    plot_eigenspectrum_overlay,
    plot_eigenvector_comparison_grid,
)

# Paths
P1_CHECKPOINT_DIR = "../results/phase1/checkpoints"
P2_CHECKPOINT_DIR = "../results/phase2/checkpoints"
FIGURE_DIR = "../results/phase2/figures"
REPORT_FIGURE_DIR = "../Report/figures"

Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORT_FIGURE_DIR).mkdir(parents=True, exist_ok=True)
```

## 2. Load All Results

```python
# Load all experimental results
all_results = load_all_results()
print(f"Total experiments: {len(all_results)}")
print(f"Phase 1 (Dense): {len(all_results[all_results['phase'] == 'P1'])}")
print(f"Phase 2 (CP): {len(all_results[all_results['phase'] == 'P2'])}")
```

## 3. Aggregated Summary Table

```python
aggregated = aggregate_results(all_results)
print(aggregated.to_markdown(index=False))

# Save for report
aggregated.to_csv("../results/all_results_summary.csv", index=False)
```

## 4. Main Result: Structure vs Regularization Pareto Frontier

```python
fig = plot_pareto_frontier(
    all_results,
    title="Structure vs Regularization: Accuracy-Interpretability Trade-off",
    save_path=f"{FIGURE_DIR}/pareto_frontier.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/structure_vs_reg.pdf")
plt.show()
```

## 5. Eigenspectrum Comparison

```python
# Load representative models
dense_none = extract_from_checkpoint(f"{P1_CHECKPOINT_DIR}/mnist_dense_none_seed42.pt")
dense_full = extract_from_checkpoint(f"{P1_CHECKPOINT_DIR}/mnist_dense_full_seed42.pt")
cp_r32 = extract_from_checkpoint(f"{P2_CHECKPOINT_DIR}/mnist_cp_r32_seed42.pt")
cp_r64 = extract_from_checkpoint(f"{P2_CHECKPOINT_DIR}/mnist_cp_r64_seed42.pt")

models_dict = {
    "Dense (no reg)": (dense_none['eigenvalues'], 'red'),
    "Dense (full reg)": (dense_full['eigenvalues'], 'green'),
    "CP R=32": (cp_r32['eigenvalues'], 'blue'),
    "CP R=64": (cp_r64['eigenvalues'], 'orange'),
}

fig = plot_eigenspectrum_overlay(
    models_dict,
    title="Eigenspectrum: Dense vs CP",
    save_path=f"{FIGURE_DIR}/eigenspectrum_overlay.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/eigenspectrum_overlay.pdf")
plt.show()
```

## 6. Eigenvector Comparison

```python
models_data = {
    "Dense (no reg)": dense_none,
    "Dense (full reg)": dense_full,
    "CP R=32 (no noise)": cp_r32,
}

fig = plot_eigenvector_comparison_grid(
    models_data,
    n_top=3,
    classes=[0, 1, 2, 3, 4],
    save_path=f"{FIGURE_DIR}/eigenvector_comparison.pdf"
)
fig.savefig(f"{REPORT_FIGURE_DIR}/eigenvector_comparison.pdf")
plt.show()
```

## 7. Statistical Analysis

### 7.1 Key Comparisons

```python
# Dense (no reg) vs Dense (full reg)
dense_none_eff = all_results[(all_results['mode']=='dense') &
                              (all_results['config']=='none')]['effective_rank']
dense_full_eff = all_results[(all_results['mode']=='dense') &
                              (all_results['config']=='full')]['effective_rank']

result1 = statistical_comparison(dense_none_eff, dense_full_eff,
                                  "Dense (no reg)", "Dense (full reg)")
print("Comparison 1: Regularization reduces effective rank")
print(f"  p-value: {result1['p_value']:.6f}")
print(f"  Cohen's d: {result1['cohens_d']:.4f}")
print(f"  Significant: {result1['significant']}")
```

```python
# Dense (full reg) vs CP R=32
cp_r32_eff = all_results[(all_results['mode']=='cp') &
                          (all_results['config']=='r32')]['effective_rank']

result2 = statistical_comparison(dense_full_eff, cp_r32_eff,
                                  "Dense (full reg)", "CP R=32")
print("\nComparison 2: Structure vs Regularization")
print(f"  Dense (full reg) eff. rank: {dense_full_eff.mean():.2f} +/- {dense_full_eff.std():.2f}")
print(f"  CP R=32 eff. rank: {cp_r32_eff.mean():.2f} +/- {cp_r32_eff.std():.2f}")
print(f"  p-value: {result2['p_value']:.6f}")
print(f"  Cohen's d: {result2['cohens_d']:.4f}")
```

### 7.2 Summary of Statistical Tests

```python
# Create summary table
stats_summary = pd.DataFrame([result1, result2])
print(stats_summary.to_markdown(index=False))
```

## 8. Robustness Results (from Person C)

```python
# Load robustness results
robustness_df = pd.read_csv("../results/phase2/robustness/rotated_mnist_results.csv")
print("Rotated MNIST Results:")
print(robustness_df.to_markdown(index=False))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(robustness_df['angle'], robustness_df['accuracy'] * 100, 'o-')
ax.set_xlabel('Rotation Angle (degrees)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Robustness to Rotation')
ax.grid(True)
plt.savefig(f"{FIGURE_DIR}/robustness_rotated.pdf")
plt.savefig(f"{REPORT_FIGURE_DIR}/robustness_rotated.pdf")
```

## 9. Final Summary Table for Report

```python
# Create publication-ready summary table
summary_rows = []

# Dense baselines
for config in ['none', 'full']:
    subset = all_results[(all_results['mode']=='dense') & (all_results['config']==config)]
    summary_rows.append({
        'Model': f"Dense ({'no reg' if config=='none' else 'full reg'})",
        'Noise': '0.4' if config == 'full' else '0.0',
        'Accuracy': f"{subset['accuracy'].mean()*100:.1f} +/- {subset['accuracy'].std()*100:.1f}",
        'Eff. Rank': f"{subset['effective_rank'].mean():.1f} +/- {subset['effective_rank'].std():.1f}",
    })

# Best CP models
for rank in [32, 64]:
    subset = all_results[(all_results['mode']=='cp') & (all_results['config']==f'r{rank}')]
    summary_rows.append({
        'Model': f"CP R={rank}",
        'Noise': '0.0',
        'Accuracy': f"{subset['accuracy'].mean()*100:.1f} +/- {subset['accuracy'].std()*100:.1f}",
        'Eff. Rank': f"{subset['effective_rank'].mean():.1f} +/- {subset['effective_rank'].std():.1f}",
    })

summary_df = pd.DataFrame(summary_rows)
print("\n=== TABLE FOR REPORT ===")
print(summary_df.to_markdown(index=False))
```

## 10. Key Findings

### Finding 1: Regularization Induces Low-Rank Structure
- Confirmed: Dense (full reg) has significantly lower effective rank than Dense (no reg)
- Statistical significance: p < 0.001

### Finding 2: Structure Can Replace Regularization
- CP models achieve low effective rank WITHOUT noise augmentation
- CP R=32 achieves comparable effective rank to Dense (full reg)
- Accuracy trade-off: [fill in based on results]

### Finding 3: Eigenvector Quality
- [Comment on whether CP eigenvectors are interpretable]
- [Compare to Dense (full reg) eigenvectors]

## 11. Conclusion

[Summarize main findings supporting or refuting the hypothesis]
```

---

## EXTENSION 5: ViT FORMULATION (CONCEPTUAL)

### 5.1 Mathematical Formulation

Write this section for `Report/sections/05_extension.tex`:

```latex
\subsection{Extension to Vision Transformers}
\label{sec:vit-formulation}

While our experiments focus on single-layer bilinear MLPs for MNIST classification,
the CP-decomposed bilinear layer naturally extends to Vision Transformers.
This section presents the mathematical formulation (implementation is beyond scope).

\subsubsection{Standard Transformer MLP Block}

The MLP block in a standard Vision Transformer is:
\begin{equation}
    \text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \text{LN}(\mathbf{x}))
\end{equation}
where $\mathbf{W}_1 \in \mathbb{R}^{d_{ff} \times d}$, $\mathbf{W}_2 \in \mathbb{R}^{d \times d_{ff}}$,
$\sigma$ is an activation function (GELU), and LN is LayerNorm.

\subsubsection{Bilinear Transformer MLP}

Following \citet{pearce2024bilinear}, we can replace the nonlinearity with a bilinear operation:
\begin{equation}
    \text{BilinearFFN}(\mathbf{x}) = \mathbf{W}_p \cdot \left[ (\mathbf{W}_l \cdot \mathbf{x}) \odot (\mathbf{W}_r \cdot \mathbf{x}) \right]
\end{equation}
where $\mathbf{W}_l, \mathbf{W}_r \in \mathbb{R}^{d_{ff} \times d}$ and $\mathbf{W}_p \in \mathbb{R}^{d \times d_{ff}}$.

This enables eigendecomposition of the interaction tensor for each layer,
providing interpretable feature interactions.

\subsubsection{CP-Decomposed Bilinear FFN}

Our proposed extension applies CP-decomposition:
\begin{equation}
    \text{CP-FFN}(\mathbf{x}) = \mathbf{W}_p \cdot \left[ \sum_{r=1}^{R} \lambda_r (\mathbf{a}_r^\top \mathbf{x}) (\mathbf{b}_r^\top \mathbf{x}) \mathbf{c}_r \right]
\end{equation}
where $\mathbf{a}_r, \mathbf{b}_r \in \mathbb{R}^d$, $\mathbf{c}_r \in \mathbb{R}^{d_{ff}}$, and $\lambda_r \in \mathbb{R}$.

\subsubsection{Connection to LoRA}

The CP factorization has an interesting connection to Low-Rank Adaptation (LoRA):
\begin{itemize}
    \item LoRA: $\mathbf{W} + \Delta\mathbf{W}$ where $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}^\top$
    \item CP-Bilinear: Interaction tensor $\mathbf{T} = \sum_r \lambda_r \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r$
\end{itemize}

Both impose low-rank structure, but CP-Bilinear does so on the \emph{interaction tensor}
rather than the weight matrix. This suggests a ``BiLoRA'' approach:
fine-tune bilinear transformers by adjusting only the CP rank $R$,
maintaining interpretability while adapting to new tasks.

\subsubsection{Scalability Considerations}

For a ViT with $L$ layers, each with a CP-Bilinear FFN of rank $R$:
\begin{itemize}
    \item Parameters per layer: $O(R \cdot d)$ vs $O(d \cdot d_{ff})$ for standard
    \item When $R \ll d_{ff}$, significant parameter savings
    \item Interpretability: Each layer's interaction tensor has rank exactly $R$
\end{itemize}

We leave implementation and empirical validation to future work.
```

### 5.2 Notes for ViT Section
- This is CONCEPTUAL ONLY - no code implementation
- Focus on mathematical elegance and potential benefits
- Connect to existing literature (LoRA)
- Be honest about limitations (not implemented, future work)

---

## EXECUTION CHECKLIST

### Day 14-15: Extension 4 (Synthesis)
- [ ] Verify all checkpoints exist (Phase 1 + Phase 2)
- [ ] Create `src/synthesize_results.py`
- [ ] Load all results into unified DataFrame
- [ ] Generate Pareto frontier figure (`structure_vs_reg.pdf`)
- [ ] Generate eigenspectrum overlay (`eigenspectrum_overlay.pdf`)
- [ ] Generate eigenvector comparison (`eigenvector_comparison.pdf`)
- [ ] Perform statistical significance tests
- [ ] Create summary table for report

### Day 16: Extension 4 Completion
- [ ] Complete `notebooks/03_all_results.ipynb`
- [ ] Copy all figures to `Report/figures/`
- [ ] Write captions for each figure

### Day 16-17: Extension 5 (ViT Formulation)
- [ ] Write mathematical formulation in LaTeX
- [ ] Add to `Report/sections/05_extension.tex`
- [ ] Review for correctness

### Day 17: Result Freeze
- [ ] All figures finalized
- [ ] All notebooks complete
- [ ] Summary table ready
- [ ] ViT section written

---

## KEY FIGURES CHECKLIST

| Figure | Source | Destination | Status |
|--------|--------|-------------|--------|
| `structure_vs_reg.pdf` | Create new | Report/figures/ | [ ] |
| `pareto_frontier.pdf` | Create new | results/phase2/figures/ | [ ] |
| `eigenspectrum_overlay.pdf` | Create new | Report/figures/ | [ ] |
| `eigenvector_comparison.pdf` | Create new | Report/figures/ | [ ] |
| `robustness_rotated.pdf` | Person C | Verify in Report/figures/ | [ ] |
| `cp_rank_sweep.pdf` | Person D | Verify in Report/figures/ | [ ] |

---

**Begin by verifying all checkpoints exist from Phase 1 and Phase 2, then create `src/synthesize_results.py`. Show me each file and ask for confirmation before proceeding.**

## PROMPT END
