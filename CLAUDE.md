# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Course**: FACT-AI (Fairness, Accountability, Confidentiality and Transparency in AI), UvA MSc AI

**Objective**: Reproduce and extend "Bilinear MLPs enable weight-based mechanistic interpretability" ([arXiv:2410.08417](https://arxiv.org/pdf/2410.08417)).

**Reproduction Scope**:
- **Section 4 (Vision)**: MNIST/Fashion-MNIST bilinear MLPs with eigendecomposition
- **Section 5 (Language)**: Negation circuit discovery in bilinear transformers via SAE analysis

**Extension Hypothesis**: Structural low-rank (CP-Decomposition) produces equivalent or better interpretability than emergent low-rank (via regularization).

## Commands

### Training

```bash
# Local test (CPU/MPS, no wandb, quick)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2

# Full training with tracking
python src/train.py --config configs/mnist_dense_full.yaml --seed 42

# Submit all Phase 1 experiments to Snellius (20 runs: 4 configs x 5 seeds)
sbatch jobs/train_array.job
```

### Environment Setup

```bash
# Snellius (GPU)
conda env create -f environment.yml && conda activate fact

# Local (CPU only)
conda env create -f environment_cpu.yml && conda activate fact_cpu
```

### Snellius HPC

```bash
ssh scur0075@Snellius  # Case-sensitive hostname
sbatch jobs/train_array.job
squeue -u scur0075
```

## Architecture

### Training Pipeline (`src/train.py`)

Wraps the original paper's implementation. Key flow:
1. Load config YAML -> Create `Model` from `image.model` (original code)
2. Apply noise augmentation via `kornia.augmentation.RandomGaussianNoise`
3. Train with `model.fit()` -> Compute eigendecomposition via `model.decompose()`
4. Save checkpoint with eigenvalues/eigenvectors pre-computed

### Bilinear Layer (`src/models/bilinear_layer.py`)

Two modes:
- **`BilinearDense`**: Wraps original `shared/components.py::Bilinear` for exact reproduction
- **`BilinearCP`**: CP-decomposed variant for structural low-rank experiments (extension)

Both expose `w_l` and `w_r` properties for eigendecomposition compatibility.

### Spectral Analysis (`src/analysis/spectral.py`)

Key metrics:
- `effective_rank(eigenvalues)`: Entropy-based rank (Roy & Bhattacharyya 2007). Lower = more interpretable.
- `top_k_coverage(eigenvalues, k)`: Variance explained by top-k eigenvalues
- `load_checkpoint_eigenvalues(path)`: Returns `(eigenvalues, eigenvectors)` tuple

### Config Format (YAML)

```yaml
model:
  mode: dense       # 'dense' or 'cp'
  d_hidden: 256
regularization:
  noise_std: 0.4    # Gaussian noise augmentation
  weight_decay: 0.5
training:
  epochs: 100
  lr: 0.001
```

### Checkpoint Format

```python
{
    'config': {                    # Flat dict for analysis compatibility
        'mode': 'dense',
        'd_hidden': 256,
        'noise_std': 0.4,
        'weight_decay': 0.5,
    },
    'model_state_dict': ...,
    'metrics': {
        'train_acc': float,
        'val_acc': float,
        'effective_rank': float,   # Mean across classes
    },
    'seed': int,
    'eigenvalues': Tensor,         # [10, 256] (n_classes, d_hidden)
    'eigenvectors': Tensor,        # [10, 256, 784] (n_classes, d_hidden, d_input)
}
```

## Original Paper Code

**Location**: `bilinear-decomposition-main/` (DO NOT MODIFY)

Key files used by our implementation:
- `shared/components.py`: `Bilinear` layer class
- `image/model.py`: `Model` class with `.fit()` and `.decompose()` methods
- `image/datasets.py`: GPU-resident `MNIST` loader

## Key Paper Claims to Verify

### Section 4 (Vision)
| Claim | Metric | Expected |
|-------|--------|----------|
| Low-rank emergence | Effective rank ratio (reg/no-reg) | < 0.5 |
| Interpretable eigenvectors | Visual inspection | Digit-like patterns |
| Cross-seed consistency | Cosine similarity | 0.8-0.9 |

### Section 5 (Language)
| Claim | Metric | Expected |
|-------|--------|----------|
| Negation features exist | SAE feature analysis | Features 3834, 751 |
| Low-rank interactions | Rank-2 correlation | >69% with >0.75 |

## Analysis Example

```python
import torch
from src.analysis.spectral import load_checkpoint_eigenvalues, spectral_summary

eigenvalues, eigenvectors = load_checkpoint_eigenvalues(
    "results/phase1/checkpoints/mnist_dense_full_seed42.pt"
)
summary = spectral_summary(eigenvalues)
print(f"Effective Rank: {summary['effective_rank_mean']:.2f}")

# Visualize eigenvector as 28x28 image
img = eigenvectors[0, 0].reshape(28, 28)  # First eigenvector of class 0
```

## Detailed Documentation

- `CONTEXT.md`: Full project knowledge (team roles, technical details, paper background)
- `WORKPLAN.md`: Experiment matrix and timeline
- `Report/`: LaTeX report (TMLR template)
