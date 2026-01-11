# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**Course**: FACT-AI (Fairness, Accountability, Confidentiality and Transparency in AI), UvA MSc AI, January 2026

**Objective**: Reproduce and extend "Bilinear MLPs enable weight-based mechanistic interpretability" ([arXiv:2410.08417](https://arxiv.org/pdf/2410.08417)) following MLRC guidelines.

**Full Reproduction Scope**:
- **Section 4 (Vision)**: MNIST/Fashion-MNIST bilinear MLPs with eigendecomposition
- **Section 5 (Language)**: Negation circuit discovery in bilinear transformers via SAE analysis

**Extension Hypothesis**: Structural low-rank (CP-Decomposition) produces equivalent or better interpretability than emergent low-rank (via regularization).

## Current State

### Section 4 (Vision) - Status: ~70% Infrastructure Complete

**Implemented**:
- Training pipeline (`src/train.py`)
- Bilinear layers (`src/models/bilinear_layer.py`) - BilinearDense + BilinearCP
- Spectral analysis (`src/analysis/spectral.py`)
- MNIST ablation configs (4 configs x 5 seeds)
- SLURM job scripts

**Missing**:
- Fashion-MNIST configs and experiments
- Cross-seed consistency analysis
- Adversarial example generation
- Visualization module (`src/analysis/visualization.py`)

### Section 5 (Language) - Status: ~5% Complete

**What it involves**:
1. Train/load bilinear transformer on TinyStories
2. Train Sparse Autoencoders (SAEs) around MLP layers
3. Discover negation circuit features (feature 3834: "not + negative", feature 751: "not + positive")
4. Analyze interaction matrices between SAE features
5. Verify low-rank structure (69% of features with >0.75 rank-2 correlation)

**Missing** (needs implementation):
- `src/language/` module
- TinyStories data loader
- SAE training pipeline
- Negation circuit discovery code
- Interaction matrix analysis

## Repository Structure

```
UvA_FACT_2025/
├── bilinear-decomposition-main/   # Original paper code (reference)
├── src/
│   ├── models/
│   │   └── bilinear_layer.py      # BilinearDense + BilinearCP
│   ├── analysis/
│   │   ├── spectral.py            # effective_rank, top_k_coverage
│   │   └── visualization.py       # [NEEDED] Plotting functions
│   ├── language/                  # [NEEDED] Section 5 implementation
│   │   ├── dataset.py             # TinyStories loader
│   │   ├── transformer.py         # Bilinear transformer wrapper
│   │   ├── sae.py                 # SAE training/loading
│   │   └── circuit_analysis.py    # Negation circuit discovery
│   └── train.py                   # Vision training script
├── configs/
│   ├── mnist_dense_*.yaml         # MNIST ablation (4 configs)
│   ├── fashion_dense_*.yaml       # [NEEDED] Fashion-MNIST
│   └── language_*.yaml            # [NEEDED] Language configs
├── jobs/
│   ├── train_array.job            # MNIST Phase 1 (20 runs)
│   ├── train_fashion.job          # [NEEDED] Fashion-MNIST
│   └── train_language.job         # [NEEDED] Language experiments
├── results/
│   ├── phase1/checkpoints/        # Vision checkpoints
│   └── language/                  # [NEEDED] Language checkpoints
├── notebooks/                     # Analysis notebooks
└── Report/                        # LaTeX report (TMLR template)
```

## Environment Setup

**For Snellius (GPU)**:
```bash
conda env create -f environment.yml
conda activate fact
```

**For Local (CPU)**:
```bash
conda env create -f environment_cpu.yml
conda activate fact_cpu
```

## Usage: Vision Training

```bash
# Local test (2 epochs, no wandb)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2

# Full training
python src/train.py --config configs/mnist_dense_full.yaml --seed 42

# Submit all MNIST experiments to Snellius
sbatch jobs/train_array.job
```

## Usage: Loading Checkpoints

```python
import torch
from src.analysis.spectral import load_checkpoint_eigenvalues, spectral_summary

# Load eigenvalues/eigenvectors
eigenvalues, eigenvectors = load_checkpoint_eigenvalues(
    "results/phase1/checkpoints/mnist_dense_full_seed42.pt"
)

# Compute summary statistics
summary = spectral_summary(eigenvalues)
print(f"Mean Effective Rank: {summary['effective_rank_mean']:.2f}")

# Reshape eigenvectors for visualization (28x28 for MNIST)
img = eigenvectors[0, 0].reshape(28, 28)  # First eigenvector of class 0
```

## Checkpoint Format

```python
checkpoint = {
    'config': {
        'mode': 'dense',           # 'dense' or 'cp'
        'd_hidden': 256,
        'epochs': 100,
        'lr': 0.001,
        'noise_std': 0.4,          # 0.0 for no noise
        'weight_decay': 0.5,       # 0.0 for no weight decay
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

## Snellius HPC Cluster

```bash
ssh scur0075@Snellius  # Case-sensitive hostname

# Submit jobs
sbatch jobs/train_array.job        # MNIST Phase 1 (20 runs)
squeue -u scur0075                 # Check status
```

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
| Sparse interaction matrices | Top-50 concentration | Small submatrix |

## Phase Summary

| Phase | Scope | Status |
|-------|-------|--------|
| **Vision (Section 4)** | MNIST + Fashion-MNIST | ~70% infrastructure |
| **Language (Section 5)** | TinyStories + SAE + Negation | ~5% infrastructure |
| **Extensions (CP)** | Structural low-rank alternative | Skeleton only |

See `CONTEXT.md` for full project knowledge and `WORKPLAN.md` for timeline.
