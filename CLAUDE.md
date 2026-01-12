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

### Environment Setup

```bash
# Snellius (GPU)
conda env create -f environment.yml && conda activate fact

# Local (CPU/MPS)
conda env create -f environment_cpu.yml && conda activate fact_cpu
```

### Section 4 (Vision) Training

```bash
# Local test (CPU/MPS, no wandb, quick)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2

# Full training with tracking
python src/train.py --config configs/mnist_dense_full.yaml --seed 42

# Fashion-MNIST variant
python src/train.py --config configs/fashion_dense_full.yaml --seed 42

# Available configs: {mnist,fashion}_dense_{none,noise,wd,full}.yaml
```

### Section 5 (Language) Experiments

```bash
# Negation circuit discovery (uses pretrained SAEs by default)
python src/language/negation_discovery.py --config configs/language_negation.yaml --use-pretrained

# Interaction matrix analysis
python src/language/interaction_analysis.py --config configs/language_interaction.yaml

# Train SAE from scratch (optional - can use pretrained)
python src/language/run_sae_training.py --config configs/language_sae.yaml --no-wandb

# Full Section 5 pipeline (Snellius)
sbatch jobs/language_full_pipeline.job
```

> **WARNING**: Language experiments are very slow on local MPS devices (4-6+ hours). Run on Snellius GPU cluster for faster iteration.

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_bilinear_layer.py -v

# Run single test by name
python -m pytest tests/ -v -k "test_effective_rank"

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Analysis Utilities

```bash
# Recompute effective rank for all checkpoints (after formula fix)
python scripts/recompute_effective_rank.py
```

### Local MPS Scripts (Apple Silicon)

```bash
# Quick vision test (2 epochs)
./scripts/test_mps_quick.sh

# Quick language test
./scripts/test_mps_language.sh

# Full overnight run (~6-8 hours on M1/M2/M3/M4)
./scripts/run_overnight_mps.sh

# Run in background
nohup ./scripts/run_overnight_mps.sh > logs/overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Snellius HPC

```bash
ssh scur0075@Snellius  # Case-sensitive hostname

# Submit jobs
sbatch jobs/train_array.job           # 20 MNIST runs (4 configs x 5 seeds)
sbatch jobs/train_fashion_array.job   # 20 Fashion-MNIST runs
sbatch jobs/language_full_pipeline.job  # Full Section 5

# Monitor
squeue -u scur0075
```

## Architecture

### Section 4 (Vision)

**Training Pipeline (`src/train.py`)**:
1. Load config YAML -> Create `Model` from `image.model` (original code)
2. Apply noise augmentation via `kornia.augmentation.RandomGaussianNoise`
3. Train with `model.fit()` -> Compute eigendecomposition via `model.decompose()`
4. Save checkpoint with eigenvalues/eigenvectors pre-computed

**Bilinear Layer (`src/models/bilinear_layer.py`)**:
- `BilinearDense`: Wraps original `shared/components.py::Bilinear` for exact reproduction
- `BilinearCP`: CP-decomposed variant for structural low-rank experiments (extension)

Both expose `w_l` and `w_r` properties for eigendecomposition compatibility.

**Spectral Analysis (`src/analysis/spectral.py`)**:
- `effective_rank(eigenvalues)`: Ratio-based rank `(L1/L2)^2`. Lower = more interpretable.
- `top_k_coverage(eigenvalues, k)`: Variance explained by top-k eigenvalues
- `load_checkpoint_eigenvalues(path)`: Returns `(eigenvalues, eigenvectors)` tuple

### Shared Utilities (`src/utils.py`)

Common functionality used across training scripts:
- `get_device()`: Auto-detect best device (cuda > mps > cpu)
- `load_config()`: Load YAML configuration
- `set_seed()`: Set random seeds for reproducibility
- `track_emissions()`: Context manager for CO2 + wall time tracking
- `init_wandb()` / `finish_wandb()`: wandb lifecycle management

### Section 5 (Language)

**SAE Training (`src/language/run_sae_training.py`)**:
- Wraps original `sae/sae.py::SAE` from the paper code
- Trains on TinyStories dataset via HuggingFace `datasets`
- Uses pretrained bilinear transformer from `tdooms/ts-medium`

**Negation Discovery (`src/language/negation_discovery.py`)**:
- Classifies samples by pattern: "not + positive", "not + negative", baseline
- Finds SAE features with differential activation on negation patterns
- Verifies opposing directions via cosine similarity of decoder weights
- Paper target features: 3834 (not+negative), 751 (not+positive)

**Interaction Analysis (`src/language/interaction_analysis.py`)**:
- Computes interaction matrices Q[i,j] for SAE feature pairs
- Verifies low-rank structure via rank-2 correlation (paper claim: >69% with >0.75)

### Config Format (YAML)

```yaml
# Vision config
model:
  mode: dense       # 'dense' or 'cp'
  d_hidden: 256
regularization:
  noise_std: 0.4    # Gaussian noise augmentation
  weight_decay: 0.5
training:
  epochs: 100
  lr: 0.001
data:
  dataset: mnist    # 'mnist' or 'fashion_mnist'
```

### Checkpoint Format

```python
# Vision checkpoint
{
    'config': {'mode': 'dense', 'd_hidden': 256, 'noise_std': 0.4, 'weight_decay': 0.5},
    'model_state_dict': ...,
    'metrics': {'train_acc': float, 'val_acc': float, 'effective_rank': float},
    'seed': int,
    'eigenvalues': Tensor,         # [10, 256] (n_classes, d_hidden)
    'eigenvectors': Tensor,        # [10, 256, 784] (n_classes, d_hidden, d_input)
}

# SAE checkpoint (language)
{
    'sae_state_dict': ...,
    'sae_config': {'point': ['mlp-out', 2], 'expansion': 8, 'k': 32, ...},
    'model_name': 'tdooms/ts-medium',
}
```

## Original Paper Code

**Location**: `bilinear-decomposition-main/` (DO NOT MODIFY)

Key files used by our wrappers:
- `shared/components.py`: `Bilinear` layer class
- `image/model.py`: `Model` class with `.fit()` and `.decompose()` methods
- `image/datasets.py`: GPU-resident `MNIST`, `FMNIST` loaders
- `sae/sae.py`: `SAE` and `SAEConfig` classes for sparse autoencoders
- `language/transformer.py`: `Transformer.from_pretrained()` for bilinear transformers

## Key Paper Claims

### Section 4 (Vision)
- **Low-rank emergence**: Regularization should reduce effective rank ratio (reg/no-reg) < 0.5
- **Accuracy trade-off**: ~94-95% test accuracy with regularization
- **Interpretable eigenvectors**: Top eigenvectors should visually resemble digits

### Section 5 (Language)
- **Negation features**: SAE features 3834 (not+negative) and 751 (not+positive) form opposing directions
- **Low-rank interactions**: >69% of interaction matrices should have >0.75 rank-2 correlation
- **Model**: Paper uses `fw-medium` (FineWeb-EDU trained) at layer 7, expansion=8

## Important Notes

### Effective Rank Formula
Two formulas available in `src/analysis/spectral.py`:
- `effective_rank(eigenvalues)`: Paper's ratio-based formula `(L1/L2)^2` (USE THIS)
- `effective_rank_entropy(eigenvalues)`: Alternative entropy-based formula

### Language Model Configuration
Configs use `fw-medium` model (335M params) with:
- Layer: 7
- Expansion: 8
- k (TopK): 30 (closest to paper's k=32 in pretrained SAEs)

See `CONTEXT.md` for detailed results and `AGENT_DISCREPENCY_FIX_EXPLANATION.md` for investigation notes.

## Analysis Examples

```python
# Vision: Load and analyze eigenspectrum
from src.analysis.spectral import load_checkpoint_eigenvalues, spectral_summary

eigenvalues, eigenvectors = load_checkpoint_eigenvalues(
    "results/phase1/checkpoints/mnist_dense_full_seed42.pt"
)
summary = spectral_summary(eigenvalues)
print(f"Effective Rank: {summary['effective_rank_mean']:.2f}")

# Visualize eigenvector as 28x28 image
img = eigenvectors[0, 0].reshape(28, 28)  # First eigenvector of class 0
```

```python
# Language: Analyze negation results
import json
with open("results/language/negation_analysis.json") as f:
    results = json.load(f)
print(f"Top not+positive feature: {results['not_positive_features'][0]}")
print(f"Opposing directions: {results['top_pair_analysis']['opposing_directions']}")
```

## Experiment Tracking (wandb)

All experiments log to a single wandb project with tags for filtering:
- **Project URL**: https://wandb.ai/itayerlich96-student/fact-bilinear
- **Entity**: `itayerlich96-student`
- **Project**: `fact-bilinear`

**Tags**:
- Vision: `["vision", "mnist"]` or `["vision", "fashion_mnist"]`
- Language: `["language", "sae"]`, `["language", "negation"]`, `["language", "interaction"]`

**Metrics logged**:
- Per-epoch: train/acc, val/acc, train/loss, val/loss
- Summary: effective_rank, top5_coverage, top10_coverage, per-class ranks
- Resources: wall_time_hours, gpu_hours, co2_kg

## Detailed Documentation

- `CONTEXT.md`: Full project knowledge (team roles, technical details, paper background)
- `WORKPLAN.md`: Experiment matrix and timeline
- `docs/WANDB_GUIDE.md`: Comprehensive wandb usage guide
- `Report/`: LaTeX report (TMLR template)
