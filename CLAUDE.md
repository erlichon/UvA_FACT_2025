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

### Vision Training (Section 4)

```bash
# Local test (CPU/MPS, no wandb, quick)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2

# Full training with tracking
python src/train.py --config configs/mnist_dense_full.yaml --seed 42

# Available configs: {mnist,fashion}_dense_{none,noise,wd,full}.yaml
```

### Language Experiments (Section 5)

```bash
# Negation circuit discovery (uses pretrained SAEs by default)
python src/language/negation_discovery.py --config configs/language_negation.yaml --use-pretrained

# Interaction matrix analysis
python src/language/interaction_analysis.py --config configs/language_interaction.yaml

# Train SAE from scratch (optional)
python src/language/run_sae_training.py --config configs/language_sae.yaml --no-wandb
```

### Testing

```bash
python -m pytest tests/ -v                          # Run all tests
python -m pytest tests/test_bilinear_layer.py -v    # Specific file
python -m pytest tests/ -v -k "test_effective_rank" # By name
python -m pytest tests/ --cov=src --cov-report=term-missing  # With coverage
```

### Analysis & Figures

```bash
python scripts/generate_figures.py       # Generate all figures from checkpoints
python scripts/recompute_effective_rank.py  # Recompute ranks after formula fix
```

### Snellius HPC

```bash
ssh scur0075@Snellius  # Case-sensitive hostname

sbatch jobs/train_array.job           # 20 MNIST runs (4 configs x 5 seeds)
sbatch jobs/train_fashion_array.job   # 20 Fashion-MNIST runs
sbatch jobs/language_full_pipeline.job  # Full Section 5

squeue -u scur0075  # Monitor jobs
```

## Architecture

### Vision Pipeline (`src/train.py`)

1. Load config YAML -> Create `Model` from `image.model` (original code)
2. Apply noise augmentation via `kornia.augmentation.RandomGaussianNoise`
3. Train with `model.fit()` -> Compute eigendecomposition via `model.decompose()`
4. Save checkpoint with eigenvalues/eigenvectors pre-computed

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/models/bilinear_layer.py` | `BilinearDense` (wraps original), `BilinearCP` (extension) |
| `src/analysis/spectral.py` | `effective_rank()`, `top_k_coverage()`, `load_checkpoint_eigenvalues()` |
| `src/plot_utils/` | Publication plotting: `style.py`, `eigenspectrum.py`, `eigenvectors.py`, `ablation.py` |
| `src/utils.py` | `get_device()`, `load_config()`, `set_seed()`, `track_emissions()`, wandb helpers |
| `src/language/` | SAE training, negation discovery, interaction analysis |

### Original Paper Code

**Location**: `bilinear-decomposition-main/` (DO NOT MODIFY)

Key files used by our wrappers:
- `shared/components.py`: `Bilinear` layer class
- `image/model.py`: `Model` class with `.fit()` and `.decompose()` methods
- `image/datasets.py`: GPU-resident `MNIST`, `FMNIST` loaders
- `sae/sae.py`: `SAE` and `SAEConfig` classes for sparse autoencoders

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
```

## Critical Notes

### MPS Device Bugs

**einsum on MPS produces incorrect results** for certain operations. The interaction analysis code forces CPU for einsum operations:
```python
# In src/language/interaction_analysis.py
if device.type == "mps":
    # Force CPU for einsum due to MPS bugs
    tensor = tensor.cpu()
```

Language experiments are slow on MPS (~4-6 hours). Prefer Snellius GPU cluster.

### Effective Rank Formula

**Use the ratio-based formula** (paper's formula), NOT entropy-based:
```python
# Correct: src/analysis/spectral.py::effective_rank()
def effective_rank(eigenvalues):
    L1 = eigenvalues.abs().sum(dim=-1)
    L2 = (eigenvalues ** 2).sum(dim=-1).sqrt()
    return (L1 / L2) ** 2
```

### Language Model Configuration

Configs use `fw-medium` model (335M params) with layer=7, expansion=8, k=30. The paper uses k=32 but pretrained SAEs only have k=30.

### Checkpoint Column Compatibility

Handle both `train_acc` and `train/acc` column formats when loading metrics from different checkpoint versions.

## wandb

- **Project**: `itayerlich96-student/fact-bilinear`
- **Tags**: Vision: `["vision", "mnist"]`, Language: `["language", "negation"]`
- **Metrics**: train/acc, val/acc, effective_rank, wall_time_hours, gpu_hours, co2_kg

## Code Conventions

### Import Original Code

```python
import sys
from pathlib import Path

_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from image.model import Model, Config
from shared.components import Bilinear
```

### Notebooks

Use project root discovery at the start:
```python
import sys
from pathlib import Path

cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
sys.path.insert(0, str(PROJECT_ROOT))
```

Import from modules rather than defining functions inline. Generate final figures via `scripts/generate_figures.py`.

## Related Documentation

- `CONTEXT.md`: Current project state, team roles, detailed results
- `WORKPLAN.md`: Experiment matrix and timeline
- `.cursorrules`: Additional code conventions and research questions
- `docs/WANDB_GUIDE.md`: Comprehensive wandb usage guide
