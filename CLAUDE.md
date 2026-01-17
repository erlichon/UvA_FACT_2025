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

### Unified Vision Script (Section 4) - PREFERRED

```bash
# Quick MPS test (2 epochs)
./scripts/train/run_vision.sh test

# Train base configs (4 configs × 5 seeds for MNIST + Fashion)
./scripts/train/run_vision.sh train base
./scripts/train/run_vision.sh train base --quick --no-wandb  # Quick test mode

# Train noise sweep (Figure 4)
./scripts/train/run_vision.sh train noise

# Train model size sweep (Figure 5)
./scripts/train/run_vision.sh train size

# Train everything
./scripts/train/run_vision.sh train all

# Generate all figures from checkpoints
./scripts/train/run_vision.sh figures
./scripts/train/run_vision.sh figures --section regularization  # Specific section

# Full pipeline (train + figures)
./scripts/train/run_vision.sh all
```

### Extension 2: Cross-Dataset Robustness

Uses Phase 1 settings (noise=0.5, weight_decay=1.0) with CoM normalization.
Cosine similarity only for subspace overlap.

```bash
# Train EMNIST models (CoM always enabled)
./scripts/train/run_extension2.sh train all

# Generate all Extension 2 figures
./scripts/train/run_extension2.sh figures

# Generate specific figure sections
./scripts/train/run_extension2.sh figures similarity 3way

# Full pipeline (train + figures)
./scripts/train/run_extension2.sh all

# Direct figure generation
python scripts/figures/generate_extension2_figures.py
python scripts/figures/generate_extension2_figures.py --sections eigenvectors heatmaps
```

### Unified Language Script (Section 5) - PREFERRED

```bash
# Quick MPS test
./scripts/train/run_language.sh test

# Figure 9: Correlation sweep (all 3 models)
./scripts/train/run_language.sh figure9
./scripts/train/run_language.sh figure9 --model fw-medium   # Single model
./scripts/train/run_language.sh figure9 --quick             # Quick mode

# Figure 8: Negation circuit visualization (memory-efficient, works on MPS)
./scripts/train/run_language.sh figure8 --device mps

# Negation discovery
./scripts/train/run_language.sh negation

# Interaction analysis
./scripts/train/run_language.sh interaction

# Generate all language figures (Figure 9 & 10)
./scripts/train/run_language.sh figures

# Full pipeline (except Figure 8)
./scripts/train/run_language.sh all
```

### Full Overnight Pipeline

```bash
# Run ALL experiments (vision + language)
./scripts/train/run_overnight_mps.sh
```

### Figure Generation Only

```bash
# Vision figures (Figures 1-7)
python scripts/figures/generate_vision_figures.py

# Language figures (Figure 9 & 10)
python scripts/figures/generate_language_figures.py
```

### Legacy Commands (still work, but deprecated)

```bash
# Vision training (direct Python)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2

# Language correlation verification (direct Python)
python src/language/verify_correlation.py --config configs/language_correlation_fw.yaml \
    --model tdooms/fw-medium --layer 7 --expansion 8 --k 30
```

### Testing

```bash
python -m pytest tests/ -v                          # Run all tests
python -m pytest tests/test_bilinear_layer.py -v    # Specific file
python -m pytest tests/ -v -k "test_effective_rank" # By name
python -m pytest tests/ --cov=src --cov-report=term-missing  # With coverage
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
| `src/data/` | Unified data module: `MNIST`, `FashionMNIST`, `EMNISTLetters`, `EMNISTDigits`, `USPS` with CoM support |
| `src/data/transforms.py` | `CenterOfMassTransform` for input geometry normalization |
| `src/training/core.py` | Training utilities: `train_model()`, `save_checkpoint()`, `log_training_history()` |
| `src/vision/spectral.py` | `effective_rank()`, `top_k_coverage()`, `load_checkpoint_eigenvalues()` |
| `src/vision/subspace.py` | `compute_subspace_overlap()`, `principal_angles()` for Extension 2 |
| `src/vision/context.py` | `VisionContext` - unified context for vision experiments |
| `src/vision/truncation.py` | `compute_truncation_accuracy()`, `compute_eigenvector_similarity()` (Figure 5) |
| `src/vision/adversarial.py` | `compute_adversarial_mask()`, `apply_adversarial_perturbation()` (Figure 7) |
| `src/data/challenge_dataset.py` | `ChallengeDataset` for similarity classification (Figure 6) |
| `src/plot_utils/` | Publication plotting: `style.py`, `eigenspectrum.py`, `eigenvectors.py`, `ablation.py`, `language.py` |
| `src/utils.py` | `get_device()`, `load_config()`, `set_seed()`, `seed_worker()`, `track_emissions()`, wandb helpers |
| `src/language/context.py` | `LanguageContext` - unified context for language experiments |
| `src/language/memory_efficient_eigen.py` | Memory-efficient iterative eigensolver for Figure 8 (avoids n_features² memory) |
| `src/language/` | SAE training, negation discovery, interaction analysis, correlation verification |

### Scripts Organization

| Directory | Purpose |
|-----------|---------|
| `scripts/train/` | Training & experiment runners (`run_vision.sh`, `run_language.sh`, `run_extension2.sh`, `run_overnight_mps.sh`) |
| `scripts/figures/` | Figure generation (`generate_vision_figures.py`, `generate_language_figures.py`, `generate_extension2_figures.py`) |
| `scripts/figures/` | Figure generation (`generate_vision_figures.py`, `generate_language_figures.py`, `paper_hub.py`) |
| `tools/` | Operational utilities (`sync_to_snellius.sh`, `sync_from_snellius.sh`, `monitor_memory.sh`) |

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

### Context Classes (DRY Pattern)

Use context classes to avoid code duplication:
```python
# Vision experiments
from src.vision import VisionContext
ctx = VisionContext()
eigenvalues, eigenvectors = ctx.load_checkpoint("path/to/checkpoint.pt")
ctx.save_figure(fig, "figure_name")

# Language experiments
from src.language import LanguageContext
ctx = LanguageContext(model_name="tdooms/fw-medium", layer=7, expansion=8)
model = ctx.get_model()
sae = ctx.get_sae(position="mlp-out")
```

### MPS Device Compatibility

**Eigendecomposition** (`model.decompose()`) is MPS-safe - the original paper code automatically moves tensors to CPU for `torch.linalg.eigh()` when on MPS.

**einsum on MPS produces incorrect results** for certain operations. The interaction analysis code forces CPU for einsum operations:
```python
# In src/language/interaction_analysis.py
if device.type == "mps":
    # Force CPU for einsum due to MPS bugs
    tensor = tensor.cpu()
```

Language experiments are slow on MPS (~4-6 hours). Prefer Snellius GPU cluster.

### Reproducibility (Seeding)

Always use `set_seed()` from `src/utils.py` at the start of experiments. It handles:
- Python `random.seed()`
- NumPy `np.random.seed()`
- PyTorch `torch.manual_seed()` and CUDA seeds
- CUDA determinism settings (`torch.backends.cudnn.deterministic = True`)

For DataLoader multi-worker determinism, use `seed_worker()`:
```python
from src.utils import set_seed, seed_worker
set_seed(42)
dataloader = DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)
```

### Effective Rank Formula

**Use the ratio-based formula** (paper's formula), NOT entropy-based:
```python
# Correct: src/vision/spectral.py::effective_rank()
def effective_rank(eigenvalues):
    L1 = eigenvalues.abs().sum(dim=-1)
    L2 = (eigenvalues ** 2).sum(dim=-1).sqrt()
    return (L1 / L2) ** 2
```

### Language Model Configuration

**IMPORTANT CLARIFICATION** (Verified 2026-01-14):

The paper has TWO different language experiments:
1. **Paper Section 5.1 (Figure 8)**: Uses ts-tiny (available as `tdooms/ts-medium`)
2. **Tutorial/Example**: Uses fw-medium for demonstration

**Model Configuration**:
| Model | Layers | d_model | Params | SAE Layer | Expansion | SAE Repo | Paper/Tutorial Use |
|-------|--------|---------|--------|-----------|-----------|----------|-------------------|
| **ts-medium** | 6 | 512 | 29.4M | 4 | 4 | `tdooms/ts-medium-scope` | Figure 9 only (no Fig 8 - missing mlp-in SAEs) |
| fw-small | 12 | - | 162M | 8 | 4 | `tdooms/fw-small-scope` | Figure 9 |
| fw-medium | 16 | 1024 | 335M | 7 | 8 | `tdooms/fw-medium-scope` | **TUTORIAL Figure 8** (features 3834/751) + Fig 9 |

**Critical Notes**:
- ✅ **VERIFIED**: `tdooms/ts-medium` IS the paper's "ts-tiny" (6L, 512d, ~29M params, TinyStories)
- ⚠️ **SAE LIMITATION**: ts-medium layer 4 does NOT have `mlp-in` SAEs available on HuggingFace
  - Available: `mlp-out`, `resid-mid`, `resid-pre` only
  - Figure 8 requires BOTH `mlp-in` and `mlp-out` SAEs (for Tracer class)
  - Paper likely used internal SAE checkpoints not publicly released
- **Figure 8 Reproduction**: Uses fw-medium (layer 7) which has both SAEs available
  - Features **3834** (not-good), **751** (not-bad) from tutorial example
  - Demonstrates the same negation circuit phenomenon
  - Clearer visualizations due to larger model capacity (335M vs 29M params)
- **Figure 9 Reproduction**: All three models work (only needs output SAEs)
  - ts-medium layer 4: expansion=4, k=30
  - fw-small layer 8: expansion=4, k=30
  - fw-medium layer 7: expansion=8, k=30
- All pretrained SAEs use k=30 (paper uses k=32, minor difference)
- Use ~2/3 model depth for SAE layer selection

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

Import from modules rather than defining functions inline. Generate final figures via `./scripts/train/run_vision.sh figures` or `python scripts/figures/generate_vision_figures.py`.

## Related Documentation

- `CONTEXT.md`: Current project state, team roles, detailed results
- `WORKPLAN.md`: Experiment matrix and timeline
- `.cursorrules`: Additional code conventions and research questions
- `docs/WANDB_GUIDE.md`: Comprehensive wandb usage guide
