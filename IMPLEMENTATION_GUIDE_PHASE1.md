# Phase 1 Implementation Guide: Person A (Infrastructure Lead)

> **Purpose**: This document provides complete specifications for implementing Phase 1 (Reproduction) infrastructure. Load this into a Claude instance along with the prompt at the bottom.

---

## 1. Project Context

**Goal**: Reproduce Section 4 (Vision) of "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., 2024).

**Your Role**: Person A - Infrastructure Lead
- Create training pipeline with experiment tracking
- Set up SLURM job submission for Snellius cluster
- Implement unified bilinear layer (dense + CP modes)

**Timeline**: Day 2-3 of 7-day Phase 1

---

## 2. Directory Structure to Create

```
UvA_FACT_2025/
├── bilinear-decomposition-main/   # UNTOUCHED - original paper code
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilinear_layer.py      # Unified dense + CP layer
│   │   └── image_model.py         # Wrapper around original Model
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── spectral.py            # Effective rank, eigenvalue metrics
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py            # Re-export original datasets
│   └── train.py                   # Main training script
├── configs/
│   ├── mnist_dense_none.yaml      # P1.1: no regularization
│   ├── mnist_dense_noise.yaml     # P1.2: noise only
│   ├── mnist_dense_wd.yaml        # P1.3: weight decay only
│   └── mnist_dense_full.yaml      # P1.4: noise + weight decay
├── jobs/
│   ├── train_single.job           # Single experiment SLURM job
│   └── train_array.job            # Array job for all 20 runs
├── scripts/
│   └── launch_phase1.sh           # Convenience launcher
├── results/
│   └── phase1/
│       ├── checkpoints/
│       └── figures/
├── notebooks/
│   └── 01_reproduction.ipynb      # Analysis notebook (Person B)
├── environment.yml                # Already exists
├── CLAUDE.md                      # Already exists
└── WORKPLAN.md                    # Already exists
```

---

## 3. Original Code Analysis

### Key Files in `bilinear-decomposition-main/`

**`shared/components.py`** - Core Bilinear layer:
```python
class Bilinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None):
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {None: nn.Identity(), "relu": nn.ReLU(), ...}[gate]

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right

    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]
```

**`image/model.py`** - Image classifier:
```python
class Model(PreTrainedModel):
    def __init__(self, config):
        self.embed = Linear(d_input, d_hidden)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output)

    def fit(self, train, test, transform=None):
        # Training loop with AdamW + CosineAnnealing
        # Returns DataFrame with train/val loss/acc

    def decompose(self):
        # Returns (eigenvalues, eigenvectors) per class
```

**`image/datasets.py`** - GPU-resident datasets:
```python
class MNIST(Dataset):
    def __init__(self, train=True, device="cuda"):
        self.x = dataset.data.float().to(device) / 255.0
        self.y = dataset.targets.to(device)
```

### How Original Training Works

```python
from image.model import Model, Config
from image.datasets import MNIST
import kornia

# Create model
model = Model(Config(epochs=100, d_hidden=256, wd=0.5))

# Load data
train, test = MNIST(train=True), MNIST(train=False)

# Add noise augmentation
transform = kornia.augmentation.RandomGaussianNoise(mean=0, std=0.4, p=1.0)

# Train
history = model.fit(train, test, transform=transform)

# Analyze
vals, vecs = model.decompose()
```

---

## 4. File Specifications

### 4.1 `src/models/bilinear_layer.py`

**Purpose**: Unified layer supporting original dense mode AND our CP extension.

**Design Decision**: For dense mode, WRAP the original `Bilinear` class (don't reimplement) to ensure exact reproduction fidelity.

```python
"""
Bilinear Layer: Dense (Original) + CP-Decomposition (Extension)

Dense Mode:
    Wraps original implementation from bilinear-decomposition-main/shared/components.py
    y = gate(W_l @ x) ⊙ (W_r @ x)

CP Mode (Our Extension):
    Explicit rank-R decomposition of the interaction tensor.
    T[i,j,k] = Σ_r λ_r * A[i,r] * B[j,r] * C[k,r]
    y_k = Σ_r λ_r * (A[:,r]ᵀ x) * (B[:,r]ᵀ x) * C[k,r]

    Rearranged for efficient computation:
    y = ((x @ A) ⊙ (x @ B) ⊙ λ) @ Cᵀ
"""
```

**Class Specification**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_in` | int | required | Input dimension |
| `d_out` | int | required | Output dimension |
| `mode` | str | `'dense'` | `'dense'` or `'cp'` |
| `rank` | int | `None` | CP rank (required if mode='cp') |
| `bias` | bool | `False` | Include bias term |
| `gate` | str | `None` | Gating for dense mode: `None`, `'relu'`, `'silu'`, `'gelu'` |

**Properties** (must match original API):
- `w_l` → `[d_out, d_in]` left projection weights
- `w_r` → `[d_out, d_in]` right projection weights

**CP Mode Math** (use `torch.einsum`):
```python
# Forward pass for CP mode:
# A: [d_in, rank]  - left input factors
# B: [d_in, rank]  - right input factors
# C: [d_out, rank] - output factors
# λ: [rank]        - scaling factors

left = torch.einsum('...i, ir -> ...r', x, self.A)    # [batch, rank]
right = torch.einsum('...i, ir -> ...r', x, self.B)   # [batch, rank]
hidden = left * right * self.lambdas                   # [batch, rank]
output = torch.einsum('...r, or -> ...o', hidden, self.C)  # [batch, d_out]
```

**Initialization**:
- Dense mode: Use original (PyTorch default)
- CP mode: `torch.randn(...) * 0.02` for A, B, C; `torch.ones(rank)` for λ

---

### 4.2 `src/models/image_model.py`

**Purpose**: Wrapper around original `image.model.Model` that adds:
1. wandb logging
2. codecarbon tracking
3. Checkpoint saving
4. Support for CP mode bilinear layers

**Key Addition**: The original model hardcodes `Bilinear` from `shared.components`. We need to optionally replace it with our `BilinearCP`.

**Approach**:
- For Phase 1 reproduction (dense mode): Use original Model directly
- For Phase 2 extensions (CP mode): Create modified Model class

```python
# Phase 1: Just wrap original for tracking
from image.model import Model as OriginalModel, Config as OriginalConfig

class TrackedModel(OriginalModel):
    """Original model with wandb + codecarbon tracking."""

    def fit(self, train, test, transform=None, wandb_run=None, tracker=None):
        # Call super().fit() but intercept to log metrics
        ...
```

---

### 4.3 `src/analysis/spectral.py`

**Purpose**: Eigenspectrum analysis utilities.

**Functions**:

```python
def effective_rank(eigenvalues: Tensor) -> float:
    """
    Entropy-based effective rank (Roy & Bhattacharyya, 2007).
    Lower = more interpretable (sharper eigenspectrum).

    Formula: exp(-Σ p_i log(p_i)) where p_i = |λ_i| / Σ|λ|
    """

def top_k_coverage(eigenvalues: Tensor, k: int = 5) -> float:
    """
    Fraction of total eigenvalue mass in top-k.
    Higher = more low-rank structure.

    Formula: Σ_{i=1}^k |λ_i| / Σ |λ|
    """

def extract_eigenspectrum(model) -> dict:
    """
    Wrapper around model.decompose() that computes all metrics.

    Returns:
        {
            'eigenvalues': Tensor[n_classes, d_hidden],
            'eigenvectors': Tensor[n_classes, d_hidden, d_input],
            'effective_rank_per_class': Tensor[n_classes],
            'mean_effective_rank': float,
            'top5_coverage_per_class': Tensor[n_classes],
        }
    """
```

---

### 4.4 `src/train.py`

**Purpose**: Main training script with full experiment tracking.

**CLI Arguments**:
```
--config PATH       Path to YAML config file
--seed INT          Random seed (default: 42)
--device STR        Device: 'cuda', 'cpu', 'mps' (default: 'cuda')
--wandb-project STR wandb project name (default: 'fact-bilinear')
--no-wandb          Disable wandb logging
--checkpoint-dir    Where to save checkpoints
```

**Workflow**:
```python
def main():
    # 1. Parse args, load config
    # 2. Set random seeds (torch, numpy)
    # 3. Initialize wandb run
    # 4. Start codecarbon tracker
    # 5. Load data (MNIST)
    # 6. Create model from config
    # 7. Create transform (if noise_std > 0)
    # 8. Train model (model.fit)
    # 9. Extract eigenspectrum metrics
    # 10. Log final metrics to wandb
    # 11. Stop codecarbon, log CO2
    # 12. Save checkpoint
```

**Config Loading**:
```python
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

---

### 4.5 Config Files (`configs/*.yaml`)

**Schema**:
```yaml
# Experiment identification
name: mnist_dense_none
experiment_id: P1.1

# Model architecture
model:
  mode: dense           # 'dense' or 'cp'
  d_hidden: 256
  n_layer: 1
  rank: null            # Only for CP mode

# Training
training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

# Regularization
regularization:
  noise_std: 0.0        # Gaussian noise augmentation
  weight_decay: 0.0     # AdamW weight decay

# Data
data:
  dataset: mnist
  device: cuda
```

**Four Configs for Phase 1**:

| File | noise_std | weight_decay | Experiment ID |
|------|-----------|--------------|---------------|
| `mnist_dense_none.yaml` | 0.0 | 0.0 | P1.1 |
| `mnist_dense_noise.yaml` | 0.4 | 0.0 | P1.2 |
| `mnist_dense_wd.yaml` | 0.0 | 0.5 | P1.3 |
| `mnist_dense_full.yaml` | 0.4 | 0.5 | P1.4 |

---

### 4.6 SLURM Job Scripts

**`jobs/train_single.job`** - Single experiment:
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=bilinear_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

python src/train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --checkpoint-dir results/phase1/checkpoints
```

**`jobs/train_array.job`** - All 20 runs (4 configs × 5 seeds):
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=bilinear_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/train_%A_%a.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

# Map array index to config and seed
CONFIGS=(none noise wd full)
SEEDS=(42 43 44 45 46)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

CONFIG_NAME="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Running config: mnist_dense_${CONFIG_NAME}, seed: ${SEED}"

python src/train.py \
    --config "configs/mnist_dense_${CONFIG_NAME}.yaml" \
    --seed "$SEED" \
    --checkpoint-dir "results/phase1/checkpoints"
```

---

## 5. Integration Points

### Import Pattern
All `src/` files should use this pattern to import from original code:

```python
import sys
from pathlib import Path

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

# Now import original modules
from image.model import Model, Config
from image.datasets import MNIST
from shared.components import Bilinear
```

### wandb Integration
```python
import wandb

# Initialize
wandb.init(
    project="fact-bilinear",
    name=f"{config['name']}_seed{seed}",
    config={
        **config,
        "seed": seed,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
)

# Log during training (modify fit loop or use callback)
wandb.log({"train/loss": loss, "train/acc": acc, "epoch": epoch})

# Log final metrics
wandb.summary["final_val_acc"] = final_acc
wandb.summary["effective_rank"] = eff_rank
wandb.summary["co2_kg"] = emissions
```

### codecarbon Integration
```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="fact-bilinear",
    output_dir="results/phase1",
    log_level="warning"  # Reduce verbosity
)
tracker.start()

# ... training ...

emissions_kg = tracker.stop()
wandb.summary["co2_kg"] = emissions_kg
```

---

## 6. Testing Checklist

Before launching all 20 jobs, verify:

- [ ] `python -c "from src.models.bilinear_layer import BilinearDense, BilinearCP"` works
- [ ] Single training run completes: `python src/train.py --config configs/mnist_dense_none.yaml --seed 42`
- [ ] wandb logs appear in project
- [ ] codecarbon saves emissions data
- [ ] Checkpoint file saved correctly
- [ ] `model.decompose()` returns valid eigenvalues/eigenvectors

---

## 7. Expected Outputs

After Phase 1 completion:

```
results/phase1/
├── checkpoints/
│   ├── mnist_dense_none_seed42.pt
│   ├── mnist_dense_none_seed43.pt
│   ├── ... (20 files total)
├── emissions.csv                    # codecarbon output
└── figures/                         # Generated by Person B
    ├── eigenspectrum_comparison.pdf
    └── eigenvectors_grid.pdf

wandb project 'fact-bilinear':
├── 20 runs with full metrics
└── Summary table exportable
```

---

## 8. Prompt for New Claude Instance

Copy this prompt to start implementation:

---

**PROMPT START**

```
I am implementing Phase 1 (Reproduction) of the FACT-AI bilinear interpretability project as Person A (Infrastructure Lead).

Please read the following context files:
- @CLAUDE.md - Project setup and conventions
- @WORKPLAN.md - Full project plan
- @IMPLEMENTATION_GUIDE_PHASE1.md - Detailed implementation specifications

My tasks for Day 2-3:
1. Create directory structure (src/, configs/, jobs/, etc.)
2. Implement src/models/bilinear_layer.py (dense mode wrapping original, CP mode as extension)
3. Implement src/train.py with wandb + codecarbon tracking
4. Create 4 config YAML files for Phase 1 experiments
5. Create SLURM job scripts for Snellius cluster

Key constraints:
- Dense mode MUST wrap the original Bilinear from bilinear-decomposition-main/shared/components.py
- Use jaxtyping for tensor type hints
- All experiments track: GPU model, wall-clock time, CO2 emissions
- 5 seeds per config: [42, 43, 44, 45, 46]

Start by creating the directory structure, then implement bilinear_layer.py first. Show me each file and ask for confirmation before moving to the next.
```

**PROMPT END**

---

## 9. Reference: Paper Baseline Values

From Pearce et al. (2024) Section 4:

| Metric | No Regularization | With Regularization |
|--------|-------------------|---------------------|
| Test Accuracy | ~97-98% | ~94-95% |
| Effective Rank | High (~150-200) | Low (~20-40) |
| Eigenvector Quality | Overfitting patterns | Digit-like features |

**Gate Check Criteria** (Day 7):
- Effective rank ratio (reg/no-reg) < 0.5
- Top eigenvectors visually resemble digits
- Accuracy within 2% of paper values
