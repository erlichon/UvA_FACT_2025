# Claude Code Prompt: Person A (Infrastructure Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 1 (Reproduction)** of a FACT-AI course project as **Person A (Infrastructure Lead)**.

### Project Summary
We are reproducing "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417) for a UvA MSc AI course. The original code exists at `bilinear-decomposition-main/`. We must create our own `src/` directory that wraps (not copies) the original code.

### Your Role
You handle all infrastructure: training pipeline, SLURM jobs, experiment tracking, checkpoint management. Person B (Analysis Lead) will use your checkpoints to generate figures and analysis.

### Timeline
- **Days 2-3**: Implement all infrastructure
- **Day 4-5**: Run all 20 Phase 1 experiments
- **Day 7**: Gate check with Person B

---

## CRITICAL CONSTRAINTS

1. **WRAP, DON'T REIMPLEMENT**: For dense mode, wrap the original `Bilinear` class from `bilinear-decomposition-main/shared/components.py`. This ensures exact reproduction fidelity.

2. **5 SEEDS**: All experiments run with seeds `[42, 43, 44, 45, 46]`

3. **ENVIRONMENTAL TRACKING**: Every run MUST log GPU hours and CO2 via codecarbon to wandb

4. **SNELLIUS SPECIFICS**:
   - Partition: `gpu_a100`
   - Modules: `module load 2025 && module load Anaconda3/2025.06-1`
   - Hostname: `ssh scur0075@Snellius` (capital S)

---

## VERIFICATION & BUDGET REQUIREMENTS

> **IMPORTANT**: The team has a total budget of **25,000 SBUs** on Snellius. Jobs must be submitted carefully.

### Local Verification (MANDATORY)

Before submitting ANY job to Snellius, you MUST verify code works locally with shallow configs:

```bash
# Test with 2 epochs locally (should complete in <1 min on CPU)
python src/train.py \
    --config configs/mnist_dense_none.yaml \
    --seed 42 \
    --no-wandb \
    --epochs 2  # Override to 2 epochs for testing
```

**Verification Checklist**:
- [ ] Training loop runs without errors
- [ ] Checkpoint saves correctly
- [ ] Eigendecomposition runs
- [ ] Metrics are computed
- [ ] Output format matches specification

### Job Submission Protocol

1. **Claude outputs job files ONLY** - The human team member will submit jobs manually
2. **Never auto-submit** - Always show the job file content and wait for human approval
3. **Test single job first** - Before array jobs, test with one seed
4. **Monitor usage** - Track SBU consumption in wandb

### Budget Tracking

| Phase | Experiments | Est. SBUs | Running Total |
|-------|-------------|-----------|---------------|
| P1 Test | 1 run | 10 | 10 |
| P1 Full | 20 runs | 200 | 210 |
| Phase 1 Buffer | debugging | 50 | 260 |

*Estimated SBU per run: ~10 (1 GPU-hour at 10 SBU/GPU-hour)*

---

## DIRECTORY STRUCTURE TO CREATE

```
UvA_FACT_2025/
├── bilinear-decomposition-main/   # UNTOUCHED - original paper code
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilinear_layer.py      # Dense wrapper + CP extension
│   │   └── image_model.py         # Wrapper with tracking
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── spectral.py            # Effective rank (coordinate with Person B)
│   └── train.py                   # Main training script
├── configs/
│   ├── mnist_dense_none.yaml      # P1.1: no regularization
│   ├── mnist_dense_noise.yaml     # P1.2: noise only
│   ├── mnist_dense_wd.yaml        # P1.3: weight decay only
│   └── mnist_dense_full.yaml      # P1.4: noise + weight decay
├── jobs/
│   ├── train_single.job           # Single experiment
│   └── train_array.job            # All 20 runs
├── results/
│   └── phase1/
│       └── checkpoints/
└── logs/                          # SLURM output logs
```

---

## FILE SPECIFICATIONS

### 1. `src/models/bilinear_layer.py`

**Purpose**: Unified layer supporting original dense mode AND our CP extension.

```python
"""
Bilinear Layer: Dense (Original) + CP-Decomposition (Extension)

Dense Mode:
    Wraps original implementation from bilinear-decomposition-main/shared/components.py
    y = gate(W_l @ x) * (W_r @ x)

CP Mode (Phase 2 Extension):
    Explicit rank-R decomposition of the interaction tensor.
    y = ((x @ A) * (x @ B) * lambdas) @ C.T
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from shared.components import Bilinear as OriginalBilinear


class BilinearDense(nn.Module):
    """
    Wrapper around original Bilinear layer for exact reproduction.

    This ensures we use the exact same implementation as the paper.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str = None):
        super().__init__()
        self._original = OriginalBilinear(d_in, d_out, bias=bias, gate=gate)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self._original(x)

    @property
    def w_l(self) -> Float[Tensor, "d_out d_in"]:
        return self._original.w_l

    @property
    def w_r(self) -> Float[Tensor, "d_out d_in"]:
        return self._original.w_r

    @property
    def weight(self):
        return self._original.weight


class BilinearCP(nn.Module):
    """
    CP-Decomposed Bilinear Layer (Phase 2 Extension).

    Explicitly parameterizes the interaction tensor with rank R:
    B[i,j,k] = sum_r lambda[r] * A[i,r] * B[j,r] * C[k,r]

    Args:
        d_in: Input dimension
        d_out: Output dimension
        rank: CP decomposition rank
        bias: Include bias term (not implemented for CP)
    """

    def __init__(self, d_in: int, d_out: int, rank: int, bias: bool = False):
        super().__init__()
        if bias:
            raise NotImplementedError("Bias not supported for CP mode")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # CP factors
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.C = nn.Parameter(torch.randn(d_out, rank) * 0.02)
        self.lambdas = nn.Parameter(torch.ones(rank))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # x: [..., d_in]
        left = x @ self.A          # [..., rank]
        right = x @ self.B         # [..., rank]
        hidden = left * right * self.lambdas  # [..., rank]
        return hidden @ self.C.T   # [..., d_out]

    @property
    def w_l(self) -> Float[Tensor, "d_out d_in"]:
        """Reconstruct W_l for compatibility with analysis code."""
        # W_l[o, i] = sum_r lambda[r] * C[o,r] * A[i,r]
        return (self.C * self.lambdas) @ self.A.T

    @property
    def w_r(self) -> Float[Tensor, "d_out d_in"]:
        """Reconstruct W_r for compatibility with analysis code."""
        # W_r[o, i] = sum_r C[o,r] * B[i,r]
        return self.C @ self.B.T


def create_bilinear(d_in: int, d_out: int, mode: str = 'dense',
                    rank: int = None, bias: bool = False, gate: str = None):
    """
    Factory function to create appropriate bilinear layer.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        mode: 'dense' (original) or 'cp' (extension)
        rank: CP rank (required if mode='cp')
        bias: Include bias term
        gate: Gating function for dense mode

    Returns:
        BilinearDense or BilinearCP instance
    """
    if mode == 'dense':
        return BilinearDense(d_in, d_out, bias=bias, gate=gate)
    elif mode == 'cp':
        if rank is None:
            raise ValueError("rank required for CP mode")
        return BilinearCP(d_in, d_out, rank=rank, bias=bias)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

---

### 2. `src/train.py`

**Purpose**: Main training script with full experiment tracking.

```python
"""
Training script for Bilinear MLP experiments.

Usage:
    python src/train.py --config configs/mnist_dense_full.yaml --seed 42
"""

import sys
from pathlib import Path
import argparse
import time
import yaml
import torch
import wandb
from codecarbon import EmissionsTracker

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "bilinear-decomposition-main"))

from image.model import Model, Config
from image.datasets import MNIST
import kornia


def load_config(path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: for full determinism, also set:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train Bilinear MLP")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wandb-project", type=str, default="fact-bilinear")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase1/checkpoints")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config_name = Path(args.config).stem

    # Set seed
    set_seed(args.seed)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{config_name}_seed{args.seed}",
            config={
                **config,
                "seed": args.seed,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "gpu_count": torch.cuda.device_count(),
            }
        )

    # Start CO2 tracking
    tracker = EmissionsTracker(
        project_name="fact-bilinear",
        log_level="warning",
    )
    tracker.start()
    start_time = time.time()

    # Load data
    train_data = MNIST(train=True, device=args.device)
    test_data = MNIST(train=False, device=args.device)

    # Create model
    model_config = Config(
        epochs=config['training']['epochs'],
        d_hidden=config['model']['d_hidden'],
        wd=config['regularization']['weight_decay'],
        lr=config['training'].get('lr', 1e-3),
    )
    model = Model(model_config)
    model = model.to(args.device)

    # Create transform (noise augmentation)
    noise_std = config['regularization']['noise_std']
    if noise_std > 0:
        transform = kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    else:
        transform = None

    # Train
    print(f"Training {config_name} with seed {args.seed}...")
    history = model.fit(train_data, test_data, transform=transform)

    # Get final metrics
    # NOTE: Original code may use 'train/acc' or 'train_acc' format - handle both
    def get_col(df, *names):
        for name in names:
            if name in df.columns:
                return df[name].iloc[-1]
        raise KeyError(f"None of {names} found in history columns: {list(df.columns)}")

    final_train_acc = get_col(history, 'train_acc', 'train/acc')
    final_val_acc = get_col(history, 'val_acc', 'val/acc', 'test_acc', 'test/acc')
    final_train_loss = get_col(history, 'train_loss', 'train/loss')
    final_val_loss = get_col(history, 'val_loss', 'val/loss', 'test_loss', 'test/loss')

    # Stop tracking
    end_time = time.time()
    emissions_kg = tracker.stop()
    wall_time_hours = (end_time - start_time) / 3600
    gpu_hours = wall_time_hours * torch.cuda.device_count()

    # Extract eigenspectrum for metrics
    vals, vecs = model.decompose()

    # Compute effective rank (simplified version)
    def effective_rank(eigenvalues):
        p = eigenvalues.abs() / eigenvalues.abs().sum(dim=-1, keepdim=True)
        p = p.clamp(min=1e-10)  # Avoid log(0)
        entropy = -(p * p.log()).sum(dim=-1)
        return entropy.exp()

    eff_rank = effective_rank(vals).mean().item()

    # Log to wandb
    if not args.no_wandb:
        wandb.summary["final_train_acc"] = final_train_acc
        wandb.summary["final_val_acc"] = final_val_acc
        wandb.summary["effective_rank"] = eff_rank
        wandb.summary["wall_time_hours"] = wall_time_hours
        wandb.summary["gpu_hours"] = gpu_hours
        wandb.summary["co2_kg"] = emissions_kg
        wandb.finish()

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Save FLAT config dict for cross-compatibility with all analysis code
    checkpoint = {
        'config': {
            'mode': 'dense',
            'd_hidden': config['model']['d_hidden'],
            'epochs': config['training']['epochs'],
            'lr': config['training'].get('lr', 1e-3),
            'noise_std': config['regularization']['noise_std'],
            'weight_decay': config['regularization']['weight_decay'],
        },
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'effective_rank': eff_rank,
        },
        'seed': args.seed,
        'eigenvalues': vals.cpu(),
        'eigenvectors': vecs.cpu(),
    }

    checkpoint_path = checkpoint_dir / f"{config_name}_seed{args.seed}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Config: {config_name}")
    print(f"Seed: {args.seed}")
    print(f"Final Val Accuracy: {final_val_acc:.4f}")
    print(f"Effective Rank: {eff_rank:.2f}")
    print(f"GPU Hours: {gpu_hours:.3f}")
    print(f"CO2 (kg): {emissions_kg:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

---

### 3. Config Files (`configs/*.yaml`)

**`configs/mnist_dense_none.yaml`** (P1.1):
```yaml
name: mnist_dense_none
experiment_id: P1.1

model:
  mode: dense
  d_hidden: 256
  n_layer: 1

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.0

data:
  dataset: mnist
```

**`configs/mnist_dense_noise.yaml`** (P1.2):
```yaml
name: mnist_dense_noise
experiment_id: P1.2

model:
  mode: dense
  d_hidden: 256
  n_layer: 1

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.5
  weight_decay: 0.0

data:
  dataset: mnist
```

**`configs/mnist_dense_wd.yaml`** (P1.3):
```yaml
name: mnist_dense_wd
experiment_id: P1.3

model:
  mode: dense
  d_hidden: 256
  n_layer: 1

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 1.0

data:
  dataset: mnist
```

**`configs/mnist_dense_full.yaml`** (P1.4):
```yaml
name: mnist_dense_full
experiment_id: P1.4

model:
  mode: dense
  d_hidden: 256
  n_layer: 1

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.5
  weight_decay: 1.0

data:
  dataset: mnist
```

---

### 4. SLURM Job Scripts

**`jobs/train_single.job`**:
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

echo "Starting training..."
echo "Config: $CONFIG"
echo "Seed: $SEED"

python src/train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --checkpoint-dir results/phase1/checkpoints
```

**`jobs/train_array.job`** (all 20 runs):
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

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: mnist_dense_${CONFIG_NAME}"
echo "Seed: $SEED"

python src/train.py \
    --config "configs/mnist_dense_${CONFIG_NAME}.yaml" \
    --seed "$SEED" \
    --checkpoint-dir "results/phase1/checkpoints"
```

---

## EXECUTION CHECKLIST

### Step 1: Create Directory Structure
```bash
mkdir -p src/models src/analysis configs jobs results/phase1/checkpoints logs
touch src/__init__.py src/models/__init__.py src/analysis/__init__.py
```

### Step 2: Create Files
Create each file as specified above. Ask for confirmation after each file.

### Step 3: Local Test (before cluster)
```bash
# Test imports work
python -c "from src.models.bilinear_layer import BilinearDense, BilinearCP; print('OK')"

# Test single training run (short)
python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb
```

### Step 4: Cluster Setup
```bash
ssh scur0075@Snellius
cd ~/fact-project
module load 2025 && module load Anaconda3/2025.06-1
conda activate fact
```

### Step 5: Submit Jobs
```bash
# Test single job first
sbatch --export=CONFIG=configs/mnist_dense_none.yaml,SEED=42 jobs/train_single.job

# If successful, submit all 20
sbatch jobs/train_array.job
```

### Step 6: Monitor
```bash
squeue -u scur0075
# Check wandb dashboard for metrics
```

---

## COORDINATION WITH PERSON B

### Checkpoint Format (AGREED)
Person B will load checkpoints like this:
```python
checkpoint = torch.load("results/phase1/checkpoints/mnist_dense_full_seed42.pt")
config = checkpoint['config']
model_state = checkpoint['model_state_dict']
eigenvalues = checkpoint['eigenvalues']  # [n_classes, d_hidden]
eigenvectors = checkpoint['eigenvectors']  # [n_classes, d_hidden, d_input]
```

### Expected Outputs After Phase 1
```
results/phase1/checkpoints/
├── mnist_dense_none_seed42.pt
├── mnist_dense_none_seed43.pt
├── ... (20 files total)

wandb project 'fact-bilinear':
├── 20 runs with full metrics
└── Summary table exportable
```

---

## SUCCESS CRITERIA

Before declaring Phase 1 infrastructure complete:
- [ ] All 20 checkpoints saved with eigenvalues/eigenvectors
- [ ] All runs logged to wandb with CO2 metrics
- [ ] Person B can load checkpoints and extract eigenspectrum
- [ ] No hardcoded paths (all use config or args)

---

**Now begin by creating the directory structure, then implement `src/models/bilinear_layer.py` first. Show me the file and ask for confirmation before proceeding.**

## PROMPT END
