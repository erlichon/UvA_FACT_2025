# Claude Code Prompt: Person D (CP Rank Sweep Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 2 Extension 3** of a FACT-AI course project as **Person D (CP Rank Sweep Lead)**.

### Project Summary
We are reproducing and extending "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417). Phase 1 (reproduction) and Extensions 1-2 are complete. You are now working on:
- **Extension 3**: CP Rank Sweep - Train CP-bilinear models with varying ranks on MNIST

### Core Hypothesis
> **Structural low-rank** (CP-Decomposition with explicit rank R) produces equivalent or better interpretability than **emergent low-rank** (via regularization), with the advantage of explicit rank control and NO need for noise augmentation.

### Your Role
- Create training configs for CP rank sweep (R = 8, 16, 32, 64, 128, 256)
- Run 30 experiments (6 ranks x 5 seeds) on Snellius
- Extract eigenspectrum and interpretability metrics for all models
- Compare CP models to Phase 1 dense models
- Provide results to Person E for final synthesis

### Timeline
- **Days 12-14** (Jan 19-21): Run CP rank sweep experiments
- **Day 14**: Handoff preliminary results to Person E

### Dependencies
- **Person F** provides: Working `BilinearCP` class and `CPImageModel` (verified via XOR gate)
- **Person A** provides: Training infrastructure (`src/train.py` pattern)
- **Person B** provides: Analysis utilities (`src/analysis/spectral.py`)
- Phase 1 checkpoints: `results/phase1/checkpoints/` (for comparison)

---

## CRITICAL CONSTRAINTS

1. **USE CP IMPLEMENTATION FROM PERSON F**: Import from `src/models/bilinear_layer.py` and `src/models/cp_model.py`. Do NOT reimplement.

2. **MINIMAL REGULARIZATION**: CP models should train with:
   - `noise_std = 0.0` (NO noise augmentation)
   - `weight_decay = 0.1` (minimal, just for stability)

   This tests whether structural rank alone provides interpretability.

3. **5 SEEDS**: All experiments use seeds `[42, 43, 44, 45, 46]`

4. **ENVIRONMENTAL TRACKING**: All runs must log CO2 via codecarbon to wandb

5. **CHECKPOINT FORMAT**: Save checkpoints compatible with Person B's analysis code

---

## VERIFICATION & BUDGET REQUIREMENTS

> **IMPORTANT**: The team has a total budget of **25,000 SBUs** on Snellius. Jobs must be submitted carefully.

### Local Verification (MANDATORY)

Before submitting ANY job to Snellius, you MUST verify code works locally:

```bash
# Test CP training with 2 epochs locally (should complete in <2 min on CPU)
python src/train_cp.py \
    --rank 32 \
    --seed 42 \
    --no-wandb \
    --epochs 2 \
    --device cpu

# Verify checkpoint saved correctly
python -c "
import torch
ckpt = torch.load('results/phase2/checkpoints/mnist_cp_r32_seed42.pt')
print('Config:', ckpt['config'])
print('Eigenvalues shape:', ckpt['eigenvalues'].shape)
print('Metrics:', ckpt['metrics'])
"
```

**Verification Checklist**:
- [ ] `train_cp.py` runs without errors
- [ ] Checkpoint saves with correct format
- [ ] Eigendecomposition produces expected shapes
- [ ] Metrics are computed and logged

### Job Submission Protocol

1. **Claude outputs job files ONLY** - The human team member will submit jobs manually
2. **Never auto-submit** - Always show the job file content and wait for human approval
3. **Test single rank first** - Run R=32 before submitting array job
4. **Monitor SBU usage** - Track consumption in wandb

### Budget Tracking

| Phase | Experiments | Est. SBUs | Running Total |
|-------|-------------|-----------|---------------|
| P1 (Persons A+B) | 20 runs | 210 | 210 |
| E1 (Person C) + E2 (Person F) | ~10 runs | 20 | 230 |
| E3 CP Sweep Test | 1 run | 10 | 240 |
| E3 CP Sweep Full | 30 runs | 300 | 540 |
| E3 Buffer | debugging | 50 | 590 |

*Total Phase 2 (E3): ~350 SBUs estimated*

---

## EXPERIMENT MATRIX

### CP Rank Sweep (30 runs total)

| Config ID | Rank | Noise | WD | Seeds | Est. GPU Hrs | Est. CO2 (kg) |
|-----------|------|-------|-----|-------|--------------|---------------|
| E3.1 | 8 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| E3.2 | 16 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| E3.3 | 32 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| E3.4 | 64 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| E3.5 | 128 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| E3.6 | 256 | 0.0 | 0.1 | 5 | 0.8 | 0.10 |
| **Total** | | | | **30** | **4.8** | **0.60** |

### Comparison Points
After training, compare to Phase 1 baselines:

| Model | Noise | WD | Expected Eff. Rank |
|-------|-------|-----|-------------------|
| Dense (no reg) | 0.0 | 0.0 | ~150-200 (high) |
| Dense (full reg) | 0.4 | 0.5 | ~20-40 (low) |
| CP R=32 (no reg) | 0.0 | 0.1 | ~32 (structural) |
| CP R=64 (no reg) | 0.0 | 0.1 | ~64 (structural) |

**Key Question**: Can CP mode achieve Dense(full reg) interpretability WITHOUT noise?

---

## FILE SPECIFICATIONS

### 1. Training Script for CP Models

**`src/train_cp.py`**:
```python
"""
Training script for CP-Bilinear models.

Usage:
    python src/train_cp.py --rank 32 --seed 42
    python src/train_cp.py --config configs/mnist_cp_r32.yaml --seed 42
"""

import sys
from pathlib import Path
import argparse
import time
import yaml
import torch
import wandb
from codecarbon import EmissionsTracker

sys.path.insert(0, str(Path(__file__).parent.parent / "bilinear-decomposition-main"))

from image.datasets import MNIST
from src.models.cp_model import CPImageModel
from src.analysis.spectral import compute_all_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train CP-Bilinear Model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--rank", type=int, default=32, help="CP rank")
    parser.add_argument("--d-hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wandb-project", type=str, default="fact-bilinear")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase2/checkpoints")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        rank = config['model']['rank']
        d_hidden = config['model'].get('d_hidden', 256)
        epochs = config['training']['epochs']
        lr = config['training'].get('lr', 1e-3)
        weight_decay = config['regularization']['weight_decay']
        config_name = Path(args.config).stem
    else:
        rank = args.rank
        d_hidden = args.d_hidden
        epochs = args.epochs
        lr = args.lr
        weight_decay = args.weight_decay
        config_name = f"mnist_cp_r{rank}"

    # Set seed
    set_seed(args.seed)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{config_name}_seed{args.seed}",
            config={
                'mode': 'cp',
                'rank': rank,
                'd_hidden': d_hidden,
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'noise_std': 0.0,  # NO noise for CP experiments
                'seed': args.seed,
                'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
            }
        )

    # Start CO2 tracking
    tracker = EmissionsTracker(project_name="fact-bilinear", log_level="warning")
    tracker.start()
    start_time = time.time()

    # Load data
    train_data = MNIST(train=True, device=args.device)
    test_data = MNIST(train=False, device=args.device)

    # Create model
    model = CPImageModel(d_hidden=d_hidden, rank=rank, n_classes=10)
    model = model.to(args.device)

    print(f"Training CP model with rank={rank}, seed={args.seed}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train (NO transform/noise)
    history = model.fit(
        train_data, test_data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        transform=None,  # NO noise augmentation
        verbose=True
    )

    # Get final metrics
    final_train_acc = history['train_acc'].iloc[-1]
    final_val_acc = history['val_acc'].iloc[-1]

    # Stop tracking
    end_time = time.time()
    emissions_kg = tracker.stop()
    wall_time_hours = (end_time - start_time) / 3600
    gpu_hours = wall_time_hours * max(1, torch.cuda.device_count())

    # Eigendecomposition
    model.eval()
    eigenvalues, eigenvectors = model.decompose()
    metrics = compute_all_metrics(eigenvalues)

    # Log to wandb
    if not args.no_wandb:
        wandb.summary["final_train_acc"] = final_train_acc
        wandb.summary["final_val_acc"] = final_val_acc
        wandb.summary["effective_rank"] = metrics['effective_rank_mean']
        wandb.summary["top5_coverage"] = metrics['top5_coverage_mean']
        wandb.summary["top10_coverage"] = metrics['top10_coverage_mean']
        wandb.summary["wall_time_hours"] = wall_time_hours
        wandb.summary["gpu_hours"] = gpu_hours
        wandb.summary["co2_kg"] = emissions_kg
        wandb.finish()

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'config': {
            'mode': 'cp',
            'rank': rank,
            'd_hidden': d_hidden,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'noise_std': 0.0,
        },
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'effective_rank': metrics['effective_rank_mean'],
            'top5_coverage': metrics['top5_coverage_mean'],
            'top10_coverage': metrics['top10_coverage_mean'],
        },
        'seed': args.seed,
        'eigenvalues': eigenvalues.cpu(),
        'eigenvectors': eigenvectors.cpu(),
    }

    checkpoint_path = checkpoint_dir / f"{config_name}_seed{args.seed}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"CP Model (rank={rank}, seed={args.seed})")
    print(f"Final Val Accuracy: {final_val_acc:.4f}")
    print(f"Effective Rank: {metrics['effective_rank_mean']:.2f}")
    print(f"Top-5 Coverage: {metrics['top5_coverage_mean']:.4f}")
    print(f"GPU Hours: {gpu_hours:.3f}")
    print(f"CO2 (kg): {emissions_kg:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

### 2. Config Files

**`configs/mnist_cp_r8.yaml`**:
```yaml
name: mnist_cp_r8
experiment_id: E3.1

model:
  mode: cp
  rank: 8
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0    # NO noise for CP
  weight_decay: 0.1  # Minimal

data:
  dataset: mnist
```

**`configs/mnist_cp_r16.yaml`**:
```yaml
name: mnist_cp_r16
experiment_id: E3.2

model:
  mode: cp
  rank: 16
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.1

data:
  dataset: mnist
```

**`configs/mnist_cp_r32.yaml`**:
```yaml
name: mnist_cp_r32
experiment_id: E3.3

model:
  mode: cp
  rank: 32
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.1

data:
  dataset: mnist
```

**`configs/mnist_cp_r64.yaml`**:
```yaml
name: mnist_cp_r64
experiment_id: E3.4

model:
  mode: cp
  rank: 64
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.1

data:
  dataset: mnist
```

**`configs/mnist_cp_r128.yaml`**:
```yaml
name: mnist_cp_r128
experiment_id: E3.5

model:
  mode: cp
  rank: 128
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.1

data:
  dataset: mnist
```

**`configs/mnist_cp_r256.yaml`**:
```yaml
name: mnist_cp_r256
experiment_id: E3.6

model:
  mode: cp
  rank: 256
  d_hidden: 256

training:
  epochs: 100
  lr: 0.001
  batch_size: 2048

regularization:
  noise_std: 0.0
  weight_decay: 0.1

data:
  dataset: mnist
```

### 3. SLURM Job Scripts

**`jobs/train_cp_single.job`**:
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=cp_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=logs/cp_train_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

echo "Training CP model"
echo "Rank: $RANK"
echo "Seed: $SEED"

python src/train_cp.py \
    --rank "$RANK" \
    --seed "$SEED" \
    --checkpoint-dir results/phase2/checkpoints
```

**`jobs/train_cp_array.job`** (all 30 runs):
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=cp_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/cp_sweep_%A_%a.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

# Map array index to rank and seed
RANKS=(8 16 32 64 128 256)
SEEDS=(42 43 44 45 46)

RANK_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

RANK="${RANKS[$RANK_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Rank: $RANK"
echo "Seed: $SEED"

python src/train_cp.py \
    --rank "$RANK" \
    --seed "$SEED" \
    --checkpoint-dir results/phase2/checkpoints
```

### 4. Analysis Notebook

**`notebooks/02b_cp_sweep.ipynb`** structure:
```markdown
# Extension 3: CP Rank Sweep Analysis

## 1. Setup
```python
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis.spectral import extract_from_checkpoint, compute_all_metrics
from src.analysis.visualization import (
    plot_eigenspectrum_comparison,
    plot_eigenvectors_grid,
    plot_accuracy_vs_effective_rank,
)

CP_CHECKPOINT_DIR = "../results/phase2/checkpoints"
P1_CHECKPOINT_DIR = "../results/phase1/checkpoints"
FIGURE_DIR = "../results/phase2/figures"
```

## 2. Load All CP Checkpoints
```python
ranks = [8, 16, 32, 64, 128, 256]
seeds = [42, 43, 44, 45, 46]

cp_results = []
for rank in ranks:
    for seed in seeds:
        path = f"{CP_CHECKPOINT_DIR}/mnist_cp_r{rank}_seed{seed}.pt"
        data = extract_from_checkpoint(path)
        cp_results.append({
            'mode': 'cp',
            'rank': rank,
            'seed': seed,
            'accuracy': data['metrics']['val_acc'],
            'effective_rank': data['metrics']['effective_rank'],
        })

cp_df = pd.DataFrame(cp_results)
```

## 3. Load Phase 1 Baselines for Comparison
```python
p1_results = []
for config in ['none', 'full']:
    for seed in seeds:
        path = f"{P1_CHECKPOINT_DIR}/mnist_dense_{config}_seed{seed}.pt"
        data = extract_from_checkpoint(path)
        metrics = compute_all_metrics(data['eigenvalues'])
        p1_results.append({
            'mode': f'dense_{config}',
            'rank': None,
            'seed': seed,
            'accuracy': data['metrics']['val_acc'],
            'effective_rank': metrics['effective_rank_mean'],
        })

p1_df = pd.DataFrame(p1_results)
```

## 4. Accuracy vs CP Rank
```python
# Plot accuracy as function of rank
fig, ax = plt.subplots(figsize=(10, 6))

cp_agg = cp_df.groupby('rank').agg({'accuracy': ['mean', 'std']}).reset_index()
cp_agg.columns = ['rank', 'acc_mean', 'acc_std']

ax.errorbar(cp_agg['rank'], cp_agg['acc_mean'] * 100,
            yerr=cp_agg['acc_std'] * 100, fmt='o-', capsize=5,
            label='CP (no noise)', markersize=10)

# Add dense baselines as horizontal lines
dense_none_acc = p1_df[p1_df['mode'] == 'dense_none']['accuracy'].mean() * 100
dense_full_acc = p1_df[p1_df['mode'] == 'dense_full']['accuracy'].mean() * 100

ax.axhline(dense_none_acc, color='red', linestyle='--', label='Dense (no reg)')
ax.axhline(dense_full_acc, color='green', linestyle='--', label='Dense (full reg)')

ax.set_xlabel('CP Rank')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Accuracy vs CP Rank')
ax.legend()
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)

plt.savefig(f"{FIGURE_DIR}/cp_accuracy_vs_rank.pdf")
```

## 5. Effective Rank vs CP Rank
```python
# Does effective rank track CP rank?
fig, ax = plt.subplots(figsize=(10, 6))

eff_agg = cp_df.groupby('rank').agg({'effective_rank': ['mean', 'std']}).reset_index()
eff_agg.columns = ['rank', 'eff_mean', 'eff_std']

ax.errorbar(eff_agg['rank'], eff_agg['eff_mean'],
            yerr=eff_agg['eff_std'], fmt='s-', capsize=5,
            label='CP effective rank', markersize=10)

# Ideal: effective_rank = cp_rank
ax.plot([8, 256], [8, 256], 'k--', alpha=0.5, label='Ideal (eff_rank = cp_rank)')

# Dense baselines
dense_none_eff = p1_df[p1_df['mode'] == 'dense_none']['effective_rank'].mean()
dense_full_eff = p1_df[p1_df['mode'] == 'dense_full']['effective_rank'].mean()

ax.axhline(dense_none_eff, color='red', linestyle='--', label='Dense (no reg)')
ax.axhline(dense_full_eff, color='green', linestyle='--', label='Dense (full reg)')

ax.set_xlabel('CP Rank')
ax.set_ylabel('Effective Rank')
ax.set_title('Effective Rank vs CP Rank')
ax.legend()
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.grid(True, alpha=0.3)

plt.savefig(f"{FIGURE_DIR}/cp_effrank_vs_rank.pdf")
```

## 6. CP Rank Sweep Summary Figure
```python
# Main figure for report: Accuracy vs Effective Rank with all models
fig, ax = plt.subplots(figsize=(10, 8))

# CP models by rank
colors = plt.cm.viridis(np.linspace(0, 1, len(ranks)))
for i, rank in enumerate(ranks):
    subset = cp_df[cp_df['rank'] == rank]
    ax.scatter(subset['effective_rank'], subset['accuracy'] * 100,
               c=[colors[i]], s=80, alpha=0.6, label=f'CP R={rank}')

# Dense baselines
dense_none = p1_df[p1_df['mode'] == 'dense_none']
dense_full = p1_df[p1_df['mode'] == 'dense_full']

ax.scatter(dense_none['effective_rank'], dense_none['accuracy'] * 100,
           c='red', marker='x', s=100, label='Dense (no reg)')
ax.scatter(dense_full['effective_rank'], dense_full['accuracy'] * 100,
           c='green', marker='+', s=100, label='Dense (full reg)')

ax.set_xlabel('Effective Rank (lower = more interpretable)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('CP Rank Sweep: Accuracy vs Interpretability')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/cp_rank_sweep.pdf")
```

## 7. Eigenvector Visualization
```python
# Compare eigenvectors: Dense (full reg) vs CP R=32
dense_data = extract_from_checkpoint(f"{P1_CHECKPOINT_DIR}/mnist_dense_full_seed42.pt")
cp32_data = extract_from_checkpoint(f"{CP_CHECKPOINT_DIR}/mnist_cp_r32_seed42.pt")

fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for c in range(10):
    # Dense eigenvector
    vec_dense = dense_data['eigenvectors'][c, 0].numpy().reshape(28, 28)
    axes[0, c].imshow(vec_dense, cmap='RdBu')
    axes[0, c].axis('off')
    if c == 0:
        axes[0, c].set_ylabel('Dense\n(full reg)', fontsize=12)

    # CP eigenvector
    vec_cp = cp32_data['eigenvectors'][c, 0].numpy().reshape(28, 28)
    axes[1, c].imshow(vec_cp, cmap='RdBu')
    axes[1, c].axis('off')
    if c == 0:
        axes[1, c].set_ylabel('CP R=32\n(no noise)', fontsize=12)

    axes[0, c].set_title(f'{c}', fontsize=10)

plt.suptitle('Top Eigenvector per Class: Dense vs CP', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/eigenvector_comparison_dense_cp.pdf")
```

## 8. Summary Table
```python
# Create summary table for report
summary_data = []

# Dense baselines
for mode in ['dense_none', 'dense_full']:
    subset = p1_df[p1_df['mode'] == mode]
    summary_data.append({
        'Model': mode.replace('_', ' ').title(),
        'Noise': '0.4' if 'full' in mode else '0.0',
        'Acc (mean)': f"{subset['accuracy'].mean()*100:.1f}%",
        'Acc (std)': f"{subset['accuracy'].std()*100:.1f}%",
        'Eff Rank (mean)': f"{subset['effective_rank'].mean():.1f}",
    })

# CP models
for rank in ranks:
    subset = cp_df[cp_df['rank'] == rank]
    summary_data.append({
        'Model': f'CP R={rank}',
        'Noise': '0.0',
        'Acc (mean)': f"{subset['accuracy'].mean()*100:.1f}%",
        'Acc (std)': f"{subset['accuracy'].std()*100:.1f}%",
        'Eff Rank (mean)': f"{subset['effective_rank'].mean():.1f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_markdown(index=False))
```

## 9. Key Findings
- Document whether CP achieves interpretability without noise
- Identify optimal rank for accuracy-interpretability trade-off
- Note any surprising results
```

---

## EXPECTED OUTPUTS

### Files to Create
```
src/
└── train_cp.py                    # CP training script

configs/
├── mnist_cp_r8.yaml
├── mnist_cp_r16.yaml
├── mnist_cp_r32.yaml
├── mnist_cp_r64.yaml
├── mnist_cp_r128.yaml
└── mnist_cp_r256.yaml

jobs/
├── train_cp_single.job
└── train_cp_array.job

results/phase2/
├── checkpoints/
│   ├── mnist_cp_r8_seed42.pt
│   ├── mnist_cp_r8_seed43.pt
│   ├── ... (30 files total)
└── figures/
    ├── cp_accuracy_vs_rank.pdf
    ├── cp_effrank_vs_rank.pdf
    └── cp_rank_sweep.pdf

notebooks/
└── 02b_cp_sweep.ipynb
```

### Handoff to Person E (Day 14)
Provide:
1. All 30 CP checkpoints in `results/phase2/checkpoints/`
2. Preliminary figures: `cp_rank_sweep.pdf`, `cp_accuracy_vs_rank.pdf`
3. Summary table comparing CP to Dense baselines
4. `notebooks/02b_cp_sweep.ipynb` with analysis code

---

## EXECUTION CHECKLIST

### Day 12: Setup and First Runs
- [ ] Verify Person F's CP implementation works (`from src.models.cp_model import CPImageModel`)
- [ ] Create `src/train_cp.py`
- [ ] Create all 6 config files
- [ ] Test single run locally: `python src/train_cp.py --rank 32 --seed 42 --no-wandb`
- [ ] Create SLURM job scripts
- [ ] Submit array job for all 30 runs

### Day 13: Monitor and Debug
- [ ] Check job status: `squeue -u scur0075`
- [ ] Verify checkpoints are being saved
- [ ] Check wandb for metrics

### Day 14: Analysis and Handoff
- [ ] All 30 checkpoints complete
- [ ] Create `notebooks/02b_cp_sweep.ipynb`
- [ ] Generate preliminary figures
- [ ] Create summary table
- [ ] Notify Person E that results are ready

---

## KEY QUESTIONS TO ANSWER

1. **Does CP rank correlate with effective rank?**
   - Expected: Effective rank should roughly match CP rank

2. **Can CP achieve Dense(full reg) interpretability without noise?**
   - Compare: CP R=32 vs Dense(full reg) eigenvectors

3. **What is the accuracy-interpretability trade-off for CP?**
   - Plot: Accuracy vs Effective Rank for all models

4. **Is there an optimal CP rank?**
   - Look for "sweet spot" balancing accuracy and interpretability

---

**Begin by verifying you can import Person C's CP implementation, then create `src/train_cp.py`. Show me each file and ask for confirmation before proceeding.**

## PROMPT END
