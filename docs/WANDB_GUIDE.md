# Weights & Biases (wandb) Guide

wandb is a cloud-based experiment tracking tool similar to TensorBoard, but with a web interface accessible from anywhere.

## Quick Start

### 1. Create Account & Login

```bash
# Install wandb (already in environment.yml)
pip install wandb

# Login (one-time setup)
wandb login
# This opens a browser to get your API key from https://wandb.ai/authorize
```

### 2. Run Experiments with wandb

```bash
# Training WITH wandb logging (requires internet + login)
python src/train.py --config configs/mnist_dense_full.yaml --seed 42

# Training WITHOUT wandb (offline mode, for overnight runs)
python src/train.py --config configs/mnist_dense_full.yaml --seed 42 --no-wandb
```

## Viewing Experiments

### Web Interface

1. Go to https://wandb.ai/itayerlich96-student/fact-bilinear
2. All experiments (vision + language, local + Snellius) are in one project
3. Use **tags** to filter:
   - `vision` + `mnist` or `fashion_mnist` for Section 4 experiments
   - `language` + `sae`, `negation`, or `interaction` for Section 5 experiments
4. You'll see:
   - **Runs table**: All experiments with metrics
   - **Charts**: Auto-generated training curves
   - **Parallel coordinates**: Compare hyperparameters across runs
   - **Custom reports**: Create publication-ready figures

### Key Features

| Feature | Description |
|---------|-------------|
| **Live tracking** | See training progress in real-time |
| **Comparison** | Select multiple runs to compare side-by-side |
| **Hyperparameter sweeps** | Visualize how parameters affect metrics |
| **Artifacts** | Store model checkpoints in the cloud |
| **Reports** | Create shareable dashboards |

## What Gets Logged

### Vision Experiments (tags: `vision`, `mnist`/`fashion_mnist`)

```python
# Config logged automatically:
wandb.config = {
    "model.d_hidden": 256,
    "regularization.noise_std": 0.4,
    "regularization.weight_decay": 0.5,
    "seed": 42,
    "gpu": "Apple M4 Pro",
}

# Per-epoch metrics (training curves):
wandb.log({
    "epoch": 0,
    "train/acc": 0.85,
    "val/acc": 0.84,
    "train/loss": 0.45,
    "val/loss": 0.48,
}, step=0)

# Final summary metrics:
wandb.summary = {
    # Accuracy
    "final_train_acc": 0.95,
    "final_val_acc": 0.94,

    # Spectral metrics (for paper Figure 4)
    "effective_rank": 35.2,           # Mean across classes
    "effective_rank_mean": 35.2,
    "effective_rank_std": 5.1,
    "top5_coverage_mean": 0.72,       # Variance explained by top-5 eigenvalues
    "top10_coverage_mean": 0.89,      # Variance explained by top-10 eigenvalues
    "decay_rate_mean": 0.65,          # λ₂/λ₁ ratio

    # Per-class effective rank
    "effective_rank_class_0": 32.1,
    "effective_rank_class_1": 38.5,
    # ... (classes 0-9)

    # Eigenvalue histograms (first 3 classes)
    "eigenvalues_class_0": wandb.Histogram(...),

    # Resource usage
    "wall_time_hours": 0.5,
    "gpu_hours": 0.5,
    "co2_kg": 0.001,
}
```

### Language Experiments (tags: `language`, `sae`/`negation`/`interaction`)

**SAE Training** (tags: `language`, `sae`):
- Config: expansion, k, layer, point, lr, n_buffers
- Summary: wall_time_hours, gpu_hours, co2_kg

**Negation Discovery** (tags: `language`, `negation`):
```python
wandb.summary = {
    "n_samples": 50000,
    "not_positive_feature": 751,      # Feature ID for "not + positive"
    "not_negative_feature": 3834,     # Feature ID for "not + negative"
    "cosine_similarity": -0.42,       # Should be negative (opposing directions)
    "opposing_directions": True,
    "wall_time_hours": 0.8,
    "co2_kg": 0.002,
}
```

**Interaction Analysis** (tags: `language`, `interaction`):
```python
wandb.summary = {
    "n_features": 500,
    "fraction_above_075": 0.71,       # Paper claims 69%
    "fraction_above_050": 0.89,
    "mean_correlation": 0.68,
    "mean_effective_rank": 4.2,
    "claim_supported": True,
    "wall_time_hours": 1.5,
    "co2_kg": 0.003,
}
```

## Offline Mode

For overnight runs without internet, use `--no-wandb`. Results are saved to:

- **Vision checkpoints**: `results/vision/checkpoints/*.pt`
- **Language results**: `results/language/*.json`

### Analyzing Offline Results

```python
import torch
import json

# Load vision checkpoint
ckpt = torch.load("results/vision/checkpoints/mnist_dense_full_seed42.pt")
print(f"Val Accuracy: {ckpt['metrics']['val_acc']:.4f}")
print(f"Effective Rank: {ckpt['metrics']['effective_rank']:.1f}")

# Load language results
with open("results/language/negation_analysis.json") as f:
    results = json.load(f)
print(f"Top negation feature: {results['not_positive_features'][0]}")
```

## Syncing Offline Runs Later

If you ran with `--no-wandb` but want to upload results later:

```bash
# wandb can sync offline runs if you set WANDB_MODE=offline initially
# This project uses --no-wandb which skips logging entirely
# To upload results, you'd need to re-run or write a sync script
```

## Creating Comparison Plots

### In wandb Web UI

1. Go to project page
2. Select runs you want to compare (checkboxes)
3. Click "Add to report" or use the Charts tab
4. Available chart types:
   - Line charts (training curves)
   - Scatter plots (accuracy vs effective rank)
   - Bar charts (final metrics comparison)
   - Parallel coordinates (hyperparameter impact)

### Local Analysis (Alternative)

Since overnight runs use `--no-wandb`, use this script for local analysis:

```python
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Load all checkpoints
checkpoints = list(Path("results/vision/checkpoints").glob("*.pt"))
results = []

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    results.append({
        'name': ckpt_path.stem,
        'config': ckpt['config'],
        'val_acc': ckpt['metrics']['val_acc'],
        'effective_rank': ckpt['metrics']['effective_rank'],
    })

# Plot accuracy vs effective rank
plt.figure(figsize=(10, 6))
for r in results:
    marker = 'o' if 'full' in r['name'] else 's'
    color = 'green' if 'full' in r['name'] else 'gray'
    plt.scatter(r['effective_rank'], r['val_acc'], marker=marker, c=color, s=100)

plt.xlabel('Effective Rank')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy vs Interpretability (Lower Rank = More Interpretable)')
plt.savefig('results/accuracy_vs_rank.pdf')
```

## CO2 Emissions Tracking

All experiments track CO2 emissions via [codecarbon](https://codecarbon.io/). This data is automatically logged to wandb:

### Metrics Logged

| Metric | Description |
|--------|-------------|
| `wall_time_hours` | Total runtime in hours |
| `gpu_hours` | GPU time (wall_time * gpu_count) |
| `co2_kg` | CO2 emissions in kilograms |

### Analyzing CO2 in wandb

1. Go to the project: https://wandb.ai/itayerlich96-student/fact-bilinear
2. In the Runs table, add columns: `co2_kg`, `wall_time_hours`, `gpu_hours`
3. Create charts:
   - **Bar chart**: CO2 by experiment type (group by tags)
   - **Scatter**: Accuracy vs CO2 (efficiency analysis)
   - **Table**: Total CO2 by regularization config

### Example: Total CO2 Report

In wandb, create a Report with:
```
# CO2 Emissions Summary

## By Experiment Type
- Vision (MNIST + Fashion-MNIST): Filter by tag "vision"
- Language (SAE + Negation + Interaction): Filter by tag "language"

## By Regularization
Group vision runs by config name to compare CO2 across:
- none (baseline)
- noise (noise augmentation)
- wd (weight decay)
- full (both)
```

## Tips

1. **Use wandb for development**: Real-time feedback is valuable
2. **Use --no-wandb for batch runs**: Avoids network dependency
3. **Compare regularization effects**: Group runs by noise_std and weight_decay
4. **Track CO2**: Required for FACT-AI report - all runs log `co2_kg` to wandb summary

## Project URL

All experiments (vision + language, local + Snellius) log to a single project:

**https://wandb.ai/itayerlich96-student/fact-bilinear**

Filter by tags to view specific experiment types. Settings are in `src/utils.py`:
- `WANDB_ENTITY = "itayerlich96-student"`
- `WANDB_PROJECT = "fact-bilinear"`
