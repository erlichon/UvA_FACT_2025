# Claude Code Prompt: Person G (Section 4 Completion + Section 5 Language Infrastructure)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Section 4 completion (Fashion-MNIST)** and **Section 5 (Language/Negation Circuits)** of a FACT-AI course project as **Person G (Language Infrastructure Lead)**.

### Project Summary
We are reproducing "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417) for a UvA MSc AI course. Section 4 (Vision/MNIST) infrastructure exists. You must:
1. Add Fashion-MNIST experiments to complete Section 4
2. Create wrapper scripts for Section 5 (Language) using the **existing original code**

### Your Role
You handle all language infrastructure: TinyStories data, SAE training, negation circuit discovery, interaction matrix analysis. You also complete Section 4 by adding Fashion-MNIST configs.

### Current State
- **Section 4 (Vision)**: ~70% complete
  - MNIST infrastructure: COMPLETE
  - Fashion-MNIST: NOT IMPLEMENTED (you add this)
- **Section 5 (Language)**: Infrastructure EXISTS in original code
  - Need: wrapper scripts, configs, and SLURM jobs

---

## CRITICAL CONSTRAINTS

1. **USE ORIGINAL CODE - IT EXISTS**: The bilinear transformer and SAE code is fully implemented in `bilinear-decomposition-main/`. DO NOT REIMPLEMENT. Wrap and use the existing code.

2. **ENVIRONMENTAL TRACKING**: Every run MUST log GPU hours and CO2 via codecarbon to wandb

3. **SNELLIUS SPECIFICS**:
   - Partition: `gpu_a100`
   - Modules: `module load 2025 && module load Anaconda3/2025.06-1`
   - Hostname: `ssh scur0075@Snellius` (capital S)

4. **OUTPUT JOB FILES**: You MUST output SLURM job files for ALL experiments. The human will submit them.

---

## ORIGINAL CODE INVENTORY (CRITICAL - READ THIS FIRST)

The `bilinear-decomposition-main/` directory contains **complete implementations** for Section 5. Here's what exists:

### Language Model (`language/transformer.py`)
```python
from language.transformer import Transformer, Config

# Load pretrained bilinear transformer from HuggingFace Hub
model = Transformer.from_pretrained("tdooms/ts-medium")  # TinyStories model

# Or create from config
config = Config(
    n_head=4, n_layer=4, n_ctx=256,
    d_model=256, d_hidden=1024,
    bilinear=True,  # Enable bilinear MLPs
    tokenizer="ts-4096",  # TinyStories tokenizer
)
model = Transformer.from_config(config)

# Weight access for interpretability
w_l, w_r = model.w_l, model.w_r  # Left/right bilinear weights per layer
w_p = model.w_p  # MLP output projections
```

### Sparse Autoencoder (`sae/sae.py`)
```python
from sae.sae import SAE, SAEConfig

# SAE config
config = SAEConfig(
    point="mlp_out",      # Hook point for activations
    target="mlp_out",     # Reconstruction target
    expansion=8,          # SAE features = expansion * d_model
    k=32,                 # Top-k sparsity
    d_model=256,
    n_ctx=256,
)

# Create and train SAE
sae = SAE(config)
# sae.fit(model, dataloader)  # Training handled internally

# Or load pretrained
sae = SAE.from_pretrained("tdooms/ts-medium-scope", point="mlp_out", layer=2)
```

### Activation Sampling (`sae/samplers.py`)
```python
from sae.samplers import ShuffleSampler
from language.utils import Sight

sight = Sight(model)
sampler = ShuffleSampler(sight, dataloader, point="mlp_out", layer=2)

for batch in sampler:
    # batch contains shuffled activations for SAE training
    pass
```

### Interaction Matrix Analysis (`sae/tracer.py`)
```python
from sae.tracer import Tracer

# Load SAEs for input and output of a layer
tracer = Tracer(model, layer=2, in_point="mlp_in", out_point="mlp_out")

# Compute interaction tensor Q for output feature idx
Q = tracer.q(output_feature_idx=3834)  # The "not + negative" feature from paper

# Q[i,j] represents interaction between input features i and j
# for producing output feature 3834
```

### Analysis Functions (`sae/functions.py`)
```python
from sae.functions import (
    compute_effective_rank,
    compute_truncated_eigenvalues,
    compute_outliers,
    compute_kurtosis,
)

# These work on Q tensors from Tracer
effective_rank = compute_effective_rank(Q)
```

### Visualization (`sae/visualizer.py`)
```python
from sae.visualizer import Visualizer

vis = Visualizer(model, sae, dataloader)
vis(feature_idx=3834)  # Show top activating examples
vis.show_logit_influence(feature_idx=3834)  # Logit effects
```

### NNSight Integration (`language/utils.py`)
```python
from language.utils import Sight, Vocab

sight = Sight(model)
vocab = Vocab(model.tokenizer)

# Access activations at any point
with sight.trace(input_ids):
    mlp_out = sight["mlp_out", layer=2].save()
```

### What You Need to Implement
The original code provides ALL the building blocks. You need to create:
1. **Wrapper scripts** that use these components for our reproduction
2. **Negation circuit discovery** - identify features 3834/751 (not in original)
3. **Interaction matrix eigendecomposition** - compute rank-2 correlations
4. **Config files** and **SLURM jobs** for experiments

---

## SECTION 4 COMPLETION: FASHION-MNIST

### Task
Add Fashion-MNIST experiments matching MNIST setup (4 configs x 5 seeds = 20 runs).

### Files to Create

**`configs/fashion_dense_none.yaml`**:
```yaml
name: fashion_dense_none
experiment_id: P1.5

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
  dataset: fashion_mnist
```

**`configs/fashion_dense_noise.yaml`** (P1.6): Same but noise_std=0.4
**`configs/fashion_dense_wd.yaml`** (P1.7): Same but weight_decay=0.5
**`configs/fashion_dense_full.yaml`** (P1.8): Both noise_std=0.4 and weight_decay=0.5

### Update src/train.py

Add Fashion-MNIST support:
```python
# In train.py, update data loading section:
if config['data']['dataset'] == 'mnist':
    from image.datasets import MNIST
    train_data = MNIST(train=True, device=args.device)
    test_data = MNIST(train=False, device=args.device)
elif config['data']['dataset'] == 'fashion_mnist':
    from image.datasets import FashionMNIST
    train_data = FashionMNIST(train=True, device=args.device)
    test_data = FashionMNIST(train=False, device=args.device)
```

### SLURM Job File

**`jobs/train_fashion_array.job`**:
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=bilinear_fashion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/fashion_%A_%a.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

CONFIGS=(none noise wd full)
SEEDS=(42 43 44 45 46)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

CONFIG_NAME="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Fashion-MNIST: config=${CONFIG_NAME}, seed=${SEED}"

python src/train.py \
    --config "configs/fashion_dense_${CONFIG_NAME}.yaml" \
    --seed "$SEED" \
    --checkpoint-dir "results/vision/checkpoints"
```

---

## SECTION 5: LANGUAGE INFRASTRUCTURE

### Overview

Section 5 of the paper discovers **negation circuits** in bilinear transformers:
- Train bilinear transformer on TinyStories (or use pretrained)
- Train Sparse Autoencoders (SAEs) around MLP layers
- Discover negation features (3834: "not + negative", 751: "not + positive")
- Analyze interaction matrices between SAE features
- Verify low-rank structure (69% of features with >0.75 rank-2 correlation)

### Directory Structure to Create

```
src/language/
├── __init__.py
├── negation_discovery.py    # Find negation features (NEW - not in original)
├── interaction_analysis.py  # Eigendecompose Q matrices (NEW)
├── run_sae_training.py      # Wrapper for original SAE code
└── run_analysis.py          # Full Section 5 pipeline

configs/
├── language_sae.yaml         # SAE training config
├── language_negation.yaml    # Negation analysis config
└── language_interaction.yaml # Interaction matrix config

jobs/
├── train_sae.job             # Train SAE (uses original code)
├── analyze_negation.job      # Negation circuit discovery
└── analyze_interaction.job   # Low-rank verification
```

**NOTE**: Most infrastructure uses original code from `bilinear-decomposition-main/`. We only create thin wrappers.

---

### FILE SPECIFICATIONS

#### 1. `src/language/__init__.py`
```python
"""
Language module for Section 5 (Negation Circuit Discovery).

This module provides wrapper scripts around the original paper code in
bilinear-decomposition-main/. We do NOT reimplement SAE or transformer code.
"""
```

#### 2. `src/language/run_sae_training.py`

```python
"""
SAE Training wrapper using original paper code.

Usage:
    python src/language/run_sae_training.py --config configs/language_sae.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml
import torch
import wandb
from codecarbon import EmissionsTracker

# Add original code to path
ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))

from language.transformer import Transformer
from sae.sae import SAE, SAEConfig
from sae.samplers import ShuffleSampler
from language.utils import Sight
from datasets import load_dataset


def create_dataloader(tokenizer, batch_size: int, n_samples: int):
    """Create TinyStories dataloader."""
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="results/language")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Init tracking
    if not args.no_wandb:
        wandb.init(project="fact-bilinear-language", name=config["name"], config=config)

    tracker = EmissionsTracker(project_name="fact-bilinear-language", log_level="warning")
    tracker.start()

    # Load pretrained bilinear transformer
    print(f"Loading model: {config['model']['pretrained']}")
    model = Transformer.from_pretrained(config["model"]["pretrained"])
    model = model.to(args.device)

    # Create dataloader
    dataloader = create_dataloader(
        model.tokenizer,
        batch_size=config["training"]["batch_size"],
        n_samples=config["data"]["n_samples"],
    )

    # Create SAE config (using original code)
    sae_config = SAEConfig(
        point=config["sae"]["point"],
        target=config["sae"]["point"],
        expansion=config["sae"]["expansion"],
        k=config["sae"]["k"],
        d_model=model.config.d_model,
        n_ctx=model.config.n_ctx,
        lr=config["training"]["lr"],
        n_batches=config["training"]["n_batches"],
    )

    # Create and train SAE (using original code)
    sae = SAE(sae_config)
    sight = Sight(model)
    sampler = ShuffleSampler(sight, dataloader, point=sae_config.point, layer=config["sae"]["layer"])

    print(f"Training SAE at {sae_config.point}, layer {config['sae']['layer']}")
    sae.fit(model, sampler)

    # Stop tracking
    emissions = tracker.stop()

    if not args.no_wandb:
        wandb.summary["co2_kg"] = emissions
        wandb.finish()

    # Save
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config,
        "sae_state_dict": sae.state_dict(),
        "sae_config": sae_config.__dict__,
    }, checkpoint_dir / f"sae_layer{config['sae']['layer']}.pt")

    print(f"Saved SAE to {checkpoint_dir}")


if __name__ == "__main__":
    main()
```

#### 3. `src/language/negation_discovery.py`

```python
"""
Negation circuit discovery in bilinear transformers.

Uses the original paper's SAE and Visualizer code to find negation features:
- Feature 3834: Activates on "not + negative words" (e.g., "not lost")
- Feature 751: Activates on "not + positive words" (e.g., "not free")

Usage:
    python src/language/negation_discovery.py --config configs/language_negation.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add original code to path
ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))

from language.transformer import Transformer
from sae.sae import SAE
from language.utils import Sight
from datasets import load_dataset


# Negation and sentiment word lists (from paper analysis)
NEGATION_WORDS = ["not", "no", "never", "neither", "none", "without", "hardly", "barely"]
POSITIVE_WORDS = ["good", "happy", "free", "nice", "great", "love", "wonderful", "beautiful", "kind", "joy"]
NEGATIVE_WORDS = ["bad", "sad", "lost", "wrong", "hate", "fear", "terrible", "awful", "angry", "hurt"]


def collect_activations_with_context(model, sae, dataloader, layer: int, device: str, n_samples: int):
    """
    Collect SAE activations along with token context for pattern matching.

    Returns:
        activations: [n_tokens, d_sae] SAE feature activations
        contexts: List of (text, token_positions) for each sample
    """
    sight = Sight(model)
    all_activations = []
    all_contexts = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            if len(all_activations) * batch["input_ids"].shape[0] >= n_samples:
                break

            input_ids = batch["input_ids"].to(device)

            # Get MLP output activations using NNSight
            with sight.trace(input_ids):
                mlp_out = sight["mlp_out", layer].save()

            # Encode through SAE
            _, sae_acts = sae(mlp_out.value.reshape(-1, mlp_out.value.shape[-1]))
            all_activations.append(sae_acts.cpu())

            # Decode tokens for context
            for ids in input_ids:
                tokens = model.tokenizer.decode(ids.tolist())
                all_contexts.append(tokens)

    activations = torch.cat(all_activations, dim=0)[:n_samples]
    return activations, all_contexts


def find_negation_features(activations, contexts, top_k: int = 20):
    """
    Find SAE features that differentially activate on negation patterns.

    Paper identifies:
    - Feature 3834: "not + negative" pattern
    - Feature 751: "not + positive" pattern
    """
    n_samples = len(contexts)
    n_features = activations.shape[1]

    # Classify samples by pattern
    neg_positive_mask = []
    neg_negative_mask = []
    baseline_mask = []

    for i, text in enumerate(contexts):
        text_lower = text.lower()
        has_negation = any(neg in text_lower for neg in NEGATION_WORDS)
        has_positive = any(pos in text_lower for pos in POSITIVE_WORDS)
        has_negative = any(neg in text_lower for neg in NEGATIVE_WORDS)

        if has_negation and has_positive and not has_negative:
            neg_positive_mask.append(i)
        elif has_negation and has_negative and not has_positive:
            neg_negative_mask.append(i)
        elif not has_negation:
            baseline_mask.append(i)

    print(f"Found {len(neg_positive_mask)} 'not+positive' samples")
    print(f"Found {len(neg_negative_mask)} 'not+negative' samples")
    print(f"Found {len(baseline_mask)} baseline samples")

    if len(neg_positive_mask) == 0 or len(neg_negative_mask) == 0:
        raise ValueError("Not enough samples with negation patterns. Need more data.")

    # Compute differential activations
    neg_pos_acts = activations[neg_positive_mask].mean(dim=0)
    neg_neg_acts = activations[neg_negative_mask].mean(dim=0)
    baseline_acts = activations[baseline_mask].mean(dim=0) if baseline_mask else activations.mean(dim=0)

    # Features that activate MORE on pattern vs baseline
    neg_pos_diff = neg_pos_acts - baseline_acts
    neg_neg_diff = neg_neg_acts - baseline_acts

    # Top features for each pattern
    _, neg_pos_top_idx = neg_pos_diff.topk(top_k)
    _, neg_neg_top_idx = neg_neg_diff.topk(top_k)

    return {
        "not_positive_features": neg_pos_top_idx.tolist(),
        "not_negative_features": neg_neg_top_idx.tolist(),
        "not_positive_activations": neg_pos_diff[neg_pos_top_idx].tolist(),
        "not_negative_activations": neg_neg_diff[neg_neg_top_idx].tolist(),
        "n_not_positive_samples": len(neg_positive_mask),
        "n_not_negative_samples": len(neg_negative_mask),
    }


def check_opposing_directions(sae, feature_idx_1: int, feature_idx_2: int) -> float:
    """
    Check if two features form opposing directions in decoder space.

    Paper claim: negation features should have negative cosine similarity.
    """
    # Decoder columns represent feature directions
    dir_1 = sae.w_dec.weight[:, feature_idx_1]
    dir_2 = sae.w_dec.weight[:, feature_idx_2]

    cosine_sim = F.cosine_similarity(dir_1.unsqueeze(0), dir_2.unsqueeze(0)).item()
    return cosine_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/language/negation_analysis.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model: {config['model']['pretrained']}")
    model = Transformer.from_pretrained(config["model"]["pretrained"])
    model = model.to(args.device)

    # Load SAE
    print(f"Loading SAE: {config['sae']['checkpoint']}")
    sae_ckpt = torch.load(config["sae"]["checkpoint"], map_location=args.device)
    sae = SAE(sae_ckpt["sae_config"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae = sae.to(args.device)

    # Create dataloader
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(min(config["analysis"]["n_samples"], len(dataset))))

    def tokenize(examples):
        return model.tokenizer(examples["text"], truncation=True, max_length=256,
                               padding="max_length", return_tensors="pt")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect activations
    activations, contexts = collect_activations_with_context(
        model, sae, dataloader, layer=config["sae"]["layer"],
        device=args.device, n_samples=config["analysis"]["n_samples"]
    )

    # Find negation features
    results = find_negation_features(activations, contexts, top_k=config["analysis"]["top_k"])

    # Check opposing directions for top features
    if results["not_positive_features"] and results["not_negative_features"]:
        feat_pos = results["not_positive_features"][0]
        feat_neg = results["not_negative_features"][0]
        cosine_sim = check_opposing_directions(sae, feat_pos, feat_neg)

        results["top_features"] = {
            "not_positive": feat_pos,
            "not_negative": feat_neg,
            "cosine_similarity": cosine_sim,
            "opposing_directions": cosine_sim < 0,
        }
        print(f"\nTop feature pair: {feat_pos} (not+pos) vs {feat_neg} (not+neg)")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"Opposing directions: {cosine_sim < 0}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
```

#### 4. `src/language/interaction_analysis.py`

```python
"""
Interaction matrix analysis using the original paper's Tracer class.

For each SAE output feature o, the Tracer computes:
    Q = W_dec_out^T @ (W_l ⊙ W_r) @ W_dec_in

where W_dec projects from SAE latent space back to model space.

Paper claim: 69% of features have >0.75 correlation with rank-2 approximation.

Usage:
    python src/language/interaction_analysis.py --config configs/language_interaction.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm

# Add original code to path
ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))

from language.transformer import Transformer
from sae.sae import SAE
from sae.tracer import Tracer
from sae.functions import compute_effective_rank


def rank_k_approximation_correlation(Q: torch.Tensor, k: int = 2) -> float:
    """
    Compute correlation between Q and its rank-k approximation.

    Q_hat_k = sum_{i=1}^k lambda_i * v_i * v_i^T

    Paper claims 69% of features have >0.75 correlation with k=2.
    """
    # Symmetrize
    Q_sym = (Q + Q.T) / 2

    # Eigendecompose
    eigenvalues, eigenvectors = torch.linalg.eigh(Q_sym)

    # Sort by absolute magnitude (descending)
    order = eigenvalues.abs().argsort(descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Rank-k approximation
    Q_hat = torch.zeros_like(Q)
    for i in range(min(k, len(eigenvalues))):
        v = eigenvectors[:, i:i+1]
        Q_hat += eigenvalues[i] * (v @ v.T)

    # Pearson correlation
    Q_flat = Q_sym.flatten()
    Q_hat_flat = Q_hat.flatten()

    Q_mean = Q_flat.mean()
    Q_hat_mean = Q_hat_flat.mean()

    numerator = ((Q_flat - Q_mean) * (Q_hat_flat - Q_hat_mean)).sum()
    denominator = torch.sqrt(
        ((Q_flat - Q_mean) ** 2).sum() *
        ((Q_hat_flat - Q_hat_mean) ** 2).sum()
    )

    return (numerator / (denominator + 1e-10)).item()


def analyze_feature_interactions(tracer: Tracer, feature_indices: list, rank_k: int = 2):
    """
    Analyze interaction matrices for specified output features.

    Uses the paper's Tracer.q() method to compute interaction tensors.
    """
    results = {
        "correlations": [],
        "effective_ranks": [],
        "feature_indices": [],
    }

    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        try:
            # Get interaction matrix Q using Tracer (paper's method)
            Q = tracer.q(feat_idx)

            if Q is None or Q.numel() == 0:
                continue

            # Compute rank-k correlation
            corr = rank_k_approximation_correlation(Q, k=rank_k)

            # Compute effective rank
            Q_sym = (Q + Q.T) / 2
            eigenvalues = torch.linalg.eigvalsh(Q_sym)
            eff_rank = compute_effective_rank(eigenvalues.abs())

            results["correlations"].append(corr)
            results["effective_ranks"].append(eff_rank)
            results["feature_indices"].append(feat_idx)

        except Exception as e:
            print(f"Error analyzing feature {feat_idx}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/language/interaction_analysis.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model: {config['model']['pretrained']}")
    model = Transformer.from_pretrained(config["model"]["pretrained"])
    model = model.to(args.device)

    # Load input and output SAEs
    layer = config["sae"]["layer"]
    print(f"Loading SAEs for layer {layer}")

    # The Tracer needs both input and output SAEs
    # These should be trained at mlp_in and mlp_out points
    in_sae = SAE.from_pretrained(config["sae"]["in_checkpoint"])
    out_sae = SAE.from_pretrained(config["sae"]["out_checkpoint"])

    # Create Tracer (uses paper's implementation)
    tracer = Tracer(model, layer=layer, in_sae=in_sae, out_sae=out_sae)

    # Analyze features
    n_features = config["analysis"]["n_features"]
    feature_indices = list(range(min(n_features, out_sae.config.d_features)))

    print(f"Analyzing {len(feature_indices)} features...")
    results = analyze_feature_interactions(
        tracer,
        feature_indices,
        rank_k=config["analysis"]["rank_k"],
    )

    # Compute summary statistics
    correlations = np.array(results["correlations"])
    fraction_above_075 = (correlations > 0.75).mean()

    summary = {
        "n_analyzed": len(correlations),
        "fraction_above_075": float(fraction_above_075),
        "mean_correlation": float(correlations.mean()),
        "std_correlation": float(correlations.std()),
        "mean_effective_rank": float(np.mean(results["effective_ranks"])),
        "paper_claim": "69% of features have >0.75 rank-2 correlation",
        "our_result": f"{fraction_above_075*100:.1f}% of features have >0.75 rank-2 correlation",
        "claim_supported": fraction_above_075 > 0.60,  # Allow some margin
    }

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Features analyzed: {summary['n_analyzed']}")
    print(f"Fraction with >0.75 correlation: {summary['fraction_above_075']*100:.1f}%")
    print(f"Mean correlation: {summary['mean_correlation']:.4f}")
    print(f"Mean effective rank: {summary['mean_effective_rank']:.2f}")
    print(f"\nPaper claim: {summary['paper_claim']}")
    print(f"Our result: {summary['our_result']}")
    print(f"Claim supported: {summary['claim_supported']}")
    print(f"{'='*60}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "summary": summary,
        "per_feature": results,
        "config": config,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
```

---

### CONFIG FILES

#### `configs/language_sae.yaml`
```yaml
name: language_sae_layer2
experiment_id: L5.1

model:
  # Use pretrained bilinear transformer from HuggingFace
  pretrained: "tdooms/ts-medium"  # TinyStories-trained model

sae:
  point: "mlp_out"    # Hook point for activations
  layer: 2            # Which transformer layer
  expansion: 8        # SAE features = expansion * d_model
  k: 32               # Top-k sparsity

training:
  lr: 0.001
  batch_size: 64
  n_batches: 10000    # Number of training batches

data:
  dataset: tinystories
  n_samples: 100000   # Samples for activation collection
```

#### `configs/language_negation.yaml`
```yaml
name: language_negation
experiment_id: L5.2

model:
  pretrained: "tdooms/ts-medium"

sae:
  checkpoint: "results/language/sae_layer2.pt"
  layer: 2

analysis:
  n_samples: 50000    # Samples to analyze
  top_k: 20           # Top features to report
```

#### `configs/language_interaction.yaml`
```yaml
name: language_interaction
experiment_id: L5.3

model:
  pretrained: "tdooms/ts-medium"

sae:
  layer: 2
  # For Tracer, we need SAEs at both input and output of MLP
  in_checkpoint: "results/language/sae_mlp_in_layer2.pt"
  out_checkpoint: "results/language/sae_mlp_out_layer2.pt"

analysis:
  n_features: 1000    # Number of output features to analyze
  rank_k: 2           # Rank for low-rank approximation
  target_correlation: 0.75  # Paper claims 69% above this
```

---

### SLURM JOB FILES

#### `jobs/train_sae.job`
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train_sae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

echo "Training SAE for Section 5 using original paper code..."
echo "Model: tdooms/ts-medium (pretrained bilinear transformer)"

python src/language/run_sae_training.py \
    --config configs/language_sae.yaml \
    --checkpoint-dir results/language
```

#### `jobs/analyze_negation.job`
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=negation_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=logs/negation_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

echo "Running negation circuit discovery..."
echo "Looking for features 3834 (not+negative) and 751 (not+positive)"

python src/language/negation_discovery.py \
    --config configs/language_negation.yaml \
    --output results/language/negation_analysis.json
```

#### `jobs/analyze_interaction.job`
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=interaction_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=logs/interaction_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

echo "Running interaction matrix analysis..."
echo "Verifying paper claim: 69% of features have >0.75 rank-2 correlation"

python src/language/interaction_analysis.py \
    --config configs/language_interaction.yaml \
    --output results/language/interaction_analysis.json
```

#### `jobs/language_full_pipeline.job`
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=language_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/language_full_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/fact-project" || exit 1
source activate fact

echo "Running full Section 5 pipeline..."

# Step 1: Train SAE
echo "Step 1/3: Training SAE..."
python src/language/run_sae_training.py \
    --config configs/language_sae.yaml \
    --checkpoint-dir results/language

# Step 2: Negation discovery
echo "Step 2/3: Discovering negation circuits..."
python src/language/negation_discovery.py \
    --config configs/language_negation.yaml \
    --output results/language/negation_analysis.json

# Step 3: Interaction analysis
echo "Step 3/3: Analyzing interaction matrices..."
python src/language/interaction_analysis.py \
    --config configs/language_interaction.yaml \
    --output results/language/interaction_analysis.json

echo "Section 5 pipeline complete!"
```

---

## EXECUTION CHECKLIST

### Step 1: Verify Original Code Works
```bash
# Test that original code imports work
cd /path/to/UvA_FACT_2025
python -c "
import sys
sys.path.insert(0, 'bilinear-decomposition-main')
from language.transformer import Transformer, Config
from sae.sae import SAE, SAEConfig
from sae.tracer import Tracer
print('All original code imports OK')
"
```

### Step 2: Create Directory Structure
```bash
mkdir -p src/language configs jobs results/language logs
touch src/language/__init__.py
```

### Step 3: Create Wrapper Scripts
Create the following files (they wrap original code, don't reimplement):
- `src/language/run_sae_training.py`
- `src/language/negation_discovery.py`
- `src/language/interaction_analysis.py`

### Step 4: Verify Imports Work
```bash
python -c "
import sys
sys.path.insert(0, 'bilinear-decomposition-main')
from src.language.negation_discovery import find_negation_features
print('Wrapper imports OK')
"
```

### Step 5: Create Fashion-MNIST Configs
Create the 4 Fashion-MNIST config files for Section 4 completion.

### Step 6: Local Testing
```bash
# Test loading pretrained model
python -c "
import sys
sys.path.insert(0, 'bilinear-decomposition-main')
from language.transformer import Transformer
model = Transformer.from_pretrained('tdooms/ts-medium')
print(f'Loaded model with {sum(p.numel() for p in model.parameters())} params')
"

# Test Fashion-MNIST (2 epochs)
python src/train.py --config configs/fashion_dense_none.yaml --seed 42 --no-wandb --epochs 2
```

### Step 7: Verify All Job Files Exist
```bash
ls jobs/
# Should show:
# - train_fashion_array.job (Section 4)
# - train_sae.job (Section 5)
# - analyze_negation.job (Section 5)
# - analyze_interaction.job (Section 5)
# - language_full_pipeline.job (Section 5 - all steps)
```

---

## SUCCESS CRITERIA

Before declaring infrastructure complete:

**Section 4 (Fashion-MNIST)**:
- [ ] 4 Fashion-MNIST config files created
- [ ] `src/train.py` updated to support Fashion-MNIST
- [ ] `jobs/train_fashion_array.job` created
- [ ] Local test passes (2 epochs, single seed)

**Section 5 (Language)**:
- [ ] Original code imports work (transformer, SAE, tracer)
- [ ] Pretrained model loads from HuggingFace (`tdooms/ts-medium`)
- [ ] `src/language/` wrapper scripts created
- [ ] SAE training script uses original `SAE.fit()` method
- [ ] Negation discovery identifies high-differential features
- [ ] Interaction analysis uses original `Tracer.q()` method
- [ ] All SLURM job files created
- [ ] Environmental tracking (codecarbon + wandb) integrated

---

## KEY METRICS TO VERIFY (from paper)

| Metric | Paper Value | Your Value |
|--------|-------------|------------|
| % features with >0.75 rank-2 correlation | 69% | ? |
| Negation feature cosine similarity | < 0 (opposing) | ? |
| SAE reconstruction loss | Converges | ? |
| Negation features identified | 3834 (not+neg), 751 (not+pos) | ? |

---

## AVAILABLE PRETRAINED MODELS (from original paper)

The paper authors have published pretrained models on HuggingFace:

| Model | Description | Usage |
|-------|-------------|-------|
| `tdooms/ts-medium` | Bilinear transformer on TinyStories | `Transformer.from_pretrained("tdooms/ts-medium")` |
| `tdooms/ts-medium-scope` | SAEs for ts-medium | `SAE.from_pretrained("tdooms/ts-medium-scope", point="mlp_out", layer=2)` |

**Check the original repo for more pretrained models**: `bilinear-decomposition-main/`

---

## IMPORTANT NOTES

1. **DO NOT REIMPLEMENT**: The original code has complete implementations. Use them.

2. **Tracer class is key**: The `sae/tracer.py` Tracer class computes interaction matrices (Q tensors) between SAE features. This is central to the paper's analysis.

3. **NNSight for activation access**: The `language/utils.py` Sight class wraps NNSight for clean activation access at any hook point.

4. **Top-k SAE**: The original code uses top-k sparsity (not L1), configured via `SAEConfig.k`.

5. **Tutorial notebook**: See `bilinear-decomposition-main/tutorials/2_language.ipynb` for usage examples.

---

**Now begin by:**
1. Verifying original code imports work
2. Creating directory structure
3. Creating `src/language/run_sae_training.py` (wrapper around original SAE code)

**Show me the file and ask for confirmation before proceeding.**

## PROMPT END
