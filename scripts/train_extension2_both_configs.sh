#!/bin/bash
# Train Extension 2 models with both regularization configurations:
# 1. Current config: noise_std=0.15, weight_decay=0.5
# 2. Phase 1 config: noise_std=0.5, weight_decay=1.0

set -e

cd /Users/maybenzion/MSCAI/FACT/UvA_FACT_2025

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fact_cpu

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

SEED=42
CHECKPOINT_DIR="results/phase1/checkpoints"

echo "============================================================"
echo "Extension 2: Training with Both Regularization Configs"
echo "Started at: $(date)"
echo "============================================================"
echo ""

# =================================================================
# Configuration 1: Current Extension 2 (lighter regularization)
# =================================================================
echo "--- Configuration 1: Current Extension 2 (noise=0.15, wd=0.5) ---"
echo ""

echo "Training EMNIST Letters (current config)..."
python src/train.py \
    --config configs/emnist_letters_regularized.yaml \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --no-wandb

echo ""
echo "Training EMNIST Digits (current config)..."
python src/train.py \
    --config configs/emnist_digits_regularized.yaml \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --no-wandb

echo ""
echo "--- Configuration 2: Phase 1 Style (noise=0.5, wd=1.0) ---"
echo ""

echo "Training EMNIST Letters (Phase 1 config)..."
python src/train.py \
    --config configs/emnist_letters_phase1_reg.yaml \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --no-wandb

echo ""
echo "Training EMNIST Digits (Phase 1 config)..."
python src/train.py \
    --config configs/emnist_digits_phase1_reg.yaml \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --no-wandb

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Completed at: $(date)"
echo "============================================================"
echo ""
echo "Checkpoints saved:"
echo "  Current config:"
echo "    - $CHECKPOINT_DIR/emnist_letters_regularized_seed${SEED}.pt"
echo "    - $CHECKPOINT_DIR/emnist_digits_regularized_seed${SEED}.pt"
echo "  Phase 1 config:"
echo "    - $CHECKPOINT_DIR/emnist_letters_phase1_reg_seed${SEED}.pt"
echo "    - $CHECKPOINT_DIR/emnist_digits_phase1_reg_seed${SEED}.pt"
echo ""
