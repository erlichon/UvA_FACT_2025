#!/bin/bash
# Train EMNIST models without CoM normalization for fair comparison with MNIST

set -e

cd /Users/maybenzion/MSCAI/FACT/UvA_FACT_2025

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fact_cpu

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "============================================================"
echo "Training EMNIST models WITHOUT CoM normalization"
echo "Regularization: Phase 1 (noise=0.5, wd=1.0)"
echo "============================================================"

echo ""
echo "Training EMNIST Letters (no CoM)..."
python src/train.py --config configs/emnist_letters_phase1_reg_nocom.yaml --seed 42 --checkpoint-dir results/phase1/checkpoints

echo ""
echo "Training EMNIST Digits (no CoM)..."
python src/train.py --config configs/emnist_digits_phase1_reg_nocom.yaml --seed 42 --checkpoint-dir results/phase1/checkpoints

echo ""
echo "============================================================"
echo "Training Complete"
echo "Completed at: $(date)"
echo "============================================================"
