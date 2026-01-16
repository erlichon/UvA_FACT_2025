#!/bin/bash
# Run Extension 2 Full Pipeline (100 epochs)

set -e

cd /Users/maybenzion/MSCAI/FACT/UvA_FACT_2025

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fact_cpu

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the full pipeline
echo "============================================================"
echo "Starting Extension 2 Full Pipeline (100 epochs, seed 42)"
echo "Started at: $(date)"
echo "============================================================"
echo ""

python src/evaluate_extension2.py --full-pipeline --epochs 100 --seed 42

echo ""
echo "============================================================"
echo "Completed at: $(date)"
echo "============================================================"
