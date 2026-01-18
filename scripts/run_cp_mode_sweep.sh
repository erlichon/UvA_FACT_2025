#!/bin/bash
# Run CP Mode Sweep: All ranks × All modes
#
# Runs 21 experiments: ranks (8, 16, 32, 64, 128, 256, 784) × modes (fixed, lambda, gated)
# Each run: 100 epochs, seed 42
# Wandb logging is DISABLED (--no-wandb)
#
# Usage:
#   chmod +x scripts/run_cp_mode_sweep.sh
#   ./scripts/run_cp_mode_sweep.sh
#
# To run in background with logging:
#   nohup ./scripts/run_cp_mode_sweep.sh > logs/cp_mode_sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create directories
mkdir -p results/phase2/checkpoints
mkdir -p logs

# Activate conda environment
echo "=========================================="
echo "CP Mode Sweep: All Ranks × All Modes"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for conda
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "fact_cpu"; then
        conda activate fact_cpu
        echo "✓ Activated fact_cpu environment"
    elif conda env list | grep -q "fact"; then
        conda activate fact
        echo "✓ Activated fact environment"
    else
        echo "Warning: fact or fact_cpu environment not found"
        echo "Continuing with current Python environment..."
    fi
else
    echo "Warning: conda not found, using current Python environment"
fi

echo ""

# CP Mode Sweep: 7 ranks × 3 modes = 21 runs
RANKS=(8 16 32 64 128 256 784)
MODES=(fixed lambda gated)
SEED=42
EPOCHS=100

TOTAL_RUNS=$((${#RANKS[@]} * ${#MODES[@]}))
CURRENT_RUN=0

echo "Running CP Mode Sweep:"
echo "  Ranks: ${RANKS[@]}"
echo "  Modes: ${MODES[@]}"
echo "  Seed: $SEED"
echo "  Epochs: $EPOCHS"
echo "  Total runs: $TOTAL_RUNS"
echo "  Wandb: DISABLED (--no-wandb)"
echo ""

for rank in "${RANKS[@]}"; do
    for mode in "${MODES[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo "=========================================="
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: Rank=$rank, Mode=$mode"
        echo "Time: $(date)"
        echo "=========================================="
        
        # Check if checkpoint already exists
        CHECKPOINT="results/phase2/checkpoints/mnist_cp_r${rank}_${mode}_seed${SEED}.pt"
        if [ -f "$CHECKPOINT" ]; then
            echo "✓ Checkpoint already exists: $CHECKPOINT"
            echo "  Skipping this run..."
            echo ""
            continue
        fi
        
        # Run training (no wandb)
        python src/train_cp.py \
            --rank "$rank" \
            --cp-init-mode "$mode" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --checkpoint-dir results/phase2/checkpoints \
            --device cpu \
            --no-wandb
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: Rank=$rank, Mode=$mode"
        else
            echo "✗ FAILED: Rank=$rank, Mode=$mode"
            echo "  Continuing with next run..."
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "CP Mode Sweep Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Checkpoints saved to: results/phase2/checkpoints/"
echo "Total runs completed: $CURRENT_RUN/$TOTAL_RUNS"
echo ""
echo "Checkpoint naming format: mnist_cp_r{rank}_{mode}_seed{seed}.pt"
echo "Examples:"
echo "  - mnist_cp_r32_fixed_seed42.pt"
echo "  - mnist_cp_r32_lambda_seed42.pt"
echo "  - mnist_cp_r32_gated_seed42.pt"
echo ""
echo "Next steps:"
echo "  1. Run notebooks/03_eigenvector_visualization.ipynb to visualize results"
echo ""


