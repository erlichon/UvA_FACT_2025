#!/bin/bash
# Run CP Rank Sweep: Quick test (7 ranks × 1 seed)
#
# Runs 7 experiments: ranks (8, 16, 32, 64, 128, 256, 784) × seed 42
# Wandb logging is DISABLED (--no-wandb)
#
# Usage:
#   chmod +x scripts/run_cp_sweep.sh
#   ./scripts/run_cp_sweep.sh
#
# To run in background with logging:
#   nohup ./scripts/run_cp_sweep.sh > logs/cp_sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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
echo "CP Rank Sweep: Quick Test (7 ranks, 1 seed)"
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

# CP Rank Sweep: 7 ranks × 1 seed = 7 runs (quick test)
RANKS=(8 16 32 64 128 256 784)
SEEDS=(42)  # Single seed for quick test

TOTAL_RUNS=$((${#RANKS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

echo "Running CP Rank Sweep (Quick Test):"
echo "  Ranks: ${RANKS[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo "  Total runs: $TOTAL_RUNS"
echo "  Wandb: DISABLED (--no-wandb)"
echo ""
echo "Note: Rank 784 is the full input dimension (28×28=784)"
echo ""

for rank in "${RANKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo "=========================================="
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: Rank=$rank, Seed=$seed"
        echo "Time: $(date)"
        echo "=========================================="
        
        # Check if checkpoint already exists
        CHECKPOINT="results/phase2/checkpoints/mnist_cp_r${rank}_seed${seed}.pt"
        if [ -f "$CHECKPOINT" ]; then
            echo "✓ Checkpoint already exists: $CHECKPOINT"
            echo "  Skipping this run..."
            echo ""
            continue
        fi
        
        # Run training (no wandb)
        python src/train_cp.py \
            --rank "$rank" \
            --seed "$seed" \
            --checkpoint-dir results/phase2/checkpoints \
            --device cpu \
            --no-wandb
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: Rank=$rank, Seed=$seed"
        else
            echo "✗ FAILED: Rank=$rank, Seed=$seed"
            echo "  Continuing with next run..."
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "CP Rank Sweep Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Checkpoints saved to: results/phase2/checkpoints/"
echo "Total runs completed: $CURRENT_RUN/$TOTAL_RUNS"
echo ""
echo "Next steps:"
echo "  1. Run notebooks/02b_cp_sweep.ipynb to visualize results"
echo ""

