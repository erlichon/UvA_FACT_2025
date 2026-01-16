#!/bin/bash
# Model Size Sweep - Figure 5 Reproduction
# Trains models with varying d_hidden sizes
#
# Usage:
#   ./scripts/run_model_size_sweep.sh           # Full sweep (6 sizes x 5 seeds)
#   ./scripts/run_model_size_sweep.sh --quick   # Quick test (2 epochs, 1 seed)
#   ./scripts/run_model_size_sweep.sh --no-wandb  # Disable wandb logging

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
WANDB_FLAG=""
EPOCHS_OVERRIDE=""
SEEDS=(42 43 44 45 46)

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS_OVERRIDE="--epochs 2"
            SEEDS=(42)  # Only one seed in quick mode
            shift
            ;;
        --no-wandb)
            WANDB_FLAG="--no-wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--no-wandb]"
            exit 1
            ;;
    esac
done

# Configuration
CHECKPOINT_DIR="results/sweeps/model_size/checkpoints"
MODEL_SIZES=(30 50 100 300 500 1000)

echo "=========================================="
echo "Model Size Sweep - Figure 5 Reproduction"
echo "=========================================="
echo "Model sizes (d_hidden): ${MODEL_SIZES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Checkpoint dir: $CHECKPOINT_DIR"
if $QUICK_MODE; then
    echo "Mode: QUICK (2 epochs, 1 seed)"
else
    echo "Mode: FULL (100 epochs, 5 seeds)"
fi
echo "=========================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Track progress
TOTAL=$((${#MODEL_SIZES[@]} * ${#SEEDS[@]}))
CURRENT=0

# Run each model size and seed
for SIZE in "${MODEL_SIZES[@]}"; do
    CONFIG="configs/sweeps/mnist_size_${SIZE}.yaml"
    
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo ">>> [$CURRENT/$TOTAL] Training d_hidden = $SIZE, seed = $SEED"
        echo "    Config: $CONFIG"
        
        python src/train.py \
            --config "$CONFIG" \
            --seed "$SEED" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            $WANDB_FLAG \
            $EPOCHS_OVERRIDE
        
        echo ">>> Completed d_hidden = $SIZE, seed = $SEED"
    done
done

echo ""
echo "=========================================="
echo "Model size sweep complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Total models trained: $TOTAL"
echo ""
echo "To generate the plots (Figure 5 + appendix extensions):"
echo "  python scripts/vision_analysis.py --sections truncation_similarity appendix"
echo "=========================================="
