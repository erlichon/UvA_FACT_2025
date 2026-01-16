#!/bin/bash
# MNIST Noise Sweep - Figure 4 Reproduction
# Trains models with varying Gaussian input noise levels
#
# Usage:
#   ./scripts/run_mnist_noise_sweep.sh           # Full sweep (6 configs x 1 seed)
#   ./scripts/run_mnist_noise_sweep.sh --quick   # Quick test (2 epochs)
#   ./scripts/run_mnist_noise_sweep.sh --no-wandb  # Disable wandb logging

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
WANDB_FLAG=""
EPOCHS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS_OVERRIDE="--epochs 2"
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
SEED=42
CHECKPOINT_DIR="results/sweeps/noise_sweep/checkpoints"
NOISE_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5)

echo "=========================================="
echo "MNIST Noise Sweep - Figure 4 Reproduction"
echo "=========================================="
echo "Seed: $SEED"
echo "Noise levels: ${NOISE_LEVELS[*]}"
echo "Checkpoint dir: $CHECKPOINT_DIR"
if $QUICK_MODE; then
    echo "Mode: QUICK (2 epochs)"
else
    echo "Mode: FULL (100 epochs)"
fi
echo "=========================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run each noise level
for NOISE in "${NOISE_LEVELS[@]}"; do
    CONFIG="configs/sweeps/mnist_noise_${NOISE}.yaml"
    
    echo ""
    echo ">>> Training with noise_std = $NOISE"
    echo "    Config: $CONFIG"
    
    python src/train.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        $WANDB_FLAG \
        $EPOCHS_OVERRIDE
    
    echo ">>> Completed noise_std = $NOISE"
done

echo ""
echo "=========================================="
echo "Noise sweep complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "To generate Figure 4:"
echo "  python scripts/vision_analysis.py --sections regularization"
echo "=========================================="
