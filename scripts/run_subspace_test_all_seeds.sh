#!/bin/bash
# Run subspace geometry test across all 5 seeds
# Uses src/run_subspace_test.py to maintain clean code architecture

set -e  # Exit on error

SEEDS=(42 43 44 45 46)
MNIST_DIR="results/phase1/checkpoints"
EMNIST_DIR="results/extension2/checkpoints"
OUTPUT_DIR="results/extension2/subspace"

echo "========================================"
echo "Running subspace geometry tests"
echo "========================================"
echo "MNIST checkpoints: $MNIST_DIR"
echo "EMNIST checkpoints: $EMNIST_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Testing seed $SEED"
    echo "----------------------------------------"
    
    MNIST_CKPT="$MNIST_DIR/mnist_dense_full_seed${SEED}.pt"
    EMNIST_CKPT="$EMNIST_DIR/emnist_regularized_seed${SEED}.pt"
    OUTPUT_FILE="$OUTPUT_DIR/subspace_seed${SEED}.json"
    
    # Check if checkpoints exist
    if [ ! -f "$MNIST_CKPT" ]; then
        echo "✗ Error: MNIST checkpoint not found: $MNIST_CKPT"
        exit 1
    fi
    
    if [ ! -f "$EMNIST_CKPT" ]; then
        echo "✗ Error: EMNIST checkpoint not found: $EMNIST_CKPT"
        echo "  Run: bash scripts/train_emnist_all_seeds.sh first"
        exit 1
    fi
    
    python src/run_subspace_test.py \
        --mnist-checkpoint "$MNIST_CKPT" \
        --emnist-checkpoint "$EMNIST_CKPT" \
        --k 10 \
        --output "$OUTPUT_FILE"
    
    echo "✓ Seed $SEED complete"
    echo ""
done

echo "========================================"
echo "All subspace tests completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next step:"
echo "  Aggregate results: python scripts/aggregate_subspace_results.py"
