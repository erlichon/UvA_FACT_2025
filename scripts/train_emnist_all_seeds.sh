#!/bin/bash
# Train EMNIST models for all 5 seeds
# Uses src/train_emnist.py to maintain clean code architecture

set -e  # Exit on error

SEEDS=(42 43 44 45 46)
OUTPUT_DIR="results/extension2/checkpoints"

echo "========================================"
echo "Training EMNIST models for 5 seeds"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Training seed $SEED"
    echo "----------------------------------------"
    
    python src/train_emnist.py \
        --seed "$SEED" \
        --epochs 100 \
        --noise-std 0.15 \
        --output-dir "$OUTPUT_DIR"
    
    echo "✓ Seed $SEED complete"
    echo ""
done

echo "========================================"
echo "All EMNIST models trained successfully!"
echo "========================================"
echo "Checkpoints saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run subspace tests: bash scripts/run_subspace_test_all_seeds.sh"
echo "  2. Aggregate results: python scripts/aggregate_subspace_results.py"
