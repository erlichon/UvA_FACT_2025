#!/bin/bash
# Train EMNIST-Digits models for all 5 seeds (Step 1 of Extension 2)
# Uses src/train_emnist.py with --dataset emnist_digits flag

set -e  # Exit on error

SEEDS=(42 43 44 45 46)
OUTPUT_DIR="results/extension2/checkpoints"

echo "========================================"
echo "Training EMNIST-Digits models (5 seeds)"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Training EMNIST-Digits seed $SEED"
    echo "----------------------------------------"
    
    python src/train_emnist.py \
        --dataset emnist_digits \
        --seed "$SEED" \
        --epochs 100 \
        --noise-std 0.15 \
        --output-dir "$OUTPUT_DIR"
    
    echo "✓ Seed $SEED complete"
    echo ""
done

echo "========================================"
echo "All EMNIST-Digits models trained successfully!"
echo "========================================"
echo "Checkpoints saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run mechanism stability test: python src/evaluate_extension2.py --mechanism-test"
echo "  2. Aggregate results: python scripts/aggregate_subspace_results.py"
