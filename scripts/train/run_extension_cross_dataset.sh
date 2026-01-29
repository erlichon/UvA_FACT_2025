#!/bin/bash
# Cross-Dataset Robustness Runner (Extension 1)
#
# Consolidated script for cross-dataset robustness experiments.
# Uses Phase 1 regularization settings (noise=0.5, wd=1.0) with CoM normalization.
#
# Usage:
#   ./scripts/train/run_extension_cross_dataset.sh train emnist-letters [--seeds 42,43,44,45,46]
#   ./scripts/train/run_extension_cross_dataset.sh train emnist-digits [--seeds ...]
#   ./scripts/train/run_extension_cross_dataset.sh train all [--seeds ...]
#   ./scripts/train/run_extension_cross_dataset.sh figures [--sections all]
#   ./scripts/train/run_extension_cross_dataset.sh all  # Full pipeline (train + figures)
#   ./scripts/train/run_extension_cross_dataset.sh test # Quick test (2 epochs, 1 seed)
#   ./scripts/train/run_extension_cross_dataset.sh help
#
# Options:
#   --quick       2 epochs, 1 seed (for testing)
#   --no-wandb    Disable wandb logging
#   --epochs N    Override number of epochs
#
# NOTE: Center-of-Mass (CoM) normalization is ALWAYS enabled for cross-dataset comparison.

set -e

# Resolve script directory and project root so the script works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure PYTHONPATH includes project root for src module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default values (paths are interpreted relative to project root)
SEEDS="42,43,44,45,46"
CHECKPOINT_DIR="checkpoints/extension2"
QUICK_MODE=false
WANDB_FLAG=""
EPOCHS_OVERRIDE=""

# Parse global options
while [[ "$1" == --* ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            EPOCHS_OVERRIDE="--epochs 2"
            SEEDS="42"  # Only one seed in quick mode
            shift
            ;;
        --no-wandb)
            WANDB_FLAG="--no-wandb"
            shift
            ;;
        --epochs)
            EPOCHS_OVERRIDE="--epochs $2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

COMMAND="${1:-help}"
SUBCOMMAND="${2:-}"

# Convert seeds string to array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

show_help() {
    cat << EOF
Extension 2: Cross-Dataset Robustness Runner

USAGE:
    ./scripts/train/run_extension_cross_dataset.sh <command> [subcommand] [options]

COMMANDS:
    train <dataset>     Train models with CoM normalization
                        Datasets: mnist, emnist-letters, emnist-digits, all
    
    figures [sections]  Generate Extension 2 figures
                        Sections: eigenvectors, heatmaps, distributions, similarity,
                                  3way, selection, angles, all (default)
    
    test                Quick test (2 epochs, 1 seed, MNIST only)
    
    all                 Full pipeline (train all + figures)
    
    help                Show this help message

OPTIONS:
    --seeds <list>      Comma-separated seeds (default: 42,43,44,45,46)
    --checkpoint-dir    Custom checkpoint directory
    --quick             2 epochs, 1 seed (for testing)
    --no-wandb          Disable wandb logging
    --epochs N          Override number of epochs

EXAMPLES:
    # Quick test (2 epochs)
    ./scripts/train/run_extension_cross_dataset.sh test
    
    # Train all cross-dataset models (MNIST + EMNIST with CoM)
    ./scripts/train/run_extension_cross_dataset.sh train all
    
    # Train with quick mode (2 epochs, 1 seed)
    ./scripts/train/run_extension_cross_dataset.sh train all --quick
    
    # Train MNIST with CoM only
    ./scripts/train/run_extension_cross_dataset.sh train mnist
    
    # Train EMNIST Letters only
    ./scripts/train/run_extension_cross_dataset.sh train emnist-letters
    
    # Generate all cross-dataset figures
    ./scripts/train/run_extension_cross_dataset.sh figures
    
    # Generate specific figure sections
    ./scripts/train/run_extension_cross_dataset.sh figures similarity 3way
    
    # Full pipeline
    ./scripts/train/run_extension_cross_dataset.sh all
EOF
}

print_header() {
    echo "=========================================="
    echo "Extension 2: Cross-Dataset Robustness"
    echo "=========================================="
    echo "Command: $COMMAND ${SUBCOMMAND:-}"
    if $QUICK_MODE; then echo "Mode: QUICK (2 epochs, 1 seed)"; fi
    if [ -n "$WANDB_FLAG" ]; then echo "wandb: disabled"; fi
    echo "=========================================="
    echo ""
}

train_model() {
    local dataset=$1
    local seed=$2
    
    case "$dataset" in
        mnist|mnist-com)
            CONFIG="configs/mnist_dense_full_com.yaml"
            ;;
        emnist-letters|emnist_letters)
            CONFIG="configs/emnist_letters_regularized.yaml"
            ;;
        emnist-digits|emnist_digits)
            CONFIG="configs/emnist_digits_regularized.yaml"
            ;;
        *)
            echo "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
    
    echo "Training $dataset seed $seed (CoM: enabled)..."
    
    # CoM is always enabled for Extension 2
    python src/train.py \
        --config "$CONFIG" \
        --seed "$seed" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --apply-com true \
        $EPOCHS_OVERRIDE \
        $WANDB_FLAG
}

run_train() {
    local dataset="${1:-all}"
    
    cd "$PROJECT_ROOT"
    mkdir -p "$CHECKPOINT_DIR"
    print_header
    
    case "$dataset" in
        mnist|mnist-com)
            for seed in "${SEED_ARRAY[@]}"; do
                train_model "mnist" "$seed"
            done
            ;;
        emnist-letters|emnist_letters)
            for seed in "${SEED_ARRAY[@]}"; do
                train_model "emnist-letters" "$seed"
            done
            ;;
        emnist-digits|emnist_digits)
            for seed in "${SEED_ARRAY[@]}"; do
                train_model "emnist-digits" "$seed"
            done
            ;;
        all)
            for seed in "${SEED_ARRAY[@]}"; do
                train_model "mnist" "$seed"
                train_model "emnist-letters" "$seed"
                train_model "emnist-digits" "$seed"
            done
            ;;
        *)
            echo "Unknown dataset: $dataset"
            echo "Valid options: mnist, emnist-letters, emnist-digits, all"
            exit 1
            ;;
    esac
    
    echo ""
    echo "Training complete! Checkpoints saved to: $CHECKPOINT_DIR"
}

run_figures() {
    local sections="${@:-all}"
    
    echo "Generating Extension 2 figures..."
    
    cd "$PROJECT_ROOT"

    if [ "$sections" = "" ] || [ "$sections" = "all" ]; then
        python scripts/figures/generate_extension_cross_dataset_figures.py
        echo ""
        echo "Running cross-dataset analysis scripts..."
        python scripts/extension_cross_dataset/calculate_eigenvector_similarity.py
        python scripts/extension_cross_dataset/calculate_mnist0_vs_all_emnist.py
        python scripts/extension_cross_dataset/generate_accuracy_tables.py --checkpoint-dir "$CHECKPOINT_DIR" --seeds "$SEEDS"
    else
        python scripts/figures/generate_extension_cross_dataset_figures.py --sections $sections
    fi
}

run_test() {
    echo "=========================================="
    echo "Extension 2: Quick Test Mode"
    echo "=========================================="
    echo "Testing with: 2 epochs, 1 seed, MNIST only"
    echo ""
    
    cd "$PROJECT_ROOT"
    mkdir -p "$CHECKPOINT_DIR"
    
    # Train one MNIST model with 2 epochs
    python src/train.py \
        --config configs/mnist_dense_full_com.yaml \
        --seed 42 \
        --epochs 2 \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --apply-com true \
        --no-wandb
    
    echo ""
    echo "Test complete! Checkpoint saved to: $CHECKPOINT_DIR"
}

run_all() {
    print_header
    echo "Seeds: ${SEEDS}"
    echo "CoM: enabled (always)"
    echo ""
    
    echo "Step 1/2: Training models (MNIST + EMNIST with CoM)..."
    run_train "all"
    
    echo ""
    echo "Step 2/2: Generating figures..."
    run_figures
    
    echo ""
    echo "=========================================="
    echo "Extension 2 pipeline complete!"
    echo "=========================================="
}

# Main command dispatch
case "$COMMAND" in
    train)
        run_train "$SUBCOMMAND"
        ;;
    figures)
        shift  # Remove 'figures' from args
        run_figures "$@"
        ;;
    test)
        run_test
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './scripts/train/run_extension_cross_dataset.sh help' for usage."
        exit 1
        ;;
esac
