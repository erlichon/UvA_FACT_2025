#!/bin/bash
# Extension 2: Cross-Dataset Robustness Runner
#
# Consolidated script for all Extension 2 experiments (replaces 7+ shell scripts).
#
# Usage:
#   ./scripts/train/run_extension2.sh train emnist-letters [--seeds 42,43,44,45,46] [--com]
#   ./scripts/train/run_extension2.sh train emnist-digits [--seeds ...] [--com]
#   ./scripts/train/run_extension2.sh train all [--seeds ...] [--com]
#   ./scripts/train/run_extension2.sh eval subspace [--seeds ...]
#   ./scripts/train/run_extension2.sh aggregate
#   ./scripts/train/run_extension2.sh all  # Full pipeline
#   ./scripts/train/run_extension2.sh help

set -e

# Default values
SEEDS="42,43,44,45,46"
USE_COM=false
CHECKPOINT_DIR="results/extension2/checkpoints"
SUBSPACE_DIR="results/extension2/subspace"

# Parse global options
while [[ "$1" == --* ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --com)
            USE_COM=true
            shift
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
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
    ./scripts/train/run_extension2.sh <command> [subcommand] [options]

COMMANDS:
    train <dataset>     Train EMNIST models
                        Datasets: emnist-letters, emnist-digits, all
    
    eval <test>         Run evaluation tests (requires trained models)
                        Tests: subspace, similarity
    
    aggregate           Aggregate subspace results across seeds
    
    all                 Full pipeline (train all + eval + aggregate)
    
    help                Show this help message

OPTIONS:
    --seeds <list>      Comma-separated seeds (default: 42,43,44,45,46)
    --com               Enable Center-of-Mass normalization
    --checkpoint-dir    Custom checkpoint directory

EXAMPLES:
    # Train EMNIST Letters with CoM for all seeds
    ./scripts/train/run_extension2.sh train emnist-letters --com
    
    # Train both datasets without CoM
    ./scripts/train/run_extension2.sh train all
    
    # Run subspace geometry test
    ./scripts/train/run_extension2.sh eval subspace
    
    # Full pipeline with CoM
    ./scripts/train/run_extension2.sh --com all
EOF
}

train_model() {
    local dataset=$1
    local seed=$2
    local com_flag=""
    local config_suffix=""
    
    if [ "$USE_COM" = true ]; then
        com_flag="--com"
        config_suffix="_com"
    fi
    
    case "$dataset" in
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
    
    echo "Training $dataset seed $seed (CoM: $USE_COM)..."
    
    # Add data.apply_com to config if needed
    COM_ARG=""
    if [ "$USE_COM" = true ]; then
        COM_ARG="--data.apply_com true"
    fi
    
    python src/train.py \
        --config "$CONFIG" \
        --seed "$seed" \
        --checkpoint-dir "$CHECKPOINT_DIR"
}

run_train() {
    local dataset="${1:-all}"
    
    mkdir -p "$CHECKPOINT_DIR"
    
    case "$dataset" in
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
                train_model "emnist-letters" "$seed"
                train_model "emnist-digits" "$seed"
            done
            ;;
        *)
            echo "Unknown dataset: $dataset"
            echo "Valid options: emnist-letters, emnist-digits, all"
            exit 1
            ;;
    esac
    
    echo ""
    echo "Training complete! Checkpoints saved to: $CHECKPOINT_DIR"
}

run_eval() {
    local test="${1:-subspace}"
    
    case "$test" in
        subspace)
            echo "Running subspace geometry tests..."
            mkdir -p "$SUBSPACE_DIR"
            
            COM_FLAG=""
            if [ "$USE_COM" = true ]; then
                COM_FLAG="--with-com"
            fi
            
            for seed in "${SEED_ARRAY[@]}"; do
                echo "Testing seed $seed..."
                python scripts/extension2/check_similarity.py \
                    --k 30 \
                    $COM_FLAG
            done
            ;;
        similarity)
            echo "Running similarity check..."
            COM_FLAG=""
            if [ "$USE_COM" = true ]; then
                COM_FLAG="--with-com"
            fi
            python scripts/extension2/check_similarity.py $COM_FLAG
            ;;
        *)
            echo "Unknown test: $test"
            echo "Valid options: subspace, similarity"
            exit 1
            ;;
    esac
}

run_aggregate() {
    echo "Aggregating subspace results..."
    python scripts/extension2/aggregate_results.py \
        --results-dir "$SUBSPACE_DIR"
}

run_all() {
    echo "="
    echo "Extension 2: Full Pipeline"
    echo "="
    echo "Seeds: ${SEEDS}"
    echo "CoM: ${USE_COM}"
    echo ""
    
    echo "Step 1/3: Training EMNIST models..."
    run_train "all"
    
    echo ""
    echo "Step 2/3: Running evaluations..."
    run_eval "subspace"
    
    echo ""
    echo "Step 3/3: Aggregating results..."
    run_aggregate
    
    echo ""
    echo "="
    echo "Extension 2 pipeline complete!"
    echo "="
}

# Main command dispatch
case "$COMMAND" in
    train)
        run_train "$SUBCOMMAND"
        ;;
    eval)
        run_eval "$SUBCOMMAND"
        ;;
    aggregate)
        run_aggregate
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './scripts/train/run_extension2.sh help' for usage."
        exit 1
        ;;
esac
