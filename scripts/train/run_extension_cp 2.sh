#!/bin/bash
# Extension CP: CP Decomposition Runner
#
# Consolidated script for Extension CP experiments.
# Trains CP-decomposed bilinear models with various ranks and initialization modes.
#
# Usage:
#   ./scripts/train/run_extension_cp.sh train ranks [--seeds 42,43,44,45,46] [--modes lambda]
#   ./scripts/train/run_extension_cp.sh train modes [--seeds 42,43,44,45,46] [--ranks 32]
#   ./scripts/train/run_extension_cp.sh train all [--seeds 42,43,44,45,46]
#   ./scripts/train/run_extension_cp.sh figures [--sections all]
#   ./scripts/train/run_extension_cp.sh test   # Quick test (2 epochs)
#   ./scripts/train/run_extension_cp.sh all    # Full pipeline (train + figures)
#   ./scripts/train/run_extension_cp.sh help
#
# NOTE: CP models use NO noise augmentation (noise_std=0.0)

set -e

# Default values (5 seeds to match paper reproduction standard)
SEEDS="42,43,44,45,46"
RANKS="8,16,32,64,128,256,784"
MODES="lambda"
CHECKPOINT_DIR="checkpoints/extension_cp"
EPOCHS=""
NO_WANDB=""

# Parse global options
while [[ "$1" == --* ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --ranks)
            RANKS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB="--no-wandb"
            shift
            ;;
        *)
            break
            ;;
    esac
done

COMMAND="${1:-help}"
SUBCOMMAND="${2:-}"

# Convert comma-separated strings to arrays
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra RANK_ARRAY <<< "$RANKS"
IFS=',' read -ra MODE_ARRAY <<< "$MODES"

show_help() {
    cat << EOF
Extension CP: CP Decomposition Runner

USAGE:
    ./scripts/train/run_extension_cp.sh <command> [subcommand] [options]

COMMANDS:
    train ranks         Train CP models across all ranks (single init mode)
                        Default: ranks 8,16,32,64,128,256,784 with lambda mode
    
    train modes         Train CP models across all init modes (single rank)
                        Default: rank 32 with fixed,lambda,gated modes
    
    train all           Train all combinations (ranks × modes)
    
    figures [sections]  Generate Extension CP figures
                        Sections: rank_comparison, mode_comparison, eigenvector_quality,
                                  spectral_analysis, all (default)
    
    test                Quick test (2 epochs, single rank/mode)
    
    all                 Full pipeline (train all + figures)
    
    help                Show this help message

OPTIONS:
    --seeds <list>      Comma-separated seeds (default: 42,43,44,45,46)
    --ranks <list>      Comma-separated ranks (default: 8,16,32,64,128,256,784)
    --modes <list>      Comma-separated init modes (default: lambda)
                        Options: fixed, lambda, gated
    --epochs <n>        Override epochs from config
    --no-wandb          Disable wandb logging
    --checkpoint-dir    Custom checkpoint directory

EXAMPLES:
    # Quick test (2 epochs)
    ./scripts/train/run_extension_cp.sh test

    # Train rank sweep with lambda init (default)
    ./scripts/train/run_extension_cp.sh train ranks

    # Train mode sweep at rank 32
    ./scripts/train/run_extension_cp.sh train modes --ranks 32 --modes fixed,lambda,gated

    # Train specific ranks with multiple seeds
    ./scripts/train/run_extension_cp.sh train ranks --ranks 32,64,128 --seeds 42,43,44

    # Generate all figures
    ./scripts/train/run_extension_cp.sh figures

    # Full pipeline
    ./scripts/train/run_extension_cp.sh all
EOF
}

train_cp_model() {
    local rank=$1
    local init_mode=$2
    local seed=$3
    
    # Check if checkpoint already exists
    local checkpoint="${CHECKPOINT_DIR}/mnist_cp_r${rank}_${init_mode}_seed${seed}.pt"
    if [ -f "$checkpoint" ]; then
        echo "✓ Checkpoint exists: $checkpoint (skipping)"
        return 0
    fi
    
    echo "Training CP model: rank=$rank, mode=$init_mode, seed=$seed..."
    
    python src/train.py \
        --mode cp \
        --rank "$rank" \
        --cp-init-mode "$init_mode" \
        --seed "$seed" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        $EPOCHS \
        $NO_WANDB
}

run_train_ranks() {
    echo "=========================================="
    echo "Extension CP: Rank Sweep"
    echo "=========================================="
    echo "Ranks: ${RANKS}"
    echo "Init mode: ${MODES}"
    echo "Seeds: ${SEEDS}"
    echo ""
    
    mkdir -p "$CHECKPOINT_DIR"
    
    for rank in "${RANK_ARRAY[@]}"; do
        for mode in "${MODE_ARRAY[@]}"; do
            for seed in "${SEED_ARRAY[@]}"; do
                train_cp_model "$rank" "$mode" "$seed"
            done
        done
    done
    
    echo ""
    echo "Rank sweep complete! Checkpoints saved to: $CHECKPOINT_DIR"
}

run_train_modes() {
    # For mode sweep, use all modes and the specified rank(s)
    local modes_to_train="fixed,lambda,gated"
    IFS=',' read -ra MODES_TO_TRAIN <<< "$modes_to_train"
    
    echo "=========================================="
    echo "Extension CP: Mode Sweep"
    echo "=========================================="
    echo "Ranks: ${RANKS}"
    echo "Init modes: ${modes_to_train}"
    echo "Seeds: ${SEEDS}"
    echo ""
    
    mkdir -p "$CHECKPOINT_DIR"
    
    for rank in "${RANK_ARRAY[@]}"; do
        for mode in "${MODES_TO_TRAIN[@]}"; do
            for seed in "${SEED_ARRAY[@]}"; do
                train_cp_model "$rank" "$mode" "$seed"
            done
        done
    done
    
    echo ""
    echo "Mode sweep complete! Checkpoints saved to: $CHECKPOINT_DIR"
}

run_train_all() {
    # Train all combinations
    local all_modes="fixed,lambda,gated"
    IFS=',' read -ra ALL_MODES <<< "$all_modes"
    
    echo "=========================================="
    echo "Extension CP: Full Training"
    echo "=========================================="
    echo "Ranks: ${RANKS}"
    echo "Init modes: ${all_modes}"
    echo "Seeds: ${SEEDS}"
    echo ""
    
    mkdir -p "$CHECKPOINT_DIR"
    
    local total=$((${#RANK_ARRAY[@]} * ${#ALL_MODES[@]} * ${#SEED_ARRAY[@]}))
    local current=0
    
    for rank in "${RANK_ARRAY[@]}"; do
        for mode in "${ALL_MODES[@]}"; do
            for seed in "${SEED_ARRAY[@]}"; do
                current=$((current + 1))
                echo "[${current}/${total}] Training rank=$rank, mode=$mode, seed=$seed..."
                train_cp_model "$rank" "$mode" "$seed"
            done
        done
    done
    
    echo ""
    echo "Full training complete! Checkpoints saved to: $CHECKPOINT_DIR"
}

run_test() {
    echo "=========================================="
    echo "Extension CP: Quick Test (2 epochs)"
    echo "=========================================="
    
    EPOCHS="--epochs 2"
    NO_WANDB="--no-wandb"
    
    mkdir -p "$CHECKPOINT_DIR"
    
    # Test with rank 32, lambda mode
    train_cp_model 32 "lambda" 42
    
    echo ""
    echo "Test complete!"
    
    # Verify checkpoint
    local checkpoint="${CHECKPOINT_DIR}/mnist_cp_r32_lambda_seed42.pt"
    if [ -f "$checkpoint" ]; then
        echo "Verifying checkpoint..."
        python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Config mode: {ckpt[\"config\"][\"mode\"]}')
print(f'Config rank: {ckpt[\"config\"][\"rank\"]}')
print(f'Eigenvalues shape: {ckpt[\"eigenvalues\"].shape}')
print(f'Val accuracy: {ckpt[\"metrics\"][\"val_acc\"]:.4f}')
print(f'Effective rank: {ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
"
        echo ""
        echo "Test PASSED!"
    else
        echo "ERROR: Checkpoint not found!"
        exit 1
    fi
}

run_figures() {
    local sections="${@:-all}"
    
    echo "Generating Extension CP figures..."
    
    if [ "$sections" = "" ] || [ "$sections" = "all" ]; then
        python scripts/figures/generate_extension_cp_figures.py
    else
        python scripts/figures/generate_extension_cp_figures.py --sections $sections
    fi
}

run_all() {
    echo "=========================================="
    echo "Extension CP: Full Pipeline"
    echo "=========================================="
    echo "Seeds: ${SEEDS}"
    echo ""
    
    echo "Step 1/2: Training all CP models..."
    run_train_all
    
    echo ""
    echo "Step 2/2: Generating figures..."
    run_figures
    
    echo ""
    echo "=========================================="
    echo "Extension CP pipeline complete!"
    echo "=========================================="
}

# Main command dispatch
case "$COMMAND" in
    train)
        case "$SUBCOMMAND" in
            ranks)
                run_train_ranks
                ;;
            modes)
                run_train_modes
                ;;
            all)
                run_train_all
                ;;
            *)
                echo "Unknown train subcommand: $SUBCOMMAND"
                echo "Valid options: ranks, modes, all"
                exit 1
                ;;
        esac
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
        echo "Run './scripts/train/run_extension_cp.sh help' for usage."
        exit 1
        ;;
esac
