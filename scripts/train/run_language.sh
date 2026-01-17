#!/bin/bash
# Unified language experiment runner (Section 5)
#
# This script consolidates all language-related experiments:
# - Figure 9: Correlation sweep (ts-medium, fw-small, fw-medium)
# - Figure 8: Negation circuit visualization
# - Figure 10: SAE training time analysis
# - Negation discovery
# - Interaction analysis
#
# Usage:
#   ./scripts/train/run_language.sh figure9 [options]     # Correlation sweep
#   ./scripts/train/run_language.sh figure8 [options]     # Negation circuit viz
#   ./scripts/train/run_language.sh figure10 [options]    # SAE training time
#   ./scripts/train/run_language.sh negation [options]    # Negation discovery
#   ./scripts/train/run_language.sh interaction [options] # Interaction analysis
#   ./scripts/train/run_language.sh figures               # Generate all figures
#   ./scripts/train/run_language.sh test                  # Quick MPS test
#   ./scripts/train/run_language.sh all [options]         # Full pipeline
#   ./scripts/train/run_language.sh help                  # Show help
#
# Options:
#   --quick       Reduced samples/features for testing
#   --device      cpu|mps|cuda (default: auto-detect)
#   --no-wandb    Disable wandb logging
#   --model       Specific model for figure9/negation/interaction
#   --sequential  Run figure9 sequentially (memory-safe)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Default values
QUICK_MODE=false
DEVICE=""
WANDB_FLAG="--no-wandb"
MODEL=""
SEQUENTIAL=false
FEATURE=3834

# Parse global options and extract command
COMMAND=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_FLAG="--no-wandb"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        figure9|correlation|figure8|negation-viz|figure10|sae-training|negation|interaction|figures|test|all|help)
            if [ -z "$COMMAND" ]; then
                COMMAND=$1
            else
                REMAINING_ARGS+=("$1")
            fi
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default command
COMMAND=${COMMAND:-help}

# Default device based on platform
if [ -z "$DEVICE" ]; then
    if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        DEVICE="mps"
    elif python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

# Check for conda environment
activate_conda() {
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate fact_cpu 2>/dev/null || conda activate fact 2>/dev/null || echo "Warning: Could not activate conda env"
    fi
}

# Print header
print_header() {
    echo "=========================================="
    echo "FACT-AI Language Experiments"
    echo "=========================================="
    echo "Command: $COMMAND"
    echo "Device: $DEVICE"
    if $QUICK_MODE; then echo "Mode: QUICK"; fi
    if [ -n "$MODEL" ]; then echo "Model: $MODEL"; fi
    echo "=========================================="
    echo ""
}

# --- FIGURE 9: Correlation Sweep ---
run_figure9() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    # Model configurations: "model layer expansion"
    declare -a MODELS=(
        "ts-medium 4 4"
        "fw-small 8 4"
        "fw-medium 7 8"
    )
    
    # Filter by model if specified
    if [ -n "$MODEL" ] && [ "$MODEL" != "all" ]; then
        case $MODEL in
            ts-medium|tdooms/ts-medium)
                MODELS=("ts-medium 4 4")
                ;;
            fw-small|tdooms/fw-small)
                MODELS=("fw-small 8 4")
                ;;
            fw-medium|tdooms/fw-medium)
                MODELS=("fw-medium 7 8")
                ;;
            *)
                echo "Unknown model: $MODEL"
                echo "Valid models: ts-medium, fw-small, fw-medium, all"
                exit 1
                ;;
        esac
    fi
    
    # Quick mode settings
    local n_features=100
    local ranks="1,2,4,8,16"
    local n_samples=3000
    local max_batches=60
    local scatter_flag=""
    
    if $QUICK_MODE; then
        n_features=50
        n_samples=1000
        max_batches=30
    else
        scatter_flag="--save-scatter --max-scatter-samples 1000"
    fi
    
    echo ">>> Running Figure 9 Correlation Sweep"
    echo "    Models: ${#MODELS[@]}"
    echo "    Features: $n_features"
    echo "    Ranks: $ranks"
    echo ""
    
    for model_config in "${MODELS[@]}"; do
        read -r model layer expansion <<< "$model_config"
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running: $model (layer=$layer, expansion=$expansion)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python src/language/verify_correlation.py \
            --config "configs/language_correlation_fw.yaml" \
            --model "tdooms/$model" \
            --layer "$layer" \
            --expansion "$expansion" \
            --k 30 \
            --output "results/language/correlation_$model.json" \
            --plot "results/language/correlation_$model.png" \
            --device "$DEVICE" \
            $WANDB_FLAG \
            --n-features "$n_features" \
            --ranks "$ranks" \
            --n-samples "$n_samples" \
            --max-batches "$max_batches" \
            $scatter_flag
        
        echo "Completed: $model"
        
        if $SEQUENTIAL; then
            echo "Waiting 5s for memory cleanup..."
            sleep 5
        fi
    done
    
    echo ""
    echo "Figure 9 sweep complete!"
    echo "Results: results/language/correlation_*.json"
    echo ""
    echo "To generate combined figures:"
    echo "  python scripts/figures/generate_language_figures.py"
}

# --- FIGURE 8: Negation Circuit Visualization ---
run_figure8() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    # Figure 8 now uses memory-efficient iterative eigensolver
    # No longer OOM-prone on MPS with 48GB RAM
    local fig8_device="$DEVICE"
    
    local n_samples=1500
    if $QUICK_MODE; then
        n_samples=500
    fi
    
    echo ">>> Running Figure 8 (Negation Circuit Visualization)"
    echo "    Feature: $FEATURE (not-good)"
    echo "    Device: $fig8_device"
    echo "    Samples: $n_samples"
    echo "    Memory: Uses iterative eigensolver (constant memory)"
    echo ""
    
    python src/language/negation_visualization.py \
        --config configs/language_negation_fw.yaml \
        --output "results/language/figure_8_data_fw_medium.json" \
        --feature "$FEATURE" \
        --device "$fig8_device" \
        --n-samples "$n_samples"
    
    echo ""
    echo "Figure 8 data generated!"
    echo "Output: results/language/figure_8_data_fw_medium.json"
}

# --- FIGURE 10: SAE TRAINING TIME ---
run_figure10() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local n_features=-1
    local n_batches=10
    if $QUICK_MODE; then
        n_features=100
        n_batches=5
    fi
    
    echo ">>> Running SAE Training Time Analysis (Figure 10)"
    echo "    Model: fw-medium, Layer: 12, Expansion: 16"
    echo "    SAE versions: v0 (1x) -> v4 (16x training)"
    echo "    Features: $n_features (-1 = all)"
    echo "    Device: $DEVICE"
    echo ""
    
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/sae_training_time_analysis.py \
        --device "$DEVICE" \
        --n-features "$n_features" \
        --n-batches "$n_batches" \
        --output "results/language/sae_training_time_comparison.json"
    
    echo ""
    echo "Figure 10 data generated!"
    echo "Output: results/language/sae_training_time_comparison.json"
}

# --- NEGATION DISCOVERY ---
run_negation() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local n_samples=50000
    if $QUICK_MODE; then
        n_samples=2000
    fi
    
    local config="configs/language_negation_fw.yaml"
    if [ -n "$MODEL" ]; then
        case $MODEL in
            ts-medium|tdooms/ts-medium)
                config="configs/language_negation_ts.yaml"
                ;;
            fw-small|tdooms/fw-small)
                config="configs/language_negation_fw_small.yaml"
                ;;
            fw-medium|tdooms/fw-medium)
                config="configs/language_negation_fw.yaml"
                ;;
        esac
    fi
    
    echo ">>> Running Negation Discovery"
    echo "    Config: $config"
    echo "    Samples: $n_samples"
    echo ""
    
    python src/language/negation_discovery.py \
        --config "$config" \
        --output "results/language/negation_analysis.json" \
        --device "$DEVICE" \
        --use-pretrained \
        $WANDB_FLAG
    
    echo ""
    echo "Negation discovery complete!"
    echo "Output: results/language/negation_analysis.json"
}

# --- INTERACTION ANALYSIS ---
run_interaction() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local n_features=500
    if $QUICK_MODE; then
        n_features=50
    fi
    
    echo ">>> Running Interaction Analysis"
    echo "    Features: $n_features"
    echo ""
    
    python src/language/interaction_analysis.py \
        --config "configs/language_interaction.yaml" \
        --output "results/language/interaction_analysis.json" \
        --device "$DEVICE" \
        $WANDB_FLAG
    
    echo ""
    echo "Interaction analysis complete!"
    echo "Output: results/language/interaction_analysis.json"
}

# --- GENERATE FIGURES ---
generate_figures() {
    print_header
    activate_conda
    
    echo ">>> Generating language figures..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/generate_language_figures.py
    
    echo ""
    echo "Figure generation complete!"
}

# --- TEST ---
run_test() {
    echo "=========================================="
    echo "Quick Language MPS Test"
    echo "=========================================="
    echo ""
    
    activate_conda
    
    echo "1. Checking device availability..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
    echo ""
    
    mkdir -p results/language/test
    
    # Test 1: Negation Discovery (minimal)
    echo "2. Testing negation discovery (2000 samples)..."
    cat > /tmp/test_negation_config.yaml << 'EOF'
name: test_negation_mps
model:
  pretrained: "tdooms/ts-medium"
sae:
  use_pretrained: true
  point: "mlp-out"
  layer: 2
  expansion: 4
  k: 30
analysis:
  n_samples: 2000
  top_k: 10
EOF
    
    python src/language/negation_discovery.py \
        --config /tmp/test_negation_config.yaml \
        --output results/language/test/negation_test.json \
        --use-pretrained \
        --device "$DEVICE" \
        --no-wandb \
        && echo "Negation discovery test PASSED" \
        || { echo "Negation discovery test FAILED"; exit 1; }
    
    echo ""
    
    # Test 2: Interaction Analysis (minimal)
    echo "3. Testing interaction analysis (20 features)..."
    cat > /tmp/test_interaction_config.yaml << 'EOF'
name: test_interaction_mps
model:
  pretrained: "tdooms/ts-medium"
sae:
  layer: 2
  input:
    name: "mlp-in"
    expansion: 4
    k: 30
  output:
    name: "mlp-out"
    expansion: 4
    k: 30
analysis:
  n_features: 20
  rank_k: 2
EOF
    
    python src/language/interaction_analysis.py \
        --config /tmp/test_interaction_config.yaml \
        --output results/language/test/interaction_test.json \
        --device "$DEVICE" \
        --no-wandb \
        && echo "Interaction analysis test PASSED" \
        || { echo "Interaction analysis test FAILED"; exit 1; }
    
    echo ""
    echo "=========================================="
    echo "ALL LANGUAGE TESTS PASSED"
    echo "=========================================="
}

# --- ALL ---
run_all() {
    echo ">>> Running full language pipeline..."
    echo ""
    run_figure9
    echo ""
    run_negation
    echo ""
    run_interaction
    echo ""
    generate_figures
    echo ""
    echo "Full pipeline complete!"
    echo ""
    echo "Note: Figure 8 was skipped (run separately with: ./scripts/train/run_language.sh figure8 --device cpu)"
}

# --- HELP ---
show_help() {
    echo "Usage: ./scripts/train/run_language.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  figure9       Correlation sweep for Figure 9 (all 3 models)"
    echo "  figure8       Negation circuit visualization (Figure 8)"
    echo "  figure10      SAE training time analysis (Figure 10)"
    echo "  negation      Negation feature discovery"
    echo "  interaction   Interaction matrix analysis"
    echo "  figures       Generate all language figures from results"
    echo "  test          Quick MPS verification tests"
    echo "  all           Full language pipeline (except Figure 8)"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  --quick       Reduced samples/features for testing"
    echo "  --device      cpu|mps|cuda (default: auto-detect)"
    echo "  --no-wandb    Disable wandb logging"
    echo "  --model       Specific model: ts-medium, fw-small, fw-medium, all"
    echo "  --sequential  Run models sequentially (memory-safe for figure9)"
    echo "  --feature     Feature index for figure8 (default: 3834)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/train/run_language.sh test                      # Quick tests"
    echo "  ./scripts/train/run_language.sh figure9 --quick           # Quick correlation sweep"
    echo "  ./scripts/train/run_language.sh figure9 --model fw-medium # Single model"
    echo "  ./scripts/train/run_language.sh figure8 --device cpu      # Figure 8 (safe mode)"
    echo "  ./scripts/train/run_language.sh all                       # Full pipeline"
    echo ""
    echo "Models for Figure 9:"
    echo "  ts-medium  (6 layers, layer 4, expansion 4) - Paper's 'ts-tiny'"
    echo "  fw-small   (12 layers, layer 8, expansion 4)"
    echo "  fw-medium  (16 layers, layer 7, expansion 8) - Primary model"
}

# --- MAIN ---
case $COMMAND in
    figure9|correlation)
        run_figure9
        ;;
    figure8|negation-viz)
        run_figure8
        ;;
    figure10|sae-training)
        run_figure10
        ;;
    negation)
        run_negation
        ;;
    interaction)
        run_interaction
        ;;
    figures)
        generate_figures
        ;;
    test)
        run_test
        ;;
    all)
        run_all
        ;;
    help|*)
        show_help
        ;;
esac
