#!/bin/bash
# Memory-safe sequential language experiment runner
# Run experiments ONE AT A TIME to avoid OOM
#
# Usage:
#   ./scripts/run_language_sequential.sh [device] [mode] [skip_figure8]
#   device: mps (default), cuda, cpu
#   mode: full (default), quick
#   skip_figure8: 1 to skip Figure 8 (default: 1)

set -e  # Exit on error

# Parse arguments
DEVICE=${1:-mps}
MODE=${2:-full}
SKIP_FIG8=${3:-1}  # Default: SKIP Figure 8 (run separately with run_figure8_safe.sh)

# Configuration based on mode
if [ "$MODE" == "full" ]; then
    N_FEATURES=-1
    RANKS="1-60"
    SCATTER_SAMPLES=1000  # Reduced from 2000 for memory safety
    echo "=== FULL MODE ==="
    echo "  - Analyzing ALL active features"
    echo "  - Ranks: 1-60 (extended)"
    echo "  - Saving scatter data (1000 samples/feature)"
elif [ "$MODE" == "quick" ]; then
    N_FEATURES=100
    RANKS="1,2,4,8,16"
    SCATTER_SAMPLES=-1
    echo "=== QUICK MODE ==="
    echo "  - Analyzing 100 features"
    echo "  - Ranks: 1,2,4,8,16 (paper)"
    echo "  - No scatter data"
else
    echo "Invalid mode: $MODE (use 'full' or 'quick')"
    exit 1
fi

echo "  - Device: $DEVICE"
echo ""

# Create results directory
mkdir -p results/language

# Function to run with memory cleanup
run_correlation() {
    local MODEL=$1
    local LAYER=$2
    local EXPANSION=$3
    local OUTPUT=$4
    local MODEL_NAME=$(basename $MODEL)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $MODEL_NAME (Layer $LAYER, Expansion $EXPANSION)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Build command
    CMD="python src/language/verify_correlation.py \
        --config configs/language_correlation_fw.yaml \
        --model $MODEL \
        --layer $LAYER \
        --expansion $EXPANSION \
        --output $OUTPUT \
        --device $DEVICE \
        --no-wandb \
        --n-features $N_FEATURES \
        --ranks \"$RANKS\" \
        --max-batches 50"
    
    # Add scatter saving for full mode
    if [ "$MODE" == "full" ]; then
        CMD="$CMD --save-scatter --max-scatter-samples $SCATTER_SAMPLES"
    fi
    
    # Run command
    eval $CMD
    
    echo ""
    echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output: $OUTPUT"
    echo ""
    
    # Memory cleanup: Force Python garbage collection
    sleep 5
    echo "Waiting 5s for memory cleanup..."
    echo ""
}

# ============================================================================
# EXPERIMENT 1: ts-medium (smallest model, ~2-3 hours)
# ============================================================================
run_correlation \
    "tdooms/ts-medium" \
    4 \
    4 \
    "results/language/correlation_ts-medium.json"

# ============================================================================
# EXPERIMENT 2: fw-small (~2-3 hours)
# ============================================================================
run_correlation \
    "tdooms/fw-small" \
    8 \
    4 \
    "results/language/correlation_fw-small.json"

# ============================================================================
# EXPERIMENT 3: fw-medium (largest, ~3-4 hours)
# ============================================================================
run_correlation \
    "tdooms/fw-medium" \
    7 \
    8 \
    "results/language/correlation_fw-medium.json"

# ============================================================================
# EXPERIMENT 4: Figure 8 - Negation Circuit Visualization (~30-60 min)
# ============================================================================
# NOTE: ts-medium layer 4 does NOT have mlp-in SAEs available on HuggingFace
# (only mlp-out, resid-mid, resid-pre). The Tracer class requires both 
# mlp-in and mlp-out SAEs, so we can only reproduce the fw-medium example.
#
# Paper's ts-medium features (1882, 1179) were likely analyzed with internal
# SAE checkpoints not publicly released.
#
# MEMORY WARNING: This step is memory-intensive and may crash on MPS.
# Run separately with: ./scripts/run_figure8_safe.sh cpu
# ============================================================================

if [ "$SKIP_FIG8" == "1" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  SKIPPING Figure 8 (memory-intensive)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Figure 8 requires heavy memory and may crash your system."
    echo "Run it separately with CPU (safer but slower):"
    echo ""
    echo "  ./scripts/run_figure8_safe.sh cpu"
    echo ""
    echo "This will take ~45-90 minutes but won't crash your PC."
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: Figure 8 Negation Circuit (fw-medium)"
    echo "Features: 3834 (not-good), 751 (not-bad)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    python src/language/negation_visualization.py \
        --config configs/language_negation_fw.yaml \
        --output results/language/figure_8_data_fw_medium.json \
        --feature 3834 \
        --device $DEVICE

    echo ""
    echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output: results/language/figure_8_data_fw_medium.json"
    echo ""
fi

# ============================================================================
# GENERATE FIGURES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Generating Language Figures (9 & 8)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/generate_language_figures.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CORRELATION EXPERIMENTS COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to:"
echo "  - results/language/correlation_*.json (Figure 9 data)"
echo ""
echo "Figures saved to:"
echo "  - Report/figures/figure_9*.pdf (Correlation analysis)"
echo ""

if [ "$SKIP_FIG8" == "1" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  NEXT STEP: Generate Figure 8"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Figure 8 was skipped to prevent memory crashes."
    echo "Run it separately (safer on CPU):"
    echo ""
    echo "  ./scripts/run_figure8_safe.sh cpu"
    echo ""
    echo "This takes ~45-90 minutes but won't crash your PC."
    echo ""
else
    echo "All experiments including Figure 8 complete!"
    echo ""
fi

echo "NOTE: ts-medium layer 4 does not have mlp-in SAEs publicly available,"
echo "      so Figure 8 uses fw-medium (layer 7) which demonstrates the same"
echo "      negation circuit phenomenon with clearer visualizations."
echo ""
