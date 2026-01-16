#!/bin/bash
# Run correlation verification across all 3 models for Figure 9 reproduction.
#
# Paper Section 5.2 (Figure 9) uses approximately 2/3 depth for each model:
#   - ts-medium (6 layers) -> Layer 4 (paper calls it "ts-tiny")
#   - fw-small (12 layers) -> Layer 8  
#   - fw-medium (16 layers) -> Layer 7 (paper's primary model for Figure 9)
#
# Figure 9 Panels:
#   - 9A: Correlation vs rank (requires extended ranks 1-60)
#   - 9B: Histogram of rank-2 correlations (requires ALL features)
#   - 9C: Scatter plots (requires --save-scatter)
#
# Available SAEs on HuggingFace (all verified):
#   - tdooms/ts-medium-scope: expansion=4, k=30
#   - tdooms/fw-small-scope: expansion=4, k=30
#   - tdooms/fw-medium-scope: expansion=8, k=30
#
# Usage:
#   chmod +x scripts/run_language_sweep.sh
#   ./scripts/run_language_sweep.sh           # Default: MPS device
#   ./scripts/run_language_sweep.sh cuda      # Use CUDA
#   ./scripts/run_language_sweep.sh cpu       # Use CPU
#   ./scripts/run_language_sweep.sh mps quick # Quick test (100 features)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DEVICE=${1:-mps}
MODE=${2:-full}  # "full" for all features, "quick" for 100 features
OUTDIR="results/language"
CONFIG="configs/language_correlation_fw.yaml"

# Set n_features and ranks based on mode
if [ "$MODE" = "quick" ]; then
    N_FEATURES=100
    RANKS="1,2,4,8,16"
    SAVE_SCATTER=""
    echo "Mode: QUICK (100 features, limited ranks)"
else
    N_FEATURES=-1  # All features
    RANKS="1-60"   # Extended ranks for Figure 9A smooth curves
    SAVE_SCATTER="--save-scatter --max-scatter-samples 2000"
    echo "Mode: FULL (all features, ranks 1-60, scatter data)"
fi

echo "=========================================="
echo "Language Correlation Sweep (Figure 9)"
echo "=========================================="
echo "Device: $DEVICE"
echo "Mode: $MODE"
echo "Output: $OUTDIR"
echo "Start: $(date)"
echo ""

# Create output directory
mkdir -p "$OUTDIR"

# Check for conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate fact_cpu 2>/dev/null || conda activate fact 2>/dev/null || echo "Warning: Could not activate conda env"
fi

# Model configurations: "model layer expansion"
# Paper Figure 9 models with correct layers:
#   - ts-medium: 6 layers, use layer 4, expansion 4 (paper's "ts-tiny")
#   - fw-small: 12 layers, use layer 8, expansion 4
#   - fw-medium: 16 layers, use layer 7, expansion 8 (paper's primary model)
declare -a MODELS=(
    "ts-medium 4 4"
    "fw-small 8 4"
    "fw-medium 7 8"
)

# Run each model
for model_config in "${MODELS[@]}"; do
    read -r model layer expansion <<< "$model_config"
    
    echo ""
    echo "=========================================="
    echo "Running: $model (layer=$layer, expansion=$expansion)"
    echo "=========================================="
    
    python src/language/verify_correlation.py \
        --config "$CONFIG" \
        --model "tdooms/$model" \
        --layer "$layer" \
        --expansion "$expansion" \
        --k 30 \
        --output "$OUTDIR/correlation_$model.json" \
        --plot "$OUTDIR/correlation_$model.png" \
        --device "$DEVICE" \
        --no-wandb \
        --n-features "$N_FEATURES" \
        --ranks "$RANKS" \
        --n-samples 3000 \
        --max-batches 60 \
        $SAVE_SCATTER
    
    echo "Completed: $model"
done

echo ""
echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "Results saved to:"
for model_config in "${MODELS[@]}"; do
    read -r model _ _ <<< "$model_config"
    echo "  - $OUTDIR/correlation_$model.json"
    echo "  - $OUTDIR/correlation_$model.png"
done
echo ""
echo "Paper claim (Figure 9): 69% of features have >0.75 rank-2 correlation"
echo ""
echo "Generate Figure 9 panels with:"
echo "  python scripts/generate_language_figures.py"
echo ""
echo "Generate Figure 8 (negation circuit) with:"
echo "  python src/language/negation_visualization.py \\"
echo "      --config configs/language_negation_fw.yaml \\"
echo "      --feature 3834 --device $DEVICE"
