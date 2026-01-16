#!/bin/bash
# Overnight training script for Apple Silicon (MPS)
#
# NOW USES UNIFIED SCRIPTS: run_vision.sh and run_language.sh
#
# Runs ALL experiments locally with wandb logging enabled.
# All results will appear at: https://wandb.ai/itayerlich96-student/fact-bilinear
#
# PHASE 1 (Vision) - ~2.5 hours:
# - 4 MNIST configs x 5 seeds = 20 runs
# - 4 Fashion-MNIST configs x 5 seeds = 20 runs
# - 20-epoch paper comparison (1 run)
#
# PHASE 2 (Noise Sweep - Figure 4) - ~1 hour:
# - 6 noise levels (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) x 1 seed = 6 runs
#
# PHASE 3 (Language - Section 5) - ~6-8 hours:
# - Section 5.1: Negation Discovery (fw-medium, layer 7, expansion 8)
#     * Paper's features: 3834 (not+negative), 751 (not+positive)
# - Section 5.2: Full Figure 9 sweep (all 3 models):
#     * ts-medium (layer 4, expansion 4)
#     * fw-small (layer 8, expansion 4)
#     * fw-medium (layer 7, expansion 8)
#
# Total estimated time: ~9-12 hours on M1/M2/M3/M4 Mac
#
# Usage:
#   chmod +x scripts/train/run_overnight_mps.sh
#   ./scripts/train/run_overnight_mps.sh
#
# Options:
#   --skip-vision     Skip Phase 1 (Vision experiments)
#   --skip-sweep      Skip Phase 2 (Noise sweep)
#   --skip-language   Skip Phase 3 (Language experiments)
#   --quick           Quick test mode (2 epochs, 100 samples)
#
# To run in background with logging:
#   nohup ./scripts/train/run_overnight_mps.sh > logs/overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Note: Requires wandb login first (run: wandb login)
#
# PREFERRED (new unified scripts):
#   ./scripts/train/run_vision.sh all        # All vision experiments
#   ./scripts/train/run_language.sh all      # All language experiments (except Figure 8)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Parse arguments
SKIP_VISION=false
SKIP_SWEEP=false
SKIP_LANGUAGE=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-vision)
            SKIP_VISION=true
            shift
            ;;
        --skip-sweep)
            SKIP_SWEEP=true
            shift
            ;;
        --skip-language)
            SKIP_LANGUAGE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-vision] [--skip-sweep] [--skip-language] [--quick]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p results/phase1/checkpoints
mkdir -p results/phase1_fashion/checkpoints
mkdir -p results/sweeps/noise_sweep/checkpoints
mkdir -p results/language
mkdir -p logs

# Header
echo "=========================================="
echo "FACT-AI Overnight Training (MPS)"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Configuration:"
echo "  Skip Vision:   $SKIP_VISION"
echo "  Skip Sweep:    $SKIP_SWEEP"
echo "  Skip Language: $SKIP_LANGUAGE"
echo "  Quick Mode:    $QUICK_MODE"
echo ""

# Check for conda
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate fact_cpu 2>/dev/null || conda activate fact 2>/dev/null || echo "Warning: Could not activate conda env"
fi

# Verify MPS is available
echo ""
echo "Checking MPS availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
echo ""

# Quick mode settings
EPOCHS_OVERRIDE=""
LANGUAGE_SAMPLES="--n-samples 100"
if $QUICK_MODE; then
    EPOCHS_OVERRIDE="--epochs 2"
    echo "QUICK MODE: Running with 2 epochs and 100 samples"
    echo ""
else
    LANGUAGE_SAMPLES=""
fi

# Configuration arrays
CONFIGS=("none" "noise" "wd" "full")
SEEDS=(42 43 44 45 46)
NOISE_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5)

# Track progress
TOTAL_RUNS=0
if ! $SKIP_VISION; then TOTAL_RUNS=$((TOTAL_RUNS + ${#CONFIGS[@]} * ${#SEEDS[@]} * 2 + 1)); fi  # +1 for 20-epoch
if ! $SKIP_SWEEP; then TOTAL_RUNS=$((TOTAL_RUNS + ${#NOISE_LEVELS[@]})); fi
if ! $SKIP_LANGUAGE; then TOTAL_RUNS=$((TOTAL_RUNS + 4)); fi  # 1 negation + 3 correlation models

CURRENT_RUN=0
START_TIME=$(date +%s)

log_progress() {
    local phase=$1
    local task=$2
    local detail=$3
    CURRENT_RUN=$((CURRENT_RUN + 1))
    local elapsed=$(($(date +%s) - START_TIME))

    echo ""
    echo "=========================================="
    echo "[$CURRENT_RUN/$TOTAL_RUNS] $phase: $task $detail"
    echo "Elapsed: $((elapsed / 60))m"
    echo "=========================================="
}


# ============================================
# PHASE 1: VISION EXPERIMENTS
# ============================================
VISION_ELAPSED=0
if ! $SKIP_VISION; then
    echo ""
    echo "############################################"
    echo "#                                          #"
    echo "#   PHASE 1: VISION EXPERIMENTS            #"
    echo "#   (41 runs, ~2.5 hours)                  #"
    echo "#                                          #"
    echo "############################################"

    VISION_START_TIME=$(date +%s)

    # MNIST Experiments
    echo ""
    echo "--- MNIST Experiments (20 runs) ---"

    for config in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            log_progress "Vision/MNIST" "$config" "seed=$seed"

            python src/train.py \
                --config "configs/mnist_dense_${config}.yaml" \
                --seed "$seed" \
                --checkpoint-dir "results/phase1/checkpoints" \
                $EPOCHS_OVERRIDE \
                || echo "Warning: MNIST ${config} seed ${seed} failed, continuing..."
        done
    done

    # Fashion-MNIST Experiments
    echo ""
    echo "--- Fashion-MNIST Experiments (20 runs) ---"

    for config in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            log_progress "Vision/Fashion" "$config" "seed=$seed"

            python src/train.py \
                --config "configs/fashion_dense_${config}.yaml" \
                --seed "$seed" \
                --checkpoint-dir "results/phase1_fashion/checkpoints" \
                $EPOCHS_OVERRIDE \
                || echo "Warning: Fashion-MNIST ${config} seed ${seed} failed, continuing..."
        done
    done

    # 20-Epoch Paper Comparison
    echo ""
    echo "--- 20-Epoch Paper Comparison (1 run) ---"
    echo "Paper reports ~95% accuracy after 20 epochs with full regularization"
    
    log_progress "Vision/Paper" "20-epoch" "seed=42"
    
    python src/train.py \
        --config "configs/mnist_dense_full_20ep.yaml" \
        --seed 42 \
        --checkpoint-dir "results/phase1/checkpoints" \
        $EPOCHS_OVERRIDE \
        || echo "Warning: 20-epoch paper comparison failed, continuing..."

    # Vision Summary
    VISION_END_TIME=$(date +%s)
    VISION_ELAPSED=$(( (VISION_END_TIME - VISION_START_TIME) / 60 ))

    echo ""
    echo "############################################"
    echo "# PHASE 1 COMPLETE                         #"
    echo "############################################"
    echo "Vision experiments completed in ${VISION_ELAPSED} minutes"
    echo ""
else
    echo ""
    echo ">>> Skipping Phase 1 (Vision) as requested"
fi


# ============================================
# PHASE 2: NOISE SWEEP (Figure 4)
# ============================================
SWEEP_ELAPSED=0
if ! $SKIP_SWEEP; then
    echo ""
    echo "############################################"
    echo "#                                          #"
    echo "#   PHASE 2: NOISE SWEEP (Figure 4)        #"
    echo "#   (6 runs, ~1 hour)                      #"
    echo "#                                          #"
    echo "############################################"

    SWEEP_START_TIME=$(date +%s)
    SWEEP_SEED=42
    SWEEP_CHECKPOINT_DIR="results/sweeps/noise_sweep/checkpoints"

    for NOISE in "${NOISE_LEVELS[@]}"; do
        log_progress "Sweep/Noise" "noise_std=$NOISE" "seed=$SWEEP_SEED"

        CONFIG="configs/sweeps/mnist_noise_${NOISE}.yaml"

        if [ -f "$CONFIG" ]; then
            python src/train.py \
                --config "$CONFIG" \
                --seed "$SWEEP_SEED" \
                --checkpoint-dir "$SWEEP_CHECKPOINT_DIR" \
                $EPOCHS_OVERRIDE \
                || echo "Warning: Noise sweep ${NOISE} failed, continuing..."
        else
            echo "WARNING: Config not found: $CONFIG"
            echo "Run: Create configs/sweeps/mnist_noise_*.yaml files first"
        fi
    done

    SWEEP_END_TIME=$(date +%s)
    SWEEP_ELAPSED=$(( (SWEEP_END_TIME - SWEEP_START_TIME) / 60 ))

    echo ""
    echo "############################################"
    echo "# PHASE 2 COMPLETE                         #"
    echo "############################################"
    echo "Noise sweep completed in ${SWEEP_ELAPSED} minutes"
    echo ""

    # Generate Figure 4
    echo "Generating Figure 4..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/generate_vision_figures.py --section regularization || echo "Warning: Figure 4 generation failed"
else
    echo ""
    echo ">>> Skipping Phase 2 (Noise Sweep) as requested"
fi


# ============================================
# PHASE 3: LANGUAGE EXPERIMENTS
# ============================================
LANGUAGE_ELAPSED=0
NEGATION_ELAPSED=0
CORRELATION_ELAPSED=0

if ! $SKIP_LANGUAGE; then
    echo ""
    echo "############################################"
    echo "#                                          #"
    echo "#   PHASE 3: LANGUAGE EXPERIMENTS          #"
    echo "#   (4 runs, ~6-8 hours)                   #"
    echo "#                                          #"
    echo "############################################"

    LANGUAGE_START_TIME=$(date +%s)

    # --- Section 5.1: Negation Discovery (fw-medium) ---
    # NOTE: Paper's primary negation results (features 3834, 751) are from fw-medium!
    log_progress "Language" "Negation Discovery" "(fw-medium, layer 7, 50k samples)"
    echo "Section 5.1: Sentiment negation circuit discovery"
    echo "Model: fw-medium (16 layers), Layer 7, Expansion 8"
    echo "Using pretrained SAEs from HuggingFace (tdooms/fw-medium-scope)"
    echo "Paper's features: 3834 (not+negative), 751 (not+positive)"
    echo "Paper expects features with opposing directions (cosine sim < 0)"
    echo "This may take 1-2 hours on MPS..."

    python src/language/negation_discovery.py \
        --config configs/language_negation_fw.yaml \
        --output results/language/negation_fw_medium.json \
        --use-pretrained \
        $LANGUAGE_SAMPLES \
        || echo "Warning: Negation discovery (fw-medium) failed, continuing..."

    NEGATION_END_TIME=$(date +%s)
    NEGATION_ELAPSED=$(( (NEGATION_END_TIME - LANGUAGE_START_TIME) / 60 ))
    echo "Negation discovery completed in ${NEGATION_ELAPSED} minutes"

    # --- Section 5.2: Full Figure 9 Sweep (all 3 models) ---
    echo ""
    echo "Section 5.2: Figure 9 reproduction (full sweep)"
    echo "Paper claim: 69% of features have >0.75 rank-2 correlation"
    echo "Running all 3 models for complete Figure 9..."
    echo ""
    
    # Model configurations: "model layer expansion"
    # Paper Figure 9 models with correct layers (~2/3 depth):
    #   - ts-medium: 6 layers, use layer 4, expansion 4 (paper's "ts-tiny")
    #   - fw-small: 12 layers, use layer 8, expansion 4
    #   - fw-medium: 16 layers, use layer 7, expansion 8 (paper's primary model)
    declare -a CORRELATION_MODELS=(
        "ts-medium 4 4"
        "fw-small 8 4"
        "fw-medium 7 8"
    )
    
    for model_config in "${CORRELATION_MODELS[@]}"; do
        read -r model layer expansion <<< "$model_config"
        
        log_progress "Language" "Correlation" "($model, layer=$layer)"
        echo "Model: $model, Layer: $layer, Expansion: $expansion"
        
        python src/language/verify_correlation.py \
            --config configs/language_correlation_fw.yaml \
            --model "tdooms/$model" \
            --layer "$layer" \
            --expansion "$expansion" \
            --k 30 \
            --output "results/language/correlation_$model.json" \
            --plot "results/language/correlation_$model.png" \
            --device mps \
            --no-wandb \
            --n-features 100 \
            --n-samples 3000 \
            --max-batches 60 \
            $LANGUAGE_SAMPLES \
            || echo "Warning: Correlation verification ($model) failed, continuing..."
        
        echo "Completed: $model"
    done

    CORRELATION_END_TIME=$(date +%s)
    CORRELATION_ELAPSED=$(( (CORRELATION_END_TIME - NEGATION_END_TIME) / 60 ))
    echo "Full Figure 9 sweep completed in ${CORRELATION_ELAPSED} minutes"

    LANGUAGE_ELAPSED=$(( (CORRELATION_END_TIME - LANGUAGE_START_TIME) / 60 ))

    echo ""
    echo "############################################"
    echo "# PHASE 3 COMPLETE                         #"
    echo "############################################"
    echo "Language experiments completed in ${LANGUAGE_ELAPSED} minutes"
    echo ""
else
    echo ""
    echo ">>> Skipping Phase 3 (Language) as requested"
fi


# ============================================
# FINAL SUMMARY
# ============================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "############################################"
echo "#                                          #"
echo "#   ALL EXPERIMENTS COMPLETE               #"
echo "#                                          #"
echo "############################################"
echo ""
echo "End time: $(date)"
echo "Total elapsed: ${TOTAL_ELAPSED} minutes (~$((TOTAL_ELAPSED / 60)) hours)"
echo ""
echo "Time breakdown:"
echo "  - Vision (Phase 1):     ${VISION_ELAPSED:-skipped} minutes"
echo "  - Noise Sweep (Phase 2): ${SWEEP_ELAPSED:-skipped} minutes"
echo "  - Language (Phase 3):   ${LANGUAGE_ELAPSED:-skipped} minutes"
if [ "$LANGUAGE_ELAPSED" -gt 0 ] 2>/dev/null; then
    echo "    - Negation (fw-medium):       ${NEGATION_ELAPSED} minutes"
    echo "    - Figure 9 sweep (3 models):  ${CORRELATION_ELAPSED} minutes"
fi
echo ""

# Count checkpoints
MNIST_COUNT=$(ls -1 results/phase1/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')
FASHION_COUNT=$(ls -1 results/phase1_fashion/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')
SWEEP_COUNT=$(ls -1 results/sweeps/noise_sweep/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')

echo "Results saved to:"
echo "  - results/phase1/checkpoints/ (MNIST: $MNIST_COUNT/21, includes 20-epoch)"
echo "  - results/phase1_fashion/checkpoints/ (Fashion-MNIST: $FASHION_COUNT/20)"
echo "  - results/sweeps/noise_sweep/checkpoints/ (Noise Sweep: $SWEEP_COUNT/6)"
echo "  - results/language/negation_fw_medium.json (Section 5.1)"
echo "  - results/language/correlation_ts-medium.json (Section 5.2 - Figure 9)"
echo "  - results/language/correlation_fw-small.json (Section 5.2 - Figure 9)"
echo "  - results/language/correlation_fw-medium.json (Section 5.2 - Figure 9)"
echo ""

# Quick summary of vision results (sample)
echo "Vision Results Summary (sample):"
echo "---------------------------------"
for config in none full; do
    checkpoint="results/phase1/checkpoints/mnist_dense_${config}_seed42.pt"
    if [ -f "$checkpoint" ]; then
        python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
print(f'mnist_{config}: val_acc={ckpt[\"metrics\"][\"val_acc\"]:.4f}, eff_rank={ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
" 2>/dev/null || true
    fi
done

# 20-epoch paper comparison
echo ""
echo "20-Epoch Paper Comparison:"
echo "--------------------------"
checkpoint="results/phase1/checkpoints/mnist_dense_full_20ep_seed42.pt"
if [ -f "$checkpoint" ]; then
    python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
print(f'20-epoch: val_acc={ckpt[\"metrics\"][\"val_acc\"]:.4f}, eff_rank={ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
print(f'Paper reports: ~95% accuracy after 20 epochs')
" 2>/dev/null || true
else
    echo "  No 20-epoch checkpoint found"
fi

# Noise sweep summary
echo ""
echo "Noise Sweep Summary:"
echo "--------------------"
for noise in 0.0 0.5; do
    checkpoint="results/sweeps/noise_sweep/checkpoints/mnist_noise_${noise}_seed42.pt"
    if [ -f "$checkpoint" ]; then
        python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
print(f'noise={noise}: val_acc={ckpt[\"metrics\"][\"val_acc\"]:.4f}, eff_rank={ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
" 2>/dev/null || true
    fi
done

# Language results summary
echo ""
echo "Language Results Summary:"
echo "-------------------------"

# Section 5.1: Negation results (fw-medium)
echo ""
echo "Section 5.1 - Negation Discovery (fw-medium, layer 7):"
echo "Paper's expected features: 3834 (not+negative), 751 (not+positive)"
if [ -f "results/language/negation_fw_medium.json" ]; then
    python -c "
import json
with open('results/language/negation_fw_medium.json') as f:
    data = json.load(f)
if 'top_pair_analysis' in data:
    print(f'  Features: not+pos={data[\"top_pair_analysis\"][\"not_positive_feature\"]}, not+neg={data[\"top_pair_analysis\"][\"not_negative_feature\"]}')
    print(f'  Cosine similarity: {data[\"top_pair_analysis\"][\"cosine_similarity\"]:.4f}')
    print(f'  Opposing directions: {data[\"top_pair_analysis\"][\"opposing_directions\"]}')
elif 'not_positive_features' in data:
    print(f'  Top not+pos features: {data[\"not_positive_features\"][:3]}')
    print(f'  Top not+neg features: {data[\"not_negative_features\"][:3]}')
else:
    print('  Results saved (check JSON for details)')
" 2>/dev/null || echo "  Could not read results"
else
    echo "  No results found"
fi

# Section 5.2: Full Figure 9 Correlation results (all 3 models)
echo ""
echo "Section 5.2 - Figure 9 Correlation (all 3 models):"
echo "Paper claim: 69% of features have >0.75 rank-2 correlation"
echo ""
for model in ts-medium fw-small fw-medium; do
    result_file="results/language/correlation_$model.json"
    if [ -f "$result_file" ]; then
        python -c "
import json
with open('$result_file') as f:
    data = json.load(f)
s = data.get('summary', {})
corr_by_rank = s.get('correlation_by_rank', {})
n_features = s.get('n_features_analyzed', 'N/A')
if '2' in corr_by_rank:
    r2 = corr_by_rank['2']
    print(f'  $model: mean={r2[\"mean\"]:.4f}, std={r2[\"std\"]:.4f} (n={n_features})')
else:
    print(f'  $model: results saved (n={n_features})')
" 2>/dev/null || echo "  $model: Could not read results"
    else
        echo "  $model: No results found"
    fi
done

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Generate vision figures:   ./scripts/train/run_vision.sh figures"
echo "  2. Generate language figures: ./scripts/train/run_language.sh figures"
echo "  3. View wandb dashboard:      https://wandb.ai/itayerlich96-student/fact-bilinear"
echo "  4. Check language results:    cat results/language/*.json"
echo "=============================================="
echo ""
echo "Done! Check logs for any errors."
