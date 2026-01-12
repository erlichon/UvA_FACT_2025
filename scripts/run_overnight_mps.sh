#!/bin/bash
# Overnight training script for Apple Silicon (MPS)
#
# Runs all experiments locally with wandb logging enabled.
# All results will appear at: https://wandb.ai/itayerlich96-student/fact-bilinear
#
# PHASE 1 (Vision) - ~2 hours:
# - 4 MNIST configs x 5 seeds = 20 runs
# - 4 Fashion-MNIST configs x 5 seeds = 20 runs
#
# PHASE 2 (Language) - ~4-6 hours:
# - SAE Training (100 buffers, 100k samples)
# - Negation Discovery (50k samples)
# - Interaction Analysis (500 features)
#
# Total estimated time: ~6-8 hours on M1/M2/M3/M4 Mac
#
# Usage:
#   chmod +x scripts/run_overnight_mps.sh
#   ./scripts/run_overnight_mps.sh
#
# To run in background with logging:
#   nohup ./scripts/run_overnight_mps.sh > logs/overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Note: Requires wandb login first (run: wandb login)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create directories
mkdir -p results/phase1/checkpoints
mkdir -p results/phase1_fashion/checkpoints
mkdir -p results/language
mkdir -p logs

# Activate conda environment
echo "=========================================="
echo "FACT-AI Overnight Training (MPS)"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
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

# Configuration arrays
CONFIGS=("none" "noise" "wd" "full")
SEEDS=(42 43 44 45 46)

# Track progress
# TOTAL_VISION_RUNS=$((${#CONFIGS[@]} * ${#SEEDS[@]} * 2))  # 2 datasets (already completed)
TOTAL_LANGUAGE_RUNS=3  # SAE training, negation, interaction
TOTAL_RUNS=$TOTAL_LANGUAGE_RUNS  # Only language experiments remaining
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
# PHASE 1: VISION EXPERIMENTS (COMPLETED)
# ============================================
# Uncomment to re-run vision experiments

echo ""
echo "############################################"
echo "#                                          #"
echo "#   PHASE 1: VISION EXPERIMENTS            #"
echo "#   (40 runs, ~2 hours)                    #"
echo "#                                          #"
echo "############################################"

# # MNIST Experiments
echo ""
echo "--- MNIST Experiments (20 runs) ---"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log_progress "Vision/MNIST" "$config" "seed=$seed"

        python src/train.py \
            --config "configs/mnist_dense_${config}.yaml" \
            --seed "$seed" \
            --checkpoint-dir "results/phase1/checkpoints" \
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
            || echo "Warning: Fashion-MNIST ${config} seed ${seed} failed, continuing..."
    done
done

# Vision Summary
VISION_END_TIME=$(date +%s)
VISION_ELAPSED=$(( (VISION_END_TIME - START_TIME) / 60 ))

echo ""
echo "############################################"
echo "# PHASE 1 COMPLETE                         #"
echo "############################################"
echo "Vision experiments completed in ${VISION_ELAPSED} minutes"
echo ""

echo ""
echo "Note: Phase 1 (Vision) already completed. Running Phase 2 (Language) only."
echo ""

# ============================================
# PHASE 2: LANGUAGE EXPERIMENTS
# ============================================
echo ""
echo "############################################"
echo "#                                          #"
echo "#   PHASE 2: LANGUAGE EXPERIMENTS          #"
echo "#   (Section 5, ~4-6 hours)                #"
echo "#                                          #"
echo "############################################"

LANGUAGE_START_TIME=$(date +%s)

# --- SAE Training ---
log_progress "Language" "SAE Training" "(100 buffers)"
echo "Training SAE on TinyStories dataset..."
echo "This may take 2-4 hours on MPS..."

python src/language/run_sae_training.py \
    --config configs/language_sae.yaml \
    --checkpoint-dir results/language \
    || echo "Warning: SAE training failed, continuing..."

SAE_END_TIME=$(date +%s)
SAE_ELAPSED=$(( (SAE_END_TIME - LANGUAGE_START_TIME) / 60 ))
echo "SAE training completed in ${SAE_ELAPSED} minutes"

# --- Negation Discovery ---
log_progress "Language" "Negation Discovery" "(50k samples)"
echo "Analyzing negation patterns in SAE features..."
echo "This may take 30-60 minutes..."

python src/language/negation_discovery.py \
    --config configs/language_negation.yaml \
    --output results/language/negation_analysis.json \
    --use-pretrained \
    || echo "Warning: Negation discovery failed, continuing..."

NEGATION_END_TIME=$(date +%s)
NEGATION_ELAPSED=$(( (NEGATION_END_TIME - SAE_END_TIME) / 60 ))
echo "Negation discovery completed in ${NEGATION_ELAPSED} minutes"

# --- Interaction Analysis ---
log_progress "Language" "Interaction Analysis" "(500 features)"
echo "Analyzing interaction matrices for low-rank structure..."
echo "This may take 1-2 hours..."

python src/language/interaction_analysis.py \
    --config configs/language_interaction.yaml \
    --output results/language/interaction_analysis.json \
    || echo "Warning: Interaction analysis failed, continuing..."

INTERACTION_END_TIME=$(date +%s)
INTERACTION_ELAPSED=$(( (INTERACTION_END_TIME - NEGATION_END_TIME) / 60 ))
echo "Interaction analysis completed in ${INTERACTION_ELAPSED} minutes"

# ============================================
# FINAL SUMMARY
# ============================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
LANGUAGE_TOTAL=$(( (END_TIME - LANGUAGE_START_TIME) / 60 ))

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
echo "  - Vision (Phase 1):  ${VISION_ELAPSED:-skipped} minutes"  # Commented out - already completed
echo "  - Language (Phase 2): ${LANGUAGE_TOTAL} minutes"
echo "    - SAE Training:     ${SAE_ELAPSED} minutes"
echo "    - Negation:         ${NEGATION_ELAPSED} minutes"
echo "    - Interaction:      ${INTERACTION_ELAPSED} minutes"
echo ""

# Count checkpoints
MNIST_COUNT=$(ls -1 results/phase1/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')
FASHION_COUNT=$(ls -1 results/phase1_fashion/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')

echo "Results saved to:"
echo "  - results/phase1/checkpoints/ (MNIST: $MNIST_COUNT/20)"
echo "  - results/phase1_fashion/checkpoints/ (Fashion-MNIST: $FASHION_COUNT/20)"
echo "  - results/language/ (SAE, negation, interaction)"
echo ""

# Quick summary of vision results
echo "Vision Results Summary:"
echo "-----------------------"
for checkpoint in results/phase1/checkpoints/*.pt; do
    if [ -f "$checkpoint" ]; then
        name=$(basename "$checkpoint" .pt)
        python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
print(f'{\"$name\"}: val_acc={ckpt[\"metrics\"][\"val_acc\"]:.4f}, eff_rank={ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
" 2>/dev/null || true
    fi
done

# Quick summary of language results
echo ""
echo "Language Results Summary:"
echo "-------------------------"

# Negation results
if [ -f "results/language/negation_analysis.json" ]; then
    python -c "
import json
with open('results/language/negation_analysis.json') as f:
    data = json.load(f)
if 'top_pair_analysis' in data:
    print(f'Negation: not+pos={data[\"top_pair_analysis\"][\"not_positive_feature\"]}, not+neg={data[\"top_pair_analysis\"][\"not_negative_feature\"]}, cosine={data[\"top_pair_analysis\"][\"cosine_similarity\"]:.4f}')
else:
    print(f'Negation: top not+pos features: {data[\"not_positive_features\"][:5]}')
" 2>/dev/null || echo "Negation: Could not read results"
fi

# Interaction results
if [ -f "results/language/interaction_analysis.json" ]; then
    python -c "
import json
with open('results/language/interaction_analysis.json') as f:
    data = json.load(f)
s = data['summary']
print(f'Interaction: {s[\"fraction_above_075\"]*100:.1f}% >0.75 corr (paper: 69%), mean_eff_rank={s[\"mean_effective_rank\"]:.2f}')
print(f'  Claim supported: {s[\"claim_supported\"]}')
" 2>/dev/null || echo "Interaction: Could not read results"
fi

echo ""
echo "Done! Check logs for any errors."
echo "For detailed results, see:"
echo "  - results/language/negation_analysis.json"
echo "  - results/language/interaction_analysis.json"
