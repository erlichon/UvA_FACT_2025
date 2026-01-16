#!/bin/bash
# Quick test script for language experiments on MPS (Apple Silicon)
#
# Tests all Section 5 language scripts with minimal settings to verify MPS compatibility.
# Full runs are in run_overnight_mps.sh
#
# Usage:
#   chmod +x scripts/test_mps_language.sh
#   ./scripts/test_mps_language.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "FACT-AI Language MPS Quick Test"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Check for conda
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate fact_cpu 2>/dev/null || conda activate fact 2>/dev/null || echo "Warning: Could not activate conda env"
fi

# Verify MPS
echo ""
echo "Checking MPS availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# Create results directories
mkdir -p results/language
mkdir -p logs

# ============================================
# Test 1: SAE Training (very minimal)
# ============================================
echo ""
echo "=========================================="
echo "Test 1: SAE Training (minimal test)"
echo "=========================================="

# Create minimal test config
cat > /tmp/test_sae_config.yaml << 'EOF'
name: test_sae_mps
model:
  pretrained: "tdooms/ts-medium"
sae:
  point: "mlp-out"
  layer: 2
  expansion: 4
  k: 16
  n_ctx: 128
training:
  lr: 0.0001
  batch_size: 16
  out_batch: 1024
  n_batches: 32
  n_buffers: 2
data:
  n_samples: 1000
EOF

echo "Running SAE training test (2 buffers, 1000 samples)..."
python src/language/run_sae_training.py \
    --config /tmp/test_sae_config.yaml \
    --checkpoint-dir results/language/test \
    --no-wandb \
    && echo "SAE training test PASSED" \
    || { echo "SAE training test FAILED"; exit 1; }

# ============================================
# Test 2: Negation Discovery (minimal)
# ============================================
echo ""
echo "=========================================="
echo "Test 2: Negation Discovery (minimal test)"
echo "=========================================="

# Create minimal test config
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

echo "Running negation discovery test (2000 samples)..."
python src/language/negation_discovery.py \
    --config /tmp/test_negation_config.yaml \
    --output results/language/test/negation_test.json \
    --use-pretrained \
    --no-wandb \
    && echo "Negation discovery test PASSED" \
    || { echo "Negation discovery test FAILED"; exit 1; }

# ============================================
# Test 3: Interaction Analysis (minimal)
# ============================================
echo ""
echo "=========================================="
echo "Test 3: Interaction Analysis (minimal test)"
echo "=========================================="

# Create minimal test config
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
  project: false
EOF

echo "Running interaction analysis test (20 features)..."
python src/language/interaction_analysis.py \
    --config /tmp/test_interaction_config.yaml \
    --output results/language/test/interaction_test.json \
    --no-wandb \
    && echo "Interaction analysis test PASSED" \
    || { echo "Interaction analysis test FAILED"; exit 1; }

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "ALL LANGUAGE MPS TESTS PASSED"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Test results saved to results/language/test/"
echo ""
echo "You can now run the full experiments with:"
echo "  ./scripts/run_overnight_mps.sh"
