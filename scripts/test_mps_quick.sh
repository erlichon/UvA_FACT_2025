#!/bin/bash
# Quick MPS test - runs 2 epochs to verify everything works
#
# Run this BEFORE the overnight script to catch any issues early.
#
# Usage:
#   chmod +x scripts/test_mps_quick.sh
#   ./scripts/test_mps_quick.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Quick MPS Test (2 epochs)"
echo "=========================================="
echo ""

# Check MPS
echo "1. Checking MPS availability..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS built: {torch.backends.mps.is_built()}')
    # Quick tensor test
    x = torch.randn(10, 10, device='mps')
    y = x @ x.T
    print(f'MPS tensor test: OK')
"
echo ""

# Test training
echo "2. Testing training (2 epochs, MNIST, no regularization)..."
mkdir -p results/test
python src/train.py \
    --config configs/mnist_dense_none.yaml \
    --seed 42 \
    --epochs 2 \
    --checkpoint-dir results/test \
    --no-wandb

echo ""
echo "3. Verifying checkpoint..."
python -c "
import torch
ckpt = torch.load('results/test/mnist_dense_none_seed42.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Eigenvalues shape: {ckpt[\"eigenvalues\"].shape}')
print(f'Eigenvectors shape: {ckpt[\"eigenvectors\"].shape}')
print(f'Val accuracy: {ckpt[\"metrics\"][\"val_acc\"]:.4f}')
print(f'Effective rank: {ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
"

echo ""
echo "=========================================="
echo "MPS TEST PASSED!"
echo "=========================================="
echo ""
echo "You can now run the overnight script:"
echo "  nohup ./scripts/run_overnight_mps.sh > logs/overnight_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo ""
echo "Or interactively:"
echo "  ./scripts/run_overnight_mps.sh"
