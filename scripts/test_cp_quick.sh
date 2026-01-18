#!/bin/bash
# Quick test script for CP implementation
# Tests basic functionality without full training

set -e

echo "=========================================="
echo "CP-Bilinear Quick Test"
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

# Test 1: Python imports
echo "Test 1: Testing imports..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
sys.path.insert(0, str(Path('bilinear-decomposition-main').absolute()))

from src.models.bilinear_layer import BilinearCP
from src.models.cp_model import CPImageModel
print('✓ Imports successful')
"

# Test 2: BilinearCP forward pass
echo ""
echo "Test 2: Testing BilinearCP forward pass..."
python -c "
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path('.').absolute()))
sys.path.insert(0, str(Path('bilinear-decomposition-main').absolute()))

from src.models.bilinear_layer import BilinearCP

layer = BilinearCP(d_in=16, d_out=8, rank=4)
x = torch.randn(4, 16)
y = layer(x)
assert y.shape == (4, 8), f'Shape mismatch: {y.shape}'
print(f'✓ Forward pass works: {x.shape} -> {y.shape}')
print(f'  Factor shapes: A={layer.A.shape}, B={layer.B.shape}, C={layer.C.shape}')
print(f'  Initialization scale: max(|A|)={layer.A.abs().max():.4f}')
"

# Test 3: CPImageModel forward pass
echo ""
echo "Test 3: Testing CPImageModel forward pass..."
python -c "
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path('.').absolute()))
sys.path.insert(0, str(Path('bilinear-decomposition-main').absolute()))

from src.models.cp_model import CPImageModel

model = CPImageModel(d_hidden=64, rank=8, n_classes=10)
x = torch.randn(8, 784)
y = model(x)
assert y.shape == (8, 10), f'Shape mismatch: {y.shape}'
print(f'✓ Forward pass works: {x.shape} -> {y.shape}')
print(f'  Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Test 4: Check w_l and w_r are removed
echo ""
echo "Test 4: Verifying w_l and w_r properties are removed..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
sys.path.insert(0, str(Path('bilinear-decomposition-main').absolute()))

from src.models.bilinear_layer import BilinearCP

layer = BilinearCP(d_in=16, d_out=8, rank=4)
assert not hasattr(layer, 'w_l'), 'w_l property should not exist'
assert not hasattr(layer, 'w_r'), 'w_r property should not exist'
print('✓ w_l and w_r properties correctly removed')
"

echo ""
echo "=========================================="
echo "✓ All quick tests passed!"
echo "=========================================="
echo ""
echo "For full tests including training, run:"
echo "  python scripts/test_cp_local.py"

