#!/usr/bin/env python3
"""
Local test script for CP-Bilinear implementation.

Tests:
1. BilinearCP initialization and forward pass
2. CPImageModel creation and forward pass
3. Training loop (2 epochs)
4. Eigendecomposition (decompose method)
5. Checkpoint saving and loading

Usage:
    python scripts/test_cp_local.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

# Import datasets directly to avoid transformers dependency
import importlib.util
_datasets_path = PROJECT_ROOT / "bilinear-decomposition-main" / "image" / "datasets.py"
spec = importlib.util.spec_from_file_location("image_datasets", _datasets_path)
image_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_datasets)
MNIST = image_datasets.MNIST

from src.models.bilinear_layer import BilinearCP
from src.models.cp_model import CPImageModel


def test_bilinear_cp():
    """Test BilinearCP layer initialization and forward pass."""
    print("=" * 60)
    print("Test 1: BilinearCP Layer")
    print("=" * 60)
    
    # Test initialization
    d_in, d_out, rank = 16, 8, 4
    layer = BilinearCP(d_in=d_in, d_out=d_out, rank=rank)
    
    print(f"✓ Initialized BilinearCP(d_in={d_in}, d_out={d_out}, rank={rank})")
    
    # Check factor shapes
    assert layer.A.shape == (d_in, rank), f"A shape mismatch: {layer.A.shape}"
    assert layer.B.shape == (d_in, rank), f"B shape mismatch: {layer.B.shape}"
    assert layer.C.shape == (d_out, rank), f"C shape mismatch: {layer.C.shape}"
    assert layer.lambdas.shape == (rank,), f"lambdas shape mismatch: {layer.lambdas.shape}"
    print(f"✓ Factor shapes correct: A={layer.A.shape}, B={layer.B.shape}, C={layer.C.shape}, lambdas={layer.lambdas.shape}")
    
    # Check initialization scale
    assert layer.A.abs().max() < 0.1, f"A initialization too large: {layer.A.abs().max()}"
    assert layer.B.abs().max() < 0.1, f"B initialization too large: {layer.B.abs().max()}"
    assert layer.C.abs().max() < 0.1, f"C initialization too large: {layer.C.abs().max()}"
    print(f"✓ Initialization scale correct (max abs: A={layer.A.abs().max():.4f}, B={layer.B.abs().max():.4f}, C={layer.C.abs().max():.4f})")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, d_in)
    y = layer(x)
    
    assert y.shape == (batch_size, d_out), f"Output shape mismatch: {y.shape}"
    print(f"✓ Forward pass works: input {x.shape} -> output {y.shape}")
    
    # Test gradients
    loss = y.sum()
    loss.backward()
    assert layer.A.grad is not None, "A gradient is None"
    assert layer.B.grad is not None, "B gradient is None"
    assert layer.C.grad is not None, "C gradient is None"
    assert layer.lambdas.grad is not None, "lambdas gradient is None"
    print("✓ Gradients flow correctly")
    
    # Test that w_l and w_r properties don't exist
    assert not hasattr(layer, 'w_l'), "w_l property should not exist"
    assert not hasattr(layer, 'w_r'), "w_r property should not exist"
    print("✓ w_l and w_r properties removed (as required)")
    
    print("✓ All BilinearCP tests passed!\n")
    return True


def test_cp_image_model():
    """Test CPImageModel creation and forward pass."""
    print("=" * 60)
    print("Test 2: CPImageModel")
    print("=" * 60)
    
    # Test initialization
    d_hidden, rank, n_classes = 64, 8, 10
    model = CPImageModel(d_hidden=d_hidden, rank=rank, n_classes=n_classes)
    
    print(f"✓ Initialized CPImageModel(d_hidden={d_hidden}, rank={rank}, n_classes={n_classes})")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 784)
    y = model(x)
    
    assert y.shape == (batch_size, n_classes), f"Output shape mismatch: {y.shape}"
    print(f"✓ Forward pass works: input {x.shape} -> output {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {total_params:,} parameters")
    
    print("✓ All CPImageModel creation tests passed!\n")
    return True


def test_training_loop():
    """Test training loop with 2 epochs."""
    print("=" * 60)
    print("Test 3: Training Loop (2 epochs)")
    print("=" * 60)
    
    # Create model
    model = CPImageModel(d_hidden=64, rank=8, n_classes=10)
    
    # Create dummy data (smaller for faster testing)
    device = 'cpu'
    train_x = torch.randn(100, 784, device=device)
    train_y = torch.randint(0, 10, (100,), device=device)
    test_x = torch.randn(50, 784, device=device)
    test_y = torch.randint(0, 10, (50,), device=device)
    
    # Create dataset-like objects
    class DummyDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    train_data = DummyDataset(train_x, train_y)
    test_data = DummyDataset(test_x, test_y)
    
    print("✓ Created model and dummy data")
    
    # Train for 2 epochs
    print("Training for 2 epochs...")
    history = model.fit(
        train_data, test_data,
        epochs=2,
        lr=1e-3,
        weight_decay=0.1,
        l1_coeff=0.0,  # No L1 penalty for this test
        transform=None,
        verbose=True
    )
    
    assert len(history) == 2, f"History length mismatch: {len(history)}"
    assert 'train_acc' in history.columns, "train_acc column missing"
    assert 'val_acc' in history.columns, "val_acc column missing"
    print(f"✓ Training completed: final train_acc={history['train_acc'].iloc[-1]:.4f}, val_acc={history['val_acc'].iloc[-1]:.4f}")
    
    # Test with L1 penalty
    print("\nTesting with L1 penalty (l1_coeff=0.01)...")
    model2 = CPImageModel(d_hidden=64, rank=8, n_classes=10)
    history2 = model2.fit(
        train_data, test_data,
        epochs=2,
        lr=1e-3,
        weight_decay=0.1,
        l1_coeff=0.01,  # With L1 penalty
        transform=None,
        verbose=False
    )
    print("✓ Training with L1 penalty works")
    
    print("✓ All training loop tests passed!\n")
    return True


def test_decompose():
    """Test eigendecomposition (decompose method)."""
    print("=" * 60)
    print("Test 4: Eigendecomposition (decompose)")
    print("=" * 60)
    
    # Create and train a small model
    model = CPImageModel(d_hidden=64, rank=8, n_classes=10)
    
    # Create dummy data
    device = 'cpu'
    train_x = torch.randn(100, 784, device=device)
    train_y = torch.randint(0, 10, (100,), device=device)
    test_x = torch.randn(50, 784, device=device)
    test_y = torch.randint(0, 10, (50,), device=device)
    
    class DummyDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    train_data = DummyDataset(train_x, train_y)
    test_data = DummyDataset(test_x, test_y)
    
    # Train briefly
    print("Training model briefly...")
    model.fit(train_data, test_data, epochs=1, lr=1e-3, weight_decay=0.1, l1_coeff=0.0, verbose=False)
    
    # Test decompose
    print("Testing decompose()...")
    eigenvalues, eigenvectors = model.decompose()
    
    assert eigenvalues.shape == (10, 784), f"Eigenvalues shape mismatch: {eigenvalues.shape}"
    assert eigenvectors.shape == (10, 784, 784), f"Eigenvectors shape mismatch: {eigenvectors.shape}"
    print(f"✓ Decompose works: eigenvalues {eigenvalues.shape}, eigenvectors {eigenvectors.shape}")
    
    # Check that eigenvalues are real (imaginary part should be ~0)
    assert eigenvalues.imag.abs().max() < 1e-5, f"Eigenvalues have large imaginary part: {eigenvalues.imag.abs().max()}"
    print("✓ Eigenvalues are real")
    
    # Check that eigenvectors are normalized (approximately)
    for c in range(10):
        vec_norm = eigenvectors[c, 0].norm().item()
        assert abs(vec_norm - 1.0) < 0.1, f"Eigenvector {c} not normalized: {vec_norm}"
    print("✓ Eigenvectors are normalized")
    
    print("✓ All decompose tests passed!\n")
    return True


def test_checkpoint():
    """Test checkpoint saving and loading."""
    print("=" * 60)
    print("Test 5: Checkpoint Saving/Loading")
    print("=" * 60)
    
    import tempfile
    from pathlib import Path
    
    # Create model and train briefly
    model = CPImageModel(d_hidden=64, rank=8, n_classes=10)
    
    device = 'cpu'
    train_x = torch.randn(50, 784, device=device)
    train_y = torch.randint(0, 10, (50,), device=device)
    test_x = torch.randn(20, 784, device=device)
    test_y = torch.randint(0, 10, (20,), device=device)
    
    class DummyDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    train_data = DummyDataset(train_x, train_y)
    test_data = DummyDataset(test_x, test_y)
    
    model.fit(train_data, test_data, epochs=1, lr=1e-3, weight_decay=0.1, l1_coeff=0.0, verbose=False)
    
    # Get eigenvalues/eigenvectors
    eigenvalues, eigenvectors = model.decompose()
    
    # Save checkpoint
    checkpoint_dir = Path(tempfile.mkdtemp())
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    
    checkpoint = {
        'config': {
            'mode': 'cp',
            'rank': 8,
            'd_hidden': 64,
            'epochs': 1,
            'lr': 1e-3,
            'noise_std': 0.0,
            'weight_decay': 0.1,
        },
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': 0.5,
            'val_acc': 0.5,
            'effective_rank': 8.0,
        },
        'seed': 42,
        'eigenvalues': eigenvalues.cpu(),
        'eigenvectors': eigenvectors.cpu(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    assert 'config' in loaded_checkpoint, "config missing from checkpoint"
    assert 'model_state_dict' in loaded_checkpoint, "model_state_dict missing from checkpoint"
    assert 'eigenvalues' in loaded_checkpoint, "eigenvalues missing from checkpoint"
    assert 'eigenvectors' in loaded_checkpoint, "eigenvectors missing from checkpoint"
    print("✓ Checkpoint loaded successfully")
    
    # Verify shapes
    assert loaded_checkpoint['eigenvalues'].shape == (10, 784), "Eigenvalues shape mismatch"
    assert loaded_checkpoint['eigenvectors'].shape == (10, 784, 784), "Eigenvectors shape mismatch"
    print("✓ Checkpoint shapes correct")
    
    print("✓ All checkpoint tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CP-Bilinear Local Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("BilinearCP Layer", test_bilinear_cp),
        ("CPImageModel", test_cp_image_model),
        ("Training Loop", test_training_loop),
        ("Eigendecomposition", test_decompose),
        ("Checkpoint", test_checkpoint),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
    
    all_passed = all(passed for _, passed, _ in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

