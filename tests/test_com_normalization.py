"""Test CoM normalization functionality."""

import torch
import pytest
from src.data.emnist import compute_center_of_mass, apply_center_of_mass_shift
from src.data.mnist_wrapper import MNIST
from src.data.emnist import EMNISTLetters, EMNISTDigits


def test_com_computation():
    """Test center of mass computation."""
    # Create test image: white square in corner
    img = torch.zeros(1, 1, 28, 28)
    img[0, 0, 5:10, 5:10] = 1.0
    
    cy, cx = compute_center_of_mass(img)
    # Center of 5:10 is 7.0 (middle index)
    assert abs(cy - 7.0) < 0.1
    assert abs(cx - 7.0) < 0.1


def test_com_shift():
    """Test CoM shifting centers images."""
    # Create batch of off-center images
    images = torch.zeros(5, 1, 28, 28)
    for i in range(5):
        images[i, 0, i:i+5, i:i+5] = 1.0
    
    shifted = apply_center_of_mass_shift(images)
    
    # Check all are now centered near (14, 14)
    for i in range(5):
        cy, cx = compute_center_of_mass(shifted[i])
        assert abs(cy - 14.0) < 1.0, f"Image {i}: cy={cy:.2f}, expected ~14.0"
        assert abs(cx - 14.0) < 1.0, f"Image {i}: cx={cx:.2f}, expected ~14.0"


def test_mnist_with_com():
    """Test MNIST wrapper with CoM."""
    # Load small subset for testing
    mnist_no_com = MNIST(train=False, device='cpu', apply_com=False)
    mnist_with_com = MNIST(train=False, device='cpu', apply_com=True)
    
    assert len(mnist_no_com) == len(mnist_with_com)
    assert mnist_no_com.x.shape == mnist_with_com.x.shape
    
    # CoM-normalized should have different pixel values
    # (at least some images should be shifted)
    assert not torch.allclose(mnist_no_com.x, mnist_with_com.x)


def test_emnist_letters_with_com():
    """Test EMNIST Letters with CoM."""
    # Load small subset for testing
    emnist_no_com = EMNISTLetters(train=False, device='cpu', apply_com=False, download=False)
    emnist_with_com = EMNISTLetters(train=False, device='cpu', apply_com=True, download=False)
    
    assert len(emnist_no_com) == len(emnist_with_com)
    # Different pixel values when CoM applied
    assert not torch.allclose(emnist_no_com.x, emnist_with_com.x)


def test_emnist_digits_with_com():
    """Test EMNIST Digits with CoM."""
    # Load small subset for testing
    emnist_no_com = EMNISTDigits(train=False, device='cpu', apply_com=False, download=False)
    emnist_with_com = EMNISTDigits(train=False, device='cpu', apply_com=True, download=False)
    
    assert len(emnist_no_com) == len(emnist_with_com)
    # Different pixel values when CoM applied
    assert not torch.allclose(emnist_no_com.x, emnist_with_com.x)


def test_mnist_num_classes():
    """Test MNIST wrapper has correct num_classes property."""
    mnist = MNIST(train=False, device='cpu', apply_com=False)
    assert mnist.num_classes == 10


def test_emnist_letters_num_classes():
    """Test EMNIST Letters has correct num_classes property."""
    emnist = EMNISTLetters(train=False, device='cpu', apply_com=False, download=False)
    assert emnist.num_classes == 26


def test_emnist_digits_num_classes():
    """Test EMNIST Digits has correct num_classes property."""
    emnist = EMNISTDigits(train=False, device='cpu', apply_com=False, download=False)
    assert emnist.num_classes == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
