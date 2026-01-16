"""
MNIST and Fashion-MNIST dataset wrappers with optional Center-of-Mass normalization.

These wrappers provide a consistent interface for loading MNIST-like datasets
with optional CoM centering for the Cross-Dataset Robustness experiments.
"""

import sys
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

# Add original paper code to path
_ORIG_PATH = Path(__file__).parent.parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from image.datasets import MNIST as OriginalMNIST, FMNIST as OriginalFMNIST

from .transforms import apply_com_to_batch


class MNIST(Dataset):
    """
    MNIST dataset wrapper with optional Center-of-Mass normalization.
    
    Wraps the original paper's GPU-resident MNIST loader and adds optional
    CoM centering for geometry normalization.
    
    Args:
        train: If True, use training set. Otherwise, use test set.
        device: Device to load data onto ('cuda', 'mps', 'cpu')
        apply_com: If True, apply Center-of-Mass centering to all images
        download: If True, download dataset if not found
        
    Example:
        >>> train_data = MNIST(train=True, device="cuda", apply_com=True)
        >>> x, y = train_data[0]  # x: [1, 28, 28], y: scalar
    """
    
    def __init__(
        self,
        train: bool = True,
        device: str = "cuda",
        apply_com: bool = False,
        download: bool = True,
    ):
        # Use original paper's loader
        self._dataset = OriginalMNIST(train=train, download=download, device=device)
        
        # Apply CoM if requested
        if apply_com:
            print(f"Applying Center-of-Mass normalization to MNIST ({'train' if train else 'test'})...")
            self._dataset.x = apply_com_to_batch(self._dataset.x)
        
        # Store references for compatibility
        self.x = self._dataset.x
        self.y = self._dataset.y
    
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self.x.size(0)
    
    @property
    def n_classes(self) -> int:
        return 10
    
    @property
    def class_names(self) -> list:
        return [str(i) for i in range(10)]


class FashionMNIST(Dataset):
    """
    Fashion-MNIST dataset wrapper with optional Center-of-Mass normalization.
    
    Wraps the original paper's GPU-resident Fashion-MNIST loader and adds
    optional CoM centering for geometry normalization.
    
    Args:
        train: If True, use training set. Otherwise, use test set.
        device: Device to load data onto ('cuda', 'mps', 'cpu')
        apply_com: If True, apply Center-of-Mass centering to all images
        download: If True, download dataset if not found
        
    Example:
        >>> train_data = FashionMNIST(train=True, device="mps", apply_com=False)
        >>> x, y = train_data[0]  # x: [1, 28, 28], y: scalar
    """
    
    FASHION_CLASSES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    def __init__(
        self,
        train: bool = True,
        device: str = "cuda",
        apply_com: bool = False,
        download: bool = True,
    ):
        # Use original paper's loader
        self._dataset = OriginalFMNIST(train=train, download=download, device=device)
        
        # Apply CoM if requested
        if apply_com:
            print(f"Applying Center-of-Mass normalization to Fashion-MNIST ({'train' if train else 'test'})...")
            self._dataset.x = apply_com_to_batch(self._dataset.x)
        
        # Store references for compatibility
        self.x = self._dataset.x
        self.y = self._dataset.y
    
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self.x.size(0)
    
    @property
    def n_classes(self) -> int:
        return 10
    
    @property
    def class_names(self) -> list:
        return self.FASHION_CLASSES
