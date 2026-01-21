"""
USPS dataset wrapper for Cross-Dataset Transfer experiments.

USPS images are 16x16, so we upscale to 28x28 for compatibility with
MNIST-trained models. Upscaling is done BEFORE CoM centering to prevent
interpolation artifacts from shifting the center of mass.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
from typing import Tuple

from .transforms import apply_com_to_batch_cached


class USPS(Dataset):
    """
    USPS dataset with upscaling to 28x28 and optional CoM normalization.
    
    USPS images are originally 16x16. For compatibility with MNIST-trained
    models (d_input=784), we upscale to 28x28 using bilinear interpolation.
    
    IMPORTANT: Upscaling is performed BEFORE CoM centering. If we upscaled
    after centering, interpolation artifacts might slightly shift the CoM.
    
    CoM-transformed data is cached to avoid recomputation on subsequent runs.
    Cache location: data/cache/com/usps_{train|test}_com.pt
    
    Args:
        train: If True, use training set. Otherwise, use test set.
        device: Device to load data onto ('cuda', 'mps', 'cpu')
        apply_com: If True, apply Center-of-Mass centering after upscaling
        download: If True, download dataset if not found
        target_size: Target image size (default: 28 for MNIST compatibility)
        use_cache: If True, use cached CoM data if available (default: True)
        
    Example:
        >>> train_data = USPS(train=True, device="cuda", apply_com=True)
        >>> x, y = train_data[0]  # x: [1, 28, 28], y: scalar (0-9)
    """
    
    def __init__(
        self,
        train: bool = True,
        device: str = "cuda",
        apply_com: bool = False,
        download: bool = True,
        target_size: int = 28,
        use_cache: bool = True,
    ):
        # Load USPS dataset
        dataset = datasets.USPS(
            root='./data',
            train=train,
            download=download,
        )
        
        # Convert to tensor: USPS data is numpy array with shape [N, 16, 16]
        # Values are in [-1, 1], convert to [0, 1]
        x = torch.tensor(dataset.data, dtype=torch.float32)
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        x = x.unsqueeze(1)  # Add channel dim: [N, 1, 16, 16]
        
        # Step 1: Upscale from 16x16 to target_size (28x28)
        # IMPORTANT: Do this BEFORE CoM to prevent interpolation artifacts
        # from shifting the center of mass
        if target_size != 16:
            print(f"Upscaling USPS images from 16x16 to {target_size}x{target_size}...")
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False,
            )
        
        # Move to device
        self.x = x.to(device)
        self.y = torch.tensor(dataset.targets, dtype=torch.long).to(device)
        
        # Step 2: Apply CoM if requested (AFTER upscaling, on 0-1 data, with caching)
        if apply_com:
            split = 'train' if train else 'test'
            print(f"Applying Center-of-Mass normalization to USPS ({split})...")
            self.x = apply_com_to_batch_cached(
                self.x,
                dataset_name='usps',
                split=split,
                device=device,
                use_cache=use_cache,
            )
        
        self.num_classes = 10
        self.target_size = target_size
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self.x.size(0)
    
    @property
    def n_classes(self) -> int:
        return self.num_classes
    
    @property
    def class_names(self) -> list:
        return [str(i) for i in range(10)]


def load_usps_normalized(
    device: str = "cuda",
    apply_com: bool = True,
    target_size: int = 28,
) -> Tuple[USPS, USPS]:
    """
    Load USPS train and test sets with upscaling and optional CoM normalization.
    
    Args:
        device: Device to load data onto
        apply_com: If True, apply Center-of-Mass centering
        target_size: Target image size (default: 28 for MNIST compatibility)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = USPS(train=True, device=device, apply_com=apply_com, target_size=target_size)
    test = USPS(train=False, device=device, apply_com=apply_com, target_size=target_size)
    return train, test
