"""
EMNIST dataset wrappers for Cross-Dataset Robustness experiments.

Provides EMNISTLetters (26 classes A-Z) and EMNISTDigits (10 classes 0-9)
with optional Center-of-Mass normalization.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Tuple, Optional

from .transforms import apply_com_to_batch_cached


# Letter-to-digit visual similarity mapping for semantic confusion analysis
# Based on visual similarity: which digit does a letter most resemble?
LETTER_DIGIT_SIMILARITY = {
    'O': 0,  # O looks like 0
    'I': 1,  # I looks like 1
    'Z': 2,  # Z looks like 2
    'S': 5,  # S looks like 5
    'B': 6,  # B looks like 6 (verified: models predict 6, not 8)
}

# EMNIST Letters class names (A-Z, 0-indexed)
EMNIST_LETTERS_CLASS_NAMES = [chr(ord('A') + i) for i in range(26)]

# EMNIST Digits class names (0-9)
EMNIST_DIGITS_CLASS_NAMES = [str(i) for i in range(10)]

# Combined for backward compatibility
EMNIST_CLASS_NAMES = EMNIST_LETTERS_CLASS_NAMES


class EMNISTLetters(Dataset):
    """
    EMNIST Letters dataset (26 classes: A-Z) with optional CoM normalization.
    
    GPU-resident dataset following the same pattern as the original paper's
    MNIST loader for training efficiency.
    
    CoM-transformed data is cached to avoid recomputation on subsequent runs.
    Cache location: data/cache/com/emnist_letters_{train|test}_com.pt
    
    Args:
        train: If True, use training set. Otherwise, use test set.
        device: Device to load data onto ('cuda', 'mps', 'cpu')
        apply_com: If True, apply Center-of-Mass centering to all images
        download: If True, download dataset if not found
        use_cache: If True, use cached CoM data if available (default: True)
        
    Example:
        >>> train_data = EMNISTLetters(train=True, device="cuda", apply_com=True)
        >>> x, y = train_data[0]  # x: [1, 28, 28], y: scalar (0-25 for A-Z)
    """
    
    def __init__(
        self,
        train: bool = True,
        device: str = "cuda",
        apply_com: bool = False,
        download: bool = True,
        use_cache: bool = True,
    ):
        # Load EMNIST Letters split
        dataset = datasets.EMNIST(
            root='./data',
            split='letters',
            train=train,
            download=download,
        )
        
        # Convert to GPU-resident tensors (matching original paper's pattern)
        # EMNIST images need transposing due to different orientation
        self.x = dataset.data.float().transpose(1, 2).to(device).unsqueeze(1) / 255.0
        
        # Labels are 1-indexed in EMNIST Letters, convert to 0-indexed
        self.y = (dataset.targets - 1).to(device)
        
        # Apply CoM if requested (BEFORE any normalization, on 0-1 data)
        if apply_com:
            split = 'train' if train else 'test'
            print(f"Applying Center-of-Mass normalization to EMNIST Letters ({split})...")
            self.x = apply_com_to_batch_cached(
                self.x,
                dataset_name='emnist_letters',
                split=split,
                device=device,
                use_cache=use_cache,
            )
        
        self.num_classes = 26
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self.x.size(0)
    
    @property
    def n_classes(self) -> int:
        return self.num_classes
    
    @property
    def class_names(self) -> list:
        return EMNIST_LETTERS_CLASS_NAMES


class EMNISTDigits(Dataset):
    """
    EMNIST Digits dataset (10 classes: 0-9) with optional CoM normalization.
    
    GPU-resident dataset for comparing with MNIST on the same domain (digits)
    but different data distribution.
    
    CoM-transformed data is cached to avoid recomputation on subsequent runs.
    Cache location: data/cache/com/emnist_digits_{train|test}_com.pt
    
    Args:
        train: If True, use training set. Otherwise, use test set.
        device: Device to load data onto ('cuda', 'mps', 'cpu')
        apply_com: If True, apply Center-of-Mass centering to all images
        download: If True, download dataset if not found
        use_cache: If True, use cached CoM data if available (default: True)
        
    Example:
        >>> train_data = EMNISTDigits(train=True, device="mps", apply_com=True)
        >>> x, y = train_data[0]  # x: [1, 28, 28], y: scalar (0-9)
    """
    
    def __init__(
        self,
        train: bool = True,
        device: str = "cuda",
        apply_com: bool = False,
        download: bool = True,
        use_cache: bool = True,
    ):
        # Load EMNIST Digits split
        dataset = datasets.EMNIST(
            root='./data',
            split='digits',
            train=train,
            download=download,
        )
        
        # Convert to GPU-resident tensors
        # EMNIST images need transposing due to different orientation
        self.x = dataset.data.float().transpose(1, 2).to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        
        # Apply CoM if requested (with caching)
        if apply_com:
            split = 'train' if train else 'test'
            print(f"Applying Center-of-Mass normalization to EMNIST Digits ({split})...")
            self.x = apply_com_to_batch_cached(
                self.x,
                dataset_name='emnist_digits',
                split=split,
                device=device,
                use_cache=use_cache,
            )
        
        self.num_classes = 10
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self.x.size(0)
    
    @property
    def n_classes(self) -> int:
        return self.num_classes
    
    @property
    def class_names(self) -> list:
        return EMNIST_DIGITS_CLASS_NAMES


def extract_emnist_letters(
    dataset: EMNISTLetters,
    letters: list,
) -> Tuple[Tensor, Tensor]:
    """
    Extract specific letters from an EMNIST Letters dataset.
    
    Args:
        dataset: EMNISTLetters dataset instance
        letters: List of letters to extract (e.g., ['O', 'I', 'Z', 'S', 'B'])
        
    Returns:
        Tuple of (images, labels) for the specified letters
    """
    # Convert letters to class indices (0-indexed: A=0, B=1, ...)
    indices = [ord(letter.upper()) - ord('A') for letter in letters]
    
    # Find samples for these classes
    mask = torch.zeros(len(dataset), dtype=torch.bool, device=dataset.y.device)
    for idx in indices:
        mask |= (dataset.y == idx)
    
    return dataset.x[mask], dataset.y[mask]


def get_emnist_letter_indices(letters: list) -> list:
    """
    Convert letter names to EMNIST class indices.
    
    Args:
        letters: List of letters (e.g., ['O', 'I', 'Z'])
        
    Returns:
        List of class indices (0-indexed: A=0, B=1, ...)
    """
    return [ord(letter.upper()) - ord('A') for letter in letters]


# Backward compatibility functions for existing code
def load_emnist_letters_normalized(
    device: str = "cuda",
    apply_com: bool = True,
) -> Tuple[EMNISTLetters, EMNISTLetters]:
    """
    Load EMNIST Letters train and test sets with optional CoM normalization.
    
    Args:
        device: Device to load data onto
        apply_com: If True, apply Center-of-Mass centering
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = EMNISTLetters(train=True, device=device, apply_com=apply_com)
    test = EMNISTLetters(train=False, device=device, apply_com=apply_com)
    return train, test


def load_emnist_digits_normalized(
    device: str = "cuda",
    apply_com: bool = True,
) -> Tuple[EMNISTDigits, EMNISTDigits]:
    """
    Load EMNIST Digits train and test sets with optional CoM normalization.
    
    Args:
        device: Device to load data onto
        apply_com: If True, apply Center-of-Mass centering
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = EMNISTDigits(train=True, device=device, apply_com=apply_com)
    test = EMNISTDigits(train=False, device=device, apply_com=apply_com)
    return train, test
