"""
Challenge Task Dataset for Figure 6.

Binary classification task from mechanistic interpretability challenge:
- Label = True if input has high cosine similarity to target OR to its complement
- Label = False otherwise

The target is a specific instance of a "1" digit from MNIST.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Float
from pathlib import Path
import sys

# Add paths for original code
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.datasets import MNIST


def cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute cosine similarity between x and y.
    
    Args:
        x: Tensor of shape [batch, d] or [d]
        y: Tensor of shape [d]
        
    Returns:
        Similarity tensor of shape [batch] or scalar
    """
    # Flatten x if needed (keeping batch dimension)
    if x.dim() > 2:
        x = x.flatten(start_dim=1)
    
    # Flatten y
    y = y.flatten()
    
    # Normalize x along last dimension
    x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    
    # Normalize y
    y_norm = y / y.norm().clamp(min=1e-10)
    
    # Compute dot product (broadcast y to match x's batch dimension)
    return torch.einsum('...d,d->...', x_norm, y_norm)


def get_target_image(train_data: MNIST, digit: int = 1, index: int = 0) -> Tensor:
    """
    Get a target image from MNIST.
    
    Args:
        train_data: MNIST dataset
        digit: Target digit class
        index: Which instance of the digit to use
        
    Returns:
        Target image tensor
    """
    mask = train_data.y == digit
    digit_images = train_data.x[mask]
    return digit_images[index]


class ChallengeDataset(Dataset):
    """
    Challenge task dataset for binary similarity classification.
    
    Labels are computed as:
    - True (1) if cosine_sim(x, target) > threshold OR cosine_sim(x, 1-target) > threshold
    - False (0) otherwise
    """
    
    def __init__(
        self,
        mnist_data: MNIST,
        target: Tensor,
        threshold: float = 0.6,
        device: str = "cpu",
    ):
        """
        Args:
            mnist_data: MNIST dataset (train or test)
            target: Target image for similarity computation
            threshold: Similarity threshold for True label
            device: Device to store data on
        """
        self.x = mnist_data.x.to(device)
        self.target = target.to(device)
        self.threshold = threshold
        self.device = device
        
        # Compute complement of target
        self.complement = 1.0 - self.target
        
        # Compute labels
        self.y = self._compute_labels()
        
    def _compute_labels(self) -> Tensor:
        """Compute binary labels based on similarity to target/complement."""
        x_flat = self.x.flatten(start_dim=1)
        target_flat = self.target.flatten()
        complement_flat = self.complement.flatten()
        
        # Compute similarities
        sim_target = cosine_similarity(x_flat, target_flat)
        sim_complement = cosine_similarity(x_flat, complement_flat)
        
        # Label is True if either similarity exceeds threshold
        labels = ((sim_target > self.threshold) | (sim_complement > self.threshold)).long()
        
        return labels
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def get_stats(self) -> dict:
        """Return statistics about the dataset."""
        n_true = (self.y == 1).sum().item()
        n_false = (self.y == 0).sum().item()
        return {
            "total": len(self),
            "n_true": n_true,
            "n_false": n_false,
            "true_ratio": n_true / len(self),
        }


def create_challenge_datasets(
    device: str = "cpu",
    threshold: float = 0.6,
    target_digit: int = 1,
    target_index: int = 0,
) -> tuple:
    """
    Create train and test datasets for the challenge task.
    
    Args:
        device: Device to store data on
        threshold: Similarity threshold for True label
        target_digit: Which digit to use as target
        target_index: Which instance of the digit
        
    Returns:
        Tuple of (train_dataset, test_dataset, target_image)
    """
    # Load MNIST
    train_mnist = MNIST(train=True, device=device)
    test_mnist = MNIST(train=False, device=device)
    
    # Get target image
    target = get_target_image(train_mnist, digit=target_digit, index=target_index)
    
    # Create challenge datasets
    train_data = ChallengeDataset(train_mnist, target, threshold=threshold, device=device)
    test_data = ChallengeDataset(test_mnist, target, threshold=threshold, device=device)
    
    return train_data, test_data, target


class ChallengeDatasetWrapper:
    """
    Wrapper to make ChallengeDataset compatible with the original training code.
    
    The original code expects dataset.x and dataset.y attributes.
    """
    
    def __init__(self, challenge_dataset: ChallengeDataset):
        self.x = challenge_dataset.x
        self.y = challenge_dataset.y
        self._dataset = challenge_dataset
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        return self._dataset[idx]
