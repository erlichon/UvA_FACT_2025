"""
Image transforms for bilinear MLP experiments.

This module provides reusable transforms, particularly Center-of-Mass (CoM)
normalization for standardizing input geometry across datasets.
"""

import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


def compute_center_of_mass(image: Tensor) -> Tuple[float, float]:
    """
    Compute the center of mass of an image.
    
    Formula: CoM_x = Σ(x · I(x,y)) / Σ(I(x,y))
    
    IMPORTANT: This should be applied to raw tensor (0-1 range) BEFORE
    normalization. Normalized data contains negative values which breaks
    the physics analogy (mass cannot be negative).
    
    Args:
        image: Tensor of shape [H, W] or [C, H, W] with values in [0, 1]
        
    Returns:
        Tuple of (com_y, com_x) - center of mass coordinates
    """
    # Handle channel dimension
    if image.dim() == 3:
        image = image.squeeze(0)  # Remove channel dim for grayscale
    
    h, w = image.shape
    
    # Ensure non-negative values for mass calculation
    image = image.clamp(min=0)
    
    # Total mass
    total_mass = image.sum()
    
    if total_mass < 1e-8:
        # If image is empty, return center
        return h / 2, w / 2
    
    # Create coordinate grids
    y_coords = torch.arange(h, device=image.device, dtype=image.dtype)
    x_coords = torch.arange(w, device=image.device, dtype=image.dtype)
    
    # Compute center of mass
    # CoM_y = Σ(y · I(y,x)) / Σ(I(y,x))
    com_y = (y_coords.view(-1, 1) * image).sum() / total_mass
    # CoM_x = Σ(x · I(y,x)) / Σ(I(y,x))
    com_x = (x_coords.view(1, -1) * image).sum() / total_mass
    
    return com_y.item(), com_x.item()


def shift_to_center(image: Tensor, target_center: Tuple[float, float] = None) -> Tensor:
    """
    Shift an image so its center of mass is at the target center.
    
    Uses affine transformation with bilinear interpolation to maintain
    smooth gradients for training.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W] with values in [0, 1]
        target_center: Target (y, x) for CoM. If None, uses image center.
        
    Returns:
        Shifted image tensor with same shape as input
    """
    # Add channel dimension if needed
    squeeze_output = False
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    
    c, h, w = image.shape
    
    # Default target is image center
    if target_center is None:
        target_center = (h / 2, w / 2)
    
    # Compute current center of mass
    com_y, com_x = compute_center_of_mass(image)
    
    # Compute shift needed (in pixels)
    shift_y = target_center[0] - com_y
    shift_x = target_center[1] - com_x
    
    # Normalize shift to [-1, 1] range for grid_sample
    # Note: grid_sample samples FROM input TO output. To shift content in
    # the positive direction, we need to sample from negative positions,
    # hence the negative sign.
    shift_x_norm = -2 * shift_x / w
    shift_y_norm = -2 * shift_y / h
    
    # Create identity affine matrix and add translation
    # Affine matrix: [[1, 0, tx], [0, 1, ty]]
    theta = torch.tensor([
        [1, 0, shift_x_norm],
        [0, 1, shift_y_norm]
    ], device=image.device, dtype=image.dtype)
    
    # Add batch dimension for grid_sample
    image_batch = image.unsqueeze(0)  # [1, C, H, W]
    theta_batch = theta.unsqueeze(0)  # [1, 2, 3]
    
    # Generate sampling grid and apply
    grid = F.affine_grid(theta_batch, image_batch.shape, align_corners=False)
    shifted = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # Remove batch dimension
    result = shifted.squeeze(0)
    
    # Remove channel dimension if input didn't have one
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


class CenterOfMassTransform:
    """
    Transform that shifts images so their center of mass is at the image center.
    
    This normalizes input geometry across datasets, allowing models to learn
    true shapes rather than overfitting to alignment artifacts.
    
    IMPORTANT: Apply this transform BEFORE any normalization (like ImageNet stats).
    The CoM calculation requires non-negative values (0-1 range).
    
    Usage:
        transform = CenterOfMassTransform()
        centered_image = transform(raw_image)  # raw_image should be in [0, 1]
    """
    
    def __init__(self, target_center: Tuple[float, float] = None):
        """
        Initialize the transform.
        
        Args:
            target_center: Target (y, x) coordinates for center of mass.
                          If None, uses image center.
        """
        self.target_center = target_center
    
    def __call__(self, image: Tensor) -> Tensor:
        """
        Apply center-of-mass centering to an image.
        
        Args:
            image: Tensor of shape [C, H, W] or [H, W] with values in [0, 1]
            
        Returns:
            Centered image tensor with same shape
        """
        return shift_to_center(image, self.target_center)
    
    def __repr__(self) -> str:
        return f"CenterOfMassTransform(target_center={self.target_center})"


def apply_com_to_batch(batch: Tensor) -> Tensor:
    """
    Apply center-of-mass centering to a batch of images.
    
    Args:
        batch: Tensor of shape [B, C, H, W] or [B, H, W] with values in [0, 1]
        
    Returns:
        Centered batch tensor with same shape
    """
    transform = CenterOfMassTransform()
    
    # Handle batched input
    if batch.dim() == 4:
        # [B, C, H, W]
        return torch.stack([transform(img) for img in batch])
    elif batch.dim() == 3:
        # [B, H, W] - add and remove channel dim
        batch_with_channel = batch.unsqueeze(1)  # [B, 1, H, W]
        result = torch.stack([transform(img) for img in batch_with_channel])
        return result.squeeze(1)  # [B, H, W]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {batch.dim()}D")
