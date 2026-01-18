"""
Vision Experiment Context - Unified context for Section 4 experiments.

This module provides a single entry point for loading checkpoints, managing
directories, and saving figures, eliminating boilerplate code across vision scripts.

Usage:
    from src.vision.context import VisionContext
    
    ctx = VisionContext()
    eigenvalues_dict = ctx.load_all_configs("mnist")
    fig = plot_eigenspectrum_comparison(eigenvalues_dict, ...)
    ctx.save_figure(fig, "eigenspectrum_comparison")
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from src.vision.spectral import load_checkpoint_eigenvalues, load_checkpoint


@dataclass
class VisionContext:
    """
    Unified context for vision experiments (Section 4).
    
    Handles:
    - Checkpoint directory management
    - Figure output directory management
    - Checkpoint loading with caching
    - Figure saving to multiple directories
    
    Attributes:
        project_root: Path to project root directory
        
    Example:
        >>> ctx = VisionContext()
        >>> vals, vecs = ctx.load_eigenvalues("mnist", "full", seed=42)
        >>> eigenvalues_dict = ctx.load_all_configs("mnist")
        >>> ctx.save_figure(fig, "my_figure")
    """
    
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    # Cache for loaded checkpoints
    _checkpoint_cache: Dict[str, dict] = field(default_factory=dict, repr=False)
    
    # Standard configurations
    CONFIGS: List[str] = field(default_factory=lambda: ["none", "noise", "wd", "full"])
    SEEDS: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    
    # --- Directory Properties ---
    
    @property
    def mnist_checkpoints(self) -> Path:
        """MNIST checkpoint directory."""
        return self.project_root / "results/vision/checkpoints"
    
    @property
    def fashion_checkpoints(self) -> Path:
        """Fashion-MNIST checkpoint directory."""
        return self.project_root / "results/vision/checkpoints_fashion"
    
    @property
    def noise_sweep_checkpoints(self) -> Path:
        """Noise sweep checkpoint directory (Figure 4)."""
        return self.project_root / "results/sweeps/noise_sweep/checkpoints"
    
    @property
    def size_sweep_checkpoints(self) -> Path:
        """Model size sweep checkpoint directory (Figure 5)."""
        return self.project_root / "results/sweeps/model_size/checkpoints"
    
    @property
    def challenge_checkpoints(self) -> Path:
        """Challenge task checkpoint directory (Figure 6)."""
        return self.project_root / "results/challenge/checkpoints"
    
    @property
    def cp_checkpoints(self) -> Path:
        """CP decomposition checkpoint directory (Extension CP)."""
        return self.project_root / "results/extension_cp/checkpoints"
    
    @property
    def cp_figures(self) -> Path:
        """CP decomposition figures directory (Extension CP)."""
        return self.project_root / "results/extension_cp/figures"
    
    # Standard CP configurations
    CP_RANKS: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 784])
    CP_INIT_MODES: List[str] = field(default_factory=lambda: ["fixed", "lambda", "gated"])
    
    @property
    def figure_dir(self) -> Path:
        """Vision figures output directory."""
        return self.project_root / "results/vision/figures"
    
    @property
    def report_dir(self) -> Path:
        """Report figures directory (for LaTeX)."""
        return self.project_root / "Report/figures"
    
    @property
    def vision_figures(self) -> Path:
        """Vision figures directory."""
        return self.project_root / "results/vision/figures"
    
    # --- Checkpoint Loading ---
    
    def get_checkpoint_dir(self, dataset: str) -> Path:
        """
        Get checkpoint directory for a dataset.
        
        Args:
            dataset: "mnist" or "fashion"
            
        Returns:
            Path to checkpoint directory
        """
        if dataset == "mnist":
            return self.mnist_checkpoints
        elif dataset in ("fashion", "fashion_mnist"):
            return self.fashion_checkpoints
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_checkpoint_path(self, dataset: str, config: str, seed: int = 42) -> Path:
        """
        Get path to a specific checkpoint.
        
        Args:
            dataset: "mnist" or "fashion"
            config: "none", "noise", "wd", or "full"
            seed: Random seed
            
        Returns:
            Path to checkpoint file
        """
        ckpt_dir = self.get_checkpoint_dir(dataset)
        return ckpt_dir / f"{dataset}_dense_{config}_seed{seed}.pt"
    
    def load_checkpoint(self, dataset: str, config: str, seed: int = 42) -> dict:
        """
        Load a checkpoint with caching.
        
        Args:
            dataset: "mnist" or "fashion"
            config: "none", "noise", "wd", or "full"
            seed: Random seed
            
        Returns:
            Full checkpoint dictionary
        """
        cache_key = f"{dataset}_{config}_{seed}"
        
        if cache_key not in self._checkpoint_cache:
            path = self.get_checkpoint_path(dataset, config, seed)
            self._checkpoint_cache[cache_key] = load_checkpoint(str(path))
        
        return self._checkpoint_cache[cache_key]
    
    def load_eigenvalues(
        self, 
        dataset: str, 
        config: str, 
        seed: int = 42
    ) -> Tuple[Tensor, Tensor]:
        """
        Load eigenvalues and eigenvectors for a specific config.
        
        Args:
            dataset: "mnist" or "fashion"
            config: "none", "noise", "wd", or "full"
            seed: Random seed
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        path = self.get_checkpoint_path(dataset, config, seed)
        return load_checkpoint_eigenvalues(str(path))
    
    def load_all_configs(
        self, 
        dataset: str = "mnist", 
        seed: int = 42,
        configs: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Load eigenvalues for all configs.
        
        Args:
            dataset: "mnist" or "fashion"
            seed: Random seed
            configs: List of configs to load. Defaults to all 4.
            
        Returns:
            Dict mapping config name to eigenvalue tensor
        """
        if configs is None:
            configs = self.CONFIGS
            
        return {
            config: self.load_eigenvalues(dataset, config, seed)[0]
            for config in configs
        }
    
    def checkpoint_exists(self, dataset: str, config: str, seed: int = 42) -> bool:
        """Check if a checkpoint exists."""
        return self.get_checkpoint_path(dataset, config, seed).exists()
    
    def count_checkpoints(self, dataset: str) -> int:
        """Count available checkpoints for a dataset."""
        ckpt_dir = self.get_checkpoint_dir(dataset)
        if not ckpt_dir.exists():
            return 0
        return len(list(ckpt_dir.glob("*.pt")))
    
    # --- CP Checkpoint Loading (Extension CP) ---
    
    def get_cp_checkpoint_path(
        self, 
        rank: int, 
        init_mode: str = "lambda", 
        seed: int = 42
    ) -> Path:
        """
        Get path to a CP checkpoint.
        
        Args:
            rank: CP decomposition rank
            init_mode: CP initialization mode ("fixed", "lambda", or "gated")
            seed: Random seed
            
        Returns:
            Path to checkpoint file
        """
        return self.cp_checkpoints / f"mnist_cp_r{rank}_{init_mode}_seed{seed}.pt"
    
    def load_cp_checkpoint(
        self, 
        rank: int, 
        init_mode: str = "lambda", 
        seed: int = 42
    ) -> dict:
        """
        Load a CP checkpoint with caching.
        
        Args:
            rank: CP decomposition rank
            init_mode: CP initialization mode
            seed: Random seed
            
        Returns:
            Full checkpoint dictionary
        """
        cache_key = f"cp_{rank}_{init_mode}_{seed}"
        
        if cache_key not in self._checkpoint_cache:
            path = self.get_cp_checkpoint_path(rank, init_mode, seed)
            self._checkpoint_cache[cache_key] = load_checkpoint(str(path))
        
        return self._checkpoint_cache[cache_key]
    
    def load_cp_eigenvalues(
        self, 
        rank: int, 
        init_mode: str = "lambda", 
        seed: int = 42
    ) -> Tuple[Tensor, Tensor]:
        """
        Load eigenvalues and eigenvectors from a CP checkpoint.
        
        Args:
            rank: CP decomposition rank
            init_mode: CP initialization mode
            seed: Random seed
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        path = self.get_cp_checkpoint_path(rank, init_mode, seed)
        return load_checkpoint_eigenvalues(str(path))
    
    def load_all_cp_ranks(
        self, 
        init_mode: str = "lambda",
        seed: int = 42,
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, Tensor]:
        """
        Load eigenvalues for all CP ranks.
        
        Args:
            init_mode: CP initialization mode
            seed: Random seed
            ranks: List of ranks to load. Defaults to standard ranks.
            
        Returns:
            Dict mapping rank to eigenvalue tensor
        """
        if ranks is None:
            ranks = self.CP_RANKS
            
        return {
            rank: self.load_cp_eigenvalues(rank, init_mode, seed)[0]
            for rank in ranks
            if self.cp_checkpoint_exists(rank, init_mode, seed)
        }
    
    def load_all_cp_modes(
        self,
        rank: int,
        seed: int = 42,
        init_modes: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Load eigenvalues for all CP initialization modes at a given rank.
        
        Args:
            rank: CP decomposition rank
            seed: Random seed
            init_modes: List of init modes to load. Defaults to all modes.
            
        Returns:
            Dict mapping init_mode to eigenvalue tensor
        """
        if init_modes is None:
            init_modes = self.CP_INIT_MODES
            
        return {
            mode: self.load_cp_eigenvalues(rank, mode, seed)[0]
            for mode in init_modes
            if self.cp_checkpoint_exists(rank, mode, seed)
        }
    
    def cp_checkpoint_exists(
        self, 
        rank: int, 
        init_mode: str = "lambda", 
        seed: int = 42
    ) -> bool:
        """Check if a CP checkpoint exists."""
        return self.get_cp_checkpoint_path(rank, init_mode, seed).exists()
    
    def count_cp_checkpoints(self) -> int:
        """Count available CP checkpoints."""
        if not self.cp_checkpoints.exists():
            return 0
        return len(list(self.cp_checkpoints.glob("*.pt")))
    
    # --- Figure Saving ---
    
    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
    
    def save_figure(
        self, 
        fig: plt.Figure, 
        name: str, 
        subfolder: str = "",
        dpi: int = 300,
        close: bool = True,
    ) -> None:
        """
        Save figure to both results and report directories.
        
        Args:
            fig: matplotlib Figure
            name: Figure name (without extension)
            subfolder: Optional subfolder within figure_dir
            dpi: DPI for saved figure
            close: If True, close figure after saving
        """
        # Determine output paths
        if subfolder:
            out_path = self.figure_dir / subfolder / f"{name}.pdf"
        else:
            out_path = self.figure_dir / f"{name}.pdf"
        
        report_path = self.report_dir / f"{name}.pdf"
        
        # Ensure directories exist
        self._ensure_dir(out_path.parent)
        self._ensure_dir(report_path.parent)
        
        # Save to both locations
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        fig.savefig(report_path, bbox_inches="tight", dpi=dpi)
        
        print(f"Saved: {out_path.name}")
        
        if close:
            plt.close(fig)
    
    def save_figure_single(
        self,
        fig: plt.Figure,
        path: Path,
        dpi: int = 300,
        close: bool = True,
    ) -> None:
        """
        Save figure to a single location.
        
        Args:
            fig: matplotlib Figure
            path: Output path
            dpi: DPI for saved figure
            close: If True, close figure after saving
        """
        self._ensure_dir(path.parent)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {path}")
        
        if close:
            plt.close(fig)
    
    def save_cp_figure(
        self,
        fig: plt.Figure,
        name: str,
        dpi: int = 300,
        close: bool = True,
    ) -> None:
        """
        Save CP figure to extension_cp figures directory and report directory.
        
        Args:
            fig: matplotlib Figure
            name: Figure name (without extension)
            dpi: DPI for saved figure
            close: If True, close figure after saving
        """
        # Save to CP figures directory
        cp_path = self.cp_figures / f"{name}.pdf"
        self._ensure_dir(cp_path.parent)
        fig.savefig(cp_path, bbox_inches="tight", dpi=dpi)
        
        # Save to report directory
        report_path = self.report_dir / f"{name}.pdf"
        self._ensure_dir(report_path.parent)
        fig.savefig(report_path, bbox_inches="tight", dpi=dpi)
        
        print(f"Saved: {cp_path.name}")
        
        if close:
            plt.close(fig)
    
    # --- Utility Methods ---
    
    def print_info(self) -> None:
        """Print context information for debugging."""
        print(f"\n{'='*60}")
        print("Vision Context Info")
        print(f"{'='*60}")
        print(f"Project root: {self.project_root}")
        print(f"MNIST checkpoints: {self.count_checkpoints('mnist')} files")
        print(f"Fashion checkpoints: {self.count_checkpoints('fashion')} files")
        print(f"CP checkpoints: {self.count_cp_checkpoints()} files")
        print(f"Figure output: {self.figure_dir}")
        print(f"CP figures: {self.cp_figures}")
        print(f"Report figures: {self.report_dir}")
        print(f"{'='*60}\n")
    
    def get_class_names(self, dataset: str) -> List[str]:
        """
        Get class names for a dataset.
        
        Args:
            dataset: "mnist" or "fashion"
            
        Returns:
            List of class names
        """
        if dataset == "mnist":
            return [str(i) for i in range(10)]
        elif dataset in ("fashion", "fashion_mnist"):
            return [
                "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
