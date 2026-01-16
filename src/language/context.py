"""
Language Experiment Context - Unified context for Section 5 experiments.

This module provides a single entry point for loading models, SAEs, and creating
Tracers, eliminating boilerplate code across language experiment scripts.

Usage:
    from src.language.context import LanguageContext
    
    ctx = LanguageContext(config, device="mps")
    model = ctx.model
    sae_out = ctx.get_sae("mlp-out")
    tracer = ctx.get_tracer()
    dataloader = ctx.get_dataloader(n_samples=2000)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from sae.sae import SAE
from sae.tracer import Tracer
from datasets import load_dataset

from src.utils import (
    get_device,
    setup_mps_fallbacks,
    is_mps_device,
)


@dataclass
class SAEConfig:
    """Normalized SAE configuration."""
    name: str
    expansion: int
    k: int


class LanguageContext:
    """
    Unified context for language experiments (Section 5).
    
    Handles:
    - Model loading with lazy initialization
    - SAE loading with caching
    - Tracer creation with normalized config
    - DataLoader creation for TinyStories
    
    Attributes:
        config: Experiment configuration dict
        device: Device string (cuda/mps/cpu)
        
    Example:
        >>> config = load_config("configs/language_correlation_fw.yaml")
        >>> ctx = LanguageContext(config, device="mps")
        >>> model = ctx.model  # Lazy-loaded on first access
        >>> sae = ctx.get_sae("mlp-out")  # Cached
        >>> tracer = ctx.get_tracer()  # Uses normalized config
    """
    
    def __init__(self, config: dict, device: Optional[str] = None):
        """
        Initialize language experiment context.
        
        Args:
            config: Experiment configuration dict (from YAML)
            device: Device string. If None, auto-detect best available.
        """
        self.config = config
        self.device = get_device(device)
        self._model: Optional[Transformer] = None
        self._sae_cache: Dict[tuple, SAE] = {}
        self._tracer: Optional[Tracer] = None
        
        # Setup MPS fallbacks if needed
        if is_mps_device(self.device):
            setup_mps_fallbacks()
    
    @property
    def model_name(self) -> str:
        """Get model name from config."""
        return self.config.get("model", {}).get("pretrained", "tdooms/fw-medium")
    
    @property
    def model(self) -> Transformer:
        """
        Lazy-load model on first access.
        
        The model is cached after first load to avoid repeated initialization.
        """
        if self._model is None:
            print(f"Loading model: {self.model_name}")
            self._model = Transformer.from_pretrained(self.model_name, device=self.device)
            print(f"  d_model: {self._model.config.d_model}")
            print(f"  d_hidden: {self._model.config.d_hidden}")
            print(f"  n_layer: {self._model.config.n_layer}")
        return self._model
    
    @property
    def layer(self) -> int:
        """Get SAE layer from config."""
        return self.config.get("sae", {}).get("layer", 7)
    
    @property
    def repo_scope(self) -> str:
        """Get the SAE repository name (model-scope)."""
        return f"{self.model.config.repo}-scope"
    
    def _get_sae_config(self, point_type: str, default_name: str) -> SAEConfig:
        """
        Extract normalized SAE config for input or output.
        
        Handles both flat and nested config formats:
        
        Flat format:
            sae:
              layer: 7
              expansion: 8
              k: 30
              
        Nested format:
            sae:
              layer: 7
              input:
                name: mlp-in
                expansion: 8
                k: 30
              output:
                name: mlp-out
                expansion: 8
                k: 30
        
        Args:
            point_type: "input" or "output"
            default_name: Default point name ("mlp-in" or "mlp-out")
            
        Returns:
            Normalized SAEConfig dataclass
        """
        sae_config = self.config.get("sae", {})
        
        # Check for nested config first
        nested = sae_config.get(point_type, {})
        if nested:
            return SAEConfig(
                name=nested.get("name", default_name),
                expansion=nested.get("expansion", sae_config.get("expansion", 8)),
                k=nested.get("k", sae_config.get("k", 30)),
            )
        
        # Fall back to flat config
        return SAEConfig(
            name=sae_config.get("point", default_name),
            expansion=sae_config.get("expansion", 8),
            k=sae_config.get("k", 30),
        )
    
    @property
    def input_sae_config(self) -> SAEConfig:
        """Get normalized input SAE config."""
        return self._get_sae_config("input", "mlp-in")
    
    @property
    def output_sae_config(self) -> SAEConfig:
        """Get normalized output SAE config."""
        return self._get_sae_config("output", "mlp-out")
    
    @property
    def expansion(self) -> int:
        """Get default expansion factor."""
        return self.output_sae_config.expansion
    
    @property
    def k(self) -> int:
        """Get default top-k value."""
        return self.output_sae_config.k
    
    def get_sae(self, point: str = "mlp-out", expansion: Optional[int] = None, k: Optional[int] = None) -> SAE:
        """
        Load SAE with caching.
        
        SAEs are cached by (point, layer, expansion, k) to avoid repeated loading.
        
        Args:
            point: SAE point name ("mlp-in" or "mlp-out")
            expansion: Expansion factor. If None, use config default.
            k: Top-k value. If None, use config default.
            
        Returns:
            Loaded SAE on the correct device
        """
        # Get config for this point type
        if point == "mlp-in":
            cfg = self.input_sae_config
        else:
            cfg = self.output_sae_config
        
        # Use provided values or config defaults
        exp = expansion if expansion is not None else cfg.expansion
        top_k = k if k is not None else cfg.k
        
        cache_key = (point, self.layer, exp, top_k)
        
        if cache_key not in self._sae_cache:
            print(f"Loading SAE from {self.repo_scope}...")
            print(f"  Point: ({point}, {self.layer})")
            print(f"  Expansion: {exp}, k: {top_k}")
            
            sae = SAE.from_pretrained(
                self.repo_scope,
                point=(point, self.layer),
                expansion=exp,
                k=top_k,
            ).to(self.device)
            
            print(f"  SAE d_model: {sae.d_model}")
            print(f"  SAE d_features: {sae.d_features}")
            
            self._sae_cache[cache_key] = sae
        
        return self._sae_cache[cache_key]
    
    def get_tracer(self, force_new: bool = False) -> Tracer:
        """
        Create Tracer with normalized config.
        
        The Tracer is cached after first creation unless force_new=True.
        
        Args:
            force_new: If True, create a new Tracer even if one is cached.
            
        Returns:
            Tracer instance for interaction analysis
        """
        if self._tracer is None or force_new:
            inp_cfg = self.input_sae_config
            out_cfg = self.output_sae_config
            
            inp_dict = {"name": inp_cfg.name, "expansion": inp_cfg.expansion, "k": inp_cfg.k}
            out_dict = {"name": out_cfg.name, "expansion": out_cfg.expansion, "k": out_cfg.k}
            
            print(f"Creating Tracer for layer {self.layer}...")
            print(f"  Input SAE: {inp_dict}")
            print(f"  Output SAE: {out_dict}")
            
            self._tracer = Tracer(
                self.model, 
                self.layer, 
                inp=inp_dict, 
                out=out_dict, 
                device=self.device
            )
            
            print(f"  out_latents shape: {self._tracer.out_latents.shape}")
            print(f"  inp_latents shape: {self._tracer.inp_latents.shape}")
        
        return self._tracer
    
    def get_dataloader(
        self, 
        n_samples: int = 2000, 
        batch_size: int = 32,
        split: str = "validation",
    ) -> DataLoader:
        """
        Create TinyStories validation dataloader.
        
        Args:
            n_samples: Number of samples to load
            batch_size: Batch size for DataLoader
            split: Dataset split ("validation" or "train")
            
        Returns:
            DataLoader yielding batches with input_ids and attention_mask
        """
        print(f"Loading TinyStories {split} data...")
        
        # Load dataset
        try:
            dataset = load_dataset("roneneldan/TinyStories", split=split)
        except Exception:
            # Fall back to train split if validation doesn't exist
            dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # Sample if dataset is larger than needed
        if len(dataset) > n_samples:
            dataset = dataset.select(range(n_samples))
        
        n_ctx = self.config.get("sae", {}).get("n_ctx", 256)
        tokenizer = self.model.tokenizer
        
        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=n_ctx,
                padding="max_length",
                return_tensors="pt",
            )
        
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset.set_format("torch")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        print(f"  Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
        
        return dataloader
    
    def print_info(self) -> None:
        """Print context information for debugging."""
        print(f"\n{'='*60}")
        print("Language Context Info")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Layer: {self.layer}")
        print(f"Input SAE: {self.input_sae_config}")
        print(f"Output SAE: {self.output_sae_config}")
        print(f"Repo scope: {self.repo_scope}")
        print(f"{'='*60}\n")
