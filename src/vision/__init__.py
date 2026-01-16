"""
Vision analysis utilities for bilinear MLP interpretability (Section 4).

Key modules:
- spectral.py: Effective rank, eigenvalue analysis, checkpoint loading
- context.py: VisionContext class for unified experiment setup
- truncation.py: Truncation accuracy and eigenvector similarity (Figure 5)
- adversarial.py: Adversarial mask computation (Figure 7)

Usage:
    # Using VisionContext (recommended for new code):
    from src.vision.context import VisionContext
    
    ctx = VisionContext()
    eigenvalues_dict = ctx.load_all_configs("mnist")
    ctx.save_figure(fig, "my_figure")
"""

from .spectral import (
    effective_rank,
    top_k_coverage,
    eigenvalue_decay_rate,
    spectral_summary,
    load_checkpoint_eigenvalues,
)
from .context import VisionContext

__all__ = [
    "effective_rank",
    "top_k_coverage",
    "eigenvalue_decay_rate",
    "spectral_summary",
    "load_checkpoint_eigenvalues",
    "VisionContext",
]
