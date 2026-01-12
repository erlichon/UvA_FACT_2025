"""
Plotting utilities for bilinear MLP interpretability analysis.

This module provides reusable plotting functions for:
- Eigenspectrum visualization
- Eigenvector visualization
- Ablation study summaries
- Trade-off analysis plots

All functions use matplotlib for compatibility with PDF export and cluster environments.
"""

from src.plot_utils.eigenspectrum import (
    plot_eigenspectrum_comparison,
    plot_eigenspectrum_per_class,
    plot_eigenvalue_decay,
)
from src.plot_utils.eigenvectors import (
    plot_eigenvectors_grid,
    plot_single_eigenvector,
)
from src.plot_utils.ablation import (
    plot_ablation_bars,
    plot_accuracy_vs_effective_rank,
    plot_metric_comparison,
)
from src.plot_utils.style import set_publication_style, COLORS, MARKERS

__all__ = [
    # Eigenspectrum plots
    "plot_eigenspectrum_comparison",
    "plot_eigenspectrum_per_class",
    "plot_eigenvalue_decay",
    # Eigenvector plots
    "plot_eigenvectors_grid",
    "plot_single_eigenvector",
    # Ablation plots
    "plot_ablation_bars",
    "plot_accuracy_vs_effective_rank",
    "plot_metric_comparison",
    # Style
    "set_publication_style",
    "COLORS",
    "MARKERS",
]
