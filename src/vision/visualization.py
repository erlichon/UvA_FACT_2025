"""
Visualization utilities (re-exports from plot_utils for backwards compatibility).

This module provides a backwards-compatible interface to the plot_utils package.
All plotting functions have been moved to src/plot_utils/ for better organization.

For new code, import directly from src.plot_utils modules instead.
"""

# Eigenspectrum plots
from src.plot_utils.eigenspectrum import (
    plot_eigenspectrum_comparison as plot_eigenspectrum,
)

# Eigenvector plots
from src.plot_utils.eigenvectors import (
    plot_eigenvectors_grid,
)

# Trade-off plots
from src.plot_utils.ablation import (
    plot_accuracy_vs_effective_rank,
)

# Extension 2 plots
from src.plot_utils.extension2 import (
    plot_usps_transfer_comparison,
    plot_semantic_confusion_distributions,
    plot_subspace_overlap_comparison,
    plot_eigenspectra_side_by_side,
)

__all__ = [
    "plot_eigenspectrum",
    "plot_eigenvectors_grid",
    "plot_accuracy_vs_effective_rank",
    "plot_usps_transfer_comparison",
    "plot_semantic_confusion_distributions",
    "plot_subspace_overlap_comparison",
    "plot_eigenspectra_side_by_side",
]
