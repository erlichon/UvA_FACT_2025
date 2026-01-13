"""
Plotting utilities for bilinear MLP interpretability analysis.

This module provides reusable plotting functions for:
- Eigenspectrum visualization
- Eigenvector visualization
- Ablation study summaries
- Trade-off analysis plots
- Sample explanations (how eigenvectors explain classification)
- Interactive Plotly visualizations

Functions use matplotlib by default, with optional Plotly for interactive HTML output.
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
from src.plot_utils.explanation import (
    plot_sample_explanation,
    plot_sample_explanation_interactive,
    compute_eigenvector_activations,
    plot_eigenspectrum_with_signs,
)
from src.plot_utils.interactive import (
    plot_eigenspectrum_interactive,
    plot_eigenspectrum_per_class_interactive,
    plot_accuracy_vs_rank_interactive,
    plot_eigenvectors_interactive,
)
from src.plot_utils.style import set_publication_style, COLORS, MARKERS

__all__ = [
    # Eigenspectrum plots (matplotlib)
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
    # Sample explanation plots
    "plot_sample_explanation",
    "plot_sample_explanation_interactive",
    "compute_eigenvector_activations",
    "plot_eigenspectrum_with_signs",
    # Interactive Plotly plots
    "plot_eigenspectrum_interactive",
    "plot_eigenspectrum_per_class_interactive",
    "plot_accuracy_vs_rank_interactive",
    "plot_eigenvectors_interactive",
    # Style
    "set_publication_style",
    "COLORS",
    "MARKERS",
]
