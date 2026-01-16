"""
Analysis utilities for bilinear MLP interpretability.
"""

from .spectral import (
    effective_rank,
    top_k_coverage,
    eigenvalue_decay_rate,
    spectral_summary,
    load_checkpoint_eigenvalues,
)

from .visualization import (
    plot_eigenspectrum,
    plot_eigenvectors_grid,
    plot_accuracy_vs_effective_rank,
    plot_usps_transfer_comparison,
    plot_semantic_confusion_distributions,
    plot_subspace_overlap_comparison,
    plot_eigenspectra_side_by_side,
)

from .subspace import (
    principal_angles,
    compute_subspace_overlap,
    pairwise_class_similarity,
    semantic_similarity_score,
    sort_eigenvectors_by_magnitude,
    select_balanced_eigenvectors,
)

__all__ = [
    # Spectral analysis
    "effective_rank",
    "top_k_coverage",
    "eigenvalue_decay_rate",
    "spectral_summary",
    "load_checkpoint_eigenvalues",
    # Visualization
    "plot_eigenspectrum",
    "plot_eigenvectors_grid",
    "plot_accuracy_vs_effective_rank",
    "plot_usps_transfer_comparison",
    "plot_semantic_confusion_distributions",
    "plot_subspace_overlap_comparison",
    "plot_eigenspectra_side_by_side",
    # Subspace analysis
    "principal_angles",
    "compute_subspace_overlap",
    "pairwise_class_similarity",
    "semantic_similarity_score",
    "sort_eigenvectors_by_magnitude",
    "select_balanced_eigenvectors",
]
