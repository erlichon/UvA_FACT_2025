"""
Language module for Section 5 (Negation Circuit Discovery).

This module provides wrapper scripts around the original paper code in
bilinear-decomposition-main/. We do NOT reimplement SAE or transformer code.

Key components (all use original paper implementations):
- run_sae_training.py: SAE training wrapper
- negation_discovery.py: Find negation features in SAE representations
- interaction_analysis.py: Analyze Q matrices for low-rank structure
- context.py: LanguageContext class for unified experiment setup

Visualization modules:
- visualizer.py: FeatureVisualizer class for token activation highlighting
- interaction_viz.py: Q matrix heatmaps and variance explained histograms

Usage:
    # Using LanguageContext (recommended for new code):
    from src.language.context import LanguageContext
    
    ctx = LanguageContext(config, device="mps")
    model = ctx.model
    sae_out = ctx.get_sae("mlp-out")
    tracer = ctx.get_tracer()
    
    # Command-line scripts:
    python -m src.language.run_sae_training --config configs/language_sae.yaml
    python -m src.language.negation_discovery --config configs/language_negation.yaml
    python -m src.language.interaction_analysis --config configs/language_interaction.yaml

Visualization:
    from src.language.visualizer import FeatureVisualizer
    from src.language.interaction_viz import plot_interaction_matrix, plot_variance_explained_histogram
"""

from src.language.context import LanguageContext

__all__ = ["LanguageContext"]
