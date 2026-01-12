"""
Publication-quality style settings for matplotlib.

Usage:
    from src.plot_utils.style import set_publication_style, COLORS, MARKERS
    set_publication_style()
"""

import matplotlib.pyplot as plt

# Color palette for consistent styling across figures
COLORS = {
    "none": "#1f77b4",      # Blue - no regularization
    "noise": "#ff7f0e",     # Orange - noise only
    "wd": "#2ca02c",        # Green - weight decay only
    "full": "#d62728",      # Red - full regularization
    "mnist": "#1f77b4",     # Blue for MNIST
    "fashion": "#9467bd",   # Purple for Fashion-MNIST
}

# Markers for scatter plots
MARKERS = {
    "none": "o",
    "noise": "s",
    "wd": "^",
    "full": "D",
}

# Config display names
CONFIG_NAMES = {
    "none": "No Reg",
    "noise": "Noise",
    "wd": "Weight Decay",
    "full": "Full Reg",
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        # Font settings
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Grid settings
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Legend settings
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Axes settings
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
