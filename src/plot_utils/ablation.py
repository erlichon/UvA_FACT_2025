"""
Ablation study visualization functions.

Functions for comparing results across configurations and visualizing trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path

from src.plot_utils.style import COLORS, MARKERS, CONFIG_NAMES


def plot_ablation_bars(
    df: pd.DataFrame,
    metrics: List[str] = ["accuracy", "effective_rank"],
    title: str = "Ablation Study Results",
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot comparing metrics across configurations.

    Args:
        df: DataFrame with columns: config, {metric}_mean, {metric}_std
        metrics: List of metrics to plot (each gets a subplot)
        title: Overall figure title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Sort configs in logical order
    config_order = ["none", "noise", "wd", "full"]
    df = df.set_index("config").loc[[c for c in config_order if c in df["config"].values]].reset_index()

    x = np.arange(len(df))
    width = 0.6

    metric_labels = {
        "accuracy": ("Test Accuracy (%)", 100),
        "effective_rank": ("Effective Rank", 1),
        "top5_coverage": ("Top-5 Coverage (%)", 100),
        "top10_coverage": ("Top-10 Coverage (%)", 100),
    }

    for ax, metric in zip(axes, metrics):
        label, scale = metric_labels.get(metric, (metric, 1))

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        values = df[mean_col].values * scale
        errors = df[std_col].values * scale

        colors = [COLORS.get(c, "steelblue") for c in df["config"]]
        ax.bar(x, values, width, yerr=errors, capsize=4, color=colors, alpha=0.85)

        ax.set_xlabel("Configuration")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_NAMES.get(c, c) for c in df["config"]], rotation=30, ha="right")

        # Add value labels on bars
        for i, (v, e) in enumerate(zip(values, errors)):
            ax.annotate(
                f"{v:.1f}",
                xy=(i, v + e + 0.02 * values.max()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_accuracy_vs_effective_rank(
    df: pd.DataFrame,
    title: str = "Accuracy vs Interpretability Trade-off",
    figsize: Tuple[float, float] = (6, 5),
    save_path: Optional[str] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Scatter plot showing accuracy vs effective rank trade-off.

    Args:
        df: DataFrame with columns: config, seed, accuracy, effective_rank
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure
        annotate: Whether to add direction annotation

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    configs = df["config"].unique()

    for config in configs:
        subset = df[df["config"] == config]
        color = COLORS.get(config, "gray")
        marker = MARKERS.get(config, "o")
        label = CONFIG_NAMES.get(config, config)

        # Plot individual seeds as smaller points
        ax.scatter(
            subset["effective_rank"],
            subset["accuracy"] * 100,
            c=color,
            marker=marker,
            s=40,
            alpha=0.4,
            label=None,
        )

        # Plot mean with error bars
        mean_rank = subset["effective_rank"].mean()
        std_rank = subset["effective_rank"].std()
        mean_acc = subset["accuracy"].mean() * 100
        std_acc = subset["accuracy"].std() * 100

        ax.errorbar(
            mean_rank,
            mean_acc,
            xerr=std_rank,
            yerr=std_acc,
            fmt=marker,
            color=color,
            markersize=10,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Effective Rank (lower = more interpretable)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    
    # Position legend in lower right
    ax.legend(title="Configuration", loc="lower right", framealpha=0.9)

    # Add direction annotation in a non-overlapping position (upper right area pointing left)
    if annotate:
        ax.annotate(
            "Better\nInterpretability",
            xy=(0.15, 0.88),
            xycoords="axes fraction",
            fontsize=9,
            ha="center",
            style="italic",
            color="gray",
        )
        ax.annotate(
            "",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            xytext=(0.27, 0.79),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_metric_comparison(
    df_list: List[pd.DataFrame],
    dataset_names: List[str],
    metric: str = "effective_rank",
    title: str = "Cross-Dataset Comparison",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare a metric across multiple datasets.

    Args:
        df_list: List of DataFrames (one per dataset)
        dataset_names: Names for each dataset
        metric: Metric to compare
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    config_order = ["none", "noise", "wd", "full"]
    n_datasets = len(df_list)
    n_configs = len(config_order)
    width = 0.8 / n_datasets

    metric_labels = {
        "accuracy": ("Test Accuracy (%)", 100),
        "effective_rank": ("Effective Rank", 1),
    }
    label, scale = metric_labels.get(metric, (metric, 1))

    for i, (df, ds_name) in enumerate(zip(df_list, dataset_names)):
        df_sorted = df.set_index("config").loc[[c for c in config_order if c in df["config"].values]].reset_index()

        x = np.arange(len(df_sorted))
        offset = (i - n_datasets / 2 + 0.5) * width

        values = df_sorted[f"{metric}_mean"].values * scale
        errors = df_sorted[f"{metric}_std"].values * scale

        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=ds_name,
            alpha=0.85,
        )

    ax.set_xlabel("Configuration")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.set_xticks(np.arange(n_configs))
    ax.set_xticklabels([CONFIG_NAMES.get(c, c) for c in config_order], rotation=30, ha="right")
    ax.legend(title="Dataset")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig
