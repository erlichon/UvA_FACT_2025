#!/usr/bin/env python3
"""
Generate all figures for Phase 1 (Vision) reproduction.

Run from project root:
    python scripts/generate_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

# Analysis utilities
from src.analysis.spectral import (
    load_checkpoint_eigenvalues,
    load_all_checkpoints,
    aggregate_by_config,
    compute_rank_ratio,
    spectral_summary,
)

# Plotting utilities
from src.plot_utils.style import set_publication_style, COLORS, CONFIG_NAMES
from src.plot_utils.eigenspectrum import (
    plot_eigenspectrum_comparison,
    plot_eigenspectrum_per_class,
    plot_eigenvalue_decay,
)
from src.plot_utils.eigenvectors import plot_eigenvectors_grid
from src.plot_utils.ablation import (
    plot_ablation_bars,
    plot_accuracy_vs_effective_rank,
    plot_metric_comparison,
)


def main():
    # Set publication style
    set_publication_style()

    # Paths
    MNIST_CHECKPOINT_DIR = PROJECT_ROOT / "results/phase1/checkpoints"
    FASHION_CHECKPOINT_DIR = PROJECT_ROOT / "results/phase1_fashion/checkpoints"
    FIGURE_DIR = PROJECT_ROOT / "results/phase1/figures"
    REPORT_FIGURE_DIR = PROJECT_ROOT / "Report/figures"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MNIST checkpoints: {len(list(MNIST_CHECKPOINT_DIR.glob('*.pt')))} files")
    print(f"Fashion checkpoints: {len(list(FASHION_CHECKPOINT_DIR.glob('*.pt')))} files")

    # Load all checkpoints
    print("\nLoading checkpoints...")
    mnist_df = load_all_checkpoints(MNIST_CHECKPOINT_DIR, dataset="mnist")
    fashion_df = load_all_checkpoints(FASHION_CHECKPOINT_DIR, dataset="fashion")

    print(f"Loaded {len(mnist_df)} MNIST experiments")
    print(f"Loaded {len(fashion_df)} Fashion-MNIST experiments")

    # Aggregate results
    mnist_agg = aggregate_by_config(mnist_df)
    fashion_agg = aggregate_by_config(fashion_df)

    # Print summary tables
    print("\n=== MNIST Results ===")
    print(mnist_agg.to_string(index=False))
    print("\n=== Fashion-MNIST Results ===")
    print(fashion_agg.to_string(index=False))

    # --- Figure 1: Eigenspectrum Comparison ---
    print("\nGenerating eigenspectrum comparison...")
    eigenvalues_dict = {}
    for config in ["none", "noise", "wd", "full"]:
        path = MNIST_CHECKPOINT_DIR / f"mnist_dense_{config}_seed42.pt"
        vals, _ = load_checkpoint_eigenvalues(str(path))
        eigenvalues_dict[config] = vals

    fig = plot_eigenspectrum_comparison(
        eigenvalues_dict,
        title="MNIST: Eigenspectrum by Regularization Type",
        top_k=100,
        save_path=str(FIGURE_DIR / "eigenspectrum_comparison.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "eigenspectrum_comparison.pdf")
    plt.close(fig)

    # --- Figure 2: Eigenvalue Decay ---
    print("Generating eigenvalue decay plot...")
    fig = plot_eigenvalue_decay(
        eigenvalues_dict,
        top_k=30,
        title="MNIST: Normalized Eigenvalue Decay",
        save_path=str(FIGURE_DIR / "eigenvalue_decay.pdf"),
    )
    plt.close(fig)

    # --- Figure 3: Per-class Eigenspectrum ---
    print("Generating per-class eigenspectrum...")
    vals_full, _ = load_checkpoint_eigenvalues(
        str(MNIST_CHECKPOINT_DIR / "mnist_dense_full_seed42.pt")
    )
    fig = plot_eigenspectrum_per_class(
        vals_full,
        title="MNIST (Full Reg): Eigenspectrum by Digit Class",
        save_path=str(FIGURE_DIR / "eigenspectrum_per_class.pdf"),
    )
    plt.close(fig)

    # --- Figure 4: Eigenvectors (No Reg) ---
    print("Generating eigenvector plots...")
    vals_none, vecs_none = load_checkpoint_eigenvalues(
        str(MNIST_CHECKPOINT_DIR / "mnist_dense_none_seed42.pt")
    )
    vals_full, vecs_full = load_checkpoint_eigenvalues(
        str(MNIST_CHECKPOINT_DIR / "mnist_dense_full_seed42.pt")
    )

    fig = plot_eigenvectors_grid(
        vecs_none,
        vals_none,
        n_top=5,
        title="MNIST (No Reg): Top Eigenvectors",
        save_path=str(FIGURE_DIR / "eigenvectors_noreg.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "eigenvectors_noreg.pdf")
    plt.close(fig)

    # --- Figure 5: Eigenvectors (Full Reg) ---
    fig = plot_eigenvectors_grid(
        vecs_full,
        vals_full,
        n_top=5,
        title="MNIST (Full Reg): Top Eigenvectors",
        save_path=str(FIGURE_DIR / "eigenvectors_reg.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "eigenvectors_reg.pdf")
    plt.close(fig)

    # --- Figure 6: MNIST Ablation ---
    print("Generating ablation plots...")
    fig = plot_ablation_bars(
        mnist_agg,
        metrics=["accuracy", "effective_rank"],
        title="MNIST: Ablation Study",
        save_path=str(FIGURE_DIR / "mnist_ablation.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "mnist_ablation.pdf")
    plt.close(fig)

    # --- Figure 7: Fashion-MNIST Ablation ---
    fig = plot_ablation_bars(
        fashion_agg,
        metrics=["accuracy", "effective_rank"],
        title="Fashion-MNIST: Ablation Study",
        save_path=str(FIGURE_DIR / "fashion_ablation.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "fashion_ablation.pdf")
    plt.close(fig)

    # --- Figure 8: Accuracy vs Effective Rank (MNIST) ---
    print("Generating trade-off plots...")
    fig = plot_accuracy_vs_effective_rank(
        mnist_df,
        title="MNIST: Accuracy vs Interpretability",
        save_path=str(FIGURE_DIR / "accuracy_vs_effrank_mnist.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "accuracy_vs_effrank_mnist.pdf")
    plt.close(fig)

    # --- Figure 9: Accuracy vs Effective Rank (Fashion) ---
    fig = plot_accuracy_vs_effective_rank(
        fashion_df,
        title="Fashion-MNIST: Accuracy vs Interpretability",
        save_path=str(FIGURE_DIR / "accuracy_vs_effrank_fashion.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "accuracy_vs_effrank_fashion.pdf")
    plt.close(fig)

    # --- Figure 10: Cross-Dataset Comparison ---
    print("Generating cross-dataset comparison...")
    fig = plot_metric_comparison(
        [mnist_agg, fashion_agg],
        ["MNIST", "Fashion-MNIST"],
        metric="effective_rank",
        title="Effective Rank: MNIST vs Fashion-MNIST",
        save_path=str(FIGURE_DIR / "cross_dataset_effrank.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "cross_dataset_effrank.pdf")
    plt.close(fig)

    fig = plot_metric_comparison(
        [mnist_agg, fashion_agg],
        ["MNIST", "Fashion-MNIST"],
        metric="accuracy",
        title="Accuracy: MNIST vs Fashion-MNIST",
        save_path=str(FIGURE_DIR / "cross_dataset_accuracy.pdf"),
    )
    fig.savefig(REPORT_FIGURE_DIR / "cross_dataset_accuracy.pdf")
    plt.close(fig)

    # --- Save CSV results ---
    print("\nSaving CSV results...")
    mnist_agg.to_csv(FIGURE_DIR / "mnist_results.csv", index=False)
    fashion_agg.to_csv(FIGURE_DIR / "fashion_results.csv", index=False)

    # --- Print Gate Check ---
    print("\n" + "=" * 60)
    print("GATE CHECK VERIFICATION")
    print("=" * 60)

    mnist_ratio = compute_rank_ratio(mnist_df, "none", "full")
    mnist_wd_ratio = compute_rank_ratio(mnist_df, "none", "wd")
    check1 = mnist_ratio < 0.5 or mnist_wd_ratio < 0.5
    print(f"\n[{'PASS' if check1 else 'CLOSE'}] Effective rank ratio < 0.5")
    print(f"       MNIST full/none: {mnist_ratio:.3f}")
    print(f"       MNIST wd/none: {mnist_wd_ratio:.3f}")

    mnist_full_acc = mnist_df[mnist_df['config'] == 'full']['accuracy'].mean()
    check3 = 0.90 < mnist_full_acc < 0.99
    print(f"\n[{'PASS' if check3 else 'FAIL'}] Accuracy in expected range (90-99%)")
    print(f"       MNIST full reg accuracy: {mnist_full_acc*100:.1f}%")

    wd_rank = mnist_df[mnist_df['config'] == 'wd']['effective_rank'].mean()
    full_rank = mnist_df[mnist_df['config'] == 'full']['effective_rank'].mean()
    noise_rank = mnist_df[mnist_df['config'] == 'noise']['effective_rank'].mean()
    check4 = wd_rank < full_rank and wd_rank < noise_rank
    print(f"\n[{'PASS' if check4 else 'INFO'}] Weight decay most effective at reducing rank")
    print(f"       WD rank: {wd_rank:.1f}")
    print(f"       Full rank: {full_rank:.1f}")
    print(f"       Noise rank: {noise_rank:.1f}")

    print("\n" + "=" * 60)
    print(f"\nFigures saved to:")
    print(f"  - {FIGURE_DIR}")
    print(f"  - {REPORT_FIGURE_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
