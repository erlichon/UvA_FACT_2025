#!/usr/bin/env python3
"""
Train EMNIST models for Extension 2.

This script trains regularized bilinear models on EMNIST letters dataset,
using shared training utilities and wandb tracking like src/train.py.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add original code to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config
from src.data.cross_dataset import load_emnist_letters_normalized
from src.utils import (
    get_device,
    get_history_column,
    track_emissions,
    init_wandb,
    finish_wandb,
)
from src.training import train_model
from src.analysis.spectral import effective_rank, spectral_summary
import wandb


def log_training_history(history, wandb_enabled: bool):
    """Log per-epoch training metrics to wandb."""
    if not wandb_enabled or wandb.run is None:
        return

    # Get column names (handle both naming conventions)
    train_acc_col = 'train/acc' if 'train/acc' in history.columns else 'train_acc'
    val_acc_col = 'val/acc' if 'val/acc' in history.columns else ('test/acc' if 'test/acc' in history.columns else 'val_acc')
    train_loss_col = 'train/loss' if 'train/loss' in history.columns else 'train_loss'
    val_loss_col = 'val/loss' if 'val/loss' in history.columns else ('test/loss' if 'test/loss' in history.columns else 'val_loss')

    # Log each epoch
    for epoch in range(len(history)):
        metrics = {"epoch": epoch}
        if train_acc_col in history.columns:
            metrics["train/acc"] = history[train_acc_col].iloc[epoch]
        if val_acc_col in history.columns:
            metrics["val/acc"] = history[val_acc_col].iloc[epoch]
        if train_loss_col in history.columns:
            metrics["train/loss"] = history[train_loss_col].iloc[epoch]
        if val_loss_col in history.columns:
            metrics["val/loss"] = history[val_loss_col].iloc[epoch]
        wandb.log(metrics, step=epoch)


def log_spectral_metrics(eigenvalues: torch.Tensor, wandb_enabled: bool):
    """Log comprehensive spectral metrics to wandb summary."""
    if not wandb_enabled or wandb.run is None:
        return {}

    # Compute full spectral summary
    summary = spectral_summary(eigenvalues)

    # Compute per-class effective rank
    per_class_eff_rank = effective_rank(eigenvalues)

    # Add per-class metrics (first 10 classes to avoid clutter)
    for cls_idx in range(min(10, len(per_class_eff_rank))):
        summary[f"effective_rank_class_{cls_idx}"] = per_class_eff_rank[cls_idx].item()

    # Log eigenvalue histogram for first 3 classes
    for cls_idx in range(min(3, eigenvalues.shape[0])):
        cls_eigenvalues = eigenvalues[cls_idx].abs().cpu().numpy()
        wandb.run.summary[f"eigenvalues_class_{cls_idx}"] = wandb.Histogram(cls_eigenvalues)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train EMNIST model for Extension 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--noise-std", type=float, default=0.15, help="Input noise std dev")
    parser.add_argument("--weight-decay", type=float, default=0.5, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--d-hidden", type=int, default=256, help="Hidden dimension")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Device
    parser.add_argument("--device", type=str, default=None, 
                       help="Device (cuda/mps/cpu), defaults to auto-detect")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/extension2/checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default=None,
                       help="Custom checkpoint name (default: emnist_regularized_seed{seed}.pt)")
    
    # Initialization
    parser.add_argument("--no-variance-correction", action="store_true",
                       help="Disable variance-corrected initialization")
    
    # Wandb
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device(args.device)
    
    print(f"Training EMNIST model:")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hidden dim: {args.d_hidden}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    # Initialize wandb
    wandb_enabled = init_wandb(
        name=f"emnist_regularized_seed{args.seed}",
        config={
            "dataset": "emnist_letters",
            "d_hidden": args.d_hidden,
            "noise_std": args.noise_std,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "variance_corrected_init": not args.no_variance_correction,
        },
        device=device,
        enabled=not args.no_wandb,
        tags=["extension2", "emnist_letters"],
    )
    
    # Load EMNIST data
    print("Loading EMNIST letters data...")
    train_data, test_data = load_emnist_letters_normalized(device=device)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Classes: {train_data.num_classes} (letters A-Z)")
    print()
    
    # Create model
    model_config = Config(
        epochs=args.epochs,
        d_hidden=args.d_hidden,
        wd=args.weight_decay,
        lr=args.lr,
        seed=args.seed,
    )
    model = Model(model_config).to(device)
    
    # Train model using shared training logic with emissions tracking
    with track_emissions("fact-bilinear-emnist") as tracker:
        eigenvalues, eigenvectors, history = train_model(
            model=model,
            train_data=train_data,
            test_data=test_data,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            noise_std=args.noise_std,
            variance_corrected_init=not args.no_variance_correction,
        )
    
    # Log training history to wandb
    log_training_history(history, wandb_enabled)
    
    # Log spectral metrics to wandb
    spectral_metrics = log_spectral_metrics(eigenvalues, wandb_enabled)
    
    # Extract final metrics using utility function
    final_train_acc = get_history_column(history, 'train_acc', 'train/acc')
    final_val_acc = get_history_column(history, 'val_acc', 'val/acc', 'test_acc', 'test/acc')
    final_train_loss = get_history_column(history, 'train_loss', 'train/loss')
    final_val_loss = get_history_column(history, 'val_loss', 'val/loss', 'test_loss', 'test/loss')
    
    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = args.checkpoint_name or f"emnist_regularized_seed{args.seed}.pt"
    checkpoint_path = output_dir / checkpoint_name
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'eigenvalues': eigenvalues.cpu(),
        'eigenvectors': eigenvectors.cpu(),
        'seed': args.seed,
        'config': {
            'dataset': 'emnist_letters',
            'd_hidden': args.d_hidden,
            'noise_std': args.noise_std,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'lr': args.lr,
            'variance_corrected_init': not args.no_variance_correction,
        },
        'metrics': {
            'train_acc': float(final_train_acc),
            'val_acc': float(final_val_acc),
            'train_loss': float(final_train_loss),
            'val_loss': float(final_val_loss),
            'effective_rank': effective_rank(eigenvalues).mean().item(),
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
    
    # Finalize wandb with all metrics
    if wandb_enabled:
        extra_summary = {
            "final_train_acc": checkpoint['metrics']['train_acc'],
            "final_val_acc": checkpoint['metrics']['val_acc'],
            "effective_rank": checkpoint['metrics']['effective_rank'],
            **spectral_metrics,  # Include all spectral metrics
        }
        finish_wandb(tracker.result, extra_summary=extra_summary)
    
    # Print summary
    metrics = checkpoint['metrics']
    result = tracker.result
    print(f"\n{'='*50}")
    print(f"EMNIST Training Complete")
    print(f"Seed: {args.seed}")
    print(f"Final Val Accuracy: {metrics['val_acc']:.4f}")
    print(f"Effective Rank: {metrics['effective_rank']:.2f}")
    print(f"Wall Time: {result.wall_time_hours*60:.1f} minutes")
    if torch.cuda.is_available():
        print(f"GPU Hours: {result.gpu_hours:.3f}")
    print(f"CO2 (kg): {result.emissions_kg:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
