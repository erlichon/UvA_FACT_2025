"""
Training script for Bilinear MLP experiments (Section 4: Vision).

Usage:
    python src/train.py --config configs/mnist_dense_full.yaml --seed 42
    python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2
"""

import sys
from pathlib import Path
import argparse
import torch

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config
from image.datasets import MNIST, FMNIST

from src.utils import (
    get_device,
    load_config,
    set_seed,
    track_emissions,
    init_wandb,
    finish_wandb,
    get_history_column,
)
from src.training import train_model
from src.analysis.spectral import effective_rank, spectral_summary, top_k_coverage
import wandb


def train_vision_model(config: dict, seed: int, device: str, epochs: int):
    """
    Train a bilinear vision model.

    Args:
        config: Experiment configuration
        seed: Random seed
        device: Device to train on
        epochs: Number of training epochs

    Returns:
        Tuple of (model, history, eigenvalues, eigenvectors)
    """
    # Load dataset
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    if dataset_name == 'mnist':
        print("Loading MNIST data...")
        train_data = MNIST(train=True, device=device)
        test_data = MNIST(train=False, device=device)
    elif dataset_name == 'fashion_mnist':
        print("Loading Fashion-MNIST data...")
        train_data = FMNIST(train=True, device=device)
        test_data = FMNIST(train=False, device=device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create model
    model_config = Config(
        epochs=epochs,
        d_hidden=config['model']['d_hidden'],
        wd=config['regularization']['weight_decay'],
        lr=config['training'].get('lr', 1e-3),
        seed=seed,
    )
    model = Model(model_config).to(device)

    # Train using shared training logic
    variance_corrected = config.get('model', {}).get('variance_corrected_init', True)
    noise_std = config['regularization']['noise_std']
    
    eigenvalues, eigenvectors, history = train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        device=device,
        seed=seed,
        epochs=epochs,
        noise_std=noise_std,
        variance_corrected_init=variance_corrected,
    )

    return model, history, eigenvalues, eigenvectors


def save_checkpoint(
    path: Path,
    config: dict,
    model,
    history,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    seed: int,
    epochs: int,
):
    """Save model checkpoint with eigenspectrum."""
    # Extract final metrics from history
    final_train_acc = get_history_column(history, 'train_acc', 'train/acc')
    final_val_acc = get_history_column(history, 'val_acc', 'val/acc', 'test_acc', 'test/acc')
    final_train_loss = get_history_column(history, 'train_loss', 'train/loss')
    final_val_loss = get_history_column(history, 'val_loss', 'val/loss', 'test_loss', 'test/loss')

    # Compute effective rank
    eff_rank = effective_rank(eigenvalues).mean().item()

    dataset_name = config.get('data', {}).get('dataset', 'mnist')

    checkpoint = {
        'config': {
            'mode': 'dense',
            'd_hidden': config['model']['d_hidden'],
            'epochs': epochs,
            'lr': config['training'].get('lr', 1e-3),
            'noise_std': config['regularization']['noise_std'],
            'weight_decay': config['regularization']['weight_decay'],
            'dataset': dataset_name,
            'variance_corrected_init': config.get('model', {}).get('variance_corrected_init', True),
        },
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': float(final_train_acc),
            'val_acc': float(final_val_acc),
            'train_loss': float(final_train_loss),
            'val_loss': float(final_val_loss),
            'effective_rank': float(eff_rank),
        },
        'seed': seed,
        'eigenvalues': eigenvalues.cpu(),
        'eigenvectors': eigenvectors.cpu(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


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

    # Add per-class metrics
    for cls_idx in range(len(per_class_eff_rank)):
        summary[f"effective_rank_class_{cls_idx}"] = per_class_eff_rank[cls_idx].item()

    # Log eigenvalue histogram for each class (first 3 classes to avoid clutter)
    for cls_idx in range(min(3, eigenvalues.shape[0])):
        cls_eigenvalues = eigenvalues[cls_idx].abs().cpu().numpy()
        wandb.run.summary[f"eigenvalues_class_{cls_idx}"] = wandb.Histogram(cls_eigenvalues)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Bilinear MLP")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase1/checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    config = load_config(args.config)
    config_name = Path(args.config).stem
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']

    # Determine dataset for tagging
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    tags = ["vision", dataset_name]

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=f"{config_name}_seed{args.seed}",
        config={**config, "seed": args.seed},
        device=device,
        enabled=not args.no_wandb,
        tags=tags,
    )

    # Train with tracking
    print(f"Training {config_name} with seed {args.seed}...")
    with track_emissions("fact-bilinear") as tracker:
        model, history, eigenvalues, eigenvectors = train_vision_model(
            config, args.seed, device, epochs
        )

    # Log training history (per-epoch metrics)
    log_training_history(history, wandb_enabled)

    # Log spectral metrics (comprehensive)
    spectral_metrics = log_spectral_metrics(eigenvalues, wandb_enabled)

    # Save checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / f"{config_name}_seed{args.seed}.pt"
    checkpoint = save_checkpoint(
        checkpoint_path, config, model, history,
        eigenvalues, eigenvectors, args.seed, epochs
    )
    print(f"Checkpoint saved to {checkpoint_path}")

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
    print(f"Config: {config_name}")
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
