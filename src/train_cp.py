"""
Training script for CP-Bilinear models.

Usage:
    python src/train_cp.py --rank 32 --seed 42
    python src/train_cp.py --config configs/mnist_cp_r32.yaml --seed 42
    python src/train_cp.py --config configs/mnist_cp_r32.yaml --cp-init-mode gated --seed 42
    python src/train_cp.py --rank 32 --cp-init-mode gated --seed 42
"""

import sys
from pathlib import Path
import argparse
import torch
import yaml

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

# Import datasets directly from file to avoid triggering image.model import
import importlib.util
_datasets_path = PROJECT_ROOT / "bilinear-decomposition-main" / "image" / "datasets.py"
spec = importlib.util.spec_from_file_location("image_datasets", _datasets_path)
image_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_datasets)
MNIST = image_datasets.MNIST
from src.models.cp_model import CPImageModel
from src.utils import (
    get_device,
    load_config,
    set_seed,
    track_emissions,
    init_wandb,
    finish_wandb,
    get_history_column,
    setup_mps_fallbacks,
    is_mps_device,
    safe_eigh,
)
from src.analysis.spectral import effective_rank, spectral_summary
import wandb


def decompose_model_mps_safe(model) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose CP model into eigenvalues and eigenvectors with MPS compatibility.

    Args:
        model: Trained CPImageModel instance

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    device = next(model.parameters()).device

    # Use model's decompose method
    eigenvalues, eigenvectors = model.decompose()

    # Move to CPU if MPS for compatibility
    if device.type == "mps":
        eigenvalues = eigenvalues.cpu()
        eigenvectors = eigenvectors.cpu()

    return eigenvalues, eigenvectors


def train_cp_model(config: dict, seed: int, device: str, epochs: int):
    """
    Train a CP-bilinear vision model.

    Args:
        config: Experiment configuration
        seed: Random seed
        device: Device to train on
        epochs: Number of training epochs

    Returns:
        Tuple of (model, history, eigenvalues, eigenvectors)
    """
    set_seed(seed)

    # Setup MPS fallbacks if needed
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - using CPU fallback for eigendecomposition")

    # Load dataset
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    if dataset_name == 'mnist':
        print("Loading MNIST data...")
        train_data = MNIST(train=True, device=device)
        test_data = MNIST(train=False, device=device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create model
    rank = config['model']['rank']
    d_hidden = config['model'].get('d_hidden', 256)
    cp_init_mode = config['model'].get('cp_init_mode', 'lambda')  # Default to lambda for backward compatibility
    print(f"Config loaded - cp_init_mode from config: {config['model'].get('cp_init_mode', 'NOT SET')}")
    print(f"Using CP initialization mode: {cp_init_mode}")
    model = CPImageModel(d_hidden=d_hidden, rank=rank, n_classes=10, cp_init_mode=cp_init_mode).to(device)

    # Training parameters
    lr = config['training'].get('lr', 1e-3)
    weight_decay = config['regularization']['weight_decay']
    l1_coeff = config['regularization'].get('l1_coeff', 0.0)  # L1 penalty for factors B and C
    lambda_l1_coeff = config['regularization'].get('lambda_l1_coeff', 1e-1)  # L1 penalty for lambda vector
    lambda_l0_coeff = config['regularization'].get('lambda_l0_coeff', 0.0)  # L0 proxy penalty for gate logits (gated mode)
    # Note: CP models use NO noise augmentation (noise_std=0.0)

    # Train
    print(f"Training CP model (rank={rank}) for {epochs} epochs...")
    history = model.fit(
        train_data, test_data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        l1_coeff=l1_coeff,
        lambda_l1_coeff=lambda_l1_coeff,
        lambda_l0_coeff=lambda_l0_coeff,
        transform=None,  # NO noise augmentation for CP
        verbose=True
    )

    # Compute eigendecomposition (MPS-safe version)
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = decompose_model_mps_safe(model)

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
            'mode': 'cp',
            'rank': config['model']['rank'],
            'd_hidden': config['model'].get('d_hidden', 256),
            'cp_init_mode': config['model'].get('cp_init_mode', 'lambda'),
            'epochs': epochs,
            'lr': config['training'].get('lr', 1e-3),
            'noise_std': 0.0,  # CP models use no noise
            'weight_decay': config['regularization']['weight_decay'],
            'dataset': dataset_name,
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
    parser = argparse.ArgumentParser(description="Train CP-Bilinear Model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--rank", type=int, default=None, help="CP rank (overrides config)")
    parser.add_argument("--d-hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--cp-init-mode", type=str, default=None, choices=['fixed', 'lambda', 'gated'],
                        help="CP initialization mode: 'fixed', 'lambda', or 'gated' (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase2/checkpoints")
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load config or use defaults
    if args.config:
        config = load_config(args.config)
        config_name_base = Path(args.config).stem
    else:
        # Create minimal config from args
        if args.rank is None:
            raise ValueError("Must provide --rank or --config")
        config = {
            'model': {
                'mode': 'cp',
                'rank': args.rank,
                'd_hidden': args.d_hidden,
                'cp_init_mode': args.cp_init_mode if args.cp_init_mode else 'lambda',  # Use CLI arg or default
            },
            'training': {
                'epochs': 100,
                'lr': args.lr,
            },
            'regularization': {
                'noise_std': 0.0,
                'weight_decay': args.weight_decay,
            },
            'data': {
                'dataset': 'mnist',
            },
        }
        config_name_base = f"mnist_cp_r{args.rank}"

    # Override rank if provided
    if args.rank is not None:
        config['model']['rank'] = args.rank
    
    # Override cp_init_mode if provided via command line
    if args.cp_init_mode is not None:
        config['model']['cp_init_mode'] = args.cp_init_mode

    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    
    # Get cp_init_mode for checkpoint naming
    cp_init_mode = config['model'].get('cp_init_mode', 'lambda')
    
    # Ensure config_name is in format mnist_cp_r{rank} for consistent checkpoint naming
    # Extract rank from config if needed
    rank = config['model']['rank']
    config_name = f"mnist_cp_r{rank}"

    # Determine dataset for tagging
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    tags = ["vision", dataset_name, "cp"]

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
        model, history, eigenvalues, eigenvectors = train_cp_model(
            config, args.seed, device, epochs
        )

    # Log training history (per-epoch metrics)
    log_training_history(history, wandb_enabled)

    # Log spectral metrics (comprehensive)
    spectral_metrics = log_spectral_metrics(eigenvalues, wandb_enabled)

    # Save checkpoint with cp_init_mode in filename
    # Format: mnist_cp_r{rank}_{cp_init_mode}_seed{seed}.pt
    # Note: config_name is already "mnist_cp_r{rank}", so we append _{cp_init_mode}
    checkpoint_path = Path(args.checkpoint_dir) / f"{config_name}_{cp_init_mode}_seed{args.seed}.pt"
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
    print(f"CP Rank: {config['model']['rank']}")
    print(f"Final Val Accuracy: {metrics['val_acc']:.4f}")
    print(f"Effective Rank: {metrics['effective_rank']:.2f}")
    print(f"Wall Time: {result.wall_time_hours*60:.1f} minutes")
    if torch.cuda.is_available():
        print(f"GPU Hours: {result.gpu_hours:.3f}")
    print(f"CO2 (kg): {result.emissions_kg:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

