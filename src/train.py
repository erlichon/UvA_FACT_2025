"""
Training script for Bilinear MLP experiments.

Usage:
    python src/train.py --config configs/mnist_dense_full.yaml --seed 42
    python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2
"""

import sys
from pathlib import Path
import argparse
import time
import yaml
import torch
import wandb
from codecarbon import EmissionsTracker

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config
from image.datasets import MNIST
import kornia


def load_config(path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Bilinear MLP")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--wandb-project", type=str, default="fact-bilinear")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase1/checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)
    config_name = Path(args.config).stem

    # Override epochs if specified
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']

    # Set seed
    set_seed(args.seed)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{config_name}_seed{args.seed}",
            config={
                **config,
                "seed": args.seed,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else device,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )

    # Start CO2 tracking
    tracker = EmissionsTracker(
        project_name="fact-bilinear",
        log_level="warning",
    )
    tracker.start()
    start_time = time.time()

    # Load data
    print("Loading MNIST data...")
    train_data = MNIST(train=True, device=device)
    test_data = MNIST(train=False, device=device)

    # Create model
    model_config = Config(
        epochs=epochs,
        d_hidden=config['model']['d_hidden'],
        wd=config['regularization']['weight_decay'],
        lr=config['training'].get('lr', 1e-3),
        seed=args.seed,
    )
    model = Model(model_config)
    model = model.to(device)

    # Create transform (noise augmentation)
    noise_std = config['regularization']['noise_std']
    if noise_std > 0:
        transform = kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    else:
        transform = None

    # Train
    print(f"Training {config_name} with seed {args.seed} for {epochs} epochs...")
    history = model.fit(train_data, test_data, transform=transform)

    # Get final metrics
    # Handle both 'train/acc' and 'train_acc' column naming conventions
    def get_col(df, *names):
        for name in names:
            if name in df.columns:
                return df[name].iloc[-1]
        raise KeyError(f"None of {names} found in history columns: {list(df.columns)}")

    final_train_acc = get_col(history, 'train_acc', 'train/acc')
    final_val_acc = get_col(history, 'val_acc', 'val/acc', 'test_acc', 'test/acc')
    final_train_loss = get_col(history, 'train_loss', 'train/loss')
    final_val_loss = get_col(history, 'val_loss', 'val/loss', 'test_loss', 'test/loss')

    # Stop tracking
    end_time = time.time()
    emissions_kg = tracker.stop()
    wall_time_hours = (end_time - start_time) / 3600
    gpu_hours = wall_time_hours * max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0

    # Extract eigenspectrum for metrics
    print("Computing eigendecomposition...")
    vals, vecs = model.decompose()

    # Compute effective rank
    def effective_rank(eigenvalues):
        p = eigenvalues.abs() / eigenvalues.abs().sum(dim=-1, keepdim=True)
        p = p.clamp(min=1e-10)  # Avoid log(0)
        entropy = -(p * p.log()).sum(dim=-1)
        return entropy.exp()

    eff_rank = effective_rank(vals).mean().item()

    # Log to wandb
    if not args.no_wandb:
        wandb.summary["final_train_acc"] = final_train_acc
        wandb.summary["final_val_acc"] = final_val_acc
        wandb.summary["effective_rank"] = eff_rank
        wandb.summary["wall_time_hours"] = wall_time_hours
        wandb.summary["gpu_hours"] = gpu_hours
        wandb.summary["co2_kg"] = emissions_kg
        wandb.finish()

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Save FLAT config dict for cross-compatibility with all analysis code
    checkpoint = {
        'config': {
            'mode': 'dense',
            'd_hidden': config['model']['d_hidden'],
            'epochs': epochs,
            'lr': config['training'].get('lr', 1e-3),
            'noise_std': config['regularization']['noise_std'],
            'weight_decay': config['regularization']['weight_decay'],
        },
        'model_state_dict': model.state_dict(),
        'metrics': {
            'train_acc': float(final_train_acc),
            'val_acc': float(final_val_acc),
            'train_loss': float(final_train_loss),
            'val_loss': float(final_val_loss),
            'effective_rank': float(eff_rank),
        },
        'seed': args.seed,
        'eigenvalues': vals.cpu(),
        'eigenvectors': vecs.cpu(),
    }

    checkpoint_path = checkpoint_dir / f"{config_name}_seed{args.seed}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Config: {config_name}")
    print(f"Seed: {args.seed}")
    print(f"Final Val Accuracy: {final_val_acc:.4f}")
    print(f"Effective Rank: {eff_rank:.2f}")
    print(f"Wall Time: {wall_time_hours*60:.1f} minutes")
    if torch.cuda.is_available():
        print(f"GPU Hours: {gpu_hours:.3f}")
    print(f"CO2 (kg): {emissions_kg:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
