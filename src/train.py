"""
Training script for Bilinear MLP experiments (Section 4: Vision).

Supports: MNIST, Fashion-MNIST, EMNIST Letters/Digits with optional CoM normalization.

Usage:
    python src/train.py --config configs/mnist_dense_full.yaml --seed 42
    python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2
    python src/train.py --config configs/emnist_letters_regularized.yaml --seed 42
"""

import sys
from pathlib import Path
import argparse
import torch
import kornia

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config

from src.utils import (
    get_device,
    load_config,
    set_seed,
    track_emissions,
    init_wandb,
    finish_wandb,
)
from src.training import (
    apply_variance_corrected_init,
    create_noise_transform,
    log_training_history,
    log_spectral_metrics,
    save_checkpoint,
)
from src.vision.spectral import effective_rank


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
    set_seed(seed)
    
    # Get CoM normalization setting (default False for backward compatibility)
    apply_com = config.get('data', {}).get('apply_com', False)
    
    # Load dataset using unified data module
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    
    if dataset_name == 'mnist':
        print("Loading MNIST data...")
        from src.data import MNIST
        train_data = MNIST(train=True, device=device, apply_com=apply_com)
        test_data = MNIST(train=False, device=device, apply_com=apply_com)
        d_output = 10
    elif dataset_name == 'fashion_mnist':
        print("Loading Fashion-MNIST data...")
        from src.data import FashionMNIST
        train_data = FashionMNIST(train=True, device=device, apply_com=apply_com)
        test_data = FashionMNIST(train=False, device=device, apply_com=apply_com)
        d_output = 10
    elif dataset_name == 'emnist_letters':
        print("Loading EMNIST Letters data...")
        from src.data import EMNISTLetters
        train_data = EMNISTLetters(train=True, device=device, apply_com=apply_com)
        test_data = EMNISTLetters(train=False, device=device, apply_com=apply_com)
        d_output = 26
    elif dataset_name == 'emnist_digits':
        print("Loading EMNIST Digits data...")
        from src.data import EMNISTDigits
        train_data = EMNISTDigits(train=True, device=device, apply_com=apply_com)
        test_data = EMNISTDigits(train=False, device=device, apply_com=apply_com)
        d_output = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if apply_com:
        print(f"  Center-of-Mass normalization applied")

    # Create model
    model_config = Config(
        epochs=epochs,
        d_hidden=config['model']['d_hidden'],
        d_output=d_output,
        wd=config['regularization']['weight_decay'],
        lr=config['training'].get('lr', 1e-3),
        seed=seed,
    )
    model = Model(model_config).to(device)

    # Apply variance-corrected initialization for Rich Training regime
    variance_corrected = config.get('model', {}).get('variance_corrected_init', False)
    if variance_corrected:
        print("Applying variance-corrected initialization (Rich Training regime):")
        apply_variance_corrected_init(model, enabled=True)

    # Create transform (noise augmentation)
    noise_std = config['regularization']['noise_std']
    transform = create_noise_transform(noise_std)

    # Train
    print(f"Training for {epochs} epochs...")
    history = model.fit(train_data, test_data, transform=transform)

    # Compute eigendecomposition (original code is MPS-safe)
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = model.decompose()

    return model, history, eigenvalues, eigenvectors


def main():
    parser = argparse.ArgumentParser(description="Train Bilinear MLP")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="results/vision/checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--apply-com", type=str, choices=["true", "false"], default=None,
                        help="Override Center-of-Mass normalization (true/false)")
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    config = load_config(args.config)
    config_name = Path(args.config).stem
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    
    # Apply CoM override if specified
    if args.apply_com is not None:
        if 'data' not in config:
            config['data'] = {}
        config['data']['apply_com'] = args.apply_com.lower() == 'true'
        print(f"  CoM override: {config['data']['apply_com']}")

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

    # Save checkpoint using consolidated function
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
