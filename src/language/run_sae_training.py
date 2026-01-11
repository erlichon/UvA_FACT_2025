"""
SAE Training wrapper using original paper code.

Uses the SAE implementation from bilinear-decomposition-main/sae/sae.py

Usage:
    python src/language/run_sae_training.py --config configs/language_sae.yaml
    python src/language/run_sae_training.py --config configs/language_sae.yaml --no-wandb
"""

import sys
from pathlib import Path
import argparse
import time
import yaml
import torch
import wandb
from codecarbon import EmissionsTracker

# Add original code to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORIG_PATH = PROJECT_ROOT / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))

from language.transformer import Transformer
from sae.sae import SAE, SAEConfig
from datasets import load_dataset


def create_dataloader(tokenizer, config: dict, device: str):
    """Create TinyStories dataloader for SAE training."""
    print(f"Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    n_samples = config.get("data", {}).get("n_samples", 100000)
    if n_samples and n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
        print(f"Using {n_samples} samples")

    n_ctx = config.get("sae", {}).get("n_ctx", 256)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=n_ctx,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    batch_size = config.get("training", {}).get("batch_size", 32)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Section 5")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint-dir", type=str, default="results/language")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect)")
    parser.add_argument("--wandb-project", type=str, default="fact-bilinear-language")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
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
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_name = config.get("name", Path(args.config).stem)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=config_name,
            config={
                **config,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else device,
            }
        )

    # Start CO2 tracking
    tracker = EmissionsTracker(project_name="fact-bilinear-language", log_level="warning")
    tracker.start()
    start_time = time.time()

    # Load pretrained bilinear transformer
    model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
    print(f"Loading pretrained model: {model_name}")
    model = Transformer.from_pretrained(model_name, device=device)

    # Create dataloader
    train_loader = create_dataloader(model.tokenizer, config, device)

    # Create validation batch (single batch for validation metrics)
    val_iter = iter(train_loader)
    validate = next(val_iter)
    validate = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in validate.items()}

    # SAE configuration
    sae_config = config.get("sae", {})
    layer = sae_config.get("layer", 2)
    point_name = sae_config.get("point", "mlp-out")
    expansion = sae_config.get("expansion", 8)
    k = sae_config.get("k", 32)

    training_config = config.get("training", {})
    n_buffers = training_config.get("n_buffers", 100)
    lr = training_config.get("lr", 1e-4)

    # Create SAE using original code
    print(f"Creating SAE at point=({point_name}, {layer}), expansion={expansion}, k={k}")
    sae_cfg = SAEConfig(
        point=(point_name, layer),
        target=(point_name, layer),
        expansion=expansion,
        k=k,
        d_model=model.config.d_model,
        n_ctx=model.config.n_ctx,
        lr=lr,
        n_buffers=n_buffers,
        in_batch=training_config.get("batch_size", 32),
        out_batch=training_config.get("out_batch", 4096),
        n_batches=training_config.get("n_batches", 256),
    )

    sae = SAE(sae_cfg).to(device)

    # Train SAE using original paper's fit method
    # Note: Pass project=None to avoid double wandb initialization
    # (we already initialized wandb above if --no-wandb was not set)
    print(f"Training SAE for {n_buffers} buffers...")
    sae.fit(model, train_loader, validate, project=None)

    # Stop tracking
    end_time = time.time()
    emissions_kg = tracker.stop()
    wall_time_hours = (end_time - start_time) / 3600

    if not args.no_wandb:
        wandb.summary["wall_time_hours"] = wall_time_hours
        wandb.summary["co2_kg"] = emissions_kg
        wandb.finish()

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save SAE state and config
    point_str = point_name.replace("-", "_")
    checkpoint_path = checkpoint_dir / f"sae_{point_str}_layer{layer}.pt"

    # Convert config to serializable dict
    sae_config_dict = {
        "point": [point_name, layer],
        "target": [point_name, layer],
        "expansion": expansion,
        "k": k,
        "d_model": model.config.d_model,
        "n_ctx": model.config.n_ctx,
        "lr": lr,
        "n_buffers": n_buffers,
    }

    torch.save({
        "config": config,
        "sae_state_dict": sae.state_dict(),
        "sae_config": sae_config_dict,
        "model_name": model_name,
        "layer": layer,
        "point": point_name,
    }, checkpoint_path)

    print(f"\n{'='*60}")
    print(f"SAE Training Complete")
    print(f"{'='*60}")
    print(f"Config: {config_name}")
    print(f"Point: ({point_name}, {layer})")
    print(f"Expansion: {expansion}, k: {k}")
    print(f"Wall Time: {wall_time_hours*60:.1f} minutes")
    print(f"CO2 (kg): {emissions_kg:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
