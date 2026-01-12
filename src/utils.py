"""
Shared utilities for FACT-AI experiments.

Consolidates common patterns from training and analysis scripts:
- Device detection
- Configuration loading
- Random seed management
- Experiment tracking (wandb + codecarbon)
- MPS compatibility helpers
"""

from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
import time
import logging
import warnings
import os
import yaml
import torch
import wandb

# Suppress codecarbon warnings about RAPL permissions (common on HPC systems)
logging.getLogger("codecarbon").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*RAPL.*")
warnings.filterwarnings("ignore", message=".*codecarbon.*")

# Import codecarbon with fallback
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def get_device(requested: Optional[str] = None) -> str:
    """
    Auto-detect best available device.

    Args:
        requested: Specific device to use. If None, auto-detect.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if requested is not None:
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_mps_device(device: str) -> bool:
    """Check if device is MPS (Apple Metal)."""
    return device == "mps" or (isinstance(device, torch.device) and device.type == "mps")


def safe_eigh(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigendecomposition with MPS compatibility.

    torch.linalg.eigh has limited MPS support, so we move to CPU if needed.

    Args:
        tensor: Symmetric matrix to decompose

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    original_device = tensor.device

    # MPS doesn't fully support eigh - move to CPU
    if original_device.type == "mps":
        tensor_cpu = tensor.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(tensor_cpu)
        return eigenvalues.to(original_device), eigenvectors.to(original_device)

    return torch.linalg.eigh(tensor)


def setup_mps_fallbacks():
    """
    Configure PyTorch for better MPS compatibility.

    Call this at the start of training scripts when using MPS.
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    # Enable MPS fallback for unsupported ops (runs on CPU automatically)
    import os
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Reduce memory pressure

    # Some operations need explicit fallback
    # This is handled automatically by PyTorch 2.0+ but we set it explicitly
    if hasattr(torch.backends.mps, "enable_fallback"):
        torch.backends.mps.enable_fallback()

    # Monkey-patch torch.cuda.empty_cache to be a no-op when CUDA isn't available
    # This fixes compatibility with original paper code that calls cuda.empty_cache()
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda: None


def load_config(path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrackingResult:
    """Results from experiment tracking."""
    wall_time_seconds: float
    wall_time_hours: float
    gpu_hours: float
    emissions_kg: float


@contextmanager
def track_emissions(project_name: str = "fact-bilinear"):
    """
    Context manager for tracking CO2 emissions and wall time.

    Gracefully handles systems where codecarbon can't read power metrics
    (e.g., HPC systems without RAPL permissions).

    Usage:
        with track_emissions("my-project") as tracker:
            # ... run experiment ...
        print(f"CO2: {tracker.result.emissions_kg} kg")

    Yields:
        Object with .result attribute (populated after context exits)
    """
    class Tracker:
        def __init__(self):
            self.result: Optional[TrackingResult] = None

    tracker = Tracker()
    emissions_tracker = None
    emissions_kg = 0.0

    # Try to initialize codecarbon, but don't fail if it doesn't work
    if CODECARBON_AVAILABLE:
        try:
            emissions_tracker = EmissionsTracker(
                project_name=project_name,
                log_level="error",  # Only show errors, not warnings
                save_to_file=False,  # Don't create emissions.csv
                tracking_mode="machine",  # More compatible with HPC
            )
            emissions_tracker.start()
        except Exception as e:
            # Codecarbon failed to start (e.g., permission issues)
            # Continue without emissions tracking
            emissions_tracker = None
            print(f"Note: CO2 tracking disabled ({type(e).__name__})")

    start_time = time.time()

    try:
        yield tracker
    finally:
        end_time = time.time()

        # Try to stop emissions tracker if it was started
        if emissions_tracker is not None:
            try:
                emissions_kg = emissions_tracker.stop() or 0.0
            except Exception:
                emissions_kg = 0.0

        wall_time_seconds = end_time - start_time
        wall_time_hours = wall_time_seconds / 3600
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_hours = wall_time_hours * max(1, gpu_count) if torch.cuda.is_available() else 0

        tracker.result = TrackingResult(
            wall_time_seconds=wall_time_seconds,
            wall_time_hours=wall_time_hours,
            gpu_hours=gpu_hours,
            emissions_kg=emissions_kg,
        )


# Default wandb settings for the FACT-AI project team
WANDB_ENTITY = "itayerlich96-student"
WANDB_PROJECT = "fact-bilinear"


def init_wandb(
    name: str,
    config: dict,
    device: str,
    enabled: bool = True,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    tags: list[str] | None = None,
) -> bool:
    """
    Initialize wandb run with standard configuration.

    Args:
        name: Run name
        config: Experiment configuration dict
        device: Device being used
        enabled: If False, skip initialization
        entity: wandb entity (team/user). Defaults to WANDB_ENTITY.
        project: wandb project name. Defaults to WANDB_PROJECT.
        tags: Optional list of tags (e.g., ["vision", "mnist"] or ["language", "sae"])

    Returns:
        True if wandb was initialized, False otherwise
    """
    if not enabled:
        return False

    wandb.init(
        entity=entity,
        project=project,
        name=name,
        tags=tags,
        config={
            **config,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else device,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    )
    return True


def finish_wandb(tracking_result: TrackingResult, extra_summary: Optional[dict] = None) -> None:
    """
    Finalize wandb run with tracking metrics.

    Args:
        tracking_result: Results from track_emissions context manager
        extra_summary: Additional metrics to add to wandb summary
    """
    if wandb.run is None:
        return

    wandb.summary["wall_time_hours"] = tracking_result.wall_time_hours
    wandb.summary["gpu_hours"] = tracking_result.gpu_hours
    wandb.summary["co2_kg"] = tracking_result.emissions_kg

    if extra_summary:
        for key, value in extra_summary.items():
            wandb.summary[key] = value

    wandb.finish()


def get_history_column(df, *names):
    """
    Get column from training history DataFrame, handling naming variations.

    The original paper code uses 'train/acc' style, our code uses 'train_acc'.

    Args:
        df: Training history DataFrame
        *names: Column name variants to try

    Returns:
        Last value from the first matching column

    Raises:
        KeyError: If none of the column names are found
    """
    for name in names:
        if name in df.columns:
            return df[name].iloc[-1]
    raise KeyError(f"None of {names} found in history columns: {list(df.columns)}")
