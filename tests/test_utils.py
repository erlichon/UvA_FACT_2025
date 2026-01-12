"""
Unit tests for shared utilities module.
"""

import sys
from pathlib import Path
import pytest
import torch
import tempfile
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    get_device,
    load_config,
    set_seed,
    track_emissions,
    get_history_column,
    TrackingResult,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_requested_device(self):
        """Should return the requested device when specified."""
        assert get_device("cpu") == "cpu"
        assert get_device("cuda") == "cuda"
        assert get_device("mps") == "mps"

    def test_auto_detect_returns_valid_device(self):
        """Auto-detect should return a valid device string."""
        device = get_device(None)
        assert device in ["cpu", "cuda", "mps"]

    def test_auto_detect_prefers_gpu(self):
        """Should prefer GPU over CPU when available."""
        device = get_device(None)
        if torch.cuda.is_available():
            assert device == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_yaml_file(self):
        """Should load and parse YAML configuration."""
        config_content = {
            "model": {"d_hidden": 256},
            "training": {"epochs": 100},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_content, f)
            f.flush()

            loaded = load_config(f.name)
            assert loaded == config_content

    def test_missing_file_raises(self):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestSetSeed:
    """Tests for set_seed function."""

    def test_reproducible_random(self):
        """Same seed should produce same random numbers."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(123)
        b = torch.randn(10)

        assert not torch.allclose(a, b)


class TestTrackEmissions:
    """Tests for track_emissions context manager."""

    def test_returns_tracking_result(self):
        """Should return TrackingResult after context exits."""
        with track_emissions("test-project") as tracker:
            # Simulate some work
            _ = sum(range(10000))

        assert tracker.result is not None
        assert isinstance(tracker.result, TrackingResult)

    def test_tracks_wall_time(self):
        """Should track wall time in seconds and hours."""
        import time

        with track_emissions("test-project") as tracker:
            time.sleep(0.1)  # Sleep 100ms

        assert tracker.result.wall_time_seconds >= 0.1
        assert tracker.result.wall_time_hours >= 0.1 / 3600

    def test_emissions_non_negative(self):
        """Emissions should be non-negative."""
        with track_emissions("test-project") as tracker:
            pass

        assert tracker.result.emissions_kg >= 0


class TestGetHistoryColumn:
    """Tests for get_history_column function."""

    def test_finds_first_matching_column(self):
        """Should return value from first matching column."""
        import pandas as pd

        df = pd.DataFrame({
            'train_acc': [0.8, 0.9, 0.95],
            'val_acc': [0.7, 0.85, 0.9],
        })

        result = get_history_column(df, 'train_acc', 'train/acc')
        assert result == 0.95  # Last value

    def test_handles_alternative_naming(self):
        """Should handle both 'train_acc' and 'train/acc' styles."""
        import pandas as pd

        # Style 1: underscore
        df1 = pd.DataFrame({'train_acc': [0.8, 0.9]})
        assert get_history_column(df1, 'train_acc', 'train/acc') == 0.9

        # Style 2: slash
        df2 = pd.DataFrame({'train/acc': [0.8, 0.9]})
        assert get_history_column(df2, 'train_acc', 'train/acc') == 0.9

    def test_raises_for_missing_column(self):
        """Should raise KeyError when no column matches."""
        import pandas as pd

        df = pd.DataFrame({'other_col': [1, 2, 3]})

        with pytest.raises(KeyError):
            get_history_column(df, 'train_acc', 'train/acc')


class TestTrackingResult:
    """Tests for TrackingResult dataclass."""

    def test_dataclass_fields(self):
        """Should have all expected fields."""
        result = TrackingResult(
            wall_time_seconds=100.0,
            wall_time_hours=100.0 / 3600,
            gpu_hours=0.5,
            emissions_kg=0.001,
        )

        assert result.wall_time_seconds == 100.0
        assert result.wall_time_hours == 100.0 / 3600
        assert result.gpu_hours == 0.5
        assert result.emissions_kg == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
