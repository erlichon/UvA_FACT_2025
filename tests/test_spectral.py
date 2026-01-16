"""
Unit tests for spectral analysis utilities.

Tests effective rank, top-k coverage, and other spectral metrics.
"""

import sys
from pathlib import Path
import pytest
import torch
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.spectral import (
    effective_rank,
    top_k_coverage,
    eigenvalue_decay_rate,
    spectral_summary,
    load_checkpoint_eigenvalues,
)


class TestEffectiveRank:
    """Tests for effective_rank function."""

    def test_uniform_eigenvalues(self):
        """Uniform eigenvalues should give maximum effective rank."""
        # All equal eigenvalues -> maximum entropy -> effective rank = n
        eigenvalues = torch.ones(10)
        eff_rank = effective_rank(eigenvalues)
        assert torch.isclose(eff_rank, torch.tensor(10.0), atol=0.01)

    def test_single_nonzero(self):
        """Single nonzero eigenvalue should give effective rank ~1."""
        eigenvalues = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        eff_rank = effective_rank(eigenvalues)
        assert torch.isclose(eff_rank, torch.tensor(1.0), atol=0.01)

    def test_two_equal_nonzero(self):
        """Two equal nonzero eigenvalues should give effective rank ~2."""
        eigenvalues = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        eff_rank = effective_rank(eigenvalues)
        assert torch.isclose(eff_rank, torch.tensor(2.0), atol=0.01)

    def test_exponential_decay(self):
        """Exponentially decaying eigenvalues should have low effective rank."""
        # Sharp decay -> low entropy -> low effective rank
        eigenvalues = torch.tensor([1.0, 0.1, 0.01, 0.001, 0.0001])
        eff_rank = effective_rank(eigenvalues)
        # Should be much less than 5
        assert eff_rank < 3.0

    def test_handles_negative_eigenvalues(self):
        """Should use absolute values for negative eigenvalues."""
        eigenvalues = torch.tensor([1.0, -1.0, 0.5, -0.5])
        eff_rank = effective_rank(eigenvalues)
        # Same as all positive version
        eigenvalues_pos = torch.tensor([1.0, 1.0, 0.5, 0.5])
        eff_rank_pos = effective_rank(eigenvalues_pos)
        assert torch.isclose(eff_rank, eff_rank_pos, atol=0.01)

    def test_batch_dimension(self):
        """Should handle batch dimensions."""
        # Shape: [3, 10] - 3 classes, 10 eigenvalues each
        eigenvalues = torch.rand(3, 10)
        eff_rank = effective_rank(eigenvalues)
        assert eff_rank.shape == (3,)

    def test_multiple_batch_dimensions(self):
        """Should handle multiple batch dimensions."""
        eigenvalues = torch.rand(2, 3, 10)
        eff_rank = effective_rank(eigenvalues)
        assert eff_rank.shape == (2, 3)

    def test_range_bounds(self):
        """Effective rank should be in [1, n]."""
        for _ in range(10):
            eigenvalues = torch.rand(20).abs() + 1e-6  # Ensure positive
            eff_rank = effective_rank(eigenvalues)
            assert 1.0 <= eff_rank <= 20.0


class TestTopKCoverage:
    """Tests for top_k_coverage function."""

    def test_full_coverage(self):
        """Top-n coverage should be 1.0."""
        eigenvalues = torch.tensor([5.0, 3.0, 2.0, 1.0])
        coverage = top_k_coverage(eigenvalues, k=4)
        assert torch.isclose(coverage, torch.tensor(1.0), atol=0.01)

    def test_single_dominant(self):
        """Single dominant eigenvalue should have high top-1 coverage."""
        eigenvalues = torch.tensor([10.0, 0.1, 0.1, 0.1])
        coverage = top_k_coverage(eigenvalues, k=1)
        assert coverage > 0.95  # 10 / 10.3 ≈ 0.97

    def test_uniform_coverage(self):
        """Uniform eigenvalues: top-k coverage = k/n."""
        eigenvalues = torch.ones(10)
        coverage_5 = top_k_coverage(eigenvalues, k=5)
        assert torch.isclose(coverage_5, torch.tensor(0.5), atol=0.01)

    def test_handles_negative_eigenvalues(self):
        """Should use absolute values for negative eigenvalues."""
        eigenvalues = torch.tensor([3.0, -2.0, 1.0, -0.5])
        coverage = top_k_coverage(eigenvalues, k=2)
        # Top 2 by absolute value: 3.0, 2.0 -> (3+2)/(3+2+1+0.5) = 5/6.5
        expected = 5.0 / 6.5
        assert torch.isclose(coverage, torch.tensor(expected), atol=0.01)

    def test_batch_dimension(self):
        """Should handle batch dimensions."""
        eigenvalues = torch.rand(3, 10)
        coverage = top_k_coverage(eigenvalues, k=5)
        assert coverage.shape == (3,)

    def test_k_larger_than_n(self):
        """k > n should still work (just return 1.0)."""
        eigenvalues = torch.tensor([1.0, 2.0, 3.0])
        coverage = top_k_coverage(eigenvalues, k=10)
        assert torch.isclose(coverage, torch.tensor(1.0), atol=0.01)


class TestEigenvalueDecayRate:
    """Tests for eigenvalue_decay_rate function."""

    def test_no_decay(self):
        """Equal top eigenvalues should give decay rate 1.0."""
        eigenvalues = torch.tensor([5.0, 5.0, 1.0, 1.0])
        decay = eigenvalue_decay_rate(eigenvalues)
        assert torch.isclose(decay, torch.tensor(1.0), atol=0.01)

    def test_fast_decay(self):
        """Fast decay should give low ratio."""
        eigenvalues = torch.tensor([10.0, 1.0, 0.1, 0.01])
        decay = eigenvalue_decay_rate(eigenvalues)
        assert torch.isclose(decay, torch.tensor(0.1), atol=0.01)

    def test_handles_negative_eigenvalues(self):
        """Should use absolute values."""
        eigenvalues = torch.tensor([-10.0, 5.0, -2.0, 1.0])
        decay = eigenvalue_decay_rate(eigenvalues)
        # Sorted by abs: 10, 5, 2, 1 -> ratio = 5/10 = 0.5
        assert torch.isclose(decay, torch.tensor(0.5), atol=0.01)

    def test_batch_dimension(self):
        """Should handle batch dimensions."""
        eigenvalues = torch.rand(3, 10)
        decay = eigenvalue_decay_rate(eigenvalues)
        assert decay.shape == (3,)


class TestSpectralSummary:
    """Tests for spectral_summary function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        eigenvalues = torch.rand(10, 256)  # 10 classes, 256 hidden
        summary = spectral_summary(eigenvalues)

        expected_keys = [
            'effective_rank_mean', 'effective_rank_std',
            'top5_coverage_mean', 'top5_coverage_std',
            'top10_coverage_mean', 'top10_coverage_std',
            'decay_rate_mean', 'decay_rate_std',
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_values_are_floats(self):
        """All summary values should be floats."""
        eigenvalues = torch.rand(10, 256)
        summary = spectral_summary(eigenvalues)

        for key, value in summary.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"

    def test_mean_std_consistency(self):
        """Mean and std should be consistent with input."""
        # All classes have same eigenvalues -> std should be 0
        eigenvalues = torch.ones(10, 256)
        summary = spectral_summary(eigenvalues)

        assert summary['effective_rank_std'] < 0.01
        assert summary['top5_coverage_std'] < 0.01


class TestLoadCheckpointEigenvalues:
    """Tests for load_checkpoint_eigenvalues function."""

    def test_loads_eigenvalues_and_eigenvectors(self):
        """Should load both eigenvalues and eigenvectors from checkpoint."""
        # Create a fake checkpoint
        eigenvalues = torch.rand(10, 256)
        eigenvectors = torch.rand(10, 256, 784)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'config': {'mode': 'dense'},
            }
            torch.save(checkpoint, f.name)

            # Load it back
            loaded_vals, loaded_vecs = load_checkpoint_eigenvalues(f.name)

            assert torch.allclose(loaded_vals, eigenvalues)
            assert torch.allclose(loaded_vecs, eigenvectors)

    def test_loads_to_cpu(self):
        """Should load tensors to CPU."""
        eigenvalues = torch.rand(10, 256)
        eigenvectors = torch.rand(10, 256, 784)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
            }
            torch.save(checkpoint, f.name)

            loaded_vals, loaded_vecs = load_checkpoint_eigenvalues(f.name)

            assert loaded_vals.device.type == 'cpu'
            assert loaded_vecs.device.type == 'cpu'

    def test_missing_file_raises(self):
        """Should raise error for missing file."""
        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_checkpoint_eigenvalues("/nonexistent/path.pt")


class TestSpectralIntegration:
    """Integration tests combining spectral functions."""

    def test_low_rank_vs_high_rank(self):
        """Low-rank eigenspectrum should have lower effective rank."""
        # Low rank: exponentially decaying eigenvalues (truly sharp spectrum)
        low_rank = torch.exp(-torch.arange(100).float() / 5)  # Fast decay

        # High rank: uniform eigenvalues
        high_rank = torch.ones(100)

        eff_rank_low = effective_rank(low_rank)
        eff_rank_high = effective_rank(high_rank)

        assert eff_rank_low < eff_rank_high
        assert eff_rank_low < 20  # Sharp spectrum has low effective rank
        assert eff_rank_high > 90  # Close to n for uniform

    def test_coverage_increases_with_k(self):
        """Top-k coverage should increase with k."""
        eigenvalues = torch.rand(100).abs()

        coverages = [top_k_coverage(eigenvalues, k=k).item() for k in [1, 5, 10, 50, 100]]

        # Should be monotonically increasing
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1]

    def test_paper_scenario(self):
        """Test scenario matching paper claims."""
        # Paper claims: regularized models have effective rank ~20-40
        # vs non-regularized with effective rank ~150-200

        # Simulate regularized (sharp spectrum)
        regularized = torch.exp(-torch.arange(256).float() / 10)

        # Simulate non-regularized (flatter spectrum)
        non_regularized = torch.exp(-torch.arange(256).float() / 50)

        eff_rank_reg = effective_rank(regularized)
        eff_rank_noreg = effective_rank(non_regularized)

        # Regularized should have lower effective rank
        assert eff_rank_reg < eff_rank_noreg

        # Ratio should be significant (paper claims < 0.5)
        ratio = eff_rank_reg / eff_rank_noreg
        assert ratio < 0.5, f"Ratio {ratio} not < 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
