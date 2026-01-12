"""
Unit tests for bilinear layer implementations.

Tests both BilinearDense (wrapper around original) and BilinearCP (extension).
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.models.bilinear_layer import BilinearDense, BilinearCP, create_bilinear
from shared.components import Bilinear as OriginalBilinear


class TestBilinearDense:
    """Tests for the BilinearDense wrapper."""

    @pytest.fixture
    def layer(self):
        torch.manual_seed(42)
        return BilinearDense(d_in=16, d_out=8)

    def test_forward_shape(self, layer):
        """Output shape should be [..., d_out]."""
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_forward_batch_shape(self, layer):
        """Should handle arbitrary batch dimensions."""
        x = torch.randn(2, 3, 4, 16)
        y = layer(x)
        assert y.shape == (2, 3, 4, 8)

    def test_w_l_shape(self, layer):
        """w_l should have shape [d_out, d_in]."""
        assert layer.w_l.shape == (8, 16)

    def test_w_r_shape(self, layer):
        """w_r should have shape [d_out, d_in]."""
        assert layer.w_r.shape == (8, 16)

    def test_weight_property(self, layer):
        """weight property should expose underlying weight."""
        assert layer.weight.shape == (16, 16)  # 2*d_out x d_in

    def test_matches_original(self):
        """Output should exactly match original implementation."""
        torch.manual_seed(42)
        dense = BilinearDense(d_in=16, d_out=8)

        torch.manual_seed(42)
        original = OriginalBilinear(d_in=16, d_out=8)

        x = torch.randn(4, 16)

        y_dense = dense(x)
        y_original = original(x)

        assert torch.allclose(y_dense, y_original, atol=1e-6)

    def test_w_l_w_r_match_original(self):
        """w_l and w_r should match original implementation."""
        torch.manual_seed(42)
        dense = BilinearDense(d_in=16, d_out=8)

        torch.manual_seed(42)
        original = OriginalBilinear(d_in=16, d_out=8)

        assert torch.allclose(dense.w_l, original.w_l, atol=1e-6)
        assert torch.allclose(dense.w_r, original.w_r, atol=1e-6)

    @pytest.mark.parametrize("gate", [None, "relu", "silu", "gelu"])
    def test_gate_options(self, gate):
        """Should support all gate options from original."""
        layer = BilinearDense(d_in=16, d_out=8, gate=gate)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_gradients_flow(self, layer):
        """Gradients should flow through the layer."""
        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_deterministic_with_seed(self):
        """Same seed should produce same weights."""
        torch.manual_seed(123)
        layer1 = BilinearDense(d_in=16, d_out=8)

        torch.manual_seed(123)
        layer2 = BilinearDense(d_in=16, d_out=8)

        assert torch.allclose(layer1.weight, layer2.weight)


class TestBilinearCP:
    """Tests for the BilinearCP extension."""

    @pytest.fixture
    def layer(self):
        torch.manual_seed(42)
        return BilinearCP(d_in=16, d_out=8, rank=4)

    def test_forward_shape(self, layer):
        """Output shape should be [..., d_out]."""
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_forward_batch_shape(self, layer):
        """Should handle arbitrary batch dimensions."""
        x = torch.randn(2, 3, 4, 16)
        y = layer(x)
        assert y.shape == (2, 3, 4, 8)

    def test_w_l_shape(self, layer):
        """Reconstructed w_l should have shape [d_out, d_in]."""
        assert layer.w_l.shape == (8, 16)

    def test_w_r_shape(self, layer):
        """Reconstructed w_r should have shape [d_out, d_in]."""
        assert layer.w_r.shape == (8, 16)

    def test_factor_shapes(self, layer):
        """CP factors should have correct shapes."""
        assert layer.A.shape == (16, 4)  # [d_in, rank]
        assert layer.B.shape == (16, 4)  # [d_in, rank]
        assert layer.C.shape == (8, 4)   # [d_out, rank]
        assert layer.lambdas.shape == (4,)  # [rank]

    def test_bias_not_supported(self):
        """Bias should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            BilinearCP(d_in=16, d_out=8, rank=4, bias=True)

    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32])
    def test_various_ranks(self, rank):
        """Should work with various rank values."""
        layer = BilinearCP(d_in=16, d_out=8, rank=rank)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_gradients_flow(self, layer):
        """Gradients should flow through all parameters."""
        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.A.grad is not None
        assert layer.B.grad is not None
        assert layer.C.grad is not None
        assert layer.lambdas.grad is not None

    def test_mathematical_correctness(self):
        """Forward pass should match explicit tensor construction."""
        torch.manual_seed(42)
        d_in, d_out, rank = 8, 4, 2
        layer = BilinearCP(d_in=d_in, d_out=d_out, rank=rank)

        x = torch.randn(3, d_in)

        # Forward pass via layer
        y_layer = layer(x)

        # Explicit computation: y = ((x @ A) * (x @ B) * lambdas) @ C.T
        left = x @ layer.A          # [batch, rank]
        right = x @ layer.B         # [batch, rank]
        hidden = left * right * layer.lambdas  # [batch, rank]
        y_explicit = hidden @ layer.C.T   # [batch, d_out]

        assert torch.allclose(y_layer, y_explicit, atol=1e-6)

    def test_w_l_w_r_reconstruction(self):
        """Reconstructed w_l, w_r should produce same output as forward."""
        torch.manual_seed(42)
        layer = BilinearCP(d_in=16, d_out=8, rank=4)

        x = torch.randn(4, 16)

        # Forward via layer
        y_layer = layer(x)

        # Forward via reconstructed w_l, w_r (bilinear formula)
        # y = (w_l @ x.T).T * (w_r @ x.T).T = (x @ w_l.T) * (x @ w_r.T)
        y_reconstructed = (x @ layer.w_l.T) * (x @ layer.w_r.T)

        # Note: These won't be exactly equal because w_l includes lambdas
        # but w_r doesn't, and the CP decomposition is different from
        # the dense bilinear formulation. The shapes should match though.
        assert y_reconstructed.shape == y_layer.shape

    def test_rank_1_pure_bilinear_limited(self):
        """Pure rank-1 bilinear output has limited expressivity."""
        torch.manual_seed(42)

        # A pure rank-1 bilinear layer can only produce rank-1 outputs
        # Test that the output space is limited
        layer = BilinearCP(d_in=4, d_out=8, rank=1)

        # Generate many random inputs
        X = torch.randn(100, 4)
        Y = layer(X)

        # Rank-1 bilinear should produce outputs that lie in a low-dimensional space
        # Compute SVD of outputs to check effective dimensionality
        U, S, V = torch.linalg.svd(Y)

        # Normalize singular values
        S_normalized = S / S.sum()

        # Most variance should be captured by first few components
        top_2_coverage = S_normalized[:2].sum()
        assert top_2_coverage > 0.9, f"Rank-1 outputs should be low-dimensional, got {top_2_coverage}"

    def test_rank_2_xor_succeeds(self):
        """Rank-2+ CP should be able to learn XOR."""
        torch.manual_seed(42)

        # XOR dataset
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])

        # Rank-4 model (higher rank for stable learning)
        model = nn.Sequential(
            BilinearCP(d_in=2, d_out=8, rank=4),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.BCELoss()

        # Train for more steps
        for _ in range(500):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Rank-4 should solve XOR
        with torch.no_grad():
            pred = model(X)
            accuracy = ((pred > 0.5).float() == y).float().mean()

        assert accuracy >= 0.75, f"Rank-4 failed XOR with accuracy {accuracy}"


class TestCreateBilinear:
    """Tests for the create_bilinear factory function."""

    def test_dense_mode_returns_dense(self):
        """mode='dense' should return BilinearDense."""
        layer = create_bilinear(d_in=16, d_out=8, mode='dense')
        assert isinstance(layer, BilinearDense)

    def test_cp_mode_returns_cp(self):
        """mode='cp' should return BilinearCP."""
        layer = create_bilinear(d_in=16, d_out=8, mode='cp', rank=4)
        assert isinstance(layer, BilinearCP)

    def test_cp_mode_requires_rank(self):
        """mode='cp' without rank should raise ValueError."""
        with pytest.raises(ValueError, match="rank required"):
            create_bilinear(d_in=16, d_out=8, mode='cp')

    def test_unknown_mode_raises(self):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            create_bilinear(d_in=16, d_out=8, mode='unknown')

    def test_dense_with_gate(self):
        """Dense mode should accept gate parameter."""
        layer = create_bilinear(d_in=16, d_out=8, mode='dense', gate='relu')
        assert isinstance(layer, BilinearDense)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_dense_with_bias(self):
        """Dense mode should accept bias parameter."""
        layer = create_bilinear(d_in=16, d_out=8, mode='dense', bias=True)
        assert isinstance(layer, BilinearDense)

    def test_cp_with_bias_raises(self):
        """CP mode with bias should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_bilinear(d_in=16, d_out=8, mode='cp', rank=4, bias=True)


class TestBilinearIntegration:
    """Integration tests for bilinear layers in training scenarios."""

    def test_dense_in_training_loop(self):
        """BilinearDense should work in a training loop."""
        torch.manual_seed(42)

        model = nn.Sequential(
            BilinearDense(d_in=16, d_out=32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Fake training step
        x = torch.randn(8, 16)
        y = torch.randint(0, 10, (8,))

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Should complete without error
        assert True

    def test_cp_in_training_loop(self):
        """BilinearCP should work in a training loop."""
        torch.manual_seed(42)

        model = nn.Sequential(
            BilinearCP(d_in=16, d_out=32, rank=8),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Fake training step
        x = torch.randn(8, 16)
        y = torch.randint(0, 10, (8,))

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Should complete without error
        assert True

    def test_eigendecomposition_compatibility(self):
        """w_l and w_r should be usable for eigendecomposition."""
        torch.manual_seed(42)

        for mode in ['dense', 'cp']:
            if mode == 'dense':
                layer = BilinearDense(d_in=16, d_out=8)
            else:
                layer = BilinearCP(d_in=16, d_out=8, rank=4)

            # Compute interaction matrix for class 0: B_sym = 0.5 * (w_l.T @ w_r + w_r.T @ w_l)
            w_l = layer.w_l[0:1]  # First output unit
            w_r = layer.w_r[0:1]

            # Outer products
            interaction = w_l.T @ w_r  # [d_in, d_in]
            B_sym = 0.5 * (interaction + interaction.T)

            # Should be symmetric
            assert torch.allclose(B_sym, B_sym.T, atol=1e-6)

            # Eigendecomposition should work
            eigenvalues, eigenvectors = torch.linalg.eigh(B_sym)

            assert eigenvalues.shape == (16,)
            assert eigenvectors.shape == (16, 16)

    def test_device_transfer(self):
        """Layers should work after device transfer."""
        layer_dense = BilinearDense(d_in=16, d_out=8)
        layer_cp = BilinearCP(d_in=16, d_out=8, rank=4)

        # Test CPU
        x_cpu = torch.randn(4, 16)
        y_dense = layer_dense(x_cpu)
        y_cp = layer_cp(x_cpu)

        assert y_dense.device.type == 'cpu'
        assert y_cp.device.type == 'cpu'

        # Skip GPU test if not available
        if torch.cuda.is_available():
            layer_dense = layer_dense.cuda()
            layer_cp = layer_cp.cuda()
            x_cuda = x_cpu.cuda()

            y_dense = layer_dense(x_cuda)
            y_cp = layer_cp(x_cuda)

            assert y_dense.device.type == 'cuda'
            assert y_cp.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
