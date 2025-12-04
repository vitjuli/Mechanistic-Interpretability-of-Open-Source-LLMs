"""Tests for the sparse autoencoder module."""

import pytest
import torch

from src.mi_pipeline.sparse_autoencoder import SparseAutoencoder, SAETrainer
from src.mi_pipeline.config import Config


class TestSparseAutoencoder:
    """Tests for the SparseAutoencoder class."""

    def test_initialization(self):
        """Test SAE initialization."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        assert sae.input_dim == 512
        assert sae.hidden_dim == 1024
        assert sae.tied_weights is True

    def test_encode(self):
        """Test encoding produces correct shape."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        x = torch.randn(32, 512)
        features = sae.encode(x)
        assert features.shape == (32, 1024)
        assert (features >= 0).all()  # ReLU should make all non-negative

    def test_decode(self):
        """Test decoding produces correct shape."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        features = torch.randn(32, 1024)
        reconstructed = sae.decode(features)
        assert reconstructed.shape == (32, 512)

    def test_forward(self):
        """Test forward pass returns all expected outputs."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        x = torch.randn(32, 512)
        reconstructed, features, original = sae(x)
        assert reconstructed.shape == (32, 512)
        assert features.shape == (32, 1024)
        assert torch.equal(original, x)

    def test_untied_weights(self):
        """Test SAE with untied weights."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024, tied_weights=False)
        assert sae.tied_weights is False
        assert hasattr(sae, "decoder")

        x = torch.randn(32, 512)
        reconstructed, features, _ = sae(x)
        assert reconstructed.shape == (32, 512)


class TestSAETrainer:
    """Tests for the SAETrainer class."""

    def test_initialization(self):
        """Test trainer initialization."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        config = Config(device="cpu", sae_epochs=2)
        trainer = SAETrainer(sae, config, device="cpu")
        assert trainer.config == config
        assert trainer.device == "cpu"

    def test_compute_loss(self):
        """Test loss computation."""
        sae = SparseAutoencoder(input_dim=512, hidden_dim=1024)
        config = Config(device="cpu")
        trainer = SAETrainer(sae, config, device="cpu")

        reconstructed = torch.randn(32, 512)
        features = torch.randn(32, 1024)
        original = torch.randn(32, 512)

        loss, loss_dict = trainer.compute_loss(reconstructed, features, original)
        assert "reconstruction" in loss_dict
        assert "sparsity" in loss_dict
        assert "total" in loss_dict
        assert loss.item() > 0

    def test_train(self):
        """Test training loop."""
        sae = SparseAutoencoder(input_dim=64, hidden_dim=128)
        config = Config(device="cpu", sae_epochs=2, sae_batch_size=16)
        trainer = SAETrainer(sae, config, device="cpu")

        activations = torch.randn(100, 64)
        history = trainer.train(activations, show_progress=False)

        assert len(history) == 2  # 2 epochs
        assert all("total" in epoch for epoch in history)

    def test_get_feature_activations(self):
        """Test getting feature activations."""
        sae = SparseAutoencoder(input_dim=64, hidden_dim=128)
        config = Config(device="cpu")
        trainer = SAETrainer(sae, config, device="cpu")

        activations = torch.randn(50, 64)
        features = trainer.get_feature_activations(activations)

        assert features.shape == (50, 128)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading SAE."""
        sae = SparseAutoencoder(input_dim=64, hidden_dim=128)
        config = Config(device="cpu")
        trainer = SAETrainer(sae, config, device="cpu")

        save_path = str(tmp_path / "test_sae.pt")
        trainer.save(save_path)

        loaded_trainer = SAETrainer.load(save_path, config, device="cpu")
        assert loaded_trainer.sae.input_dim == 64
        assert loaded_trainer.sae.hidden_dim == 128
