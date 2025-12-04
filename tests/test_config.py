"""Tests for the configuration module."""

import pytest

from src.mi_pipeline.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.model_name == "Qwen/Qwen3-4B"
        assert config.target_layers == (8, 16, 24, 31)
        assert config.sae_hidden_dim == 4096
        assert config.sae_sparsity_penalty == 1e-3

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = Config(
            model_name="test/model",
            target_layers=(1, 2, 3),
            sae_hidden_dim=2048,
        )
        assert config.model_name == "test/model"
        assert config.target_layers == (1, 2, 3)
        assert config.sae_hidden_dim == 2048

    def test_invalid_target_layers(self):
        """Test that empty target_layers raises error."""
        with pytest.raises(ValueError, match="target_layers cannot be empty"):
            Config(target_layers=())

    def test_invalid_sparsity_penalty(self):
        """Test that negative sparsity penalty raises error."""
        with pytest.raises(ValueError, match="sae_sparsity_penalty must be non-negative"):
            Config(sae_sparsity_penalty=-0.1)

    def test_invalid_prune_threshold(self):
        """Test that invalid prune threshold raises error."""
        with pytest.raises(ValueError, match="graph_prune_threshold must be between 0 and 1"):
            Config(graph_prune_threshold=1.5)

    def test_behaviors_tuple(self):
        """Test that behaviors is a tuple with expected values."""
        config = Config()
        assert isinstance(config.behaviors, tuple)
        assert "factual_recall" in config.behaviors
        assert "reasoning" in config.behaviors
