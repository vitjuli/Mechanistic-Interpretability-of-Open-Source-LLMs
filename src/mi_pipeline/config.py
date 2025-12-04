"""Configuration module for mechanistic interpretability pipeline.

This module defines the configuration settings for running Qwen3-4B
on prompt sets and capturing activations for interpretability analysis.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for the mechanistic interpretability pipeline.

    Attributes:
        model_name: HuggingFace model identifier for Qwen3-4B
        target_layers: Layer indices to capture activations from
        behaviors: List of behavior categories to analyze
        sae_hidden_dim: Hidden dimension size for sparse autoencoder
        sae_sparsity_penalty: L1 sparsity regularization coefficient
        sae_learning_rate: Learning rate for SAE training
        sae_epochs: Number of training epochs for SAE
        graph_prune_threshold: Threshold for pruning dependency graph edges
        device: Device to run inference on
        batch_size: Batch size for processing prompts
        max_length: Maximum sequence length for tokenization
    """

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B"
    target_layers: tuple = (8, 16, 24, 31)  # Subset of layers to analyze

    # Behavior categories to analyze (based on biology paper)
    behaviors: tuple = (
        "factual_recall",
        "reasoning",
        "code_generation",
        "multilingual",
    )

    # Sparse Autoencoder configuration
    sae_hidden_dim: int = 4096
    sae_sparsity_penalty: float = 1e-3
    sae_learning_rate: float = 1e-4
    sae_epochs: int = 10
    sae_batch_size: int = 32

    # Dependency graph configuration
    graph_prune_threshold: float = 0.01
    graph_top_k_features: int = 50

    # Inference configuration
    device: str = "cuda"
    batch_size: int = 8
    max_length: int = 512

    # Output paths
    output_dir: str = "outputs"
    activations_dir: str = "outputs/activations"
    sae_dir: str = "outputs/sae_models"
    graphs_dir: str = "outputs/graphs"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.target_layers:
            raise ValueError("target_layers cannot be empty")
        if self.sae_sparsity_penalty < 0:
            raise ValueError("sae_sparsity_penalty must be non-negative")
        if self.graph_prune_threshold < 0 or self.graph_prune_threshold > 1:
            raise ValueError("graph_prune_threshold must be between 0 and 1")
