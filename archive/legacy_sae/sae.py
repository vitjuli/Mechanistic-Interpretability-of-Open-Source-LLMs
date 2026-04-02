"""
Sparse Autoencoder (SAE) implementation for feature decomposition.

Based on the architecture from:
- Anthropic's "Towards Monosemanticity" (Bricken et al., 2023)
- "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for decomposing neural network activations.

    Architecture:
        x -> encoder -> ReLU -> latent -> decoder -> reconstruction

    Loss:
        L = MSE(x, x_hat) + lambda * L1(latent)
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 4,
        l1_lambda: float = 0.005,
    ):
        """
        Initialize SAE.

        Args:
            input_dim: Dimension of input activations (e.g., MLP hidden size)
            expansion_factor: Multiplier for latent dimension (4x to 8x typical)
            l1_lambda: Sparsity penalty coefficient
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.l1_lambda = l1_lambda

        # Encoder: input -> latent
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

        # Decoder: latent -> reconstruction
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)

        # Pre-bias (learned offset to subtract before encoding)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Initialize weights
        self._initialize_weights()

        logger.info(f"SAE initialized: {input_dim} -> {self.latent_dim} ({expansion_factor}x)")

    def _initialize_weights(self):
        """Initialize encoder/decoder weights."""
        # Xavier initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        # Normalize decoder columns (unit norm constraint)
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=1, keepdim=True) + 1e-8
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse latent features.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            Latent features (batch, latent_dim)
        """
        # Center activations
        x_centered = x - self.pre_bias

        # Encode with ReLU for sparsity
        latent = F.relu(self.encoder(x_centered))

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.

        Args:
            latent: Latent features (batch, latent_dim)

        Returns:
            Reconstructed activations (batch, input_dim)
        """
        reconstruction = self.decoder(latent) + self.pre_bias
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder pass.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            - reconstruction: Decoded activations (batch, input_dim)
            - latent: Sparse latent features (batch, latent_dim)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss with sparsity penalty.

        Args:
            x: Original activations
            reconstruction: Decoded activations
            latent: Sparse latent features

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstruction, x)

        # Sparsity loss (L1 on latent activations)
        l1_loss = latent.abs().mean()

        # Total loss
        total_loss = mse_loss + self.l1_lambda * l1_loss

        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
        }

    def get_feature_activations(
        self,
        x: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k active features for input.

        Args:
            x: Input activations (batch, input_dim)
            top_k: Number of top features to return

        Returns:
            - feature_indices: Top-k feature indices (batch, top_k)
            - feature_values: Top-k feature values (batch, top_k)
        """
        latent = self.encode(x)
        values, indices = torch.topk(latent, k=top_k, dim=-1)
        return indices, values

    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction and sparsity metrics.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            Dictionary with metrics
        """
        reconstruction, latent = self.forward(x)

        # Reconstruction error
        mse = F.mse_loss(reconstruction, x).item()

        # R-squared
        ss_res = ((x - reconstruction) ** 2).sum()
        ss_tot = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2 = r2.item()

        # Sparsity metrics
        l0 = (latent > 0).float().mean().item()  # Fraction of active features
        l1 = latent.abs().mean().item()

        # Dead features (never activate)
        feature_max_acts = latent.max(dim=0)[0]
        dead_fraction = (feature_max_acts == 0).float().mean().item()

        return {
            "mse": mse,
            "r2": r2,
            "l0": l0,
            "l1": l1,
            "dead_fraction": dead_fraction,
        }

    def normalize_decoder(self):
        """Normalize decoder weights to unit norm (column normalization)."""
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=1, keepdim=True) + 1e-8
            )


class SAETrainer:
    """Trainer for sparse autoencoders."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        learning_rate: float = 3e-4,
        normalize_decoder: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            sae: SparseAutoencoder model
            learning_rate: Adam learning rate
            normalize_decoder: Whether to normalize decoder after each step
        """
        self.sae = sae
        self.normalize_decoder = normalize_decoder
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

        self.step_count = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Activation batch (batch_size, input_dim)

        Returns:
            Dictionary with loss values
        """
        self.sae.train()
        self.optimizer.zero_grad()

        # Forward pass
        reconstruction, latent = self.sae(batch)

        # Compute loss
        losses = self.sae.loss(batch, reconstruction, latent)

        # Backward pass
        losses["total"].backward()
        self.optimizer.step()

        # Normalize decoder weights
        if self.normalize_decoder:
            self.sae.normalize_decoder()

        self.step_count += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, val_data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate on validation data.

        Args:
            val_data: Validation activations (N, input_dim)

        Returns:
            Dictionary with metrics
        """
        self.sae.eval()
        return self.sae.compute_metrics(val_data)
