"""Sparse Autoencoder module for learning interpretable features.

This module implements sparse autoencoders (SAEs) to decompose
model activations into interpretable, sparse feature representations.
"""

import os
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import Config


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features from activations.

    Implements a single-layer autoencoder with L1 sparsity regularization
    to learn sparse, interpretable feature dictionaries from model activations.

    Attributes:
        input_dim: Dimension of input activations
        hidden_dim: Dimension of sparse feature representation
        encoder: Linear layer mapping inputs to sparse features
        decoder: Linear layer reconstructing inputs from features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tied_weights: bool = True,
    ):
        """Initialize the sparse autoencoder.

        Args:
            input_dim: Dimension of input activations.
            hidden_dim: Dimension of the sparse hidden representation.
            tied_weights: Whether to tie encoder and decoder weights.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights

        # Encoder: input -> sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: sparse features -> reconstructed input
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse feature representation.

        Args:
            x: Input activations of shape (batch, input_dim).

        Returns:
            Sparse features of shape (batch, hidden_dim).
        """
        # Apply encoder and ReLU for non-negativity (common in SAEs)
        return torch.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space.

        Args:
            features: Sparse features of shape (batch, hidden_dim).

        Returns:
            Reconstructed activations of shape (batch, input_dim).
        """
        if self.tied_weights:
            # Use transposed encoder weights for decoding
            return nn.functional.linear(
                features, self.encoder.weight.t(), self.decoder_bias
            )
        else:
            return self.decoder(features)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input activations of shape (batch, input_dim).

        Returns:
            Tuple of (reconstructed, features, input) tensors.
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features, x


class SAETrainer:
    """Trainer for Sparse Autoencoders.

    Handles training loop, loss computation, and model saving/loading.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        config: Optional[Config] = None,
        device: str = "cuda",
    ):
        """Initialize the SAE trainer.

        Args:
            sae: The sparse autoencoder to train.
            config: Configuration object for training parameters.
            device: Device to train on.
        """
        self.sae = sae.to(device)
        self.config = config or Config()
        self.device = device

        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=self.config.sae_learning_rate,
        )

    def compute_loss(
        self,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        original: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the SAE training loss.

        Loss = reconstruction_loss + sparsity_penalty * L1(features)

        Args:
            reconstructed: Reconstructed activations.
            features: Sparse feature activations.
            original: Original input activations.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # MSE reconstruction loss
        reconstruction_loss = nn.functional.mse_loss(reconstructed, original)

        # L1 sparsity loss on features
        sparsity_loss = features.abs().mean()

        # Total loss
        total_loss = (
            reconstruction_loss
            + self.config.sae_sparsity_penalty * sparsity_loss
        )

        loss_dict = {
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def train(
        self,
        activations: torch.Tensor,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """Train the SAE on activation data.

        Args:
            activations: Activation tensor of shape (num_samples, input_dim).
            show_progress: Whether to show training progress.

        Returns:
            List of loss dictionaries for each epoch.
        """
        # Create data loader
        dataset = TensorDataset(activations)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.sae_batch_size,
            shuffle=True,
        )

        self.sae.train()
        history = []

        epochs_iter = range(self.config.sae_epochs)
        if show_progress:
            epochs_iter = tqdm(epochs_iter, desc="Training SAE")

        for epoch in epochs_iter:
            epoch_losses: dict[str, list[float]] = {
                "reconstruction": [],
                "sparsity": [],
                "total": [],
            }

            for (batch,) in dataloader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                reconstructed, features, original = self.sae(batch)
                loss, loss_dict = self.compute_loss(
                    reconstructed, features, original
                )

                loss.backward()
                self.optimizer.step()

                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)

            # Average losses for epoch
            avg_losses = {
                key: sum(vals) / len(vals)
                for key, vals in epoch_losses.items()
            }
            history.append(avg_losses)

            if show_progress:
                tqdm.write(
                    f"Epoch {epoch + 1}: "
                    f"recon={avg_losses['reconstruction']:.6f}, "
                    f"sparse={avg_losses['sparsity']:.6f}"
                )

        self.sae.eval()
        return history

    def get_feature_activations(
        self, activations: torch.Tensor
    ) -> torch.Tensor:
        """Get sparse feature activations for input activations.

        Args:
            activations: Input activations of shape (num_samples, input_dim).

        Returns:
            Feature activations of shape (num_samples, hidden_dim).
        """
        self.sae.eval()
        with torch.no_grad():
            activations = activations.to(self.device)
            features = self.sae.encode(activations)
        return features.cpu()

    def save(self, path: str) -> None:
        """Save the SAE model to disk.

        Args:
            path: Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.sae.state_dict(),
                "input_dim": self.sae.input_dim,
                "hidden_dim": self.sae.hidden_dim,
                "tied_weights": self.sae.tied_weights,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        config: Optional[Config] = None,
        device: str = "cuda",
    ) -> "SAETrainer":
        """Load an SAE model from disk.

        Args:
            path: Path to the saved model.
            config: Configuration object.
            device: Device to load the model to.

        Returns:
            SAETrainer instance with loaded model.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sae = SparseAutoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            tied_weights=checkpoint["tied_weights"],
        )
        sae.load_state_dict(checkpoint["state_dict"])
        return cls(sae, config, device)
