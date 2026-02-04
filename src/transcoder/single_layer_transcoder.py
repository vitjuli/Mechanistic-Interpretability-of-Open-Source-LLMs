"""
Single-layer transcoder (PLT) implementation.

A per-layer transcoder decomposes the output of a single MLP layer into
sparsely active features that often correspond to interpretable concepts.

This implementation is adapted from the circuit-tracer project:
https://github.com/safety-research/circuit-tracer

Key differences from SAEs:
- Transcoders replace MLP computation, not just encode/decode residual stream
- Input: MLP input activations (pre-MLP residual stream)
- Output: MLP output (what the MLP would have produced)
- Features represent "what the MLP is computing" rather than "what is in the residual"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging

from safetensors import safe_open
from safetensors.torch import save_file

from src.transcoder.activation_functions import JumpReLU

logger = logging.getLogger(__name__)


class SingleLayerTranscoder(nn.Module):
    """
    A per-layer transcoder (PLT) that decomposes MLP computation into features.

    Per-layer transcoders decompose the output of a single MLP layer into
    sparsely active features. Unlike cross-layer transcoders (CLTs), each PLT
    operates independently on its assigned layer.

    Architecture:
        MLP_input -> W_enc -> activation_fn -> W_dec -> MLP_output_approx

    The transcoder learns to approximate:
        MLP(x) ≈ W_dec @ activation_fn(W_enc @ x + b_enc) + b_dec

    Attributes:
        d_model: Dimension of the transformer's residual stream
        d_transcoder: Number of learned features (typically >> d_model)
        layer_idx: Which transformer layer this transcoder is for
        W_enc: Encoder weights (d_transcoder, d_model)
        W_dec: Decoder weights (d_transcoder, d_model)
        b_enc: Encoder bias (d_transcoder,)
        b_dec: Decoder bias (d_model,)
        activation_function: Sparsity-inducing nonlinearity (e.g., JumpReLU)
    """

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function: nn.Module,
        layer_idx: int,
        skip_connection: bool = False,
        transcoder_path: Optional[str] = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize a single-layer transcoder.

        Args:
            d_model: Dimension of the transformer's residual stream
            d_transcoder: Number of transcoder features
            activation_function: Activation function (e.g., JumpReLU)
            layer_idx: Layer index this transcoder is associated with
            skip_connection: Whether to include a skip connection
            transcoder_path: Path to safetensors file for lazy loading
            lazy_encoder: If True, load encoder weights on-demand
            lazy_decoder: If True, load decoder weights on-demand
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layer_idx = layer_idx
        self.transcoder_path = transcoder_path
        self.lazy_encoder = lazy_encoder
        self.lazy_decoder = lazy_decoder
        self._device = device
        self._dtype = dtype

        # Encoder weights
        if not lazy_encoder:
            self.W_enc = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("W_enc", None)

        # Decoder weights
        if not lazy_decoder:
            self.W_dec = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("W_dec", None)

        # Biases (always loaded eagerly as they're small)
        self.b_enc = nn.Parameter(torch.zeros(d_transcoder, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        # Optional skip connection
        if skip_connection:
            self.W_skip = nn.Parameter(
                torch.zeros(d_model, d_model, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("W_skip", None)

        self.activation_function = activation_function

        logger.debug(
            f"Initialized transcoder for layer {layer_idx}: "
            f"d_model={d_model}, d_transcoder={d_transcoder}"
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the module's parameters."""
        return self.b_enc.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    def _load_weight(self, weight_name: str) -> torch.Tensor:
        """Load a weight tensor from the safetensors file."""
        if self.transcoder_path is None:
            raise ValueError(f"Cannot load {weight_name}: transcoder_path not set")

        with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
            return f.get_tensor(weight_name).to(self.dtype)

    def __getattr__(self, name: str):
        """Dynamically load weights when accessed if lazy loading is enabled."""
        if name == "W_enc" and self.lazy_encoder:
            return self._load_weight("W_enc")
        elif name == "W_dec" and self.lazy_decoder:
            return self._load_weight("W_dec")
        return super().__getattr__(name)

    def _get_decoder_vectors(
        self,
        feat_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get decoder vectors, optionally for specific features only.

        Args:
            feat_ids: Feature indices to retrieve (None = all)

        Returns:
            Decoder weight matrix or subset
        """
        if feat_ids is not None:
            to_read = feat_ids
        else:
            to_read = slice(None)

        if not self.lazy_decoder:
            W_dec = self.W_dec
            if isinstance(to_read, torch.Tensor):
                return W_dec[to_read].to(self.dtype)
            return W_dec[to_read].to(self.dtype)

        # Lazy loading with slicing
        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
            return f.get_slice("W_dec")[to_read].to(self.dtype)

    def encode(
        self,
        input_acts: torch.Tensor,
        apply_activation_function: bool = True,
    ) -> torch.Tensor:
        """
        Encode MLP inputs to transcoder features.

        Args:
            input_acts: MLP input activations (..., d_model)
            apply_activation_function: Whether to apply JumpReLU/activation

        Returns:
            Feature activations (..., d_transcoder)
        """
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)

        if not apply_activation_function:
            return pre_acts

        return self.activation_function(pre_acts)

    def decode(
        self,
        acts: torch.Tensor,
        input_acts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode transcoder features to MLP output approximation.

        Args:
            acts: Feature activations (..., d_transcoder)
            input_acts: Original MLP inputs (needed if skip_connection=True)

        Returns:
            Reconstructed MLP output (..., d_model)
        """
        W_dec = self.W_dec
        reconstruction = acts @ W_dec + self.b_dec

        if self.W_skip is not None:
            if input_acts is None:
                raise ValueError(
                    "Transcoder has skip connection but no input_acts provided"
                )
            reconstruction = reconstruction + self.compute_skip(input_acts)

        return reconstruction

    def compute_skip(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Compute skip connection output."""
        if self.W_skip is None:
            raise ValueError("Transcoder has no skip connection")
        return input_acts @ self.W_skip.T

    def forward(self, input_acts: torch.Tensor) -> torch.Tensor:
        """
        Full transcoder pass: encode then decode.

        Args:
            input_acts: MLP input activations (..., d_model)

        Returns:
            Approximated MLP output (..., d_model)
        """
        transcoder_acts = self.encode(input_acts)
        return self.decode(transcoder_acts, input_acts)

    def forward_with_features(
        self,
        input_acts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the intermediate features.

        Args:
            input_acts: MLP input activations (..., d_model)

        Returns:
            - reconstruction: Approximated MLP output (..., d_model)
            - features: Feature activations (..., d_transcoder)
        """
        features = self.encode(input_acts)
        reconstruction = self.decode(features, input_acts)
        return reconstruction, features

    def get_active_features(
        self,
        input_acts: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get indices and values of active (non-zero) features.

        Args:
            input_acts: MLP input activations (..., d_model)
            top_k: If specified, return only top-k features by magnitude

        Returns:
            - indices: Feature indices that are active
            - values: Corresponding activation values
        """
        features = self.encode(input_acts)

        if top_k is not None:
            values, indices = torch.topk(features, k=min(top_k, features.shape[-1]), dim=-1)
            return indices, values

        # Return all non-zero features
        active_mask = features > 0
        # For batched input, return per-sample active features
        if features.dim() == 2:
            results = []
            for i in range(features.shape[0]):
                indices = torch.where(active_mask[i])[0]
                values = features[i, indices]
                results.append((indices, values))
            return results
        else:
            indices = torch.where(active_mask)[0]
            values = features[indices]
            return indices, values

    def compute_reconstruction_error(
        self,
        input_acts: torch.Tensor,
        target_acts: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute reconstruction metrics.

        Args:
            input_acts: MLP input activations
            target_acts: True MLP output activations

        Returns:
            Dictionary with MSE, R^2, and cosine similarity
        """
        with torch.no_grad():
            reconstruction = self.forward(input_acts)

            # MSE
            mse = F.mse_loss(reconstruction, target_acts).item()

            # R^2
            ss_res = ((target_acts - reconstruction) ** 2).sum()
            ss_tot = ((target_acts - target_acts.mean(dim=0, keepdim=True)) ** 2).sum()
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                reconstruction.flatten(),
                target_acts.flatten(),
                dim=0,
            ).item()

            return {
                "mse": mse,
                "r2": r2,
                "cosine_similarity": cos_sim,
            }

    def to_safetensors(self, save_path: str):
        """
        Save transcoder to safetensors format.

        Args:
            save_path: Path to save the safetensors file
        """
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_enc": self.b_enc.cpu(),
            "b_dec": self.b_dec.cpu(),
        }

        if isinstance(self.activation_function, JumpReLU):
            state_dict["activation_function.threshold"] = (
                self.activation_function.threshold.cpu()
            )

        if self.W_skip is not None:
            state_dict["W_skip"] = self.W_skip.cpu()

        save_file(state_dict, save_path)
        logger.info(f"Saved transcoder to {save_path}")

    @classmethod
    def from_safetensors(
        cls,
        path: str,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        lazy_encoder: bool = False,
        lazy_decoder: bool = False,
    ) -> "SingleLayerTranscoder":
        """
        Load a transcoder from a safetensors file.

        Args:
            path: Path to the safetensors file
            layer_idx: Layer index for this transcoder
            device: Device to load to
            dtype: Data type for parameters
            lazy_encoder: Whether to lazy-load encoder weights
            lazy_decoder: Whether to lazy-load decoder weights

        Returns:
            Loaded SingleLayerTranscoder instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with safe_open(path, framework="pt", device=str(device)) as f:
            # Get dimensions from encoder shape
            W_enc_shape = f.get_slice("W_enc").get_shape()
            d_transcoder, d_model = W_enc_shape

            # Check for threshold (JumpReLU) vs plain ReLU
            tensor_names = list(f.keys())
            if "activation_function.threshold" in tensor_names:
                threshold = f.get_tensor("activation_function.threshold").to(dtype)
                activation_fn = JumpReLU(threshold)
            else:
                activation_fn = JumpReLU(0.0)  # Default to ReLU-like behavior

            has_skip = "W_skip" in tensor_names

        # Create transcoder
        transcoder = cls(
            d_model=d_model,
            d_transcoder=d_transcoder,
            activation_function=activation_fn,
            layer_idx=layer_idx,
            skip_connection=has_skip,
            transcoder_path=path,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
            device=device,
            dtype=dtype,
        )

        # Load weights if not lazy
        with safe_open(path, framework="pt", device=str(device)) as f:
            if not lazy_encoder:
                transcoder.W_enc.data = f.get_tensor("W_enc").to(dtype)
            if not lazy_decoder:
                transcoder.W_dec.data = f.get_tensor("W_dec").to(dtype)

            transcoder.b_enc.data = f.get_tensor("b_enc").to(dtype)
            transcoder.b_dec.data = f.get_tensor("b_dec").to(dtype)

            if has_skip:
                transcoder.W_skip.data = f.get_tensor("W_skip").to(dtype)

        logger.info(
            f"Loaded transcoder for layer {layer_idx} from {path}: "
            f"d_model={d_model}, d_transcoder={d_transcoder}"
        )

        return transcoder

    def __repr__(self) -> str:
        return (
            f"SingleLayerTranscoder(layer={self.layer_idx}, "
            f"d_model={self.d_model}, d_transcoder={self.d_transcoder}, "
            f"activation={self.activation_function.__class__.__name__})"
        )
