"""
Transcoder loading utilities for pre-trained transcoders from HuggingFace.

This module provides utilities to download and load pre-trained transcoders
from the circuit-tracer project's HuggingFace repositories.

Available Qwen3 transcoders:
- mwhanna/qwen3-0.6b-transcoders-lowl0
- mwhanna/qwen3-1.7b-transcoders-lowl0
- mwhanna/qwen3-4b-transcoders
- mwhanna/qwen3-8b-transcoders
- mwhanna/qwen3-14b-transcoders-lowl0

Reference:
- https://github.com/safety-research/circuit-tracer
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from src.transcoder.single_layer_transcoder import SingleLayerTranscoder
from src.transcoder.activation_functions import JumpReLU

logger = logging.getLogger(__name__)


# Known transcoder repositories for Qwen3 models
QWEN3_TRANSCODERS = {
    "0.6b": "mwhanna/qwen3-0.6b-transcoders-lowl0",
    "1.7b": "mwhanna/qwen3-1.7b-transcoders-lowl0",
    "4b": "mwhanna/qwen3-4b-transcoders",
    "8b": "mwhanna/qwen3-8b-transcoders",
    "14b": "mwhanna/qwen3-14b-transcoders-lowl0",
}

# Model configurations (layers, hidden dim)
QWEN3_CONFIGS = {
    "0.6b": {"num_layers": 28, "hidden_size": 1024, "model_name": "Qwen/Qwen3-0.6B"},
    "1.7b": {"num_layers": 28, "hidden_size": 1536, "model_name": "Qwen/Qwen3-1.7B"},
    "4b": {"num_layers": 36, "hidden_size": 2560, "model_name": "Qwen/Qwen3-4B"},
    "8b": {"num_layers": 36, "hidden_size": 4096, "model_name": "Qwen/Qwen3-8B"},
    "14b": {"num_layers": 40, "hidden_size": 5120, "model_name": "Qwen/Qwen3-14B"},
}


@dataclass
class TranscoderConfig:
    """Configuration for a transcoder set."""

    model_name: str
    model_kind: str
    feature_input_hook: str
    feature_output_hook: str
    num_layers: int
    hidden_size: int
    repo_id: str


class TranscoderSet:
    """
    A collection of transcoders for all layers of a model.

    This class manages loading, caching, and accessing transcoders for
    different layers of a transformer model.

    Attributes:
        config: TranscoderConfig with model and transcoder details
        transcoders: Dict mapping layer index to SingleLayerTranscoder
        cache_dir: Local directory where transcoder weights are cached
    """

    def __init__(
        self,
        config: TranscoderConfig,
        cache_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        lazy_load: bool = True,
    ):
        """
        Initialize a TranscoderSet.

        Args:
            config: Configuration for the transcoder set
            cache_dir: Directory to cache downloaded weights
            device: Device to load transcoders to
            dtype: Data type for transcoder parameters
            lazy_load: If True, only load transcoders when accessed
        """
        self.config = config
        self.cache_dir = cache_dir or Path.home() / ".cache" / "transcoders"
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.lazy_load = lazy_load

        self._transcoders: Dict[int, SingleLayerTranscoder] = {}
        self._layer_paths: Dict[int, Path] = {}

        # Set up cache directory for transcoder weights
        # FIXED: Handle case where cache_dir exists as a file (not directory)
        if self.cache_dir.exists() and not self.cache_dir.is_dir():
            logger.warning(f"Cache path {self.cache_dir} exists as file, not directory. Removing...")
            self.cache_dir.unlink()  # Remove the file
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized TranscoderSet for {config.model_name} "
            f"({config.num_layers} layers)"
        )

    def __getitem__(self, layer_idx: int) -> SingleLayerTranscoder:
        """Get transcoder for a specific layer, loading if necessary."""
        if layer_idx not in self._transcoders:
            self._load_layer(layer_idx)
        return self._transcoders[layer_idx]

    def __contains__(self, layer_idx: int) -> bool:
        """Check if a layer index is valid."""
        return 0 <= layer_idx < self.config.num_layers

    def __len__(self) -> int:
        """Return number of layers."""
        return self.config.num_layers

    def __iter__(self):
        """Iterate over all layers."""
        for i in range(self.config.num_layers):
            yield i, self[i]

    def _get_layer_path(self, layer_idx: int) -> Path:
        """Get the path to a layer's safetensors file, downloading if needed."""
        if layer_idx in self._layer_paths:
            return self._layer_paths[layer_idx]

        # Download from HuggingFace
        filename = f"layer_{layer_idx}.safetensors"
        local_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=filename,
            cache_dir=self.cache_dir,
        )

        self._layer_paths[layer_idx] = Path(local_path)
        return self._layer_paths[layer_idx]

    def _load_layer(self, layer_idx: int) -> None:
        """Load a single layer's transcoder."""
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(
                f"Layer {layer_idx} out of range [0, {self.config.num_layers})"
            )

        logger.info(f"Loading transcoder for layer {layer_idx}")

        path = self._get_layer_path(layer_idx)
        transcoder = SingleLayerTranscoder.from_safetensors(
            path=str(path),
            layer_idx=layer_idx,
            device=self.device,
            dtype=self.dtype,
            lazy_encoder=self.lazy_load,
            lazy_decoder=self.lazy_load,
        )

        self._transcoders[layer_idx] = transcoder

    def load_layers(self, layer_indices: Optional[List[int]] = None) -> None:
        """
        Pre-load transcoders for specified layers.

        Args:
            layer_indices: List of layer indices to load (None = all)
        """
        if layer_indices is None:
            layer_indices = list(range(self.config.num_layers))

        for idx in layer_indices:
            if idx not in self._transcoders:
                self._load_layer(idx)

        logger.info(f"Loaded {len(layer_indices)} transcoders")

    def get_loaded_layers(self) -> List[int]:
        """Return list of currently loaded layer indices."""
        return sorted(self._transcoders.keys())

    def unload_layer(self, layer_idx: int) -> None:
        """Unload a layer's transcoder to free memory."""
        if layer_idx in self._transcoders:
            del self._transcoders[layer_idx]
            torch.cuda.empty_cache()
            logger.debug(f"Unloaded transcoder for layer {layer_idx}")

    def encode(
        self,
        layer_idx: int,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode activations through a layer's transcoder.

        Args:
            layer_idx: Layer index
            activations: MLP input activations (..., hidden_size)

        Returns:
            Feature activations (..., d_transcoder)
        """
        return self[layer_idx].encode(activations)

    def decode(
        self,
        layer_idx: int,
        features: torch.Tensor,
        input_acts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode features through a layer's transcoder.

        Args:
            layer_idx: Layer index
            features: Feature activations (..., d_transcoder)
            input_acts: Original MLP inputs (for skip connection)

        Returns:
            Reconstructed MLP output (..., hidden_size)
        """
        return self[layer_idx].decode(features, input_acts)

    def get_feature_info(self, layer_idx: int) -> Dict:
        """Get information about features in a layer's transcoder."""
        tc = self[layer_idx]
        return {
            "layer_idx": layer_idx,
            "d_model": tc.d_model,
            "d_transcoder": tc.d_transcoder,
            "activation_type": tc.activation_function.__class__.__name__,
            "has_skip_connection": tc.W_skip is not None,
        }


def download_transcoder_weights(
    repo_id: str,
    cache_dir: Optional[Path] = None,
    layers: Optional[List[int]] = None,
) -> Path:
    """
    Download transcoder weights from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "mwhanna/qwen3-4b-transcoders")
        cache_dir: Local directory to cache downloads
        layers: Specific layers to download (None = all)

    Returns:
        Path to the cached repository
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "transcoders"

    if layers is None:
        # Download entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.yaml", "*.json"],
        )
        logger.info(f"Downloaded full transcoder set to {local_dir}")
        return Path(local_dir)
    else:
        # Download specific layers
        for layer_idx in layers:
            filename = f"layer_{layer_idx}.safetensors"
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
        logger.info(f"Downloaded {len(layers)} transcoder layers")
        return cache_dir


def load_transcoder_set(
    model_size: str = "4b",
    repo_id: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_load: bool = True,
    layers: Optional[List[int]] = None,
) -> TranscoderSet:
    """
    Load a set of pre-trained transcoders for a Qwen3 model.

    Args:
        model_size: Model size key ("0.6b", "1.7b", "4b", "8b", "14b")
        repo_id: Override HuggingFace repository ID
        cache_dir: Local cache directory for downloads
        device: Device to load transcoders to
        dtype: Data type for parameters
        lazy_load: If True, load individual transcoders on-demand
        layers: Specific layers to pre-load (None = load on-demand)

    Returns:
        TranscoderSet instance ready to use

    Example:
        >>> transcoders = load_transcoder_set("4b", layers=[10, 15, 20])
        >>> features = transcoders.encode(15, mlp_input_activations)
    """
    model_size = model_size.lower()

    if model_size not in QWEN3_CONFIGS:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Available: {list(QWEN3_CONFIGS.keys())}"
        )

    model_config = QWEN3_CONFIGS[model_size]
    repo = repo_id or QWEN3_TRANSCODERS[model_size]

    # Create config
    config = TranscoderConfig(
        model_name=model_config["model_name"],
        model_kind="transcoder_set",
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        num_layers=model_config["num_layers"],
        hidden_size=model_config["hidden_size"],
        repo_id=repo,
    )

    # Create transcoder set
    transcoder_set = TranscoderSet(
        config=config,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
        lazy_load=lazy_load,
    )

    # Pre-load specified layers
    if layers is not None:
        transcoder_set.load_layers(layers)

    return transcoder_set


def get_available_transcoders() -> Dict[str, str]:
    """Return dictionary of available Qwen3 transcoder repositories."""
    return QWEN3_TRANSCODERS.copy()


def get_model_config(model_size: str) -> Dict:
    """Get configuration for a specific model size."""
    model_size = model_size.lower()
    if model_size not in QWEN3_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")
    return QWEN3_CONFIGS[model_size].copy()
