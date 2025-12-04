"""Activation capture module for extracting hidden states from Qwen3-4B.

This module provides functionality to run the model on prompts and
capture intermediate layer activations for interpretability analysis.
"""

from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


class ActivationCapture:
    """Captures activations from specified layers of a transformer model.

    This class hooks into transformer layers to extract hidden states
    during forward passes, enabling analysis of internal representations.

    Attributes:
        config: Configuration object with model and capture settings
        model: The loaded transformer model
        tokenizer: The tokenizer for the model
        hooks: List of registered forward hooks
        activations: Dictionary mapping layer indices to captured activations
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """Initialize the activation capture module.

        Args:
            config: Configuration object. If None, uses default Config.
            model: Pre-loaded model. If None, loads from config.model_name.
            tokenizer: Pre-loaded tokenizer. If None, loads from config.model_name.
        """
        self.config = config or Config()
        self.hooks: list = []
        self.activations: dict[int, list[torch.Tensor]] = {}

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model = None
            self.tokenizer = None

    def load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _get_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer.

        Args:
            layer_idx: Index of the layer to hook.

        Returns:
            A hook function that captures the layer's output.
        """

        def hook(module: nn.Module, input_t: tuple, output: tuple) -> None:
            # For transformer layers, output is typically (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            # Detach and move to CPU to avoid memory issues
            self.activations[layer_idx].append(hidden_states.detach().cpu())

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        self.remove_hooks()

        # Get the transformer layers
        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer"):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        for layer_idx in self.config.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._get_hook(layer_idx)
                )
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_activations(self) -> None:
        """Clear captured activations."""
        self.activations.clear()

    @torch.no_grad()
    def capture(
        self,
        prompts: list[str],
        show_progress: bool = True,
    ) -> dict[int, torch.Tensor]:
        """Capture activations for a list of prompts.

        Args:
            prompts: List of text prompts to process.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping layer indices to concatenated activation tensors.
            Each tensor has shape (total_tokens, hidden_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.clear_activations()
        self.register_hooks()

        iterator = tqdm(prompts, desc="Capturing activations") if show_progress else prompts

        for prompt in iterator:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Forward pass (hooks capture activations)
            self.model(**inputs)

        self.remove_hooks()

        # Concatenate activations across prompts for each layer
        result = {}
        for layer_idx, act_list in self.activations.items():
            # Each activation is (batch, seq_len, hidden_dim)
            # Flatten to (total_tokens, hidden_dim)
            concatenated = torch.cat(
                [a.view(-1, a.size(-1)) for a in act_list], dim=0
            )
            result[layer_idx] = concatenated

        return result

    def save_activations(
        self,
        activations: dict[int, torch.Tensor],
        behavior: str,
    ) -> None:
        """Save captured activations to disk.

        Args:
            activations: Dictionary of layer activations.
            behavior: Behavior category name for the file prefix.
        """
        import os

        os.makedirs(self.config.activations_dir, exist_ok=True)

        for layer_idx, activation in activations.items():
            path = os.path.join(
                self.config.activations_dir,
                f"{behavior}_layer_{layer_idx}.pt",
            )
            torch.save(activation, path)

    def load_activations(self, behavior: str) -> dict[int, torch.Tensor]:
        """Load previously saved activations from disk.

        Args:
            behavior: Behavior category name for the file prefix.

        Returns:
            Dictionary mapping layer indices to activation tensors.
        """
        import os

        activations = {}
        for layer_idx in self.config.target_layers:
            path = os.path.join(
                self.config.activations_dir,
                f"{behavior}_layer_{layer_idx}.pt",
            )
            if os.path.exists(path):
                activations[layer_idx] = torch.load(path, weights_only=True)
        return activations
