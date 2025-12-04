"""Intervention validation module for mechanistic interpretability.

This module implements inhibition and swap-in style interventions
to validate the causal importance of identified features.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from tqdm import tqdm

from .config import Config
from .sparse_autoencoder import SparseAutoencoder


@dataclass
class InterventionResult:
    """Results from an intervention experiment.

    Attributes:
        feature_id: Identifier of the intervened feature
        baseline_logits: Logits without intervention
        intervention_logits: Logits with intervention
        logit_diff: Difference in target logit
        kl_divergence: KL divergence between output distributions
        success: Whether the intervention had the expected effect
    """

    feature_id: str
    baseline_logits: torch.Tensor
    intervention_logits: torch.Tensor
    logit_diff: float
    kl_divergence: float
    success: bool

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "feature_id": self.feature_id,
            "logit_diff": self.logit_diff,
            "kl_divergence": self.kl_divergence,
            "success": self.success,
        }


class InterventionValidator:
    """Validates feature importance through causal interventions.

    Implements inhibition (zeroing out features) and swap-in (replacing
    features with values from other contexts) to test whether identified
    features causally affect model outputs.

    Attributes:
        config: Configuration object
        model: The transformer model
        saes: Dictionary mapping layer indices to trained SAEs
    """

    def __init__(
        self,
        model: nn.Module,
        saes: dict[int, SparseAutoencoder],
        config: Optional[Config] = None,
    ):
        """Initialize the intervention validator.

        Args:
            model: The transformer model to intervene on.
            saes: Dictionary mapping layer indices to SAEs.
            config: Configuration object.
        """
        self.model = model
        self.saes = saes
        self.config = config or Config()
        self.hooks: list = []

    def _get_model_layers(self):
        """Get the transformer layers from the model.

        Returns:
            The model's transformer layers.

        Raises:
            ValueError: If the model architecture is not recognized.
        """
        if hasattr(self.model, "model"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer"):
            return self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

    def _get_inhibition_hook(
        self,
        layer: int,
        feature_indices: list[int],
        sae: SparseAutoencoder,
    ) -> Callable:
        """Create a hook that zeros out specific SAE features.

        Args:
            layer: Layer index.
            feature_indices: Indices of features to inhibit.
            sae: The SAE for this layer.

        Returns:
            A forward hook function.
        """
        device = next(sae.parameters()).device

        def hook(
            module: nn.Module, input_t: tuple, output: tuple
        ) -> tuple:
            # Get hidden states from output
            hidden_states = output[0] if isinstance(output, tuple) else output
            original_shape = hidden_states.shape
            original_dtype = hidden_states.dtype

            # Flatten for SAE processing
            flat_hidden = hidden_states.view(-1, hidden_states.size(-1))

            # Encode through SAE
            with torch.no_grad():
                features = sae.encode(flat_hidden.to(device).float())

                # Zero out target features (inhibition)
                for idx in feature_indices:
                    if idx < features.size(-1):
                        features[:, idx] = 0.0

                # Decode back to activation space
                modified = sae.decode(features)

            # Reshape to original
            modified = modified.view(original_shape).to(original_dtype)

            # Return modified output
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def _get_swap_hook(
        self,
        layer: int,
        feature_indices: list[int],
        source_features: torch.Tensor,
        sae: SparseAutoencoder,
    ) -> Callable:
        """Create a hook that swaps in features from another context.

        Args:
            layer: Layer index.
            feature_indices: Indices of features to swap.
            source_features: Feature values to swap in.
            sae: The SAE for this layer.

        Returns:
            A forward hook function.
        """
        device = next(sae.parameters()).device

        def hook(
            module: nn.Module, input_t: tuple, output: tuple
        ) -> tuple:
            hidden_states = output[0] if isinstance(output, tuple) else output
            original_shape = hidden_states.shape
            original_dtype = hidden_states.dtype

            flat_hidden = hidden_states.view(-1, hidden_states.size(-1))

            with torch.no_grad():
                features = sae.encode(flat_hidden.to(device).float())

                # Swap in source features
                for idx in feature_indices:
                    if idx < features.size(-1) and idx < source_features.size(-1):
                        # Use mean of source features for this index
                        swap_value = source_features[:, idx].mean()
                        features[:, idx] = swap_value

                modified = sae.decode(features)

            modified = modified.view(original_shape).to(original_dtype)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def _register_hook(
        self, layer: int, hook: Callable
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific layer.

        Args:
            layer: Layer index.
            hook: Hook function to register.

        Returns:
            Handle for the registered hook.
        """
        layers = self._get_model_layers()
        handle = layers[layer].register_forward_hook(hook)
        self.hooks.append(handle)
        return handle

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def get_baseline_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get model logits without any intervention.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Logits tensor.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    @torch.no_grad()
    def inhibit_features(
        self,
        input_ids: torch.Tensor,
        layer: int,
        feature_indices: list[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model with specified features inhibited (zeroed out).

        Args:
            input_ids: Input token IDs.
            layer: Layer to intervene on.
            feature_indices: Feature indices to inhibit.
            attention_mask: Attention mask.

        Returns:
            Tuple of (baseline_logits, intervention_logits).
        """
        # Get baseline
        baseline_logits = self.get_baseline_logits(input_ids, attention_mask)

        # Register inhibition hook
        if layer not in self.saes:
            raise ValueError(f"No SAE available for layer {layer}")

        sae = self.saes[layer]
        hook = self._get_inhibition_hook(layer, feature_indices, sae)
        self._register_hook(layer, hook)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            intervention_logits = outputs.logits
        finally:
            self._clear_hooks()

        return baseline_logits, intervention_logits

    @torch.no_grad()
    def swap_features(
        self,
        input_ids: torch.Tensor,
        source_input_ids: torch.Tensor,
        layer: int,
        feature_indices: list[int],
        attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model with features swapped from a source context.

        Args:
            input_ids: Target input token IDs.
            source_input_ids: Source input token IDs (to get features from).
            layer: Layer to intervene on.
            feature_indices: Feature indices to swap.
            attention_mask: Attention mask for target.
            source_attention_mask: Attention mask for source.

        Returns:
            Tuple of (baseline_logits, intervention_logits).
        """
        if layer not in self.saes:
            raise ValueError(f"No SAE available for layer {layer}")

        sae = self.saes[layer]

        # Get source activations and features
        source_activations: list = []

        def capture_hook(module: nn.Module, input_t: tuple, output: tuple):
            hidden = output[0] if isinstance(output, tuple) else output
            source_activations.append(hidden.detach())

        layers = self._get_model_layers()
        handle = layers[layer].register_forward_hook(capture_hook)

        try:
            self.model(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
            )
        finally:
            handle.remove()

        # Get source features
        source_hidden = source_activations[0]
        flat_source = source_hidden.view(-1, source_hidden.size(-1))
        source_features = sae.encode(
            flat_source.to(next(sae.parameters()).device).float()
        )

        # Get baseline
        baseline_logits = self.get_baseline_logits(input_ids, attention_mask)

        # Register swap hook
        hook = self._get_swap_hook(layer, feature_indices, source_features, sae)
        self._register_hook(layer, hook)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            intervention_logits = outputs.logits
        finally:
            self._clear_hooks()

        return baseline_logits, intervention_logits

    def compute_kl_divergence(
        self,
        baseline_logits: torch.Tensor,
        intervention_logits: torch.Tensor,
    ) -> float:
        """Compute KL divergence between baseline and intervention distributions.

        Args:
            baseline_logits: Logits from baseline model.
            intervention_logits: Logits from intervened model.

        Returns:
            KL divergence value.
        """
        baseline_probs = torch.softmax(baseline_logits[:, -1, :], dim=-1)
        intervention_probs = torch.softmax(intervention_logits[:, -1, :], dim=-1)

        # Add small epsilon for numerical stability
        eps = 1e-10
        kl = torch.sum(
            baseline_probs
            * (
                torch.log(baseline_probs + eps)
                - torch.log(intervention_probs + eps)
            ),
            dim=-1,
        )
        return kl.mean().item()

    def validate_inhibition(
        self,
        input_ids: torch.Tensor,
        layer: int,
        feature_indices: list[int],
        target_logit: int,
        expected_decrease: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> InterventionResult:
        """Validate a feature's importance through inhibition.

        Args:
            input_ids: Input token IDs.
            layer: Layer to intervene on.
            feature_indices: Features to inhibit.
            target_logit: Target logit index to measure.
            expected_decrease: Whether we expect the target logit to decrease.
            attention_mask: Attention mask.

        Returns:
            InterventionResult with validation results.
        """
        baseline, intervention = self.inhibit_features(
            input_ids, layer, feature_indices, attention_mask
        )

        # Measure change in target logit
        baseline_target = baseline[0, -1, target_logit].item()
        intervention_target = intervention[0, -1, target_logit].item()
        logit_diff = intervention_target - baseline_target

        # Compute KL divergence
        kl_div = self.compute_kl_divergence(baseline, intervention)

        # Check if intervention had expected effect
        if expected_decrease:
            success = logit_diff < 0
        else:
            success = logit_diff > 0

        feature_id = f"L{layer}_F{feature_indices}"

        return InterventionResult(
            feature_id=feature_id,
            baseline_logits=baseline,
            intervention_logits=intervention,
            logit_diff=logit_diff,
            kl_divergence=kl_div,
            success=success,
        )

    def validate_features(
        self,
        input_ids: torch.Tensor,
        layer: int,
        feature_indices: list[int],
        target_logit: int,
        attention_mask: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> list[InterventionResult]:
        """Validate multiple features through individual inhibitions.

        Args:
            input_ids: Input token IDs.
            layer: Layer to intervene on.
            feature_indices: List of feature indices to validate.
            target_logit: Target logit index.
            attention_mask: Attention mask.
            show_progress: Whether to show progress bar.

        Returns:
            List of InterventionResults for each feature.
        """
        results = []
        iterator = (
            tqdm(feature_indices, desc="Validating features")
            if show_progress
            else feature_indices
        )

        for feat_idx in iterator:
            result = self.validate_inhibition(
                input_ids=input_ids,
                layer=layer,
                feature_indices=[feat_idx],
                target_logit=target_logit,
                attention_mask=attention_mask,
            )
            results.append(result)

        return results

    def summarize_results(
        self, results: list[InterventionResult]
    ) -> dict:
        """Summarize validation results.

        Args:
            results: List of intervention results.

        Returns:
            Dictionary with summary statistics.
        """
        if not results:
            return {"num_features": 0}

        successful = [r for r in results if r.success]
        logit_diffs = [r.logit_diff for r in results]
        kl_divs = [r.kl_divergence for r in results]

        return {
            "num_features": len(results),
            "num_successful": len(successful),
            "success_rate": len(successful) / len(results),
            "mean_logit_diff": sum(logit_diffs) / len(logit_diffs),
            "max_logit_diff": max(logit_diffs),
            "min_logit_diff": min(logit_diffs),
            "mean_kl_divergence": sum(kl_divs) / len(kl_divs),
        }
