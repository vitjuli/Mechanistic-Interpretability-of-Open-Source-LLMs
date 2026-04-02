"""
Intervention methods for validating circuit hypotheses.

Implements three intervention types from Anthropic's methodology:
1. Feature Ablation: Zero or negate feature activations
2. Activation Patching: Swap activations between prompts
3. Feature Steering: Inject feature activations to modify behaviour

Reference: "On the Biology of a Large Language Model" (Lindsey et al., 2025)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from src.sae import SparseAutoencoder


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    prompt: str
    baseline_logit_diff: float
    intervened_logit_diff: float
    effect_size: float  # Change in logit diff
    relative_effect: float  # Effect / baseline
    intervention_type: str
    target_features: List[Tuple[int, int]]  # (layer, feature_idx)
    metadata: Dict


class FeatureAblation:
    """
    Zero-ablation and inhibition of SAE features.

    Ablation: Set feature activation to 0
    Inhibition: Set feature activation to -M * original (M > 0)
    """

    def __init__(
        self,
        model,
        saes: Dict[int, SparseAutoencoder],
        device: torch.device,
    ):
        self.model = model
        self.saes = saes
        self.device = device

    def ablate_features(
        self,
        activations: torch.Tensor,
        sae: SparseAutoencoder,
        feature_indices: List[int],
        mode: str = "zero",
        inhibition_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Modify activations by ablating specific SAE features.

        Args:
            activations: Original hidden state (batch, hidden_dim)
            sae: SAE for the layer
            feature_indices: Features to ablate
            mode: "zero" (set to 0) or "inhibit" (negate)
            inhibition_factor: Multiplier for inhibition mode

        Returns:
            Modified activations (batch, hidden_dim)
        """
        with torch.no_grad():
            # Encode to feature space
            features = sae.encode(activations)

            # Store original features
            original_features = features.clone()

            # Apply intervention
            for idx in feature_indices:
                if mode == "zero":
                    features[:, idx] = 0.0
                elif mode == "inhibit":
                    features[:, idx] = -inhibition_factor * original_features[:, idx]

            # Decode back
            modified_activations = sae.decode(features)

        return modified_activations

    def compute_ablation_effect(
        self,
        activations: torch.Tensor,
        sae: SparseAutoencoder,
        feature_indices: List[int],
    ) -> Dict[str, float]:
        """
        Compute the effect of ablating features on reconstruction.

        Returns:
            Dictionary with reconstruction error metrics
        """
        with torch.no_grad():
            # Original reconstruction
            original_recon, original_features = sae(activations)

            # Ablated reconstruction
            ablated_features = original_features.clone()
            for idx in feature_indices:
                ablated_features[:, idx] = 0.0
            ablated_recon = sae.decode(ablated_features)

            # Compute difference
            diff = (original_recon - ablated_recon).norm(dim=-1).mean().item()
            original_norm = original_recon.norm(dim=-1).mean().item()

            return {
                "reconstruction_diff": diff,
                "relative_diff": diff / (original_norm + 1e-8),
                "n_ablated_features": len(feature_indices),
            }


class ActivationPatching:
    """
    Swap activations between prompts to test causal relationships.

    Given prompts A and B, replace activations from A with those from B
    at specific layers/features and measure the effect on output.
    """

    def __init__(
        self,
        model,
        saes: Dict[int, SparseAutoencoder],
        device: torch.device,
    ):
        self.model = model
        self.saes = saes
        self.device = device

    def patch_features(
        self,
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
        sae: SparseAutoencoder,
        feature_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Replace specific features from source with those from target.

        Args:
            source_activations: Original activations to modify
            target_activations: Activations to patch in
            sae: SAE for the layer
            feature_indices: Which features to patch (None = all)

        Returns:
            Patched activations
        """
        with torch.no_grad():
            source_features = sae.encode(source_activations)
            target_features = sae.encode(target_activations)

            if feature_indices is None:
                # Patch all features
                patched_features = target_features
            else:
                patched_features = source_features.clone()
                for idx in feature_indices:
                    patched_features[:, idx] = target_features[:, idx]

            patched_activations = sae.decode(patched_features)

        return patched_activations

    def compute_feature_similarity(
        self,
        acts_a: torch.Tensor,
        acts_b: torch.Tensor,
        sae: SparseAutoencoder,
    ) -> Dict[str, float]:
        """
        Compute similarity between feature representations.

        Returns:
            Dictionary with similarity metrics
        """
        with torch.no_grad():
            features_a = sae.encode(acts_a)
            features_b = sae.encode(acts_b)

            cosine_sim = F.cosine_similarity(features_a, features_b, dim=-1).mean().item()
            l2_dist = (features_a - features_b).norm(dim=-1).mean().item()

            # Overlap: fraction of mutually active features
            active_a = (features_a > 0).float()
            active_b = (features_b > 0).float()
            intersection = (active_a * active_b).sum(dim=-1)
            union = ((active_a + active_b) > 0).float().sum(dim=-1)
            iou = (intersection / (union + 1e-8)).mean().item()

            return {
                "cosine_similarity": cosine_sim,
                "l2_distance": l2_dist,
                "feature_iou": iou,
            }


class FeatureSteering:
    """
    Inject or amplify feature activations to steer model behaviour.

    Based on Anthropic's finding that adding feature activations
    can predictably modify model outputs.
    """

    def __init__(
        self,
        model,
        saes: Dict[int, SparseAutoencoder],
        device: torch.device,
    ):
        self.model = model
        self.saes = saes
        self.device = device

    def steer_features(
        self,
        activations: torch.Tensor,
        sae: SparseAutoencoder,
        feature_idx: int,
        steering_strength: float = 5.0,
    ) -> torch.Tensor:
        """
        Amplify a specific feature to steer behaviour.

        Args:
            activations: Original hidden state
            sae: SAE for the layer
            feature_idx: Which feature to amplify
            steering_strength: How much to add (in activation units)

        Returns:
            Modified activations
        """
        with torch.no_grad():
            features = sae.encode(activations)
            features[:, feature_idx] += steering_strength
            steered_activations = sae.decode(features)

        return steered_activations
