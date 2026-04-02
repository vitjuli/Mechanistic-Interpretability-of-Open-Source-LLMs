"""
Gradient-based attribution for building feature dependency graphs.

Implements a simplified version of the attribution graph methodology from
Anthropic's "On the Biology of a Large Language Model" (2025) and the
companion methods paper.

Key concepts:
- Edge weight = activation x gradient (first-order attribution)
- Graph nodes: input tokens, SAE features, output logits
- Pruning: keep only edges above attribution threshold
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx

from src.sae import SparseAutoencoder


class GradientAttribution:
    """
    Compute gradient-based attribution scores between SAE features and outputs.

    Simplified version of Anthropic's backward Jacobian approach.
    Uses activation x gradient as the attribution score.
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

    @torch.no_grad()
    def get_feature_activations(
        self,
        hidden_states: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Encode hidden states through SAEs to get feature activations."""
        features = {}
        for layer, sae in self.saes.items():
            if layer in hidden_states:
                h = hidden_states[layer]
                features[layer] = sae.encode(h)
        return features

    def compute_logit_attribution(
        self,
        hidden_state: torch.Tensor,
        sae: SparseAutoencoder,
        target_logit_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attribution of each SAE feature to target logit.

        Uses the chain: feature activation * (encoder weight^T @ logit gradient)

        Args:
            hidden_state: Residual stream activation (hidden_dim,)
            sae: Trained SAE for this layer
            target_logit_grad: Gradient of target logit w.r.t. hidden state

        Returns:
            Attribution scores per feature (latent_dim,)
        """
        with torch.no_grad():
            # Get feature activations
            features = sae.encode(hidden_state.unsqueeze(0)).squeeze(0)

            # Project gradient into feature space via encoder
            grad_in_feature_space = F.linear(
                target_logit_grad - sae.pre_bias,
                sae.encoder.weight,
                sae.encoder.bias,
            )

            # Attribution = activation * projected gradient
            attribution = features * grad_in_feature_space

        return attribution

    def compute_feature_to_feature_attribution(
        self,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
        sae_source: SparseAutoencoder,
        sae_target: SparseAutoencoder,
    ) -> torch.Tensor:
        """
        Compute attribution between features in adjacent layers.

        Returns:
            Attribution matrix (source_features, target_features)
        """
        with torch.no_grad():
            # Decoder of source -> encoder of target gives virtual weight
            # virtual_weight[i,j] = sum of paths from source feature i to target feature j
            virtual_weight = sae_target.encoder.weight @ sae_source.decoder.weight.T
            # Scale by activations
            attribution = features_source.unsqueeze(1) * virtual_weight.T * features_target.unsqueeze(0)

        return attribution


class AttributionGraph:
    """
    Directed graph representing feature dependencies from input to output.

    Nodes: input tokens, SAE features, output logits
    Edges: attribution scores between connected nodes
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_input_node(self, token: str, position: int):
        """Add an input token node."""
        node_id = f"input_{position}"
        self.graph.add_node(node_id, type="input", token=token, position=position)
        return node_id

    def add_feature_node(
        self,
        layer: int,
        feature_idx: int,
        activation: float,
        label: str = "",
    ):
        """Add an SAE feature node."""
        node_id = f"L{layer}_F{feature_idx}"
        self.graph.add_node(
            node_id,
            type="feature",
            layer=layer,
            feature_idx=feature_idx,
            activation=activation,
            label=label,
        )
        return node_id

    def add_output_node(self, token: str, logit: float):
        """Add an output logit node."""
        node_id = f"output_{token}"
        self.graph.add_node(node_id, type="output", token=token, logit=logit)
        return node_id

    def add_edge(self, source: str, target: str, attribution: float):
        """Add a directed edge with attribution weight."""
        self.graph.add_edge(source, target, weight=attribution)

    def prune(
        self,
        top_k_edges_per_node: int = 10,
        min_attribution: float = 0.01,
    ):
        """Remove low-attribution edges and disconnected nodes."""
        # For each node, keep only top-k outgoing edges
        for node in list(self.graph.nodes()):
            out_edges = list(self.graph.out_edges(node, data=True))
            if len(out_edges) > top_k_edges_per_node:
                out_edges.sort(key=lambda e: abs(e[2].get("weight", 0)), reverse=True)
                for _, target, _ in out_edges[top_k_edges_per_node:]:
                    self.graph.remove_edge(node, target)

        # Remove edges below threshold
        edges_to_remove = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if abs(d.get("weight", 0)) < min_attribution
        ]
        self.graph.remove_edges_from(edges_to_remove)

        # Remove isolated nodes (except input/output)
        isolated = [
            n for n in self.graph.nodes()
            if self.graph.degree(n) == 0
            and self.graph.nodes[n].get("type") == "feature"
        ]
        self.graph.remove_nodes_from(isolated)

    def get_top_features(self, n: int = 20) -> List[Dict]:
        """Get top features by total attribution flow."""
        features = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "feature":
                continue

            in_attr = sum(
                abs(d.get("weight", 0))
                for _, _, d in self.graph.in_edges(node, data=True)
            )
            out_attr = sum(
                abs(d.get("weight", 0))
                for _, _, d in self.graph.out_edges(node, data=True)
            )

            features.append({
                "node_id": node,
                "layer": data["layer"],
                "feature_idx": data["feature_idx"],
                "activation": data.get("activation", 0),
                "total_in_attribution": in_attr,
                "total_out_attribution": out_attr,
                "total_flow": in_attr + out_attr,
            })

        features.sort(key=lambda x: x["total_flow"], reverse=True)
        return features[:n]

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {"id": n, **d}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
        }

    def to_networkx(self) -> nx.DiGraph:
        """Return underlying NetworkX graph."""
        return self.graph
