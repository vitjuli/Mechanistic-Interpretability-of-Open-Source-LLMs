"""Dependency graph module for analyzing feature relationships.

This module builds and prunes dependency graphs that trace information
flow from input tokens through SAE features to output logits.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch import nn

from .config import Config
from .sparse_autoencoder import SparseAutoencoder


@dataclass
class GraphNode:
    """Represents a node in the dependency graph.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node ('input', 'feature', 'logit')
        layer: Layer index (None for input/logit nodes)
        feature_idx: Feature index within the layer (for feature nodes)
        token: Token string (for input nodes)
        logit_idx: Logit index (for logit nodes)
    """

    node_id: str
    node_type: str
    layer: Optional[int] = None
    feature_idx: Optional[int] = None
    token: Optional[str] = None
    logit_idx: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "layer": self.layer,
            "feature_idx": self.feature_idx,
            "token": self.token,
            "logit_idx": self.logit_idx,
        }


@dataclass
class GraphEdge:
    """Represents an edge in the dependency graph.

    Attributes:
        source: Source node ID
        target: Target node ID
        weight: Edge weight (attribution score)
        attribution_type: Type of attribution ('gradient', 'activation')
    """

    source: str
    target: str
    weight: float
    attribution_type: str = "gradient"

    def to_dict(self) -> dict:
        """Convert edge to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "attribution_type": self.attribution_type,
        }


class DependencyGraph:
    """Builds and manages dependency graphs for interpretability analysis.

    Traces information flow from inputs through SAE features to logits,
    with support for pruning to focus on the most important pathways.

    Attributes:
        config: Configuration object
        graph: NetworkX directed graph
        nodes: Dictionary of GraphNode objects
        edges: List of GraphEdge objects
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the dependency graph.

        Args:
            config: Configuration object for graph settings.
        """
        self.config = config or Config()
        self.graph = nx.DiGraph()
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def _make_node_id(
        self,
        node_type: str,
        layer: Optional[int] = None,
        idx: Optional[int] = None,
    ) -> str:
        """Create a unique node ID.

        Args:
            node_type: Type of the node.
            layer: Layer index (if applicable).
            idx: Index within the layer or vocabulary.

        Returns:
            Unique string identifier for the node.
        """
        if node_type == "input":
            return f"input_{idx}"
        elif node_type == "feature":
            return f"L{layer}_F{idx}"
        elif node_type == "logit":
            return f"logit_{idx}"
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def add_input_nodes(self, tokens: list[str]) -> list[str]:
        """Add input token nodes to the graph.

        Args:
            tokens: List of input token strings.

        Returns:
            List of node IDs for the input tokens.
        """
        node_ids = []
        for idx, token in enumerate(tokens):
            node_id = self._make_node_id("input", idx=idx)
            node = GraphNode(
                node_id=node_id,
                node_type="input",
                token=token,
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
            node_ids.append(node_id)
        return node_ids

    def add_feature_nodes(
        self, layer: int, num_features: int
    ) -> list[str]:
        """Add SAE feature nodes for a layer.

        Args:
            layer: Layer index.
            num_features: Number of features in the SAE.

        Returns:
            List of node IDs for the features.
        """
        node_ids = []
        for idx in range(num_features):
            node_id = self._make_node_id("feature", layer=layer, idx=idx)
            node = GraphNode(
                node_id=node_id,
                node_type="feature",
                layer=layer,
                feature_idx=idx,
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
            node_ids.append(node_id)
        return node_ids

    def add_logit_nodes(self, logit_indices: list[int]) -> list[str]:
        """Add output logit nodes.

        Args:
            logit_indices: List of vocabulary indices for logit nodes.

        Returns:
            List of node IDs for the logits.
        """
        node_ids = []
        for idx in logit_indices:
            node_id = self._make_node_id("logit", idx=idx)
            node = GraphNode(
                node_id=node_id,
                node_type="logit",
                logit_idx=idx,
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
            node_ids.append(node_id)
        return node_ids

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        attribution_type: str = "gradient",
    ) -> None:
        """Add an edge between nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge weight (attribution score).
            attribution_type: Type of attribution method used.
        """
        edge = GraphEdge(
            source=source,
            target=target,
            weight=weight,
            attribution_type=attribution_type,
        )
        self.edges.append(edge)
        self.graph.add_edge(source, target, **edge.to_dict())

    def compute_feature_attributions(
        self,
        activations: torch.Tensor,
        feature_activations: torch.Tensor,
        layer: int,
    ) -> np.ndarray:
        """Compute attribution scores from inputs to features.

        Uses activation-based attribution: the product of input activations
        and feature activations provides a measure of information flow.

        Args:
            activations: Input activations of shape (seq_len, hidden_dim).
            feature_activations: SAE features of shape (seq_len, num_features).
            layer: Layer index.

        Returns:
            Attribution matrix of shape (seq_len, num_features).
        """
        # Simple activation-based attribution
        # More sophisticated methods (gradients, attention) could be added
        input_norms = torch.norm(activations, dim=-1, keepdim=True)
        feature_norms = torch.abs(feature_activations)

        # Attribution = input_norm * feature_activation
        attributions = (input_norms * feature_norms).numpy()
        return attributions

    def build_from_activations(
        self,
        tokens: list[str],
        layer_activations: dict[int, torch.Tensor],
        layer_features: dict[int, torch.Tensor],
        decisive_logits: list[int],
        logit_attributions: Optional[dict[int, np.ndarray]] = None,
    ) -> None:
        """Build the dependency graph from activations and features.

        Args:
            tokens: Input token strings.
            layer_activations: Dict mapping layer indices to activations.
            layer_features: Dict mapping layer indices to SAE feature activations.
            decisive_logits: List of important logit indices.
            logit_attributions: Optional pre-computed logit attributions.
        """
        # Add input nodes
        input_ids = self.add_input_nodes(tokens)

        # Add logit nodes
        self.add_logit_nodes(decisive_logits)

        prev_layer_ids = input_ids

        # Process each layer
        for layer in sorted(layer_activations.keys()):
            activations = layer_activations[layer]
            features = layer_features[layer]

            # Add feature nodes for this layer
            num_features = features.size(-1)
            feature_ids = self.add_feature_nodes(layer, num_features)

            # Compute attributions from previous layer to features
            attributions = self.compute_feature_attributions(
                activations, features, layer
            )

            # Add edges from inputs/previous features to current features
            # For simplicity, connect tokens to features based on position
            for pos_idx, input_id in enumerate(prev_layer_ids[:len(tokens)]):
                if pos_idx < attributions.shape[0]:
                    for feat_idx, feat_id in enumerate(feature_ids):
                        weight = float(attributions[pos_idx, feat_idx])
                        if weight > self.config.graph_prune_threshold:
                            self.add_edge(input_id, feat_id, weight)

            # For connecting to next layer
            prev_layer_ids = feature_ids

        # Connect final layer features to logits
        if logit_attributions is not None:
            last_layer = max(layer_activations.keys())
            last_features = layer_features[last_layer]
            num_features = last_features.size(-1)

            for logit_idx in decisive_logits:
                logit_id = self._make_node_id("logit", idx=logit_idx)
                if logit_idx in logit_attributions:
                    attr = logit_attributions[logit_idx]
                    for feat_idx in range(min(num_features, len(attr))):
                        weight = float(attr[feat_idx])
                        if weight > self.config.graph_prune_threshold:
                            feat_id = self._make_node_id(
                                "feature", layer=last_layer, idx=feat_idx
                            )
                            self.add_edge(feat_id, logit_id, weight)

    def prune(self, threshold: Optional[float] = None) -> "DependencyGraph":
        """Prune edges below the threshold weight.

        Args:
            threshold: Minimum edge weight to keep. Uses config if None.

        Returns:
            New pruned DependencyGraph.
        """
        threshold = threshold or self.config.graph_prune_threshold

        pruned = DependencyGraph(self.config)
        pruned.nodes = dict(self.nodes)

        # Copy nodes
        for node_id, node in self.nodes.items():
            pruned.graph.add_node(node_id, **node.to_dict())

        # Copy edges above threshold
        for edge in self.edges:
            if abs(edge.weight) >= threshold:
                pruned.edges.append(edge)
                pruned.graph.add_edge(
                    edge.source, edge.target, **edge.to_dict()
                )

        # Remove isolated nodes (except input and logit nodes)
        isolated = [
            node
            for node in pruned.graph.nodes()
            if pruned.graph.degree(node) == 0
            and pruned.nodes[node].node_type == "feature"
        ]
        pruned.graph.remove_nodes_from(isolated)
        for node_id in isolated:
            del pruned.nodes[node_id]

        return pruned

    def get_top_features(
        self, k: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """Get the top-k most important features by total edge weight.

        Args:
            k: Number of features to return. Uses config if None.

        Returns:
            List of (node_id, importance_score) tuples.
        """
        k = k or self.config.graph_top_k_features

        feature_importance: dict[str, float] = {}
        for edge in self.edges:
            if self.nodes[edge.source].node_type == "feature":
                feature_importance[edge.source] = (
                    feature_importance.get(edge.source, 0) + abs(edge.weight)
                )
            if self.nodes[edge.target].node_type == "feature":
                feature_importance[edge.target] = (
                    feature_importance.get(edge.target, 0) + abs(edge.weight)
                )

        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:k]

    def find_paths_to_logit(
        self, logit_idx: int, max_paths: int = 10
    ) -> list[list[str]]:
        """Find paths from input nodes to a specific logit.

        Args:
            logit_idx: Target logit index.
            max_paths: Maximum number of paths to return.

        Returns:
            List of paths, where each path is a list of node IDs.
        """
        logit_id = self._make_node_id("logit", idx=logit_idx)

        if logit_id not in self.graph:
            return []

        paths = []
        for node_id, node in self.nodes.items():
            if node.node_type == "input":
                try:
                    for path in nx.all_simple_paths(
                        self.graph, node_id, logit_id
                    ):
                        paths.append(path)
                        if len(paths) >= max_paths:
                            return paths
                except nx.NetworkXNoPath:
                    continue

        return paths

    def to_dict(self) -> dict:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary with nodes and edges.
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def save(self, path: str) -> None:
        """Save the graph to a JSON file.

        Args:
            path: Path to save the graph.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, config: Optional[Config] = None) -> "DependencyGraph":
        """Load a graph from a JSON file.

        Args:
            path: Path to the saved graph.
            config: Configuration object.

        Returns:
            Loaded DependencyGraph.
        """
        graph = cls(config)

        with open(path, "r") as f:
            data = json.load(f)

        for node_dict in data["nodes"]:
            node = GraphNode(**node_dict)
            graph.nodes[node.node_id] = node
            graph.graph.add_node(node.node_id, **node.to_dict())

        for edge_dict in data["edges"]:
            edge = GraphEdge(**edge_dict)
            graph.edges.append(edge)
            graph.graph.add_edge(edge.source, edge.target, **edge.to_dict())

        return graph
