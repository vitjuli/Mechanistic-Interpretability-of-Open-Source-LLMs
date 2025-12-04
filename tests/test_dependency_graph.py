"""Tests for the dependency graph module."""

import pytest
import json
import torch
import numpy as np

from src.mi_pipeline.dependency_graph import (
    DependencyGraph,
    GraphNode,
    GraphEdge,
)
from src.mi_pipeline.config import Config


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_input_node(self):
        """Test creating an input node."""
        node = GraphNode(
            node_id="input_0",
            node_type="input",
            token="hello",
        )
        assert node.node_id == "input_0"
        assert node.node_type == "input"
        assert node.token == "hello"

    def test_feature_node(self):
        """Test creating a feature node."""
        node = GraphNode(
            node_id="L8_F42",
            node_type="feature",
            layer=8,
            feature_idx=42,
        )
        assert node.layer == 8
        assert node.feature_idx == 42

    def test_to_dict(self):
        """Test converting node to dictionary."""
        node = GraphNode(
            node_id="logit_100",
            node_type="logit",
            logit_idx=100,
        )
        d = node.to_dict()
        assert d["node_id"] == "logit_100"
        assert d["logit_idx"] == 100


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = GraphEdge(
            source="input_0",
            target="L8_F42",
            weight=0.5,
        )
        assert edge.source == "input_0"
        assert edge.target == "L8_F42"
        assert edge.weight == 0.5

    def test_to_dict(self):
        """Test converting edge to dictionary."""
        edge = GraphEdge(
            source="L8_F42",
            target="logit_100",
            weight=0.3,
            attribution_type="activation",
        )
        d = edge.to_dict()
        assert d["attribution_type"] == "activation"


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    def test_initialization(self):
        """Test graph initialization."""
        config = Config()
        graph = DependencyGraph(config)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_input_nodes(self):
        """Test adding input nodes."""
        graph = DependencyGraph()
        tokens = ["hello", "world"]
        node_ids = graph.add_input_nodes(tokens)

        assert len(node_ids) == 2
        assert "input_0" in graph.nodes
        assert graph.nodes["input_0"].token == "hello"

    def test_add_feature_nodes(self):
        """Test adding feature nodes."""
        graph = DependencyGraph()
        node_ids = graph.add_feature_nodes(layer=8, num_features=10)

        assert len(node_ids) == 10
        assert "L8_F0" in graph.nodes
        assert graph.nodes["L8_F5"].layer == 8
        assert graph.nodes["L8_F5"].feature_idx == 5

    def test_add_logit_nodes(self):
        """Test adding logit nodes."""
        graph = DependencyGraph()
        node_ids = graph.add_logit_nodes([100, 200, 300])

        assert len(node_ids) == 3
        assert graph.nodes["logit_200"].logit_idx == 200

    def test_add_edge(self):
        """Test adding edges."""
        graph = DependencyGraph()
        graph.add_input_nodes(["test"])
        graph.add_feature_nodes(layer=8, num_features=5)

        graph.add_edge("input_0", "L8_F2", weight=0.5)
        assert len(graph.edges) == 1
        assert graph.edges[0].weight == 0.5

    def test_prune(self):
        """Test graph pruning."""
        config = Config(graph_prune_threshold=0.3)
        graph = DependencyGraph(config)

        graph.add_input_nodes(["test"])
        graph.add_feature_nodes(layer=8, num_features=3)
        graph.add_logit_nodes([0])

        graph.add_edge("input_0", "L8_F0", weight=0.5)  # Keep
        graph.add_edge("input_0", "L8_F1", weight=0.1)  # Prune
        graph.add_edge("L8_F0", "logit_0", weight=0.4)  # Keep

        pruned = graph.prune(threshold=0.3)

        assert len(pruned.edges) == 2
        # L8_F1 should be removed as isolated feature node
        assert "L8_F1" not in pruned.nodes

    def test_get_top_features(self):
        """Test getting top features by importance."""
        graph = DependencyGraph()
        graph.add_input_nodes(["test"])
        graph.add_feature_nodes(layer=8, num_features=5)

        graph.add_edge("input_0", "L8_F0", weight=0.5)
        graph.add_edge("input_0", "L8_F1", weight=0.3)
        graph.add_edge("input_0", "L8_F2", weight=0.8)

        top = graph.get_top_features(k=2)
        assert len(top) == 2
        assert top[0][0] == "L8_F2"  # Highest weight

    def test_save_and_load(self, tmp_path):
        """Test saving and loading graph."""
        graph = DependencyGraph()
        graph.add_input_nodes(["hello", "world"])
        graph.add_feature_nodes(layer=8, num_features=3)
        graph.add_edge("input_0", "L8_F1", weight=0.5)

        save_path = str(tmp_path / "test_graph.json")
        graph.save(save_path)

        loaded = DependencyGraph.load(save_path)
        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.edges) == len(graph.edges)

    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = DependencyGraph()
        graph.add_input_nodes(["test"])
        graph.add_feature_nodes(layer=8, num_features=2)

        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 3

    def test_compute_feature_attributions(self):
        """Test computing feature attributions."""
        graph = DependencyGraph()

        activations = torch.randn(10, 64)
        features = torch.abs(torch.randn(10, 128))

        attributions = graph.compute_feature_attributions(
            activations, features, layer=8
        )

        assert attributions.shape == (10, 128)
        assert (attributions >= 0).all()

    def test_find_paths_to_logit(self):
        """Test finding paths to a logit."""
        graph = DependencyGraph()
        graph.add_input_nodes(["test"])
        graph.add_feature_nodes(layer=8, num_features=3)
        graph.add_logit_nodes([0])

        graph.add_edge("input_0", "L8_F0", weight=0.5)
        graph.add_edge("L8_F0", "logit_0", weight=0.5)

        paths = graph.find_paths_to_logit(logit_idx=0)
        assert len(paths) >= 1
        assert paths[0][0] == "input_0"
        assert paths[0][-1] == "logit_0"
