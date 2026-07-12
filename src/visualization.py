"""
Visualization utilities for circuit analysis.

Provides functions for:
- Attribution graph rendering
- Feature activation heatmaps
- Intervention effect plots
- Publication-quality figure generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import networkx as nx

# Cambridge colour palette
PALETTE = {
    "blue": "#0072B2",
    "red": "#D55E00",
    "green": "#009E73",
    "purple": "#7570B3",
    "orange": "#E69F00",
    "grey": "#999999",
}

# Publication defaults
def set_publication_style():
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def plot_attribution_graph(
    graph: nx.DiGraph,
    title: str = "Attribution Graph",
    figsize: Tuple[int, int] = (14, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Render an attribution graph as a hierarchical diagram.

    Nodes are coloured by type (input/feature/output) and positioned
    hierarchically by layer.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Classify nodes
    input_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "input"]
    feature_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "feature"]
    output_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "output"]

    # Group features by layer
    layers = {}
    for node in feature_nodes:
        layer = graph.nodes[node].get("layer", 0)
        layers.setdefault(layer, []).append(node)

    sorted_layers = sorted(layers.keys())
    n_levels = len(sorted_layers) + 2  # +2 for input and output

    # Position nodes
    pos = {}
    for i, node in enumerate(input_nodes):
        pos[node] = ((i + 1) / (len(input_nodes) + 1), 0)

    for level, layer in enumerate(sorted_layers):
        nodes = layers[layer]
        y = (level + 1) / n_levels
        for i, node in enumerate(nodes):
            pos[node] = ((i + 1) / (len(nodes) + 1), y)

    for i, node in enumerate(output_nodes):
        pos[node] = ((i + 1) / (len(output_nodes) + 1), 1.0)

    # Colours
    node_colors = []
    for node in graph.nodes():
        t = graph.nodes[node].get("type", "feature")
        if t == "input":
            node_colors.append(PALETTE["green"])
        elif t == "output":
            node_colors.append(PALETTE["red"])
        else:
            node_colors.append(PALETTE["blue"])

    # Edge widths
    weights = [abs(graph.edges[e].get("weight", 1.0)) for e in graph.edges()]
    if weights:
        max_w = max(weights) or 1
        edge_widths = [2.0 * w / max_w for w in weights]
    else:
        edge_widths = []

    # Draw
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths,
                           edge_color="grey", alpha=0.5, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                           node_size=400, alpha=0.9)

    labels = {}
    for n in graph.nodes():
        d = graph.nodes[n]
        if d.get("type") == "input":
            labels[n] = "Input"
        elif d.get("type") == "output":
            labels[n] = d.get("token", n)[:12]
        else:
            labels[n] = n
    nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=7)

    legend = [
        mpatches.Patch(color=PALETTE["green"], label="Input"),
        mpatches.Patch(color=PALETTE["blue"], label="SAE Feature"),
        mpatches.Patch(color=PALETTE["red"], label="Output"),
    ]
    ax.legend(handles=legend, loc="upper right")
    ax.set_title(title)
    ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def plot_feature_heatmap(
    feature_matrix: np.ndarray,
    feature_labels: Optional[List[str]] = None,
    prompt_labels: Optional[List[str]] = None,
    title: str = "Feature Activations",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a heatmap of feature activations across prompts.

    Args:
        feature_matrix: (n_prompts, n_features)
        feature_labels: Labels for features
        prompt_labels: Labels for prompts
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        feature_matrix,
        ax=ax,
        cmap="viridis",
        xticklabels=feature_labels or False,
        yticklabels=prompt_labels or False,
        cbar_kws={"label": "Activation"},
    )
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Prompt")
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def plot_intervention_effects(
    baseline_diffs: np.ndarray,
    intervened_diffs: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Intervention Effects",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the effect of interventions on logit differences.

    Shows baseline vs intervened logit differences as paired comparisons.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Paired comparison
    ax1 = axes[0]
    for i in range(len(baseline_diffs)):
        ax1.plot([0, 1], [baseline_diffs[i], intervened_diffs[i]],
                 color="grey", alpha=0.3)
    ax1.scatter([0] * len(baseline_diffs), baseline_diffs,
                color=PALETTE["blue"], label="Baseline", zorder=5)
    ax1.scatter([1] * len(intervened_diffs), intervened_diffs,
                color=PALETTE["red"], label="Intervened", zorder=5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Baseline", "Intervened"])
    ax1.set_ylabel("Logit Difference")
    ax1.set_title("Paired Comparison")
    ax1.legend()

    # Effect size distribution
    ax2 = axes[1]
    effects = intervened_diffs - baseline_diffs
    ax2.hist(effects, bins=20, color=PALETTE["purple"], edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="black", linestyle="--")
    ax2.axvline(x=effects.mean(), color=PALETTE["red"], linestyle="-",
                label=f"Mean: {effects.mean():.2f}")
    ax2.set_xlabel("Change in Logit Difference")
    ax2.set_ylabel("Count")
    ax2.set_title("Effect Size Distribution")
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def plot_logit_diff_distribution(
    logit_diffs: np.ndarray,
    threshold: float = 2.0,
    behaviour_name: str = "",
    figsize: Tuple[int, int] = (8, 5),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution of logit differences with threshold line."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(logit_diffs, bins=30, color=PALETTE["blue"], edgecolor="black", alpha=0.7)
    ax.axvline(x=threshold, color=PALETTE["red"], linestyle="--", linewidth=2,
               label=f"Threshold ({threshold})")
    ax.axvline(x=np.mean(logit_diffs), color=PALETTE["green"], linestyle="-",
               linewidth=2, label=f"Mean ({np.mean(logit_diffs):.2f})")
    ax.set_xlabel("Logit Difference (correct - incorrect)")
    ax.set_ylabel("Count")
    ax.set_title(f"Logit Difference Distribution: {behaviour_name}")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def plot_sae_reconstruction_quality(
    r2_scores: Dict[int, float],
    l0_scores: Dict[int, float],
    dead_fractions: Dict[int, float],
    figsize: Tuple[int, int] = (14, 5),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot SAE reconstruction quality across layers."""
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    layers = sorted(r2_scores.keys())

    # R-squared
    ax1 = axes[0]
    ax1.bar(range(len(layers)), [r2_scores[l] for l in layers],
            color=PALETTE["blue"], edgecolor="black")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}" for l in layers], rotation=45)
    ax1.set_ylabel("R-squared")
    ax1.set_title("Reconstruction Quality")
    ax1.axhline(y=0.85, color=PALETTE["red"], linestyle="--", label="Target")
    ax1.set_ylim(0, 1)
    ax1.legend()

    # L0 sparsity
    ax2 = axes[1]
    ax2.bar(range(len(layers)), [l0_scores[l] for l in layers],
            color=PALETTE["green"], edgecolor="black")
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45)
    ax2.set_ylabel("L0 (fraction active)")
    ax2.set_title("Feature Sparsity")

    # Dead features
    ax3 = axes[2]
    ax3.bar(range(len(layers)), [dead_fractions[l] for l in layers],
            color=PALETTE["orange"], edgecolor="black")
    ax3.set_xticks(range(len(layers)))
    ax3.set_xticklabels([f"L{l}" for l in layers], rotation=45)
    ax3.set_ylabel("Dead Feature Fraction")
    ax3.set_title("Dead Features")
    ax3.axhline(y=0.2, color=PALETTE["red"], linestyle="--", label="Threshold")
    ax3.set_ylim(0, 1)
    ax3.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig
