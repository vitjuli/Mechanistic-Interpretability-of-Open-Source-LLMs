"""
Generate publication-quality visualizations for circuit analysis.

Creates figures suitable for thesis including:
- Attribution graph diagrams
- Feature activation heatmaps
- Intervention effect plots
- Comparison with Anthropic's results

Usage:
    python scripts/08_generate_figures.py
    python scripts/08_generate_figures.py --split test
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Cambridge blue color scheme
CAMBRIDGE_BLUE = '#0072B2'
CAMBRIDGE_RED = '#D55E00'
CAMBRIDGE_GREEN = '#009E73'
CAMBRIDGE_PURPLE = '#7570B3'


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_baseline_metrics(results_path: Path, split: str) -> Optional[Dict]:
    """Load baseline metrics summary."""
    metrics_file = results_path / f"baseline_metrics_{split}.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None


def load_attribution_graph(graph_path: Path, behaviour: str, split: str) -> Optional[Dict]:
    """Load attribution graph from JSON."""
    json_file = graph_path / behaviour / f"attribution_graph_{split}.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            return json.load(f)
    return None


def load_intervention_results(results_path: Path, behaviour: str) -> Dict[str, pd.DataFrame]:
    """Load intervention experiment results."""
    intervention_path = results_path / "interventions" / behaviour
    results = {}

    if intervention_path.exists():
        for csv_file in intervention_path.glob("*.csv"):
            name = csv_file.stem.replace(f"intervention_", "").replace(f"_{behaviour}", "")
            results[name] = pd.read_csv(csv_file)

    return results


def visualize_attribution_graph(
    graph_data: Dict,
    output_path: Path,
    behaviour: str,
    title: str = None,
):
    """
    Create a visual representation of the attribution graph.

    Mimics the style from Anthropic's Biology paper with:
    - Nodes colored by layer
    - Edge thickness proportional to attribution
    - Hierarchical layout
    """
    if not graph_data or not graph_data.get("nodes"):
        print(f"  No graph data for {behaviour}")
        return

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in graph_data["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, **node)

    # Add edges
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

    # Separate nodes by type
    input_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "input"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "feature"]
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "output"]

    if not feature_nodes:
        print(f"  No feature nodes found for {behaviour}")
        return

    # Group features by layer
    layers = {}
    for node in feature_nodes:
        layer = G.nodes[node].get("layer", 0)
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Position nodes in hierarchical layout
    pos = {}

    # Input nodes at bottom
    for i, node in enumerate(input_nodes):
        pos[node] = (0.5, 0)

    # Feature nodes in middle (one row per layer)
    sorted_layers = sorted(layers.keys())
    n_layers = len(sorted_layers)

    for layer_idx, layer in enumerate(sorted_layers):
        layer_nodes = layers[layer]
        n_nodes = len(layer_nodes)
        y = (layer_idx + 1) / (n_layers + 2)

        for i, node in enumerate(layer_nodes):
            x = (i + 1) / (n_nodes + 1)
            pos[node] = (x, y)

    # Output nodes at top
    for i, node in enumerate(output_nodes):
        x = (i + 1) / (len(output_nodes) + 1)
        pos[node] = (x, 1.0)

    # Color nodes by type/layer
    colors = []
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get("type") == "input":
            colors.append(CAMBRIDGE_GREEN)
        elif node_data.get("type") == "output":
            colors.append(CAMBRIDGE_RED)
        else:
            # Color by layer
            layer = node_data.get("layer", 0)
            layer_idx = sorted_layers.index(layer) if layer in sorted_layers else 0
            cmap = plt.cm.Blues
            colors.append(cmap(0.3 + 0.7 * layer_idx / max(n_layers - 1, 1)))

    # Draw edges with thickness proportional to weight
    edge_weights = [G.edges[e].get("weight", 1.0) for e in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [2 * w / max_weight for w in edge_weights]
    else:
        edge_widths = [1] * len(G.edges())

    # Draw the graph
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="gray",
        width=edge_widths,
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors,
        node_size=500,
        alpha=0.9,
    )

    # Add labels
    labels = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get("type") == "input":
            labels[node] = "Input"
        elif node_data.get("type") == "output":
            labels[node] = node.replace("output_", "")[:10]
        else:
            labels[node] = node

    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_weight="bold",
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=CAMBRIDGE_GREEN, label='Input'),
        mpatches.Patch(color=CAMBRIDGE_BLUE, label='SAE Features'),
        mpatches.Patch(color=CAMBRIDGE_RED, label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Title and styling
    ax.set_title(title or f"Attribution Graph: {behaviour}")
    ax.axis("off")

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"attribution_graph_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def visualize_baseline_comparison(
    metrics: Dict,
    output_path: Path,
):
    """
    Create comparison visualization of all behaviours.
    """
    if not metrics or "behaviours" not in metrics:
        print("  No baseline metrics to visualize")
        return

    behaviours_data = metrics["behaviours"]
    behaviours = list(behaviours_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax1 = axes[0]
    accuracies = [behaviours_data[b]["accuracy"] for b in behaviours]
    colors = [CAMBRIDGE_GREEN if behaviours_data[b]["passed"] else CAMBRIDGE_RED
              for b in behaviours]

    bars = ax1.bar(range(len(behaviours)), accuracies, color=colors, edgecolor='black')
    ax1.axhline(y=0.8, color='gray', linestyle='--', linewidth=2, label='Threshold (80%)')
    ax1.set_xticks(range(len(behaviours)))
    ax1.set_xticklabels([b.replace("_", "\n") for b in behaviours], fontsize=9)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Baseline Accuracy by Behaviour")
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.0%}', ha='center', va='bottom', fontsize=9)

    # Logit difference comparison
    ax2 = axes[1]
    mean_diffs = [behaviours_data[b]["mean_logit_diff"] for b in behaviours]

    bars = ax2.bar(range(len(behaviours)), mean_diffs, color=CAMBRIDGE_BLUE, edgecolor='black')
    ax2.axhline(y=2.0, color='gray', linestyle='--', linewidth=2, label='Min Threshold')
    ax2.set_xticks(range(len(behaviours)))
    ax2.set_xticklabels([b.replace("_", "\n") for b in behaviours], fontsize=9)
    ax2.set_ylabel("Mean Logit Difference")
    ax2.set_title("Mean Logit Difference by Behaviour")
    ax2.legend()

    for bar, diff in zip(bars, mean_diffs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{diff:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "baseline_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def visualize_feature_importance(
    intervention_results: Dict[str, pd.DataFrame],
    output_path: Path,
    behaviour: str,
):
    """
    Create feature importance visualization from intervention experiments.
    """
    if "feature_importance" not in intervention_results:
        return

    df = intervention_results["feature_importance"]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top features by correlation
    ax1 = axes[0]
    top_features = df.head(15)

    colors = [CAMBRIDGE_BLUE if c > 0 else CAMBRIDGE_RED
              for c in top_features["correlation_with_logit_diff"]]

    ax1.barh(range(len(top_features)),
             top_features["correlation_with_logit_diff"],
             color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f"L{int(row['layer'])}_F{int(row['feature_idx'])}"
                         for _, row in top_features.iterrows()], fontsize=9)
    ax1.set_xlabel("Correlation with Logit Difference")
    ax1.set_title(f"Top Features by Correlation\n({behaviour})")
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.invert_yaxis()

    # Activation distribution for top features
    ax2 = axes[1]
    top_5 = df.head(5)
    x = range(len(top_5))
    width = 0.35

    ax2.bar([i - width/2 for i in x], top_5["mean_activation"], width,
            label='Mean', color=CAMBRIDGE_BLUE, edgecolor='black')
    ax2.bar([i + width/2 for i in x], top_5["std_activation"], width,
            label='Std', color=CAMBRIDGE_PURPLE, edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{int(row['layer'])}_F{int(row['feature_idx'])}"
                         for _, row in top_5.iterrows()], fontsize=9)
    ax2.set_ylabel("Activation")
    ax2.set_title("Activation Statistics (Top 5 Features)")
    ax2.legend()

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"feature_importance_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def visualize_patching_results(
    intervention_results: Dict[str, pd.DataFrame],
    output_path: Path,
    behaviour: str,
):
    """
    Create visualization of counterfactual patching results.
    """
    if "patching" not in intervention_results:
        return

    df = intervention_results["patching"]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Feature similarity distribution
    ax1 = axes[0]
    ax1.hist(df["feature_cosine_similarity"], bins=20, color=CAMBRIDGE_BLUE,
             edgecolor='black', alpha=0.7)
    ax1.axvline(x=df["feature_cosine_similarity"].mean(), color=CAMBRIDGE_RED,
                linestyle='--', linewidth=2,
                label=f'Mean: {df["feature_cosine_similarity"].mean():.3f}')
    ax1.set_xlabel("Feature Cosine Similarity")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Feature Similarity Between Prompt Pairs\n({behaviour})")
    ax1.legend()

    # Baseline logit diff comparison
    ax2 = axes[1]
    ax2.scatter(df["baseline_diff_a"], df["baseline_diff_b"],
                c=df["feature_cosine_similarity"], cmap="viridis",
                s=50, alpha=0.7, edgecolor='black')

    # Add diagonal
    min_val = min(df["baseline_diff_a"].min(), df["baseline_diff_b"].min())
    max_val = max(df["baseline_diff_a"].max(), df["baseline_diff_b"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    ax2.set_xlabel("Baseline Logit Diff (Prompt A)")
    ax2.set_ylabel("Baseline Logit Diff (Prompt B)")
    ax2.set_title("Logit Difference Comparison")

    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("Feature Similarity")

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"patching_results_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def create_summary_figure(
    config: Dict,
    results_path: Path,
    output_path: Path,
):
    """
    Create a summary figure suitable for thesis.
    """
    # Load all available metrics
    metrics = load_baseline_metrics(results_path, "train")

    if not metrics:
        print("  No metrics available for summary figure")
        return

    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        f"Mechanistic Interpretability Analysis\n"
        f"Model: {config['model']['name']}",
        fontsize=14, fontweight='bold'
    )

    # Panel 1: Accuracy by behaviour
    ax1 = fig.add_subplot(gs[0, 0])
    if "behaviours" in metrics:
        behaviours = list(metrics["behaviours"].keys())
        accuracies = [metrics["behaviours"][b]["accuracy"] for b in behaviours]
        colors = [CAMBRIDGE_GREEN if metrics["behaviours"][b]["passed"] else CAMBRIDGE_RED
                  for b in behaviours]
        ax1.bar(range(len(behaviours)), accuracies, color=colors, edgecolor='black')
        ax1.axhline(y=0.8, color='gray', linestyle='--')
        ax1.set_xticks(range(len(behaviours)))
        ax1.set_xticklabels([b.replace("_", "\n")[:15] for b in behaviours], fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("A) Baseline Accuracy")

    # Panel 2: Logit differences
    ax2 = fig.add_subplot(gs[0, 1])
    if "behaviours" in metrics:
        mean_diffs = [metrics["behaviours"][b]["mean_logit_diff"] for b in behaviours]
        ax2.bar(range(len(behaviours)), mean_diffs, color=CAMBRIDGE_BLUE, edgecolor='black')
        ax2.set_xticks(range(len(behaviours)))
        ax2.set_xticklabels([b.replace("_", "\n")[:15] for b in behaviours], fontsize=8)
        ax2.set_ylabel("Mean Logit Diff")
        ax2.set_title("B) Logit Differences")

    # Panel 3: Summary statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    if "behaviours" in metrics:
        for b in behaviours:
            m = metrics["behaviours"][b]
            status = "PASS" if m["passed"] else "FAIL"
            summary_text += f"{b}:\n"
            summary_text += f"  Acc: {m['accuracy']:.1%} [{status}]\n"
            summary_text += f"  Diff: {m['mean_logit_diff']:.2f}\n\n"

    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 4-6: Placeholder for circuit diagrams (would load from saved graphs)
    for i, (gs_pos, title) in enumerate([
        (gs[1, 0], "C) Circuit: Grammar"),
        (gs[1, 1], "D) Circuit: Factual"),
        (gs[1, 2], "E) Circuit: Sentiment"),
    ]):
        ax = fig.add_subplot(gs_pos)
        ax.text(0.5, 0.5, "Circuit Diagram\n(See individual figures)",
                ha='center', va='center', fontsize=10, style='italic',
                transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "thesis_summary_figure.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Generate circuit visualizations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        choices=["grammar_agreement"],
        default="grammar_agreement",
        help="Which behaviour to visualize (currently only grammar_agreement)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    print("=" * 70)
    print("CIRCUIT VISUALIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Split: {args.split}")
    print(f"  Behaviours: {', '.join(behaviours) if behaviours else 'all (summary only)'}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Setup paths
    results_path = Path(config["paths"]["results"])
    figures_path = Path(config["paths"]["figures"])
    graph_path = results_path / "attribution_graphs"

    # Create baseline comparison
    print("\n[1] Creating baseline comparison...")
    metrics = load_baseline_metrics(results_path, args.split)
    if metrics:
        visualize_baseline_comparison(metrics, figures_path)
    else:
        print("  No baseline metrics found")

    # Visualize each behaviour
    for behaviour in behaviours:
        print(f"\n[2] Visualizing {behaviour}...")

        # Attribution graph
        print("  Loading attribution graph...")
        graph_data = load_attribution_graph(graph_path, behaviour, args.split)
        if graph_data:
            visualize_attribution_graph(graph_data, figures_path, behaviour)
        else:
            print("  No attribution graph found")

        # Intervention results
        print("  Loading intervention results...")
        intervention_results = load_intervention_results(results_path, behaviour)

        if intervention_results:
            visualize_feature_importance(intervention_results, figures_path, behaviour)
            visualize_patching_results(intervention_results, figures_path, behaviour)
        else:
            print("  No intervention results found")

    # Create thesis summary figure
    print("\n[3] Creating thesis summary figure...")
    create_summary_figure(config, results_path, figures_path)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {figures_path.absolute()}")
    print("\nGenerated figures can be used directly in thesis.")


if __name__ == "__main__":
    main()
