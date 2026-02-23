"""
Generate publication-quality visualizations for circuit analysis.

Creates figures suitable for thesis including:
- Attribution graph diagrams
- Feature activation heatmaps
- Intervention effect plots

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

import matplotlib
matplotlib.use("Agg") # Safe for HPC/headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def load_baseline_metrics(results_path: Path, split: str, behaviour: Optional[str] = None) -> Optional[Dict]:
    """Load baseline metrics summary (checks general and behaviour-specific paths)."""
    # 1) Check general path
    p0 = results_path / f"baseline_metrics_{split}.json"
    if p0.exists():
        with open(p0, "r") as f:
            return json.load(f)
    
    # 2) Check behaviour-specific path
    if behaviour is not None:
        p1 = results_path / behaviour / f"baseline_metrics_{split}.json"
        if p1.exists():
            with open(p1, "r") as f:
                return json.load(f)
            
    return None


def load_attribution_graph(graph_path: Path, behaviour: str, split: str) -> Optional[Dict]:
    """Load attribution graph from JSON (supports suffixes like _n80)."""
    folder = graph_path / behaviour
    if not folder.exists():
        return None

    # try exact match first
    exact = folder / f"attribution_graph_{split}.json"
    if exact.exists():
        with open(exact, "r") as f:
            return json.load(f)

    # fallback: any suffix (e.g., attribution_graph_train_n80.json)
    candidates = sorted(
        folder.glob(f"attribution_graph_{split}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  Loaded graph: {candidates[0].name}")
        with open(candidates[0], "r") as f:
            return json.load(f)

    return None


def load_importance_results(results_path: Path, behaviour: str) -> Optional[pd.DataFrame]:
    """
    Load per-layer importance CSVs saved by 07_run_interventions.py and concatenate into one DF.
    Expects files like:
      results/interventions/<behaviour>/importance/feature_importance_layer_15.csv
    """
    imp_dir = results_path / "interventions" / behaviour / "importance"
    if not imp_dir.exists():
        return None

    frames = []
    # Sort by layer number for cleanliness (extract number from filename)
    # Filename format: feature_importance_layer_15.csv
    files = sorted(imp_dir.glob("feature_importance_layer_*.csv"), 
                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1)

    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {csv_file.name}: {e}")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def visualize_importance_results(
    importance_df: pd.DataFrame,
    output_path: Path,
    behaviour: str,
    top_n: int = 20,
):
    """
    Plot:
      (A) Top-N features by abs_correlation across all layers
      (B) Per-layer summary (mean abs_correlation)
    """
    if importance_df is None or importance_df.empty:
        print("  No importance results to visualize")
        return

    df = importance_df.copy()
    
    # Handle possible column name variations
    # We need: layer, feature_idx, correlation_with_logit_diff (signed), abs_correlation
    
    # 1. Check for signed correlation
    signed_col = None
    for c in ["correlation_with_logit_diff", "correlation", "corr"]:
        if c in df.columns:
            signed_col = c
            break
            
    # 2. Check for abs correlation
    abs_col = "abs_correlation" 
    if abs_col not in df.columns:
        if "abs_corr" in df.columns:
            abs_col = "abs_corr"
        elif signed_col:
            df[abs_col] = df[signed_col].abs()
        else:
            print(f"  Importance DF missing correlation columns. Found: {df.columns}")
            return
            
    if signed_col is None:
        # If we only have abs, we can't show sign color, but can still plot
        signed_col = abs_col
        
    required = {"layer", "feature_idx"}
    if not required.issubset(df.columns):
        print(f"  Importance DF missing columns: {required - set(df.columns)}")
        return

    # Top-N overall
    top = df.sort_values(abs_col, ascending=False).head(top_n).copy()
    top["label"] = top.apply(lambda r: f"L{int(r['layer'])}_F{int(r['feature_idx'])}", axis=1)

    # Per-layer summary
    by_layer = (
        df.groupby("layer", as_index=False)
          .agg(mean_abs=(abs_col, "mean"),
               max_abs=(abs_col, "max"))
          .sort_values("layer")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (A) Top-N barh by signed corr, colored by sign
    ax = axes[0]
    # Color logic: Blue if positive, Red if negative (if signed col exists and is not abs)
    if signed_col != abs_col:
        colors = [CAMBRIDGE_BLUE if c >= 0 else CAMBRIDGE_RED for c in top[signed_col]]
    else:
        colors = [CAMBRIDGE_BLUE] * len(top)
        
    ax.barh(range(len(top)), top[signed_col], color=colors, edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(f"Top {len(top)} Features by |corr| (signed shown)\n{behaviour}")
    ax.set_xlabel(f"Correlation ({signed_col})")

    # (B) Per-layer mean/max abs corr
    ax = axes[1]
    ax.plot(by_layer["layer"], by_layer["mean_abs"], marker="o", label="Mean |corr|")
    ax.plot(by_layer["layer"], by_layer["max_abs"], marker="o", label="Max |corr|")
    ax.set_title("Importance summary by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Correlation|")
    ax.legend()

    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"importance_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_path.name}")
    return fig_path


def visualize_activation_heatmaps(
    importance_df: pd.DataFrame,
    output_path: Path,
    behaviour: str,
    metrics: Optional[List[str]] = None,
):
    """
    Heatmaps over (layer x feature_idx) for activation/correlation metrics.

    Requires columns: layer, feature_idx and each metric in `metrics`.
    Produces one heatmap per metric.
    """
    if importance_df is None or importance_df.empty:
        print("  No importance results for heatmaps")
        return

    df = importance_df.copy()

    required = {"layer", "feature_idx"}
    if not required.issubset(df.columns):
        print(f"  Heatmaps: missing columns: {required - set(df.columns)}")
        return

    if metrics is None:
        metrics = ["mean_activation", "std_activation", "activation_frequency", "abs_correlation"]

    # Ensure numeric layer/feature_idx (safe)
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["feature_idx"] = pd.to_numeric(df["feature_idx"], errors="coerce")
    df = df.dropna(subset=["layer", "feature_idx"])

    output_path.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        if m not in df.columns:
            # Try to be smart about abs_correlation if missing but correlation exists
            if m == "abs_correlation" and "abs_corr" in df.columns:
                 m = "abs_corr" # use the short name
            elif m == "abs_correlation" and "correlation_with_logit_diff" in df.columns:
                 df["abs_correlation"] = df["correlation_with_logit_diff"].abs()
            elif m == "abs_correlation" and "correlation" in df.columns:
                 df["abs_correlation"] = df["correlation"].abs()
            else:
                 print(f"  Heatmaps: metric '{m}' not found; skipping")
                 continue

        # Build matrix: rows=layer, cols=feature_idx
        mat = (
            df.pivot_table(index="layer", columns="feature_idx", values=m, aggfunc="mean")
              .sort_index()
        )

        if mat.empty:
            print(f"  Heatmaps: empty matrix for {m}; skipping")
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": m},
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(f"{behaviour}: Heatmap of {m} (layer × feature_idx)")
        ax.set_xlabel("feature_idx")
        ax.set_ylabel("layer")

        fig_path = output_path / f"heatmap_{m}_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")


def canonical_intervention_name(stem: str, behaviour: str) -> str:
    """Map messy filenames to standard keys."""
    # Safety not needed here as this is unique
    s = stem.replace("intervention_", "")
    s = s.replace(f"_{behaviour}", "")
    
    if s.startswith("ablation"): return "ablation"
    if s.startswith("patching"): return "patching"
    if s.startswith("feature_importance"): return "feature_importance"
    
    return s.strip("_")


def load_intervention_results(results_path: Path, behaviour: str) -> Dict[str, pd.DataFrame]:
    """Load intervention experiment results."""
    intervention_path = results_path / "interventions" / behaviour
    results = {}

    if not intervention_path.exists():
        return results

    # Only load intervention_*.csv to avoid loading importance files or other junk
    for csv_file in intervention_path.glob("intervention_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Use canonical name to map filenames like "intervention_ablation_grammar_agreement.csv" -> "ablation"
            key = canonical_intervention_name(csv_file.stem, behaviour)
            if key:
                results[key] = df
                # Log the type of experiment found
                if "experiment_type" in df.columns:
                    types = df["experiment_type"].unique()
                    print(f"    Loaded {key} (types: {types})")
                else:
                    print(f"    Loaded {key}")
        except Exception as e:
            print(f"    Warning: failed to read {csv_file.name}: {e}")

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
        x = (i + 1) / (len(input_nodes) + 1)
        pos[node] = (x, 0.0)

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

    # Node sizing by importance (if available)
    node_sizes = []
    for node in G.nodes():
        node_data = G.nodes[node]
        # Default size
        size = 500
        val = None
        if "score" in node_data:
            val = node_data["score"]
        elif "corr" in node_data:
            val = node_data["corr"]
            
        if val is not None:
             size = 250 + 2000 * abs(val)
        elif node_data.get("type") in ["input", "output"]:
            size = 800
        node_sizes.append(size)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # Add labels (cleaner: only top nodes if many)
    labels = {}
    
    # Simple heuristic: label inputs/outputs and top 15 features
    feature_nodes_with_score = []
    for n in feature_nodes:
        d = G.nodes[n]
        val = d.get("score", d.get("corr", 0))
        feature_nodes_with_score.append((n, abs(val)))
    
    scores = [s for _, s in feature_nodes_with_score]
    nonzero = sum(1 for s in scores if s > 0)
    
    if nonzero < max(3, int(0.1 * len(scores))):
        # Too sparse or missing scores -> don't label features to avoid misleading random top-k
        top_features = set()
    else:
        top_features = set(n for n, s in sorted(feature_nodes_with_score, key=lambda x: x[1], reverse=True)[:15])
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get("type") == "input":
            labels[node] = "Input"
        elif node_data.get("type") == "output":
            labels[node] = node.replace("output_", "")[:10]
        elif node in top_features:
            labels[node] = node
        # else: no label

    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_weight="bold",
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=CAMBRIDGE_GREEN, label='Input'),
        mpatches.Patch(color=plt.cm.Blues(0.6), label='Features (by layer)'),
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
    if not metrics:
        print("  No baseline metrics to visualize")
        return

    # Check for multi-behaviour format
    if "behaviours" not in metrics:
        # Fallback for single behaviour file
        if "accuracy" in metrics:
            print("  Single behaviour metrics found, generating summary plot...")
            acc = metrics["accuracy"]
            diff = metrics.get("mean_logprob_diff_normalized", None)
            passed = metrics.get("passed", False)
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            
            # Accuracy
            axes[0].bar([0], [acc], color=CAMBRIDGE_GREEN if passed else CAMBRIDGE_RED, edgecolor="black")
            axes[0].set_ylim(0, 1)
            axes[0].set_xticks([0])
            axes[0].set_xticklabels(["behaviour"])
            axes[0].set_title("Baseline Accuracy")
            axes[0].axhline(y=0.8, color='gray', linestyle='--')
            
            # Logit Diff
            if diff is not None:
                axes[1].bar([0], [diff], color=CAMBRIDGE_BLUE, edgecolor="black")
                axes[1].set_xticks([0])
                axes[1].set_xticklabels(["behaviour"])
                axes[1].set_title("Mean Normalized Logprob Diff")
            else:
                axes[1].axis("off")
                
            plt.tight_layout()
            output_path.mkdir(parents=True, exist_ok=True)
            fig_path = output_path / "baseline_comparison.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()
            print(f"  Saved: {fig_path.name}")
            return fig_path
            
        print("  No baseline metrics 'behaviours' key found to visualize")
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
    mean_diffs = [behaviours_data[b]["mean_logprob_diff_normalized"] for b in behaviours]

    bars = ax2.bar(range(len(behaviours)), mean_diffs, color=CAMBRIDGE_BLUE, edgecolor='black')
    ax2.axhline(y=2.0, color='gray', linestyle='--', linewidth=2, label='Min Threshold')
    ax2.set_xticks(range(len(behaviours)))
    ax2.set_xticklabels([b.replace("_", "\n") for b in behaviours], fontsize=9)
    ax2.set_ylabel("Mean Normalized Logprob Diff")
    ax2.set_title("Mean Normalized Logprob Difference by Behaviour")
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

    # Identify correlation column
    corr_col = None
    for c in ["correlation_with_logit_diff", "correlation", "corr", "abs_correlation"]:
        if c in df.columns:
            corr_col = c
            break
            
    if corr_col is None:
        print("  feature_importance: no correlation column found; skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top features by correlation
    ax1 = axes[0]
    top_features = df.head(15)

    if corr_col == "abs_correlation":
        colors = [CAMBRIDGE_BLUE] * len(top_features)
    else:
        colors = [CAMBRIDGE_BLUE if c > 0 else CAMBRIDGE_RED
                  for c in top_features[corr_col]]

    ax1.barh(range(len(top_features)),
             top_features[corr_col],
             color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f"L{int(row['layer'])}_F{int(row['feature_idx'])}"
                         for _, row in top_features.iterrows()], fontsize=9)
    ax1.set_xlabel(f"Correlation ({corr_col})")
    ax1.set_title(f"Top Features by Correlation\n({behaviour})")
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.invert_yaxis()

    # Activation stats (if available)
    ax2 = axes[1]
    
    if not {"mean_activation", "std_activation"}.issubset(df.columns):
        ax2.axis("off")
        ax2.set_title("Missing mean_activation/std_activation")
    else:
        top_5 = df.head(5)
        x = np.arange(len(top_5))
        width = 0.35

        ax2.bar(x - width/2, top_5["mean_activation"], width, label='Mean Act', color=CAMBRIDGE_PURPLE)
        ax2.bar(x + width/2, top_5["std_activation"], width, label='Std Act', color=CAMBRIDGE_GREEN)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L{int(row['layer'])}_F{int(row['feature_idx'])}"
                             for _, row in top_5.iterrows()], rotation=45, ha='right')
        ax2.set_ylabel("Activation")
        ax2.set_title("Activation Stats (Top 5)")
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

    if "effect_size" not in df.columns:
        if "abs_effect_size" in df.columns:
            df = df.copy()
            df["effect_size"] = df["abs_effect_size"]
        else:
            print("  Patching file missing effect_size; skipping patching plots")
            return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Effect size distribution
    ax1 = axes[0]
    ax1.hist(df["effect_size"], bins=20, color=CAMBRIDGE_BLUE,
             edgecolor='black', alpha=0.7)
    ax1.axvline(x=df["effect_size"].mean(), color=CAMBRIDGE_RED,
                linestyle='--', linewidth=2,
                label=f'Mean: {df["effect_size"].mean():.3f}')
    ax1.set_xlabel("Effect Size")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Patching Effect Size Distribution\n({behaviour})")
    ax1.legend()

    # Baseline vs intervened logit diff
    ax2 = axes[1]
    if "baseline_logit_diff" in df.columns and "intervened_logit_diff" in df.columns:
        ax2.scatter(df["baseline_logit_diff"], df["intervened_logit_diff"],
                    c=df["effect_size"], cmap="viridis",
                    s=50, alpha=0.7, edgecolor='black')

        # Add diagonal (no effect line)
        min_val = min(df["baseline_logit_diff"].min(), df["intervened_logit_diff"].min())
        max_val = max(df["baseline_logit_diff"].max(), df["intervened_logit_diff"].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No effect')

        ax2.set_xlabel("Baseline Logit Diff")
        ax2.set_ylabel("Intervened Logit Diff")
        ax2.set_title("Patching Effect on Logit Difference")
        ax2.legend()
        
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label("Effect Size")
    else:
        ax2.axis("off")
        ax2.set_title("Missing baseline/intervened logit diff")

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"patching_results_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"  Saved: {fig_path.name}")
    return fig_path


def visualize_layer_profile(graph_data: Dict, output_path: Path, behaviour: str):
    """Plot layer-wise circuit statistics."""
    if not graph_data or not graph_data.get("nodes"):
        return

    # Extract node data
    nodes = pd.DataFrame(graph_data["nodes"])
    
    # Determine importance column
    importance_col = None
    if "score" in nodes.columns:
        importance_col = "score"
    elif "corr" in nodes.columns:
        importance_col = "corr"
        
    if "layer" not in nodes.columns or importance_col is None:
        return

    features = nodes[nodes["type"] == "feature"]
    if features.empty:
        return

    g = features.groupby("layer").agg(
        n_features=("id", "count"),
        total_abs_importance=(importance_col, lambda x: np.sum(np.abs(x))),
        max_score=(importance_col, lambda x: np.max(np.abs(x))),
        mean_score=(importance_col, lambda x: np.mean(np.abs(x)))
    ).reset_index().sort_values("layer")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = CAMBRIDGE_BLUE
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Features', color=color)
    ax1.bar(g["layer"], g["n_features"], color=color, alpha=0.6, label="Count")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = CAMBRIDGE_RED
    ax2.set_ylabel(f'Max |{importance_col}|', color=color)
    ax2.plot(g["layer"], g["max_score"], color=color, marker="o", linewidth=2, label="Max Score")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Circuit Layer Profile ({behaviour})")
    fig.tight_layout()
    
    fig_path = output_path / f"layer_profile_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")


def visualize_prompt_stability(intervention_results: Dict[str, pd.DataFrame], output_path: Path, behaviour: str):
    """Boxplot of effect variability across prompts/layers."""
    if "patching" not in intervention_results and "ablation" not in intervention_results:
        return
        
    # Use ablation if available, else patching
    key = "ablation" if "ablation" in intervention_results else "patching"
    df = intervention_results[key]
    
    # Ensure abs_effect_size exists
    if "abs_effect_size" not in df.columns:
        if "effect_size" in df.columns:
            df = df.copy()
            df["abs_effect_size"] = df["effect_size"].abs()
        else:
            return
            
    if "prompt_idx" not in df.columns:
        return

    # Aggregate by prompt (mean abs effect across layers)
    prompt_stats = df.groupby("prompt_idx")["abs_effect_size"].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Overall prompt stability
    sns.violinplot(data=prompt_stats, x="abs_effect_size", ax=axes[0], color=CAMBRIDGE_BLUE)
    axes[0].set_xlabel("Mean |Effect Size| (averaged over layers)")
    axes[0].set_title(f"Prompt Stability Distribution ({behaviour})")
    
    # 2. Layer-wise stability
    # Check if 'layer' column exists (it should if derived from standard results)
    if "layer" in df.columns:
        sns.violinplot(data=df, x="layer", y="abs_effect_size", ax=axes[1], color=CAMBRIDGE_BLUE, cut=0)
        axes[1].set_ylabel("|Effect Size|")
        axes[1].set_title("Stability by Layer")
    
    plt.tight_layout()
    fig_path = output_path / f"prompt_stability_{behaviour}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")


def visualize_freq_vs_importance(graph_data: Dict, output_path: Path, behaviour: str):
    """Scatter of Feature Frequency vs Importance Score."""
    # Assuming we can link freq data, but for now let's just use score distribution
    # If we had freq in node data, we'd use it. 
    # Let's check if 'count' or 'freq' is in node attributes (it might be in graph_data nodes)
    pass # Skipped for now as we don't strictly have freq loaded in graph json


def create_summary_figure(
    config: Dict,
    results_path: Path,
    output_path: Path,
    split: str,
    behaviour: Optional[str] = None
):
    """
    Create a summary figure suitable for thesis.
    """
    # Load all available metrics
    metrics = load_baseline_metrics(results_path, split, behaviour)

    if not metrics:
        print("  No metrics available for summary figure")
        return

    if "behaviours" not in metrics and "accuracy" in metrics:
        metrics = {"behaviours": {behaviour or "behaviour": metrics}}

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
        mean_diffs = [metrics["behaviours"][b]["mean_logprob_diff_normalized"] for b in behaviours]
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
            summary_text += f"  Diff: {m['mean_logprob_diff_normalized']:.2f}\n\n"

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


def visualize_steering_results(intervention_results: Dict[str, pd.DataFrame],
                               output_path: Path, behaviour: str):
    if "steering" not in intervention_results:
        return
    df = intervention_results["steering"]
    if df.empty:
        return

    print(f"Visualizing steering results ({len(df)} rows)...")
    # ensure effect_size
    if "effect_size" not in df.columns:
        if "abs_effect_size" in df.columns:
            df = df.copy()
            df["effect_size"] = df["abs_effect_size"]
        else:
            print("  Steering missing effect_size; skipping")
            return

    # Layer summary
    if "layer" in df.columns:
        g = df.groupby("layer").agg(
            mean_abs=("effect_size", lambda x: np.mean(np.abs(x))),
            mean_signed=("effect_size", "mean"),
            std=("effect_size", "std"),
        ).reset_index().sort_values("layer")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(g["layer"], g["mean_abs"], marker="o", linewidth=2, label="Mean |effect|")
        ax.errorbar(g["layer"], g["mean_signed"], yerr=g["std"], fmt='o', linestyle="--", alpha=0.6, label="Mean signed ± std")
        ax.set_title(f"Steering impact by layer ({behaviour})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effect size")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / f"steering_layer_effect_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")

    # Scatter baseline vs intervened
    if "baseline_logit_diff" in df.columns and "intervened_logit_diff" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df["baseline_logit_diff"], df["intervened_logit_diff"], alpha=0.5, s=12)
        
        try:
            lo = min(df["baseline_logit_diff"].min(), df["intervened_logit_diff"].min())
            hi = max(df["baseline_logit_diff"].max(), df["intervened_logit_diff"].max())
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        except ValueError:
            pass # handle empty or nan cases gracefully

        ax.set_xlabel("Baseline logit diff")
        ax.set_ylabel("Steered logit diff")
        ax.set_title(f"Steering: baseline vs steered ({behaviour})")
        ax.grid(True, alpha=0.3)
        
        fig_path = output_path / f"steering_logits_scatter_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")


def visualize_ablation_results(intervention_results: Dict[str, pd.DataFrame],
                               output_path: Path, behaviour: str):
    if "ablation" not in intervention_results:
        return
    df = intervention_results["ablation"]
    if df.empty:
        return

    # Ensure abs_effect_size exists
    if "abs_effect_size" not in df.columns:
        if "effect_size" in df.columns:
            df = df.copy()
            df["abs_effect_size"] = df["effect_size"].abs()
        else:
            print("  Ablation missing effect_size/abs_effect_size; skipping")
            return

    # Ensure effect_size exists (for histogram)
    # Layer-wise mean |effect|
    if "layer" not in df.columns:
        print("  Ablation missing 'layer'; skipping layer-wise plots")
    else:
        # Sign check
        if "sign_flipped" not in df.columns:
            df = df.copy()
            df["sign_flipped"] = np.nan
        
        g = df.groupby("layer").agg(
            mean_abs=("abs_effect_size", "mean"),
            mean_signed=("effect_size", "mean"),
            std=("effect_size", "std")
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean absolute effect
        ax.plot(g["layer"], g["mean_abs"], color=CAMBRIDGE_RED, linewidth=2, marker='o', label="Mean |Effect|")
        
        # Plot signed effect with error bars
        ax.errorbar(g["layer"], g["mean_signed"], yerr=g["std"], 
                    color=CAMBRIDGE_BLUE, linestyle='--', alpha=0.6, label="Mean Signed Effect")
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effect Size")
        ax.set_title(f"Ablation Impact by Layer ({behaviour})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = output_path / f"ablation_layer_effect_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")

    # Histogram of effect sizes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter finite values for histogram
    vals = df["effect_size"].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(vals) == 0:
        print("  Ablation: no finite effect_size values; skipping hist")
        plt.close()
    else:
        ax.hist(vals, bins=30, color=CAMBRIDGE_PURPLE, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=vals.mean(), color=CAMBRIDGE_RED, linestyle='--', linewidth=2, 
                   label=f"Mean: {vals.mean():.3f}")
        
        ax.set_xlabel("Effect Size (Logit Diff Change)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Ablation Effect Size Distribution ({behaviour})")
        ax.legend()
    
        fig_path = output_path / f"ablation_effect_hist_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")
        
    # Scatter: Baseline vs Intervened Logits (Validation)
    if "baseline_logit_diff" in df.columns and "intervened_logit_diff" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df["baseline_logit_diff"], df["intervened_logit_diff"], alpha=0.5, s=10)
        
        # diagonal
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        
        ax.plot([lo, hi], [lo, hi], 'k-', alpha=0.75, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        
        ax.set_xlabel("Baseline Logit Diff")
        ax.set_ylabel("Ablated Logit Diff")
        ax.set_title(f"Ablation Impact ({behaviour})")
        ax.grid(True, alpha=0.3)
        
        fig_path = output_path / f"ablation_logits_scatter_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")
        
    # Top Prompts by Impact
    if "prompt_idx" in df.columns:
        top_prompts = df.groupby("prompt_idx")["abs_effect_size"].mean().sort_values(ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot barh
        y_pos = range(len(top_prompts))
        ax.barh(y_pos, top_prompts.values, color=CAMBRIDGE_RED, alpha=0.7, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i}" for i in top_prompts.index])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Mean |Effect Size|")
        ax.set_ylabel("prompt_idx")
        ax.set_title(f"Top 15 Most Sensitive Prompts ({behaviour})")
        
        fig_path = output_path / f"ablation_top_prompts_{behaviour}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fig_path.name}")


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
        choices=["grammar_agreement", "physics_scalar_vector_operator"],
        default="grammar_agreement",
        help="Which behaviour to visualize",
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
    metrics = load_baseline_metrics(results_path, args.split, args.behaviour)
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
            visualize_layer_profile(graph_data, figures_path, behaviour)
        else:
            print("  No attribution graph found")

        # Intervention results
        # Load intervention results
        intervention_results = load_intervention_results(results_path, behaviour)
        
        # Importance results (per-layer CSVs)
        print("  Loading importance results...")
        importance_df = load_importance_results(results_path, behaviour)
        if importance_df is not None:
             visualize_importance_results(importance_df, figures_path, behaviour, top_n=20)
             visualize_activation_heatmaps(importance_df, figures_path, behaviour)
        else:
             print("  No importance results found")
        
        if intervention_results:
            visualize_patching_results(intervention_results, figures_path, behaviour)
            visualize_ablation_results(intervention_results, figures_path, behaviour)
            visualize_steering_results(intervention_results, figures_path, behaviour)
            visualize_prompt_stability(intervention_results, figures_path, behaviour)

        else:
            print("  No intervention results found")

    # Create thesis summary figure
    print("\n[3] Creating thesis summary figure...")
    create_summary_figure(config, results_path, figures_path, args.split, args.behaviour)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {figures_path.absolute()}")
    print("\nGenerated figures can be used directly in thesis.")


if __name__ == "__main__":
    main()
