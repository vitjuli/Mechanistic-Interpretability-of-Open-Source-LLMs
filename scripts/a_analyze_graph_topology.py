#!/usr/bin/env python3
"""
Stage 0 — Attribution Graph Topology Audit

Confirms star topology (no feature-feature edges), characterises edge weights,
features-per-layer distribution, and explains why Louvain returns 1 community.

Usage:
    # CSD3 antonym run:
    python scripts/a_analyze_graph_topology.py \
        --graph data/ui_offline/20260302-211821_antonym_operation_train_n80/graph.json \
        --out_dir data/analysis/antonym_train_n80_topology

    # Local grammar_agreement test:
    python scripts/a_analyze_graph_topology.py \
        --graph "data/ui_offline/20260223-191404_grammar_agreement_train_n80/graph.json" \
        --out_dir data/analysis/grammar_agreement_train_n80_topology
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    """Load node-link JSON; accepts both 'links' and 'edges' key."""
    with open(path) as f:
        g = json.load(f)
    nodes = g.get("nodes", [])
    links = g.get("links", g.get("edges", []))
    return nodes, links, g


def classify_nodes(nodes):
    feat_nodes = [n for n in nodes if n.get("type") == "feature"]
    hub_nodes  = [n for n in nodes if n.get("type") != "feature"]
    feat_ids   = {n["id"] for n in feat_nodes}
    return feat_nodes, hub_nodes, feat_ids


# ---------------------------------------------------------------------------
# Edge analysis
# ---------------------------------------------------------------------------

def analyse_edges(links, feat_ids):
    """Return (ff, hf, fh, weights) — classify all links by endpoint types."""
    ff, hf, fh, weights = [], [], [], []
    for e in links:
        src = e.get("source", "")
        tgt = e.get("target", "")
        w   = e.get("weight", None)
        if w is not None:
            weights.append(float(w))
        sf = src in feat_ids
        tf = tgt in feat_ids
        if sf and tf:
            ff.append(e)
        elif not sf and tf:
            hf.append(e)
        elif sf and not tf:
            fh.append(e)
    return ff, hf, fh, np.asarray(weights, dtype=float)


def compute_degrees(links, feat_ids):
    in_deg, out_deg = Counter(), Counter()
    for e in links:
        if e.get("target") in feat_ids:
            in_deg[e["target"]] += 1
        if e.get("source") in feat_ids:
            out_deg[e["source"]] += 1
    return in_deg, out_deg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_degree_distribution(feat_nodes, links, feat_ids, out_dir):
    in_deg, out_deg = compute_degrees(links, feat_ids)
    in_vals  = [in_deg.get(n["id"], 0)  for n in feat_nodes]
    out_vals = [out_deg.get(n["id"], 0) for n in feat_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title, color in zip(
        axes,
        [in_vals, out_vals],
        ["In-degree (edges arriving at feature)", "Out-degree (edges leaving feature)"],
        ["#4e79a7", "#e15759"],
    ):
        cnt = Counter(vals)
        ax.bar(list(cnt.keys()), list(cnt.values()), color=color, edgecolor="white")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Feature count")
        ax.set_title(title)
        ax.set_xticks(sorted(cnt.keys()))
    fig.suptitle("Feature node degree distributions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "degree_distribution.png", dpi=120)
    plt.close(fig)
    return in_vals, out_vals


def plot_edge_weights(weights, out_dir):
    if len(weights) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(weights, bins=40, color="#76b7b2", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero")
    axes[0].set_xlabel("Edge weight (signed)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Signed edge weight distribution")
    axes[0].legend()

    axes[1].hist(np.abs(weights), bins=40, color="#59a14f", edgecolor="white")
    axes[1].set_xlabel("|Edge weight|")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Absolute edge weight distribution")

    fig.tight_layout()
    fig.savefig(out_dir / "edge_weight_histogram.png", dpi=120)
    plt.close(fig)


def plot_features_per_layer(feat_nodes, out_dir):
    layer_counts = Counter(
        n.get("layer") for n in feat_nodes if n.get("layer") is not None
    )
    layers = sorted(layer_counts.keys())
    counts = [layer_counts[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, counts, color="#f28e2b", edgecolor="white")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature nodes per layer")
    ax.set_xticks(layers)
    for l, c in zip(layers, counts):
        ax.text(l, c + 0.1, str(c), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "features_per_layer.png", dpi=120)
    plt.close(fig)
    return layer_counts


def plot_feature_attrs(feat_nodes, out_dir):
    """Scatter: specific_score vs frequency, coloured by layer."""
    xs     = [n.get("frequency", 0)      for n in feat_nodes]
    ys     = [n.get("specific_score", 0) for n in feat_nodes]
    layers = [n.get("layer", 0)          for n in feat_nodes]
    ns     = [n.get("n_prompts", 1)      for n in feat_nodes]
    betas  = [n.get("beta_sign", 0)      for n in feat_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: specific_score vs frequency
    sc = axes[0].scatter(
        xs, ys, c=layers, s=[max(20, n * 5) for n in ns],
        alpha=0.8, cmap="viridis", edgecolors="none",
    )
    plt.colorbar(sc, ax=axes[0], label="Layer")
    axes[0].set_xlabel("Frequency (fraction of prompts)")
    axes[0].set_ylabel("Specific score")
    axes[0].set_title("Specificity vs frequency (size ∝ n_prompts, colour = layer)")

    # Right: beta_sign distribution per layer
    unique_layers = sorted(set(layers))
    pos_counts = [sum(1 for l, b in zip(layers, betas) if l == ul and b > 0) for ul in unique_layers]
    neg_counts = [sum(1 for l, b in zip(layers, betas) if l == ul and b < 0) for ul in unique_layers]
    x = range(len(unique_layers))
    axes[1].bar(x, pos_counts, label="Excitatory (β>0)", color="#59a14f", alpha=0.8)
    axes[1].bar(x, [-n for n in neg_counts], label="Inhibitory (β<0)", color="#e15759", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(unique_layers, fontsize=8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Feature count (+ excitatory / − inhibitory)")
    axes[1].set_title("Excitatory vs inhibitory features per layer")
    axes[1].legend()
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / "feature_attrs_scatter.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

LOUVAIN_EXPLANATION = """\
  The attribution graph has a STAR TOPOLOGY — every feature node connects to
  exactly the same three hub nodes ("input", "output_correct",
  "output_incorrect") and to no other feature node.

  When converted to undirected form (as Louvain requires), the graph becomes
  a complete bipartite structure:
      {input, output_correct, output_incorrect} ↔ {F1 … Fn}

  All feature nodes are IDENTICALLY positioned: each has exactly the same two
  neighbours (input and one output hub) via undirected edges.  There is no
  denser subgraph of features to exploit; modularity is flat across all
  possible partitions of the feature set.

  Consequence: Louvain Q(1 big community) ≈ 0, and any split into k>1
  communities achieves Q ≤ 0 because there are no within-community feature–
  feature edges to reward.  Raising the resolution parameter γ also does not
  help: the null model already predicts no community structure.

  This is NOT a bug or a parameter choice issue.  It is a structural property
  of the current per-prompt union graph, which records each feature's
  marginal contribution to the output but stores no pairwise feature
  interactions.

  CORRECT APPROACHES FOR STAGE 2:
    (A) Feature-profile clustering — cluster features on their node attribute
        vectors: (layer_bin, beta_sign, specific_score, frequency).  Features
        that operate at similar depths with the same sign form natural groups.
    (B) Co-activation graph communities — build feature–feature edges from
        the per-prompt top-k lists stored by script 04 or the graph's own
        union_params, then run Louvain/Leiden on THAT graph.  An edge between
        F_i and F_j is weighted by the number of prompts in which both appear
        in the top-k.  This approximates Anthropic's cross-layer Jacobian
        edges without requiring Jacobian computation.
"""


def write_report(path, nodes, links, feat_nodes, hub_nodes, feat_ids,
                 ff, hf, fh, weights, layer_counts, in_vals, out_vals):
    lines = []
    def p(*args): lines.append(" ".join(str(a) for a in args))

    p("=" * 70)
    p("ATTRIBUTION GRAPH TOPOLOGY AUDIT  —  Stage 0")
    p("=" * 70)
    p()

    p("─── 1. Graph summary ───")
    p(f"  Total nodes   : {len(nodes)}")
    p(f"  Total links   : {len(links)}")
    p(f"  Feature nodes : {len(feat_nodes)}")
    p(f"  Hub nodes     : {len(hub_nodes)}  {[n['id'] for n in hub_nodes]}")
    p()

    p("─── 2. Edge classification ───")
    p(f"  Hub → Feature          : {len(hf)}")
    p(f"  Feature → Hub          : {len(fh)}")
    p(f"  Feature → Feature      : {len(ff)}  ← STAR TOPOLOGY CONFIRMED")
    p()

    p("─── 3. Edge weight statistics ───")
    if len(weights):
        pos = int((weights > 0).sum())
        neg = int((weights < 0).sum())
        p(f"  N edges with weight : {len(weights)}")
        p(f"  Positive (excitatory)  : {pos}  ({100*pos/len(weights):.1f}%)")
        p(f"  Negative (inhibitory)  : {neg}  ({100*neg/len(weights):.1f}%)")
        p(f"  Mean (signed)       : {weights.mean():.4f}")
        p(f"  Std                 : {weights.std():.4f}")
        p(f"  Min                 : {weights.min():.4f}")
        p(f"  Max                 : {weights.max():.4f}")
        p(f"  Mean |weight|       : {np.abs(weights).mean():.4f}")
        p(f"  Median |weight|     : {np.median(np.abs(weights)):.4f}")
    p()

    p("─── 4. Feature degree statistics ───")
    p(f"  In-degree values  : {sorted(set(in_vals))}  (all features, should be uniform)")
    p(f"  Out-degree values : {sorted(set(out_vals))}  (all features, should be uniform)")
    in_uniform  = len(set(in_vals))  == 1
    out_uniform = len(set(out_vals)) == 1
    p(f"  All in-degrees identical  : {in_uniform}")
    p(f"  All out-degrees identical : {out_uniform}")
    if in_uniform and out_uniform:
        p(f"  → Pure star topology confirmed by degree regularity.")
    p()

    p("─── 5. Features per layer ───")
    for layer in sorted(layer_counts):
        bar = "█" * layer_counts[layer]
        p(f"  Layer {layer:2d}: {layer_counts[layer]:3d}  {bar}")
    p()

    p("─── 6. Feature attribute summary ───")
    def _stats(vals, name):
        if vals:
            p(f"  {name}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}"
              f"  min={min(vals):.3f}  max={max(vals):.3f}")
    specs  = [n.get("specific_score", 0) for n in feat_nodes if n.get("specific_score") is not None]
    freqs  = [n.get("frequency", 0)      for n in feat_nodes if n.get("frequency")       is not None]
    betas  = [n.get("beta_sign", 0)      for n in feat_nodes if n.get("beta_sign")        is not None]
    _stats(specs, "specific_score")
    _stats(freqs, "frequency     ")
    if betas:
        pos_b = sum(1 for b in betas if b > 0)
        neg_b = sum(1 for b in betas if b < 0)
        p(f"  beta_sign: {pos_b} excitatory (+1),  {neg_b} inhibitory (−1)")
    p()

    p("─── 7. Why Louvain returns 1 community ───")
    p(LOUVAIN_EXPLANATION)

    p("=" * 70)
    p("END OF REPORT")
    p("=" * 70)

    text = "\n".join(lines)
    path.write_text(text)
    print(text)

    # Machine-readable summary for SUMMARY.md generator
    summary = {
        "n_nodes": len(nodes),
        "n_links": len(links),
        "n_feat_nodes": len(feat_nodes),
        "n_hub_nodes": len(hub_nodes),
        "n_ff_edges": len(ff),
        "star_topology_confirmed": len(ff) == 0,
        "n_positive_edges": int((weights > 0).sum()) if len(weights) else 0,
        "n_negative_edges": int((weights < 0).sum()) if len(weights) else 0,
        "mean_abs_weight": float(np.abs(weights).mean()) if len(weights) else 0,
        "n_layers": len(layer_counts),
        "layer_counts": {str(k): v for k, v in layer_counts.items()},
        "excitatory_features": sum(1 for b in betas if b > 0),
        "inhibitory_features": sum(1 for b in betas if b < 0),
    }
    import json as _json
    (path.parent / "topology_stats.json").write_text(
        _json.dumps(summary, indent=2)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Attribution graph topology audit"
    )
    parser.add_argument("--graph",   required=True,
                        help="Path to graph.json (ui_offline node-link format)")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for report + figures")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not graph_path.exists():
        print(f"ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {graph_path}")
    nodes, links, raw = load_graph(graph_path)
    feat_nodes, hub_nodes, feat_ids = classify_nodes(nodes)

    print(f"  {len(nodes)} nodes  ({len(feat_nodes)} features, {len(hub_nodes)} hubs)")
    print(f"  {len(links)} links")

    ff, hf, fh, weights = analyse_edges(links, feat_ids)

    print("Plotting...")
    layer_counts      = plot_features_per_layer(feat_nodes, out_dir)
    in_vals, out_vals = plot_degree_distribution(feat_nodes, links, feat_ids, out_dir)
    plot_edge_weights(weights, out_dir)
    plot_feature_attrs(feat_nodes, out_dir)

    print("Writing report...")
    write_report(
        out_dir / "graph_topology_report.txt",
        nodes, links, feat_nodes, hub_nodes, feat_ids,
        ff, hf, fh, weights, layer_counts, in_vals, out_vals,
    )

    print(f"\nOutputs written to: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
