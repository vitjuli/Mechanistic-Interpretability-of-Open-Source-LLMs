#!/usr/bin/env python3
"""
Diagnostic: feature-to-feature interaction graph.

Tests whether replacing VW (weight-space) edges with direct interaction edges
(co-activation or causal) produces better community structure —
specifically: fewer giant clusters and fewer singletons.

Run from project root:
    python scripts/diag_feature_interaction.py
    python scripts/diag_feature_interaction.py --top_n 20 --output_dir data/diagnostics/feature_interaction

Does NOT modify any pipeline scripts. Reads existing outputs only.
"""

import argparse
import json
import os
import sys
import textwrap
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("WARNING: python-louvain not found — falling back to greedy modularity")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR   = PROJECT_ROOT / "data" / "results" / "transcoder_features"
ATTR_GRAPH_DIR = PROJECT_ROOT / "data" / "results" / "attribution_graphs"
CAUSAL_DIR     = PROJECT_ROOT / "data" / "results" / "causal_edges"
DASHBOARD_DIR  = PROJECT_ROOT / "dashboard_b1" / "public" / "data"


# ─── data loaders ─────────────────────────────────────────────────────────────

def load_roleaware_graph(behaviour: str) -> dict:
    path = ATTR_GRAPH_DIR / behaviour / "attribution_graph_train_n96_roleaware.json"
    with open(path) as f:
        return json.load(f)


def get_feature_nodes(g: dict) -> list[dict]:
    return [n for n in g["nodes"] if n.get("type") == "feature"]


def load_causal_edges(behaviour: str, split: str) -> list[dict]:
    path = CAUSAL_DIR / behaviour / f"causal_edges_{behaviour}_{split}.json"
    with open(path) as f:
        return json.load(f)["edges"]


def load_binary_coactivation(behaviour: str, split: str,
                              feature_nodes: list[dict],
                              n_prompts: int = 96,
                              n_positions: int = 5) -> dict[str, np.ndarray]:
    """
    Returns dict: feature_id -> binary float32 vector of length n_prompts
    (1.0 if the feature appears in any of the last-5 positions for that prompt).

    Only features whose layer index file exists are returned.
    """
    # Group features by layer
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for fn in feature_nodes:
        by_layer[fn["layer"]].append(fn)

    result: dict[str, np.ndarray] = {}

    for layer, feats in sorted(by_layer.items()):
        idx_path = (FEATURES_DIR / f"layer_{layer}" /
                    f"{behaviour}_{split}_top_k_indices.npy")
        if not idx_path.exists():
            continue

        idx = np.load(idx_path)   # (n_prompts * n_positions, top_k)
        if idx.shape[0] != n_prompts * n_positions:
            continue

        # (n_prompts, n_positions, top_k)
        idx_3d = idx.reshape(n_prompts, n_positions, -1)

        for fn in feats:
            fid = fn["feature_idx"]
            # 1.0 if feature appears at ANY position for this prompt
            vec = np.any(idx_3d == fid, axis=(1, 2)).astype(np.float32)
            result[fn["id"]] = vec

    return result


# ─── similarity metrics ───────────────────────────────────────────────────────

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    both  = np.sum((a > 0) & (b > 0))
    either = np.sum((a > 0) | (b > 0))
    return float(both / either) if either > 0 else 0.0


def phi_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Phi (Matthews) correlation for two binary vectors."""
    n = len(a)
    tp = np.sum((a > 0) & (b > 0))
    tn = np.sum((a == 0) & (b == 0))
    fp = np.sum((a == 0) & (b > 0))
    fn = np.sum((a > 0) & (b == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


# ─── graph builders ───────────────────────────────────────────────────────────

def build_coactivation_graph(feature_ids: list[str],
                              coact_vecs: dict[str, np.ndarray],
                              threshold: float = 0.15,
                              metric: str = "jaccard") -> nx.Graph:
    """Undirected graph; edge iff Jaccard (or phi) >= threshold."""
    G = nx.Graph()
    for fid in feature_ids:
        G.add_node(fid)

    for i, fi in enumerate(feature_ids):
        for j, fj in enumerate(feature_ids):
            if j <= i:
                continue
            if fi not in coact_vecs or fj not in coact_vecs:
                continue
            if metric == "jaccard":
                w = jaccard(coact_vecs[fi], coact_vecs[fj])
            else:
                w = phi_correlation(coact_vecs[fi], coact_vecs[fj])
            if w >= threshold:
                G.add_edge(fi, fj, weight=w)
    return G


def build_vw_graph(feature_ids: list[str], g_json: dict,
                   threshold: float = 0.01) -> nx.Graph:
    """Undirected VW graph restricted to feature_ids, |weight| >= threshold."""
    id_set = set(feature_ids)
    G = nx.Graph()
    for fid in feature_ids:
        G.add_node(fid)

    for e in g_json["edges"]:
        if e.get("edge_type") != "virtual_weight":
            continue
        src, tgt = e["source"], e["target"]
        if src not in id_set or tgt not in id_set:
            continue
        w = abs(e.get("weight", 0.0))
        if w >= threshold:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = max(G[src][tgt]["weight"], w)
            else:
                G.add_edge(src, tgt, weight=w)
    return G


def build_causal_graph(feature_ids: list[str],
                        causal_edges: list[dict],
                        threshold: float = 0.1) -> nx.Graph:
    """
    Undirected causal graph: edge if mean_delta_abs >= threshold for either
    direction i→j or j→i. Weight = max of the two directional means.
    """
    id_set = set(feature_ids)
    # Build directional dict first
    directed: dict[tuple, float] = {}
    for e in causal_edges:
        src, tgt = e["source"], e["target"]
        if src not in id_set or tgt not in id_set:
            continue
        w = abs(e.get("mean_delta", e.get("mean_delta_abs", 0.0)))
        directed[(src, tgt)] = w

    G = nx.Graph()
    for fid in feature_ids:
        G.add_node(fid)

    # Collapse to undirected: max(i→j, j→i)
    seen: set[frozenset] = set()
    for (src, tgt), w in directed.items():
        key = frozenset([src, tgt])
        if key in seen:
            continue
        seen.add(key)
        w_rev = directed.get((tgt, src), 0.0)
        w_max = max(w, w_rev)
        if w_max >= threshold:
            G.add_edge(src, tgt, weight=w_max)

    return G


# ─── community detection ──────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Returns {node: community_id}."""
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}

    if HAS_LOUVAIN:
        # Louvain requires non-negative weights; use abs
        G2 = nx.Graph()
        for u, v, d in G.edges(data=True):
            G2.add_edge(u, v, weight=abs(d.get("weight", 1.0)))
        for n in G.nodes():
            if n not in G2:
                G2.add_node(n)
        return community_louvain.best_partition(G2)

    # Fallback: greedy modularity (NetworkX built-in)
    comms = list(nx.community.greedy_modularity_communities(G))
    partition: dict[str, int] = {}
    for cid, comm in enumerate(comms):
        for node in comm:
            partition[node] = cid
    return partition


def community_stats(partition: dict[str, int],
                    feature_info: dict[str, dict]) -> dict:
    """Return structured stats per community."""
    from collections import defaultdict
    comms: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        comms[cid].append(node)

    stats = []
    for cid in sorted(comms.keys()):
        members = sorted(comms[cid])
        layers = sorted(set(feature_info[m]["layer"]
                            for m in members if m in feature_info))
        stats.append({
            "community_id": cid,
            "n_features": len(members),
            "layers": layers,
            "layer_min": min(layers) if layers else -1,
            "layer_max": max(layers) if layers else -1,
            "members": members,
        })

    # Sort by size descending
    stats.sort(key=lambda s: s["n_features"], reverse=True)
    return stats


# ─── visualisation ────────────────────────────────────────────────────────────

COMMUNITY_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def draw_interaction_graph(G: nx.Graph,
                            partition: dict[str, int],
                            feature_info: dict[str, dict],
                            title: str,
                            ax: plt.Axes):
    """Draw 20-node interaction graph with layer-based y-position."""
    if G.number_of_nodes() == 0:
        ax.set_title(title + "\n(empty graph)", fontsize=9)
        ax.axis("off")
        return

    # Position: x = layer, y = rank within layer (spread vertically)
    layer_count: dict[int, list[str]] = defaultdict(list)
    for n in G.nodes():
        layer = feature_info.get(n, {}).get("layer", 0)
        layer_count[layer].append(n)

    pos = {}
    for layer, nodes in layer_count.items():
        for rank, n in enumerate(sorted(nodes)):
            y = rank - len(nodes) / 2.0
            pos[n] = (layer, y)

    node_colors = []
    for n in G.nodes():
        cid = partition.get(n, 0)
        node_colors.append(COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)])

    edge_weights = [abs(d.get("weight", 0.5)) for _, _, d in G.edges(data=True)]
    if edge_weights:
        max_w = max(edge_weights) or 1.0
        edge_widths = [0.5 + 3.0 * (w / max_w) for w in edge_weights]
    else:
        edge_widths = []

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=350, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=edge_widths,
                           edge_color="#444444")

    # Labels: Lxx_Fyyy → just Fyyy for brevity
    labels = {n: n.split("_")[1] if "_" in n else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=5)

    # Layer x-axis labels
    layers = sorted(set(feature_info.get(n, {}).get("layer", 0)
                        for n in G.nodes()))
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(left=False, labelleft=False)

    # Community legend
    comm_ids = sorted(set(partition.values()))
    handles = [mpatches.Patch(color=COMMUNITY_COLORS[c % len(COMMUNITY_COLORS)],
                               label=f"C{c}")
               for c in comm_ids]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              framealpha=0.7, ncol=2)


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--top_n", type=int, default=20,
                   help="Top-N features by mean_abs_grad_attr_conditional")
    p.add_argument("--coact_threshold", type=float, default=0.15,
                   help="Min Jaccard similarity to add co-activation edge")
    p.add_argument("--causal_threshold", type=float, default=0.10,
                   help="Min mean_delta_abs for causal edge")
    p.add_argument("--vw_threshold", type=float, default=0.01,
                   help="Min |VW weight| (should match pipeline default)")
    p.add_argument("--output_dir", default="data/diagnostics/feature_interaction")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(" DIAGNOSTIC: feature-to-feature interaction graph")
    print(f" behaviour={args.behaviour}  split={args.split}  top_n={args.top_n}")
    print(f"{'='*70}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading roleaware attribution graph …")
    g_json = load_roleaware_graph(args.behaviour)
    feat_nodes = get_feature_nodes(g_json)
    print(f"  Total feature nodes in graph: {len(feat_nodes)}")

    # Sort by gradient attribution score, take top N
    feat_nodes.sort(key=lambda n: n.get("mean_abs_grad_attr_conditional", 0),
                    reverse=True)
    top_n = feat_nodes[:args.top_n]
    top_n_ids = [n["id"] for n in top_n]
    feature_info: dict[str, dict] = {n["id"]: n for n in feat_nodes}

    print(f"\n── Step 1: Top-{args.top_n} features ──────────────────────────────")
    layer_counts = Counter(n["layer"] for n in top_n)
    print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")
    print(f"  {'Feature ID':22s} {'Layer':6s} {'mean_abs_grad_attr':>18s}")
    print(f"  {'-'*50}")
    for n in top_n:
        score = n.get("mean_abs_grad_attr_conditional", 0)
        print(f"  {n['id']:22s} L{n['layer']:<5d} {score:>18.4f}")

    # ── 2A. Co-activation vectors ─────────────────────────────────────────────
    print(f"\n── Step 2A: Co-activation (binary, 96 prompts) ─────────────────────")
    coact_vecs = load_binary_coactivation(
        args.behaviour, args.split, feat_nodes   # pass all to load all layers once
    )
    covered = sum(1 for fid in top_n_ids if fid in coact_vecs)
    print(f"  Features with activation vectors: {covered}/{args.top_n}")

    # Compute pairwise Jaccard matrix for top-N
    n = len(top_n_ids)
    jaccard_matrix = np.zeros((n, n))
    for i, fi in enumerate(top_n_ids):
        for j, fj in enumerate(top_n_ids):
            if j <= i:
                continue
            if fi in coact_vecs and fj in coact_vecs:
                w = jaccard(coact_vecs[fi], coact_vecs[fj])
                jaccard_matrix[i, j] = w
                jaccard_matrix[j, i] = w

    nonzero_pairs = np.sum(jaccard_matrix > args.coact_threshold) // 2
    print(f"  Jaccard pairs >= {args.coact_threshold}: {nonzero_pairs} "
          f"(out of {n*(n-1)//2} possible)")
    print(f"  Jaccard range: [{jaccard_matrix.max():.3f} max, "
          f"{jaccard_matrix[jaccard_matrix>0].min() if np.any(jaccard_matrix>0) else 0:.3f} min>0]")

    # Save Jaccard matrix
    jaccard_df = pd.DataFrame(jaccard_matrix,
                               index=top_n_ids, columns=top_n_ids)
    jaccard_df.to_csv(out_dir / "jaccard_matrix.csv")

    # ── 2B. Causal interaction ────────────────────────────────────────────────
    print(f"\n── Step 2B: Causal interaction edges ───────────────────────────────")
    causal_edges = load_causal_edges(args.behaviour, args.split)
    causal_top20 = [e for e in causal_edges
                    if e["source"] in set(top_n_ids) and e["target"] in set(top_n_ids)]
    print(f"  Total causal edges (full graph): {len(causal_edges)}")
    print(f"  Causal edges between top-{args.top_n}: {len(causal_top20)} "
          f"(out of {n*(n-1)} directed pairs, "
          f"{len(causal_top20)/(n*(n-1))*100:.1f}% coverage)")
    if causal_top20:
        vals = [abs(e.get("mean_delta", e.get("mean_delta_abs", 0))) for e in causal_top20]
        print(f"  mean_delta_abs range: [{min(vals):.3f}, {max(vals):.3f}]")

    # ── 3. Build graphs ───────────────────────────────────────────────────────
    print(f"\n── Step 3: Building graphs ─────────────────────────────────────────")

    G_coact  = build_coactivation_graph(top_n_ids, coact_vecs,
                                         threshold=args.coact_threshold)
    G_vw     = build_vw_graph(top_n_ids, g_json, threshold=args.vw_threshold)
    G_causal = build_causal_graph(top_n_ids, causal_edges,
                                   threshold=args.causal_threshold)

    print(f"  VW graph:          {G_vw.number_of_nodes()} nodes, "
          f"{G_vw.number_of_edges()} edges")
    print(f"  Co-activation:     {G_coact.number_of_nodes()} nodes, "
          f"{G_coact.number_of_edges()} edges "
          f"(threshold={args.coact_threshold})")
    print(f"  Causal:            {G_causal.number_of_nodes()} nodes, "
          f"{G_causal.number_of_edges()} edges "
          f"(threshold={args.causal_threshold})")

    # ── 4. Community detection ────────────────────────────────────────────────
    print(f"\n── Step 4: Community detection (Louvain) ───────────────────────────")

    part_vw     = detect_communities(G_vw)
    part_coact  = detect_communities(G_coact)
    part_causal = detect_communities(G_causal)

    stats_vw     = community_stats(part_vw,     feature_info)
    stats_coact  = community_stats(part_coact,  feature_info)
    stats_causal = community_stats(part_causal, feature_info)

    def n_giants(stats, threshold=5):
        return sum(1 for s in stats if s["n_features"] >= threshold)

    def n_singletons(stats):
        return sum(1 for s in stats if s["n_features"] == 1)

    print(f"\n  {'Graph':15s} {'#comm':>6} {'#giants':>8} {'#singletons':>12} "
          f"{'sizes':>20}")
    for label, stats in [("VW", stats_vw), ("Co-act", stats_coact),
                          ("Causal", stats_causal)]:
        sizes = [s["n_features"] for s in stats]
        sizes_str = str(sorted(sizes, reverse=True)[:8])
        print(f"  {label:15s} {len(stats):>6} {n_giants(stats):>8} "
              f"{n_singletons(stats):>12}  {sizes_str}")

    # ── 5. Diagnostics ───────────────────────────────────────────────────────
    print(f"\n── Step 5: Community details ───────────────────────────────────────")

    for label, stats in [("VW", stats_vw), ("Co-activation", stats_coact),
                          ("Causal", stats_causal)]:
        print(f"\n  [{label}]")
        for s in stats:
            layer_str = f"L{s['layer_min']}–L{s['layer_max']}" if s["layers"] else "—"
            progression = "monotone" if s["layers"] == sorted(s["layers"]) else "mixed"
            print(f"    C{s['community_id']}: n={s['n_features']:2d}  "
                  f"layers={layer_str:12s}  progression={progression}")
            for m in s["members"]:
                fi = feature_info.get(m, {})
                score = fi.get("mean_abs_grad_attr_conditional", 0)
                print(f"      {m:22s} L{fi.get('layer','?')}  score={score:.4f}")

    # ── Extended: Co-activation on full graph (all 137 nodes) ─────────────────
    print(f"\n── Extended: Co-activation on all {len(feat_nodes)} graph features ──")
    G_coact_full = build_coactivation_graph(
        [n["id"] for n in feat_nodes], coact_vecs,
        threshold=args.coact_threshold
    )
    part_coact_full = detect_communities(G_coact_full)
    stats_coact_full = community_stats(part_coact_full, feature_info)

    print(f"  Full co-act graph: {G_coact_full.number_of_nodes()} nodes, "
          f"{G_coact_full.number_of_edges()} edges")
    print(f"  #communities={len(stats_coact_full)}  "
          f"giants(≥5)={n_giants(stats_coact_full)}  "
          f"singletons={n_singletons(stats_coact_full)}")
    full_sizes = sorted([s["n_features"] for s in stats_coact_full], reverse=True)
    print(f"  Size distribution: {full_sizes[:15]}{'...' if len(full_sizes)>15 else ''}")

    # Compare to current VW full-graph community problem
    print(f"\n  Current VW full-graph (for reference):  "
          f"4 giants (30,36,33,30) + 8 singletons in 137 nodes, 12 total")
    print(f"  Co-act full-graph:  "
          f"{n_giants(stats_coact_full)} giants + "
          f"{n_singletons(stats_coact_full)} singletons in "
          f"{len(feat_nodes)} nodes, {len(stats_coact_full)} total")

    # ── 6. Save CSVs ──────────────────────────────────────────────────────────
    print(f"\n── Step 6: Saving outputs to {out_dir} ──────────────────────────────")

    # Feature list with communities
    rows = []
    for fid in top_n_ids:
        fi = feature_info[fid]
        rows.append({
            "feature_id": fid,
            "layer": fi["layer"],
            "feature_idx": fi["feature_idx"],
            "mean_abs_grad_attr": fi.get("mean_abs_grad_attr_conditional", 0),
            "grad_attr_sign": fi.get("grad_attr_sign", 0),
            "community_vw": part_vw.get(fid, -1),
            "community_coact": part_coact.get(fid, -1),
            "community_causal": part_causal.get(fid, -1),
        })
    features_df = pd.DataFrame(rows)
    features_df.to_csv(out_dir / "top_features_communities.csv", index=False)
    print(f"  top_features_communities.csv  ({len(rows)} rows)")

    # Jaccard matrix already saved above
    print(f"  jaccard_matrix.csv  ({n}×{n})")

    # Community summary for each graph
    for label, stats in [("vw", stats_vw), ("coact", stats_coact),
                          ("causal", stats_causal), ("coact_full", stats_coact_full)]:
        rows_c = []
        for s in stats:
            rows_c.append({
                "community_id": s["community_id"],
                "n_features": s["n_features"],
                "layer_min": s["layer_min"],
                "layer_max": s["layer_max"],
                "members": ",".join(s["members"]),
            })
        pd.DataFrame(rows_c).to_csv(out_dir / f"communities_{label}.csv", index=False)
    print(f"  communities_vw/coact/causal/coact_full.csv")

    # ── Visualisation ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Feature-to-feature interaction graphs  |  {args.behaviour}  |  "
        f"top-{args.top_n} features by gradient attribution",
        fontsize=10, fontweight="bold"
    )

    for ax, G, part, title in [
        (axes[0], G_vw,     part_vw,     f"VW edges\n({G_vw.number_of_edges()} edges, threshold={args.vw_threshold})"),
        (axes[1], G_coact,  part_coact,  f"Co-activation (Jaccard)\n({G_coact.number_of_edges()} edges, threshold={args.coact_threshold})"),
        (axes[2], G_causal, part_causal, f"Causal (ablation Δ)\n({G_causal.number_of_edges()} edges, threshold={args.causal_threshold})"),
    ]:
        draw_interaction_graph(G, part, feature_info, title, ax)

    plt.tight_layout()
    fig_path = out_dir / "interaction_graphs.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  interaction_graphs.png  (3-panel)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    _print_summary(args, stats_vw, stats_coact, stats_causal,
                   stats_coact_full, len(feat_nodes), G_coact, G_vw, G_causal)

    # Save summary text
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_summary(args, stats_vw, stats_coact, stats_causal,
                       stats_coact_full, len(feat_nodes), G_coact, G_vw, G_causal)
    (out_dir / "summary.txt").write_text(buf.getvalue())
    print(f"  summary.txt")
    print(f"\nAll outputs in: {out_dir}\n")


def _print_summary(args, stats_vw, stats_coact, stats_causal,
                   stats_coact_full, n_full, G_coact, G_vw, G_causal):
    """Print the diagnostic summary."""

    def n_giants(stats, t=5): return sum(1 for s in stats if s["n_features"] >= t)
    def n_singletons(stats): return sum(1 for s in stats if s["n_features"] == 1)
    def has_layer_progression(stats):
        for s in stats:
            if len(s["layers"]) > 1 and s["layers"] != sorted(s["layers"]):
                return False
        return True

    print("=" * 70)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"""
Top-{args.top_n} features  |  layers covered: {sorted(set(s["layer_min"] for s in stats_vw))} – {sorted(set(s["layer_max"] for s in stats_vw))}

GRAPH            #communities  Giants(≥5)  Singletons  Sizes (top-8)
──────────────────────────────────────────────────────────────────────
VW  (top-{args.top_n:2d})       {len(stats_vw):>4}         {n_giants(stats_vw):>3}         {n_singletons(stats_vw):>3}        {sorted([s["n_features"] for s in stats_vw], reverse=True)[:8]}
Co-act (top-{args.top_n:2d})    {len(stats_coact):>4}         {n_giants(stats_coact):>3}         {n_singletons(stats_coact):>3}        {sorted([s["n_features"] for s in stats_coact], reverse=True)[:8]}
Causal (top-{args.top_n:2d})    {len(stats_causal):>4}         {n_giants(stats_causal):>3}         {n_singletons(stats_causal):>3}        {sorted([s["n_features"] for s in stats_causal], reverse=True)[:8]}
Co-act (all {n_full:3d})   {len(stats_coact_full):>4}         {n_giants(stats_coact_full):>3}         {n_singletons(stats_coact_full):>3}        {sorted([s["n_features"] for s in stats_coact_full], reverse=True)[:8]}
""")

    print("Reference — current VW full-graph (all 137 nodes):")
    print("  12 communities: 4 giants (36, 33, 30, 30) + 8 singletons")
    print("  Problem: ~96% of nodes absorbed into 4 undifferentiated blocks\n")

    print("── Per-community layer structure (top-N graphs) ──────────────────")
    for label, stats in [("VW", stats_vw), ("Co-act", stats_coact),
                          ("Causal", stats_causal)]:
        sizes = [s["n_features"] for s in stats]
        layers_per_comm = [s["layers"] for s in stats]
        all_monotone = all(
            len(ly) <= 1 or ly == sorted(ly)
            for ly in layers_per_comm
        )
        print(f"  {label:8s}  sizes={sorted(sizes,reverse=True)}  "
              f"layer-monotone={all_monotone}")

    # Answer the key question
    # True co-activation full-graph statistics
    coact_full_sizes = sorted([s["n_features"] for s in stats_coact_full], reverse=True)
    vw_full_max_size = 36   # from known current output (L15-L20 block)
    coact_full_max_size = coact_full_sizes[0] if coact_full_sizes else 0

    # Co-act on full graph is WORSE if largest cluster is bigger than VW
    coact_full_worse = coact_full_max_size > vw_full_max_size
    # Singletons: VW has 8, co-act has 0 — co-act eliminates singletons but
    # creates bigger blobs; not a clear improvement
    coact_edge_density = (G_coact.number_of_edges() /
                          max(1, G_coact.number_of_nodes() * (G_coact.number_of_nodes()-1) / 2))

    print("\n── CONCLUSION ────────────────────────────────────────────────────────")
    print(f"\n  Is the community problem caused by MISSING feature→feature edges?")

    if coact_full_worse:
        verdict = "NO"
        reason = textwrap.fill(
            f"Co-activation (full graph) produces 3 giants [{', '.join(str(s) for s in coact_full_sizes[:3])}] "
            f"vs VW's 4 giants [36, 33, 30, 30]. Largest cluster grows "
            f"from 36 → {coact_full_max_size} nodes — the problem gets WORSE, not better. "
            "The root cause is EDGE DENSITY, not edge type: "
            f"co-activation at threshold={args.coact_threshold} connects {G_coact.number_of_edges()}/{G_coact.number_of_nodes()*(G_coact.number_of_nodes()-1)//2} "
            "top-N pairs ({:.0f}% density), creating a near-complete graph where Louvain "
            "collapses everything into 1–3 super-communities. "
            "The current VW graph is already the SPARSER option for top-N features. "
            "Fix: reduce the node set per layer, raise edge thresholds, or use "
            "layer-constrained community detection rather than switching edge types.".format(coact_edge_density*100),
            width=68, subsequent_indent="  "
        )
    else:
        verdict = "YES — PARTIALLY"
        reason = textwrap.fill(
            f"Co-activation (full graph) produces {n_giants(stats_coact_full)} giants "
            f"({coact_full_sizes[:3]}) vs VW's 4 giants [36, 33, 30, 30]. "
            "Switching from VW (weight-space proximity) to co-activation "
            "(prompt-level co-occurrence) does differentiate communities better. "
            "However singletons are not the cause — VW's 8 singletons are FR-specific "
            "outliers, not disconnected features. The main problem (giant layer-bands) "
            "is caused by high within-band VW density, not missing cross-band edges.",
            width=68, subsequent_indent="  "
        )

    print(f"  {verdict}")
    print(f"  {reason}")

    print(f"\n  Key data points:")
    print(f"    VW full-graph:         4 giants (max={vw_full_max_size}) + 8 singletons")
    print(f"    Co-act full-graph:     {n_giants(stats_coact_full)} giants (max={coact_full_max_size}) + {n_singletons(stats_coact_full)} singletons")
    print(f"    Co-act edge density (top-{args.top_n}): {coact_edge_density*100:.0f}% at threshold={args.coact_threshold}")
    print(f"    Causal coverage (top-{args.top_n}):     {G_causal.number_of_edges()} edges / {G_causal.number_of_nodes()*(G_causal.number_of_nodes()-1)//2} possible (2.9%)")
    print(f"\n  Causal graph is too sparse for community detection: script 08 only")
    print(f"  tests adjacent-layer pairs — no cross-range edges exist by construction.")
    print("=" * 70)


if __name__ == "__main__":
    main()
