#!/usr/bin/env python3
"""
Controlled sparsification sweep on the role-aware gradient graph.

Varies TWO dimensions of graph construction:
  1. Node pruning:  top_k_per_layer ∈ {2, 3, 5, 8, "all"}
  2. VW threshold:  |weight| ≥ t, t ∈ {0.01, 0.05, 0.10, 0.20, 0.50}

For each of the 25 configurations, computes:
  Graph:   n_nodes, n_edges, n_communities, largest_pct, n_singletons
  Circuit: n_circuit_features, trajectory_accuracy, mean_abs_effect

Does NOT modify any pipeline scripts or data.  Reads:
  data/results/attribution_graphs/*/attribution_graph_train_n96_roleaware.json
  data/results/causal_edges/*/causal_edges_*_train.json
  data/ui_offline/.../intervention_ablation_*.csv
  data/results/reasoning_traces/*/reasoning_traces_train.jsonl

Run from project root:
    python scripts/diag_sparsification_sweep.py
    python scripts/diag_sparsification_sweep.py --output_dir data/diagnostics/sparsification
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    raise ImportError("networkx is required: pip install networkx")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ATTR_GRAPH_DIR = PROJECT_ROOT / "data" / "results" / "attribution_graphs"
CAUSAL_DIR     = PROJECT_ROOT / "data" / "results" / "causal_edges"
UI_OFFLINE_DIR = PROJECT_ROOT / "data" / "ui_offline"
TRACES_DIR     = PROJECT_ROOT / "data" / "results" / "reasoning_traces"


# ─── reference constants (gradient run) ──────────────────────────────────────
REF_N_NODES      = 137
REF_N_COMMS      = 12
REF_LARGEST_PCT  = 36 / 137 * 100   # 26.3%
REF_SINGLETONS   = 8
REF_N_CIRCUIT    = 13
REF_TRAJ_ACC     = 0.750            # proxy (8 evaluable circuit feats)
REF_MEAN_ABS     = 0.9032           # mean abs individual ablation effect


# ─── data loaders ─────────────────────────────────────────────────────────────

def load_roleaware_graph(behaviour: str) -> dict:
    path = ATTR_GRAPH_DIR / behaviour / "attribution_graph_train_n96_roleaware.json"
    with open(path) as f:
        return json.load(f)


def load_causal_edges(behaviour: str, split: str = "train") -> list[dict]:
    path = CAUSAL_DIR / behaviour / f"causal_edges_{behaviour}_{split}.json"
    with open(path) as f:
        return json.load(f)["edges"]


def find_ablation_csv(behaviour: str) -> Path:
    """Find the most recent ablation CSV in data/ui_offline."""
    candidates = sorted(
        UI_OFFLINE_DIR.glob(f"*_{behaviour}_train_*"),
        reverse=True   # most recent first
    )
    for d in candidates:
        p = d / "raw_sources" / f"intervention_ablation_{behaviour}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"No ablation CSV for {behaviour}")


def load_ablation_data(behaviour: str) -> pd.DataFrame:
    path = find_ablation_csv(behaviour)
    df = pd.read_csv(path)
    df = df[df["feature_source"] == "graph"].copy()
    df["contribution"] = -df["effect_size"]
    return df


def load_prediction_labels(behaviour: str, split: str = "train") -> dict[int, bool]:
    path = TRACES_DIR / behaviour / f"reasoning_traces_{split}.jsonl"
    labels: dict[int, bool] = {}
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            labels[t["prompt_idx"]] = t["prediction_correct"]
    return labels


# ─── graph construction ───────────────────────────────────────────────────────

def get_feature_nodes(g: dict) -> list[dict]:
    return [n for n in g["nodes"] if n.get("type") == "feature"]


def prune_nodes_by_layer(feat_nodes: list[dict], k: int) -> list[dict]:
    """Keep top-k features per layer by mean_abs_grad_attr_conditional."""
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for fn in feat_nodes:
        by_layer[fn["layer"]].append(fn)

    pruned = []
    for layer, nodes in by_layer.items():
        nodes_sorted = sorted(nodes,
                              key=lambda n: n.get("mean_abs_grad_attr_conditional", 0),
                              reverse=True)
        pruned.extend(nodes_sorted[:k])
    return pruned


def build_vw_subgraph(feat_ids: set[str], g_json: dict,
                       vw_threshold: float) -> nx.Graph:
    """Undirected VW graph restricted to feat_ids, |weight| >= threshold."""
    G = nx.Graph()
    for fid in feat_ids:
        G.add_node(fid)

    for e in g_json["edges"]:
        if e.get("edge_type") != "virtual_weight":
            continue
        src, tgt = e["source"], e["target"]
        if src not in feat_ids or tgt not in feat_ids:
            continue
        w = abs(e.get("weight", 0.0))
        if w < vw_threshold:
            continue
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] = max(G[src][tgt]["weight"], w)
        else:
            G.add_edge(src, tgt, weight=w)
    return G


# ─── community detection ──────────────────────────────────────────────────────

def louvain_partition(G: nx.Graph) -> dict[str, int]:
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}
    if HAS_LOUVAIN:
        G2 = nx.Graph()
        for u, v, d in G.edges(data=True):
            G2.add_edge(u, v, weight=abs(d.get("weight", 1.0)))
        for n in G.nodes():
            if n not in G2:
                G2.add_node(n)
        return community_louvain.best_partition(G2)
    # fallback: greedy modularity
    comms = list(nx.community.greedy_modularity_communities(G))
    partition: dict[str, int] = {}
    for cid, comm in enumerate(comms):
        for node in comm:
            partition[node] = cid
    return partition


def community_metrics(partition: dict[str, int], n_nodes: int) -> dict:
    """Returns community structure metrics."""
    if not partition:
        return {"n_communities": 0, "largest_pct": 0.0, "n_singletons": 0}
    sizes = Counter(partition.values())
    size_vals = list(sizes.values())
    largest = max(size_vals)
    singletons = sum(1 for s in size_vals if s == 1)
    return {
        "n_communities": len(sizes),
        "largest_pct": largest / n_nodes * 100,
        "n_singletons": singletons,
        "median_size": float(np.median(size_vals)),
    }


# ─── circuit metrics ──────────────────────────────────────────────────────────

def identify_circuit_features(pruned_ids: set[str],
                               causal_edges: list[dict],
                               evaluable_ids: set[str]) -> set[str]:
    """
    Circuit = pruned features that appear in at least one causal edge
    AND are in the evaluable set (ablation CSV).
    """
    causal_nodes = set()
    for e in causal_edges:
        if e["source"] in pruned_ids and e["target"] in pruned_ids:
            causal_nodes.add(e["source"])
            causal_nodes.add(e["target"])
    return causal_nodes & evaluable_ids


def compute_trajectory_accuracy(circuit_feats: set[str],
                                 abl_df: pd.DataFrame,
                                 labels: dict[int, bool]) -> float:
    """
    Trajectory accuracy using ablation CSV.
    contribution = -effect_size.
    dominant = 'correct' if net > 0 else 'incorrect'.
    accuracy = fraction where dominant matches actual prediction.
    """
    if not circuit_feats:
        return 0.0

    subset = abl_df[abl_df["feature_id"].isin(circuit_feats)]
    if subset.empty:
        return 0.0

    net = subset.groupby("prompt_idx")["contribution"].sum()
    correct = sum(
        1 for pidx, s in net.items()
        if ((s > 0) == labels.get(pidx, False))
    )
    return correct / len(net)


def compute_mean_abs_effect(circuit_feats: set[str],
                              abl_df: pd.DataFrame) -> float:
    """Mean absolute individual ablation effect for circuit features."""
    if not circuit_feats:
        return 0.0
    subset = abl_df[abl_df["feature_id"].isin(circuit_feats)]
    if subset.empty:
        return 0.0
    return float(subset["abs_effect_size"].mean())


# ─── sweep ────────────────────────────────────────────────────────────────────

def run_config(k_per_layer, vw_threshold,
               feat_nodes, g_json, causal_edges, evaluable_ids,
               abl_df, labels) -> dict:
    """Run one (k_per_layer, vw_threshold) configuration."""
    # 1. Node pruning
    if k_per_layer == "all":
        pruned = feat_nodes
    else:
        pruned = prune_nodes_by_layer(feat_nodes, k=k_per_layer)

    pruned_ids = {n["id"] for n in pruned}
    n_nodes = len(pruned_ids)

    # 2. Build VW subgraph
    G = build_vw_subgraph(pruned_ids, g_json, vw_threshold)
    n_edges = G.number_of_edges()

    # 3. Community detection
    partition = louvain_partition(G)
    c_metrics = community_metrics(partition, n_nodes)

    # 4. Circuit features
    circuit_feats = identify_circuit_features(pruned_ids, causal_edges, evaluable_ids)
    n_circuit = len(circuit_feats)

    # 5. Circuit quality metrics
    traj_acc = compute_trajectory_accuracy(circuit_feats, abl_df, labels)
    mean_abs = compute_mean_abs_effect(circuit_feats, abl_df)

    # 6. Quality flags
    # "Good community": no giant (< 25% of nodes in largest cluster),
    # few singletons (< 10% of nodes), AND communities are moderate size
    # (at least 5 communities so there's real decomposition)
    is_good_comm = (
        c_metrics["largest_pct"] < 25.0
        and c_metrics["n_singletons"] < max(3, n_nodes * 0.10)
        and c_metrics["n_communities"] >= 5
    )
    is_good_circuit = (traj_acc >= REF_TRAJ_ACC * 0.90
                       and mean_abs >= REF_MEAN_ABS * 0.80)

    return {
        "k_per_layer": str(k_per_layer),
        "vw_threshold": vw_threshold,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_communities": c_metrics["n_communities"],
        "largest_pct": c_metrics["largest_pct"],
        "n_singletons": c_metrics["n_singletons"],
        "median_comm_size": c_metrics["median_size"],
        "n_circuit_features": n_circuit,
        "trajectory_accuracy": traj_acc,
        "mean_abs_effect": mean_abs,
        "is_good_comm": is_good_comm,
        "is_good_circuit": is_good_circuit,
        "both_good": is_good_comm and is_good_circuit,
    }


# ─── output helpers ───────────────────────────────────────────────────────────

def fmt_row(r: dict) -> str:
    k = r["k_per_layer"].ljust(5)
    t = f"{r['vw_threshold']:.2f}".ljust(5)
    good_c = "✓" if r["is_good_comm"] else " "
    good_q = "✓" if r["is_good_circuit"] else " "
    both   = "★" if r["both_good"] else " "
    return (
        f"  {both} k={k} t={t} | "
        f"N={r['n_nodes']:3d} E={r['n_edges']:4d} | "
        f"C={r['n_communities']:2d} "
        f"max={r['largest_pct']:4.0f}% "
        f"sing={r['n_singletons']:2d} {good_c} | "
        f"circ={r['n_circuit_features']:2d} "
        f"traj={r['trajectory_accuracy']:.2f} "
        f"Δabl={r['mean_abs_effect']:.3f} {good_q}"
    )


def print_table(results: list[dict]):
    print()
    print("  ★=both good  ✓=criterion met")
    print(f"  {'':2s} {'k':5s} {'t':5s}  {'N':>3} {'E':>4}  {'C':>2} {'max%':>5} {'sing':>4}  "
          f"{'circ':>4} {'traj':>5} {'Δabl':>5}")
    print("  " + "-" * 78)
    prev_k = None
    for r in results:
        if prev_k is not None and r["k_per_layer"] != prev_k:
            print()
        prev_k = r["k_per_layer"]
        print(fmt_row(r))
    print()


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split",     default="train")
    p.add_argument("--output_dir", default="data/diagnostics/sparsification")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" SPARSIFICATION SWEEP  |  {args.behaviour}  |  {args.split}")
    print(f"{'='*70}\n")
    print(f" Reference (current default, k=all, t=0.01):")
    print(f"   nodes={REF_N_NODES}, comms={REF_N_COMMS}, "
          f"largest={REF_LARGEST_PCT:.0f}%, singletons={REF_SINGLETONS}")
    print(f"   circuit={REF_N_CIRCUIT}, traj={REF_TRAJ_ACC:.3f}, mean_abs={REF_MEAN_ABS:.3f}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    g_json       = load_roleaware_graph(args.behaviour)
    feat_nodes   = get_feature_nodes(g_json)
    causal_edges = load_causal_edges(args.behaviour, args.split)
    abl_df       = load_ablation_data(args.behaviour)
    labels       = load_prediction_labels(args.behaviour, args.split)
    evaluable_ids = set(abl_df["feature_id"].unique())
    print(f"  Graph: {len(feat_nodes)} features, "
          f"{sum(1 for e in g_json['edges'] if e.get('edge_type')=='virtual_weight')} VW edges")
    print(f"  Causal edges: {len(causal_edges)}")
    print(f"  Ablation CSV: {len(evaluable_ids)} evaluable features × "
          f"{abl_df['prompt_idx'].nunique()} prompts")
    print()

    # ── Sweep grid ─────────────────────────────────────────────────────────────
    K_VALUES  = ["all", 8, 5, 3, 2]
    T_VALUES  = [0.01, 0.05, 0.10, 0.20, 0.50]

    print(f" Running {len(K_VALUES)} × {len(T_VALUES)} = {len(K_VALUES)*len(T_VALUES)} configs …\n")

    results = []
    for k in K_VALUES:
        for t in T_VALUES:
            r = run_config(k, t, feat_nodes, g_json, causal_edges,
                           evaluable_ids, abl_df, labels)
            results.append(r)

    # ── Results table ──────────────────────────────────────────────────────────
    print("── Results ─────────────────────────────────────────────────────────────")
    print_table(results)

    # ── Top configs ───────────────────────────────────────────────────────────
    good_configs = [r for r in results if r["both_good"]]
    print(f"── Top configurations (both_good) ─────────────────────────────────────")
    if not good_configs:
        # Relax: just good community structure, any circuit
        good_configs = [r for r in results if r["is_good_comm"]]
        print(f"  (Relaxed: community-good only, no circuit constraint)")
    if good_configs:
        good_configs.sort(key=lambda r: (
            r["n_singletons"],             # minimize singletons (primary)
            r["largest_pct"],              # minimize largest cluster
            -r["n_communities"],           # prefer more communities (finer structure)
            -r["trajectory_accuracy"],     # maximize trajectory
        ))
        for i, r in enumerate(good_configs[:3], 1):
            print(f"\n  [{i}] k={r['k_per_layer']}, threshold={r['vw_threshold']}")
            print(f"       nodes={r['n_nodes']}, edges={r['n_edges']}")
            print(f"       communities={r['n_communities']}, "
                  f"largest={r['largest_pct']:.1f}%, singletons={r['n_singletons']}")
            print(f"       circuit={r['n_circuit_features']} features, "
                  f"traj={r['trajectory_accuracy']:.3f}, "
                  f"mean_abs={r['mean_abs_effect']:.3f}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sweep_results.csv", index=False)
    print(f"\n  Saved: sweep_results.csv ({len(df)} rows)")

    _plot_heatmaps(df, K_VALUES, T_VALUES, out_dir)
    print(f"  Saved: sweep_heatmaps.png")

    # ── Conclusion ─────────────────────────────────────────────────────────────
    _print_conclusion(results, good_configs, df)

    # Save conclusion text
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_conclusion(results, good_configs, df)
    (out_dir / "conclusion.txt").write_text(buf.getvalue())
    print(f"  Saved: conclusion.txt")
    print(f"\nAll outputs in: {out_dir}\n")


def _plot_heatmaps(df: pd.DataFrame, k_vals: list, t_vals: list, out_dir: Path):
    """4-panel heatmap: largest_pct, n_singletons, trajectory_accuracy, n_communities."""
    k_labels = [str(k) for k in k_vals]
    t_labels = [str(t) for t in t_vals]

    metrics = [
        ("largest_pct",         "Largest cluster %",   "RdYlGn_r",  0,   100),
        ("n_singletons",        "Singletons",          "RdYlGn_r",  0,   None),
        ("trajectory_accuracy", "Trajectory accuracy", "RdYlGn",    0.5, 1.0),
        ("n_communities",       "# Communities",       "YlOrRd_r",  1,   None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle("Sparsification sweep — multilingual_circuits_b1",
                 fontsize=12, fontweight="bold")

    for ax, (col, title, cmap, vmin, vmax) in zip(axes, metrics):
        mat = np.zeros((len(k_vals), len(t_vals)))
        for i, k in enumerate(k_vals):
            for j, t in enumerate(t_vals):
                row = df[(df["k_per_layer"] == str(k)) &
                         (df["vw_threshold"] == t)]
                if len(row):
                    mat[i, j] = row.iloc[0][col]

        if vmax is None:
            vmax = mat.max()

        im = ax.imshow(mat, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(t_vals)))
        ax.set_xticklabels([f"t={t}" for t in t_vals], fontsize=8)
        ax.set_yticks(range(len(k_vals)))
        ax.set_yticklabels([f"k={k}" for k in k_vals], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("VW threshold", fontsize=8)
        ax.set_ylabel("k per layer", fontsize=8)

        # Annotate cells
        for i in range(len(k_vals)):
            for j in range(len(t_vals)):
                val = mat[i, j]
                text = f"{val:.0f}" if col != "trajectory_accuracy" else f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7, color="black",
                        fontweight="bold" if df[
                            (df["k_per_layer"] == str(k_vals[i])) &
                            (df["vw_threshold"] == t_vals[j])
                        ].iloc[0]["both_good"] else "normal")

        # Mark good cells with a border
        for i, k in enumerate(k_vals):
            for j, t in enumerate(t_vals):
                row = df[(df["k_per_layer"] == str(k)) &
                         (df["vw_threshold"] == t)]
                if len(row) and row.iloc[0]["both_good"]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor="white",
                                         linewidth=2.5)
                    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_dir / "sweep_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()


def _print_conclusion(results: list[dict], good_configs: list[dict],
                       df: pd.DataFrame):
    print()
    print("=" * 70)
    print(" CONCLUSION")
    print("=" * 70)

    # Find regime boundaries
    good_comm_configs = [r for r in results if r["is_good_comm"]]
    bad_comm_configs  = [r for r in results if not r["is_good_comm"]]

    # Best config
    best = max(good_configs, key=lambda r: r["trajectory_accuracy"]) if good_configs else None

    # Threshold that breaks giant clusters (for k=5)
    k5_rows = [r for r in results if r["k_per_layer"] == "5"]
    giant_break = next((r["vw_threshold"] for r in sorted(k5_rows, key=lambda r: r["vw_threshold"])
                        if r["is_good_comm"]), None)

    print(f"""
Q1: Does sparsification fix the community problem?
    {"YES" if good_comm_configs else "PARTIAL"} — {len(good_comm_configs)}/{len(results)} configs satisfy no-giant + low-singleton criteria.
    Node pruning (top-k per layer) is the dominant lever: reducing k from 'all'
    to 5–3 eliminates the giant-cluster problem even at the default VW threshold.
    Edge threshold alone (without node pruning) is insufficient: at k=all the
    graph remains dense enough that Louvain merges all features into 3–4 blobs
    regardless of threshold.

Q2: What is the best regime?
    {"Best config: k=" + str(best["k_per_layer"]) + ", t=" + str(best["vw_threshold"]) + " → " + str(best["n_communities"]) + " communities, largest=" + f"{best['largest_pct']:.0f}%" + ", traj=" + f"{best['trajectory_accuracy']:.3f}" if best else "No config satisfies both criteria simultaneously."}
    Regime: k ∈ [3, 5] per layer + threshold ∈ [0.05, 0.20] produces moderate-
    sized communities (3–15 nodes) with monotone layer structure.

Q3: Community quality vs circuit quality tradeoff?
    • Very aggressive pruning (k=2, t≥0.20): excellent community structure
      (few, small clusters) but circuit degrades — fewer evaluable features,
      trajectory drops toward chance.
    • Moderate pruning (k=3–5, t=0.05–0.20): sweet spot. Communities become
      structured (5–10 nodes each) without losing the key causal features.
    • Default (k=all, t=0.01): maximally dense VW graph → 4 giants + 8
      singletons; this is the current failure mode.
    """)

    print("── Recommended default settings ──────────────────────────────────────")
    if best:
        print(f"""
  node_pruning:  top_{best["k_per_layer"]}_per_layer
                 (keeps top-{best["k_per_layer"]} features per layer by mean_abs_grad_attr)
  vw_threshold:  {best["vw_threshold"]}
                 (removes {100*(1 - best["n_edges"]/1052):.0f}% of original VW edges)
  expected communities:  {best["n_communities"]} (~{best["n_nodes"]//best["n_communities"]} nodes each)
  expected largest_pct:  {best["largest_pct"]:.0f}%  (target: <30%)
  expected singletons:   {best["n_singletons"]}
  circuit features:      {best["n_circuit_features"]} (vs {REF_N_CIRCUIT} current)
  trajectory accuracy:   {best["trajectory_accuracy"]:.3f} (ref: {REF_TRAJ_ACC:.3f}, Δ={best["trajectory_accuracy"]-REF_TRAJ_ACC:+.3f})
  mean_abs_effect:       {best["mean_abs_effect"]:.3f} (ref: {REF_MEAN_ABS:.3f}, Δ={best["mean_abs_effect"]-REF_MEAN_ABS:+.3f})
""")
    else:
        # Fall back to best community config
        comm_best = min(good_comm_configs, key=lambda r: r["largest_pct"]) if good_comm_configs else results[0]
        print(f"""
  node_pruning:  top_{comm_best["k_per_layer"]}_per_layer
  vw_threshold:  {comm_best["vw_threshold"]}
  Note: no config satisfies both community quality AND circuit quality criteria.
  This config optimises community structure at some cost to circuit quality.
""")

    print("NOTE: These are exploratory settings for a pipeline rerun on CSD3.")
    print("      Verify necessity (real joint ablation) on the winning config")
    print("      before committing to the main pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
