"""
Post-hoc comparison of internal (v2) vs explicit (v3) candidate-selection experiments.

Uses only attribution graph JSON files — no transcoder feature files needed.
Can run immediately without syncing feature data from CSD3.

Compares:
  - Feature overlap by layer (Jaccard, shared feature IDs)
  - Node statistics (mean_attr, frequency, specificity) across experiments
  - Circuit feature overlap (if both circuit JSONs available)
  - Exclusive features: v2-only (possible candidate-retrieval stage) vs v3-only
    (possible explicit list-reading)

Hypothesis:
  Shared L22-L25 features  →  common final selection/output circuit
  v2-only L10-L21 features →  candidate pool construction / internal retrieval
  v3-only features         →  explicit option-reading / anchor on provided tokens

Usage:
  python scripts/44_explicit_vs_internal_candidate_comparison.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BEHAVIOUR_V2 = "physics_internal_candidate_selection_v2"
BEHAVIOUR_V3 = "physics_particle_candidate_selection_v3"
SPLIT        = "train"
LAYERS       = list(range(10, 26))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def get_graph_path(behaviour, split):
    base = Path("data/results/attribution_graphs")
    return base / behaviour / f"attribution_graph_{split}_n120_roleaware.json"


def get_circuit_path(behaviour, split):
    return Path("data/results") / f"circuits_{behaviour}_{split}_roleaware.json"


def load_graph_nodes(behaviour, split):
    """Returns {(layer, feat_idx): node_dict} for feature nodes."""
    path = get_graph_path(behaviour, split)
    if not path.exists():
        print(f"[WARN] Graph not found: {path}")
        return {}
    with open(path) as f:
        g = json.load(f)
    result = {}
    for n in g["nodes"]:
        if n.get("type") == "feature":
            result[(n["layer"], n["feature_idx"])] = n
    return result


def load_circuit_nodes(behaviour, split):
    """Returns set of (layer, feat_idx) tuples."""
    path = get_circuit_path(behaviour, split)
    if not path.exists():
        return set()
    with open(path) as f:
        c = json.load(f)
    result = set()
    for node_id in c.get("circuit", {}).get("feature_nodes", []):
        if "_F" in node_id and node_id.startswith("L"):
            l, f = node_id.split("_F")
            result.add((int(l[1:]), int(f)))
    return result


def load_causal_edges(behaviour, split):
    path = Path("data/results/causal_edges") / behaviour / f"causal_edges_{behaviour}_{split}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("edges", [])


# ─── Layer-level feature overlap ─────────────────────────────────────────────

def compute_layer_overlap(nodes_v2, nodes_v3):
    rows = []
    for layer in LAYERS:
        feats_v2 = {fidx for (l, fidx) in nodes_v2 if l == layer}
        feats_v3 = {fidx for (l, fidx) in nodes_v3 if l == layer}

        intersect = feats_v2 & feats_v3
        union     = feats_v2 | feats_v3
        jaccard   = len(intersect) / len(union) if union else float("nan")

        rows.append({
            "layer":        layer,
            "n_v2":         len(feats_v2),
            "n_v3":         len(feats_v3),
            "n_shared":     len(intersect),
            "n_v2_only":    len(feats_v2 - feats_v3),
            "n_v3_only":    len(feats_v3 - feats_v2),
            "jaccard":      jaccard,
            "shared_feats": sorted(intersect),
            "v2_only_feats": sorted(feats_v2 - feats_v3),
            "v3_only_feats": sorted(feats_v3 - feats_v2),
        })
    return pd.DataFrame(rows)


# ─── Node statistics comparison ───────────────────────────────────────────────

def compare_node_stats(nodes_v2, nodes_v3, layer_overlap_df):
    """Build a per-feature comparison table with stats from both experiments."""
    all_feats = set(nodes_v2.keys()) | set(nodes_v3.keys())
    rows = []
    for (layer, fidx) in sorted(all_feats):
        n2 = nodes_v2.get((layer, fidx))
        n3 = nodes_v3.get((layer, fidx))
        membership = (
            "shared"   if n2 and n3 else
            "v2_only"  if n2 else
            "v3_only"
        )
        row = {
            "feature_id":   f"L{layer}_F{fidx}",
            "layer":        layer,
            "feature_idx":  fidx,
            "membership":   membership,
        }
        for label, node in [("v2", n2), ("v3", n3)]:
            if node:
                row[f"{label}_mean_attr"]     = node.get("mean_grad_attr_conditional", float("nan"))
                row[f"{label}_mean_abs_attr"] = node.get("mean_abs_grad_attr_conditional", float("nan"))
                row[f"{label}_frequency"]     = node.get("frequency", float("nan"))
                row[f"{label}_specific_score"] = node.get("specific_score", float("nan"))
                row[f"{label}_mean_activation"] = node.get("mean_activation_conditional", float("nan"))
                row[f"{label}_position_role"]  = node.get("position_role", "")
                row[f"{label}_causal_status"]  = node.get("causal_status", "")
            else:
                for suffix in ["mean_attr", "mean_abs_attr", "frequency",
                               "specific_score", "mean_activation",
                               "position_role", "causal_status"]:
                    row[f"{label}_{suffix}"] = float("nan") if suffix != "position_role" and suffix != "causal_status" else ""
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Circuit overlap ──────────────────────────────────────────────────────────

def compare_circuits(circ_v2, circ_v3):
    shared  = circ_v2 & circ_v3
    v2_only = circ_v2 - circ_v3
    v3_only = circ_v3 - circ_v2
    union   = circ_v2 | circ_v3
    jaccard = len(shared) / len(union) if union else float("nan")

    # By layer
    rows = []
    for layer in LAYERS:
        c2 = {f for (l, f) in circ_v2 if l == layer}
        c3 = {f for (l, f) in circ_v3 if l == layer}
        i  = c2 & c3
        rows.append({
            "layer":     layer,
            "n_v2":      len(c2),
            "n_v3":      len(c3),
            "n_shared":  len(i),
            "n_v2_only": len(c2 - c3),
            "n_v3_only": len(c3 - c2),
            "jaccard":   len(i) / len(c2 | c3) if (c2 | c3) else float("nan"),
        })
    return {
        "n_v2":     len(circ_v2),
        "n_v3":     len(circ_v3),
        "n_shared": len(shared),
        "n_v2_only": len(v2_only),
        "n_v3_only": len(v3_only),
        "jaccard":  jaccard,
        "shared_features":  sorted(f"L{l}_F{f}" for l, f in shared),
        "v2_only_features": sorted(f"L{l}_F{f}" for l, f in v2_only),
        "v3_only_features": sorted(f"L{l}_F{f}" for l, f in v3_only),
        "by_layer": pd.DataFrame(rows),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def make_comparison_plot(layer_overlap_df, circuit_by_layer, output_dir):
    if not MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layers = layer_overlap_df["layer"].tolist()

    ax = axes[0]
    ax.bar(layers, layer_overlap_df["n_v2"], alpha=0.7, label="v2 (internal)", color="#1f77b4")
    ax.bar(layers, layer_overlap_df["n_v3"], alpha=0.5, label="v3 (explicit)", color="#ff7f0e",
           width=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of graph features")
    ax.set_title("Graph features by layer:\nInternal (v2) vs Explicit (v3)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(layers, layer_overlap_df["n_shared"],  label="shared",   color="steelblue", alpha=0.8)
    ax.bar(layers, layer_overlap_df["n_v2_only"], label="v2 only",  color="#2ca02c",   alpha=0.8,
           bottom=layer_overlap_df["n_shared"])
    ax.bar(layers, layer_overlap_df["n_v3_only"], label="v3 only",  color="#d62728",   alpha=0.8,
           bottom=layer_overlap_df["n_shared"] + layer_overlap_df["n_v2_only"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature membership by layer")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    jaccard = layer_overlap_df["jaccard"].fillna(0)
    ax.bar(layers, jaccard, color="mediumpurple", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Jaccard overlap: v2 vs v3 graph features")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_overlap_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output_dir}/feature_overlap_by_layer.png")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(layer_overlap_df, node_stats_df, circuit_summary, output_dir):
    lines = [
        "# Explicit vs Internal Candidate Comparison Report",
        f"## {BEHAVIOUR_V2} (internal) vs {BEHAVIOUR_V3} (explicit)",
        "",
        "## Hypothesis",
        "- **Shared L22-L25 features** → common final selection/output circuit",
        "- **v2-only L10-L21 features** → candidate pool construction / internal retrieval stage",
        "- **v3-only features** → explicit option-reading from prompt tokens",
        "",
    ]

    # Overall node overlap
    total = len(node_stats_df)
    n_shared   = (node_stats_df["membership"] == "shared").sum()
    n_v2_only  = (node_stats_df["membership"] == "v2_only").sum()
    n_v3_only  = (node_stats_df["membership"] == "v3_only").sum()

    lines += [
        "## 1. Graph Feature Overlap (all layers)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| v2 total graph features | {len([n for n in node_stats_df['membership'] if n in ['shared','v2_only']])} |",
        f"| v3 total graph features | {len([n for n in node_stats_df['membership'] if n in ['shared','v3_only']])} |",
        f"| Shared features | {n_shared} |",
        f"| v2-only features | {n_v2_only} |",
        f"| v3-only features | {n_v3_only} |",
        f"| Overall Jaccard | {n_shared / (n_shared + n_v2_only + n_v3_only):.3f} |",
        "",
    ]

    # Layer-by-layer table
    lines += [
        "## 2. Feature overlap by layer",
        "",
        "| Layer | v2 | v3 | shared | v2_only | v3_only | Jaccard |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in layer_overlap_df.sort_values("layer").iterrows():
        lines.append(
            f"| L{int(r['layer'])} | {int(r['n_v2'])} | {int(r['n_v3'])} | "
            f"{int(r['n_shared'])} | {int(r['n_v2_only'])} | {int(r['n_v3_only'])} | "
            f"{r['jaccard']:.3f} |"
        )
    lines += [""]

    # Circuit overlap
    if circuit_summary:
        v3_missing_note = " **(v3 circuit not synced locally — all v2 features appear as v2-only)**" \
            if circuit_summary.get("v3_missing") else ""
        lines += [
            f"## 3. Circuit Feature Overlap{v3_missing_note}",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| v2 circuit features | {circuit_summary['n_v2']} |",
            f"| v3 circuit features | {circuit_summary['n_v3']} |",
            f"| Shared circuit features | {circuit_summary['n_shared']} |",
            f"| v2-only circuit features | {circuit_summary['n_v2_only']} |",
            f"| v3-only circuit features | {circuit_summary['n_v3_only']} |",
            f"| Jaccard | {circuit_summary['jaccard']:.3f} |",
            "",
            "**Shared circuit features:**",
            ", ".join(circuit_summary["shared_features"]) or "none",
            "",
            "**v2-only circuit features** (possible internal retrieval):",
            ", ".join(circuit_summary["v2_only_features"]) or "none",
            "",
            "**v3-only circuit features** (possible explicit list-reading):",
            ", ".join(circuit_summary["v3_only_features"]) or "none",
            "",
        ]
        # By layer
        lines += [
            "### Circuit overlap by layer",
            "",
            "| Layer | v2 | v3 | shared | Jaccard |",
            "|---|---|---|---|---|",
        ]
        for _, r in circuit_summary["by_layer"].sort_values("layer").iterrows():
            if r["n_v2"] + r["n_v3"] > 0:
                lines.append(
                    f"| L{int(r['layer'])} | {int(r['n_v2'])} | {int(r['n_v3'])} | "
                    f"{int(r['n_shared'])} | {r['jaccard']:.3f} |"
                )
    lines += [""]

    # v2-only features by layer (potential retrieval stage)
    lines += ["## 4. v2-only features (potential candidate-retrieval stage)", ""]
    v2_only = node_stats_df[node_stats_df["membership"] == "v2_only"].copy()
    if len(v2_only):
        v2_only_sorted = v2_only.sort_values(["layer", "v2_specific_score"],
                                              ascending=[True, False])
        lines += ["| Feature | Layer | v2_specific_score | v2_frequency | v2_mean_attr | v2_position_role |"]
        lines += ["|---|---|---|---|---|---|"]
        for _, r in v2_only_sorted.head(25).iterrows():
            lines.append(
                f"| {r['feature_id']} | {r['layer']} | "
                f"{r['v2_specific_score']:.4f} | {r['v2_frequency']:.3f} | "
                f"{r['v2_mean_attr']:.3f} | {r['v2_position_role']} |"
            )
    lines += [""]

    # v3-only features (potential explicit list-reading)
    lines += ["## 5. v3-only features (potential explicit list-reading)", ""]
    v3_only = node_stats_df[node_stats_df["membership"] == "v3_only"].copy()
    if len(v3_only):
        v3_only_sorted = v3_only.sort_values(["layer", "v3_specific_score"],
                                              ascending=[True, False])
        lines += ["| Feature | Layer | v3_specific_score | v3_frequency | v3_mean_attr | v3_position_role |"]
        lines += ["|---|---|---|---|---|---|"]
        for _, r in v3_only_sorted.head(25).iterrows():
            lines.append(
                f"| {r['feature_id']} | {r['layer']} | "
                f"{r['v3_specific_score']:.4f} | {r['v3_frequency']:.3f} | "
                f"{r['v3_mean_attr']:.3f} | {r['v3_position_role']} |"
            )
    lines += [""]

    # Shared features
    lines += ["## 6. Shared features (common selection circuit)", ""]
    shared = node_stats_df[node_stats_df["membership"] == "shared"].copy()
    if len(shared):
        shared_sorted = shared.sort_values(["layer", "v2_specific_score"],
                                            ascending=[True, False])
        lines += ["| Feature | Layer | v2_attr | v3_attr | v2_freq | v3_freq | v2_role | v3_role |"]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for _, r in shared_sorted.head(25).iterrows():
            lines.append(
                f"| {r['feature_id']} | {r['layer']} | "
                f"{r['v2_mean_attr']:.3f} | {r['v3_mean_attr']:.3f} | "
                f"{r['v2_frequency']:.3f} | {r['v3_frequency']:.3f} | "
                f"{r['v2_position_role']} | {r['v3_position_role']} |"
            )
    lines += [""]

    # Interpretation
    early_v2 = v2_only[v2_only["layer"] <= 21]
    late_shared = shared[shared["layer"] >= 22] if len(shared) else pd.DataFrame()
    lines += [
        "## 7. Interpretation",
        "",
        f"- **Early (L10-L21) v2-only features**: {len(early_v2)} — these may encode the "
        f"internal candidate pool construction process absent in the explicit case",
        f"- **Late (L22-L25) shared features**: {len(late_shared)} — the final decision "
        f"mechanism appears shared between explicit and internal selection",
    ]
    if len(early_v2) >= 3 and len(late_shared) >= 2:
        interp = (
            "**TWO-STAGE INTERPRETATION SUPPORTED**: "
            "v2 has additional early-layer candidate retrieval features "
            "while sharing late-layer selection features with v3"
        )
    elif len(late_shared) >= 3 and len(early_v2) < 2:
        interp = (
            "**SHARED CIRCUIT ONLY**: both experiments use the same late-layer circuit; "
            "no evidence for a distinct internal retrieval stage"
        )
    else:
        interp = (
            "**AMBIGUOUS**: overlap patterns are mixed — "
            "manual feature-level investigation recommended"
        )
    lines += [f"- Overall: {interp}", ""]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "explicit_vs_internal_report.md").write_text("\n".join(lines))
    print(f"Report written to {output_dir}/explicit_vs_internal_report.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour_v2", default=BEHAVIOUR_V2)
    ap.add_argument("--behaviour_v3", default=BEHAVIOUR_V3)
    ap.add_argument("--split",        default=SPLIT)
    ap.add_argument("--no_plots",     action="store_true")
    args = ap.parse_args()

    output_dir = Path("data/results/internal_candidate_analysis/explicit_vs_internal")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graphs
    print(f"Loading v2 graph: {args.behaviour_v2}")
    nodes_v2 = load_graph_nodes(args.behaviour_v2, args.split)
    print(f"  {len(nodes_v2)} feature nodes")

    print(f"Loading v3 graph: {args.behaviour_v3}")
    nodes_v3 = load_graph_nodes(args.behaviour_v3, args.split)
    print(f"  {len(nodes_v3)} feature nodes")

    if not nodes_v2 and not nodes_v3:
        print("[ERROR] Neither graph found — cannot compare")
        return

    # Layer overlap
    layer_overlap = compute_layer_overlap(nodes_v2, nodes_v3)
    layer_overlap.to_csv(output_dir / "feature_overlap_by_layer.csv", index=False)

    # Node stats
    node_stats = compare_node_stats(nodes_v2, nodes_v3, layer_overlap)
    node_stats.to_csv(output_dir / "candidate_feature_overlap.csv", index=False)

    # Circuit overlap
    circ_v2 = load_circuit_nodes(args.behaviour_v2, args.split)
    circ_v3 = load_circuit_nodes(args.behaviour_v3, args.split)
    circuit_summary = None
    v3_circuit_path = get_circuit_path(args.behaviour_v3, args.split)
    if circ_v2 and not circ_v3:
        print(f"[NOTE] v3 circuit JSON not found locally ({v3_circuit_path})")
        print(f"       Sync from CSD3 to enable circuit comparison.")
        print(f"       v2 circuit: {len(circ_v2)} features")
        circ_v3 = set()  # comparison will show all v2 as v2-only
        circuit_summary = compare_circuits(circ_v2, circ_v3)
        circuit_summary["v3_missing"] = True
        circuit_summary["by_layer"].to_csv(output_dir / "circuit_overlap.csv", index=False)
    elif circ_v2 or circ_v3:
        print(f"Circuit features: v2={len(circ_v2)}, v3={len(circ_v3)}")
        circuit_summary = compare_circuits(circ_v2, circ_v3)
        circuit_summary["by_layer"].to_csv(output_dir / "circuit_overlap.csv", index=False)
    else:
        print("[WARN] No circuit JSON files found")

    # Plots
    if not args.no_plots:
        make_comparison_plot(layer_overlap, circuit_summary, output_dir)

    # Report
    write_report(layer_overlap, node_stats, circuit_summary, output_dir)

    # Console summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("\nLayer | v2 | v3 | shared | Jaccard")
    for _, r in layer_overlap.sort_values("layer").iterrows():
        bar = "▓" * int(r["jaccard"] * 10) if not np.isnan(r["jaccard"]) else ""
        print(f"  L{int(r['layer']):2d} | {int(r['n_v2'])} | {int(r['n_v3'])} | "
              f"{int(r['n_shared'])} | {r['jaccard']:.3f} {bar}")

    if circuit_summary:
        print(f"\nCircuit Jaccard: {circuit_summary['jaccard']:.3f}")
        print(f"  Shared: {circuit_summary['shared_features']}")
        print(f"  v2-only: {circuit_summary['v2_only_features'][:5]}")
        print(f"  v3-only: {circuit_summary['v3_only_features'][:5]}")


if __name__ == "__main__":
    main()
