"""
diag_build_v2_features.py

Build v2-compatible feature sets for H1/H2/H3/H4 scripts (53, 54):
  1. circuit_features_for_h1.json — top features by CAUSAL effect (mean_abs_effect),
     not graph edge weight (v1→v2 lesson: attribution sign ≠ causal direction)
  2. cluster_semantics_v2.json — compatible with script 53 (h2_pairs mode)
     and script 54 (cluster ablation), built from runD_v2 outputs

Source data:
  data/analysis/runD_v2/cluster_semantics/cluster_feature_summary.csv
  data/analysis/runD_v2/results_summary_ru.md (cluster names + polarity)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

# v2 cluster polarity from results_summary_ru.md §2
V2_CLUSTER_META = {
    0:  {"name": "C0 L10 α-supporting",        "polarity": "alpha", "orient_delta": -0.483},
    1:  {"name": "C1 L11 β-supporting weak",   "polarity": "beta",  "orient_delta": +0.161},
    2:  {"name": "C2 L12 β-supporting weak",   "polarity": "beta",  "orient_delta": +0.059},
    3:  {"name": "C3 L13 α-supporting",        "polarity": "alpha", "orient_delta": -0.609},
    4:  {"name": "C4 L14+L17 β-supporting",    "polarity": "beta",  "orient_delta": +0.178},
    5:  {"name": "C5 L15 α-supporting weak",   "polarity": "alpha", "orient_delta": -0.122},
    6:  {"name": "C6 L16 β-supporting",        "polarity": "beta",  "orient_delta": +0.417},
    7:  {"name": "C7 L25 β-supporting",        "polarity": "beta",  "orient_delta": +0.445},
    8:  {"name": "C8 L18 α-supporting STRONGEST", "polarity": "alpha", "orient_delta": -0.896},
    9:  {"name": "C9 L19 α-supporting",        "polarity": "alpha", "orient_delta": -0.410},
    10: {"name": "C10 L20 β-supporting",       "polarity": "beta",  "orient_delta": +0.244},
    11: {"name": "C11 L21 β-supporting",       "polarity": "beta",  "orient_delta": +0.709},
    12: {"name": "C12 L22+L23 β-supporting",   "polarity": "beta",  "orient_delta": +0.734},
    13: {"name": "C13 L24 β-supporting STRONGEST", "polarity": "beta", "orient_delta": +1.334},
}


def main():
    fs = pd.read_csv(ROOT / "data/analysis/runD_v2/cluster_semantics/cluster_feature_summary.csv")
    out_dir = ROOT / "data/analysis/iia_failure_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = fs.dropna(subset=["layer"])
    fs["layer"] = fs["layer"].astype(int)

    # ── 1. Top-K features by mean_abs_effect (causal magnitude) ─────────────
    fs_sorted = fs.sort_values("mean_abs_effect", ascending=False)

    circuit_features = {
        "source":            "runD_v2 cluster_feature_summary.csv",
        "ranking_metric":    "mean_abs_effect (causal effect on logit)",
        "v2_note":           "v1 dropped negative-attribution features; v2 includes them. Top features include L24/L18 strong polar clusters.",
        "top_by_causal_effect": [],
        "top_per_cluster":   {},
    }

    for _, row in fs_sorted.head(30).iterrows():
        circuit_features["top_by_causal_effect"].append({
            "feature_id":    row["feature_id"],
            "layer":         int(row["layer"]),
            "cluster_id":    int(row["cluster_id"]),
            "mean_abs":      float(row["mean_abs_effect"]),
            "mean_signed":   float(row["mean_signed_effect"]),
        })

    # Top 3 per cluster (for stratified pairs)
    for cid, sub in fs.groupby("cluster_id"):
        top = sub.nlargest(3, "mean_abs_effect")
        circuit_features["top_per_cluster"][int(cid)] = {
            "name": V2_CLUSTER_META[int(cid)]["name"],
            "polarity": V2_CLUSTER_META[int(cid)]["polarity"],
            "orient_delta": V2_CLUSTER_META[int(cid)]["orient_delta"],
            "features": [
                {"feature_id": r["feature_id"], "layer": int(r["layer"]),
                 "mean_abs": float(r["mean_abs_effect"])}
                for _, r in top.iterrows()
            ],
        }

    with open(out_dir / "circuit_features_for_h1.json", "w") as f:
        json.dump(circuit_features, f, indent=2)
    print(f"Saved → circuit_features_for_h1.json")
    print(f"  Top 5 features by causal effect:")
    for f_ in circuit_features["top_by_causal_effect"][:5]:
        print(f"    {f_['feature_id']:>20s}  L{f_['layer']:2d}  C{f_['cluster_id']:>2d}  "
              f"|effect|={f_['mean_abs']:.3f}")

    # ── 2. cluster_semantics_v2.json (compatible with scripts 52/53/54) ─────
    clusters_out = []
    for cid, sub in fs.groupby("cluster_id"):
        layers = sorted(sub["layer"].unique().tolist())
        clusters_out.append({
            "id":          int(cid),
            "name":        V2_CLUSTER_META[int(cid)]["name"],
            "polarity":    V2_CLUSTER_META[int(cid)]["polarity"],
            "orient_delta":V2_CLUSTER_META[int(cid)]["orient_delta"],
            "n_features":  len(sub),
            "layer_min":   int(min(layers)),
            "layer_max":   int(max(layers)),
            "layers":      [int(l) for l in layers],
            "features":    [
                {"id": r["feature_id"],
                 "layer": int(r["layer"]),
                 "mean_abs": float(r["mean_abs_effect"]),
                 "mean_signed": float(r["mean_signed_effect"])}
                for _, r in sub.iterrows()
            ],
        })

    semantics = {
        "meta": {
            "source": "runD_v2",
            "method": "coimp_louvain k=14",
            "n_features": int(fs.shape[0]),
            "n_clusters": len(clusters_out),
        },
        "clusters": clusters_out,
    }
    with open(out_dir / "cluster_semantics_v2.json", "w") as f:
        json.dump(semantics, f, indent=2)
    print(f"\nSaved → cluster_semantics_v2.json ({len(clusters_out)} clusters)")

    # Print α-supporters vs β-supporters for H2 pairing
    print("\n=== v2 cluster polarity ===")
    def fmt(c):
        L = f"L{c['layer_min']}" + ("+" if c['layer_min'] != c['layer_max'] else "")
        return f"C{c['id']} {L} ({c['orient_delta']:+.2f})"
    a_clust = [fmt(c) for c in clusters_out if c["polarity"] == "alpha"]
    b_clust = [fmt(c) for c in clusters_out if c["polarity"] == "beta"]
    print(f"  α-supporters: {a_clust}")
    print(f"  β-supporters: {b_clust}")


if __name__ == "__main__":
    main()
