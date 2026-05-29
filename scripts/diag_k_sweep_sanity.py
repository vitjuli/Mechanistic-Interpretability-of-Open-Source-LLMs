"""
diag_k_sweep_sanity.py

Recut agglomerative clustering on W_coimportance at k=15, 16, 17, 18 and check:
  1. Do problematic mergers (L14+L17, L22+L23) split?
  2. Do strong causal clusters (L18=C8, L24=C13 in k=14) stay intact?
  3. What's the polarity (orient_delta) of new sub-clusters?

If problematic clusters split with clean polarity AND strong clusters preserved,
we have evidence that k=16 (or 17) gives cleaner structure with minimal cost.

Reads:
  data/analysis/runD_v2/clustering_full/W_coimportance.npy
  data/analysis/runD_v2/clustering_full/feat_ids.json
  data/analysis/runD_v2/clustering_full/cluster_labels.csv  (agglo_coimp_k14)
  data/analysis/runD_v2/cluster_semantics/cluster_feature_summary.csv (for mean_signed_effect)

Outputs:
  data/analysis/iia_failure_diagnosis/k_sweep_sanity.json
  stdout: per-k summary
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def cluster_polarity(cluster_feats, fs):
    """Compute α/β polarity for a set of feature_ids."""
    sub = fs[fs["feature_id"].isin(cluster_feats)]
    if sub.empty:
        return None
    n = len(sub)
    mean_signed = sub["mean_signed_effect"].mean()
    n_neg = (sub["mean_signed_effect"] < 0).sum()
    n_pos = (sub["mean_signed_effect"] > 0).sum()
    layers = sub["layer"].dropna().astype(int).unique().tolist()
    return {
        "n": int(n),
        "layers": sorted(layers),
        "mean_signed": float(mean_signed),
        "n_neg_sign": int(n_neg),
        "n_pos_sign": int(n_pos),
        "polarity": "alpha" if mean_signed < -0.05 else (
                    "beta"  if mean_signed > +0.05 else "mixed"),
        "purity": float(max(n_neg, n_pos) / n) if n > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering_dir",
                    default="data/analysis/runD_v2/clustering_full")
    ap.add_argument("--semantics_csv",
                    default="data/analysis/runD_v2/cluster_semantics/cluster_feature_summary.csv")
    ap.add_argument("--k_values", nargs="+", type=int, default=[14, 15, 16, 17, 18])
    ap.add_argument("--out",
                    default="data/analysis/iia_failure_diagnosis/k_sweep_sanity.json")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    cdir = root / args.clustering_dir

    # Load matrix and feature IDs
    W = np.load(cdir / "W_coimportance.npy")
    feat_ids = json.load(open(cdir / "feat_ids.json"))
    print(f"W shape: {W.shape}  features: {len(feat_ids)}")

    # Convert co-importance to distance: D = 1 - W (clipped to [0,1])
    D = np.clip(1.0 - W, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    # Ensure symmetric for scipy
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)

    # Average linkage (same as agglo_coimp method)
    Z = linkage(condensed, method="average")
    print(f"Linkage matrix: {Z.shape[0]} merges")

    # Load polarity info per feature
    fs = pd.read_csv(root / args.semantics_csv)

    # Load existing k=14 labels for comparison
    labels_csv = pd.read_csv(cdir / "cluster_labels.csv")
    k14_col = labels_csv.set_index("feature_id")["agglo_coimp_k14"].to_dict()

    # Build feature→layer map from existing summary
    feat_to_layer = dict(zip(fs["feature_id"], fs["layer"]))

    results = {}

    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        print(f"{'='*60}")

        labels = fcluster(Z, t=k, criterion="maxclust")
        clusters = defaultdict(list)
        for fid, lab in zip(feat_ids, labels):
            clusters[int(lab)].append(fid)

        cluster_info = []
        for cid, feats in clusters.items():
            pol = cluster_polarity(feats, fs)
            if pol is None:
                continue
            cluster_info.append({"cluster_id": cid, **pol, "features": feats})

        # Sort by mean_signed (α first, then β)
        cluster_info.sort(key=lambda c: c["mean_signed"])

        print(f"\n  {'cid':>4}  {'n':>3}  {'layers':>15}  {'pol':>6}  {'mean_signed':>11}  "
              f"{'purity':>6}")
        for c in cluster_info:
            layers_str = ",".join(str(l) for l in c["layers"][:4])
            if len(c["layers"]) > 4:
                layers_str += "+"
            print(f"  C{c['cluster_id']:>3}  {c['n']:>3}  {layers_str:>15}  "
                  f"{c['polarity']:>6}  {c['mean_signed']:>+11.4f}  {c['purity']:>6.2f}")

        # ── Check problematic mergers ───────────────────────────────────────
        # L14+L17 problematic at k=14: which clusters contain L14 & L17 features?
        L14_feats = [fid for fid, lay in feat_to_layer.items() if lay == 14]
        L17_feats = [fid for fid, lay in feat_to_layer.items() if lay == 17]
        L22_feats = [fid for fid, lay in feat_to_layer.items() if lay == 22]
        L23_feats = [fid for fid, lay in feat_to_layer.items() if lay == 23]
        L18_feats = [fid for fid, lay in feat_to_layer.items() if lay == 18]
        L24_feats = [fid for fid, lay in feat_to_layer.items() if lay == 24]

        def cluster_of(fid):
            for c in cluster_info:
                if fid in c["features"]:
                    return c["cluster_id"]
            return None

        def split_check(group_a, group_b, name_a, name_b):
            ca = {cluster_of(f) for f in group_a} - {None}
            cb = {cluster_of(f) for f in group_b} - {None}
            both = ca & cb
            return {
                "in_a": sorted(ca), "in_b": sorted(cb),
                "shared": sorted(both),
                "split": len(both) == 0,
                "name": f"{name_a} vs {name_b}",
            }

        l14_l17 = split_check(L14_feats, L17_feats, "L14", "L17")
        l22_l23 = split_check(L22_feats, L23_feats, "L22", "L23")

        print(f"\n  L14 vs L17: {'SPLIT ✓' if l14_l17['split'] else 'STILL MERGED ✗'}  "
              f"L14 in {l14_l17['in_a']}, L17 in {l14_l17['in_b']}")
        print(f"  L22 vs L23: {'SPLIT ✓' if l22_l23['split'] else 'STILL MERGED ✗'}  "
              f"L22 in {l22_l23['in_a']}, L23 in {l22_l23['in_b']}")

        # ── Strong clusters intact? ─────────────────────────────────────────
        l18_clusters = {cluster_of(f) for f in L18_feats} - {None}
        l24_clusters = {cluster_of(f) for f in L24_feats} - {None}
        print(f"  L18 (k=14: C8 α STRONGEST): now in {len(l18_clusters)} cluster(s) — "
              f"{'INTACT ✓' if len(l18_clusters) == 1 else 'SPLIT ⚠️'}")
        print(f"  L24 (k=14: C13 β STRONGEST): now in {len(l24_clusters)} cluster(s) — "
              f"{'INTACT ✓' if len(l24_clusters) == 1 else 'SPLIT ⚠️'}")

        results[f"k_{k}"] = {
            "n_clusters": len(cluster_info),
            "clusters":   [{k_: v for k_, v in c.items() if k_ != "features"}
                           for c in cluster_info],
            "L14_L17_split": l14_l17["split"],
            "L22_L23_split": l22_l23["split"],
            "L18_intact":    len(l18_clusters) == 1,
            "L24_intact":    len(l24_clusters) == 1,
            "n_alpha":  sum(1 for c in cluster_info if c["polarity"] == "alpha"),
            "n_beta":   sum(1 for c in cluster_info if c["polarity"] == "beta"),
            "n_mixed":  sum(1 for c in cluster_info if c["polarity"] == "mixed"),
        }

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"{'k':>4}  {'#clusters':>10}  {'L14|L17':>9}  {'L22|L23':>9}  "
          f"{'L18 ok':>7}  {'L24 ok':>7}  {'#α':>3}  {'#β':>3}  {'#mix':>5}")
    for k in args.k_values:
        r = results[f"k_{k}"]
        print(f"  {k:>2}  {r['n_clusters']:>10}  {'✓' if r['L14_L17_split'] else '✗':>9}  "
              f"{'✓' if r['L22_L23_split'] else '✗':>9}  "
              f"{'✓' if r['L18_intact'] else '✗':>7}  "
              f"{'✓' if r['L24_intact'] else '✗':>7}  "
              f"{r['n_alpha']:>3}  {r['n_beta']:>3}  {r['n_mixed']:>5}")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
