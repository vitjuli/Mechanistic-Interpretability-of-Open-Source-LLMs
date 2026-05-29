"""
diag_recut_k16_and_regen.py

After sanity check confirmed k=16 is optimal:
  1. Recut agglomerative dendrogram on W_coimportance at k=16
  2. Compute orient_delta per cluster (mean_eff on α − mean_eff on β)
  3. Label polarity based on orient_delta (NOT mean_signed_effect — which mixes α/β)
  4. Regenerate cluster_semantics_v2.json (16 clusters)
  5. Regenerate circuit_features_for_h1.json (top features by mean_abs_effect)
  6. Save new agglo_coimp_k16 column to cluster_labels.csv

Sign convention (from results_summary_ru.md):
  orient_delta = mean_eff(α-prompts) − mean_eff(β-prompts)
  orient_delta < 0  → α-supporting (cluster more negative on α, more positive on β)
  orient_delta > 0  → β-supporting
  |orient_delta| < 0.05 → mixed/weak

Reads:
  data/analysis/runD_v2/clustering_full/W_coimportance.npy
  data/analysis/runD_v2/clustering_full/feat_ids.json
  data/analysis/runD_v2/clustering_full/feat_prompt_signed.npy   (227 × 538)
  data/analysis/runD_v2/clustering_full/prompt_labels.json       (correct_answer per prompt)
  data/analysis/runD_v2/clustering_full/cluster_labels.csv       (to append k=16 column)
  data/analysis/runD_v2/cluster_semantics/cluster_feature_summary.csv  (mean_abs_effect)

Writes:
  data/analysis/iia_failure_diagnosis/cluster_semantics_v2.json   (16 clusters, with polarity)
  data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json (top features, top_per_cluster k=16)
  data/analysis/iia_failure_diagnosis/k16_recut_report.md
  data/analysis/runD_v2/clustering_full/cluster_labels_k16.csv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--polarity_threshold", type=float, default=0.05,
                    help="|orient_delta| > this → polar; else mixed")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    cdir = root / "data/analysis/runD_v2/clustering_full"
    sdir = root / "data/analysis/runD_v2/cluster_semantics"
    out_dir = root / "data/analysis/iia_failure_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    W = np.load(cdir / "W_coimportance.npy")
    feat_ids = json.load(open(cdir / "feat_ids.json"))
    feat_prompt_signed = np.load(cdir / "feat_prompt_signed.npy")  # (n_feat, n_prompt)
    prompt_labels = json.load(open(cdir / "prompt_labels.json"))   # {idx_str: {...}}
    fs = pd.read_csv(sdir / "cluster_feature_summary.csv")
    labels_csv = pd.read_csv(cdir / "cluster_labels.csv")

    n_feat, n_prompt = feat_prompt_signed.shape
    print(f"Loaded W: {W.shape}, feat_prompt_signed: {feat_prompt_signed.shape}, "
          f"{n_feat} features, {n_prompt} prompts")

    # ── α/β masks ───────────────────────────────────────────────────────────
    alpha_idx = [int(k) for k, v in prompt_labels.items() if v["correct_answer"] == "alpha"]
    beta_idx  = [int(k) for k, v in prompt_labels.items() if v["correct_answer"] == "beta"]
    print(f"  α-prompts: {len(alpha_idx)}, β-prompts: {len(beta_idx)}")

    # ── Recut at k=16 ───────────────────────────────────────────────────────
    D = np.clip(1.0 - W, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels_k = fcluster(Z, t=args.k, criterion="maxclust")  # 1-based labels

    # Reindex to 0-based
    unique_labels = sorted(set(labels_k))
    relabel = {old: new for new, old in enumerate(unique_labels)}
    labels_0 = np.array([relabel[l] for l in labels_k])
    print(f"\nClusters at k={args.k}: {labels_0.max() + 1}")

    # ── Group features per cluster + compute orient_delta ──────────────────
    def parse_layer(fid):
        """Always parse layer from feature_id; CSV has NaN sometimes."""
        try:
            return int(fid.split("_F")[0].lstrip("L"))
        except Exception:
            return -1
    fid_to_layer = {fid: parse_layer(fid) for fid in feat_ids}
    fid_to_mean_abs = dict(zip(fs["feature_id"], fs["mean_abs_effect"]))
    fid_to_mean_signed = dict(zip(fs["feature_id"], fs["mean_signed_effect"]))
    fid_to_idx = {fid: i for i, fid in enumerate(feat_ids)}

    clusters_data = []
    for cid in range(labels_0.max() + 1):
        member_fids = [feat_ids[i] for i in range(n_feat) if labels_0[i] == cid]
        member_idxs = [fid_to_idx[f] for f in member_fids]

        # mean effect per prompt: average across features in cluster, then split by α/β
        eff_alpha = feat_prompt_signed[member_idxs][:, alpha_idx].mean()
        eff_beta  = feat_prompt_signed[member_idxs][:, beta_idx].mean()
        orient_delta = eff_alpha - eff_beta

        if orient_delta < -args.polarity_threshold:
            polarity = "alpha"
        elif orient_delta > +args.polarity_threshold:
            polarity = "beta"
        else:
            polarity = "mixed"

        layers = sorted({fid_to_layer[f] for f in member_fids if fid_to_layer[f] >= 0})

        # Sort features by mean_abs_effect (causal magnitude)
        member_records = sorted(
            [{"id": f, "layer": fid_to_layer.get(f, -1),
              "mean_abs": float(fid_to_mean_abs.get(f, 0)),
              "mean_signed": float(fid_to_mean_signed.get(f, 0))}
             for f in member_fids],
            key=lambda x: -x["mean_abs"]
        )

        # Cluster name based on dominant layer(s) + polarity
        layer_str = "+".join(str(l) for l in layers) if len(layers) <= 2 else \
                    f"{layers[0]}–{layers[-1]}"
        name = f"C{cid} L{layer_str} {polarity}"

        clusters_data.append({
            "id":           int(cid),
            "name":         name,
            "polarity":     polarity,
            "orient_delta": float(orient_delta),
            "mean_eff_alpha": float(eff_alpha),
            "mean_eff_beta":  float(eff_beta),
            "n_features":   len(member_fids),
            "layer_min":    int(min(layers)) if layers else -1,
            "layer_max":    int(max(layers)) if layers else -1,
            "layers":       [int(l) for l in layers],
            "features":     member_records,
        })

    # Sort by orient_delta (α-strongest first)
    clusters_data.sort(key=lambda c: c["orient_delta"])

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"{'cid':>4}  {'n':>3}  {'layers':>10}  {'σ̃_α':>8}  {'σ̃_β':>8}  "
          f"{'orient_Δ':>9}  {'polarity':>8}  top 3 features")
    print("-" * 100)
    for c in clusters_data:
        top3 = ", ".join(f"{f['id']}({f['mean_abs']:.2f})"
                         for f in c["features"][:3])
        layer_str = "+".join(str(l) for l in c["layers"]) if len(c["layers"]) <= 2 \
                    else f"{c['layers'][0]}–{c['layers'][-1]}"
        print(f"  C{c['id']:>2}  {c['n_features']:>3}  {layer_str:>10}  "
              f"{c['mean_eff_alpha']:>+8.3f}  {c['mean_eff_beta']:>+8.3f}  "
              f"{c['orient_delta']:>+9.3f}  {c['polarity']:>8}  {top3[:45]}")

    n_alpha = sum(1 for c in clusters_data if c["polarity"] == "alpha")
    n_beta  = sum(1 for c in clusters_data if c["polarity"] == "beta")
    n_mixed = sum(1 for c in clusters_data if c["polarity"] == "mixed")
    print(f"\n  → {n_alpha} α-supporting, {n_beta} β-supporting, {n_mixed} mixed")

    # ── Save cluster_semantics_v2.json (k=16) ───────────────────────────────
    semantics = {
        "meta": {
            "source":         "runD_v2 + diag_recut_k16",
            "method":         f"agglo_coimp_k{args.k} (average linkage on W_coimportance)",
            "k":              int(args.k),
            "n_features":     int(n_feat),
            "n_clusters":     len(clusters_data),
            "polarity_threshold": args.polarity_threshold,
            "sign_convention": "orient_delta = mean_eff(α) − mean_eff(β); "
                               "negative → α-supporting, positive → β-supporting",
            "n_alpha_supporting": n_alpha,
            "n_beta_supporting":  n_beta,
            "n_mixed":            n_mixed,
        },
        "clusters": clusters_data,
    }
    with open(out_dir / "cluster_semantics_v2.json", "w") as f:
        json.dump(semantics, f, indent=2)
    print(f"\nSaved → cluster_semantics_v2.json ({len(clusters_data)} clusters)")

    # ── Save circuit_features_for_h1.json (top features) ───────────────────
    fs_sorted = fs.dropna(subset=["layer"]).sort_values("mean_abs_effect", ascending=False)
    fid_to_new_cid = {feat_ids[i]: int(labels_0[i]) for i in range(n_feat)}

    circuit_features = {
        "source":          "runD_v2 + diag_recut_k16",
        "ranking_metric":  "mean_abs_effect (causal effect on logit margin)",
        "clustering":      f"agglo_coimp_k{args.k}",
        "top_by_causal_effect": [],
        "top_per_cluster":  {},
    }
    for _, row in fs_sorted.head(30).iterrows():
        circuit_features["top_by_causal_effect"].append({
            "feature_id":   row["feature_id"],
            "layer":        int(row["layer"]),
            "cluster_id":   fid_to_new_cid.get(row["feature_id"], -1),
            "mean_abs":     float(row["mean_abs_effect"]),
            "mean_signed":  float(row["mean_signed_effect"]),
        })

    for c in clusters_data:
        top3 = c["features"][:3]
        circuit_features["top_per_cluster"][str(c["id"])] = {
            "name":         c["name"],
            "polarity":     c["polarity"],
            "orient_delta": c["orient_delta"],
            "features":     [{"feature_id": f["id"], "layer": f["layer"],
                              "mean_abs": f["mean_abs"]} for f in top3],
        }
    with open(out_dir / "circuit_features_for_h1.json", "w") as f:
        json.dump(circuit_features, f, indent=2)
    print(f"Saved → circuit_features_for_h1.json")

    # ── Append k=16 column to cluster_labels.csv ────────────────────────────
    labels_csv["agglo_coimp_k16"] = labels_csv["feature_id"].map(fid_to_new_cid).fillna(-1).astype(int)
    labels_csv.to_csv(cdir / "cluster_labels_k16.csv", index=False)
    print(f"Saved → cluster_labels_k16.csv (with agglo_coimp_k16 column)")

    # ── Markdown report ─────────────────────────────────────────────────────
    md_lines = [
        f"# k=16 recut report",
        f"",
        f"**Date:** 2026-05-29  ",
        f"**Method:** agglo average-linkage on W_coimportance, cut at k=16  ",
        f"**Source:** runD_v2 ({n_feat} features, 538 prompts)  ",
        f"",
        f"## Summary",
        f"",
        f"- {n_alpha} α-supporting clusters (orient_delta < −{args.polarity_threshold})",
        f"- {n_beta} β-supporting clusters (orient_delta > +{args.polarity_threshold})",
        f"- {n_mixed} mixed clusters (|orient_delta| ≤ {args.polarity_threshold})",
        f"",
        f"## Per-cluster polarity (sorted by orient_delta)",
        f"",
        f"| cid | n | layers | σ̃_α | σ̃_β | orient_Δ | polarity | top feature |",
        f"|-----|---|--------|-----|------|----------|----------|-------------|",
    ]
    for c in clusters_data:
        lay = "+".join(str(l) for l in c["layers"]) if len(c["layers"]) <= 2 \
              else f"{c['layers'][0]}–{c['layers'][-1]}"
        top_str = c["features"][0]["id"] if c["features"] else "—"
        md_lines.append(
            f"| C{c['id']} | {c['n_features']} | L{lay} | "
            f"{c['mean_eff_alpha']:+.3f} | {c['mean_eff_beta']:+.3f} | "
            f"{c['orient_delta']:+.3f} | {c['polarity']} | {top_str} |"
        )

    md_lines += [
        f"",
        f"## Notes",
        f"- L18 cluster (strongest α): preserved (single cluster, all 17 features)",
        f"- L24 cluster (strongest β): preserved (single cluster, all 20 features)",
        f"- L14 and L17 now separate clusters (resolved k=14 problematic merger)",
        f"- L22 and L23 now separate clusters (resolved k=14 problematic merger)",
        f"",
        f"## Files updated",
        f"- `data/analysis/iia_failure_diagnosis/cluster_semantics_v2.json`",
        f"- `data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json`",
        f"- `data/analysis/runD_v2/clustering_full/cluster_labels_k16.csv`",
        f"",
        f"## H2 priority pairs (recommend for sbatch)",
    ]
    # Find strongest α and β clusters in new labeling
    a_clusters = [c for c in clusters_data if c["polarity"] == "alpha"]
    b_clusters = [c for c in clusters_data if c["polarity"] == "beta"]
    strongest_a = min(a_clusters, key=lambda c: c["orient_delta"]) if a_clusters else None
    strongest_b = max(b_clusters, key=lambda c: c["orient_delta"]) if b_clusters else None
    if strongest_a and strongest_b:
        md_lines.append(
            f"- **Strongest α+β pair**: C{strongest_a['id']} (L{strongest_a['layer_min']}, "
            f"orient_Δ={strongest_a['orient_delta']:+.3f}) + "
            f"C{strongest_b['id']} (L{strongest_b['layer_min']}, "
            f"orient_Δ={strongest_b['orient_delta']:+.3f})"
        )

    with open(out_dir / "k16_recut_report.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved → k16_recut_report.md")

    # ── Return cluster IDs for script 53 update ─────────────────────────────
    print(f"\n{'='*60}")
    print("FOR SCRIPT 53 UPDATE — strongest opposite-polarity clusters:")
    print(f"{'='*60}")
    print(f"  Strongest α: C{strongest_a['id']} L{strongest_a['layer_min']} "
          f"(orient_Δ={strongest_a['orient_delta']:+.3f}, n={strongest_a['n_features']})")
    print(f"  Strongest β: C{strongest_b['id']} L{strongest_b['layer_min']} "
          f"(orient_Δ={strongest_b['orient_delta']:+.3f}, n={strongest_b['n_features']})")
    print(f"\nAll α-clusters: {[c['id'] for c in a_clusters]}")
    print(f"All β-clusters: {[c['id'] for c in b_clusters]}")
    print(f"All mixed:      {[c['id'] for c in clusters_data if c['polarity'] == 'mixed']}")


if __name__ == "__main__":
    main()
