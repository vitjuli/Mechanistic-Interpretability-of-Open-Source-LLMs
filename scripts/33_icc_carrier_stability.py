#!/usr/bin/env python3
"""
Script 33: ICC-based carrier stability test for Condition I.a.

For each cluster C_i, computes A_{C_i}(p) = mean activation of cluster features
on prompt p. Then uses ICC (Intraclass Correlation Coefficient, one-way random
effects) to measure what fraction of variance in A_{C_i}(p) is explained by the
physical situation (semantic_equiv_group) vs surface wording (wording_variant).

ICC = sigma^2_between / (sigma^2_between + sigma^2_within)

High ICC -> cluster activation is determined by the physical state V_h,
            not by how the question is phrased. This is carrier stability.
Low ICC  -> cluster cannot distinguish meaning from surface form.

ICC computed via one-way ANOVA (Shrout & Fleiss ICC(1,1)):
  MS_B = between-groups mean square
  MS_W = within-groups mean square
  n_0  = effective group size (harmonic-mean correction for unequal groups)
  ICC  = (MS_B - MS_W) / (MS_B + (n_0 - 1) * MS_W)

Outputs
-------
data/analysis/carrier_stability/
    icc_stability.csv     per-cluster ICC results
    icc_stability.json    summary
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import argparse

ROOT     = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
CLU      = ROOT / "data/results/clustering"
OUT      = ROOT / "data/analysis/carrier_stability"

CLUSTER_COL = "agglo_coimp_k12"
TAU_ICC     = 0.5    # a priori threshold: moderate reliability
MIN_GROUPS  = 5      # skip clusters with too few active groups


def icc_oneway(values_by_group):
    """
    One-way random effects ICC (Shrout & Fleiss ICC(1,1)).
    values_by_group: list of arrays, one per group.
    Returns ICC, MS_B, MS_W.
    """
    k   = len(values_by_group)               # number of groups
    ns  = np.array([len(g) for g in values_by_group])
    N   = ns.sum()
    grand_mean = np.concatenate(values_by_group).mean()

    # between-groups sum of squares
    group_means = np.array([np.mean(g) for g in values_by_group])
    SS_B = np.sum(ns * (group_means - grand_mean) ** 2)
    MS_B = SS_B / (k - 1)

    # within-groups sum of squares
    SS_W = sum(np.sum((g - group_means[i]) ** 2)
               for i, g in enumerate(values_by_group))
    df_W = N - k
    MS_W = SS_W / df_W if df_W > 0 else 0.0

    # effective n (harmonic-mean correction for unequal groups)
    n0 = (N - np.sum(ns ** 2) / N) / (k - 1)

    if MS_B + (n0 - 1) * MS_W < 1e-12:
        return 0.0, MS_B, MS_W

    icc_val = (MS_B - MS_W) / (MS_B + (n0 - 1) * MS_W)
    return float(np.clip(icc_val, 0.0, 1.0)), float(MS_B), float(MS_W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="agglomerative cluster count")
    ap.add_argument("--grouping_dir", default=str(GROUPING))
    ap.add_argument("--clustering_dir", default=str(CLU))
    ap.add_argument("--out_dir", default=str(OUT))
    args = ap.parse_args()

    grouping_dir   = Path(args.grouping_dir)
    clustering_dir = Path(args.clustering_dir)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_col = f"agglo_coimp_k{args.k}"

    # ── load data ─────────────────────────────────────────────────────────────
    act = pd.read_csv(grouping_dir / "feature_prompt_activation_matrix.csv", index_col=0)
    act.columns = [int(c) for c in act.columns]

    cl = pd.read_csv(clustering_dir / "cluster_labels.csv")[["feature_id", cluster_col]]
    cl = cl.rename(columns={cluster_col: "cluster"})

    pm = pd.read_csv(grouping_dir / "prompt_metadata.csv")
    pm = pm[["prompt_idx", "semantic_equiv_group", "wording_variant",
             "correct_answer", "level"]].copy()

    shared = sorted(set(act.columns) & set(pm["prompt_idx"]))
    act = act[shared]
    pm  = pm[pm["prompt_idx"].isin(shared)].copy()
    cl  = cl[cl["feature_id"].isin(act.index)].copy()

    clusters = sorted(cl["cluster"].unique())
    groups   = pm.groupby("semantic_equiv_group")

    print(f"Features: {len(act)}  |  Clusters: {len(clusters)}  |  "
          f"Semantic groups: {pm['semantic_equiv_group'].nunique()}")
    print(f"Threshold tau_ICC = {TAU_ICC}\n")

    rows = []
    for cid in clusters:
        feats = cl[cl["cluster"] == cid]["feature_id"].tolist()
        n_feat = len(feats)

        # A_{C}(p) = mean activation across cluster features
        A = act.loc[feats].mean(axis=0)   # Series: prompt_idx -> mean activation

        # collect per-group activation arrays
        vbg = []    # values_by_group: only groups with >= 2 wording variants
        group_names, group_ns, group_means_list = [], [], []
        for gname, gdf in groups:
            pidxs = [p for p in gdf["prompt_idx"].tolist() if p in A.index]
            if len(pidxs) < 2:
                continue
            vals = A[pidxs].values
            vbg.append(vals)
            group_names.append(gname)
            group_ns.append(len(vals))
            group_means_list.append(float(vals.mean()))

        n_groups = len(vbg)
        if n_groups < MIN_GROUPS:
            rows.append(dict(cluster=cid, n_features=n_feat,
                             n_groups=n_groups, icc=None, ms_b=None, ms_w=None,
                             sigma_between=None, sigma_within=None,
                             passes_threshold=None, note="too_few_groups"))
            continue

        icc_val, ms_b, ms_w = icc_oneway(vbg)

        # sigma_between = std of group means; sigma_within = mean within-group std
        sigma_between = float(np.std(group_means_list))
        sigma_within  = float(np.mean([np.std(g) for g in vbg]))

        rows.append(dict(
            cluster=cid,
            n_features=n_feat,
            n_groups=n_groups,
            icc=round(icc_val, 3),
            ms_b=round(ms_b, 4),
            ms_w=round(ms_w, 4),
            sigma_between=round(sigma_between, 4),
            sigma_within=round(sigma_within, 4),
            passes_threshold=bool(icc_val >= TAU_ICC),
            note="ok",
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "icc_stability.csv", index=False)

    valid = df[df["note"] == "ok"].copy()
    print(valid[["cluster","n_features","n_groups","icc",
                 "sigma_between","sigma_within","passes_threshold"]].to_string(index=False))

    n_pass = valid["passes_threshold"].sum()
    print(f"\nICC >= {TAU_ICC}: {n_pass}/{len(valid)} clusters")
    print(f"Mean ICC: {valid['icc'].mean():.3f}  |  Min: {valid['icc'].min():.3f}  "
          f"|  Max: {valid['icc'].max():.3f}")

    summary = dict(
        tau_icc=TAU_ICC,
        n_clusters_tested=len(valid),
        n_pass=int(n_pass),
        mean_icc=round(float(valid["icc"].mean()), 3),
        min_icc=round(float(valid["icc"].min()), 3),
        max_icc=round(float(valid["icc"].max()), 3),
    )
    json.dump(summary, open(out_dir / "icc_stability.json", "w"), indent=2)
    print(f"\nSaved: icc_stability.csv, icc_stability.json")


if __name__ == "__main__":
    main()
