#!/usr/bin/env python3
"""
Script 31: Activation-stability test for Condition I.a (carrier stability).

For each cluster C and each semantic_equiv_group g, computes:
  sigma_C(p) = mean activation of features in C on prompt p

Then measures how consistently sigma_C is stable across wording variants within g:
  sign_consistency = fraction of prompts in g where sign(sigma_C(p)) == majority sign
  cv               = std(sigma_C(p)) / mean(|sigma_C(p)|)   (low = stable)

Groups with only 1 prompt are skipped (no within-group variance to measure).

This is a more direct test of Condition I.a than the Jaccard recovery test:
instead of asking "do the same features cluster together on paraphrase splits?",
we ask "does cluster C activate consistently across paraphrases of the same concept?"

Outputs
-------
data/analysis/carrier_stability/
    activation_stability_per_cluster_group.csv   per (cluster, group) metrics
    activation_stability_per_cluster.csv         per cluster summary
    activation_stability_heatmap.html            interactive heatmap
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import argparse

ROOT = Path(__file__).parent.parent
GROUPING  = ROOT / "data/results/grouping"
CLUSTERING = ROOT / "data/results/clustering"
OUT = ROOT / "data/analysis/carrier_stability"

CLUSTER_COL = "agglo_coimp_k12"
MIN_PROMPTS_PER_GROUP = 2   # skip singleton groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="agglomerative cluster count")
    ap.add_argument("--grouping_dir", default=str(GROUPING))
    ap.add_argument("--clustering_dir", default=str(CLUSTERING))
    ap.add_argument("--out_dir", default=str(OUT))
    args = ap.parse_args()

    grouping_dir   = Path(args.grouping_dir)
    clustering_dir = Path(args.clustering_dir)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_col = f"agglo_coimp_k{args.k}"

    # ── load data ─────────────────────────────────────────────────────────────
    act = pd.read_csv(grouping_dir / "feature_prompt_activation_matrix.csv", index_col=0)
    # act: rows = feature_id, columns = prompt_idx (as string)

    cl = pd.read_csv(clustering_dir / "cluster_labels.csv")
    cl = cl[["feature_id", cluster_col]].rename(columns={cluster_col: "cluster"})

    pm = pd.read_csv(grouping_dir / "prompt_metadata.csv")
    pm = pm[["prompt_idx", "semantic_equiv_group", "wording_variant",
             "correct_answer", "level"]].copy()

    # align: act columns are prompt_idx as strings
    act.columns = [int(c) for c in act.columns]

    # only keep prompts that are in both act and pm
    shared_prompts = sorted(set(act.columns) & set(pm["prompt_idx"]))
    act = act[shared_prompts]
    pm  = pm[pm["prompt_idx"].isin(shared_prompts)].copy()

    feat_ids = list(act.index)
    cl = cl[cl["feature_id"].isin(feat_ids)].copy()

    clusters = sorted(cl["cluster"].unique())
    groups = pm.groupby("semantic_equiv_group")

    print(f"Features: {len(feat_ids)}  |  Clusters: {len(clusters)}  |  "
          f"Semantic groups: {pm['semantic_equiv_group'].nunique()}")

    # ── per (cluster, group) ─────────────────────────────────────────────────
    rows = []
    for cid in clusters:
        feat_in_cluster = cl[cl["cluster"] == cid]["feature_id"].tolist()
        # activation matrix for this cluster: shape (n_feat, n_prompts)
        act_c = act.loc[feat_in_cluster]

        for gname, gdf in groups:
            pidxs = gdf["prompt_idx"].tolist()
            if len(pidxs) < MIN_PROMPTS_PER_GROUP:
                continue

            # sigma_C(p) = mean activation of cluster features on prompt p
            sigma = act_c[pidxs].mean(axis=0).values   # shape (n_prompts_in_group,)

            mean_act    = float(np.mean(np.abs(sigma)))
            mean_signed = float(np.mean(sigma))

            # sign consistency: fraction agreeing with majority sign
            signs = np.sign(sigma)
            majority = 1.0 if np.mean(signs > 0) >= 0.5 else -1.0
            sign_cons = float(np.mean(signs == majority))

            # coefficient of variation (lower = more stable)
            cv = float(np.std(sigma) / mean_act) if mean_act > 1e-8 else np.nan

            # alpha/beta label of this group
            answers = gdf["correct_answer"].mode()
            answer  = answers.iloc[0] if len(answers) else "?"
            level   = gdf["level"].mode().iloc[0] if "level" in gdf.columns else "?"

            rows.append(dict(
                cluster=cid,
                group=gname,
                n_prompts=len(pidxs),
                mean_abs_activation=round(mean_act, 4),
                mean_signed_activation=round(mean_signed, 4),
                sign_consistency=round(sign_cons, 3),
                cv=round(cv, 3) if not np.isnan(cv) else None,
                correct_answer=answer,
                level=level,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "activation_stability_per_cluster_group.csv", index=False)
    print(f"Saved: activation_stability_per_cluster_group.csv  ({len(df)} rows)")

    # ── per cluster summary ───────────────────────────────────────────────────
    summary_rows = []
    for cid in clusters:
        sub = df[df["cluster"] == cid]
        # only groups where cluster is actually active (mean_abs > threshold)
        active_thresh = sub["mean_abs_activation"].quantile(0.25)
        active = sub[sub["mean_abs_activation"] >= active_thresh]

        summary_rows.append(dict(
            cluster=cid,
            n_groups_tested=len(sub),
            n_groups_active=len(active),
            mean_sign_consistency=round(active["sign_consistency"].mean(), 3),
            frac_groups_sign_cons_ge075=round(
                (active["sign_consistency"] >= 0.75).mean(), 3),
            mean_cv=round(active["cv"].dropna().mean(), 3),
            mean_abs_activation_overall=round(sub["mean_abs_activation"].mean(), 4),
        ))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "activation_stability_per_cluster.csv", index=False)
    print(f"Saved: activation_stability_per_cluster.csv")
    print()
    print(summary[["cluster", "mean_sign_consistency",
                   "frac_groups_sign_cons_ge075", "mean_cv"]].to_string(index=False))

    # ── heatmap: sign_consistency per (cluster, group) ───────────────────────
    try:
        import plotly.graph_objects as go

        pivot = df.pivot(index="cluster", columns="group",
                         values="sign_consistency").fillna(0)

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[f"C{c}" for c in pivot.index],
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            colorbar=dict(title="sign consistency"),
        ))
        fig.update_layout(
            title="Activation sign consistency per cluster × semantic group",
            xaxis_title="semantic group",
            yaxis_title="cluster",
            height=600,
        )
        fig.write_html(str(out_dir / "activation_stability_heatmap.html"))
        print("Saved: activation_stability_heatmap.html")
    except ImportError:
        print("plotly not available, skipping heatmap")


if __name__ == "__main__":
    main()
