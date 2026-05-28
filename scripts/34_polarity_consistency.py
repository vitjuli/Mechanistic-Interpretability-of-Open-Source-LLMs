#!/usr/bin/env python3
"""
Script 34: Polarity consistency test for Condition I.b.

For each cluster C_i computes the signed cluster contribution:
  sigma_G(p) = mean signed effect of cluster features on prompt p
             = (1/|G|) * sum_{f in G} effect(f, p)

where effect(f, p) is the gradient×activation attribution score from
feature_prompt_effect_matrix.csv. Positive = feature helps the correct
answer on this prompt; negative = feature hurts it.

Two tests:

(A) Mann-Whitney U: compare sigma_G distribution between alpha-prompts
    and beta-prompts.  A polarized cluster has opposite sign on the two
    classes: significant U with large effect size (rank-biserial r).

(B) Paired sign-flip rate (contrastive pairs): for each matched
    (p_alpha, p_beta) pair from the same semantic situation, check whether
    sign(sigma_G(p_alpha)) != sign(sigma_G(p_beta)).
    This directly estimates Pr[sign-flip | |sigma| > 0] from the formula.

Additional metrics per cluster:
  - median_alpha, median_beta: median sigma_G by answer class
  - frac_correct_alpha: fraction of alpha-prompts where sigma_G > 0
  - frac_correct_beta:  fraction of beta-prompts where sigma_G > 0

BH FDR correction applied to Mann-Whitney p-values across 12 clusters.

Outputs
-------
data/analysis/carrier_stability/
    polarity_consistency.csv    per-cluster results
    polarity_consistency.json   summary
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, rankdata

ROOT     = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
CLU      = ROOT / "data/results/clustering"
OUT      = ROOT / "data/analysis/carrier_stability"

CLUSTER_COL = "agglo_coimp_k12"
ALPHA_BH    = 0.05


def bh_correction(pvals):
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = rankdata(pvals, method="ordinal")
    adj   = np.minimum(1.0, np.array(pvals) * n / ranks)
    mono  = adj.copy()
    for i in range(n - 2, -1, -1):
        mono[order[i]] = min(mono[order[i]], mono[order[i + 1]])
    return mono


def rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return float(1 - 2 * u_stat / (n1 * n2))


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

    # ── load ─────────────────────────────────────────────────────────────────
    eff = pd.read_csv(grouping_dir / "feature_prompt_effect_matrix.csv", index_col=0)
    eff.columns = [int(c) for c in eff.columns]

    cl = pd.read_csv(clustering_dir / "cluster_labels.csv")[["feature_id", cluster_col]]
    cl = cl.rename(columns={cluster_col: "cluster"})

    pm = pd.read_csv(grouping_dir / "prompt_metadata.csv")
    pm = pm[["prompt_idx", "correct_answer", "semantic_equiv_group",
             "contrastive_pair_id", "contrastive_role"]].copy()

    # align prompts
    shared = sorted(set(eff.columns) & set(pm["prompt_idx"]))
    eff = eff[shared]
    pm  = pm[pm["prompt_idx"].isin(shared)].copy()
    cl  = cl[cl["feature_id"].isin(eff.index)].copy()

    # prompt index sets
    alpha_idx = pm[pm["correct_answer"] == "alpha"]["prompt_idx"].tolist()
    beta_idx  = pm[pm["correct_answer"] == "beta"]["prompt_idx"].tolist()

    # contrastive pairs (matched alpha/beta per situation)
    pairs_df = pm[pm["contrastive_pair_id"].notna()].copy()
    alpha_pairs = (pairs_df[pairs_df["contrastive_role"] == "alpha_member"]
                   .set_index("contrastive_pair_id")["prompt_idx"])
    beta_pairs  = (pairs_df[pairs_df["contrastive_role"] == "beta_member"]
                   .set_index("contrastive_pair_id")["prompt_idx"])
    common_pairs = sorted(set(alpha_pairs.index) & set(beta_pairs.index))

    clusters = sorted(cl["cluster"].unique())
    print(f"Clusters: {len(clusters)}  |  "
          f"alpha prompts: {len(alpha_idx)}  |  beta prompts: {len(beta_idx)}")
    print(f"Contrastive pairs available: {len(common_pairs)}\n")

    rows    = []
    pvals_raw = []

    for cid in clusters:
        feats = cl[cl["cluster"] == cid]["feature_id"].tolist()

        # sigma_G(p) = mean signed effect across cluster features
        sigma = eff.loc[feats].mean(axis=0)   # Series: prompt_idx -> sigma

        sig_alpha = sigma[alpha_idx].values
        sig_beta  = sigma[beta_idx].values

        # ── (A) Mann-Whitney U ────────────────────────────────────────────
        stat, pval = mannwhitneyu(sig_alpha, sig_beta, alternative="two-sided")
        rb = rank_biserial(stat, len(sig_alpha), len(sig_beta))

        # medians
        med_alpha = float(np.median(sig_alpha))
        med_beta  = float(np.median(sig_beta))

        # fraction of prompts where sigma has the "correct" sign
        # correct sign for alpha-prompts = positive (cluster helps alpha)
        # correct sign for beta-prompts  = positive (cluster helps beta)
        # BUT we want opposite signs → polarity = |med_alpha| and |med_beta| have opposite signs
        polarity_direction = "alpha" if med_alpha > 0 else "beta"  # which class gets positive sigma
        frac_pos_alpha = float((sig_alpha > 0).mean())
        frac_pos_beta  = float((sig_beta > 0).mean())

        # ── (B) Paired sign-flip rate ─────────────────────────────────────
        flip_count = 0
        n_pairs_used = 0
        for pid in common_pairs:
            pa = alpha_pairs[pid]
            pb = beta_pairs[pid]
            if pa in sigma.index and pb in sigma.index:
                sa, sb = sigma[pa], sigma[pb]
                # sign flip: one positive, one negative
                if sa != 0 and sb != 0:
                    flip_count += int(np.sign(sa) != np.sign(sb))
                    n_pairs_used += 1

        sign_flip_rate = round(flip_count / n_pairs_used, 3) if n_pairs_used > 0 else None

        pvals_raw.append(pval)
        rows.append(dict(
            cluster=cid,
            n_features=len(feats),
            layers=",".join(sorted(set(f.split("_")[0] for f in feats))),
            median_alpha=round(med_alpha, 4),
            median_beta=round(med_beta, 4),
            polarity_direction=polarity_direction,
            frac_pos_alpha=round(frac_pos_alpha, 3),
            frac_pos_beta=round(frac_pos_beta, 3),
            mw_stat=round(stat, 1),
            mw_pval=round(pval, 6),
            rank_biserial=round(rb, 3),
            sign_flip_rate=sign_flip_rate,
            n_pairs_used=n_pairs_used,
        ))

    # BH correction
    adj_pvals = bh_correction(pvals_raw)
    for i, row in enumerate(rows):
        row["mw_pval_bh"] = round(adj_pvals[i], 6)
        row["polarized_bh"] = bool(adj_pvals[i] < ALPHA_BH)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "polarity_consistency.csv", index=False)

    print("── Per-cluster polarity results ─────────────────────────────────────")
    print(df[["cluster","n_features","layers","median_alpha","median_beta",
              "rank_biserial","mw_pval_bh","sign_flip_rate","polarized_bh"]].to_string(index=False))

    n_polarized = df["polarized_bh"].sum()
    print(f"\nPolarized (MW BH < {ALPHA_BH}): {n_polarized}/{len(df)}")
    print(f"Mean |rank-biserial|: {df['rank_biserial'].abs().mean():.3f}")
    print(f"Mean sign-flip rate (contrastive pairs): "
          f"{df['sign_flip_rate'].dropna().mean():.3f}")

    summary = dict(
        alpha_bh=ALPHA_BH,
        n_clusters=len(df),
        n_polarized=int(n_polarized),
        mean_abs_rank_biserial=round(float(df["rank_biserial"].abs().mean()), 3),
        mean_sign_flip_rate=round(float(df["sign_flip_rate"].dropna().mean()), 3),
        n_pairs=int(df["n_pairs_used"].iloc[0]),
    )
    json.dump(summary, open(out_dir / "polarity_consistency.json", "w"), indent=2)
    print(f"\nSaved: polarity_consistency.csv, polarity_consistency.json")


if __name__ == "__main__":
    main()
