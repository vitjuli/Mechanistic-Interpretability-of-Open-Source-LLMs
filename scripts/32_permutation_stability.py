#!/usr/bin/env python3
"""
Script 32: Permutation-based null distribution for carrier stability (Condition I.a).

For each cluster G_i in the full-corpus agglomerative clustering:
  J_obs(G_i) = best Jaccard between G_i and any cluster in the
               paraphrase-split clusterings (mean of A-split and B-split).

Null distribution: randomly permute prompt indices between A and B
(preserving |A| and |B|), re-cluster both halves, compute best Jaccard
for each G_i. Repeat R times.

Per-cluster p-value (right-tail):
  p(G_i) = |{r : J_null(r) >= J_obs(G_i)}| / R
  Low p -> observed Jaccard is in the UPPER tail of null
           (paraphrase recovery is BETTER than random splits)

Per-cluster p-value (left-tail, for the "not worse than random" test):
  p_left(G_i) = |{r : J_null(r) <= J_obs(G_i)}| / R
  Low p_left -> observed Jaccard is in the LOWER tail of null
                (paraphrase recovery is WORSE than random splits)

For Condition I.a we want to show "paraphrase does not hurt":
  i.e., J_obs is NOT significantly below the null.
  Criterion: p_left > alpha (cluster survives the "is it worse?" test).
  Equivalently: J_obs >= tau_null_lower(G_i) = alpha-quantile of null.

FDR correction (Benjamini-Hochberg) applied to both p-value sets.

Outputs
-------
data/analysis/carrier_stability/
    permutation_stability.csv    per-cluster results
    permutation_stability.json   summary
"""
import json, collections
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import argparse

ROOT = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
CLU     = ROOT / "data/results/clustering"
OUT     = ROOT / "data/analysis/carrier_stability"

TOP_K = 10
K     = 12
R     = 1000   # permutation draws
ALPHA = 0.05
SEED  = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def build_coimp(X, cols):
    Xs = X[:, cols]
    k  = min(TOP_K, len(cols))
    tops = [set(np.argsort(Xs[i])[::-1][:k].tolist()) for i in range(X.shape[0])]
    n  = X.shape[0]
    W  = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            u = len(tops[i] | tops[j])
            W[i, j] = W[j, i] = len(tops[i] & tops[j]) / u if u else 0.0
    return W


def cluster_agglo(W, k=K):
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, k, criterion="maxclust")


def best_jaccard_per_cluster(full_labels, other_labels):
    """For each cluster in full, return best Jaccard with any cluster in other."""
    results = {}
    for c in sorted(set(full_labels)):
        S = set(np.where(full_labels == c)[0])
        best = max(
            len(S & set(np.where(other_labels == c2)[0])) /
            len(S | set(np.where(other_labels == c2)[0]))
            for c2 in set(other_labels)
        )
        results[c] = best
    return results


def paraphrase_split(seg, wv, n_prompt):
    order = collections.defaultdict(list)
    for j in range(n_prompt):
        order[seg[j]].append((wv[j], j))
    A, B, tog = [], [], 0
    for g, lst in order.items():
        lst.sort()
        if len(lst) == 1:
            (A if tog % 2 == 0 else B).append(lst[0][1]); tog += 1
        else:
            for r, (w, j) in enumerate(lst):
                (A if r % 2 == 0 else B).append(j)
    return A, B


def bh_correction(pvals, alpha=ALPHA):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = rankdata(pvals, method="ordinal")
    adjusted = np.minimum(1.0, np.array(pvals) * n / ranks)
    # enforce monotonicity
    adjusted_mono = adjusted.copy()
    for i in range(n - 2, -1, -1):
        adjusted_mono[order[i]] = min(adjusted_mono[order[i]],
                                      adjusted_mono[order[i + 1]])
    return adjusted_mono


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K, help="agglomerative cluster count")
    ap.add_argument("--grouping_dir", default=str(GROUPING))
    ap.add_argument("--clustering_dir", default=str(CLU))
    ap.add_argument("--out_dir", default=str(OUT))
    args = ap.parse_args()

    grouping_dir   = Path(args.grouping_dir)
    clustering_dir = Path(args.clustering_dir)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_ids = json.load(open(clustering_dir / "feat_ids.json"))
    Xdf = pd.read_csv(grouping_dir / "feature_prompt_abs_effect_matrix.csv", index_col=0)
    assert list(Xdf.index) == feat_ids
    X = Xdf.values.astype(float)
    n_prompt = X.shape[1]

    pm    = pd.read_csv(grouping_dir / "prompt_metadata.csv")
    meta  = pm.set_index("prompt_idx").reindex([int(c) for c in Xdf.columns])
    wv    = meta["wording_variant"].fillna(1).astype(int).values
    seg   = meta["semantic_equiv_group"].astype(str).values

    # full-corpus clustering (use --k if supplied)
    Wf = build_coimp(X, list(range(n_prompt)))
    lf = cluster_agglo(Wf, k=args.k)
    clusters = sorted(set(lf))
    n_feat_per_cluster = {c: int((lf == c).sum()) for c in clusters}

    # paraphrase split
    A, B = paraphrase_split(seg, wv, n_prompt)
    WA, WB = build_coimp(X, A), build_coimp(X, B)
    la, lb = cluster_agglo(WA, k=args.k), cluster_agglo(WB, k=args.k)

    jac_A = best_jaccard_per_cluster(lf, la)
    jac_B = best_jaccard_per_cluster(lf, lb)
    J_obs = {c: (jac_A[c] + jac_B[c]) / 2 for c in clusters}

    print(f"Full corpus: {n_prompt} prompts | A={len(A)} B={len(B)}")
    print(f"Clusters: {len(clusters)} | R={R} permutations\n")

    # permutation null
    rng  = np.random.default_rng(SEED)
    all_idx = np.arange(n_prompt)
    nA = len(A)

    null = {c: [] for c in clusters}   # J_null values per cluster

    for r in range(R):
        perm = rng.permutation(all_idx)
        Ar, Br = perm[:nA].tolist(), perm[nA:].tolist()
        WAr, WBr = build_coimp(X, Ar), build_coimp(X, Br)
        lar, lbr = cluster_agglo(WAr, k=args.k), cluster_agglo(WBr, k=args.k)
        jAr = best_jaccard_per_cluster(lf, lar)
        jBr = best_jaccard_per_cluster(lf, lbr)
        for c in clusters:
            null[c].append((jAr[c] + jBr[c]) / 2)
        if (r + 1) % 100 == 0:
            print(f"  {r+1}/{R} done")

    # per-cluster stats and p-values
    rows = []
    p_right_raw = []   # fraction null >= J_obs  (paraphrase BETTER than random)
    p_left_raw  = []   # fraction null <= J_obs  (paraphrase NOT WORSE than random)

    for c in clusters:
        null_arr = np.array(null[c])
        j_obs    = J_obs[c]
        p_right  = float((null_arr >= j_obs).mean())
        p_left   = float((null_arr <= j_obs).mean())
        tau_05   = float(np.percentile(null_arr, 5))    # lower 5th pctile of null
        tau_95   = float(np.percentile(null_arr, 95))   # upper 95th pctile of null
        pctile   = float((null_arr < j_obs).mean() * 100)

        p_right_raw.append(p_right)
        p_left_raw.append(p_left)

        rows.append(dict(
            cluster=c,
            n_features=n_feat_per_cluster[c],
            J_obs=round(j_obs, 3),
            null_mean=round(float(null_arr.mean()), 3),
            null_std=round(float(null_arr.std()), 3),
            tau_lower5=round(tau_05, 3),
            tau_upper95=round(tau_95, 3),
            pctile_among_null=round(pctile, 1),
            p_right=round(p_right, 4),
            p_left=round(p_left, 4),
        ))

    # BH correction
    p_right_adj = bh_correction(p_right_raw)
    p_left_adj  = bh_correction(p_left_raw)

    df = pd.DataFrame(rows)
    df["p_right_bh"] = p_right_adj.round(4)
    df["p_left_bh"]  = p_left_adj.round(4)

    # stability verdict: J_obs NOT significantly below null
    # (p_left > alpha means J_obs is not in the lower tail)
    df["stable_not_worse"] = df["p_left_bh"] > ALPHA
    # strong stability verdict: J_obs significantly ABOVE null
    df["stable_better"]    = df["p_right_bh"] < ALPHA

    df.to_csv(out_dir / "permutation_stability.csv", index=False)

    print("\n── Per-cluster permutation results ──────────────────────────────")
    print(df[["cluster","n_features","J_obs","null_mean","tau_lower5",
              "pctile_among_null","p_left_bh","p_right_bh",
              "stable_not_worse","stable_better"]].to_string(index=False))

    n_not_worse = df["stable_not_worse"].sum()
    n_better    = df["stable_better"].sum()
    print(f"\nStable (not worse than random, p_left_BH > {ALPHA}): "
          f"{n_not_worse}/{len(df)}")
    print(f"Strongly stable (better than random, p_right_BH < {ALPHA}): "
          f"{n_better}/{len(df)}")

    summary = dict(
        R=R, K=args.k, alpha=ALPHA,
        n_stable_not_worse=int(n_not_worse),
        n_stable_better=int(n_better),
        n_clusters=len(df),
        mean_J_obs=round(float(df["J_obs"].mean()), 3),
        mean_null_mean=round(float(df["null_mean"].mean()), 3),
    )
    json.dump(summary, open(out_dir / "permutation_stability.json", "w"), indent=2)
    print(f"\nSaved: permutation_stability.csv, permutation_stability.json")


if __name__ == "__main__":
    main()
