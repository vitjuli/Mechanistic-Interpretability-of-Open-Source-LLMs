#!/usr/bin/env python3
"""
Script 30: Carrier-stability recovery test + agglomerative re-clustering.

Operationalises Condition I.a (carrier stability) of Definition 1:

    Pr_{p~P, t~T_irr}[ exists G' in Phi(t(p)) s.t. G' = G ] >= 1 - eta_irr

T_irr here = paraphrase (wording_variant within a semantic_equivalence_group).
We split the corpus into two paraphrase halves (A = odd wording variants,
B = even), rebuild the co-importance graph on each half, re-cluster, and
measure how well the full-corpus clusters are recovered.

A random-split control (same sizes, ignoring paraphrase structure) gives the
data-reduction floor: if paraphrase-split recovery ~ random-split recovery,
the clustering is invariant to paraphrasing up to the cost of having less data.

Finding: the original method (co-importance Louvain, resolution=1.0) is
UNSTABLE (paraphrase ARI below the random-split distribution). Average-linkage
agglomerative on the same co-importance graph is stable. We therefore select
Phi = agglomerative(co-importance, average linkage, k=12) by carrier stability.

Outputs
-------
data/analysis/carrier_stability/
    recovery_summary.json          metrics for both methods
    recovery_k_sweep.csv           agglomerative k sweep
Adds column `agglo_coimp_k{K}` to data/results/clustering/cluster_labels.csv
(coimp_louvain is preserved for reversibility).
"""
import json, argparse, collections, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
CLU = ROOT / "data/results/clustering"
OUT = ROOT / "data/analysis/carrier_stability"

TOP_K = 10          # top-k prompts per feature (matches script 22)
LOUVAIN_THR = 0.15  # original co-importance Louvain threshold (script 23)
LOUVAIN_RES = 1.0
N_RANDOM = 40       # random-split control draws
SEED = 42


def build_coimp(X, cols):
    """Co-importance Jaccard graph over a subset of prompt columns."""
    Xs = X[:, cols]
    k = min(TOP_K, len(cols))
    tops = [set(np.argsort(Xs[i])[::-1][:k].tolist()) for i in range(X.shape[0])]
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            u = len(tops[i] | tops[j])
            W[i, j] = W[j, i] = (len(tops[i] & tops[j]) / u if u else 0.0)
    return W


def cluster_agglo(W, k):
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, k, criterion="maxclust")


def cluster_louvain(W, thr=LOUVAIN_THR, res=LOUVAIN_RES):
    import community as community_louvain
    import networkx as nx
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > thr:
                G.add_edge(i, j, weight=float(W[i, j]))
    if G.number_of_edges() == 0:
        return np.zeros(n, dtype=int)
    part = community_louvain.best_partition(G, weight="weight",
                                            random_state=SEED, resolution=res)
    return np.array([part[i] for i in range(n)])


def recovery(full, other):
    """For each full cluster, best Jaccard with any cluster in `other`."""
    out = []
    for c in set(full):
        S = set(np.where(full == c)[0])
        best = max((len(S & set(np.where(other == c2)[0])) /
                    len(S | set(np.where(other == c2)[0]))
                    for c2 in set(other)), default=0.0)
        out.append(best)
    return float(np.mean(out)), float(np.mean([x >= 0.5 for x in out]))


def paraphrase_split(seg, wv, n_prompt):
    """A = odd wording variants per semantic group, B = even."""
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


def stability(clfun, X, A, B, rand_splits):
    Wf, WA, WB = (build_coimp(X, list(range(X.shape[1]))),
                  build_coimp(X, A), build_coimp(X, B))
    lf, la, lb = clfun(Wf), clfun(WA), clfun(WB)
    mj1, fr1 = recovery(lf, la)
    mj2, fr2 = recovery(lf, lb)
    ari_p = adjusted_rand_score(la, lb)
    ari_r = [adjusted_rand_score(clfun(build_coimp(X, ra)),
                                 clfun(build_coimp(X, rb)))
             for ra, rb in rand_splits]
    pct = float((np.array(ari_r) < ari_p).mean() * 100)
    return dict(n_clusters=len(set(lf)),
                mean_jaccard=(mj1 + mj2) / 2,
                frac_recovered_ge_0_5=(fr1 + fr2) / 2,
                ari_paraphrase=float(ari_p),
                ari_random_mean=float(np.mean(ari_r)),
                ari_random_std=float(np.std(ari_r)),
                paraphrase_percentile_of_random=pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="agglomerative cluster count")
    ap.add_argument("--no_inject", action="store_true",
                    help="skip writing the agglo column into cluster_labels.csv")
    ap.add_argument("--grouping_dir", default=str(GROUPING),
                    help="directory with feature_prompt_* matrices and prompt_metadata.csv")
    ap.add_argument("--clustering_dir", default=str(CLU),
                    help="directory with feat_ids.json, W_coimportance.npy, cluster_labels.csv")
    ap.add_argument("--out_dir", default=str(OUT),
                    help="output directory for carrier_stability results")
    args = ap.parse_args()

    grouping_dir = Path(args.grouping_dir)
    clustering_dir = Path(args.clustering_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_ids = json.load(open(clustering_dir / "feat_ids.json"))
    Xdf = pd.read_csv(grouping_dir / "feature_prompt_abs_effect_matrix.csv", index_col=0)
    assert list(Xdf.index) == feat_ids, "feature order mismatch"
    X = Xdf.values.astype(float)
    n_prompt = X.shape[1]

    pm = pd.read_csv(grouping_dir / "prompt_metadata.csv")
    col_idx = [int(c) for c in Xdf.columns]
    meta = pm.set_index("prompt_idx").reindex(col_idx)
    wv = meta["wording_variant"].fillna(1).astype(int).values
    seg = meta["semantic_equiv_group"].astype(str).values

    A, B = paraphrase_split(seg, wv, n_prompt)
    rng = np.random.default_rng(0)
    allidx = np.arange(n_prompt)
    rand_splits = [(lambda p: (p[:len(A)].tolist(), p[len(A):].tolist()))(rng.permutation(allidx))
                   for _ in range(N_RANDOM)]
    print(f"Corpus: {n_prompt} prompts | paraphrase split A={len(A)} B={len(B)}")

    # ── stability of both methods ────────────────────────────────────────────
    summary = {
        "config": {"top_k": TOP_K, "n_random_splits": N_RANDOM,
                   "agglo_k": args.k, "louvain_thr": LOUVAIN_THR,
                   "louvain_res": LOUVAIN_RES},
        "louvain_coimp_res1.0": stability(cluster_louvain, X, A, B, rand_splits),
        f"agglo_coimp_k{args.k}": stability(lambda W: cluster_agglo(W, args.k),
                                            X, A, B, rand_splits),
    }
    print("\nMethod comparison (carrier stability):")
    for name in ("louvain_coimp_res1.0", f"agglo_coimp_k{args.k}"):
        s = summary[name]
        print(f"  {name:22} meanJac={s['mean_jaccard']:.2f} "
              f"frac>=.5={s['frac_recovered_ge_0_5']:.2f} "
              f"ARI_p={s['ari_paraphrase']:.2f} ARI_r={s['ari_random_mean']:.2f} "
              f"pctile={s['paraphrase_percentile_of_random']:.0f}")

    # ── k sweep (for the record) ─────────────────────────────────────────────
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Wf = build_coimp(X, list(range(n_prompt)))
    rows = []
    for k in range(8, 23):
        s = stability(lambda W, k=k: cluster_agglo(W, k), X, A, B, rand_splits)
        lf = cluster_agglo(Wf, k)
        sizes = collections.Counter(lf)
        s.update(k=k, n_singletons=sum(1 for v in sizes.values() if v == 1),
                 silhouette=float(silhouette_score(Xn, lf, metric="cosine"))
                 if len(set(lf)) > 1 else float("nan"))
        rows.append(s)
    pd.DataFrame(rows).to_csv(out_dir / "recovery_k_sweep.csv", index=False)
    json.dump(summary, open(out_dir / "recovery_summary.json", "w"), indent=2)
    print(f"\nSaved {out_dir/'recovery_summary.json'} and recovery_k_sweep.csv")

    # ── inject agglomerative labels into cluster_labels.csv ──────────────────
    if not args.no_inject:
        W_saved = np.load(clustering_dir / "W_coimportance.npy")
        labels_full = cluster_agglo(W_saved, args.k)
        col = f"agglo_coimp_k{args.k}"
        ldf = pd.read_csv(clustering_dir / "cluster_labels.csv")
        assert list(ldf["feature_id"]) == feat_ids, "label-file feature order mismatch"
        ldf[col] = labels_full.astype(int)
        ldf.to_csv(clustering_dir / "cluster_labels.csv", index=False)
        print(f"Injected column '{col}' into cluster_labels.csv "
              f"({len(set(labels_full))} clusters; coimp_louvain preserved)")


if __name__ == "__main__":
    main()
