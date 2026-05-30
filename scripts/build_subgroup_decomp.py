"""
Script: build_subgroup_decomp.py

Builds the fine-resolution sub-cluster partition `agglo_coimp_subgroup_k30`
by combining HAC average-linkage at k_target=60 with iterative singleton merging
(min_size=2).

This is the canonical implementation of the procedure described in
docs/thesis_clean/results_summary_ru.md section 5.6.1.

Inputs:
    data/analysis/runD_v2/clustering_full/W_coimportance.npy
    data/analysis/runD_v2/clustering_full/feat_ids.json
    data/analysis/runD_v2/clustering_full/cluster_labels.csv

Output:
    Updates cluster_labels.csv to add column `agglo_coimp_subgroup_k30`
    Also saves feature_subgroup_assignments.csv

Algorithm:
    1. Build HAC linkage Z from W_coimportance (distance = 1 - W, average linkage)
    2. Cut at k_target=60 via scipy.cluster.hierarchy.fcluster(..., criterion='maxclust')
    3. While any cluster has size < min_size (=2):
       a. Pick first singleton S
       b. Find candidate cluster C with maximum mean(W[i,j] for i∈S, j∈C)
       c. Merge S → C (relabel)
    4. Return final partition (all clusters size ≥ 2)

Reference: this matches the function cluster_no_singletons() used throughout
the validation pipeline (scripts 37b, etc.).

Usage:
    python3 scripts/build_subgroup_decomp.py \\
        --clustering_dir data/analysis/runD_v2/clustering_full \\
        --k_target 60 \\
        --min_size 2 \\
        --col_name agglo_coimp_subgroup_k30
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cluster_no_singletons(Z: np.ndarray, k_target: int, W: np.ndarray,
                          min_size: int = 2) -> Tuple[np.ndarray, dict]:
    """
    HAC partition with iterative singleton merging.

    Args:
        Z: scipy linkage matrix (n-1, 4) from linkage()
        k_target: requested number of clusters (input to scipy fcluster)
        W: co-importance similarity matrix (n_features × n_features)
        min_size: minimum allowed cluster size (singletons merged below this)

    Returns:
        labels: array of length n_features with cluster IDs (all clusters size ≥ min_size)
        stats: dict with merge statistics
    """
    labels = fcluster(Z, k_target, criterion="maxclust")
    log.info(f"Raw fcluster(Z, k_target={k_target}) → {len(set(labels))} unique clusters")

    raw_sizes = pd.Series(labels).value_counts()
    initial_singletons = int((raw_sizes < min_size).sum())
    log.info(f"Initial singletons (size < {min_size}): {initial_singletons}")

    n_iters_into_existing = 0
    n_iters_into_singleton = 0

    while True:
        sizes = pd.Series(labels).value_counts()
        small = sizes[sizes < min_size].index.tolist()
        if not small:
            break
        for small_c in small:
            idxs = np.where(labels == small_c)[0]
            if len(idxs) == 0:
                continue
            best_c, best_sim = None, -np.inf
            for other_c in sizes.index:
                if other_c == small_c:
                    continue
                other_idxs = np.where(labels == other_c)[0]
                sim = float(W[np.ix_(idxs, other_idxs)].mean())
                if sim > best_sim:
                    best_sim, best_c = sim, other_c
            if best_c is not None:
                # Track whether target was already non-singleton or also singleton
                if sizes.get(best_c, 0) >= min_size:
                    n_iters_into_existing += 1
                else:
                    n_iters_into_singleton += 1
                labels[labels == small_c] = best_c
                break  # re-evaluate sizes from scratch

    final_k = len(set(labels))
    stats = {
        "raw_k": int(len(raw_sizes)),
        "initial_singletons": initial_singletons,
        "iterations": n_iters_into_existing + n_iters_into_singleton,
        "merged_into_existing": n_iters_into_existing,
        "merged_into_singleton": n_iters_into_singleton,
        "final_k": final_k,
    }
    log.info(f"Singleton merging complete: {stats}")
    return labels, stats


def build_partition(W: np.ndarray, k_target: int, min_size: int = 2) -> Tuple[np.ndarray, dict]:
    """Build HAC linkage and run singleton merging."""
    D = 1.0 - W
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    D = np.clip(D, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    labels, stats = cluster_no_singletons(Z, k_target, W, min_size)
    return labels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering_dir", type=Path, required=True,
                        help="Directory containing W_coimportance.npy, feat_ids.json, cluster_labels.csv")
    parser.add_argument("--k_target", type=int, default=60,
                        help="HAC parameter (default 60). Will produce actual_k after singleton merging.")
    parser.add_argument("--min_size", type=int, default=2,
                        help="Minimum cluster size (default 2; clusters smaller will be merged)")
    parser.add_argument("--col_name", type=str, default="agglo_coimp_subgroup_k30",
                        help="Column name to add to cluster_labels.csv")
    parser.add_argument("--save_assignments", action="store_true",
                        help="Also save feature_subgroup_assignments.csv to subgroup_decomp dir")
    args = parser.parse_args()

    # Load W and feat_ids
    W = np.load(args.clustering_dir / "W_coimportance.npy")
    feat_ids = json.loads((args.clustering_dir / "feat_ids.json").read_text())
    assert W.shape[0] == W.shape[1] == len(feat_ids), \
        f"W shape {W.shape} doesn't match feat_ids count {len(feat_ids)}"
    log.info(f"Loaded W ({W.shape}) and feat_ids ({len(feat_ids)} features)")

    # Build partition
    labels, stats = build_partition(W, args.k_target, args.min_size)

    # Update cluster_labels.csv
    cl_path = args.clustering_dir / "cluster_labels.csv"
    cl = pd.read_csv(cl_path)
    feat_id_to_label = dict(zip(feat_ids, labels.astype(int)))
    cl[args.col_name] = cl["feature_id"].map(feat_id_to_label).astype("Int64")
    n_missing = cl[args.col_name].isna().sum()
    assert n_missing == 0, f"{n_missing} features have no assignment after merge"
    cl.to_csv(cl_path, index=False)
    log.info(f"Updated {cl_path} with column '{args.col_name}' "
             f"(unique values: {cl[args.col_name].nunique()})")

    # Optionally save standalone assignments
    if args.save_assignments:
        out_dir = args.clustering_dir.parent / "carrier_stability/subgroup_decomp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame({
            "feature_id": feat_ids,
            "subgroup_cluster": labels.astype(int),
            "layer": [int(f.split("_F")[0][1:]) for f in feat_ids],
        })
        out_df.to_csv(out_dir / "feature_subgroup_assignments.csv", index=False)
        log.info(f"Saved feature_subgroup_assignments.csv to {out_dir}")

    print(f"\n=== Final stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
