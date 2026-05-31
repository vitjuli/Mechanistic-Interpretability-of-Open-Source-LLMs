"""
diag_pca_effective_rank.py

Approach 3: Effective rank of cluster activation matrix via PCA.

For 30 sub-clusters × 538 prompts, compute:
  - SVD of (centered, scaled) cluster activations
  - Explained variance ratio per principal component
  - Effective rank at 90%, 95%, 99% variance thresholds
  - Compare full 30-cluster ensemble vs L1-selected 14-cluster ensemble

Output: data/analysis/iia_failure_diagnosis/pca_effective_rank.{json,csv}
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
ACT  = ROOT / "data/analysis/runD_v2/activations"
CSUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp/feature_subgroup_assignments.csv"
PM   = ROOT / "data/analysis/runD_v2/grouping/prompt_metadata.csv"
WEIGHTS = ROOT / "data/analysis/iia_failure_diagnosis/linear_ensemble_weights.csv"
OUT  = ROOT / "data/analysis/iia_failure_diagnosis"

# Load activations and labels
act = np.load(ACT / "activation_matrix.npy")
feat_ids = (ACT / "feature_ids.txt").read_text().strip().split("\n")
prompt_idxs = [int(x) for x in (ACT / "prompt_idxs.txt").read_text().strip().split("\n")]

cl = pd.read_csv(CSUB)
fid_to_cid = dict(zip(cl["feature_id"], cl["subgroup_cluster"]))
clusters = defaultdict(list)
for fid in feat_ids:
    if fid in fid_to_cid:
        clusters[fid_to_cid[fid]].append(fid)
cluster_ids = sorted(clusters.keys())
n_clusters = len(cluster_ids)

feat_to_row = {f: i for i, f in enumerate(feat_ids)}
cluster_act = np.zeros((n_clusters, len(prompt_idxs)))
for j, cid in enumerate(cluster_ids):
    rows = [feat_to_row[f] for f in clusters[cid] if f in feat_to_row]
    cluster_act[j] = act[rows].mean(axis=0)

pm = pd.read_csv(PM).set_index("prompt_idx")
y = np.array([1 if pm.loc[p, "correct_answer"] == "alpha" else 0 for p in prompt_idxs])

# X: (n_prompts, n_clusters) — standardized
X_full = StandardScaler().fit_transform(cluster_act.T)


def effective_rank(X, thresholds=(0.90, 0.95, 0.99)):
    """Returns explained variance ratios + effective rank at given thresholds."""
    # SVD on centered, scaled X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = s**2
    var_ratio = var / var.sum()
    cum = np.cumsum(var_ratio)
    ranks = {f"rank_{int(t*100)}": int(np.searchsorted(cum, t) + 1) for t in thresholds}
    return ranks, var_ratio, cum, s


# ── Full 30-cluster ensemble ─────────────────────────────────────────────────
ranks_full, var_full, cum_full, s_full = effective_rank(X_full)

print(f"=== PCA Effective Rank for FULL 30-cluster ensemble ===\n")
print(f"Matrix shape: {X_full.shape}  (538 prompts × 30 clusters, standardized)")
print(f"\nExplained variance ratio per PC:")
for i in range(min(15, len(var_full))):
    bar = "█" * int(var_full[i] * 100)
    print(f"  PC{i+1:>2}  var={var_full[i]:.3f}  cum={cum_full[i]:.3f}  {bar}")
print(f"  ...")
print(f"  PC30 var={var_full[-1]:.4f}  cum={cum_full[-1]:.3f}")

print(f"\nEffective rank thresholds:")
for k, v in ranks_full.items():
    print(f"  {k}% variance → {v} components")

# ── L1-selected 14-cluster subset ────────────────────────────────────────────
w_df = pd.read_csv(WEIGHTS)
used_mask = w_df["abs_weight"] > 1e-6
used_idx = [cluster_ids.index(int(w_df.iloc[i]["cluster_id"])) for i in range(len(w_df)) if used_mask[i]]
X_sub = X_full[:, used_idx]
ranks_sub, var_sub, cum_sub, _ = effective_rank(X_sub)

print(f"\n=== PCA Effective Rank for L1-SELECTED 14-cluster ensemble ===\n")
print(f"Matrix shape: {X_sub.shape}")
print(f"\nExplained variance per PC:")
for i in range(min(14, len(var_sub))):
    bar = "█" * int(var_sub[i] * 100)
    print(f"  PC{i+1:>2}  var={var_sub[i]:.3f}  cum={cum_sub[i]:.3f}  {bar}")

print(f"\nEffective rank thresholds:")
for k, v in ranks_sub.items():
    print(f"  {k}% variance → {v} components")

# ── Compare with V_h signal direction ────────────────────────────────────────
# Project y (V_h labels) into PC space — does V_h align with top PCs?
# Use the centered y signal: y_centered = y - mean(y)
y_centered = y.astype(float) - y.mean()
U_full, _, _ = np.linalg.svd(X_full, full_matrices=False)
# Correlation between y and each PC
pc_corr = np.array([abs(np.corrcoef(U_full[:, i], y_centered)[0, 1]) for i in range(U_full.shape[1])])

print(f"\n=== Where does V_h signal live in PCA decomposition? ===")
print(f"Top-10 PCs by |corr(PC, V_h)|:")
top_pc_idx = np.argsort(-pc_corr)[:10]
for rank, pc_i in enumerate(top_pc_idx):
    print(f"  PC{pc_i+1:>2} (explains {var_full[pc_i]*100:.1f}% variance)  |corr with V_h| = {pc_corr[pc_i]:.3f}")

# How many top PCs explain 95% of V_h-correlated variance?
pc_corr_squared = pc_corr ** 2
sorted_corr_sq = np.sort(pc_corr_squared)[::-1]
cum_corr = np.cumsum(sorted_corr_sq) / sorted_corr_sq.sum()
n_pc_for_Vh = int(np.searchsorted(cum_corr, 0.90) + 1)
print(f"\nNumber of PCs to capture 90% of V_h-correlated signal: {n_pc_for_Vh}")
print(f"Number of PCs to capture 95% of V_h-correlated signal: {int(np.searchsorted(cum_corr, 0.95) + 1)}")

# ── Save ─────────────────────────────────────────────────────────────────────
summary = {
    "n_clusters_full": n_clusters,
    "n_prompts": len(prompt_idxs),
    "full_ensemble": {
        **ranks_full,
        "var_ratio_top10": [float(v) for v in var_full[:10]],
        "cum_var_top10": [float(c) for c in cum_full[:10]],
    },
    "L1_selected_subset": {
        "n_clusters_in_subset": len(used_idx),
        **ranks_sub,
        "var_ratio_top10": [float(v) for v in var_sub[:10]],
    },
    "Vh_alignment": {
        "top_PC_idx_by_Vh_corr": [int(i)+1 for i in top_pc_idx[:5]],
        "top_PC_corrs": [float(pc_corr[i]) for i in top_pc_idx[:5]],
        "n_pc_for_90pct_Vh": n_pc_for_Vh,
        "n_pc_for_95pct_Vh": int(np.searchsorted(cum_corr, 0.95) + 1),
    },
}
with open(OUT / "pca_effective_rank.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save per-PC table for plot
per_pc_df = pd.DataFrame({
    "PC": list(range(1, n_clusters + 1)),
    "var_ratio_full": var_full,
    "cum_var_full": cum_full,
    "corr_with_Vh": pc_corr,
})
per_pc_df.to_csv(OUT / "pca_effective_rank.csv", index=False)
print(f"\nSaved → {OUT}/pca_effective_rank.{{json,csv}}")
