"""
diag_pca_causal_metric.py

Re-do PCA on cluster activations in a CAUSAL metric (whitened by Σ^{-1/2})
instead of Euclidean. Hypothesis (Park & Veitch / Geiger causal inner product):
after whitening, share of variance along V_h direction γ̄ should rise sharply,
while format-direction variance should be suppressed.

Two whitenings tested:
  (1) Full-covariance whitening:    W = Σ^{-1/2},  Σ = Cov(X)
       — analog of Park & Veitch causal inner product
       — removes overall variance structure, exposes concept-direction
  (2) Within-class whitening:       W = Σ_within^{-1/2}
       — Fisher LDA-style: normalizes out within-class noise
       — V_h becomes the maximum-variance direction by construction

Compares:
  - Euclidean PCA (Approach 3)        — V_h on PC2/PC3, PC1 = format
  - Full-cov whitened PCA             — does V_h rise?
  - Within-class whitened PCA         — V_h should dominate

Output:
  data/analysis/iia_failure_diagnosis/pca_causal_metric.{json,csv}
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
ACT  = ROOT / "data/analysis/runD_v2/activations"
CSUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp/feature_subgroup_assignments.csv"
PM   = ROOT / "data/analysis/runD_v2/grouping/prompt_metadata.csv"
OUT  = ROOT / "data/analysis/iia_failure_diagnosis"

# ── Load + aggregate to cluster activations ─────────────────────────────────
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

feat_to_row = {f: i for i, f in enumerate(feat_ids)}
cluster_act = np.zeros((len(cluster_ids), len(prompt_idxs)))
for j, cid in enumerate(cluster_ids):
    rows = [feat_to_row[f] for f in clusters[cid] if f in feat_to_row]
    cluster_act[j] = act[rows].mean(axis=0)

pm = pd.read_csv(PM).set_index("prompt_idx")
y = np.array([1 if pm.loc[p, "correct_answer"] == "alpha" else 0 for p in prompt_idxs])

X_raw = cluster_act.T   # (538, 30)
X = StandardScaler().fit_transform(X_raw)
n, d = X.shape
y_c = y - y.mean()
print(f"X: {X.shape}, n_alpha={y.sum()}, n_beta={(1-y).sum()}")


def variance_decomposition(X, gamma, label=""):
    """Compute total variance, variance along gamma, and PC structure."""
    Xc = X - X.mean(axis=0)
    # SVD
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_per_pc = s**2 / (len(Xc) - 1)
    total_var = var_per_pc.sum()
    var_ratio = var_per_pc / total_var
    cum = np.cumsum(var_ratio)
    rank_90 = int(np.searchsorted(cum, 0.90) + 1)
    rank_95 = int(np.searchsorted(cum, 0.95) + 1)

    # Project onto gamma (unit-normalized)
    gamma_unit = gamma / np.linalg.norm(gamma)
    proj = Xc @ gamma_unit
    var_gamma = proj.var(ddof=1)
    frac_gamma = var_gamma / total_var

    # Alignment of gamma with each PC
    # gamma in PC basis: gamma · v_i
    alignment = np.array([abs(gamma_unit @ Vt[i]) for i in range(d)])

    # Top-3 PC indices most aligned with gamma
    top3 = np.argsort(-alignment)[:3]
    print(f"\n--- {label} ---")
    print(f"  total var          = {total_var:.3f}")
    print(f"  var along gamma    = {var_gamma:.4f}  (frac = {frac_gamma*100:.2f}%)")
    print(f"  effective rank 90% = {rank_90}")
    print(f"  effective rank 95% = {rank_95}")
    print(f"  top-3 PCs aligned with gamma:")
    for i, pc in enumerate(top3):
        print(f"    PC{pc+1:>2}: var_ratio={var_ratio[pc]*100:5.2f}%, "
              f"|cos(gamma, PC)|={alignment[pc]:.3f}, "
              f"|corr(PC, V_h)|={abs(np.corrcoef(U[:,pc], y_c)[0,1]):.3f}")
    return dict(
        total_var=float(total_var),
        var_along_gamma=float(var_gamma),
        frac_var_along_gamma=float(frac_gamma),
        rank_90=rank_90,
        rank_95=rank_95,
        top3_PC=[int(p)+1 for p in top3],
        top3_alignment=[float(alignment[p]) for p in top3],
        top3_pc_corr_with_Vh=[float(abs(np.corrcoef(U[:,p], y_c)[0,1])) for p in top3],
        pc1_corr_Vh=float(abs(np.corrcoef(U[:,0], y_c)[0,1])),
        pc1_var_ratio=float(var_ratio[0]),
    )


# ── Step 1: V_h direction γ̄ (from logistic regression L1) ─────────────────
clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000)
clf.fit(X, y)
gamma = clf.coef_[0]   # (30,)
print(f"\nγ̄ (V_h direction from L1 logreg): ||γ||={np.linalg.norm(gamma):.3f}, "
      f"non-zero={(np.abs(gamma) > 1e-6).sum()}/{d}")

# ── Euclidean PCA (baseline) ────────────────────────────────────────────────
res_eucl = variance_decomposition(X, gamma, label="EUCLIDEAN (standard PCA)")

# ── Whitening 1: Full covariance ────────────────────────────────────────────
Sigma_full = np.cov(X.T)
eigvals, eigvecs = np.linalg.eigh(Sigma_full)
# Avoid numerical issues for tiny eigenvalues
eigvals = np.maximum(eigvals, 1e-8)
W_full = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # Σ^{-1/2}
X_w_full = X @ W_full
# γ in whitened basis
gamma_w_full = W_full @ gamma   # direction in new basis
res_full = variance_decomposition(X_w_full, gamma_w_full,
                                  label="WHITENED by full Σ (Park & Veitch causal inner product)")

# ── Whitening 2: Within-class covariance (Fisher LDA) ───────────────────────
X_a = X[y == 1]
X_b = X[y == 0]
mu_a, mu_b = X_a.mean(0), X_b.mean(0)
Sigma_within = ((X_a - mu_a).T @ (X_a - mu_a) + (X_b - mu_b).T @ (X_b - mu_b)) / (len(X) - 2)
eigvals_w, eigvecs_w = np.linalg.eigh(Sigma_within)
eigvals_w = np.maximum(eigvals_w, 1e-8)
W_within = eigvecs_w @ np.diag(1.0 / np.sqrt(eigvals_w)) @ eigvecs_w.T
X_w_within = X @ W_within
gamma_w_within = W_within @ gamma
res_within = variance_decomposition(X_w_within, gamma_w_within,
                                    label="WHITENED by Σ_within (Fisher LDA)")

# ── Comparison summary ──────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"COMPARISON: where does V_h sit in each metric?")
print(f"{'='*80}")
print(f"{'Metric':>50}  {'PC1 var':>8}  {'PC1↔V_h':>8}  {'γ frac of total var':>20}")
print("-"*100)
print(f"{'Euclidean':>50}  {res_eucl['pc1_var_ratio']*100:>7.2f}%  "
      f"{res_eucl['pc1_corr_Vh']:>8.3f}  {res_eucl['frac_var_along_gamma']*100:>19.2f}%")
print(f"{'Whitened by Σ_full (causal IP)':>50}  {res_full['pc1_var_ratio']*100:>7.2f}%  "
      f"{res_full['pc1_corr_Vh']:>8.3f}  {res_full['frac_var_along_gamma']*100:>19.2f}%")
print(f"{'Whitened by Σ_within (Fisher LDA)':>50}  {res_within['pc1_var_ratio']*100:>7.2f}%  "
      f"{res_within['pc1_corr_Vh']:>8.3f}  {res_within['frac_var_along_gamma']*100:>19.2f}%")

# Fold rise in γ-variance share
fold_full   = res_full['frac_var_along_gamma'] / res_eucl['frac_var_along_gamma']
fold_within = res_within['frac_var_along_gamma'] / res_eucl['frac_var_along_gamma']
print(f"\nγ̄-variance share rise (whitened vs Euclidean):")
print(f"  full-cov whitening:    {fold_full:.2f}x")
print(f"  within-class whitening:{fold_within:.2f}x")

# ── Save ─────────────────────────────────────────────────────────────────────
summary = dict(
    Euclidean=res_eucl,
    whitened_full=res_full,
    whitened_within=res_within,
    fold_rise_full=float(fold_full),
    fold_rise_within=float(fold_within),
    n_features_used_by_gamma=int((np.abs(gamma) > 1e-6).sum()),
)
with open(OUT / "pca_causal_metric.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {OUT}/pca_causal_metric.json")
