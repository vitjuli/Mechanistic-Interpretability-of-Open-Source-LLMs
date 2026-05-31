"""
diag_pca_causal_metric_148.py

Re-run causal-metric PCA with Σ_within computed over **148 semantic-equivalence
groups**, not binary V_h. This is the meaningful version: it whitens out
paraphrase noise (within-cue variation), then asks which between-cue direction
the cluster activations are organized around.

Key question: is V_h direction γ̄ the dominant between-cue PC, or just one of
many between-cue concepts?

If γ̄ ≈ PC1 after 148-group whitening → V_h is THE main between-cue concept
                                          (method B in pilot can stay 1-D)
If γ̄ spread across multiple PCs        → V_h is one of multiple concepts
                                          (method B needs r > 1, multidim subspace)

Compares with the binary-V_h whitening from diag_pca_causal_metric.py.

Output: data/analysis/iia_failure_diagnosis/pca_causal_metric_148.json
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

# ── Load + aggregate ────────────────────────────────────────────────────────
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
sem_grp = np.array([str(pm.loc[p, "semantic_equiv_group"]) for p in prompt_idxs])

X = StandardScaler().fit_transform(cluster_act.T)
n, d = X.shape
y_c = y - y.mean()
print(f"X: {X.shape}  |  semantic-equivalence-groups: {len(set(sem_grp))}")

# ── V_h direction γ̄ from logistic regression on X ───────────────────────────
clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000)
clf.fit(X, y)
gamma = clf.coef_[0]
gamma_unit = gamma / np.linalg.norm(gamma)

# ── Σ_within over 148 semantic-equivalence-groups ────────────────────────────
unique_groups = sorted(set(sem_grp))
within_sum = np.zeros((d, d))
n_eff = 0
group_sizes = []
for g in unique_groups:
    mask = sem_grp == g
    n_g = mask.sum()
    if n_g < 2:
        continue   # singleton can't contribute to within-class variance
    Xg = X[mask]
    mu_g = Xg.mean(0)
    Xg_c = Xg - mu_g
    within_sum += Xg_c.T @ Xg_c
    n_eff += n_g - 1
    group_sizes.append(n_g)

Sigma_within_148 = within_sum / n_eff
print(f"Within-class groups used (n≥2): {len(group_sizes)} / {len(unique_groups)}")
print(f"Group size: median={int(np.median(group_sizes))}, "
      f"min={min(group_sizes)}, max={max(group_sizes)}, total n_eff={n_eff}")
print(f"Σ_within_148 rank: {np.linalg.matrix_rank(Sigma_within_148)}, "
      f"min eigval: {np.linalg.eigh(Sigma_within_148)[0].min():.6f}")

# ── Whitening ────────────────────────────────────────────────────────────────
eigvals, eigvecs = np.linalg.eigh(Sigma_within_148)
eigvals = np.maximum(eigvals, 1e-6)
W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

X_w = X @ W
gamma_w = W @ gamma_unit

# ── PCA on whitened ─────────────────────────────────────────────────────────
Xc = X_w - X_w.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
var_per_pc = s**2 / (n - 1)
var_ratio = var_per_pc / var_per_pc.sum()
cum = np.cumsum(var_ratio)

# Alignment of γ_w with each PC
g_w_unit = gamma_w / np.linalg.norm(gamma_w)
alignment = np.array([abs(g_w_unit @ Vt[i]) for i in range(d)])

# Correlation of each PC scores with V_h
pc_corr_Vh = np.array([abs(np.corrcoef(U[:, i], y_c)[0, 1]) for i in range(d)])

# Top-K PCs aligned with γ̄ — how many PCs to cover γ̄?
align_sq = alignment**2  # sums to ||γ_w_unit||² = 1
sorted_idx = np.argsort(-align_sq)
cum_align = np.cumsum(align_sq[sorted_idx])
rank_gamma_90 = int(np.searchsorted(cum_align, 0.90) + 1)
rank_gamma_95 = int(np.searchsorted(cum_align, 0.95) + 1)
rank_gamma_99 = int(np.searchsorted(cum_align, 0.99) + 1)

print(f"\n=== PCA after whitening by Σ_within (148 cue-groups) ===")
print(f"PC1 var={var_ratio[0]*100:5.2f}%  cum={cum[0]*100:5.2f}%")
print(f"PC2 var={var_ratio[1]*100:5.2f}%  cum={cum[1]*100:5.2f}%")
print(f"PC3 var={var_ratio[2]*100:5.2f}%  cum={cum[2]*100:5.2f}%")
print(f"PC4 var={var_ratio[3]*100:5.2f}%  cum={cum[3]*100:5.2f}%")
print(f"PC5 var={var_ratio[4]*100:5.2f}%  cum={cum[4]*100:5.2f}%")
print(f"...")
print(f"\nTop-5 PCs by |cos(γ̄_w, PC)|:")
print(f"{'rank':>4} {'PC_idx':>6} {'var%':>7} {'cos(γ̄,PC)':>10} {'|corr(PC,V_h)|':>14} {'cum_γ̄_share':>13}")
for i in range(5):
    pc_i = sorted_idx[i]
    print(f"{i+1:>4} {pc_i+1:>6} {var_ratio[pc_i]*100:>6.2f}% "
          f"{alignment[pc_i]:>10.3f} {pc_corr_Vh[pc_i]:>14.3f} {cum_align[i]:>13.3f}")

print(f"\nNumber of PCs to cover γ̄ direction:")
print(f"  90% of γ̄ direction → {rank_gamma_90} PCs")
print(f"  95% of γ̄ direction → {rank_gamma_95} PCs")
print(f"  99% of γ̄ direction → {rank_gamma_99} PCs")

# Compare with binary V_h whitening
print(f"\n=== COMPARISON ===")
print(f"{'Metric':>45}  {'PC1↔γ̄':>8}  {'PC1 corr V_h':>14}  {'#PCs for γ̄(90%)':>17}")
print("-"*90)
print(f"{'Σ_within(2-class V_h) [tautological]':>45}  {0.818:>8.3f}  {0.758:>14.3f}  {'1 (by construction)':>17}")
print(f"{'Σ_within(148 cue-groups) [meaningful]':>45}  "
      f"{alignment[sorted_idx[0]]:>8.3f}  "
      f"{pc_corr_Vh[sorted_idx[0]]:>14.3f}  "
      f"{rank_gamma_90:>15}d")

# ── Verdict ──────────────────────────────────────────────────────────────────
top_align = alignment[sorted_idx[0]]
top_pc_var = var_ratio[sorted_idx[0]]
top_corr = pc_corr_Vh[sorted_idx[0]]

print(f"\n=== VERDICT for pilot design ===")
if top_align > 0.7 and top_pc_var > 0.10 and rank_gamma_90 <= 3:
    verdict = "STRONG: γ̄ is dominant between-cue direction → pilot can use 1-D method (B≈C)"
elif top_align > 0.5 and rank_gamma_90 <= 6:
    verdict = "MODERATE: γ̄ multi-dimensional but compact → pilot method B with r∈[2,6]"
else:
    verdict = "DISTRIBUTED: γ̄ spread across many PCs → method B needs r large (>6), or γ̄ not main between-cue concept"
print(f"  Verdict: {verdict}")
print(f"  Pilot recommendation: r = {rank_gamma_90}")

# ── Save ─────────────────────────────────────────────────────────────────────
summary = {
    "n_clusters": d,
    "n_groups_used": len(group_sizes),
    "n_eff_within": n_eff,
    "PC1_var_ratio": float(var_ratio[0]),
    "PC1_corr_with_Vh": float(pc_corr_Vh[0]),
    "top5_by_gamma_alignment": [
        {"rank": i+1, "pc_idx": int(sorted_idx[i])+1,
         "var_ratio": float(var_ratio[sorted_idx[i]]),
         "cos_with_gamma": float(alignment[sorted_idx[i]]),
         "corr_with_Vh": float(pc_corr_Vh[sorted_idx[i]]),
         "cum_gamma_share": float(cum_align[i])} for i in range(5)
    ],
    "rank_for_gamma_90": rank_gamma_90,
    "rank_for_gamma_95": rank_gamma_95,
    "rank_for_gamma_99": rank_gamma_99,
    "verdict": verdict,
}
with open(OUT / "pca_causal_metric_148.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {OUT}/pca_causal_metric_148.json")
