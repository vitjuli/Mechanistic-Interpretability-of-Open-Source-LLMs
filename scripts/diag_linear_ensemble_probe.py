"""
diag_linear_ensemble_probe.py

Approach 1: L1-regularized logistic regression on cluster activations to find
minimal sufficient linear ensemble that predicts V_h (α/β).

Math: V_h(p) ≈ sign(Σ_i w_i · A_{C_i}(p) + b)

where A_{C_i}(p) = mean activation of cluster C_i features on prompt p.

L1 penalty → automatic sparse selection: clusters with w_i = 0 are excluded
from the minimal ensemble.

Outputs:
  data/analysis/iia_failure_diagnosis/linear_ensemble_weights.csv
  data/analysis/iia_failure_diagnosis/linear_ensemble_summary.json
  stdout: weights, accuracy, sparsity, comparison
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).parent.parent
ACT  = ROOT / "data/analysis/runD_v2/activations"
CSUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp/feature_subgroup_assignments.csv"
PM   = ROOT / "data/analysis/runD_v2/grouping/prompt_metadata.csv"
OUT  = ROOT / "data/analysis/iia_failure_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
act = np.load(ACT / "activation_matrix.npy")  # (n_feat, n_prompts)
feat_ids = (ACT / "feature_ids.txt").read_text().strip().split("\n")
prompt_idxs = [int(x) for x in (ACT / "prompt_idxs.txt").read_text().strip().split("\n")]

cl = pd.read_csv(CSUB)
fid_to_cid = dict(zip(cl["feature_id"], cl["subgroup_cluster"]))
fid_to_layer = dict(zip(cl["feature_id"], cl["layer"]))

clusters = defaultdict(list)
for fid in feat_ids:
    if fid in fid_to_cid:
        clusters[fid_to_cid[fid]].append(fid)
cluster_ids = sorted(clusters.keys())
n_clusters = len(cluster_ids)

print(f"Activations: {act.shape}  |  Sub-clusters: {n_clusters}")

# Aggregate to cluster level: cluster_act[n_clusters, n_prompts]
feat_to_row = {f: i for i, f in enumerate(feat_ids)}
cluster_act = np.zeros((n_clusters, len(prompt_idxs)))
cluster_meta = []
for j, cid in enumerate(cluster_ids):
    rows = [feat_to_row[f] for f in clusters[cid] if f in feat_to_row]
    cluster_act[j] = act[rows].mean(axis=0)
    layers = sorted(set(fid_to_layer[f] for f in clusters[cid] if f in fid_to_layer))
    cluster_meta.append({"cluster_id": int(cid), "n_features": len(rows),
                         "layer_min": min(layers), "layer_max": max(layers)})

# Labels
pm = pd.read_csv(PM).set_index("prompt_idx")
y = np.array([1 if pm.loc[p, "correct_answer"] == "alpha" else 0 for p in prompt_idxs])
print(f"Labels: {y.sum()} α, {(1-y).sum()} β")

# Z-score cluster activations (so L1 weights are comparable)
X = cluster_act.T  # (n_prompts, n_clusters)
X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

# ── L1 logistic regression with multiple C values ────────────────────────────
print(f"\n=== L1-regularized Logistic Regression ===")
print(f"{'C':>8} {'n_used':>7} {'train_acc':>9} {'cv_acc(5fold)':>13}")
print("-"*50)
results = []
for C in [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
    clf.fit(X, y)
    w = clf.coef_[0]
    n_used = int((np.abs(w) > 1e-6).sum())
    train_acc = clf.score(X, y)
    cv_scores = cross_val_score(
        LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy"
    )
    cv_acc = cv_scores.mean()
    cv_std = cv_scores.std()
    results.append({"C": C, "n_used": n_used, "train_acc": train_acc,
                    "cv_acc": cv_acc, "cv_std": cv_std,
                    "weights": w.tolist()})
    print(f"{C:>8.3f} {n_used:>7d} {train_acc:>9.3f} {cv_acc:>9.3f}±{cv_std:.3f}")

# Pick "best" sparse model: smallest n_used such that cv_acc ≥ 0.95 * max(cv_acc)
max_cv = max(r["cv_acc"] for r in results)
best_sparse = min((r for r in results if r["cv_acc"] >= 0.95 * max_cv),
                  key=lambda r: r["n_used"])
print(f"\n→ Best sparse model: C={best_sparse['C']}, "
      f"{best_sparse['n_used']}/{n_clusters} clusters used, "
      f"CV acc = {best_sparse['cv_acc']:.3f}")

# ── L2 (dense) baseline for comparison ──────────────────────────────────────
clf_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
clf_l2.fit(X, y)
cv_l2 = cross_val_score(
    LogisticRegression(penalty="l2", C=1.0, max_iter=2000),
    X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
).mean()
print(f"L2 dense baseline (all 30 clusters): CV acc = {cv_l2:.3f}, train = {clf_l2.score(X, y):.3f}")

# ── Random baseline ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
rand_scores = []
for _ in range(20):
    y_perm = rng.permutation(y)
    clf_r = LogisticRegression(penalty="l1", solver="liblinear", C=best_sparse["C"], max_iter=2000)
    cv_r = cross_val_score(clf_r, X, y_perm, cv=5).mean()
    rand_scores.append(cv_r)
print(f"Random (shuffled labels) baseline CV: {np.mean(rand_scores):.3f} ± {np.std(rand_scores):.3f}")

# ── Per-cluster weight from best sparse model ───────────────────────────────
best_weights = np.array(best_sparse["weights"])
print(f"\n=== Cluster weights (best sparse model, C={best_sparse['C']}) ===")
print(f"{'cluster_id':>10} {'layer':>6} {'nF':>3} {'weight':>9} {'role':>15}")
print("-"*55)
wdf = pd.DataFrame(cluster_meta)
wdf["weight"] = best_weights
wdf["abs_weight"] = np.abs(best_weights)
wdf["role"] = wdf["weight"].apply(lambda w: "α-supporter" if w > 0.01
                                           else "β-supporter" if w < -0.01
                                           else "ZERO (excluded)")
wdf = wdf.sort_values("weight", ascending=False)
for _, r in wdf.iterrows():
    w_str = f"{r['weight']:+.3f}"
    L_str = f"L{int(r['layer_min'])}" if r['layer_min'] == r['layer_max'] else \
            f"L{int(r['layer_min'])}-{int(r['layer_max'])}"
    print(f"{int(r['cluster_id']):>10d} {L_str:>6} {int(r['n_features']):>3d} "
          f"{w_str:>9} {r['role']:>15}")

# Used clusters only
used = wdf[wdf["abs_weight"] > 1e-6].copy()
print(f"\n=== Minimal sufficient ensemble: {len(used)} clusters out of {n_clusters} ===")
print(f"  α-supporters: {(used['weight'] > 0).sum()}")
print(f"  β-supporters: {(used['weight'] < 0).sum()}")
print(f"  Total features used: {used['n_features'].sum()}")
print(f"  Layers covered: {sorted(set(used['layer_min'].tolist() + used['layer_max'].tolist()))}")

# ── Save ─────────────────────────────────────────────────────────────────────
wdf.to_csv(OUT / "linear_ensemble_weights.csv", index=False)
summary = {
    "n_clusters_total": n_clusters,
    "n_clusters_used_in_minimal_ensemble": int(len(used)),
    "minimal_ensemble_cv_acc": float(best_sparse["cv_acc"]),
    "minimal_ensemble_train_acc": float(best_sparse["train_acc"]),
    "minimal_ensemble_C": float(best_sparse["C"]),
    "l2_dense_cv_acc": float(cv_l2),
    "random_baseline_cv_acc": float(np.mean(rand_scores)),
    "random_baseline_std": float(np.std(rand_scores)),
    "n_features_total_used": int(used["n_features"].sum()),
    "alpha_supporters_in_ensemble": int((used["weight"] > 0).sum()),
    "beta_supporters_in_ensemble": int((used["weight"] < 0).sum()),
    "regularization_sweep": [{"C": r["C"], "n_used": r["n_used"],
                              "cv_acc": r["cv_acc"]} for r in results],
}
with open(OUT / "linear_ensemble_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {OUT}/linear_ensemble_weights.csv + linear_ensemble_summary.json")
