#!/usr/bin/env python3
"""
Script 22: Prepare derived matrix representations for the clustering benchmark.

Reads grouping CSVs, constructs multiple data views, and saves them to
data/results/clustering/ as .npy files + JSON metadata.

Outputs
-------
data/results/clustering/
  feat_prompt_abs.npy      (40, 470) abs ablation effects
  feat_prompt_signed.npy   (40, 470) signed ablation effects
  feat_group_abs.npy       (40, 132) group-level mean abs effect
  feat_group_signed.npy    (40, 132) group-level mean signed effect
  feat_group_sfr.npy       (40, 132) group-level sign-flip rate
  feat_group_act.npy       (40, 132) group-level mean activation
  W_abs_cosine.npy         (40, 40)  feature–feature cosine similarity from abs
  W_signed_cosine.npy      (40, 40)  feature–feature cosine similarity from signed
  W_pearson_abs.npy        (40, 40)  feature–feature Pearson r from abs
  W_coimportance.npy       (40, 40)  co-importance Jaccard graph (top-k prompts)
  feat_delta_abs.npy       (40,)     scalar: mean_alpha_abs - mean_beta_abs
  feat_residual.npy        (40, 470) prompt-level residuals after subtracting group mean
  feat_ids.json            feature ID list (row order for all feat × * matrices)
  group_ids.json           group ID list (column order for feat_group_* matrices)
  prompt_idxs.json         prompt index list (column order for feat_prompt_* matrices)
  prompt_labels.json       per-prompt metadata for stratified evaluation
  feat_labels.json         per-feature known labels (circuit, role, community, etc.)
  subsets.json             index sets for level / answer / core subsets
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
OUT = ROOT / "data/results/clustering"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K_COIMP = 10   # top-k prompts per feature for co-importance Jaccard


def pearson_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation matrix."""
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xn = Xc / norms
    return Xn @ Xn.T


def main():
    print("Loading source CSVs...")
    abs_df  = pd.read_csv(GROUPING / "feature_prompt_abs_effect_matrix.csv",   index_col=0)
    eff_df  = pd.read_csv(GROUPING / "feature_prompt_effect_matrix.csv",        index_col=0)
    grp_eff = pd.read_csv(GROUPING / "feature_group_effect_matrix.csv",         index_col=0)
    grp_sfr = pd.read_csv(GROUPING / "feature_group_sfr_matrix.csv",            index_col=0)
    grp_act = pd.read_csv(GROUPING / "feature_group_activation_matrix.csv",     index_col=0)
    pm      = pd.read_csv(GROUPING / "prompt_metadata.csv")
    fm      = pd.read_csv(GROUPING / "feature_metadata.csv")
    fa      = pd.read_csv(GROUPING / "feature_by_answer_summary.csv")
    contrib = pd.read_csv(GROUPING / "feature_prompt_contributions.csv")
    grp_sum = pd.read_csv(GROUPING / "probe_group_summary.csv")

    feat_ids   = list(abs_df.index)
    group_ids  = list(grp_eff.columns)
    prompt_idxs = [int(c) for c in abs_df.columns]
    n_feat, n_prompt = abs_df.shape
    n_group = grp_eff.shape[1]
    print(f"  {n_feat} features × {n_prompt} prompts × {n_group} groups")

    # Only keep features in ablation matrix (40 of the 69 in feature_metadata)
    fm_ab = fm[fm.feature_id.isin(feat_ids)].set_index("feature_id").reindex(feat_ids)

    # ── Raw matrices ─────────────────────────────────────────────────────────────
    X_abs    = abs_df.values.astype(np.float32)    # (40, 470)
    X_signed = eff_df.values.astype(np.float32)
    X_ga     = grp_eff.values.astype(np.float32)  # (40, 132)
    X_gsfr   = grp_sfr.values.astype(np.float32)
    X_gact   = grp_act.values.astype(np.float32)

    # For group-level abs, derive from grp_eff carefully
    # feature_group_effect_matrix holds mean signed effect; abs is not stored directly
    # use the by_group CSV for proper abs values
    by_grp = pd.read_csv(GROUPING / "feature_by_group_effect.csv")
    # Build group_abs from pivot
    ga_abs_df = by_grp.pivot_table(
        index="feature_id", columns="group_id", values="mean_abs_effect", aggfunc="first"
    ).reindex(index=feat_ids, columns=group_ids)
    X_ga_abs = ga_abs_df.fillna(0.0).values.astype(np.float32)

    # ── Feature-feature similarity matrices ──────────────────────────────────────
    print("Computing feature–feature similarity matrices...")
    # L2-normalise rows so cosine = dot product
    X_abs_norm    = X_abs    / (np.linalg.norm(X_abs,    axis=1, keepdims=True) + 1e-12)
    X_signed_norm = X_signed / (np.linalg.norm(X_signed, axis=1, keepdims=True) + 1e-12)
    X_ga_abs_norm = X_ga_abs / (np.linalg.norm(X_ga_abs, axis=1, keepdims=True) + 1e-12)

    W_abs_cos    = (X_abs_norm    @ X_abs_norm.T).clip(-1, 1).astype(np.float32)
    W_signed_cos = (X_signed_norm @ X_signed_norm.T).clip(-1, 1).astype(np.float32)
    W_pearson    = pearson_rows(X_abs).clip(-1, 1).astype(np.float32)
    np.fill_diagonal(W_abs_cos, 1.0)
    np.fill_diagonal(W_signed_cos, 1.0)
    np.fill_diagonal(W_pearson, 1.0)

    # ── Co-importance Jaccard graph (novel) ──────────────────────────────────────
    print("Building co-importance Jaccard graph (novel)...")
    # For each feature, find top-k prompts by abs effect
    top_k_sets = []
    for i, fid in enumerate(feat_ids):
        vals = X_abs[i]
        top_idx = set(np.argsort(vals)[::-1][:TOP_K_COIMP].tolist())
        top_k_sets.append(top_idx)

    W_coimp = np.zeros((n_feat, n_feat), dtype=np.float32)
    for i in range(n_feat):
        for j in range(i, n_feat):
            inter = len(top_k_sets[i] & top_k_sets[j])
            union = len(top_k_sets[i] | top_k_sets[j])
            v = inter / union if union > 0 else 0.0
            W_coimp[i, j] = v
            W_coimp[j, i] = v

    # ── Delta abs (α − β scalar per feature, novel) ─────────────────────────────
    print("Computing α–β contrastive delta...")
    fa_idx = fa.set_index(["feature_id", "correct_answer"])
    delta_abs = np.zeros(n_feat, dtype=np.float32)
    for i, fid in enumerate(feat_ids):
        a = fa_idx.loc[(fid, "alpha"), "mean_abs_effect"] if (fid, "alpha") in fa_idx.index else 0.0
        b = fa_idx.loc[(fid, "beta"),  "mean_abs_effect"] if (fid, "beta")  in fa_idx.index else 0.0
        delta_abs[i] = float(a) - float(b)

    # ── Residual matrix (novel) ──────────────────────────────────────────────────
    print("Computing residual matrix (subtract group mean per prompt)...")
    # Map prompt_idx → group_id
    pid2gid = dict(zip(pm.prompt_idx, pm.group_id))
    # For each prompt column, subtract the group mean for that feature
    X_residual = X_signed.copy()
    for col_j, pidx in enumerate(prompt_idxs):
        gid = pid2gid.get(pidx)
        if gid and gid in group_ids:
            gcol = group_ids.index(gid)
            X_residual[:, col_j] -= X_ga[:, gcol]

    # ── Subsets of prompts ───────────────────────────────────────────────────────
    pm_sorted = pm.set_index("prompt_idx").reindex(prompt_idxs)
    level_idx  = {lv: [j for j, p in enumerate(prompt_idxs) if pm_sorted.loc[p, "level"] == lv]
                  for lv in ["1","2","3","AUX"]}
    answer_idx = {ans: [j for j, p in enumerate(prompt_idxs) if pm_sorted.loc[p, "correct_answer"] == ans]
                  for ans in ["alpha","beta"]}
    core_idx   = [j for j, p in enumerate(prompt_idxs)
                  if pm_sorted.loc[p, "level"] in ("1","2")
                  or pm_sorted.loc[p, "group_id"] in ("L3-FA","L3-FB")]
    pass_idx   = [j for j, p in enumerate(prompt_idxs) if pm_sorted.loc[p, "sign_correct"]]
    fail_idx   = [j for j, p in enumerate(prompt_idxs) if not pm_sorted.loc[p, "sign_correct"]]

    subsets = {
        "level": {k: v for k,v in level_idx.items()},
        "answer": answer_idx,
        "core": core_idx,
        "pass": pass_idx,
        "fail": fail_idx,
    }

    # ── Feature labels for evaluation ────────────────────────────────────────────
    feat_labels = {}
    for fid in feat_ids:
        row = fm_ab.loc[fid]
        feat_labels[fid] = {
            "layer":       int(row.layer),
            "community":   int(row.community) if pd.notna(row.community) else -1,
            "role_label":  str(row.role_label),
            "is_circuit":  bool(row.is_circuit_feature),
            "is_alpha_d":  bool(row.is_global_alpha_discrim),
            "is_beta_d":   bool(row.is_global_beta_discrim),
            "grad_sign":   int(row.grad_attr_sign),
        }

    # ── Prompt labels ─────────────────────────────────────────────────────────────
    prompt_labels = {}
    for j, pidx in enumerate(prompt_idxs):
        row = pm_sorted.loc[pidx]
        prompt_labels[str(pidx)] = {
            "j":            j,
            "level":        str(row.level),
            "group_id":     str(row.group_id),
            "correct_answer": str(row.correct_answer),
            "sign_correct": bool(row.sign_correct),
            "is_anchor":    bool(row.is_anchor),
            "is_kw":        bool(row.is_kw_variant),
            "is_aux":       bool(row.is_auxiliary),
        }

    # ── Save ─────────────────────────────────────────────────────────────────────
    print("Saving...")

    def sv(arr, name):
        path = OUT / name
        np.save(path, arr)
        print(f"  {name}: {arr.shape}  [{arr.min():.3f}, {arr.max():.3f}]")

    sv(X_abs,       "feat_prompt_abs.npy")
    sv(X_signed,    "feat_prompt_signed.npy")
    sv(X_ga,        "feat_group_signed.npy")
    sv(X_ga_abs,    "feat_group_abs.npy")
    sv(X_gsfr,      "feat_group_sfr.npy")
    sv(X_gact,      "feat_group_act.npy")
    sv(W_abs_cos,   "W_abs_cosine.npy")
    sv(W_signed_cos,"W_signed_cosine.npy")
    sv(W_pearson,   "W_pearson_abs.npy")
    sv(W_coimp,     "W_coimportance.npy")
    sv(delta_abs,   "feat_delta_abs.npy")
    sv(X_residual,  "feat_residual.npy")

    def sj(obj, name):
        path = OUT / name
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"  {name}")

    sj(feat_ids,      "feat_ids.json")
    sj(group_ids,     "group_ids.json")
    sj(prompt_idxs,   "prompt_idxs.json")
    sj(feat_labels,   "feat_labels.json")
    sj(prompt_labels, "prompt_labels.json")
    sj(subsets,       "subsets.json")

    # Summary stats
    print(f"\nDone.  Features={n_feat}, Prompts={n_prompt}, Groups={n_group}")
    print(f"  co-importance Jaccard range: [{W_coimp.min():.3f}, {W_coimp.max():.3f}]")
    print(f"  delta_abs range: [{delta_abs.min():.3f}, {delta_abs.max():.3f}]")
    print(f"  residual range: [{X_residual.min():.3f}, {X_residual.max():.3f}]")
    print(f"  core prompts: {len(core_idx)}, fail: {len(fail_idx)}")


if __name__ == "__main__":
    main()
