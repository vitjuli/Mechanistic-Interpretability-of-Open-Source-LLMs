#!/usr/bin/env python3
"""
Script 21: Prepare derived JSON artifacts for the Latent State Explorer dashboard tab.

Reads:  data/results/grouping/*.csv
Writes: dashboard_probe/public/data/
  explorer_prompts.json     — per-prompt: metadata, t-SNE position, top features, causal weight
  explorer_features.json    — per-feature: all summary stats, by-level, by-group
  explorer_similarity.json  — prompt–prompt cosine similarity edges (top-k)
  explorer_bipartite.json   — prompt–feature bipartite edges

Usage:
  python scripts/21_prepare_explorer_ui.py [--dashboard dashboard_probe]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent.parent
GROUPING_DIR = ROOT / "data/results/grouping"


def _f(v):
    """Convert numpy scalar to Python float/int/bool, pass None through."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _s(v):
    """Convert to str, return None for NaN."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def main(dashboard: str):
    out_dir = ROOT / dashboard / "public/data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source files...")
    abs_mat_df   = pd.read_csv(GROUPING_DIR / "feature_prompt_abs_effect_matrix.csv", index_col=0)
    contrib_df   = pd.read_csv(GROUPING_DIR / "feature_prompt_contributions.csv")
    prompt_meta  = pd.read_csv(GROUPING_DIR / "prompt_metadata.csv")
    feat_meta    = pd.read_csv(GROUPING_DIR / "feature_metadata.csv")
    feat_by_ans  = pd.read_csv(GROUPING_DIR / "feature_by_answer_summary.csv")
    feat_by_lv   = pd.read_csv(GROUPING_DIR / "feature_by_level_summary.csv")
    feat_by_grp  = pd.read_csv(GROUPING_DIR / "feature_by_group_effect.csv")
    feat_top_df  = pd.read_csv(GROUPING_DIR / "feature_top_prompts.csv")
    grp_summary  = pd.read_csv(GROUPING_DIR / "probe_group_summary.csv")

    n_feats, n_prompts = abs_mat_df.shape
    feat_ids = list(abs_mat_df.index)  # 40 features with ablation data
    print(f"  {n_feats} features × {n_prompts} prompts")

    # Only use features that appear in the ablation matrix
    feat_meta = feat_meta[feat_meta["feature_id"].isin(feat_ids)].copy()

    # ── 1. t-SNE embedding of prompts ──────────────────────────────────────────
    print("Computing t-SNE embedding (perp=30, n_iter=1000)...")
    # Prompt vectors = columns of abs_mat_df → shape (n_prompts, n_feats)
    prompt_vectors = abs_mat_df.values.T          # (470, 40)
    prompt_vectors_norm = normalize(prompt_vectors)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, n_prompts - 1),
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    positions = tsne.fit_transform(prompt_vectors_norm)   # (470, 2)

    # Scale to [-10, 10]
    for i in range(2):
        r = positions[:, i].max() - positions[:, i].min()
        if r > 0:
            positions[:, i] = (positions[:, i] - positions[:, i].mean()) / r * 20

    # ── 2. Prompt–prompt cosine similarity graph ───────────────────────────────
    print("Computing prompt–prompt cosine similarity...")
    sim_matrix = cosine_similarity(prompt_vectors_norm)   # (470, 470)
    np.fill_diagonal(sim_matrix, 0.0)

    TOP_K_SIM = 7
    sim_edges = []
    for i in range(n_prompts):
        top_j = np.argsort(sim_matrix[i])[::-1][:TOP_K_SIM]
        for j in top_j:
            if j > i and sim_matrix[i, j] > 0.5:
                sim_edges.append({
                    "s": int(i),
                    "t": int(j),
                    "v": round(float(sim_matrix[i, j]), 4),
                })
    print(f"  {len(sim_edges)} similarity edges (cosine > 0.5, top-{TOP_K_SIM}/prompt)")

    # ── 3. Per-prompt summary ──────────────────────────────────────────────────
    print("Building prompt summaries...")
    causal_weights = abs_mat_df.values.T.mean(axis=1)   # mean abs effect over all feats

    # Top-5 features per prompt by abs_effect_size
    top_per_prompt: dict[int, list] = {}
    for pid, grp in contrib_df.groupby("prompt_idx"):
        top5 = grp.nlargest(5, "abs_effect_size")
        top_per_prompt[int(pid)] = [
            {
                "f":   str(r["feature_id"]),
                "a":   round(float(r["abs_effect_size"]), 4),
                "e":   round(float(r["effect_size"]), 4),
                "sf":  bool(r["sign_flipped"]),
                "act": round(float(r["activation_value"]), 3) if pd.notna(r["activation_value"]) else None,
            }
            for _, r in top5.iterrows()
        ]

    # Group-level sign accuracy for lookup
    grp_sign_acc = {}
    if "sign_acc" in grp_summary.columns and "group_id" in grp_summary.columns:
        for _, r in grp_summary.iterrows():
            grp_sign_acc[str(r["group_id"])] = _f(r["sign_acc"])

    prompts_out = []
    for _, row in prompt_meta.iterrows():
        pidx = int(row["prompt_idx"])
        gid  = str(row["group_id"])
        prompts_out.append({
            "idx":    pidx,
            "id":     str(row["prompt_id"]),
            "short":  str(row["prompt_short"])[:80],
            "level":  str(row["level"]),
            "group_id": gid,
            "cue_label": _s(row["cue_label"]),
            "cue_type":  _s(row["cue_type"]),
            "correct_answer": str(row["correct_answer"]),
            "latent_state_target": _s(row.get("latent_state_target")),
            "difficulty": _s(row.get("difficulty")),
            "inference_steps": int(row["inference_steps"]) if pd.notna(row.get("inference_steps")) else None,
            "is_anchor": bool(row["is_anchor"]),
            "is_kw":     bool(row["is_kw_variant"]),
            "is_aux":    bool(row["is_auxiliary"]),
            "sign_correct": bool(row["sign_correct"]),
            "hard_correct": bool(row.get("hard_correct", row["sign_correct"])),
            "baseline_diff": round(float(row["baseline_sign_diff"]), 4) if pd.notna(row.get("baseline_sign_diff")) else None,
            "causal_weight": round(float(causal_weights[pidx]), 4),
            "x":   round(float(positions[pidx, 0]), 3),
            "y":   round(float(positions[pidx, 1]), 3),
            "top_feats": top_per_prompt.get(pidx, []),
            "group_sign_acc": grp_sign_acc.get(gid),
        })

    # ── 4. Per-feature summary ─────────────────────────────────────────────────
    print("Building feature summaries...")

    # Lookups from by-answer/level/group CSVs
    ans_lu:  dict[tuple, pd.Series] = {}
    for _, r in feat_by_ans.iterrows():
        ans_lu[(r["feature_id"], r["correct_answer"])] = r

    lv_lu: dict[tuple, pd.Series] = {}
    for _, r in feat_by_lv.iterrows():
        lv_lu[(r["feature_id"], str(r["level"]))] = r

    grp_lu: dict[str, list] = {}
    for _, r in feat_by_grp.iterrows():
        grp_lu.setdefault(r["feature_id"], []).append(r)

    # Top-10 prompts per feature per ranking_metric
    top_feats_lu: dict[str, dict] = {}
    for fid, fgrp in feat_top_df.groupby("feature_id"):
        top_feats_lu[str(fid)] = {}
        for metric, mgrp in fgrp.groupby("ranking_metric"):
            top_feats_lu[str(fid)][str(metric)] = [
                {
                    "idx": int(r["prompt_idx"]),
                    "short": str(r["prompt_short"])[:60],
                    "level": str(r["level"]),
                    "gid": str(r["group_id"]),
                    "ans": str(r["correct_answer"]),
                    "v": round(float(r["metric_value"]), 4),
                    "ok": bool(r["sign_correct"]),
                }
                for _, r in mgrp.head(10).iterrows()
            ]

    # Count connected prompts per feature (abs_effect > 0.1)
    conn_counts = (
        contrib_df[contrib_df["abs_effect_size"] > 0.1]
        .groupby("feature_id")["prompt_idx"]
        .nunique()
        .to_dict()
    )

    features_out = []
    for _, fm in feat_meta.iterrows():
        fid = str(fm["feature_id"])
        a_r = ans_lu.get((fid, "alpha"))
        b_r = ans_lu.get((fid, "beta"))

        by_level: dict[str, dict] = {}
        for lv in ("1", "2", "3", "AUX"):
            r = lv_lu.get((fid, lv))
            if r is not None:
                by_level[lv] = {
                    "n":    int(r["n_prompts"]),
                    "mae":  round(float(r["mean_abs_effect"]), 4),
                    "me":   round(float(r["mean_effect"]), 4),
                    "sfr":  round(float(r["sfr"]), 4),
                    "act":  round(float(r["mean_activation"]), 3) if "mean_activation" in r.index and pd.notna(r["mean_activation"]) else None,
                }

        by_group = sorted(
            [
                {
                    "gid": str(r["group_id"]),
                    "lv":  str(r["level"]),
                    "ans": str(r["correct_answer"]),
                    "n":   int(r["n_prompts"]),
                    "mae": round(float(r["mean_abs_effect"]), 4),
                    "me":  round(float(r["mean_effect"]), 4),
                    "sfr": round(float(r["sfr"]), 4),
                    "act": round(float(r["mean_activation"]), 3) if "mean_activation" in r.index and pd.notna(r["mean_activation"]) else None,
                }
                for r in (grp_lu.get(fid) or [])
            ],
            key=lambda x: -x["mae"],
        )[:30]

        features_out.append({
            "id":          fid,
            "layer":       int(fm["layer"]),
            "community":   int(fm["community"]) if pd.notna(fm["community"]) else None,
            "role":        str(fm["role_label"]),
            "is_circuit":  bool(fm["is_circuit_feature"]),
            "is_alpha_d":  bool(fm["is_global_alpha_discrim"]),
            "is_beta_d":   bool(fm["is_global_beta_discrim"]),
            "grad_sign":   int(fm["grad_attr_sign"]),
            "freq":        round(float(fm["frequency_pdt"]), 4),
            "mean_act":    round(float(fm["mean_activation_pdt"]), 3),
            "n_conn":      conn_counts.get(fid, 0),
            # alpha stats
            "a_mae":  round(float(a_r["mean_abs_effect"]), 4) if a_r is not None else None,
            "a_me":   round(float(a_r["mean_effect"]),     4) if a_r is not None else None,
            "a_sfr":  round(float(a_r["sfr"]),             4) if a_r is not None else None,
            # beta stats
            "b_mae":  round(float(b_r["mean_abs_effect"]), 4) if b_r is not None else None,
            "b_me":   round(float(b_r["mean_effect"]),     4) if b_r is not None else None,
            "b_sfr":  round(float(b_r["sfr"]),             4) if b_r is not None else None,
            # delta
            "delta_me": round(float(a_r["mean_effect"]) - float(b_r["mean_effect"]), 4)
                        if (a_r is not None and b_r is not None) else None,
            "by_level": by_level,
            "by_group": by_group,
            "top_prompts": top_feats_lu.get(fid, {}),
        })

    # ── 5. Bipartite edges ─────────────────────────────────────────────────────
    print("Building bipartite edges...")
    bip_rows = contrib_df[contrib_df["abs_effect_size"] > 0][
        ["prompt_idx", "feature_id", "abs_effect_size", "effect_size", "sign_flipped"]
    ]
    bipartite_edges = [
        {
            "p":  int(r["prompt_idx"]),
            "f":  str(r["feature_id"]),
            "a":  round(float(r["abs_effect_size"]), 4),
            "e":  round(float(r["effect_size"]),     4),
            "sf": bool(r["sign_flipped"]),
        }
        for _, r in bip_rows.iterrows()
    ]
    print(f"  {len(bipartite_edges)} bipartite edges")

    # ── Save ───────────────────────────────────────────────────────────────────
    def save(obj, name):
        path = out_dir / name
        with open(path, "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
        print(f"  Saved {name}  ({path.stat().st_size / 1024:.1f} KB)")

    print("Saving...")
    save(prompts_out,                           "explorer_prompts.json")
    save(features_out,                          "explorer_features.json")
    save({"top_k": TOP_K_SIM, "edges": sim_edges}, "explorer_similarity.json")
    save({"edges": bipartite_edges},            "explorer_bipartite.json")

    print(f"\nDone:  {len(prompts_out)} prompts, {len(features_out)} features")
    print(f"       {len(sim_edges)} similarity edges, {len(bipartite_edges)} bipartite edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Latent State Explorer data.")
    parser.add_argument("--dashboard", default="dashboard_probe",
                        help="Target dashboard directory (default: dashboard_probe)")
    args = parser.parse_args()
    main(args.dashboard)
