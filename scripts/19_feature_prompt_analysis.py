"""
Script 19: Feature × prompt analysis for latent-state interpretation.

Creates a family of files for heatmap / clustering / latent-state analysis of
the physics_decay_type_probe ablation results.

Data sources used:
  - Ablation CSV (step 07): effect_size, abs_effect_size, sign_flipped,
    baseline_logit_diff, intervened_logit_diff — causal/ablation quantities
  - Graph JSON (UI run): community, grad_attr_sign, mean_activation_conditional,
    frequency — feature metadata from physics_decay_type corpus (not probe)
  - Probe JSONL (step 01): full prompt metadata
  - Baseline CSV (step 02): per-prompt sign accuracy

NOTE on quantities:
  effect_size > 0  → ablating feature reduced correct-answer logit → feature was HELPING
  effect_size < 0  → ablating feature increased correct-answer logit → feature was HURTING
  abs_effect_size  → causal importance regardless of direction
  sign_flipped     → strongest causal indicator: feature changed model's decision

  Activation values (top_k_values.npy) are on CSD3 only and are NOT included here.
  To add them, rsync:
    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/transcoder_features/layer_*/physics_decay_type_probe_train_top_k_{indices,values}.npy data/results/transcoder_features/layer_*/

Outputs:
  data/results/grouping/feature_prompt_contributions.csv  — full (feature, prompt) table
  data/results/grouping/feature_top_prompts.csv           — top-k prompts per feature
  data/results/grouping/feature_by_group_effect.csv       — (feature, group) aggregates
  data/results/grouping/feature_by_level_summary.csv      — (feature, level) aggregates
  data/results/grouping/feature_by_answer_summary.csv     — (feature, correct_answer)
  data/results/grouping/feature_prompt_effect_matrix.csv  — 40×470 heatmap matrix
  data/results/grouping/feature_group_effect_matrix.csv   — 40×N_groups heatmap matrix
  data/results/grouping/feature_group_sfr_matrix.csv      — 40×N_groups sfr matrix
  data/results/grouping/feature_metadata.csv              — one row per feature (graph info)
  data/results/grouping/prompt_metadata.csv               — one row per prompt (metadata)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data/results/grouping"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Canonical circuit features from physics_decay_type analysis ──────────────
# Top path: input→L22_F110496→L23_F83556→L24_F60777→L25_F71226→output
# 11-feature circuit (SLURM 28420228)
CIRCUIT_FEATURES = {
    "L22_F110496", "L23_F83556", "L24_F60777", "L25_F71226",
    "L24_F60777",  # appears in canonical path
}
# Newly identified global discriminators from CP test (probe run)
GLOBAL_DISCRIMINATORS_ALPHA = {"L24_F52031", "L24_F18943", "L24_F88968", "L24_F249"}
GLOBAL_DISCRIMINATORS_BETA  = {"L18_F145795", "L10_F128064", "L10_F35580", "L10_F80002", "L10_F83063"}


def load_graph_feature_metadata(ui_run_dir: Path) -> pd.DataFrame:
    """Load feature-level metadata from graph JSON."""
    g = json.load(open(ui_run_dir / "graph.json"))
    rows = []
    for n in g["nodes"]:
        if n.get("type") != "feature":
            continue
        fid = n["id"]
        rows.append({
            "feature_id":               fid,
            "layer":                    n["layer"],
            "feature_idx":              n["feature_idx"],
            "community":                n["community"],
            "grad_attr_sign":           n["grad_attr_sign"],
            "frequency_pdt":            round(n["frequency"], 4),          # on pdt corpus
            "mean_activation_pdt":      round(n["mean_activation_conditional"], 4),
            "mean_grad_attr_pdt":       round(n["mean_grad_attr_conditional"], 4),
            "mean_abs_grad_attr_pdt":   round(n["mean_abs_grad_attr_conditional"], 4),
            "n_prompts_pdt":            n["n_prompts"],
            "position_role":            n.get("position_role", ""),
            "causal_status":            n.get("causal_status", ""),
            "is_circuit_feature":       fid in CIRCUIT_FEATURES,
            "is_global_alpha_discrim":  fid in GLOBAL_DISCRIMINATORS_ALPHA,
            "is_global_beta_discrim":   fid in GLOBAL_DISCRIMINATORS_BETA,
        })
    df = pd.DataFrame(rows).sort_values(["layer", "feature_idx"])
    df["role_label"] = df.apply(lambda r:
        "α-circuit" if r["is_circuit_feature"] and r["grad_attr_sign"] > 0
        else "β-circuit" if r["is_circuit_feature"] and r["grad_attr_sign"] < 0
        else "α-discrim" if r["is_global_alpha_discrim"]
        else "β-discrim" if r["is_global_beta_discrim"]
        else "α-attr" if r["grad_attr_sign"] > 0
        else "β-attr", axis=1)
    return df


def load_prompt_metadata(behaviour: str, split: str) -> pd.DataFrame:
    """Load all prompt metadata from JSONL, indexed by prompt_idx (row order)."""
    rows = []
    for i, line in enumerate(open(ROOT / f"data/prompts/{behaviour}_{split}.jsonl")):
        p = json.loads(line)
        lv = p.get("level")
        rows.append({
            "prompt_idx":               i,
            "prompt_id":                p.get("prompt_id", f"p{i:04d}"),
            "prompt_text":              p["prompt"],
            "prompt_short":             p["prompt"][:80],
            "level":                    lv if lv is not None else "AUX",
            "level_label":              p.get("level_label", "auxiliary" if lv is None else f"L{lv}"),
            "group_id":                 p.get("group_id", "?"),
            "surface_family":           p.get("surface_family", ""),
            "cue_type":                 p.get("cue_type", ""),
            "cue_set":                  p.get("cue_set", ""),
            "relation_type":            p.get("relation_type", ""),
            "concept_route":            p.get("concept_route", ""),
            "cue_label":                (p.get("concept_route") or p.get("relation_type") or p.get("cue_type") or ""),
            "latent_state_target":      p.get("ic_concept_group", ""),
            "correct_answer":           p.get("correct_answer", "").strip(),
            "incorrect_answer":         p.get("incorrect_answer", "").strip(),
            "physics_concept":          p.get("physics_concept", ""),
            "difficulty":               p.get("difficulty", ""),
            "inference_steps":          p.get("inference_steps"),
            "wording_variant":          p.get("wording_variant"),
            "semantic_equiv_group":     p.get("semantic_equivalence_group", ""),
            "contrastive_pair_id":      p.get("contrastive_pair_id", ""),
            "contrastive_role":         p.get("contrastive_role", ""),
            "is_anchor":                bool(p.get("is_anchor", False)),
            "is_kw_variant":            bool(p.get("is_kw_variant", False)),
            "is_auxiliary":             bool(p.get("is_auxiliary", False)),
            "has_alpha_keyword":        bool(p.get("has_alpha_keyword", False)),
            "has_beta_keyword":         bool(p.get("has_beta_keyword", False)),
            "keyword_free":             bool(p.get("keyword_free", True)),
            "evidence_completeness":    p.get("evidence_completeness", ""),
            "is_uniquely_determining":  bool(p.get("is_uniquely_determining", True)),
        })
    return pd.DataFrame(rows)


def load_baseline(behaviour: str, split: str) -> pd.DataFrame:
    p = ROOT / f"data/results/baseline_{behaviour}_{split}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["prompt_idx"] = range(len(df))
    df["sign_correct"] = df["logprob_diff_normalized"] > 0
    df["hard_correct"] = df["logprob_diff_normalized"] > 0.5
    return df[["prompt_idx", "logprob_diff_normalized", "sign_correct", "hard_correct"]].rename(
        columns={"logprob_diff_normalized": "baseline_sign_diff"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour", default="physics_decay_type_probe")
    parser.add_argument("--split",     default="train")
    parser.add_argument("--ui_run",    type=Path, default=None)
    parser.add_argument("--top_k",     type=int, default=10, help="Top prompts per feature")
    args = parser.parse_args()

    # ── Locate UI run ───────────────────────────────────────────────────────
    if args.ui_run is None:
        runs = sorted((ROOT / "data/ui_offline").glob(f"*_{args.behaviour}_*"), reverse=True)
        if not runs:
            raise FileNotFoundError(f"No UI run found for {args.behaviour}")
        args.ui_run = runs[0]
    print(f"UI run: {args.ui_run.name}")

    # ── Load data ───────────────────────────────────────────────────────────
    print("Loading graph feature metadata...")
    feat_meta = load_graph_feature_metadata(args.ui_run)
    feat_dict = feat_meta.set_index("feature_id").to_dict("index")
    print(f"  {len(feat_meta)} feature nodes")

    print("Loading prompt metadata...")
    prompt_meta = load_prompt_metadata(args.behaviour, args.split)
    prompt_dict = prompt_meta.set_index("prompt_idx").to_dict("index")
    print(f"  {len(prompt_meta)} prompts")

    print("Loading ablation CSV...")
    abl_candidates = list((args.ui_run / "raw_sources").glob(f"intervention_ablation_{args.behaviour}*.csv"))
    if not abl_candidates:
        abl_candidates = list((ROOT / f"data/results/interventions/{args.behaviour}").glob(f"intervention_ablation_{args.behaviour}*.csv"))
    if not abl_candidates:
        raise FileNotFoundError("Ablation CSV not found")
    abl = pd.read_csv(abl_candidates[0])
    print(f"  {len(abl)} rows, {abl['feature_id'].nunique()} features, {abl['prompt_idx'].nunique()} prompts")

    print("Loading baseline CSV...")
    baseline = load_baseline(args.behaviour, args.split)

    # ── Build feature_prompt_contributions.csv ──────────────────────────────
    print("\nBuilding feature_prompt_contributions.csv ...")

    # Merge ablation with prompt metadata
    contrib = abl[["prompt_idx", "feature_id", "layer", "effect_size", "abs_effect_size",
                    "sign_flipped", "baseline_logit_diff", "intervened_logit_diff",
                    "relative_effect"]].copy()
    contrib["sign_flipped"] = contrib["sign_flipped"].astype(bool)

    # Add feature metadata from graph
    for col in ["community", "grad_attr_sign", "frequency_pdt", "mean_activation_pdt",
                "mean_abs_grad_attr_pdt", "position_role", "is_circuit_feature",
                "is_global_alpha_discrim", "is_global_beta_discrim", "role_label"]:
        contrib[col] = contrib["feature_id"].map(lambda fid: feat_dict.get(fid, {}).get(col))

    # Add prompt metadata
    for col in ["prompt_id", "prompt_short", "level", "level_label", "group_id",
                "cue_label", "cue_type", "relation_type", "concept_route",
                "latent_state_target", "correct_answer", "difficulty",
                "inference_steps", "wording_variant", "semantic_equiv_group",
                "contrastive_pair_id", "contrastive_role",
                "is_anchor", "is_kw_variant", "is_auxiliary", "keyword_free",
                "has_alpha_keyword", "has_beta_keyword", "is_uniquely_determining"]:
        contrib[col] = contrib["prompt_idx"].map(lambda i: prompt_dict.get(i, {}).get(col))

    # Add baseline sign accuracy per prompt
    if not baseline.empty:
        bl = baseline.set_index("prompt_idx")
        contrib["baseline_sign_diff"] = contrib["prompt_idx"].map(bl["baseline_sign_diff"])
        contrib["sign_correct"]        = contrib["prompt_idx"].map(bl["sign_correct"])
        contrib["hard_correct"]        = contrib["prompt_idx"].map(bl["hard_correct"])

    # Sort
    contrib = contrib.sort_values(["feature_id", "prompt_idx"]).reset_index(drop=True)

    out = OUT / "feature_prompt_contributions.csv"
    contrib.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(contrib)} rows)")

    # ── Save feature_metadata.csv ────────────────────────────────────────────
    feat_out = OUT / "feature_metadata.csv"
    feat_meta.to_csv(feat_out, index=False)
    print(f"  Saved: {feat_out}  ({len(feat_meta)} features)")

    # ── Save prompt_metadata.csv ─────────────────────────────────────────────
    if not baseline.empty:
        prompt_meta = prompt_meta.merge(baseline, on="prompt_idx", how="left")
    pout = OUT / "prompt_metadata.csv"
    prompt_meta.to_csv(pout, index=False)
    print(f"  Saved: {pout}  ({len(prompt_meta)} prompts)")

    # ── Build feature_top_prompts.csv ────────────────────────────────────────
    print("\nBuilding feature_top_prompts.csv ...")
    top_rows = []
    keep_cols = ["prompt_idx", "prompt_id", "prompt_short", "level", "group_id",
                 "cue_label", "latent_state_target", "correct_answer",
                 "is_anchor", "is_kw_variant", "is_auxiliary",
                 "baseline_sign_diff", "sign_correct"]

    for metric, ascending in [("abs_effect_size", False), ("effect_size", False)]:
        for fid, grp in contrib.groupby("feature_id"):
            top = grp.nlargest(args.top_k, metric) if not ascending else grp.nsmallest(args.top_k, metric)
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                r = {"feature_id": fid, "layer": row["layer"],
                     "ranking_metric": metric, "rank": rank,
                     "metric_value": row[metric],
                     "community": row["community"],
                     "role_label": row["role_label"]}
                for c in keep_cols:
                    r[c] = row.get(c)
                top_rows.append(r)

    top_df = pd.DataFrame(top_rows)
    tpath = OUT / "feature_top_prompts.csv"
    top_df.to_csv(tpath, index=False)
    print(f"  Saved: {tpath}  ({len(top_df)} rows, {args.top_k} prompts × 40 features × 2 metrics)")

    # ── Aggregated: feature × group ──────────────────────────────────────────
    print("\nBuilding aggregated tables ...")

    def aggregate(df, group_cols, value_col_pairs):
        """Aggregate multiple (value, agg_fn) pairs over group_cols."""
        result = df.groupby(group_cols).agg(
            n_prompts=("prompt_idx", "nunique"),
            n_nonzero=("abs_effect_size", lambda x: (x > 0).sum()),
            mean_effect=("effect_size", "mean"),
            mean_abs_effect=("abs_effect_size", "mean"),
            max_abs_effect=("abs_effect_size", "max"),
            sfr=("sign_flipped", "mean"),
            n_flip=("sign_flipped", "sum"),
        ).reset_index()
        return result

    # feature × group
    fg = aggregate(contrib, ["feature_id", "group_id"], [])
    # add feature meta cols
    for col in ["layer", "community", "grad_attr_sign", "role_label"]:
        fg[col] = fg["feature_id"].map(lambda fid: feat_dict.get(fid, {}).get(col))
    # add group-level info (level, latent_state_target, correct_answer, cue_label)
    grp_info = contrib.groupby("group_id").first()[["level", "latent_state_target", "correct_answer", "cue_label"]].reset_index()
    fg = fg.merge(grp_info, on="group_id", how="left")
    fg = fg.sort_values(["feature_id", "level", "group_id"])
    fg.to_csv(OUT / "feature_by_group_effect.csv", index=False)
    print(f"  Saved: feature_by_group_effect.csv  ({len(fg)} rows)")

    # feature × level
    fl = aggregate(contrib, ["feature_id", "level"], [])
    for col in ["layer", "community", "grad_attr_sign", "role_label"]:
        fl[col] = fl["feature_id"].map(lambda fid: feat_dict.get(fid, {}).get(col))
    fl = fl.sort_values(["feature_id", "level"])
    fl.to_csv(OUT / "feature_by_level_summary.csv", index=False)
    print(f"  Saved: feature_by_level_summary.csv  ({len(fl)} rows)")

    # feature × correct_answer
    fa = aggregate(contrib, ["feature_id", "correct_answer"], [])
    for col in ["layer", "community", "grad_attr_sign", "role_label"]:
        fa[col] = fa["feature_id"].map(lambda fid: feat_dict.get(fid, {}).get(col))
    fa.to_csv(OUT / "feature_by_answer_summary.csv", index=False)
    print(f"  Saved: feature_by_answer_summary.csv  ({len(fa)} rows)")

    # feature × latent_state_target (α vs β)
    ft = aggregate(contrib, ["feature_id", "latent_state_target"], [])
    for col in ["layer", "community", "grad_attr_sign", "role_label"]:
        ft[col] = ft["feature_id"].map(lambda fid: feat_dict.get(fid, {}).get(col))
    ft.to_csv(OUT / "feature_by_target_summary.csv", index=False)
    print(f"  Saved: feature_by_target_summary.csv  ({len(ft)} rows)")

    # ── Heatmap matrices ─────────────────────────────────────────────────────
    print("\nBuilding heatmap matrices ...")

    # Feature × prompt (effect_size) — 40 × 470
    feat_ids = sorted(contrib["feature_id"].unique())
    prompt_ids = sorted(contrib["prompt_idx"].unique())

    pivot_effect = contrib.pivot_table(
        index="feature_id", columns="prompt_idx", values="effect_size", aggfunc="first"
    ).reindex(index=feat_ids, columns=prompt_ids)
    pivot_effect.to_csv(OUT / "feature_prompt_effect_matrix.csv")
    print(f"  Saved: feature_prompt_effect_matrix.csv  ({pivot_effect.shape[0]} features × {pivot_effect.shape[1]} prompts)")

    # Feature × prompt (abs_effect_size)
    pivot_abs = contrib.pivot_table(
        index="feature_id", columns="prompt_idx", values="abs_effect_size", aggfunc="first"
    ).reindex(index=feat_ids, columns=prompt_ids)
    pivot_abs.to_csv(OUT / "feature_prompt_abs_effect_matrix.csv")
    print(f"  Saved: feature_prompt_abs_effect_matrix.csv")

    # Feature × group (mean_abs_effect)
    fg_pivot = fg.pivot_table(index="feature_id", columns="group_id", values="mean_abs_effect", aggfunc="first")
    fg_pivot.to_csv(OUT / "feature_group_effect_matrix.csv")
    print(f"  Saved: feature_group_effect_matrix.csv  ({fg_pivot.shape[0]} features × {fg_pivot.shape[1]} groups)")

    # Feature × group (sfr)
    fg_sfr = fg.pivot_table(index="feature_id", columns="group_id", values="sfr", aggfunc="first")
    fg_sfr.to_csv(OUT / "feature_group_sfr_matrix.csv")
    print(f"  Saved: feature_group_sfr_matrix.csv")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY (top features by mean abs effect on probe corpus)")
    print("=" * 60)
    feat_summary = contrib.groupby("feature_id").agg(
        layer=("layer", "first"),
        community=("community", "first"),
        role=("role_label", "first"),
        mean_abs_effect=("abs_effect_size", "mean"),
        max_abs_effect=("abs_effect_size", "max"),
        sfr=("sign_flipped", "mean"),
        mean_effect_alpha=("effect_size", lambda x: x[contrib.loc[x.index, "latent_state_target"] == "alpha"].mean()),
        mean_effect_beta=("effect_size", lambda x: x[contrib.loc[x.index, "latent_state_target"] == "beta"].mean()),
    ).sort_values("mean_abs_effect", ascending=False)

    print(f"{'Feature':<18} {'L':>3} {'C':>2} {'Role':<12} {'MeanAbs':>8} {'MaxAbs':>8} {'SFR':>6} {'α_eff':>7} {'β_eff':>7}")
    print("-" * 80)
    for fid, r in feat_summary.iterrows():
        print(f"  {fid:<16} {r['layer']:>3} {r['community']:>2} {str(r['role']):<12} "
              f"{r['mean_abs_effect']:>7.4f}  {r['max_abs_effect']:>7.4f}  {r['sfr']:>5.3f}  "
              f"{r['mean_effect_alpha']:>6.3f}  {r['mean_effect_beta']:>6.3f}")

    print(f"\nAll outputs saved to: {OUT}")
    print("\nNote: Activation values (top_k_values.npy) are on CSD3.")
    print("To add them locally, rsync:")
    print("  rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/transcoder_features/layer_*/physics_decay_type_probe_train_top_k_values.npy data/results/transcoder_features/layer_*/")
    print("  rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/transcoder_features/layer_*/physics_decay_type_probe_train_top_k_indices.npy data/results/transcoder_features/layer_*/")


if __name__ == "__main__":
    main()
