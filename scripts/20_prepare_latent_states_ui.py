"""
Script 20: Build latent_states.json for the Latent States dashboard tab.

Reads the grouping CSVs produced by scripts 18+19 and creates a single
compact JSON consumed by the LatentStatesTab React component.

Usage:
    python scripts/20_prepare_latent_states_ui.py
    # or point at a specific dashboard:
    python scripts/20_prepare_latent_states_ui.py --dashboard dashboard_physics

Output:
    dashboard_physics/public/data/latent_states.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRP  = ROOT / "data/results/grouping"

# Groups to exclude from heatmap columns (keep clean)
EXCLUDE_GROUPS = set()  # add noisy group_ids here if needed

# Core group definition (matches script 18)
CORE_GROUPS_ANCHOR = {"L3-FA", "L3-FB"}
def is_core(row):
    lv = row.get("level")
    gid = row.get("group_id", "")
    if lv in (1, 2):
        return True
    if gid in CORE_GROUPS_ANCHOR:
        return True
    return False


# ── Sorting helpers ────────────────────────────────────────────────────────────
LEVEL_ORDER = {1: 0, 2: 1, 3: 2, "AUX": 3, None: 3}
TARGET_ORDER = {"alpha": 0, "beta": 1, "": 2, None: 2}

def group_sort_key(g):
    lv = g.get("level", "AUX")
    target = g.get("correct_answer", "").strip().replace(" ", "")
    gid = g.get("group_id", "")
    # special order: anchors last within L3
    anchor = 1 if gid.startswith("L3-F") else 0
    return (LEVEL_ORDER.get(lv, 3), TARGET_ORDER.get(target, 2), anchor, gid)

def feature_sort_key(f):
    # Sort: layer asc, then by Δeff desc (most α-promoting first)
    return (f.get("layer", 99), -f.get("delta_eff", 0))


def safe_float(v, ndigits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), ndigits)


def load_csv(name):
    p = GRP / name
    if not p.exists():
        print(f"  [WARN] {name} not found, skipping")
        return pd.DataFrame()
    return pd.read_csv(p)


def main(dashboard: str):
    out_dir = ROOT / dashboard / "public/data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading grouping CSVs...")
    feat_meta     = load_csv("feature_metadata.csv")
    feat_grp      = load_csv("feature_by_group_effect.csv")
    feat_lv       = load_csv("feature_by_level_summary.csv")
    feat_ans      = load_csv("feature_by_answer_summary.csv")
    prompt_meta   = load_csv("prompt_metadata.csv")
    top_prompts   = load_csv("feature_top_prompts.csv")
    grp_summary   = load_csv("probe_group_summary.csv")
    lv_summary    = load_csv("probe_level_summary.csv")
    ic_df         = load_csv("probe_ic_analysis.csv")
    se_df         = load_csv("probe_se_analysis.csv")
    cp_df         = load_csv("probe_cp_analysis.csv")
    kw_df         = load_csv("probe_kw_analysis.csv")
    contrib       = load_csv("feature_prompt_contributions.csv")

    # ── Build feature objects ──────────────────────────────────────────────────
    print("Building feature objects...")

    # Filter to only the 40 ablated features (appear in feat_grp)
    ablated_ids = set(feat_grp["feature_id"].unique()) if not feat_grp.empty else set()
    if feat_meta.empty:
        features = []
    else:
        feat_meta_40 = feat_meta[feat_meta["feature_id"].isin(ablated_ids)].copy() \
                       if ablated_ids else feat_meta.copy()

        # Per-answer stats (α vs β effect and activation)
        ans_dict = {}
        if not feat_ans.empty:
            for _, r in feat_ans.iterrows():
                key = (r["feature_id"], str(r.get("correct_answer","")).strip().replace(" ",""))
                ans_dict[key] = r

        # Per-level stats (level stored as string in CSV: '1','2','3','AUX')
        lv_dict = {}
        if not feat_lv.empty:
            for _, r in feat_lv.iterrows():
                lv_dict[(r["feature_id"], str(r["level"]))] = r

        # Per-group effect dict for this feature
        fg_dict = {}
        if not feat_grp.empty:
            for _, r in feat_grp.iterrows():
                fg_dict[(r["feature_id"], r["group_id"])] = r

        # Top prompts per feature
        tp_dict = {}
        if not top_prompts.empty:
            for fid, grp in top_prompts.groupby("feature_id"):
                tp_dict[fid] = {}
                for metric, mgrp in grp.groupby("ranking_metric"):
                    tp_dict[fid][metric] = [
                        {
                            "rank":       int(r["rank"]),
                            "prompt_short": str(r.get("prompt_short",""))[:80],
                            "level":       r.get("level"),
                            "group_id":    str(r.get("group_id","")),
                            "cue_label":   str(r.get("cue_label","") or ""),
                            "target":      str(r.get("latent_state_target","")),
                            "answer":      str(r.get("correct_answer","")).strip(),
                            "metric_value": safe_float(r.get("metric_value"), 4),
                            "is_anchor":   bool(r.get("is_anchor", False)),
                            "is_kw":       bool(r.get("is_kw_variant", False)),
                            "sign_correct": bool(r.get("sign_correct", True)),
                        }
                        for _, r in mgrp.sort_values("rank").iterrows()
                    ]

        # IC results per feature
        ic_by_gid = {}
        if not ic_df.empty:
            for _, r in ic_df.iterrows():
                ic_by_gid[r["group_id"]] = {
                    "sim_to_anchor": safe_float(r.get("sim_to_anchor")),
                    "concept": str(r.get("concept","")),
                    "level": r.get("level"),
                    "anchor_id": str(r.get("anchor_id","")),
                }

        # Compute Δact and Δeff from contrib
        delta_dict = {}
        if not contrib.empty:
            for fid, fgrp in contrib.groupby("feature_id"):
                alpha_rows = fgrp[fgrp["latent_state_target"] == "alpha"]
                beta_rows  = fgrp[fgrp["latent_state_target"] == "beta"]
                delta_dict[fid] = {
                    "alpha_act": safe_float(alpha_rows["activation_value"].mean()) if "activation_value" in fgrp.columns else None,
                    "beta_act":  safe_float(beta_rows["activation_value"].mean()) if "activation_value" in fgrp.columns else None,
                    "alpha_eff": safe_float(alpha_rows["effect_size"].mean()),
                    "beta_eff":  safe_float(beta_rows["effect_size"].mean()),
                    "mean_act":  safe_float(fgrp["activation_value"].mean()) if "activation_value" in fgrp.columns else None,
                    "pct_active": safe_float((fgrp.get("activation_value", pd.Series([0]*len(fgrp))) > 0).mean(), 3),
                }

        features = []
        for _, fm in feat_meta_40.iterrows():
            fid = fm["feature_id"]
            d = delta_dict.get(fid, {})
            a_act = d.get("alpha_act"); b_act = d.get("beta_act")
            a_eff = d.get("alpha_eff"); b_eff = d.get("beta_eff")

            # by-level stats
            by_level = {}
            for lv in ["1", "2", "3", "AUX"]:
                r = lv_dict.get((fid, lv))
                if r is not None:
                    by_level[str(lv)] = {
                        "mean_abs_effect":  safe_float(r.get("mean_abs_effect")),
                        "mean_effect":      safe_float(r.get("mean_effect")),
                        "sfr":              safe_float(r.get("sfr"), 3),
                        "mean_activation":  safe_float(r.get("mean_activation")),
                    }

            features.append({
                "id":                  fid,
                "layer":               int(fm["layer"]),
                "community":           int(fm["community"]) if pd.notna(fm.get("community")) else None,
                "grad_attr_sign":      int(fm.get("grad_attr_sign", 0) or 0),
                "role_label":          str(fm.get("role_label", "")),
                "is_circuit":          bool(fm.get("is_circuit_feature", False)),
                "is_alpha_discrim":    bool(fm.get("is_global_alpha_discrim", False)),
                "is_beta_discrim":     bool(fm.get("is_global_beta_discrim", False)),
                "frequency_pdt":       safe_float(fm.get("frequency_pdt"), 3),
                "mean_activation_pdt": safe_float(fm.get("mean_activation_pdt"), 3),
                "mean_act":            d.get("mean_act"),
                "pct_active":          d.get("pct_active"),
                "alpha_act":           a_act,
                "beta_act":            b_act,
                "delta_act":           safe_float((a_act - b_act) if (a_act is not None and b_act is not None) else None),
                "alpha_eff":           a_eff,
                "beta_eff":            b_eff,
                "delta_eff":           safe_float((a_eff - b_eff) if (a_eff is not None and b_eff is not None) else None),
                "by_level":            by_level,
                "top_prompts":         tp_dict.get(fid, {}),
            })

        features.sort(key=feature_sort_key)
        print(f"  {len(features)} features")

    # ── Build group objects ────────────────────────────────────────────────────
    print("Building group objects...")

    # Merge group summary with prompt metadata for group-level info
    grp_meta_rows = {}
    if not prompt_meta.empty:
        for gid, g in prompt_meta.groupby("group_id"):
            first = g.iloc[0]
            grp_meta_rows[gid] = {
                "level":       first.get("level"),
                "correct_answer": str(first.get("correct_answer","")).strip(),
                "cue_label":   str(first.get("cue_label","") or first.get("cue_type","") or ""),
                "is_anchor":   bool(g["is_anchor"].any()),
                "is_kw":       bool(g["is_kw_variant"].any()),
                "is_auxiliary":bool(g["is_auxiliary"].any()),
            }

    grp_acc = {}  # group_id -> {sign_acc, n, mean_diff}
    if not grp_summary.empty:
        for _, r in grp_summary.iterrows():
            grp_acc[str(r.get("group_id",""))] = {
                "n": int(r.get("n", 0)),
                "sign_acc": safe_float(r.get("sign_acc"), 3),
                "hard_acc":  safe_float(r.get("hard_acc"), 3),
                "mean_diff": safe_float(r.get("mean_diff"), 3),
            }

    # Collect all group IDs from feat_grp
    all_gids = sorted(feat_grp["group_id"].unique()) if not feat_grp.empty else []

    groups = []
    for gid in all_gids:
        if gid in EXCLUDE_GROUPS:
            continue
        meta = grp_meta_rows.get(gid, {})
        acc  = grp_acc.get(gid, {})
        lv   = meta.get("level")
        grp_obj = {
            "group_id":      gid,
            "level":         lv,
            "correct_answer": meta.get("correct_answer", ""),
            "cue_label":     meta.get("cue_label", ""),
            "is_anchor":     meta.get("is_anchor", False),
            "is_kw":         meta.get("is_kw", False),
            "is_auxiliary":  meta.get("is_auxiliary", False),
            "is_core":       is_core({"level": lv, "group_id": gid}),
            "n":             acc.get("n"),
            "sign_acc":      acc.get("sign_acc"),
            "hard_acc":      acc.get("hard_acc"),
            "mean_diff":     acc.get("mean_diff"),
            "ic_sim":        ic_by_gid.get(gid, {}).get("sim_to_anchor") if not ic_df.empty else None,
        }
        groups.append(grp_obj)

    groups.sort(key=group_sort_key)
    print(f"  {len(groups)} groups")

    # ── Build heatmap matrices ─────────────────────────────────────────────────
    print("Building matrices...")
    feat_ids  = [f["id"] for f in features]
    group_ids = [g["group_id"] for g in groups]

    def build_matrix(metric_col):
        if feat_grp.empty:
            return []
        matrix = []
        for fid in feat_ids:
            row = []
            for gid in group_ids:
                r = feat_grp[(feat_grp["feature_id"] == fid) & (feat_grp["group_id"] == gid)]
                if len(r) == 0:
                    row.append(None)
                else:
                    v = r.iloc[0].get(metric_col)
                    row.append(safe_float(v))
            matrix.append(row)
        return matrix

    matrices = {
        "mean_abs_effect": build_matrix("mean_abs_effect"),
        "mean_effect":     build_matrix("mean_effect"),
        "sfr":             build_matrix("sfr"),
        "mean_activation": build_matrix("mean_activation"),
    }

    # ── Assemble SE/IC/CP/KW ─────────────────────────────────────────────────
    def df_to_records(df):
        if df.empty:
            return []
        df = df.copy()
        for col in df.select_dtypes(include=[float]).columns:
            df[col] = df[col].round(4)
        return df.where(pd.notna(df), None).to_dict(orient="records")

    # ── Level summary ──────────────────────────────────────────────────────────
    level_rows = []
    if not lv_summary.empty:
        level_rows = df_to_records(lv_summary)

    # ── Final JSON ─────────────────────────────────────────────────────────────
    payload = {
        "meta": {
            "behaviour": "physics_decay_type_probe",
            "n_features": len(features),
            "n_groups":   len(groups),
            "n_prompts":  470,
        },
        "features":     features,
        "groups":       groups,
        "feat_ids":     feat_ids,
        "group_ids":    group_ids,
        "matrices":     matrices,
        "ic_analysis":  df_to_records(ic_df),
        "se_analysis":  df_to_records(se_df),
        "cp_analysis":  df_to_records(cp_df),
        "kw_analysis":  df_to_records(kw_df),
        "level_summary": level_rows,
    }

    out_path = out_dir / "latent_states.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved: {out_path}  ({size_kb:.1f} KB)")
    print(f"  features: {len(features)}, groups: {len(groups)}")
    print(f"  matrices: {list(matrices.keys())}")
    print(f"  IC: {len(payload['ic_analysis'])} rows, SE: {len(payload['se_analysis'])}, "
          f"CP: {len(payload['cp_analysis'])}, KW: {len(payload['kw_analysis'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", default="dashboard_probe")
    args = parser.parse_args()
    main(args.dashboard)
