"""
Comprehensive analysis of gating_probe_v1 baseline results.

Computes per-family:
  - sign_acc, mean |nd|, AUC(gate_label)
  - pair_flip_rate: fraction of (allowed, blocked) pairs where model correctly flips
  - direction_sensitivity: mean |nd_allowed - nd_blocked| per pair
  - gate_asymmetry: P(model outputs Yes) — 0.5=balanced, 1.0=always Yes
  - wording_robustness: std of sign_acc across wording families
  - causal_readiness_score (CRS): pair_flip_rate × sign_acc × direction_sensitivity_normalized × wording_consistency
  - ambiguity_report: worst-performing templates and concepts

Ranking by CRS determines recommended behaviours for full mechanistic pipeline.

Outputs:
  data/results/gating_probe_v1/
    family_metrics.csv
    pair_analysis.csv
    ambiguity_report.csv
    wording_breakdown.csv
  docs/gating_probe_screen_results.md  (partial — full report in script 83)

Usage:
    python scripts/82_analyze_gating_probe_results.py
"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

IN_CSV   = Path("data/results/gating_probe_v1/baseline_results.csv")
OUT_DIR  = Path("data/results/gating_probe_v1")
DOCS_DIR = Path("docs")


def auc_score(y_true, scores):
    """One-vs-rest AUC; y_true: 0/1, scores: higher = more 'allowed'."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def pair_analysis(df):
    """For each counterfactual pair, check if model correctly flips."""
    pairs = df.groupby("pair_id")
    rows  = []
    for pid, grp in pairs:
        allowed  = grp[grp["pair_role"] == "allowed"]
        blocked  = grp[grp["pair_role"] == "blocked"]
        if allowed.empty or blocked.empty:
            continue
        # Mean nd over wording variants for each role
        nd_all = float(allowed["nd"].mean())
        nd_blk = float(blocked["nd"].mean())

        # Correct flip: nd_allowed > 0 AND nd_blocked < 0 (for majority of wording variants)
        frac_correct_all = float((allowed["nd"] > 0).mean())   # fraction saying Yes for allowed
        frac_correct_blk = float((blocked["nd"] < 0).mean())   # fraction saying No for blocked
        flip_correct     = frac_correct_all > 0.5 and frac_correct_blk > 0.5

        rows.append({
            "pair_id":              pid,
            "family":               grp["family"].iloc[0],
            "concept_key":          grp["concept_key"].iloc[0],
            "nd_allowed_mean":      round(nd_all, 4),
            "nd_blocked_mean":      round(nd_blk, 4),
            "direction_sensitivity":round(abs(nd_all - nd_blk), 4),
            "frac_correct_allowed": round(frac_correct_all, 4),
            "frac_correct_blocked": round(frac_correct_blk, 4),
            "flip_correct":         flip_correct,
            "n_allowed":            len(allowed),
            "n_blocked":            len(blocked),
        })
    return pd.DataFrame(rows)


def family_metrics(df, pair_df):
    rows = []
    for fam, grp in df.groupby("family"):
        n          = len(grp)
        sign_acc   = float(grp["correct"].mean())
        mean_nd    = float(grp["nd"].mean())
        mean_absnd = float(grp["nd"].abs().mean())
        gate_asym  = float((grp["nd"] > 0).mean())   # fraction predicting Yes

        # AUC: gate_label allow=1, block=0; score = nd
        y_true  = (grp["gate_label"] == "allow").astype(int).values
        auc     = auc_score(y_true, grp["nd"].values)

        # Wording robustness
        wf_accs = grp.groupby("wording_family")["correct"].mean()
        wf_std  = float(wf_accs.std()) if len(wf_accs) > 1 else 0.0

        # Pair metrics
        fam_pairs = pair_df[pair_df["family"] == fam]
        pair_flip_rate = float(fam_pairs["flip_correct"].mean()) if len(fam_pairs) else 0.0
        dir_sens_mean  = float(fam_pairs["direction_sensitivity"].mean()) if len(fam_pairs) else 0.0

        # Causal Readiness Score:
        #   CRS = pair_flip_rate × sign_acc × dir_sens_normalized × wording_consistency
        # wording_consistency = 1 - wf_std (penalizes variance across wordings)
        # dir_sens_normalized: rescaled 0→1 within each run (we use tanh scaling)
        dir_norm   = float(np.tanh(dir_sens_mean / 2.0))   # maps [0,∞) → [0,1)
        wrd_consist = max(0.0, 1.0 - wf_std * 2)           # 1=perfectly consistent
        crs = pair_flip_rate * sign_acc * dir_norm * wrd_consist

        # Confound check: how far from ceiling (sign_acc ≤ 0.95 ideal)
        not_ceiling = 1.0 - max(0.0, (sign_acc - 0.95) / 0.05)

        rows.append({
            "family":            fam,
            "family_name":       grp["family_name"].iloc[0],
            "n_prompts":         n,
            "n_pairs":           len(fam_pairs),
            "sign_acc":          round(sign_acc, 4),
            "mean_nd":           round(mean_nd, 4),
            "mean_abs_nd":       round(mean_absnd, 4),
            "gate_asymmetry":    round(gate_asym, 4),
            "auc_gate_label":    round(auc, 4),
            "pair_flip_rate":    round(pair_flip_rate, 4),
            "dir_sensitivity":   round(dir_sens_mean, 4),
            "wording_std":       round(wf_std, 4),
            "causal_readiness":  round(crs, 4),
            "not_ceiling":       round(not_ceiling, 4),
        })
    return pd.DataFrame(rows).sort_values("causal_readiness", ascending=False)


def wording_breakdown(df):
    rows = []
    for (fam, wf), grp in df.groupby(["family", "wording_family"]):
        rows.append({
            "family":         fam,
            "family_name":    grp["family_name"].iloc[0],
            "wording_family": wf,
            "sign_acc":       round(float(grp["correct"].mean()), 4),
            "mean_nd":        round(float(grp["nd"].mean()), 4),
            "n":              len(grp),
        })
    return pd.DataFrame(rows)


def ambiguity_report(df):
    """Worst-performing concept × wording combinations."""
    rows = []
    for (fam, ck, wf), grp in df.groupby(["family","concept_key","wording_family"]):
        rows.append({
            "family":         fam,
            "concept_key":    ck,
            "wording_family": wf,
            "sign_acc":       round(float(grp["correct"].mean()), 4),
            "mean_nd":        round(float(grp["nd"].mean()), 4),
            "n":              len(grp),
        })
    return pd.DataFrame(rows).sort_values("sign_acc")


def write_report(fam_df, pair_df, wf_df, amb_df, overall_acc):
    lines = [
        "# Gating Probe Screen — Baseline Results",
        "",
        "## Overview",
        "",
        f"- **Total prompts**: {fam_df['n_prompts'].sum()}",
        f"- **Families**: {len(fam_df)}",
        f"- **Overall sign accuracy**: {overall_acc:.4f}",
        "",
        "## Family Ranking (by Causal Readiness Score)",
        "",
        "CRS = pair_flip_rate × sign_acc × direction_sensitivity_normalized × wording_consistency",
        "",
        "| Rank | Family | Name | sign_acc | AUC | flip_rate | dir_sens | wording_std | CRS |",
        "|------|--------|------|----------|-----|-----------|----------|-------------|-----|",
    ]
    for rank, (_, r) in enumerate(fam_df.iterrows(), 1):
        lines.append(
            f"| {rank} | {r['family']} | {r['family_name']} | {r['sign_acc']:.3f} | "
            f"{r['auc_gate_label']:.3f} | {r['pair_flip_rate']:.3f} | "
            f"{r['dir_sensitivity']:.3f} | {r['wording_std']:.3f} | **{r['causal_readiness']:.4f}** |"
        )

    lines += [
        "",
        "## Per-Family Detail",
        "",
        "| Family | n | sign_acc | gate_asymmetry | AUC | pair_flip | dir_sens | CRS |",
        "|--------|---|----------|----------------|-----|-----------|----------|-----|",
    ]
    for _, r in fam_df.iterrows():
        lines.append(
            f"| {r['family']}: {r['family_name']} | {r['n_prompts']} | "
            f"{r['sign_acc']:.3f} | {r['gate_asymmetry']:.3f} | "
            f"{r['auc_gate_label']:.3f} | {r['pair_flip_rate']:.3f} | "
            f"{r['dir_sensitivity']:.3f} | {r['causal_readiness']:.4f} |"
        )

    lines += [
        "",
        "## Wording Robustness",
        "",
        "| Family | Wording | sign_acc | mean_nd | n |",
        "|--------|---------|----------|---------|---|",
    ]
    for _, r in wf_df.sort_values(["family","sign_acc"]).iterrows():
        lines.append(
            f"| {r['family']} | {r['wording_family']} | {r['sign_acc']:.3f} | {r['mean_nd']:.3f} | {r['n']} |"
        )

    lines += [
        "",
        "## Worst-Performing Templates (sign_acc < 0.60)",
        "",
        "| Family | Concept | Wording | sign_acc | mean_nd | n |",
        "|--------|---------|---------|----------|---------|---|",
    ]
    worst = amb_df[amb_df["sign_acc"] < 0.60].head(20)
    for _, r in worst.iterrows():
        lines.append(
            f"| {r['family']} | {r['concept_key']} | {r['wording_family']} | "
            f"{r['sign_acc']:.3f} | {r['mean_nd']:.3f} | {r['n']} |"
        )

    lines += [
        "",
        "## Counterfactual Pair Analysis",
        "",
        "| Pair ID | Family | Concept | nd_allowed | nd_blocked | dir_sens | flip_correct |",
        "|---------|--------|---------|-----------|-----------|----------|--------------|",
    ]
    for _, r in pair_df.sort_values(["family","direction_sensitivity"], ascending=[True,False]).iterrows():
        flip_str = "✓" if r["flip_correct"] else "✗"
        lines.append(
            f"| {r['pair_id']} | {r['family']} | {r['concept_key']} | "
            f"{r['nd_allowed_mean']:+.3f} | {r['nd_blocked_mean']:+.3f} | "
            f"{r['direction_sensitivity']:.3f} | {flip_str} |"
        )

    lines += [
        "",
        "## Gate Asymmetry Analysis",
        "",
        "gate_asymmetry = P(model outputs Yes). 0.5=balanced, >0.5=Yes-biased, <0.5=No-biased.",
        "",
        "| Family | gate_asymmetry | Interpretation |",
        "|--------|----------------|----------------|",
    ]
    for _, r in fam_df.iterrows():
        g = r["gate_asymmetry"]
        interp = "balanced" if 0.40<=g<=0.60 else ("Yes-biased (allow default)" if g>0.60 else "No-biased (block default)")
        lines.append(f"| {r['family']}: {r['family_name']} | {g:.3f} | {interp} |")

    # Top recommendations
    top2 = fam_df.head(2)
    lines += [
        "",
        "## Recommendations for Full Mechanistic Pipeline",
        "",
    ]
    for rank, (_, r) in enumerate(top2.iterrows(), 1):
        lines.append(f"### Rank {rank}: Family {r['family']} — {r['family_name']}")
        lines.append("")
        lines.append(f"- sign_acc = {r['sign_acc']:.3f}")
        lines.append(f"- AUC(gate_label) = {r['auc_gate_label']:.3f}")
        lines.append(f"- pair_flip_rate = {r['pair_flip_rate']:.3f}")
        lines.append(f"- direction_sensitivity = {r['dir_sensitivity']:.3f}")
        lines.append(f"- wording_std = {r['wording_std']:.3f}")
        lines.append(f"- **Causal Readiness Score = {r['causal_readiness']:.4f}**")
        lines.append("")

    lines += [
        "*Data: data/results/gating_probe_v1/*",
        "*Scripts: 80 (generate), 81 (baseline), 82 (analysis), 83 (representation probe)*",
    ]

    path = DOCS_DIR / "gating_probe_screen_results.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_CSV.exists():
        print(f"Baseline CSV not found: {IN_CSV}")
        print("Run script 81 first.")
        return

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} rows from {IN_CSV}")

    pair_df  = pair_analysis(df)
    fam_df   = family_metrics(df, pair_df)
    wf_df    = wording_breakdown(df)
    amb_df   = ambiguity_report(df)

    pair_df.to_csv(OUT_DIR / "pair_analysis.csv",      index=False)
    fam_df.to_csv( OUT_DIR / "family_metrics.csv",     index=False)
    wf_df.to_csv(  OUT_DIR / "wording_breakdown.csv",  index=False)
    amb_df.to_csv( OUT_DIR / "ambiguity_report.csv",   index=False)

    overall_acc = float(df["correct"].mean())

    print(f"\n=== FAMILY METRICS (ranked by CRS) ===")
    print(fam_df[["family","family_name","sign_acc","auc_gate_label",
                   "pair_flip_rate","dir_sensitivity","wording_std",
                   "causal_readiness"]].to_string(index=False))

    print(f"\nOverall sign accuracy: {overall_acc:.4f}")
    print(f"\nTop recommendation: Family {fam_df.iloc[0]['family']} — {fam_df.iloc[0]['family_name']}")
    print(f"  CRS = {fam_df.iloc[0]['causal_readiness']:.4f}")

    write_report(fam_df, pair_df, wf_df, amb_df, overall_acc)


if __name__ == "__main__":
    main()
