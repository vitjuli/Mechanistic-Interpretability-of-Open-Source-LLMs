"""
31_analyze_physics_e1_selection.py

First-pass analysis for physics_e1_selection behaviour.

Produces:
  data/results/physics_e1_selection/baseline_breakdown.csv
  data/results/physics_e1_selection/contrastive_pair_analysis.csv
  data/results/physics_e1_selection/delta_l_profile.csv
  data/results/physics_e1_selection/analysis_report.md

Usage:
  python scripts/31_analyze_physics_e1_selection.py [--split train|test] [--run_dir PATH]

Requirements:
  - Baseline CSV must exist: data/results/baseline_physics_e1_selection_{split}.csv
  - Ablation CSV is optional: data/results/interventions/physics_e1_selection/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
OUTDIR   = ROOT / "data/results/physics_e1_selection"
OUTDIR.mkdir(parents=True, exist_ok=True)

BEHAVIOUR = "physics_e1_selection"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--split",   default="train", choices=["train", "test"])
parser.add_argument("--run_dir", default=None, help="Path to UI offline run directory (optional)")
args = parser.parse_args()
SPLIT = args.split

# ── Load prompts ──────────────────────────────────────────────────────────────
PROMPT_FILE = ROOT / f"data/prompts/{BEHAVIOUR}_{SPLIT}.jsonl"
if not PROMPT_FILE.exists():
    sys.exit(f"[FATAL] Prompt file not found: {PROMPT_FILE}\n"
             f"  Run: python scripts/30_generate_physics_e1_selection_prompts.py")

prompts = [json.loads(l) for l in open(PROMPT_FILE, encoding="utf-8")]
pm = pd.DataFrame(prompts)
print(f"Loaded {len(pm)} prompts from {PROMPT_FILE.name}")

# ── Load baseline results ─────────────────────────────────────────────────────
BASELINE_FILE = ROOT / f"data/results/baseline_{BEHAVIOUR}_{SPLIT}.csv"
has_baseline  = BASELINE_FILE.exists()

if has_baseline:
    bl = pd.read_csv(BASELINE_FILE)
    print(f"Loaded {len(bl)} baseline rows from {BASELINE_FILE.name}")
    # Merge with prompt metadata
    if "prompt_idx" in bl.columns:
        df = bl.merge(pm, on="prompt_idx", how="left", suffixes=("", "_meta"))
    else:
        df = bl.merge(pm, left_index=True, right_on="prompt_idx", how="left")
    # Ensure sign_correct column
    if "sign_correct" not in df.columns and "baseline_logit_diff" in df.columns:
        df["sign_correct"] = df["baseline_logit_diff"] > 0
else:
    print("[WARN] Baseline CSV not found — producing metadata-only analysis.")
    df = pm.copy()
    df["sign_correct"]       = None
    df["baseline_logit_diff"] = None

# ── Try to load ablation CSV ───────────────────────────────────────────────────
ABLATION_FILE = ROOT / f"data/results/interventions/{BEHAVIOUR}/intervention_ablation_{BEHAVIOUR}_{SPLIT}.csv"
if not ABLATION_FILE.exists():
    ABLATION_FILE = ROOT / f"data/results/interventions/{BEHAVIOUR}/intervention_ablation_{BEHAVIOUR}.csv"
has_ablation = ABLATION_FILE.exists()

if has_ablation:
    abl = pd.read_csv(ABLATION_FILE)
    print(f"Loaded {len(abl)} ablation rows from {ABLATION_FILE.name}")
else:
    print("[INFO] Ablation CSV not found — skipping feature-level analysis.")
    abl = None

# ── 1. Baseline breakdown ─────────────────────────────────────────────────────
print("\n=== BASELINE BREAKDOWN ===")

report_lines = [
    f"# physics_e1_selection — Analysis Report",
    f"Split: {SPLIT} | Prompts: {len(pm)}",
    f"Baseline available: {has_baseline}",
    f"Ablation available: {has_ablation}",
    "",
]

def acc(sub, col="sign_correct"):
    vals = sub[col].dropna()
    return vals.mean() if len(vals) > 0 else float("nan")

breakdown_rows = []

if has_baseline:
    overall_acc = acc(df)
    print(f"Overall sign accuracy: {overall_acc:.3f}  ({df['sign_correct'].sum()}/{len(df)})")
    report_lines += [
        "## Baseline accuracy",
        f"Overall: **{overall_acc:.1%}**  ({int(df['sign_correct'].sum())}/{len(df)})",
        "",
    ]

    # By wording family
    report_lines.append("### By wording family")
    for fam in sorted(df["wording_family"].unique()):
        sub = df[df["wording_family"] == fam]
        a = acc(sub)
        print(f"  {fam}: {a:.3f}  (n={len(sub)})")
        report_lines.append(f"- {fam}: {a:.1%}  (n={len(sub)})")
        breakdown_rows.append({"breakdown": "wording_family", "value": fam, "n": len(sub),
                                "accuracy": a, "allowed_n": (sub.selection_result=="allowed").sum(),
                                "forbidden_n": (sub.selection_result=="forbidden").sum()})
    report_lines.append("")

    # By transition group
    report_lines.append("### By transition group")
    for gid in ["sp", "ps", "pd", "dp", "ss", "pp", "dd", "sd", "dl0", "dlp1", "dlm1", "dlp2"]:
        sub = df[df["group_id"] == gid]
        if len(sub) == 0:
            continue
        a = acc(sub)
        result = sub.iloc[0]["selection_result"] if "selection_result" in sub.columns else "?"
        print(f"  {gid:8s} ({result:9s}): {a:.3f}  (n={len(sub)})")
        report_lines.append(f"- {gid} ({result}): {a:.1%}  (n={len(sub)})")
        breakdown_rows.append({"breakdown": "group_id", "value": gid, "n": len(sub),
                                "accuracy": a, "result": result})
    report_lines.append("")

    # By selection result
    report_lines.append("### By selection result (allowed vs forbidden)")
    for res in ["allowed", "forbidden"]:
        sub = df[df["selection_result"] == res]
        a = acc(sub)
        print(f"  {res:10s}: {a:.3f}  (n={len(sub)})")
        report_lines.append(f"- {res}: {a:.1%}  (n={len(sub)})")
        breakdown_rows.append({"breakdown": "selection_result", "value": res, "n": len(sub),
                                "accuracy": a})
    report_lines.append("")

    # By Δl value
    report_lines.append("### By |Δl| value")
    for dl in sorted(df["abs_delta_l"].dropna().unique()):
        sub = df[df["abs_delta_l"] == dl]
        a = acc(sub)
        res = "allowed" if dl == 1 else "forbidden"
        print(f"  |Δl|={int(dl)}  ({res:9s}): {a:.3f}  (n={len(sub)})")
        report_lines.append(f"- |Δl|={int(dl)} ({res}): {a:.1%}  (n={len(sub)})")
        breakdown_rows.append({"breakdown": "abs_delta_l", "value": int(dl), "n": len(sub),
                                "accuracy": a, "expected_result": res})
    report_lines.append("")

    # By keyword_free
    if "keyword_free" in df.columns:
        report_lines.append("### By keyword-free status (E1/dipole keyword)")
        for kf in [False, True]:
            sub = df[df["keyword_free"] == kf]
            if len(sub) == 0:
                continue
            a = acc(sub)
            print(f"  keyword_free={kf}: {a:.3f}  (n={len(sub)})")
            report_lines.append(f"- keyword_free={kf}: {a:.1%}  (n={len(sub)})")
        report_lines.append("")

bl_df = pd.DataFrame(breakdown_rows)
bl_df.to_csv(OUTDIR / "baseline_breakdown.csv", index=False)
print(f"Saved baseline_breakdown.csv ({len(bl_df)} rows)")

# ── 2. Contrastive pair analysis ──────────────────────────────────────────────
print("\n=== CONTRASTIVE PAIR ANALYSIS ===")

cp_rows = []
PAIRS = {
    "sp_vs_ss": ("allowed", "forbidden"),
    "pd_vs_pp": ("allowed", "forbidden"),
    "dp_vs_dd": ("allowed", "forbidden"),
    "sp_vs_sd": ("allowed", "forbidden"),
}

report_lines.append("## Contrastive pair analysis")
report_lines.append("For each pair: accuracy_allowed, accuracy_forbidden, joint accuracy")
report_lines.append("")

for pair_id, (roleA, roleB) in PAIRS.items():
    subA = df[df["contrastive_pair_id"] == pair_id]
    if len(subA) == 0:
        continue
    allowed_sub   = subA[subA["contrastive_role"] == "allowed"]
    forbidden_sub = subA[subA["contrastive_role"] == "forbidden"]
    a_acc = acc(allowed_sub)  if has_baseline else float("nan")
    f_acc = acc(forbidden_sub) if has_baseline else float("nan")

    # Joint consistency: for each matching wording variant, are both correct?
    joint_consec = float("nan")
    if has_baseline and "wording_variant" in df.columns:
        n_joint = 0
        n_both_correct = 0
        for fam in subA["wording_family"].unique():
            for v in subA["wording_variant"].unique():
                a_row = subA[(subA["contrastive_role"] == "allowed") &
                             (subA["wording_family"] == fam) &
                             (subA["wording_variant"] == v)]
                f_row = subA[(subA["contrastive_role"] == "forbidden") &
                             (subA["wording_family"] == fam) &
                             (subA["wording_variant"] == v)]
                if len(a_row) == 1 and len(f_row) == 1:
                    n_joint += 1
                    if a_row.iloc[0]["sign_correct"] and f_row.iloc[0]["sign_correct"]:
                        n_both_correct += 1
        joint_consec = n_both_correct / n_joint if n_joint > 0 else float("nan")

    print(f"  {pair_id:12s}: allowed={a_acc:.3f} forbidden={f_acc:.3f} joint={joint_consec:.3f}")
    report_lines.append(f"- **{pair_id}**: allowed={a_acc:.1%}, forbidden={f_acc:.1%}, "
                        f"joint={joint_consec:.1%}  (n_allowed={len(allowed_sub)}, "
                        f"n_forbidden={len(forbidden_sub)})")

    cp_rows.append({
        "pair_id": pair_id,
        "n_allowed": len(allowed_sub),
        "n_forbidden": len(forbidden_sub),
        "accuracy_allowed": a_acc,
        "accuracy_forbidden": f_acc,
        "joint_consistency": joint_consec,
    })

cp_df = pd.DataFrame(cp_rows)
cp_df.to_csv(OUTDIR / "contrastive_pair_analysis.csv", index=False)
print(f"Saved contrastive_pair_analysis.csv ({len(cp_df)} rows)")
report_lines.append("")

# ── 3. Delta-l profile ────────────────────────────────────────────────────────
print("\n=== DELTA-L PROFILE ===")

dl_rows = []
report_lines.append("## Delta-l profile")
report_lines.append("Accuracy per signed Δl value (tests whether model tracks sign vs magnitude)")
report_lines.append("")

if has_baseline:
    for dl in sorted(df["delta_l"].dropna().unique(), key=lambda x: int(x)):
        sub = df[df["delta_l"] == dl]
        a   = acc(sub)
        res = sub.iloc[0]["selection_result"] if len(sub) > 0 and "selection_result" in sub.columns else "?"
        print(f"  Δl={int(dl):+3d} ({res:9s}): {a:.3f}  (n={len(sub)})")
        report_lines.append(f"- Δl={int(dl):+d} ({res}): {a:.1%}  (n={len(sub)})")
        dl_rows.append({"delta_l": int(dl), "abs_delta_l": abs(int(dl)),
                        "result": res, "n": len(sub), "accuracy": a})

dl_df = pd.DataFrame(dl_rows)
dl_df.to_csv(OUTDIR / "delta_l_profile.csv", index=False)
print(f"Saved delta_l_profile.csv ({len(dl_df)} rows)")
report_lines.append("")

# ── 4. Ablation feature analysis (if available) ───────────────────────────────
if has_ablation:
    print("\n=== ABLATION FEATURE ANALYSIS ===")
    report_lines.append("## Ablation feature analysis")

    abl2 = abl.merge(pm[["prompt_idx","group_id","selection_result","wording_family",
                          "delta_l","abs_delta_l"]].drop_duplicates("prompt_idx"),
                     on="prompt_idx", how="left")

    # Mean effect per feature, by selection result
    allowed_eff  = abl2[abl2["selection_result"]=="allowed"].groupby("feature_id")["effect_size"].mean()
    forbidden_eff = abl2[abl2["selection_result"]=="forbidden"].groupby("feature_id")["effect_size"].mean()
    feat_contrast = (allowed_eff - forbidden_eff).dropna().sort_values(key=abs, ascending=False)

    print(f"Top features by |allowed_mean_eff - forbidden_mean_eff|:")
    top10 = feat_contrast.head(10)
    for fid, v in top10.items():
        print(f"  {fid:20s}: contrast={v:+.4f}")

    feat_df = pd.DataFrame({
        "feature_id":          feat_contrast.index,
        "allowed_minus_forbidden_effect": feat_contrast.values,
        "mean_effect_allowed":  [allowed_eff.get(f, float("nan")) for f in feat_contrast.index],
        "mean_effect_forbidden": [forbidden_eff.get(f, float("nan")) for f in feat_contrast.index],
    })
    feat_df.to_csv(OUTDIR / "feature_contrast_allowed_forbidden.csv", index=False)
    print(f"Saved feature_contrast_allowed_forbidden.csv ({len(feat_df)} rows)")

    # Cross-family consistency: same feature, same transition type, different wording
    report_lines.append("Top 10 features by |allowed_eff − forbidden_eff|:")
    for fid, v in top10.items():
        report_lines.append(f"- {fid}: {v:+.4f}")
    report_lines.append("")

# ── 5. Corpus statistics ──────────────────────────────────────────────────────
report_lines += [
    "## Corpus statistics",
    f"- Total prompts: {len(pm)}",
    f"- Allowed: {(pm['selection_result']=='allowed').sum()}  "
    f"Forbidden: {(pm['selection_result']=='forbidden').sum()}",
    f"- Wording families: {sorted(pm['wording_family'].unique())}",
    f"- Transition groups: {sorted(pm['group_id'].unique())}",
    f"- keyword_free prompts: {pm.get('keyword_free', pd.Series()).sum() if 'keyword_free' in pm.columns else 'N/A'}",
    "",
]

# ── 6. Decision thresholds ────────────────────────────────────────────────────
if has_baseline:
    overall_acc = acc(df)
    f4_acc = acc(df[df["wording_family"] == "F4"]) if "F4" in df["wording_family"].values else float("nan")
    report_lines += [
        "## Gate check",
        f"- Overall sign accuracy: {overall_acc:.1%}  (gate: ≥ 85%)",
        f"  {'✓ PASS' if overall_acc >= 0.85 else '✗ FAIL — consider fallback to F4-only analysis'}",
        f"- F4 (explicit Δl) accuracy: {f4_acc:.1%}  (gate: ≥ 95%)",
        f"  {'✓ PASS' if f4_acc >= 0.95 else '✗ FAIL — model may not know E1 rule'}",
        "",
        "## Recommended next steps" if overall_acc < 0.85 else "## Baseline PASSED",
    ]
    if overall_acc < 0.85:
        report_lines += [
            "- Accuracy below threshold. Check breakdown by family.",
            "- If F4 acc < 95%: model does not reliably know E1 rule — do NOT proceed with mechanistic analysis.",
            "- If F4 acc ≥ 95% but F1/F2 acc < 85%: orbital name extraction is the bottleneck.",
            "  → Add more F2 (quantum number) prompts or simplify to l-value-only analysis.",
        ]
    else:
        report_lines += [
            "- Proceed with feature extraction (scripts/04) and intervention analysis (scripts/07).",
            "- Key analyses: cross-family convergence (F1 vs F2 vs F3 for same transition group),",
            "  contrastive pair feature analysis (sp vs ss, pd vs pp).",
        ]

# ── Write report ───────────────────────────────────────────────────────────────
report_text = "\n".join(report_lines)
(OUTDIR / "analysis_report.md").write_text(report_text, encoding="utf-8")
print(f"\nSaved analysis_report.md")
print("\n=== DONE ===")
print(f"Outputs in: {OUTDIR}")
