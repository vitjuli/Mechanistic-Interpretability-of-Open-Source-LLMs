#!/usr/bin/env python3
"""
runB_three_mode_analysis.py

Analyses the three-mode (zero / mean / resample) ablation control for
the Run B L18/L24 gating clusters.

Run locally AFTER syncing:
  rsync -avz iv294@login.hpc.cam.ac.uk:.../data/analysis/runB_validation/three_mode_ablation_control/ \
      data/analysis/runB_validation/three_mode_ablation_control/

Outputs to data/analysis/runB_validation/three_mode_ablation_control/:
  three_mode_ablation_control.csv
  three_mode_ablation_control_summary.md
  L18_L24_three_mode_gating_validation.md
"""

import ast, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
THREE_MODE_DIR = ROOT / "data/analysis/runB_validation/three_mode_ablation_control"
OUT = THREE_MODE_DIR

MODES = ["zero", "mean", "resample"]
BEHAVIOUR = "physics_decay_type_probe"

# ── Load all three mode CSVs ───────────────────────────────────────────────────
dfs = {}
for mode in MODES:
    p = THREE_MODE_DIR / f"intervention_ablation_{BEHAVIOUR}_{mode}.csv"
    if not p.exists():
        print(f"WARNING: {p} not found — skipping mode '{mode}'")
        continue
    df = pd.read_csv(p)
    df["correct_token"] = df["metadata"].apply(lambda x: ast.literal_eval(x)["correct_token"])
    df["is_alpha_prompt"] = df["correct_token"] == " alpha"
    df["delta_alpha_baseline"]   = np.where(df.is_alpha_prompt,  df.baseline_logit_diff,  -df.baseline_logit_diff)
    df["delta_alpha_intervened"] = np.where(df.is_alpha_prompt,  df.intervened_logit_diff,-df.intervened_logit_diff)
    df["effect_alpha_axis"]      = df.delta_alpha_intervened - df.delta_alpha_baseline
    df["ablation_mode"] = mode
    dfs[mode] = df
    print(f"  Loaded {mode}: {len(df)} rows, {df.feature_id.nunique()} features")

if not dfs:
    print("ERROR: No mode CSVs found. Sync from CSD3 first:")
    print("  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/"
          "data/analysis/runB_validation/three_mode_ablation_control/ "
          "data/analysis/runB_validation/three_mode_ablation_control/")
    exit(1)

combined = pd.concat(dfs.values(), ignore_index=True)

# ── Bootstrap CI helper ────────────────────────────────────────────────────────
def bci(x, n=2000, alpha=0.05):
    if len(x) == 0:
        return (np.nan, np.nan)
    m = [np.mean(np.random.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(m, [100*alpha/2, 100*(1-alpha/2)]))

# ── Per-feature, per-mode fixed-axis stats ────────────────────────────────────
rows = []
for mode, df in dfs.items():
    for fid, grp in df.groupby("feature_id"):
        lyr = int(grp.layer.iloc[0])
        eff = grp.effect_alpha_axis.values
        ci  = bci(eff)
        a_g = grp[grp.is_alpha_prompt]
        b_g = grp[~grp.is_alpha_prompt]
        rows.append({
            "ablation_mode":        mode,
            "feature_id":           fid,
            "layer":                lyr,
            "mean_effect_alpha_ax": eff.mean(),
            "ci_lo":                ci[0],
            "ci_hi":                ci[1],
            "ci_excl_zero":         bool(ci[0] > 0 or ci[1] < 0),
            "causal_dir":           "anti_alpha" if eff.mean() > 0 else "pro_alpha",
            "sfr_alpha_prompts":    a_g.sign_flipped.mean(),
            "sfr_beta_prompts":     b_g.sign_flipped.mean(),
        })

feat_mode_df = pd.DataFrame(rows)
feat_mode_df.to_csv(OUT / "three_mode_ablation_control.csv", index=False)
print(f"\nSaved: three_mode_ablation_control.csv ({len(feat_mode_df)} rows)")

# ── Stability check: does direction agree across modes? ───────────────────────
pivot = feat_mode_df.pivot_table(
    index="feature_id", columns="ablation_mode", values="causal_dir", aggfunc="first"
)
pivot["all_agree"] = pivot.apply(
    lambda r: len(set(r.dropna())) == 1, axis=1
)
print("\nDirection agreement across modes:")
print(pivot.to_string())

# ── L18 and L24 group-level summaries ─────────────────────────────────────────
reports = {}
for lyr, claimed_dir in [(18, "pro_alpha"), (24, "anti_alpha")]:
    sub = feat_mode_df[feat_mode_df.layer == lyr]
    r = {}
    for mode in MODES:
        m_sub = sub[sub.ablation_mode == mode]
        if m_sub.empty:
            r[mode] = {"n": 0, "pct_correct_dir": np.nan, "all_ci_excl": False}
            continue
        pct = (m_sub.causal_dir == claimed_dir).mean()
        all_ci = m_sub.ci_excl_zero.all()
        r[mode] = {"n": len(m_sub), "pct_correct_dir": pct, "all_ci_excl": all_ci,
                   "mean_eff": m_sub.mean_effect_alpha_ax.mean()}
    reports[lyr] = r

# ── Write L18/L24 three-mode gating validation ────────────────────────────────
with open(OUT / "L18_L24_three_mode_gating_validation.md", "w") as f:
    f.write("# L18/L24 Three-Mode Gating Validation\n\n")
    f.write("Fixed-axis effect = logit(α) − logit(β) after ablation − baseline.\n")
    f.write("pro_alpha: ablating LOWERS logit(α) → feature supports α.\n")
    f.write("anti_alpha: ablating RAISES logit(α) → feature suppresses α.\n\n")

    for lyr, claimed in [(18, "pro_alpha"), (24, "anti_alpha")]:
        f.write(f"## L{lyr} — claimed direction under zero-ablation: {claimed}\n\n")
        r = reports[lyr]
        rows_md = []
        for mode in MODES:
            if mode not in r:
                rows_md.append(f"| {mode} | — | — | — | — |")
                continue
            rd = r[mode]
            status = "✓ confirmed" if rd["pct_correct_dir"] == 1.0 else \
                     "⚠ partial" if rd["pct_correct_dir"] >= 0.5 else "✗ reversed"
            f.write(f"- **{mode}**: {rd['pct_correct_dir']:.0%} of features confirm {claimed}, "
                    f"CIs all excl zero: {rd['all_ci_excl']}, "
                    f"mean fixed-axis effect: {rd.get('mean_eff', float('nan')):.3f} → {status}\n")
        f.write("\n")

        # Per-feature table for this layer
        sub = feat_mode_df[feat_mode_df.layer == lyr]
        if not sub.empty:
            f.write("### Per-feature breakdown\n\n")
            f.write(sub[["ablation_mode","feature_id","mean_effect_alpha_ax","ci_lo","ci_hi",
                          "ci_excl_zero","causal_dir","sfr_alpha_prompts","sfr_beta_prompts"]
                        ].to_markdown(index=False))
            f.write("\n\n")

    f.write("## Answers to validation questions\n\n")
    modes_present = list(dfs.keys())

    def direction_stable(lyr, claimed):
        r = reports.get(lyr, {})
        return all(r.get(m, {}).get("pct_correct_dir", 0) == 1.0 for m in modes_present)

    l18_stable = direction_stable(18, "pro_alpha")
    l24_stable = direction_stable(24, "anti_alpha")

    f.write(f"**1. Does L18 remain pro-α / anti-β under all three modes?** "
            f"{'YES — confirmed across all modes run.' if l18_stable else 'PARTIALLY — see table above.'}\n\n")
    f.write(f"**2. Does L24 remain anti-α / pro-β under all three modes?** "
            f"{'YES — confirmed across all modes run.' if l24_stable else 'PARTIALLY — see table above.'}\n\n")

    # Check C6 / L16 stability
    l16_sub = feat_mode_df[feat_mode_df.layer == 16]
    if not l16_sub.empty:
        l16_dirs = l16_sub.groupby("ablation_mode").causal_dir.apply(list)
        l16_stable = all(len(set(v)) == 1 for v in l16_dirs.values)
        f.write(f"**3. Does L16 (C6) remain directionally stable?** "
                f"{'YES.' if l16_stable else 'Mixed — see per-mode table.'}\n\n")

    both_stable = l18_stable and l24_stable
    f.write(f"**4. Is the L18/L24 finding a zero-ablation artifact?** "
            f"{'NO — direction is preserved across all ablation modes tested.' if both_stable else 'UNCERTAIN — one or more modes show divergence; see details above.'}\n\n")

    f.write("**5. Which ablation mode should be used for the thesis main text?** ")
    if both_stable:
        f.write("Zero-ablation is appropriate as the primary mode. "
                "Mean and resample ablation serve as robustness checks confirming the direction. "
                "Report all three modes in supplementary material.\n\n")
    else:
        f.write("Direction is not fully stable across modes. "
                "Use zero-ablation results with explicit caveat; report discrepancies in full.\n\n")
    f.write("*Generated by `scripts/runB_three_mode_analysis.py`*\n")

# ── Summary report ─────────────────────────────────────────────────────────────
with open(OUT / "three_mode_ablation_control_summary.md", "w") as f:
    f.write("# Three-Mode Ablation Control: Summary\n\n")
    f.write("Modes compared: " + ", ".join(modes_present) + "\n\n")
    f.write("## Direction agreement per feature\n\n")
    f.write(pivot.reset_index().to_markdown(index=False))
    f.write("\n\n## Layer-level summary\n\n")
    by_layer = feat_mode_df.groupby(["layer","ablation_mode"]).agg(
        n=("feature_id","nunique"),
        pct_anti_alpha=("causal_dir", lambda x: (x=="anti_alpha").mean()),
        pct_ci_excl=("ci_excl_zero","mean"),
    ).reset_index()
    f.write(by_layer.to_markdown(index=False))
    f.write("\n\n## Key finding\n\n")
    both = l18_stable and l24_stable
    f.write(
        "**L18/L24 gating interpretation is ROBUST across ablation modes.**\n"
        if both else
        "**L18/L24 gating interpretation is PARTIALLY supported — check details.**\n"
    )
    f.write("\n*Generated by `scripts/runB_three_mode_analysis.py`*\n")

print("\nOutputs:")
for p in sorted(OUT.iterdir()):
    if p.suffix in (".csv", ".md"):
        print(f"  {p.name}")
