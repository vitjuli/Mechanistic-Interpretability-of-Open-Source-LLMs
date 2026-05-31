"""
diag_concept_specific_ablation.py

Causal test for cluster's "concept detector" hypothesis using EXISTING ablation data.

For each cluster G:
  1. From cluster_group_scores.csv, get SFR per cue_group when G is ablated.
  2. Identify G's "concept" groups: TOP-K cue_groups by |mean_effect|.
  3. Compute:
       SFR_TOP  = mean SFR over G's concept groups (TOP-K)
       SFR_BOT  = mean SFR over non-concept groups (BOTTOM-K by |mean_effect|)
       SFR_other_Vh = mean SFR over same-V_h groups NOT in TOP-K
       SFR_random = mean SFR over K random groups (50 replicates)
  4. If SFR_TOP >> SFR_BOT and SFR_TOP >> SFR_random
     → cluster causally specific to its concept groups (not just V_h)
  5. If SFR_TOP ≈ SFR_other_Vh
     → cluster is just a V_h-class ablator (whole-class effect, not concept-specific)

This uses pure observational ablation data — no new GPU runs.

Reads:
  data/analysis/runD_v2/cluster_semantics/cluster_group_scores.csv

Outputs:
  data/analysis/iia_failure_diagnosis/causal_concept_specificity.csv
  data/analysis/iia_failure_diagnosis/causal_concept_specificity.md
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CG_PATH = ROOT / "data/analysis/runD_v2/cluster_semantics/cluster_group_scores.csv"
OUT_DIR = ROOT / "data/analysis/iia_failure_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10
N_RANDOM = 50
RNG = np.random.default_rng(42)

# Load
df = pd.read_csv(CG_PATH)
print(f"Loaded: {len(df)} (cluster, group) rows for {df.cluster_id.nunique()} clusters, "
      f"{df.group_id.nunique()} groups")

# Filter to groups with reasonable sample size (avoid n=1 singletons)
df = df[df.n_prompts >= 3].copy()
print(f"After n_prompts >= 3 filter: {len(df)} rows")

# Per cluster
results = []
for cid in sorted(df.cluster_id.unique()):
    sub = df[df.cluster_id == cid].copy()
    if len(sub) < TOP_K * 2:
        continue

    # Sort groups by |mean_effect| to identify concept-aligned groups
    sub = sub.sort_values("mean_abs_effect", ascending=False)
    top_groups = set(sub.head(TOP_K).group_id.tolist())
    bot_groups = set(sub.tail(TOP_K).group_id.tolist())

    # SFR per partition (weighted by n_prompts)
    def weighted_sfr(s):
        if len(s) == 0:
            return float("nan")
        return float((s["sfr"] * s["n_prompts"]).sum() / s["n_prompts"].sum())

    sfr_top = weighted_sfr(sub[sub.group_id.isin(top_groups)])
    sfr_bot = weighted_sfr(sub[sub.group_id.isin(bot_groups)])

    # V_h composition of TOP
    top_correct = sub[sub.group_id.isin(top_groups)].correct_answer.values
    top_alpha_frac = float((top_correct == "alpha").mean())

    # Same-V_h control: SFR on groups of dominant V_h class NOT in TOP
    dominant_vh = "alpha" if top_alpha_frac >= 0.5 else "beta"
    same_vh_not_top = sub[(sub.correct_answer == dominant_vh)
                          & (~sub.group_id.isin(top_groups))]
    sfr_same_vh_other = weighted_sfr(same_vh_not_top) if len(same_vh_not_top) > 0 else float("nan")

    # Random baseline: pick K random groups, compute SFR
    all_groups = sub.group_id.tolist()
    rand_sfrs = []
    for _ in range(N_RANDOM):
        rand_pick = set(RNG.choice(all_groups, size=TOP_K, replace=False))
        rand_sfrs.append(weighted_sfr(sub[sub.group_id.isin(rand_pick)]))
    sfr_random_mean = float(np.mean(rand_sfrs))
    sfr_random_std  = float(np.std(rand_sfrs))

    # Per-cluster overall metrics
    overall_sfr = weighted_sfr(sub)
    mean_top_abs_eff = float(sub.head(TOP_K).mean_abs_effect.mean())

    # CONCEPT-SPECIFICITY ratio
    # Concept-specific if SFR_TOP >> SFR_BOT AND SFR_TOP >> SFR_other_VH (same class)
    sfr_top_minus_bot = sfr_top - sfr_bot
    sfr_top_minus_same_vh = sfr_top - sfr_same_vh_other if not np.isnan(sfr_same_vh_other) else float("nan")
    sfr_top_minus_random = sfr_top - sfr_random_mean
    z_random = (sfr_top - sfr_random_mean) / sfr_random_std if sfr_random_std > 1e-6 else float("inf")

    # Classification
    if sfr_top - sfr_random_mean < 0.05:
        kind = "no concept signal"
    elif not np.isnan(sfr_top_minus_same_vh) and sfr_top_minus_same_vh < 0.05:
        kind = "V_h-class effect"
    elif sfr_top_minus_bot > 0.15 and z_random > 2:
        kind = "CONCEPT-SPECIFIC"
    else:
        kind = "weak / mixed"

    results.append(dict(
        cluster_id=cid,
        sfr_top=round(sfr_top, 3),
        sfr_bot=round(sfr_bot, 3),
        sfr_same_vh_other=round(sfr_same_vh_other, 3) if not np.isnan(sfr_same_vh_other) else None,
        sfr_random_mean=round(sfr_random_mean, 3),
        sfr_random_std=round(sfr_random_std, 3),
        sfr_top_minus_bot=round(sfr_top_minus_bot, 3),
        sfr_top_minus_same_vh=round(sfr_top_minus_same_vh, 3) if not np.isnan(sfr_top_minus_same_vh) else None,
        z_random=round(z_random, 2) if not np.isinf(z_random) else float("inf"),
        top_alpha_frac=round(top_alpha_frac, 2),
        overall_sfr=round(overall_sfr, 3),
        mean_top_abs_eff=round(mean_top_abs_eff, 3),
        classification=kind,
    ))

R = pd.DataFrame(results).sort_values("sfr_top_minus_bot", ascending=False)
R.to_csv(OUT_DIR / "causal_concept_specificity.csv", index=False)

# Pretty print
print(f"\n{'='*120}")
print(f"CAUSAL CONCEPT-SPECIFICITY (TOP={TOP_K}, weighted by n_prompts)")
print(f"{'='*120}")
print(R[["cluster_id", "sfr_top", "sfr_bot", "sfr_same_vh_other",
        "sfr_random_mean", "sfr_top_minus_bot", "sfr_top_minus_same_vh",
        "z_random", "top_alpha_frac", "classification"]].to_string(index=False))

print(f"\nClassification counts: {R.classification.value_counts().to_dict()}")

# Markdown report
md_lines = [
    "# Causal Concept-Specificity (using existing runD ablation data)",
    "",
    f"**Date**: 2026-05-31",
    f"**Source**: `data/analysis/runD_v2/cluster_semantics/cluster_group_scores.csv` "
    f"(per-cluster × per-group SFR from runD ablation on 538 prompts)",
    f"**Method**: For each cluster, compare SFR on TOP-{TOP_K} concept groups vs BOTTOM-{TOP_K} vs random vs same-V_h-other",
    "",
    "## Test logic",
    "",
    "1. **SFR_top**  = mean SFR on top-K cue_groups by |attribution effect| (cluster's 'concept' groups)",
    "2. **SFR_bot**  = mean SFR on bottom-K cue_groups (control: clusters' weakest groups)",
    "3. **SFR_same_vh_other** = mean SFR on dominant-V_h groups NOT in TOP (control: V_h-class effect)",
    "4. **SFR_random** = mean SFR on K random cue_groups (50 reps, baseline)",
    "5. **z_random** = (SFR_top − SFR_random_mean) / SFR_random_std",
    "",
    "**Classifications**:",
    "- CONCEPT-SPECIFIC: SFR_top − SFR_bot > 0.15 AND z_random > 2",
    "- V_h-class effect: SFR_top ≈ SFR_same_vh_other (cluster ablates the WHOLE V_h class, not concept)",
    "- no concept signal: SFR_top − SFR_random < 0.05 (no specificity)",
    "- weak / mixed: intermediate",
    "",
    "## Results",
    "",
    "| cid | SFR_top | SFR_bot | SFR_same_vh | SFR_rand | Δ(top−bot) | Δ(top−sameVh) | z_rand | α_frac | classification |",
    "|-----|---------|---------|-------------|----------|-----------|---------------|--------|--------|----------------|",
]
for _, r in R.iterrows():
    md_lines.append(
        f"| C{int(r.cluster_id)} | {r.sfr_top:.3f} | {r.sfr_bot:.3f} | "
        f"{r.sfr_same_vh_other if r.sfr_same_vh_other is not None else '—'} | "
        f"{r.sfr_random_mean:.3f} | {r.sfr_top_minus_bot:+.3f} | "
        f"{r.sfr_top_minus_same_vh if r.sfr_top_minus_same_vh is not None else '—'} | "
        f"{r.z_random:.2f} | {r.top_alpha_frac:.2f} | {r.classification} |"
    )
md_lines += [
    "",
    "## Summary",
    "",
    f"- CONCEPT-SPECIFIC clusters: {(R.classification == 'CONCEPT-SPECIFIC').sum()}/{len(R)}",
    f"- V_h-class effect clusters: {(R.classification == 'V_h-class effect').sum()}/{len(R)}",
    f"- no concept signal: {(R.classification == 'no concept signal').sum()}/{len(R)}",
    f"- weak / mixed: {(R.classification == 'weak / mixed').sum()}/{len(R)}",
]
(OUT_DIR / "causal_concept_specificity.md").write_text("\n".join(md_lines))
print(f"\nSaved → {OUT_DIR}/causal_concept_specificity.{{csv,md}}")
