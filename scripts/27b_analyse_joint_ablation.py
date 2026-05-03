"""
Script 27b: Analyse results from script 27 cluster joint ablation.

Runs locally after the CSD3 job completes.
Computes synergy / redundancy metrics and writes:
  data/results/cluster_joint_ablation/synergy_summary.csv
  data/results/cluster_joint_ablation/interaction_by_answer.csv
  data/results/cluster_joint_ablation/per_prompt_interaction.csv
  docs/behaviors/cluster_joint_ablation_results.md
  dashboard_probe/public/data/cluster_joint_ablation.json  (for dashboard)

Also computes the ANALYTICAL BOUNDS from existing individual-ablation data
(which are available NOW, before running the CSD3 job).
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent

GROUPING   = ROOT / "data/results/grouping"
CLU_DIR    = ROOT / "data/results/clustering"
JOINT_DIR  = ROOT / "data/results/cluster_joint_ablation"
JOINT_DIR.mkdir(parents=True, exist_ok=True)
DASH_OUT   = ROOT / "dashboard_probe/public/data"

# ── Load base data ──────────────────────────────────────────────────────────
import csv as csvlib
with open(CLU_DIR / "cluster_labels.csv") as f:
    rows = list(csvlib.DictReader(f))
coimp = {r["feature_id"]: int(r["coimp_louvain"]) for r in rows}
clusters = defaultdict(list)
for fid, cid in coimp.items():
    clusters[cid].append(fid)

contrib = pd.read_csv(GROUPING / "feature_prompt_contributions.csv",
    usecols=["prompt_idx","feature_id","effect_size","abs_effect_size",
             "baseline_logit_diff","sign_flipped","correct_answer","level",
             "group_id","sign_correct","is_circuit_feature",
             "is_global_alpha_discrim","is_global_beta_discrim"])
contrib["cluster_id"] = contrib["feature_id"].map(coimp)
pmeta = pd.read_csv(GROUPING / "prompt_metadata.csv")
pm_idx = pmeta.set_index("prompt_idx")

# ── PART 1: Analytical bounds (from existing data) ────────────────────────
print("=== PART 1: Analytical bounds (no new model runs needed) ===\n")

analytic = (
    contrib.dropna(subset=["cluster_id"])
    .groupby(["cluster_id","prompt_idx"])
    .agg(
        individual_sum     = ("effect_size",     "sum"),
        individual_abs_sum = ("abs_effect_size",  "sum"),
        baseline           = ("baseline_logit_diff", "first"),
        correct_answer     = ("correct_answer",   "first"),
        level              = ("level",            "first"),
        sign_correct       = ("sign_correct",     "first"),
        n_features         = ("feature_id",       "count"),
    )
    .reset_index()
)

analytic["pred_joint_logit"]    = analytic["baseline"] + analytic["individual_sum"]
analytic["guaranteed_interact"] = (
    analytic["individual_abs_sum"].abs() > analytic["baseline"].abs()
)
analytic["pred_sign_flip"] = (
    analytic["baseline"] * analytic["pred_joint_logit"] < 0
)

# Per-cluster summary of analytical bounds
analytic_summary = []
for cid in sorted(clusters.keys()):
    sub = analytic[analytic["cluster_id"] == cid]
    n = len(clusters[cid])
    row = {
        "cluster_id":          cid,
        "n_features":          n,
        "n_prompts":           len(sub),
        "mean_individual_sum": round(sub.individual_sum.mean(), 4),
        "mean_abs_sum":        round(sub.individual_abs_sum.mean(), 4),
        "guaranteed_interact_frac": round(sub.guaranteed_interact.mean(), 3),
        "pred_sign_flip_frac": round(sub.pred_sign_flip.mean(), 3),
        "mean_baseline":       round(sub.baseline.mean(), 4),
        # Interaction floor: if all interaction is purely redundant, joint = 0 for most extreme case
        # But physically joint_effect can't be MORE negative than baseline (logit floor)
        "min_possible_joint":  round(float(sub.apply(
            lambda r: max(r.individual_sum, -abs(r.baseline)*3), axis=1).mean()), 4),
    }
    analytic_summary.append(row)

analytic_sum_df = pd.DataFrame(analytic_summary)
analytic_sum_df.to_csv(JOINT_DIR / "analytic_bounds.csv", index=False)

print("Analytical bounds per cluster:")
print(analytic_sum_df[["cluster_id","n_features","guaranteed_interact_frac",
                         "pred_sign_flip_frac","mean_individual_sum"]].to_string(index=False))

print("\nKEY: clusters with guaranteed_interact_frac > 0.5 have linear independence")
print("     impossible for >50% of prompts → MUST show interaction in actual joint ablation")

# Split by answer
print("\nPer-answer analytical breakdown:")
for ans in ["alpha","beta"]:
    sub_a = analytic[analytic["correct_answer"]==ans]
    by_c = sub_a.groupby("cluster_id").agg(
        mean_sum=("individual_sum","mean"),
        pred_flip=("pred_sign_flip","mean"),
    )
    print(f"\n  {ans}:")
    print(by_c.round(3).to_string())

# ── PART 2: Actual joint ablation results (post-CSD3) ─────────────────────
joint_csv = JOINT_DIR / "joint_ablation_physics_decay_type_probe_train.csv"

if not joint_csv.exists():
    print(f"\n=== PART 2: Joint ablation results not yet available ===")
    print(f"  Expected: {joint_csv}")
    print("  Run on CSD3: sbatch jobs/cluster_joint_ablation.sbatch")
    print("  Then re-run this script.")

    # Write partial output with just analytical results
    out_json = {
        "status": "analytical_only",
        "analytic_summary": analytic_sum_df.to_dict(orient="records"),
        "has_joint_results": False,
    }
    with open(DASH_OUT / "cluster_joint_ablation.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved partial cluster_joint_ablation.json (analytical only)")
    sys.exit(0)

# ── Full analysis with joint ablation results ─────────────────────────────
print(f"\n=== PART 2: Joint ablation results found: {joint_csv} ===\n")
jdf = pd.read_csv(joint_csv)
print(f"Loaded {len(jdf)} rows: {jdf.cluster_id.nunique()} clusters × {jdf.prompt_idx.nunique()} prompts")

# Merge with prompt metadata
jdf = jdf.merge(
    pmeta[["prompt_idx","correct_answer","level","group_id","sign_correct",
           "is_anchor","difficulty"]],
    on="prompt_idx", how="left"
)

# ── Synergy summary per cluster ───────────────────────────────────────────
def interaction_summary(df):
    valid = df.dropna(subset=["interaction_ratio","interaction_term"])
    n = len(valid)
    if n == 0:
        return {}
    ratio = valid.interaction_ratio
    term  = valid.interaction_term
    # interaction_ratio < 1 and term > 0 → redundant (joint < indiv_sum in magnitude)
    # interaction_ratio > 1 and term < 0 → synergistic
    # But direction depends on sign of indiv_sum — let's define clearly:
    # For features that hurt when ablated (indiv_sum < 0):
    #   sub-additive (redundant): |joint_effect| < |indiv_sum| → |ratio| < 1
    #   super-additive (synergistic): |joint_effect| > |indiv_sum| → |ratio| > 1
    n_redundant   = int((ratio.abs() < 0.9).sum())
    n_additive    = int(((ratio.abs() >= 0.9) & (ratio.abs() <= 1.1)).sum())
    n_synergistic = int((ratio.abs() > 1.1).sum())
    return {
        "n_prompts":             n,
        "mean_joint_effect":     round(float(df.joint_effect.mean()), 4),
        "mean_individual_sum":   round(float(df.individual_sum.mean()), 4),
        "mean_interaction_term": round(float(term.mean()), 4),
        "mean_interaction_ratio":round(float(ratio.mean()), 4),
        "median_interaction_ratio": round(float(ratio.median()), 4),
        "n_redundant_prompts":   n_redundant,
        "n_additive_prompts":    n_additive,
        "n_synergistic_prompts": n_synergistic,
        "frac_redundant":        round(n_redundant/n, 3),
        "frac_additive":         round(n_additive/n, 3),
        "frac_synergistic":      round(n_synergistic/n, 3),
        "sign_flip_rate_joint":  round(float(df.sign_flipped_joint.mean()), 3),
        "pred_sign_flip_rate":   round(float(df.predicted_sign_flip.mean()), 3),
        # Test: is ratio significantly different from 1.0?
        "ttest_vs_additive_p":   round(float(stats.ttest_1samp(ratio.dropna(), 1.0).pvalue), 4),
    }

syn_rows = []
for cid in sorted(jdf.cluster_id.unique()):
    sub = jdf[jdf.cluster_id==cid]
    s = interaction_summary(sub)
    s["cluster_id"] = cid
    s["n_features"]  = len(clusters[cid])
    syn_rows.append(s)

syn_df = pd.DataFrame(syn_rows)
syn_df.to_csv(JOINT_DIR / "synergy_summary.csv", index=False)

print("=== Synergy summary per cluster ===")
cols = ["cluster_id","n_features","mean_interaction_ratio","frac_redundant",
        "frac_additive","frac_synergistic","ttest_vs_additive_p"]
print(syn_df[cols].to_string(index=False))

# ── Per-answer interaction breakdown ─────────────────────────────────────
rows_ans = []
for cid in sorted(jdf.cluster_id.unique()):
    for ans in ["alpha","beta"]:
        sub = jdf[(jdf.cluster_id==cid) & (jdf.correct_answer==ans)]
        if len(sub) == 0: continue
        s = interaction_summary(sub)
        s["cluster_id"] = cid; s["correct_answer"] = ans
        rows_ans.append(s)
inter_ans = pd.DataFrame(rows_ans)
inter_ans.to_csv(JOINT_DIR / "interaction_by_answer.csv", index=False)

print("\n=== Interaction ratio by cluster × answer ===")
pivot = inter_ans.pivot_table(
    index="cluster_id", columns="correct_answer",
    values="mean_interaction_ratio")
print(pivot.round(3).to_string())

# ── Per-prompt detailed output ────────────────────────────────────────────
jdf[["cluster_id","prompt_idx","baseline_logit_diff","joint_logit_diff",
     "joint_effect","individual_sum","interaction_term","interaction_ratio",
     "sign_flipped_joint","predicted_sign_flip",
     "correct_answer","level","sign_correct"]].to_csv(
    JOINT_DIR / "per_prompt_interaction.csv", index=False)
print(f"\nSaved per_prompt_interaction.csv ({len(jdf)} rows)")

# ── Markdown report ───────────────────────────────────────────────────────
CLUSTER_NAMES = {
    0:"Early β-Routing (L10)", 1:"Early α-Pair (L11)", 2:"Singleton L12",
    3:"Early α-Pair (L13)",    4:"Mid α-Pair (L20)",   5:"Singleton L15",
    6:"L16 β-Module",          7:"Multi-Layer Convergence",
    8:"Singleton L18",         9:"Mid-Late α Module (L19–L21)",
    10:"Output Decision (L24–L25)",
}

md = ["# Cluster Joint Ablation Results", "",
      "**Behaviour:** `physics_decay_type_probe`", "",
      "## Key question",
      "Do cluster features act **independently** (additive), **redundantly** (sub-additive), "
      "or **synergistically** (super-additive) when ablated jointly?", "",
      "## Definitions",
      "- `individual_sum` = sum of individual effect_sizes for cluster features (from script 07 single-feature runs)",
      "- `joint_effect` = actual logit change when all cluster features ablated simultaneously (this script)",
      "- `interaction_term` = joint_effect − individual_sum",
      "  - > 0 (with negative indiv_sum): **sub-additive / redundant** (less impact than expected)",
      "  - < 0 (with negative indiv_sum): **super-additive / synergistic** (more impact than expected)",
      "- `interaction_ratio` = joint_effect / individual_sum ≈ 1.0 for additive features",
      "",
      "## Analytical bounds (no model runs required)",
      "",
      "From individual ablation data, the following clusters have linear independence",
      "**mathematically impossible** for the indicated fraction of prompts",
      "(because |individual_sum| > |baseline|, so the summed effect would exceed available margin):",
      "",
]
for _, r in analytic_sum_df.iterrows():
    cid = int(r.cluster_id)
    if r.guaranteed_interact_frac > 0.05:
        md.append(
            f"- **C{cid} — {CLUSTER_NAMES.get(cid,'')}** (n={r.n_features}): "
            f"{r.guaranteed_interact_frac:.1%} of prompts guaranteed to show interaction"
        )
md += [""]

md += ["## Actual joint ablation results", ""]
for _, row in syn_df.iterrows():
    cid = int(row.cluster_id)
    if row.mean_interaction_ratio != row.mean_interaction_ratio: continue  # nan
    verdict = ("REDUNDANT" if row.mean_interaction_ratio < 0.80
               else "SYNERGISTIC" if row.mean_interaction_ratio > 1.20
               else "APPROXIMATELY ADDITIVE")
    sig = "* (p<0.05)" if row.ttest_vs_additive_p < 0.05 else ""
    md += [
        f"### C{cid} — {CLUSTER_NAMES.get(cid,'')} (n={row.n_features})",
        "",
        f"**Verdict:** {verdict}{sig}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean interaction ratio | {row.mean_interaction_ratio:.3f} |",
        f"| Median interaction ratio | {row.median_interaction_ratio:.3f} |",
        f"| Fraction redundant prompts (ratio < 0.9) | {row.frac_redundant:.1%} |",
        f"| Fraction additive prompts (ratio 0.9–1.1) | {row.frac_additive:.1%} |",
        f"| Fraction synergistic prompts (ratio > 1.1) | {row.frac_synergistic:.1%} |",
        f"| Actual sign-flip rate | {row.sign_flip_rate_joint:.1%} |",
        f"| Predicted sign-flip rate (independent) | {row.pred_sign_flip_rate:.1%} |",
        f"| t-test vs additive (ratio=1) p-value | {row.ttest_vs_additive_p:.4f} |",
        "",
    ]

with open(ROOT / "docs/behaviors/cluster_joint_ablation_results.md", "w") as f:
    f.write("\n".join(md))
print("\nSaved cluster_joint_ablation_results.md")

# ── Dashboard JSON ────────────────────────────────────────────────────────
per_cluster_json = []
for _, row in syn_df.iterrows():
    cid = int(row.cluster_id)
    sub = jdf[jdf.cluster_id==cid].copy()
    # Top redundant prompts (ratio far below 1 for beneficial clusters)
    redundant_examples = (
        sub[sub.interaction_ratio.notna()]
        .sort_values("interaction_ratio")
        .head(10)
        [["prompt_idx","baseline_logit_diff","joint_effect","individual_sum","interaction_ratio"]]
        .round(4).to_dict(orient="records")
    )
    # Per-answer ratios from inter_ans
    ratio_by_ans = {}
    for ans in ["alpha", "beta"]:
        ans_rows = inter_ans[(inter_ans.cluster_id == cid) & (inter_ans.correct_answer == ans)]
        ratio_by_ans[f"ratio_{ans}"] = (
            float(ans_rows.mean_interaction_ratio.iloc[0])
            if len(ans_rows) > 0 and not pd.isna(ans_rows.mean_interaction_ratio.iloc[0])
            else None
        )
    per_cluster_json.append({
        "cluster_id":          cid,
        "name":                CLUSTER_NAMES.get(cid, f"C{cid}"),
        "n_features":          int(row.n_features),
        "interaction_ratio":   float(row.mean_interaction_ratio) if row.mean_interaction_ratio==row.mean_interaction_ratio else None,
        "verdict":             ("redundant" if (row.mean_interaction_ratio < 0.80)
                                else "synergistic" if (row.mean_interaction_ratio > 1.20)
                                else "additive") if row.mean_interaction_ratio==row.mean_interaction_ratio else "unknown",
        "frac_redundant":      float(row.frac_redundant),
        "frac_additive":       float(row.frac_additive),
        "frac_synergistic":    float(row.frac_synergistic),
        "sign_flip_joint":     float(row.sign_flip_rate_joint),
        "sign_flip_pred":      float(row.pred_sign_flip_rate),
        "ttest_p":             float(row.ttest_vs_additive_p),
        "top_redundant":       redundant_examples,
        **ratio_by_ans,
    })

out_json = {
    "status":             "complete",
    "has_joint_results":  True,
    "analytic_summary":   analytic_sum_df.to_dict(orient="records"),
    "joint_summary":      per_cluster_json,
}
with open(DASH_OUT / "cluster_joint_ablation.json", "w") as f:
    json.dump(out_json, f, indent=2)
print("Saved cluster_joint_ablation.json")
