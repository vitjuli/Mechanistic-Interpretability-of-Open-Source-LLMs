#!/usr/bin/env python3
"""
runC_comparison_report.py

Generates RUN_C_COMPARISON_REPORT.md comparing Run B vs Run C.
Run after run_runC_pipeline.sh completes.

INTERPRETATION DISCIPLINE:
- Does not claim "11/12 accepted clusters are statistically significant"
- Does not assert L18/L24 are confirmed gating clusters unless Task 1 three-mode
  ablation confirms direction under mean and resample ablation
- Acceptance status is always labeled DESCRIPTIVE
- Primary statistical signature = low interaction ratio vs matched null clusters
"""

import warnings, json, ast
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--runB_base",   type=Path, default=None)
parser.add_argument("--runC_base",   type=Path, default=None)
parser.add_argument("--validation",  type=Path, default=None)
args, _ = parser.parse_known_args()

ROOT = Path(__file__).resolve().parents[1]
RUNB = Path(args.runB_base)  if args.runB_base  else ROOT / "data/analysis/runB"
RUNC = Path(args.runC_base)  if args.runC_base  else ROOT / "data/analysis/runC_top10_sign_complete"
VAL  = Path(args.validation) if args.validation else ROOT / "data/analysis/runB_validation"
OUT  = RUNC

def safe_load(p, loader=pd.read_csv, **kw):
    p = Path(p)
    return loader(p, **kw) if p.exists() else None

# ── Load Run B data ───────────────────────────────────────────────────────────
runB_fv   = safe_load(RUNB / "cluster_semantics/final_cluster_validation_table.csv")
runB_cl   = safe_load(RUNB / "clustering/cluster_labels.csv")
runB_abl  = safe_load(
    ROOT / "data/results/interventions/physics_decay_type_probe/runB"
           "/intervention_ablation_physics_decay_type_probe.csv")
runB_joint = safe_load(RUNB / "cluster_joint_ablation/joint_ablation_physics_decay_type_probe_train.csv")
runB_null  = safe_load(VAL / "null_cluster_acceptance_results.csv")
runB_stats = {"n_features": 69, "n_pos": 40, "n_neg": 29, "top_k_per_layer": 5}

# ── Load Run C data ───────────────────────────────────────────────────────────
runC_fv    = safe_load(RUNC / "cluster_semantics/final_cluster_validation_table.csv")
runC_cl    = safe_load(RUNC / "clustering/cluster_labels.csv")
runC_abl   = safe_load(
    ROOT / "data/results/interventions/physics_decay_type_probe/runC"
           "/intervention_ablation_physics_decay_type_probe.csv")
runC_joint  = safe_load(RUNC / "cluster_joint_ablation/joint_ablation_physics_decay_type_probe_train.csv")
runC_null   = safe_load(RUNC / "runC_null_cluster_acceptance_results.csv")
runC_stats_path = RUNC / "runC_graph_stats.json"
runC_stats  = json.load(open(runC_stats_path)) if runC_stats_path.exists() else {}

# ── Three-mode task 1 results (optional — may not be done yet) ────────────────
three_mode_path = VAL / "three_mode_ablation_control/L18_L24_three_mode_gating_validation.md"
three_mode_done = three_mode_path.exists()

# ── Fixed-axis analysis for Run C ─────────────────────────────────────────────
runC_gate_l18_dir = runC_gate_l24_dir = "unknown"
if runC_abl is not None:
    runC_abl["correct_token"] = runC_abl["metadata"].apply(
        lambda x: ast.literal_eval(x)["correct_token"])
    runC_abl["is_alpha"] = runC_abl["correct_token"] == " alpha"
    runC_abl["eff_ax"]   = np.where(runC_abl.is_alpha,
                                     runC_abl.effect_size, -runC_abl.effect_size)
    l18_sub = runC_abl[runC_abl.layer == 18]
    l24_sub = runC_abl[runC_abl.layer == 24]
    if len(l18_sub) > 0:
        runC_gate_l18_dir = "pro_alpha"  if l18_sub.eff_ax.mean() < 0 else "anti_alpha"
    if len(l24_sub) > 0:
        runC_gate_l24_dir = "anti_alpha" if l24_sub.eff_ax.mean() > 0 else "pro_alpha"

# ── Run B → Run C cluster mapping ────────────────────────────────────────────
def cluster_features(cl_df):
    if cl_df is None:
        return {}
    return {c: set(g.feature_id) for c, g in cl_df.groupby("coimp_louvain")}

runB_clusters = cluster_features(runB_cl)
runC_clusters = cluster_features(runC_cl)

mapping = []
if runB_cl is not None and runC_cl is not None:
    runB_feats = set(runB_cl.feature_id)
    runC_feats = set(runC_cl.feature_id)
    new_feats  = runC_feats - runB_feats

    for ca, fa in sorted(runB_clusters.items()):
        overlaps = sorted([
            (cb, len(fa & fb) / len(fa | fb), len(fa & fb))
            for cb, fb in runC_clusters.items() if fa & fb
        ], key=lambda x: -x[1])
        if overlaps:
            best_cb, best_j, best_inter = overlaps[0]
            fb = runC_clusters[best_cb]
            n_new = len(fb - fa)
            transition = ("persisted" if best_j > 0.8 else
                          "gained_members" if n_new > 0 and best_j > 0.4 else
                          "split" if len(overlaps) > 1 and overlaps[1][1] > 0.3 else
                          "changed")
            mapping.append({"runB_cluster": ca, "runB_n": len(fa),
                             "best_runC_cluster": best_cb, "jaccard": round(best_j, 3),
                             "n_new": n_new, "n_new_from_runC": len((fb - fa) & new_feats),
                             "transition": transition})
        else:
            mapping.append({"runB_cluster": ca, "runB_n": len(fa),
                             "best_runC_cluster": -1, "jaccard": 0,
                             "n_new": 0, "n_new_from_runC": 0, "transition": "disappeared"})
mapping_df = pd.DataFrame(mapping)

# ── Write report ───────────────────────────────────────────────────────────────
with open(OUT / "RUN_C_COMPARISON_REPORT.md", "w") as f:
    f.write("# Run C Comparison Report: Run B vs Run C\n\n")
    f.write("> **Interpretation discipline**: Acceptance status = DESCRIPTIVE ONLY. "
            "Primary statistical signature = low interaction ratio vs matched null clusters.\n"
            "> L18/L24 gating claims require Task 1 (three-mode ablation) confirmation.\n\n")
    f.write("---\n\n")

    # ── Section 1: Feature budget ───────────────────────────────────────────────
    f.write("## 1. Feature Budget\n\n")
    runC_n_feat = runC_stats.get("n_features", runC_abl.feature_id.nunique() if runC_abl is not None else "unknown")
    runC_n_pos  = runC_stats.get("n_pos", "?")
    runC_n_neg  = runC_stats.get("n_neg", "?")
    f.write("| Property | Run B | Run C |\n|---|---|---|\n")
    f.write(f"| Feature count | {runB_stats['n_features']} | {runC_n_feat} |\n")
    f.write(f"| top_k_per_layer | {runB_stats['top_k_per_layer']} | 10 |\n")
    f.write(f"| Positive attribution | {runB_stats['n_pos']} | {runC_n_pos} |\n")
    f.write(f"| Negative attribution | {runB_stats['n_neg']} | {runC_n_neg} |\n")
    if isinstance(runC_n_pos, int) and isinstance(runC_n_neg, int) and runC_n_pos > 0:
        runC_ratio = runC_n_pos / max(runC_n_neg, 1)
        runB_ratio = runB_stats['n_pos'] / max(runB_stats['n_neg'], 1)
        f.write(f"| Pos:Neg ratio | {runB_ratio:.2f} | {runC_ratio:.2f} |\n")
        if abs(runC_ratio - runB_ratio) > 0.2:
            f.write(f"\n⚠ Pos:Neg ratio changed from {runB_ratio:.2f} to {runC_ratio:.2f}. "
                    "Negative-sign (suppressor) features carry the Run B gating mechanism — "
                    "verify whether suppressor fraction changes material conclusions.\n\n")
    by_layer = runC_stats.get("by_layer", {})
    if by_layer:
        f.write("\n**Run C features by layer:**\n\n")
        f.write("| Layer | n_features |\n|---|---|\n")
        for l, n in sorted(by_layer.items(), key=lambda x: int(x[0])):
            f.write(f"| L{l} | {n} |\n")
    f.write("\n")

    # ── Section 2: Cluster stability ───────────────────────────────────────────
    f.write("## 2. Cluster Stability\n\n")
    if not mapping_df.empty:
        f.write(mapping_df.to_markdown(index=False))
        f.write("\n\n### Specific checks\n\n")
        # L18/L24 persistence
        l18_in_runC = any(runC_cl.feature_id.str.startswith("L18_")) if runC_cl is not None else False
        l24_in_runC = any(runC_cl.feature_id.str.startswith("L24_")) if runC_cl is not None else False
        f.write(f"- **L18 features in Run C**: {'YES' if l18_in_runC else 'NO'} — "
                f"cluster structure {'preserved' if l18_in_runC else 'not present'}\n")
        f.write(f"- **L24 features in Run C**: {'YES' if l24_in_runC else 'NO'} — "
                f"cluster structure {'preserved' if l24_in_runC else 'not present'}\n\n")
        f.write(f"> **Interpretation gate**: L18/L24 structural presence is noted. "
                f"Gating interpretation requires Task 1 (three-mode ablation) confirmation. "
                f"Three-mode analysis {'complete' if three_mode_done else 'PENDING'}.\n\n")
    else:
        f.write("Cluster mapping not available (Run C analysis pending).\n\n")

    # ── Section 3: Redundancy and null comparison ───────────────────────────────
    f.write("## 3. Redundancy and Null Comparison\n\n")
    if runB_fv is not None and runC_fv is not None:
        f.write("### Interaction ratios: Run B vs Run C\n\n")
        f.write("| Run | mean IR | median IR | n_clusters |\n|---|---|---|---|\n")
        for run, fv in [("Run B", runB_fv), ("Run C", runC_fv)]:
            if fv is not None and "interaction_ratio" in fv.columns:
                ir = fv.interaction_ratio
                f.write(f"| {run} | {ir.mean():.4f} | {ir.median():.4f} | {len(fv)} |\n")
        f.write("\n")
    if runC_null is not None:
        n_ir_sig = int(runC_null.ir_significant.sum())
        n_cl     = len(runC_null)
        f.write(f"**Run C null-cluster test**: {n_ir_sig}/{n_cl} real clusters have IR "
                f"significantly below matched null (p<0.05). ")
        if runB_null is not None:
            runB_ir_sig = int(runB_null.ir_significant.sum())
            f.write(f"Run B had {runB_ir_sig}/12. ")
        f.write("Low IR remains the primary statistically robust signature.\n\n")
        f.write("**Acceptance status is DESCRIPTIVE** — null acceptance rate is high "
                f"({runC_null.null_accept_rate.mean():.0%}) as in Run B.\n\n")
    else:
        f.write("Run C null-cluster test not yet run.\n\n")

    # ── Section 4: Beta-prompt asymmetry ───────────────────────────────────────
    f.write("## 4. Beta-Prompt Asymmetry\n\n")
    if runC_abl is not None:
        l24_b_sfr = runC_abl[(runC_abl.layer==24)&~runC_abl.is_alpha].sign_flipped.mean()
        l24_a_sfr = runC_abl[(runC_abl.layer==24)& runC_abl.is_alpha].sign_flipped.mean()
        overall_b = runC_abl[~runC_abl.is_alpha].sign_flipped.mean()
        overall_a = runC_abl[ runC_abl.is_alpha].sign_flipped.mean()
        f.write(f"- Overall β-SFR: {overall_b:.4f}, α-SFR: {overall_a:.4f}, "
                f"ratio: {overall_b/max(overall_a,1e-9):.2f}× (Run B was 5.5% overall)\n")
        f.write(f"- L24 β-SFR: {l24_b_sfr:.4f}, α-SFR: {l24_a_sfr:.4f}, "
                f"ratio: {l24_b_sfr/max(l24_a_sfr,1e-9):.2f}×\n\n")
        asymm = overall_b > overall_a * 2
        f.write(f"**β-prompt asymmetry {'persists' if asymm else 'weaker'} in Run C.**\n\n")
    else:
        f.write("Run C ablation data not yet available.\n\n")

    # ── Section 5: Attribution vs causal direction ──────────────────────────────
    f.write("## 5. Attribution vs Causal Direction\n\n")
    f.write(f"- Run C L18 fixed-axis causal direction: **{runC_gate_l18_dir}** "
            f"(Run B: pro_alpha)\n")
    f.write(f"- Run C L24 fixed-axis causal direction: **{runC_gate_l24_dir}** "
            f"(Run B: anti_alpha)\n\n")
    f.write("⚠ These are layer-aggregate summaries. Feature-level verification requires "
            "running `scripts/runB_validation.py` logic on Run C data.\n\n")
    f.write("> Do not claim global attribution failure. The localized L18/L24 divergence "
            "is restricted to these specific competitive-gating layers unless Run C confirms "
            "it more broadly.\n\n")

    # ── Section 6: Recommendation ──────────────────────────────────────────────
    f.write("## 6. Recommendation\n\n")
    if runC_abl is not None and runC_fv is not None:
        runC_n = runC_abl.feature_id.nunique()
        runC_sfr = runC_abl.sign_flipped.mean()
        runB_sfr = runB_abl.sign_flipped.mean() if runB_abl is not None else 0.055
        sfr_stable = abs(runC_sfr - runB_sfr) < 0.02
        structure_stable = (runC_gate_l18_dir == "pro_alpha" and runC_gate_l24_dir == "anti_alpha")
        f.write(f"**Should Run C replace Run B as the primary analysis?** ")
        if sfr_stable and structure_stable:
            f.write("RUN C AS ROBUSTNESS CHECK — Run B remains primary. "
                    f"Run C confirms structure ({runC_n} features, SFR={runC_sfr:.3f} vs "
                    f"Run B {runB_stats['n_features']} features, SFR={runB_sfr:.3f}). "
                    "Run C adds resolution; Run B is the canonical sign-complete analysis.\n\n")
        else:
            f.write("REVIEW REQUIRED — Run C shows structural differences from Run B. "
                    "Do not switch primary analysis until differences are understood.\n\n")
    else:
        f.write("Run C data incomplete — recommendation pending.\n\n")

    f.write("**Which clusters for cross-prompt patching?**\n")
    f.write("- Priority: Run B C6 (L24-25) and C7 (L18) gating clusters, IF Task 1 confirms "
            "direction under mean+resample ablation.\n")
    f.write("- If Run C confirms the same L18/L24 structural groups, use Run C clusters "
            "for patching experiments (larger feature set = more complete ablation).\n")
    f.write("- Do not use Run C clusters for patching until joint-ablation and null-cluster "
            "tests are complete for Run C.\n\n")

    three_mode_status = "COMPLETE" if three_mode_done else "PENDING"
    f.write("**What remains before writing final thesis chapter?**\n")
    f.write(f"1. Task 1 three-mode ablation: {three_mode_status}\n")
    f.write("2. Run C joint ablation (sbatch: run_probe_runC_joint_ablation.sbatch)\n")
    f.write("3. Run C null-cluster test (after joint ablation)\n")
    f.write("4. Feature-level fixed-axis analysis for Run C (extend runB_validation.py)\n")
    f.write("5. Cross-prompt patching experiments (after above)\n\n")
    f.write("---\n\n*Generated by `scripts/runC_comparison_report.py`*\n")

print(f"Report written: {OUT}/RUN_C_COMPARISON_REPORT.md")
