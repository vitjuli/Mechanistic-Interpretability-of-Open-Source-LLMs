"""
Script 27d: Local analysis of IR null control results.

Run after syncing ir_null_control/ from CSD3 (script 27c output).
Produces plots and a concise statistical report.

Usage:
    python3 scripts/27d_ir_null_analysis.py \\
        --null_dir  data/analysis/runD_v2/ir_null_control \\
        --joint_dir data/analysis/runD_v2/cluster_joint_ablation
"""
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--null_dir",  type=Path, required=True)
parser.add_argument("--joint_dir", type=Path, default=None,
                    help="Dir with real joint ablation results. Optional.")
parser.add_argument("--cluster_col", type=str, default="coimp_louvain",
                    help="Cluster column in cluster_labels.csv (default coimp_louvain; "
                         "use agglo_coimp_subgroup_k30 for sub-cluster IR null)")
args = parser.parse_args()

null_dir  = args.null_dir
joint_dir = args.joint_dir

# ── Load null distribution ────────────────────────────────────────────────────
df_null = pd.read_csv(null_dir / "ir_null_distribution.csv")
df_sum  = pd.read_csv(null_dir / "ir_null_summary.csv")
print(f"Null distribution: {len(df_null)} rows | {df_null['bundle_id'].nunique()} bundles")

print("\n=== Null IR by cluster size ===")
print(df_sum[["size_k", "mean_ir", "median_ir", "std_ir",
              "p5_ir", "p95_ir", "frac_near_1", "frac_below_05"]].to_string(index=False))

# ── Overall null IR stats ─────────────────────────────────────────────────────
ir_all = df_null["ir"].dropna()
print(f"\nOverall null IR across all sizes:")
print(f"  mean={ir_all.mean():.4f}  median={ir_all.median():.4f}  std={ir_all.std():.4f}")
print(f"  p5={np.percentile(ir_all,5):.4f}  p95={np.percentile(ir_all,95):.4f}")

t, p = stats.ttest_1samp(ir_all, 1.0)
print(f"  t-test vs IR=1: t={t:.3f}, p={p:.4g}")
print(f"  → Null IR {'is NOT' if p > 0.05 else 'IS'} significantly different from 1.0 (α=0.05)")

# ── Per-bundle mean IR (bundle-level distribution) ────────────────────────────
bundle_ir = df_null.dropna(subset=["ir"]).groupby("bundle_id")["ir"].mean()
print(f"\nBundle-level mean IR (each bundle averaged over {df_null['prompt_i'].nunique()} prompts):")
print(f"  mean={bundle_ir.mean():.4f}  median={bundle_ir.median():.4f}  std={bundle_ir.std():.4f}")
t2, p2 = stats.ttest_1samp(bundle_ir, 1.0)
print(f"  t-test vs IR=1: t={t2:.3f}, p={p2:.4g}")

# ── Compare against real cluster IR ──────────────────────────────────────────
if args.joint_dir is not None:
    vs_real_csv = null_dir / "ir_null_vs_real.csv"
    real_csv_candidates = (
        [vs_real_csv] if vs_real_csv.exists() else
        list(args.joint_dir.glob("joint_ablation_*.csv"))
    )

    if vs_real_csv.exists():
        comp = pd.read_csv(vs_real_csv)
        print("\n=== Real cluster IR vs Null (from ir_null_vs_real.csv) ===")
        print(comp[["cluster_id", "size_k", "real_mean_ir", "null_mean_ir", "p_value", "significant"]].to_string(index=False))
        n_sig = int(comp["significant"].sum())
        print(f"\n{n_sig}/{len(comp)} clusters have IR significantly below null (p < 0.05)")
        print("Interpretation: p_value = fraction of null bundles with mean IR ≤ real cluster mean IR")
    elif real_csv_candidates:
        real_csv = [f for f in real_csv_candidates if "joint_ablation" in str(f)]
        if real_csv:
            real_df = pd.read_csv(real_csv[0])
            print("\n=== Quick comparison: real cluster IR ===")
            real_ir = (real_df.dropna(subset=["interaction_ratio"])
                       .groupby("cluster_id")["interaction_ratio"]
                       .agg(["mean", "median"]).round(3))
            print(real_ir)

            # Bundle-level comparison
            print("\nPer-cluster empirical p-values:")
            from collections import defaultdict
            clu_csv = args.joint_dir.parent / "clustering_full/cluster_labels.csv"
            if not clu_csv.exists():
                clu_csv = args.joint_dir.parent / "clustering_full/cluster_labels.csv"
            sizes_by_cid = {}
            if clu_csv.exists():
                import csv as csvlib
                with open(clu_csv) as f_:
                    rows_ = list(csvlib.DictReader(f_))
                coimp_ = defaultdict(list)
                for r_ in rows_:
                    if r_.get(args.cluster_col, "") not in ("", None):
                        coimp_[int(r_[args.cluster_col])].append(r_["feature_id"])
                sizes_by_cid = {cid: len(fids) for cid, fids in coimp_.items()}

            for cid, row in real_ir.iterrows():
                real_k = sizes_by_cid.get(int(cid), None)
                null_sub = df_null[df_null["size_k"] == real_k]["ir"].dropna() if real_k else pd.Series([], dtype=float)
                p_val = float((null_sub <= row["mean"]).mean()) if len(null_sub) > 0 else float("nan")
                print(f"  C{cid:2d} (k={real_k:2d}): real_ir={row['mean']:.3f}  p={p_val:.3f}")
else:
    print("\n(No --joint_dir provided; skipping real vs null comparison)")

# ── Markdown report ───────────────────────────────────────────────────────────
md_lines = [
    "# IR Null Control Results",
    "",
    "**Experiment**: Random feature bundles — same sizes as real co-importance clusters.",
    "Tests whether IR ≈ 1 for random features (empirical validation of additivity baseline).",
    "",
    "## Overall null IR distribution",
    "",
    f"- Mean IR_null = {ir_all.mean():.4f} (SD={ir_all.std():.4f})",
    f"- Median IR_null = {ir_all.median():.4f}",
    f"- 5th–95th percentile: [{np.percentile(ir_all,5):.3f}, {np.percentile(ir_all,95):.3f}]",
    f"- t-test vs IR=1.0: t={t:.3f}, p={p:.4g}",
    "",
    "## Null IR by cluster size",
    "",
]
md_lines.append("| k | n_bundles | mean_IR | median_IR | SD | p5 | p95 | frac≈1 |")
md_lines.append("|---|-----------|---------|-----------|-----|-----|------|--------|")
for _, r in df_sum.iterrows():
    md_lines.append(
        f"| {int(r.size_k)} | {int(r.n_bundles)} | {r.mean_ir:.3f} | {r.median_ir:.3f} | "
        f"{r.std_ir:.3f} | {r.p5_ir:.3f} | {r.p95_ir:.3f} | {r.frac_near_1:.2f} |"
    )

md_lines += [
    "",
    "## Interpretation",
    "",
    "If `mean_IR_null ≈ 1.0` for all sizes: random bundles are approximately additive, confirming",
    "that co-importance structure (not chance) drives the low IR observed in real clusters.",
    "",
    "**Conclusion**: " + (
        f"IR_null = {ir_all.mean():.3f} ± {ir_all.std():.3f} "
        f"({'not ' if p > 0.05 else ''}significantly different from 1.0, p={p:.3g})."
    ),
]

report_path = null_dir / "ir_null_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"\nReport saved: {report_path}")
