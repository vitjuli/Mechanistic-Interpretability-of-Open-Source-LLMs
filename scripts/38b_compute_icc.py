"""
Script 38b: Compute ICC (Condition I.a) from raw transcoder activations.

Implements exactly the formula from Definition 1 (thesis_1_1.md):

    A_G(p) = (1/|G|) * Σ_{(ℓ,k)∈G} a_k^(ℓ)(p)

    σ²_V_h  = Var_g(μ_g)                        between-group variance
    σ²_irr  = E_g[Var_{p∈g}(A_G(p))]            within-group variance (paraphrase)
    ICC(G)  = σ²_V_h / (σ²_V_h + σ²_irr)

    Threshold: τ_ICC = 0.5  (Koo & Mae, 2016 — moderate reliability)

Semantic_equivalence_groups are groups of prompts with the same physical cue
and different wording variants. Within-group variation = paraphrase noise.
Between-group variation = physical state V_h signal.

Also computes one-way ANOVA F-stat and p-value per cluster to test
H0: all group means equal (no V_h signal).

Prerequisites:
    data/analysis/runD_v2/activations/activation_matrix.npy   (script 38)
    data/analysis/runD_v2/activations/feature_ids.txt
    data/analysis/runD_v2/activations/prompt_idxs.txt

Outputs (in --out_dir):
    icc_per_cluster.csv       per-cluster ICC, F-stat, p-value, status
    icc_summary.json          summary statistics

Usage:
    python3 scripts/38b_compute_icc.py \\
        --act_dir       data/analysis/runD_v2/activations \\
        --grouping_dir  data/analysis/runD_v2/grouping \\
        --clustering_dir data/analysis/runD_v2/clustering_full \\
        --out_dir       data/analysis/runD_v2/carrier_stability
"""
import argparse, json, csv as csvlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--act_dir",        type=Path, required=True)
parser.add_argument("--grouping_dir",   type=Path, required=True)
parser.add_argument("--clustering_dir", type=Path, required=True)
parser.add_argument("--out_dir",        type=Path, required=True)
parser.add_argument("--tau_icc",        type=float, default=0.5,
                    help="ICC threshold (default 0.5, Koo & Mae 2016)")
parser.add_argument("--clustering_col", type=str, default="coimp_louvain",
                    help="Column in cluster_labels.csv to use for cluster assignments "
                         "(default: coimp_louvain; use agglo_coimp_k16 for k=16)")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

# ── Load activation matrix ────────────────────────────────────────────────────
act = np.load(args.act_dir / "activation_matrix.npy")  # (n_features, n_prompts)
feature_ids = (args.act_dir / "feature_ids.txt").read_text().strip().split("\n")
prompt_idxs = [int(x) for x in (args.act_dir / "prompt_idxs.txt").read_text().strip().split("\n")]
print(f"Activation matrix: {act.shape}  ({len(feature_ids)} features × {len(prompt_idxs)} prompts)")

feat_to_row = {fid: i for i, fid in enumerate(feature_ids)}
pidx_to_col = {pidx: j for j, pidx in enumerate(prompt_idxs)}

# ── Load cluster assignments ──────────────────────────────────────────────────
with open(args.clustering_dir / "cluster_labels.csv") as f:
    rows = list(csvlib.DictReader(f))
col = args.clustering_col
coimp = {r["feature_id"]: int(r[col]) for r in rows if r.get(col, "") != ""}
from collections import defaultdict
clusters: dict[int, list[str]] = defaultdict(list)
for fid, cid in coimp.items():
    clusters[cid].append(fid)

# ── Load prompt metadata ──────────────────────────────────────────────────────
pm = pd.read_csv(args.grouping_dir / "prompt_metadata.csv")
pm_indexed = pm.set_index("prompt_idx")

# Build array: for each prompt column index j, get semantic_equiv_group and V_h
col_group   = []
col_vh      = []
for pidx in prompt_idxs:
    row = pm_indexed.loc[pidx]
    col_group.append(str(row["semantic_equiv_group"]))
    col_vh.append(str(row["correct_answer"]))

col_group = np.array(col_group)
col_vh    = np.array(col_vh)

unique_groups = sorted(set(col_group))
print(f"Semantic equivalence groups: {len(unique_groups)}")
print(f"V_h values: {sorted(set(col_vh))}")
print(f"Groups with >1 prompt: {sum(1 for g in unique_groups if (col_group==g).sum() > 1)}")

# Groups with only 1 prompt can't contribute to within-group variance
# Keep them for between-group variance but note they have 0 within-group variance
multi_groups = [g for g in unique_groups if (col_group == g).sum() > 1]
print(f"Groups with ≥2 prompts (contribute to σ²_irr): {len(multi_groups)}")


def compute_icc(A_G: np.ndarray) -> tuple[float, float, float, float]:
    """
    A_G: (n_prompts,) — mean cluster activation per prompt.

    Returns (ICC, sigma2_Vh, sigma2_irr, F_stat, p_anova).
    """
    # μ_g = mean within each semantic_equiv_group
    mu_g = np.array([A_G[col_group == g].mean() for g in unique_groups])
    sigma2_Vh = float(np.var(mu_g, ddof=1))  # Var_g(μ_g), between groups

    # E_g[Var_{p∈g}(A_G(p))] — within-group variance, only for multi-prompt groups
    within_vars = []
    for g in unique_groups:
        mask = col_group == g
        n_g = mask.sum()
        if n_g > 1:
            within_vars.append(float(np.var(A_G[mask], ddof=1)))
    sigma2_irr = float(np.mean(within_vars)) if within_vars else 0.0

    denom = sigma2_Vh + sigma2_irr
    icc = float(sigma2_Vh / denom) if denom > 1e-10 else float("nan")

    # One-way ANOVA across all semantic groups
    groups_data = [A_G[col_group == g].tolist() for g in unique_groups
                   if (col_group == g).sum() > 0]
    groups_data = [g for g in groups_data if len(g) > 0]
    if len(groups_data) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups_data)
    else:
        f_stat, p_anova = float("nan"), float("nan")

    return icc, sigma2_Vh, sigma2_irr, float(f_stat), float(p_anova)


# ── Compute ICC per cluster ───────────────────────────────────────────────────
results = []
for cid in sorted(clusters.keys()):
    fids = clusters[cid]
    # Get row indices in activation matrix
    row_idxs = [feat_to_row[fid] for fid in fids if fid in feat_to_row]
    if not row_idxs:
        print(f"  C{cid}: no features found in activation matrix — skip")
        continue

    # A_G(p) = mean activation across cluster features, for each prompt
    A_G = act[row_idxs, :].mean(axis=0)  # (n_prompts,)

    icc, s2_vh, s2_irr, f_stat, p_anova = compute_icc(A_G)

    passes = (icc >= args.tau_icc) if not np.isnan(icc) else False
    status = "PASS" if passes else ("FAIL" if not np.isnan(icc) else "N/A")

    # Mean activation by V_h
    mean_alpha = float(A_G[col_vh == "alpha"].mean()) if (col_vh == "alpha").any() else float("nan")
    mean_beta  = float(A_G[col_vh == "beta"].mean())  if (col_vh == "beta").any()  else float("nan")

    results.append({
        "cluster_id":   cid,
        "n_features":   len(row_idxs),
        "mean_act_alpha": round(mean_alpha, 5),
        "mean_act_beta":  round(mean_beta,  5),
        "mean_act_diff":  round(mean_alpha - mean_beta, 5),
        "sigma2_Vh":    round(s2_vh,   6),
        "sigma2_irr":   round(s2_irr,  6),
        "ICC":          round(icc,     4) if not np.isnan(icc) else float("nan"),
        "F_stat":       round(f_stat,  3) if not np.isnan(f_stat) else float("nan"),
        "p_anova":      float(p_anova) if not np.isnan(p_anova) else float("nan"),
        "passes_tau_icc": passes,
        "status":       status,
    })
    print(f"  C{cid:2d} (n={len(row_idxs):2d}): ICC={icc:.3f}  σ²_Vh={s2_vh:.5f}  σ²_irr={s2_irr:.5f}  "
          f"F={f_stat:.1f}  p={p_anova:.3g}  → {status}")

df = pd.DataFrame(results)
df.to_csv(args.out_dir / "icc_per_cluster.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
n_pass = int(df["passes_tau_icc"].sum())
n_total = len(df)
print(f"\n=== ICC Summary (τ_ICC = {args.tau_icc}) ===")
print(df[["cluster_id", "n_features", "ICC", "F_stat", "p_anova", "status"]].to_string(index=False))
print(f"\n{n_pass}/{n_total} clusters pass ICC ≥ {args.tau_icc}")
print(f"Mean ICC: {df['ICC'].mean():.3f}  Median: {df['ICC'].median():.3f}")

summary = {
    "tau_icc":      args.tau_icc,
    "n_clusters":   n_total,
    "n_pass":       n_pass,
    "n_fail":       n_total - n_pass,
    "mean_icc":     round(float(df["ICC"].mean()), 4),
    "median_icc":   round(float(df["ICC"].median()), 4),
    "min_icc":      round(float(df["ICC"].min()), 4),
    "max_icc":      round(float(df["ICC"].max()), 4),
    "n_groups":     len(unique_groups),
    "n_prompts":    len(prompt_idxs),
}
with open(args.out_dir / "icc_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: icc_per_cluster.csv, icc_summary.json → {args.out_dir}")
print(f"\nInterpretation (Koo & Mae 2016):")
print(f"  ICC < 0.50  — poor reliability (σ²_irr dominates → paraphrase-sensitive)")
print(f"  0.50–0.75   — moderate (V_h signal present but some surface sensitivity)")
print(f"  0.75–0.90   — good (mostly V_h-determined)")
print(f"  ≥ 0.90      — excellent (near-perfect stability)")
