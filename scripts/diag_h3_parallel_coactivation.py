"""
diag_h3_parallel_coactivation.py

H3-parallel (OASR-style): do multiple clusters co-activate on the same prompts
during normal forward passes? If yes → parallel circuits, ablating one is
absorbed because others independently produce the same answer.

Distinct from H3-reactive (Wang 2022 backup heads) which appears ONLY post-ablation.

Reads:
  dashboard_probe/public/data/cluster_activation_map.json

Outputs:
  data/analysis/iia_failure_diagnosis/h3_parallel_coactivation.csv
  data/analysis/iia_failure_diagnosis/h3_summary.json

Logic:
  - For each prompt, get cluster_eff (signed cluster effect = grad × act)
  - Split prompts by correct_answer (α vs β)
  - On α-prompts: cluster contributes "for α" iff cluster_eff > 0 (helps correct answer)
  - Compute pairwise Pearson correlation between clusters on α-prompts only and β-prompts only
  - Positive correlations between several clusters on α-prompts → coordinated parallel ensemble
  - Also report co-activation count: in how many prompts do ≥ K clusters fire above threshold?

Literature:
  - Chen et al. 2026 OASR — parallel sheaves of circuits
  - Anthropic 2025 — "many features fire jointly during ordinary inference"
  - Wang et al. 2023 — IOI backup heads (this is the REACTIVE alternative)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_act_json",
                    default="dashboard_probe/public/data/cluster_activation_map.json")
    ap.add_argument("--threshold_abs", type=float, default=0.3,
                    help="cluster_abs > threshold counts as 'firing'")
    ap.add_argument("--out_dir",
                    default="data/analysis/iia_failure_diagnosis")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    d = json.load(open(root / args.cluster_act_json))

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = sorted([c["id"] for c in d["clusters"]])
    cluster_names = {c["id"]: c["name"] for c in d["clusters"]}
    K = len(cluster_ids)
    n_prompts = len(d["groups"])

    # ── build matrices ──────────────────────────────────────────────────────
    eff = np.zeros((n_prompts, K))   # signed effect
    act = np.zeros((n_prompts, K))   # abs activation
    answers = []
    levels = []
    for i, g in enumerate(d["groups"]):
        for cid in cluster_ids:
            eff[i, cid] = float(g["cluster_eff"].get(str(cid), 0))
            act[i, cid] = float(g["cluster_abs"].get(str(cid), 0))
        answers.append(g["correct_answer"])
        levels.append(g.get("level_label", ""))

    answers = np.array(answers)
    alpha_mask = answers == "alpha"
    beta_mask  = answers == "beta"

    print(f"Loaded {n_prompts} prompts, {K} clusters, "
          f"{alpha_mask.sum()} α, {beta_mask.sum()} β")

    # ── pairwise correlations on α-prompts and β-prompts separately ─────────
    eff_alpha = eff[alpha_mask]   # (n_α, K)
    eff_beta  = eff[beta_mask]

    corr_alpha = np.corrcoef(eff_alpha.T)   # (K, K)
    corr_beta  = np.corrcoef(eff_beta.T)

    # Off-diagonal mean (strength of coordinated activation)
    mask_off = ~np.eye(K, dtype=bool)
    mean_corr_alpha = corr_alpha[mask_off].mean()
    mean_corr_beta  = corr_beta[mask_off].mean()
    pos_pairs_alpha = ((corr_alpha > 0.3) & mask_off).sum() // 2
    pos_pairs_beta  = ((corr_beta  > 0.3) & mask_off).sum() // 2

    print(f"\n=== Pairwise correlations of cluster_eff ===")
    print(f"  α-prompts: mean off-diag corr = {mean_corr_alpha:+.3f}, "
          f"{pos_pairs_alpha} pairs with r > 0.3 (max possible = {K*(K-1)//2})")
    print(f"  β-prompts: mean off-diag corr = {mean_corr_beta:+.3f}, "
          f"{pos_pairs_beta} pairs with r > 0.3")

    # ── co-activation count: how many clusters fire on the same prompt? ────
    firing = act > args.threshold_abs                # (n_prompts, K)
    n_firing = firing.sum(axis=1)
    print(f"\n=== Co-firing distribution (|act|>{args.threshold_abs}) ===")
    print(f"  Mean clusters firing per prompt: {n_firing.mean():.2f}")
    print(f"  Median: {np.median(n_firing)}, Max: {n_firing.max()}")
    print(f"  Prompts with ≥ 3 clusters firing: {(n_firing >= 3).sum()}/{n_prompts} "
          f"({100*(n_firing >= 3).mean():.1f}%)")
    print(f"  Prompts with ≥ 5 clusters firing: {(n_firing >= 5).sum()}/{n_prompts} "
          f"({100*(n_firing >= 5).mean():.1f}%)")

    # ── identify the "parallel ensemble": which clusters are co-active most? ─
    co_act_matrix = firing.astype(int).T @ firing.astype(int)  # (K, K)
    np.fill_diagonal(co_act_matrix, 0)
    # Top co-firing pairs
    top_pairs = []
    for i in range(K):
        for j in range(i+1, K):
            if co_act_matrix[i, j] > 0:
                top_pairs.append({
                    "ci": i, "cj": j,
                    "co_fire_count": int(co_act_matrix[i, j]),
                    "co_fire_frac": float(co_act_matrix[i, j] / n_prompts),
                    "corr_alpha": float(corr_alpha[i, j]),
                    "corr_beta":  float(corr_beta[i, j]),
                })
    top_pairs.sort(key=lambda x: x["co_fire_count"], reverse=True)
    print(f"\n=== Top 10 co-firing cluster pairs ===")
    print(f"{'(i,j)':>9}  {'co_fire':>8}  {'frac':>6}  {'r_α':>7}  {'r_β':>7}  i.name → j.name")
    for p in top_pairs[:10]:
        name_i = cluster_names[p["ci"]][:25]
        name_j = cluster_names[p["cj"]][:25]
        print(f"  C{p['ci']:2d},C{p['cj']:2d}  {p['co_fire_count']:>8d}  "
              f"{p['co_fire_frac']:.3f}  {p['corr_alpha']:+.3f}  {p['corr_beta']:+.3f}  "
              f"{name_i} → {name_j}")

    # ── α/β-specific: which clusters help α and which help β? ───────────────
    # On α-prompts, cluster_eff > 0 means cluster helps α. We want clusters
    # that help α on α-prompts AND help β on β-prompts (single-role).
    # If multiple clusters help α SAME prompts → parallel α-circuit.
    cluster_role = []
    for cid in cluster_ids:
        e_a = eff[alpha_mask, cid]
        e_b = eff[beta_mask,  cid]
        helps_alpha = (e_a > 0).mean()
        helps_beta  = (e_b > 0).mean()
        cluster_role.append({
            "cluster_id":   int(cid),
            "name":         cluster_names[cid],
            "mean_eff_α":   float(e_a.mean()),
            "mean_eff_β":   float(e_b.mean()),
            "frac_helps_α_on_α_prompts": float(helps_alpha),
            "frac_helps_β_on_β_prompts": float(helps_beta),
        })
    print(f"\n=== Cluster role (signed effect on α vs β prompts) ===")
    print(f"{'cid':>4}  {'mean_eff_α':>11}  {'mean_eff_β':>11}  {'role':>15}  name")
    for r in cluster_role:
        if r["mean_eff_α"] > 0.05 and r["mean_eff_β"] < -0.05:
            role = "α-supporting"
        elif r["mean_eff_α"] < -0.05 and r["mean_eff_β"] > 0.05:
            role = "β-supporting"
        elif r["mean_eff_α"] > 0.05 and r["mean_eff_β"] > 0.05:
            role = "both/generic"
        else:
            role = "neutral/weak"
        print(f"  C{r['cluster_id']:2d}  {r['mean_eff_α']:>+11.4f}  {r['mean_eff_β']:>+11.4f}  "
              f"{role:>15}  {r['name'][:40]}")

    # Count of α-supporting clusters and β-supporting clusters
    n_alpha_clusters = sum(1 for r in cluster_role
                           if r["mean_eff_α"] > 0.05 and r["mean_eff_β"] < -0.05)
    n_beta_clusters  = sum(1 for r in cluster_role
                           if r["mean_eff_α"] < -0.05 and r["mean_eff_β"] > 0.05)
    print(f"\n  → {n_alpha_clusters} clearly α-supporting, "
          f"{n_beta_clusters} clearly β-supporting clusters")

    # ── verdict ──────────────────────────────────────────────────────────────
    parallel_evidence = []
    if n_firing.mean() >= 5:
        parallel_evidence.append("≥5 clusters fire per prompt on average (high co-firing)")
    if mean_corr_alpha > 0.3 or mean_corr_beta > 0.3:
        parallel_evidence.append("mean inter-cluster corr > 0.3 (coordinated activation)")
    if n_alpha_clusters >= 2 and n_beta_clusters >= 2:
        parallel_evidence.append(f"{n_alpha_clusters} α-clusters + {n_beta_clusters} β-clusters "
                                 "(redundant role coverage)")

    if len(parallel_evidence) >= 2:
        verdict = "H3-parallel SUPPORTED: multiple coordinated cluster ensembles"
    elif len(parallel_evidence) == 1:
        verdict = "H3-parallel WEAK: some parallel structure but not strong"
    else:
        verdict = "H3-parallel NOT SUPPORTED in co-activation"

    print(f"\n=== Verdict ===")
    print(f"  {verdict}")
    for e in parallel_evidence:
        print(f"    + {e}")

    # ── save ─────────────────────────────────────────────────────────────────
    pd.DataFrame(top_pairs).to_csv(out_dir / "h3_top_cofire_pairs.csv", index=False)
    pd.DataFrame(cluster_role).to_csv(out_dir / "h3_cluster_roles.csv", index=False)

    summary = {
        "n_prompts": n_prompts,
        "n_alpha":   int(alpha_mask.sum()),
        "n_beta":    int(beta_mask.sum()),
        "n_clusters": K,
        "threshold_abs": args.threshold_abs,
        "mean_corr_off_diag_alpha": float(mean_corr_alpha),
        "mean_corr_off_diag_beta":  float(mean_corr_beta),
        "n_pos_pairs_alpha":  int(pos_pairs_alpha),
        "n_pos_pairs_beta":   int(pos_pairs_beta),
        "mean_clusters_firing_per_prompt": float(n_firing.mean()),
        "frac_prompts_ge3_firing": float((n_firing >= 3).mean()),
        "frac_prompts_ge5_firing": float((n_firing >= 5).mean()),
        "n_alpha_clusters":  n_alpha_clusters,
        "n_beta_clusters":   n_beta_clusters,
        "parallel_evidence": parallel_evidence,
        "verdict":           verdict,
    }
    with open(out_dir / "h3_parallel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_dir}/h3_*.{{csv,json}}")


if __name__ == "__main__":
    main()
