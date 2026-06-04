"""
For each of the 30 sub-clusters, identify which cue groups are simultaneously:
  (1) significantly moved by joint ablation (large |joint_effect|)
  (2) actually flipped (sign_flipped_joint == True)

The intersection tells us which cue groups the cluster CAUSALLY controls
(not just moves but actually changes the answer).

Output: per-cluster table with top cue groups, ranked by (n_flipped, mean |effect|).
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

ROOT = Path("/Users/julia/Desktop/courses/thesis/project")
JAS = ROOT / "data/analysis/runD_v2/cluster_joint_ablation_subgroup"
SUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp"
PRMS = ROOT / "data/prompts/physics_decay_type_probe_v2_train.jsonl"

# ── Load ─────────────────────────────────────────────────────────────────────
ja = pd.read_csv(JAS / "joint_ablation_physics_decay_type_probe_v2_train.csv")
prompts = [json.loads(l) for l in open(PRMS)]

cue_of = {i: (p.get("cue_type") or "AUX") for i, p in enumerate(prompts)}
ans_of = {i: p.get("correct_answer", "").strip() for i, p in enumerate(prompts)}

ja["cue_type"]      = ja["prompt_idx"].map(cue_of)
ja["correct_ans"]   = ja["prompt_idx"].map(ans_of)
ja["abs_effect"]    = ja["joint_effect"].abs() if "joint_effect" in ja.columns else (ja["joint_logit_diff"] - ja["baseline_logit_diff"]).abs()
ja["sign_flipped"]  = ja["sign_flipped_joint"].astype(bool)

print(f"Loaded: {len(ja)} rows, {ja['cluster_id'].nunique()} clusters × {ja['prompt_idx'].nunique()} prompts")
print()

# ── Per-cluster × per-cue table ──────────────────────────────────────────────
# For each (cluster_id, cue_type): n_prompts, n_flipped, mean |effect|, max |effect|
agg = ja.groupby(["cluster_id", "cue_type", "correct_ans"]).agg(
    n_prompts=("prompt_idx", "count"),
    n_flipped=("sign_flipped", "sum"),
    mean_abs_effect=("abs_effect", "mean"),
    max_abs_effect=("abs_effect", "max"),
    mean_signed_effect=("joint_effect", "mean") if "joint_effect" in ja.columns else (
        "joint_logit_diff", lambda s: float((s - ja.loc[s.index, "baseline_logit_diff"]).mean())
    ),
).reset_index()
agg["flip_rate"] = agg["n_flipped"] / agg["n_prompts"]

# Filter to cue groups with at least 3 prompts (otherwise meaningless)
agg = agg[agg["n_prompts"] >= 3].copy()

# ── For each cluster, find top cue groups (dual criterion) ───────────────────
# Score: combination of flip rate AND mean abs effect normalized
# Rank within cluster by n_flipped descending, then by mean_abs_effect

print("="*100)
print("PER-CLUSTER TOP CUE GROUPS (DUAL CRITERION: large |effect| AND flips)")
print("="*100)
print()

per_cluster_top_rows = []
cluster_ids = sorted(agg["cluster_id"].unique())

# Also need per-cluster metadata (layers, n_features, single SFR)
iia_meta = pd.read_csv(SUB / "iia_probe_clusters.csv").set_index("cluster")
single_meta = pd.read_csv(JAS / "single_cluster_additive_sfr.csv").set_index("id") if (JAS / "single_cluster_additive_sfr.csv").exists() else None

for cid in cluster_ids:
    sub = agg[agg["cluster_id"] == cid].copy()
    # rank: by n_flipped first, then by mean_abs_effect
    sub = sub.sort_values(["n_flipped", "mean_abs_effect"], ascending=[False, False])

    layers_str = str(iia_meta.loc[cid, "layers"]) if cid in iia_meta.index else "?"
    nfeat = int(iia_meta.loc[cid, "n_features"]) if cid in iia_meta.index else 0
    single_sfr = float(single_meta.loc[cid, "additive_sfr"]) if (single_meta is not None and cid in single_meta.index) else None

    print(f"━━━ CLUSTER {cid}  (layers={layers_str}, n_features={nfeat}"
          + (f", single SFR={single_sfr:.3f}" if single_sfr is not None else "")
          + ") ━━━")

    if (sub["n_flipped"] > 0).sum() == 0:
        print("  (no cue group has any flip — cluster doesn't causally control any cue group)")
        per_cluster_top_rows.append({
            "cluster": cid, "rank": 0, "cue_type": "—",
            "ans_class": "—", "n_prompts": 0, "n_flipped": 0,
            "flip_rate": 0.0, "mean_abs_effect": 0.0, "mean_signed_effect": 0.0,
        })
        print()
        continue

    print(f"{'rank':>4} {'cue_type':<35} {'class':<6} {'n':>4} {'flipped':>8} "
          f"{'rate':>6} {'mean|eff|':>10} {'mean(eff)':>10}")
    print("  " + "-"*95)
    top_n = sub.head(8)
    for rank, (_, r) in enumerate(top_n.iterrows(), 1):
        if r["n_flipped"] == 0 and rank > 3:
            break
        ind = "★" if (r["n_flipped"] >= 3 and r["mean_abs_effect"] > 0.3) else (
              "•" if r["n_flipped"] >= 1 else " ")
        print(f"  {ind}{rank:>3} {r['cue_type']:<35} {r['correct_ans']:<6} "
              f"{int(r['n_prompts']):>4} {int(r['n_flipped']):>8} "
              f"{r['flip_rate']:>6.2f} {r['mean_abs_effect']:>10.3f} "
              f"{r['mean_signed_effect']:>+10.3f}")
        per_cluster_top_rows.append({
            "cluster": cid, "rank": rank,
            "cue_type": r["cue_type"], "ans_class": r["correct_ans"],
            "n_prompts": int(r["n_prompts"]),
            "n_flipped": int(r["n_flipped"]),
            "flip_rate": float(r["flip_rate"]),
            "mean_abs_effect": float(r["mean_abs_effect"]),
            "mean_signed_effect": float(r["mean_signed_effect"]),
            "layers": layers_str,
            "cluster_n_features": nfeat,
            "single_sfr": single_sfr,
        })
    print()

# ── Save the compact table ───────────────────────────────────────────────────
out_df = pd.DataFrame(per_cluster_top_rows)
out_path = JAS / "cluster_cue_dual_criterion.csv"
out_df.to_csv(out_path, index=False)
print(f"\nsaved → {out_path}")

# Also save the full agg
agg.to_csv(JAS / "cluster_cue_full.csv", index=False)
print(f"saved → {JAS / 'cluster_cue_full.csv'}")

# ── Cross-cluster cue heatmap data ───────────────────────────────────────────
# Pivot: rows = cluster, cols = cue_type, values = (n_flipped, mean_abs_effect)
pivot_flips = agg.pivot_table(index="cluster_id", columns="cue_type",
                              values="n_flipped", fill_value=0).astype(int)
pivot_effect = agg.pivot_table(index="cluster_id", columns="cue_type",
                                values="mean_abs_effect", fill_value=0).round(3)

pivot_flips.to_csv(JAS / "pivot_cluster_cue_flips.csv")
pivot_effect.to_csv(JAS / "pivot_cluster_cue_effect.csv")
print(f"saved → pivot_cluster_cue_flips.csv  (rows=cluster, cols=cue, value=n_flipped)")
print(f"saved → pivot_cluster_cue_effect.csv (rows=cluster, cols=cue, value=mean|effect|)")

# Quick summary
print()
print("="*100)
print("CROSS-CLUSTER PATTERN: which cue groups have ANY cluster causally controlling them?")
print("="*100)
cue_summary = agg.groupby("cue_type").agg(
    n_clusters_with_flips=("n_flipped", lambda s: (s > 0).sum()),
    total_flips=("n_flipped", "sum"),
    max_flip_rate=("flip_rate", "max"),
    best_cluster=("flip_rate", lambda s: int(agg.loc[s.idxmax(), "cluster_id"])),
).reset_index().sort_values("total_flips", ascending=False)
print(cue_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
