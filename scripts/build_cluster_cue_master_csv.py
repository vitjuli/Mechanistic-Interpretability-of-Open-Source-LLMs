"""
Build a comprehensive cluster × cue master CSV for union analysis.

Outputs:
  1. cluster_cue_master_long.csv     — long format, one row per (cluster, cue, class)
                                       with all metrics (n, flipped, effect, etc.)
  2. cluster_cue_signed_effect.csv   — wide pivot, rows=cluster, cols=cue
                                       value = mean signed joint_effect
                                       (use for additive union prediction)
  3. cluster_cue_flip_rate.csv       — wide pivot, value = flip_rate
  4. cluster_cue_strength.csv        — wide pivot, value = combined strength
                                       (flip_rate × mean|effect|)
  5. cluster_cue_n_prompts.csv       — wide pivot, value = n_prompts tested
"""
import json
from pathlib import Path
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

ja["cue_type"] = ja["prompt_idx"].map(cue_of)
ja["correct_ans"] = ja["prompt_idx"].map(ans_of)
ja["signed_effect"] = ja["joint_logit_diff"] - ja["baseline_logit_diff"]
ja["abs_effect"] = ja["signed_effect"].abs()
ja["sign_flipped"] = ja["sign_flipped_joint"].astype(bool)

# Per-cluster meta
iia_meta = pd.read_csv(SUB / "iia_probe_clusters.csv").set_index("cluster")

print(f"Loaded: {len(ja)} rows, {ja['cluster_id'].nunique()} clusters, "
      f"{ja['cue_type'].nunique()} cue types")
print()

# ── 1. Long-format master CSV ────────────────────────────────────────────────
# Group by (cluster, cue, class) and aggregate everything
agg = ja.groupby(["cluster_id", "cue_type", "correct_ans"]).agg(
    n_prompts=("prompt_idx", "count"),
    n_flipped=("sign_flipped", "sum"),
    mean_signed_effect=("signed_effect", "mean"),
    median_signed_effect=("signed_effect", "median"),
    std_signed_effect=("signed_effect", "std"),
    mean_abs_effect=("abs_effect", "mean"),
    max_abs_effect=("abs_effect", "max"),
    mean_baseline=("baseline_logit_diff", "mean"),
).reset_index()
agg["flip_rate"] = agg["n_flipped"] / agg["n_prompts"]
agg["strength"] = agg["flip_rate"] * agg["mean_abs_effect"]   # composite metric

# Add cluster metadata
def get_layers(cid):
    return str(iia_meta.loc[cid, "layers"]) if cid in iia_meta.index else "?"
def get_nfeat(cid):
    return int(iia_meta.loc[cid, "n_features"]) if cid in iia_meta.index else 0
def get_iia(cid):
    return float(iia_meta.loc[cid, "iia"]) if cid in iia_meta.index else np.nan

agg["cluster_layers"] = agg["cluster_id"].map(get_layers)
agg["cluster_n_features"] = agg["cluster_id"].map(get_nfeat)
agg["cluster_iia"] = agg["cluster_id"].map(get_iia)

# Reorder columns
agg = agg[[
    "cluster_id", "cluster_layers", "cluster_n_features", "cluster_iia",
    "cue_type", "correct_ans",
    "n_prompts", "n_flipped", "flip_rate",
    "mean_signed_effect", "median_signed_effect", "std_signed_effect",
    "mean_abs_effect", "max_abs_effect",
    "mean_baseline", "strength",
]]
agg = agg.sort_values(["cluster_id", "correct_ans", "cue_type"])

out_long = JAS / "cluster_cue_master_long.csv"
agg.to_csv(out_long, index=False)
print(f"✓ saved long-format: {out_long}")
print(f"  shape: {agg.shape}")

# ── 2-5. Wide pivots ─────────────────────────────────────────────────────────
# Combine α and β tests per (cluster, cue) for the wide pivots
agg_combined = ja.groupby(["cluster_id", "cue_type"]).agg(
    n_prompts=("prompt_idx", "count"),
    n_flipped=("sign_flipped", "sum"),
    mean_signed_effect=("signed_effect", "mean"),
    mean_abs_effect=("abs_effect", "mean"),
).reset_index()
agg_combined["flip_rate"] = agg_combined["n_flipped"] / agg_combined["n_prompts"]
agg_combined["strength"] = agg_combined["flip_rate"] * agg_combined["mean_abs_effect"]

# Wide pivots
def write_pivot(df, value_col, fname):
    piv = df.pivot(index="cluster_id", columns="cue_type", values=value_col)
    out = JAS / fname
    piv.round(4).to_csv(out)
    print(f"✓ saved wide pivot: {out}   shape: {piv.shape}")
    return piv

print()
piv_eff = write_pivot(agg_combined, "mean_signed_effect", "cluster_cue_signed_effect.csv")
piv_flp = write_pivot(agg_combined, "flip_rate", "cluster_cue_flip_rate.csv")
piv_str = write_pivot(agg_combined, "strength", "cluster_cue_strength.csv")
piv_n   = write_pivot(agg_combined, "n_prompts", "cluster_cue_n_prompts.csv")

# ── 6. Master per-prompt × per-cluster effect matrix for unions ──────────────
print()
print("Building per-prompt × per-cluster matrix for union analysis...")
effect_matrix = ja.pivot(index="prompt_idx", columns="cluster_id", values="signed_effect").fillna(0)
baseline_per_prompt = (
    ja.drop_duplicates("prompt_idx").set_index("prompt_idx")["baseline_logit_diff"]
)
prompts_meta = pd.DataFrame({
    "prompt_idx": range(len(prompts)),
    "cue_type": [cue_of[i] for i in range(len(prompts))],
    "correct_ans": [ans_of[i] for i in range(len(prompts))],
    "baseline_logit_diff": [baseline_per_prompt.get(i, np.nan) for i in range(len(prompts))],
})
out_eff = JAS / "per_prompt_cluster_effect_matrix.csv"
combined = prompts_meta.merge(
    effect_matrix.rename(columns=lambda c: f"C{int(c)}").reset_index(),
    on="prompt_idx", how="left"
)
combined.to_csv(out_eff, index=False)
print(f"✓ saved per-prompt × per-cluster effect matrix: {out_eff}")
print(f"  shape: {combined.shape}  (538 prompts × {effect_matrix.shape[1]} clusters + 4 metadata cols)")

# ── 7. Quick verification print ──────────────────────────────────────────────
print()
print("="*80)
print("PREVIEW OF CLUSTER × CUE  SIGNED EFFECT TABLE")
print("(rows = cluster, cols = cue_type, value = mean signed joint_effect)")
print("="*80)
print()
# Show only cues with strong activity
cue_order = piv_eff.abs().mean(axis=0).sort_values(ascending=False).index[:12]
print(piv_eff[cue_order].round(2).to_string())

print()
print("="*80)
print("PREVIEW OF CLUSTER × CUE  FLIP RATE TABLE")
print("="*80)
print()
print(piv_flp[cue_order].round(2).to_string())

print()
print(f"\nMain output files in {JAS}:")
print(f"  cluster_cue_master_long.csv          ← long format with all metrics")
print(f"  cluster_cue_signed_effect.csv        ← wide: rows=cluster, cols=cue, value=mean signed effect")
print(f"  cluster_cue_flip_rate.csv            ← wide: flip rate")
print(f"  cluster_cue_strength.csv             ← wide: flip_rate × mean|effect|")
print(f"  cluster_cue_n_prompts.csv            ← wide: n_prompts tested")
print(f"  per_prompt_cluster_effect_matrix.csv ← prompt-level effect matrix")
