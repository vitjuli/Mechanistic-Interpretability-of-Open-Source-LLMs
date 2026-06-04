"""
For ALL 30 sub-clusters (Louvain subgroup decomposition at k=30):
  exhaustively enumerate combinations of size 2-5
  use per-prompt joint_ablation data to approximate combined effect
    (assumption: additive — combined_margin ≈ baseline + Σ_c joint_effect[c])
  rank by overall flip rate, selectivity, and per-cue coverage
  save top combinations and a per-cue breakdown for the best ones

Caveat: additivity is an approximation. The L18↔L24 antagonism shows joint patching
can have less effect than the sum. So this OVER-ESTIMATES large-union flip rates.
For RANKING purposes it still gives a useful order of candidates.
"""
import json
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path("/Users/julia/Desktop/courses/thesis/project")
JAS = ROOT / "data/analysis/runD_v2/cluster_joint_ablation_subgroup"
SUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp"
PRMS = ROOT / "data/prompts/physics_decay_type_probe_v2_train.jsonl"

prompts = [json.loads(l) for l in open(PRMS)]
cue_of = [p.get("cue_type") or "AUX" for p in prompts]
ans_of = [p.get("correct_answer", "").strip() for p in prompts]

# ── Load joint ablation per-prompt × per-cluster data ────────────────────────
ja = pd.read_csv(JAS / "joint_ablation_physics_decay_type_probe_v2_train.csv")
print(f"Loaded joint_ablation: {len(ja)} rows, "
      f"{ja['cluster_id'].nunique()} clusters × {ja['prompt_idx'].nunique()} prompts")

# Pivot: row=prompt, col=cluster, value=joint_effect (= joint_logit_diff - baseline)
ja["joint_effect"] = ja["joint_logit_diff"] - ja["baseline_logit_diff"]
effect = ja.pivot(index="prompt_idx", columns="cluster_id", values="joint_effect").fillna(0)
baseline = (
    ja.drop_duplicates("prompt_idx").set_index("prompt_idx")["baseline_logit_diff"]
      .reindex(effect.index)
)
print(f"Effect matrix: {effect.shape}  (rows=prompts, cols=clusters)")

cluster_ids = sorted(effect.columns.tolist())
n_clusters = len(cluster_ids)
print(f"All sub-cluster IDs: {cluster_ids}")
print()

# Load per-cluster metadata (n_features, layer, IIA)
ja_meta = ja.drop_duplicates("cluster_id")[["cluster_id", "n_cluster_features"]].set_index("cluster_id")
iia = pd.read_csv(SUB / "iia_probe_clusters.csv")
iia_meta = iia.set_index("cluster")[["iia", "layers", "n_features"]]

# Index of α vs β prompts (using metadata)
prompts_df = pd.DataFrame({
    "idx": range(len(prompts)),
    "cue": cue_of, "ans": ans_of,
})
is_alpha = (prompts_df["ans"].values == "alpha")
is_beta = ~is_alpha

# Convert to fast numpy arrays
E = effect[cluster_ids].values    # (n_prompts, n_clusters)
B = baseline.values                # (n_prompts,)
sign_B = np.sign(B)

print(f"Class balance: α={int(is_alpha.sum())}  β={int(is_beta.sum())}")
print()

# ── Single-cluster baseline: which clusters alone give meaningful flips ──────
print("="*80)
print("SINGLE-CLUSTER flip rates (alone, sanity check on additive assumption)")
print("="*80)
single_rows = []
for ci in range(n_clusters):
    cid = cluster_ids[ci]
    new_B = B + E[:, ci]
    flip = (np.sign(new_B) != sign_B) & (sign_B != 0)
    sfr = float(flip.mean())
    sfr_a = float(flip[is_alpha].mean())
    sfr_b = float(flip[is_beta].mean())
    single_rows.append({
        "id": cid,
        "n_features": int(ja_meta.loc[cid, "n_cluster_features"]),
        "iia": float(iia_meta.loc[cid, "iia"]) if cid in iia_meta.index else np.nan,
        "layers": str(iia_meta.loc[cid, "layers"]) if cid in iia_meta.index else "?",
        "additive_sfr": sfr,
        "sfr_alpha": sfr_a,
        "sfr_beta": sfr_b,
        "asymmetry": sfr_b - sfr_a,
    })
single_df = pd.DataFrame(single_rows).sort_values("additive_sfr", ascending=False)
print(single_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()
single_df.to_csv(JAS / "single_cluster_additive_sfr.csv", index=False)

# ── Combinations 2..5 ────────────────────────────────────────────────────────
print("="*80)
print("EXHAUSTIVE COMBINATIONS (size 2 to 5)")
print("="*80)
print(f"  C(30,2) = {len(list(combinations(range(n_clusters), 2)))}  "
      f"C(30,3) = {len(list(combinations(range(n_clusters), 3)))}  "
      f"C(30,4) = {len(list(combinations(range(n_clusters), 4)))}  "
      f"C(30,5) = {len(list(combinations(range(n_clusters), 5)))}")

all_results = []
for k in [2, 3, 4, 5]:
    print(f"\nSize {k}: enumerating {sum(1 for _ in combinations(range(n_clusters), k))} combos...")
    for combo in combinations(range(n_clusters), k):
        sum_eff = E[:, list(combo)].sum(axis=1)
        new_B = B + sum_eff
        flip = (np.sign(new_B) != sign_B) & (sign_B != 0)
        sfr = float(flip.mean())
        sfr_a = float(flip[is_alpha].mean())
        sfr_b = float(flip[is_beta].mean())
        if sfr < 0.05:  # skip very weak combos to save memory
            continue
        all_results.append({
            "k": k,
            "ids": "+".join(str(cluster_ids[i]) for i in combo),
            "additive_sfr": sfr,
            "sfr_alpha": sfr_a,
            "sfr_beta": sfr_b,
            "selectivity": sfr_b - sfr_a,
            "n_features_total": sum(int(ja_meta.loc[cluster_ids[i], "n_cluster_features"]) for i in combo),
        })

results_df = pd.DataFrame(all_results)
print(f"\nTotal candidate combinations with sfr ≥ 0.05: {len(results_df)}")

# Save full
results_df.to_csv(JAS / "all_combinations_additive_sfr.csv", index=False)

# ── Top combinations ────────────────────────────────────────────────────────
print()
print("="*80)
print("TOP 30 combinations by additive SFR (overall)")
print("="*80)
top = results_df.sort_values("additive_sfr", ascending=False).head(30)
print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print()
print("="*80)
print("TOP 20 combinations by selectivity (β→α flip prefers β-prompts)")
print("="*80)
top_sel = results_df.sort_values("selectivity", ascending=False).head(20)
print(top_sel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── Per-cue breakdown for top-5 combinations ─────────────────────────────────
print()
print("="*80)
print("PER-CUE breakdown for top-5 combinations by overall SFR")
print("="*80)
top5 = results_df.sort_values("additive_sfr", ascending=False).head(5)
for _, row in top5.iterrows():
    ids = [int(x) for x in row["ids"].split("+")]
    cluster_cols = [cluster_ids.index(i) for i in ids]
    sum_eff = E[:, cluster_cols].sum(axis=1)
    new_B = B + sum_eff
    flip = (np.sign(new_B) != sign_B) & (sign_B != 0)
    print(f"\nCombination: {row['ids']}  (k={row['k']}, additive sfr={row['additive_sfr']:.3f})")
    cue_counts_flipped = Counter()
    cue_counts_total = Counter()
    for i, c in enumerate(cue_of):
        cue_counts_total[c] += 1
        if flip[i]: cue_counts_flipped[c] += 1
    rows = sorted(
        [(c, cue_counts_flipped[c], cue_counts_total[c]) for c in cue_counts_flipped],
        key=lambda x: -x[1]/max(x[2],1)
    )
    print(f"{'cue_type':<35} {'flipped':>8} {'total':>6} {'rate':>6}")
    for c, f, t in rows[:12]:
        if t < 3: continue
        print(f"  {c:<33} {f:>8} {t:>6} {f/t:>6.2f}")

# ── Summary statistics ──────────────────────────────────────────────────────
print()
print("="*80)
print("BEST COMBINATIONS BY SIZE")
print("="*80)
for k in [2, 3, 4, 5]:
    sub = results_df[results_df["k"] == k]
    if len(sub) == 0:
        print(f"k={k}: no combinations above sfr=0.05")
        continue
    best = sub.sort_values("additive_sfr", ascending=False).head(3)
    print(f"\nTop 3 of size {k}:")
    for _, r in best.iterrows():
        print(f"  {r['ids']:<25} sfr={r['additive_sfr']:.3f}  "
              f"α={r['sfr_alpha']:.3f}  β={r['sfr_beta']:.3f}  "
              f"n_feats={r['n_features_total']}")

# Saved
print()
print(f"saved → {JAS / 'all_combinations_additive_sfr.csv'}")
print(f"saved → {JAS / 'single_cluster_additive_sfr.csv'}")
