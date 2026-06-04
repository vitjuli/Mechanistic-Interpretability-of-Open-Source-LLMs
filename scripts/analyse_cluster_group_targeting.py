"""
Manual analysis: for each cluster, which cue_type groups does it preferentially flip?
Can specific UNIONS of clusters cover specific groups that no single cluster does?

5 clusters with per-pair data: 16, 18, 19, 20 (L24 sub-detectors), 35 (L18 monolithic).

For each flipped pair:
  - direction a_to_b: target prompt = pb  (β-prompt pushed toward α)
  - direction b_to_a: target prompt = pa  (α-prompt pushed toward β)
Get cue_type of the target prompt → which group did this cluster "successfully manipulate"?

Then for each cluster (and union of clusters): which cue_types are over-represented
in the flipped set, vs the natural base rate of cue_types in tested pairs?
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path("/Users/julia/Desktop/courses/thesis/project")
SUB = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp"
PRMS = ROOT / "data/prompts/physics_decay_type_probe_v2_train.jsonl"

# ── Load ─────────────────────────────────────────────────────────────────────
flips = pd.read_csv(SUB / "iia_per_pair_flips.csv")
with open(PRMS) as f:
    prompts = [json.loads(l) for l in f]

cue_of = {i: (p.get("cue_type") or "AUX") for i, p in enumerate(prompts)}
ans_of = {i: p.get("correct_answer", "").strip() for i, p in enumerate(prompts)}

# ── Compute per-flip target prompt + its cue ─────────────────────────────────
# CORRECT convention (verified against baseline_margin signs):
#   a_to_b:  push α→β, target = pa_idx (α-prompt being pushed to β-side)
#   b_to_a:  push β→α, target = pb_idx (β-prompt being pushed to α-side)
def target_idx(row):
    return row["pa_idx"] if row["direction"] == "a_to_b" else row["pb_idx"]

flips["target_idx"]  = flips.apply(target_idx, axis=1)
flips["target_cue"]  = flips["target_idx"].map(cue_of)
flips["target_ans"]  = flips["target_idx"].map(ans_of)

# ── Per-cluster: which cues does it flip + base rate of cues among TESTED ────
clusters = sorted(flips["target"].unique())
print("="*80)
print("PER-CLUSTER: cue distribution of FLIPPED targets vs TESTED base rate")
print("="*80)

per_cluster_flipped_cues = {}
per_cluster_tested_cues = {}
cluster_size_by_id = {16: 2, 18: 10, 19: 3, 20: 5, 35: 17}

for cid in clusters:
    sub = flips[flips["target"] == cid]
    tested = Counter(sub["target_cue"])
    flipped = Counter(sub[sub["flipped"] == 1]["target_cue"])

    per_cluster_tested_cues[cid] = tested
    per_cluster_flipped_cues[cid] = flipped

    print(f"\nCluster {cid}  (n_features={cluster_size_by_id.get(cid, '?')}, "
          f"flips={sum(flipped.values())}/{sum(tested.values())} pairs)")
    print(f"{'cue_type':<35} {'flipped':>9} {'tested':>9} {'flip rate':>10} {'baseline':>10}")
    print("-"*80)
    total_t = sum(tested.values())
    total_f = sum(flipped.values())
    base_rate = total_f / max(total_t, 1)
    # sort by flip rate
    cue_rates = []
    for cue, ft in flipped.items():
        nt = tested[cue]
        if nt < 5:  # ignore low-n cues
            continue
        rate = ft / nt
        cue_rates.append((cue, ft, nt, rate, base_rate))
    cue_rates.sort(key=lambda x: -x[3])
    for cue, ft, nt, rate, base in cue_rates[:8]:
        marker = "★" if rate > base * 1.5 else ("•" if rate > base else " ")
        print(f"{marker} {cue:<33} {ft:>9} {nt:>9} {rate:>10.3f} {base:>10.3f}")

# ── UNION analysis: which cues are flipped by UNION but not by any single ────
print("\n" + "="*80)
print("UNION ANALYSIS: which target prompts (and their cues) become flipped")
print("when COMBINING clusters, that no single cluster flipped?")
print("="*80)

# Set of flipped target_idx per cluster
flipped_targets_per_cluster = {
    cid: set(flips[(flips["target"] == cid) & (flips["flipped"] == 1)]["target_idx"])
    for cid in clusters
}

print("\n(For unions: 'gained' = prompts in union not flipped by any individual cluster — N/A,")
print(" since union flips can only equal union-of-individual flips in this offline view.")
print(" But we CAN look at the OVERLAP: which prompts are flipped by ALL members of a group?)")

# Overlap structure
print("\nCommon-prompts overlap matrix (size of intersection of flipped sets):")
print(f"{'cluster':<10}", *[f"C{c:>4}" for c in clusters])
for c1 in clusters:
    row = [f"C{c1:<8}"]
    for c2 in clusters:
        inter = len(flipped_targets_per_cluster[c1] & flipped_targets_per_cluster[c2])
        row.append(f"{inter:>5}")
    print(*row)

# Per-cluster unique prompts (flipped by THIS cluster only)
print(f"\nPer-cluster UNIQUE flipped prompts (and their cues):")
all_other_flipped = {cid: set().union(*(flipped_targets_per_cluster[c] for c in clusters if c != cid))
                    for cid in clusters}
for cid in clusters:
    unique = flipped_targets_per_cluster[cid] - all_other_flipped[cid]
    cues = Counter(cue_of[i] for i in unique)
    print(f"\nCluster {cid}: {len(unique)} unique prompts only it flips")
    for cue, n in cues.most_common(8):
        ex = next((p["prompt"][:75] for i, p in enumerate(prompts) if i in unique and (p.get("cue_type") or "AUX") == cue), "")
        print(f"  {cue:<35} n={n:>2}  ex: {ex}")

# ── Union coverage by cue ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("UNION COVERAGE: for each cue group, which clusters flip it (combined)?")
print("="*80)
print(f"{'cue_type':<35} {'flipped by clusters (union)':<35} {'n_union':>7}")
print("-"*80)

# For each cue, build the set of clusters that flip ≥1 prompt of that cue
cue_to_clusters = defaultdict(set)
cue_to_total_flips = defaultdict(int)
cue_to_total_tested = defaultdict(int)

for cid in clusters:
    sub = flips[(flips["target"] == cid)]
    for _, r in sub.iterrows():
        cue = r["target_cue"]
        cue_to_total_tested[cue] += 1
        if r["flipped"] == 1:
            cue_to_clusters[cue].add(cid)
            cue_to_total_flips[cue] += 1

rows = []
for cue, cids in sorted(cue_to_clusters.items(), key=lambda kv: -len(kv[1])):
    cstr = "+".join(str(c) for c in sorted(cids))
    n_flipped = sum(1 for cid in cids for _ in flips[(flips["target"]==cid)&(flips["flipped"]==1)&(flips["target_cue"]==cue)].iterrows())
    rows.append((cue, cstr, len(cids), n_flipped, cue_to_total_tested[cue]))

for cue, cstr, k, nf, nt in rows[:25]:
    print(f"{cue:<35} {cstr:<35} {k:>7}/{len(clusters)}   ({nf} flips in {nt} tests)")

# Cues NOT flipped by anyone:
not_flipped = [c for c in cue_to_total_tested if cue_to_total_flips.get(c, 0) == 0]
print(f"\nCues NEVER flipped by any of 5 clusters: {len(not_flipped)}")
for c in not_flipped[:10]:
    print(f"  {c:<35} (n_tested={cue_to_total_tested[c]})")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
n_cues_total = len(cue_to_total_tested)
n_cues_some = len(cue_to_clusters)
n_cues_none = len(not_flipped)
print(f"Total cue groups represented in tested pairs: {n_cues_total}")
print(f"Cues with ≥1 flip by some cluster: {n_cues_some}")
print(f"Cues with 0 flips by any cluster: {n_cues_none}")
print()

# Save
out_dir = ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp"
summary = {
    "per_cluster_flipped_cue_counts": {str(c): {str(k): int(v) for k, v in per_cluster_flipped_cues[c].items()} for c in clusters},
    "per_cluster_tested_cue_counts": {str(c): {str(k): int(v) for k, v in per_cluster_tested_cues[c].items()} for c in clusters},
    "cue_to_clusters_that_flip_it": {str(c): sorted(int(x) for x in cids) for c, cids in cue_to_clusters.items()},
    "cues_never_flipped": sorted(not_flipped),
}
out_path = out_dir / "cluster_group_targeting_analysis.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"saved analysis to: {out_path}")
