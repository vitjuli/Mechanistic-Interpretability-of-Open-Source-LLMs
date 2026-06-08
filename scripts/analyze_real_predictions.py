"""
Honest re-analysis of error structure on REAL canonical predictions (j94 output).

No assumptions carried over from broken baseline_logit_diff CSV. Build error
structure from scratch:
  (1) Marginal distribution: where do β-prompts pile up?
  (2) Per-metadata breakdown: β-recall by each metadata cell
  (3) Cue-token contrast: does presence of nucleon/lepton tokens predict success?
  (4) Minimal-pair analysis via contrastive_pair_id if available
  (5) Aggregate verdict: is there a clean interpretable surface crutch?
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("data/analysis/runD_v2/real_predictions/real_predictions.csv")
PROMPTS = Path("data/prompts/physics_decay_type_probe_v2_train.jsonl")
OUT = Path("data/analysis/runD_v2/real_predictions/analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
print("=" * 84)
print(f"Loaded {len(df)} rows, columns ({len(df.columns)}): {list(df.columns)}")
print("=" * 84)

# Sanity check
print(f"\nOverall: accuracy = {df['correct'].mean():.3f}")
print(f"α-recall = {df[df['correct_answer']=='alpha']['correct'].mean():.3f} ({df[df['correct_answer']=='alpha']['correct'].sum()}/{(df['correct_answer']=='alpha').sum()})")
print(f"β-recall = {df[df['correct_answer']=='beta']['correct'].mean():.3f} ({df[df['correct_answer']=='beta']['correct'].sum()}/{(df['correct_answer']=='beta').sum()})")

# Group masks
is_alpha = df["correct_answer"] == "alpha"
is_beta = df["correct_answer"] == "beta"
failed_beta = is_beta & ~df["correct"]      # true β, predicted α
succ_beta = is_beta & df["correct"]          # true β, predicted β
print(f"\nfailed_β = {failed_beta.sum()}, succ_β = {succ_beta.sum()}")

# Need prompt text for cue analysis - load from jsonl
import json
prompts_jsonl = [json.loads(l) for l in open(PROMPTS)]
df["prompt_text"] = [p["prompt"] for p in prompts_jsonl]

print("\n" + "=" * 84)
print("(1) MARGIN DISTRIBUTION")
print("=" * 84)
print(f"\nα-prompts margin (α minus β; positive = correctly picks α):")
print(df[is_alpha]["margin_alpha_minus_beta"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
print(f"\nβ-prompts margin (α minus β; NEGATIVE = correctly picks β):")
print(df[is_beta]["margin_alpha_minus_beta"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
print(f"\nFailed β-prompts (model picks α when answer is β):")
print(df[failed_beta]["margin_alpha_minus_beta"].describe(percentiles=[0.05, 0.5, 0.95]).to_string())
print(f"\nSucceeded β-prompts (model correctly picks β):")
print(df[succ_beta]["margin_alpha_minus_beta"].describe(percentiles=[0.05, 0.5, 0.95]).to_string())

print("\n" + "=" * 84)
print("(2) PER-METADATA BREAKDOWN — β-recall by each metadata cell (β-prompts only)")
print("=" * 84)
META_FIELDS = ["cue_type", "relation_type", "concept_route", "level_label",
               "abstraction_level", "evidence_completeness", "prompt_format",
               "surface_family", "difficulty", "physics_concept", "keyword_type"]

beta_df = df[is_beta].copy()
for field in META_FIELDS:
    if field not in beta_df.columns:
        continue
    rec = (beta_df.groupby(field)["correct"]
           .agg(["count", "sum", "mean"])
           .rename(columns={"count": "n", "sum": "succ", "mean": "β_recall"})
           .sort_values("β_recall"))
    rec = rec[rec["n"] >= 3]  # ignore tiny cells
    if len(rec) > 0:
        print(f"\n--- β-recall by {field} (n≥3) ---")
        print(rec.to_string())

print("\n" + "=" * 84)
print("(3) CUE-TOKEN CONTRAST — does presence of nucleon/lepton tokens predict success?")
print("=" * 84)
NUCL_PAT = re.compile(r"\b(proton|neutron|nucleon|nucleus|nuclei|nuclear)\b", re.IGNORECASE)
LEPT_PAT = re.compile(r"\b(electron|positron|lepton|antineutrino|neutrino|muon|tauon)\b", re.IGNORECASE)
WEAK_PAT = re.compile(r"\b(weak\s*(force|interaction)|W\s*boson|quark|flavour|flavor)\b", re.IGNORECASE)
HELI_PAT = re.compile(r"\b(helium|alpha\s*particle|α[-\s]particle|2\s*protons?|2\s*neutrons?)\b", re.IGNORECASE)

def cue_flags(text):
    return {
        "has_nucl": bool(NUCL_PAT.search(text)),
        "has_lept": bool(LEPT_PAT.search(text)),
        "has_weak": bool(WEAK_PAT.search(text)),
        "has_heli": bool(HELI_PAT.search(text)),
    }

for k in ["has_nucl", "has_lept", "has_weak", "has_heli"]:
    df[k] = df["prompt_text"].map(lambda t: cue_flags(t)[k])

beta_df = df[is_beta].copy()
print("\nβ-recall conditional on cue-token presence (n≥5 cells only):")
for col in ["has_nucl", "has_lept", "has_weak"]:
    grp = beta_df.groupby(col)["correct"].agg(["count", "sum", "mean"])
    grp = grp[grp["count"] >= 5]
    print(f"\n{col}:")
    print(grp.to_string())

# Combined: nucl × lept
print("\nβ-recall by joint (has_nucl, has_lept):")
joint = beta_df.groupby(["has_nucl", "has_lept"])["correct"].agg(["count", "sum", "mean"])
joint = joint[joint["count"] >= 3]
print(joint.to_string())

# Same for α-prompts to check baseline (α should succeed almost everywhere)
print("\nα-recall by (has_nucl, has_lept):")
joint_a = df[is_alpha].groupby(["has_nucl", "has_lept"])["correct"].agg(["count", "sum", "mean"])
joint_a = joint_a[joint_a["count"] >= 3]
print(joint_a.to_string())

print("\n" + "=" * 84)
print("(4) CONTRASTIVE PAIRS — same physics, different framing")
print("=" * 84)
if "contrastive_pair_id" in df.columns:
    pair_counts = df["contrastive_pair_id"].value_counts()
    real_pairs = pair_counts[pair_counts >= 2].index
    real_pairs = [p for p in real_pairs if p not in ("NA", "")]
    print(f"Found {len(real_pairs)} pair IDs with ≥2 members.")
    if len(real_pairs) > 0:
        # Pairs where one fails and the other succeeds
        split_pairs = []
        for pid in real_pairs:
            sub = df[df["contrastive_pair_id"] == pid]
            if sub["correct"].nunique() == 2:
                split_pairs.append(pid)
        print(f"  Of those, {len(split_pairs)} pairs have a split outcome (one succ, one fail).")
        if split_pairs:
            print(f"  Showing first 10:")
            for pid in split_pairs[:10]:
                sub = df[df["contrastive_pair_id"] == pid][[
                    "prompt_idx", "correct_answer", "pred", "correct",
                    "contrastive_role", "margin_alpha_minus_beta"]].sort_values("contrastive_role")
                print(f"\n  pair {pid}:")
                print(sub.to_string(index=False))

print("\n" + "=" * 84)
print("(5) AGGREGATE VERDICT")
print("=" * 84)

# Strongest cell-level signals
print("\nWorst β-recall cells across all metadata fields (n≥5):")
all_worst = []
for field in META_FIELDS:
    if field not in beta_df.columns:
        continue
    grp = beta_df.groupby(field)["correct"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "β_recall"})
    grp = grp[grp["n"] >= 5]
    for val, row in grp.iterrows():
        all_worst.append({"field": field, "value": val, "n": int(row["n"]), "β_recall": float(row["β_recall"])})
all_worst_df = pd.DataFrame(all_worst).sort_values("β_recall").head(15)
print(all_worst_df.to_string(index=False))

print("\nBest β-recall cells (n≥5):")
all_best = pd.DataFrame(all_worst).sort_values("β_recall", ascending=False).head(15)
print(all_best.to_string(index=False))

# Save
df.to_csv(OUT / "real_predictions_with_cues.csv", index=False)
all_worst_df.to_csv(OUT / "worst_beta_recall_cells.csv", index=False)
pd.DataFrame(all_worst).sort_values("β_recall", ascending=False).head(15).to_csv(
    OUT / "best_beta_recall_cells.csv", index=False)

print(f"\n[saved] {OUT}/")
print("=" * 84)
