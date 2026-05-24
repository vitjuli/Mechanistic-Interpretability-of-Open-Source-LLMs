"""
Two complementary robustness checks:

ANALYSIS 1 — L21/L25 reversal: near-boundary hypothesis
  Compare |delta_orig| between flip vs ok groups at L21 and L25.
  If flip group has smaller |delta_orig| → prompts were near decision boundary;
  any perturbation direction suffices → cosine alignment irrelevant → explains reversal.

ANALYSIS 2 — L24 SFR sanity check
  Split prompts by no-op cosine at L24 (gradient_sensitivity CSV):
    "noisy"  = cosine > median  (reconstruction error more aligned with gradient)
    "clean"  = cosine ≤ median  (reconstruction error less aligned with gradient)
  Recompute SFR from Run B ablation on clean subset.
  If SFR stays high → causal result survives even after removing the noisiest prompts.

Run:
  python3 scripts/analyse_recon_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GS_CSV  = Path("data/analysis/gradient_sensitivity/physics_decay_type_probe_v2/"
               "gradient_sensitivity_physics_decay_type_probe_v2_train.csv")
ABL_CSV = Path("data/results/interventions/physics_decay_type_probe_v2/runB/"
               "intervention_ablation_physics_decay_type_probe_v2.csv")

for p in [GS_CSV, ABL_CSV]:
    if not p.exists():
        sys.exit(f"File not found: {p}")

gs  = pd.read_csv(GS_CSV)
abl = pd.read_csv(ABL_CSV)

# ── ANALYSIS 1: L21/L25 near-boundary hypothesis ──────────────────────────────
print("=" * 65)
print("ANALYSIS 1: |delta_orig| for flip vs ok — L21 and L25")
print("=" * 65)
print()
print("Hypothesis: flip group has smaller |delta_orig| → near boundary")
print()

print(f"{'L':>4}  {'group':>8}  {'n':>5}  {'mean|Δ|':>9}  "
      f"{'median|Δ|':>11}  {'<0.5 (%)':>10}")
print("-" * 55)

for layer in [21, 25]:
    g    = gs[gs["layer"] == layer].copy()
    g["abs_delta"] = g["delta_orig"].abs()

    for label, subset in [("flip", g[g["sign_flipped"] == 1]),
                           ("ok",   g[g["sign_flipped"] == 0])]:
        near = (subset["abs_delta"] < 0.5).mean() * 100
        print(f"  L{layer:>2d}  {label:>8}  {len(subset):>5}  "
              f"{subset['abs_delta'].mean():>9.4f}  "
              f"{subset['abs_delta'].median():>11.4f}  "
              f"{near:>9.1f}%")

    flip_d = g[g["sign_flipped"] == 1]["abs_delta"].values
    ok_d   = g[g["sign_flipped"] == 0]["abs_delta"].values
    t, p   = stats.ttest_ind(flip_d, ok_d, equal_var=False)
    print(f"       Welch t={t:.3f}  p={p:.4f}  "
          f"mean diff = {flip_d.mean()-ok_d.mean():+.4f}")
    print()

print("Interpretation:")
print("  p < 0.05 + smaller |delta_orig| in flip group → near-boundary hypothesis confirmed.")
print("  This explains why cosine(flip) < cosine(ok) at L21/L25:")
print("  near-boundary prompts flip regardless of error direction.")

# ── ANALYSIS 2: L24 SFR robustness on clean subset ───────────────────────────
print()
print("=" * 65)
print("ANALYSIS 2: L24 SFR on clean vs noisy prompts (Run B)")
print("=" * 65)
print()

l24_gs = gs[gs["layer"] == 24][["prompt_idx", "cosine_err_grad"]].copy()
median_cosine = l24_gs["cosine_err_grad"].median()
l24_gs["noisy"] = l24_gs["cosine_err_grad"] > median_cosine

clean_idx = set(l24_gs[~l24_gs["noisy"]]["prompt_idx"])
noisy_idx = set(l24_gs[ l24_gs["noisy"]]["prompt_idx"])

print(f"L24 cosine median: {median_cosine:.4f}")
print(f"Clean prompts (cosine ≤ median): {len(clean_idx)}")
print(f"Noisy prompts (cosine > median): {len(noisy_idx)}")
print()

# Filter ablation to L24, graph features only
abl_l24 = abl[(abl["layer"] == 24) & (abl["feature_source"] == "graph")].copy()
if abl_l24.empty:
    # fallback: no feature_source filter
    abl_l24 = abl[abl["layer"] == 24].copy()

abl_l24_clean = abl_l24[abl_l24["prompt_idx"].isin(clean_idx)]
abl_l24_noisy = abl_l24[abl_l24["prompt_idx"].isin(noisy_idx)]
abl_l24_all   = abl_l24

def sfr(df):
    return df["sign_flipped"].mean() if len(df) else float("nan")

sfr_all   = sfr(abl_l24_all)
sfr_clean = sfr(abl_l24_clean)
sfr_noisy = sfr(abl_l24_noisy)

print(f"{'Subset':<20}  {'n_rows':>8}  {'n_prompts':>10}  {'SFR':>8}")
print("-" * 52)
print(f"{'All prompts':<20}  {len(abl_l24_all):>8}  "
      f"{abl_l24_all['prompt_idx'].nunique():>10}  {sfr_all:.4f}")
print(f"{'Clean (cosine≤med)':<20}  {len(abl_l24_clean):>8}  "
      f"{abl_l24_clean['prompt_idx'].nunique():>10}  {sfr_clean:.4f}")
print(f"{'Noisy (cosine>med)':<20}  {len(abl_l24_noisy):>8}  "
      f"{abl_l24_noisy['prompt_idx'].nunique():>10}  {sfr_noisy:.4f}")

print()

# Also per correct_answer (alpha vs beta) within clean
print("Clean subset SFR by answer type (alpha vs beta):")
gs_prompt_ans = gs[gs["layer"] == 24][["prompt_idx", "correct_answer"]].drop_duplicates()
abl_l24_clean = abl_l24_clean.merge(gs_prompt_ans, on="prompt_idx", how="left")
for ans, grp in abl_l24_clean.groupby("correct_answer"):
    print(f"  {ans:<10}  n_rows={len(grp):>6}  SFR={sfr(grp):.4f}")

print()

# ── Per-prompt join: ablation flip × no-op flip ───────────────────────────────
# no-op sign_flipped in gs = enc→dec reconstruction flip (same op as script 51)
# Join on (prompt_idx, layer=24) to compare per (prompt, feature) pair.
l24_noop = (gs[gs["layer"] == 24][["prompt_idx", "cosine_err_grad", "sign_flipped"]]
            .rename(columns={"sign_flipped": "noop_flipped"})
            .copy())
l24_noop["noisy"] = l24_noop["cosine_err_grad"] > median_cosine

abl_l24_joined = abl_l24.merge(l24_noop, on="prompt_idx", how="inner")

def flip_breakdown(df, label):
    abl  = df["sign_flipped"].astype(bool)
    noop = df["noop_flipped"].astype(bool)
    n = len(df)
    abl_only  = (abl  & ~noop).sum()   # genuinely causal
    both      = (abl  &  noop).sum()   # ambiguous
    noop_only = (~abl &  noop).sum()   # noise but not causal
    neither   = (~abl & ~noop).sum()
    print(f"  {label:<22}  n={n:>5}  "
          f"abl_only={abl_only/n:.3f}  both={both/n:.3f}  "
          f"noop_only={noop_only/n:.3f}  neither={neither/n:.3f}")
    return abl_only / n

print("Per-prompt flip breakdown: abl_only = genuinely causal (ablation flips, no-op does not)")
print(f"{'Subset':<24}  {'n':>6}  {'abl_only':>9}  {'both':>7}  "
      f"{'noop_only':>10}  {'neither':>8}")
print("-" * 70)
rate_all   = flip_breakdown(abl_l24_joined,                             "All prompts")
rate_clean = flip_breakdown(abl_l24_joined[~abl_l24_joined["noisy"]],  "Clean (cosine≤med)")
rate_noisy = flip_breakdown(abl_l24_joined[ abl_l24_joined["noisy"]],  "Noisy (cosine>med)")

print()
print("Interpretation:")
if rate_clean > 0.03:
    print(f"  ✓ Genuine causal SFR (abl_only) = {rate_clean:.3f} on clean subset.")
    print("    L24 causal result survives: ablation flips prompts that reconstruction alone does not.")
elif rate_clean > 0.01:
    print(f"  ~ Genuine causal SFR = {rate_clean:.3f} — modest but present on clean subset.")
else:
    print(f"  ✗ Genuine causal SFR = {rate_clean:.3f} — L24 effect is driven by reconstruction noise.")

# Also break down by answer type on clean subset
print()
clean_joined = abl_l24_joined[~abl_l24_joined["noisy"]].copy()
gs_ans = gs[gs["layer"] == 24][["prompt_idx", "correct_answer"]].drop_duplicates()
clean_joined = clean_joined.merge(gs_ans, on="prompt_idx", how="left")
print("Genuine causal SFR (abl_only) on clean subset by answer type:")
for ans, grp in clean_joined.groupby("correct_answer"):
    abl_only_rate = ((grp["sign_flipped"].astype(bool)) &
                     (~grp["noop_flipped"].astype(bool))).mean()
    print(f"  {ans:<10}  n={len(grp):>5}  abl_only={abl_only_rate:.4f}")
