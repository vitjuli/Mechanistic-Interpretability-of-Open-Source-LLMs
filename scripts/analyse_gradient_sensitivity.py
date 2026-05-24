"""
Significance test for cosine(flip) - cosine(ok) per layer.

Tests whether flipped prompts have higher cosine alignment between
reconstruction error and logit-diff gradient than non-flipped prompts.

Uses both Welch t-test and bootstrap CI (5000 resamples) since n_flip
is small for some layers. Bonferroni correction applied for 16 tests.

Run after syncing:
  rsync ... data/analysis/gradient_sensitivity/
  python3 scripts/analyse_gradient_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path("data/analysis/gradient_sensitivity/physics_decay_type_probe_v2/"
           "gradient_sensitivity_physics_decay_type_probe_v2_train.csv")

if not CSV.exists():
    sys.exit(f"CSV not found: {CSV}\nRun rsync first.")

df = pd.read_csv(CSV)
layers = sorted(df["layer"].unique())
N_LAYERS = len(layers)
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_LAYERS
N_BOOT = 5000
rng = np.random.default_rng(42)

print("=== cosine(flip) − cosine(ok): significance per layer ===\n")
print(f"Bonferroni threshold: p < {ALPHA_BONF:.4f}  (α={ALPHA}, {N_LAYERS} tests)\n")
print(f"{'L':>4}  {'n_flip':>7}  {'n_ok':>6}  "
      f"{'cos(flip)':>10}  {'cos(ok)':>9}  {'Δcosine':>9}  "
      f"{'t-stat':>8}  {'p(t)':>8}  {'boot_CI_95':>22}  {'sig':>5}")
print("-" * 103)

results = []
for layer in layers:
    g     = df[df["layer"] == layer]
    flip  = g[g["sign_flipped"] == 1]["cosine_err_grad"].values
    ok    = g[g["sign_flipped"] == 0]["cosine_err_grad"].values

    if len(flip) < 2 or len(ok) < 2:
        print(f"  L{layer:>2d}  {len(flip):>7}  {len(ok):>6}  — skipped (too few flips)")
        continue

    mean_flip = flip.mean()
    mean_ok   = ok.mean()
    delta     = mean_flip - mean_ok

    # Welch t-test (unequal variance, unequal n)
    t_stat, p_val = stats.ttest_ind(flip, ok, equal_var=False)

    # Bootstrap CI on delta = mean(flip) - mean(ok)
    boot_deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s_flip = rng.choice(flip, size=len(flip), replace=True)
        s_ok   = rng.choice(ok,   size=len(ok),   replace=True)
        boot_deltas[i] = s_flip.mean() - s_ok.mean()
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

    sig_bonf = p_val < ALPHA_BONF
    sig_nom  = p_val < ALPHA
    sig_str  = "**" if sig_bonf else ("*" if sig_nom else "ns")

    print(f"  L{layer:>2d}  {len(flip):>7}  {len(ok):>6}  "
          f"{mean_flip:>10.4f}  {mean_ok:>9.4f}  {delta:>+9.4f}  "
          f"{t_stat:>8.3f}  {p_val:>8.4f}  "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig_str:>5}")

    results.append(dict(layer=layer, n_flip=len(flip), n_ok=len(ok),
                        mean_flip=mean_flip, mean_ok=mean_ok, delta=delta,
                        t_stat=t_stat, p_val=p_val,
                        ci_lo=ci_lo, ci_hi=ci_hi,
                        sig_bonf=sig_bonf, sig_nom=sig_nom))

print()
print("** = Bonferroni-significant  * = nominally significant  ns = not significant")

rdf = pd.DataFrame(results)

print("\n=== SUMMARY ===\n")

sig_b = rdf[rdf["sig_bonf"]]
sig_n = rdf[rdf["sig_nom"] & ~rdf["sig_bonf"]]
ns    = rdf[~rdf["sig_nom"]]

print(f"Bonferroni-significant layers ({len(sig_b)}):")
for _, r in sig_b.sort_values("p_val").iterrows():
    print(f"  L{int(r.layer):>2d}  Δ={r.delta:+.4f}  p={r.p_val:.4f}  "
          f"CI=[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]")

print(f"\nNominally significant only ({len(sig_n)}):")
for _, r in sig_n.sort_values("p_val").iterrows():
    print(f"  L{int(r.layer):>2d}  Δ={r.delta:+.4f}  p={r.p_val:.4f}  "
          f"CI=[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]")

print(f"\nNot significant ({len(ns)}) — Δcosine is noise:")
for _, r in ns.sort_values("delta").iterrows():
    print(f"  L{int(r.layer):>2d}  Δ={r.delta:+.4f}  p={r.p_val:.4f}  "
          f"CI=[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]")

print("\n=== INTERPRETATION ===\n")
l18 = rdf[rdf["layer"] == 18]
l24 = rdf[rdf["layer"] == 24]
l21 = rdf[rdf["layer"] == 21]

for layer, label in [(18, "L18 (adversarial candidate)"),
                     (24, "L24 (magnitude candidate)"),
                     (21, "L21 (reversed sign)")]:
    row = rdf[rdf["layer"] == layer]
    if row.empty:
        continue
    r = row.iloc[0]
    verdict = ("Bonferroni-sig" if r.sig_bonf
               else "nominally sig" if r.sig_nom
               else "NOT significant — noise")
    print(f"  {label}: Δ={r.delta:+.4f}  p={r.p_val:.4f}  → {verdict}")
    zero_in_ci = r.ci_lo <= 0 <= r.ci_hi
    print(f"    95% CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  "
          f"(zero {'IN' if zero_in_ci else 'OUTSIDE'} CI)")
