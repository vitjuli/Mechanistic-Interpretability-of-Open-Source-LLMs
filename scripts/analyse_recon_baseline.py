"""
Sanity-check analysis for recon_baseline CSV (script 51 output).

Tests the hypothesis that L18 sign flips are caused by "adversarially directed"
reconstruction error rather than large error — i.e. small errors in sensitive
subspace directions cause flips, not overall reconstruction magnitude.

Run locally after syncing:
  rsync iv294@login.hpc.cam.ac.uk:.../data/analysis/recon_baseline/ data/analysis/recon_baseline/
  python scripts/analyse_recon_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("data/analysis/recon_baseline/physics_decay_type_probe_v2/"
           "recon_baseline_physics_decay_type_probe_v2_train.csv")

if not CSV.exists():
    sys.exit(f"CSV not found: {CSV}\nRun rsync first.")

df = pd.read_csv(CSV)
layers = sorted(df["layer"].unique())

print("=== RECON ERROR: flipped vs non-flipped (norm analysis) ===\n")
print(f"{'L':>4}  {'n_flip':>7}  {'flip%':>6}  "
      f"{'err(flip)':>10}  {'err(no flip)':>13}  {'Δerr':>8}  "
      f"{'corr(err,flip)':>15}  {'err_med_flip':>13}")
print("-" * 90)

rows = []
for layer in layers:
    g = df[df["layer"] == layer]
    n_flip   = g["sign_flipped"].sum()
    flip_pct = g["sign_flipped"].mean() * 100

    flipped  = g[g["sign_flipped"] == 1]["recon_err_norm"]
    ok       = g[g["sign_flipped"] == 0]["recon_err_norm"]

    mean_flip = flipped.mean() if len(flipped) else float("nan")
    mean_ok   = ok.mean()      if len(ok)      else float("nan")
    delta_err = mean_flip - mean_ok

    corr = g["recon_err_norm"].corr(g["sign_flipped"])

    med_flip = flipped.median() if len(flipped) else float("nan")

    print(f"  L{layer:>2d}  {n_flip:>7}  {flip_pct:>5.1f}%  "
          f"{mean_flip:>10.4f}  {mean_ok:>13.4f}  {delta_err:>+8.4f}  "
          f"{corr:>15.4f}  {med_flip:>13.4f}")

    rows.append(dict(layer=layer, n_flip=int(n_flip), flip_pct=flip_pct,
                     mean_err_flip=mean_flip, mean_err_ok=mean_ok,
                     delta_err=delta_err, corr_err_flip=corr, med_err_flip=med_flip))

print()

# ── Highlight anomalies ───────────────────────────────────────────────────────
rdf = pd.DataFrame(rows)
mean_delta = rdf["delta_err"].mean()
mean_corr  = rdf["corr_err_flip"].mean()

print("=== ANOMALY CHECK ===\n")
print(f"Mean Δerr across layers : {mean_delta:+.4f}")
print(f"Mean corr(err, flip)    : {mean_corr:.4f}\n")

print("Layers where Δerr is below mean (flip errors are NOT bigger than usual):")
low_delta = rdf[rdf["delta_err"] < mean_delta].sort_values("delta_err")
for _, r in low_delta.iterrows():
    flag = "  ← L18 hypothesis" if r["layer"] == 18 else ""
    print(f"  L{int(r['layer']):>2d}  Δerr={r['delta_err']:+.4f}  "
          f"corr={r['corr_err_flip']:.4f}  flip%={r['flip_pct']:.1f}%{flag}")

print()
print("Layers where corr(err, flip) is below mean (error magnitude poorly predicts flip):")
low_corr = rdf[rdf["corr_err_flip"] < mean_corr].sort_values("corr_err_flip")
for _, r in low_corr.iterrows():
    flag = "  ← L18 hypothesis" if r["layer"] == 18 else ""
    print(f"  L{int(r['layer']):>2d}  corr={r['corr_err_flip']:.4f}  "
          f"Δerr={r['delta_err']:+.4f}  flip%={r['flip_pct']:.1f}%{flag}")

print()
print("=== INTERPRETATION ===")
l18 = rdf[rdf["layer"] == 18].iloc[0]
l24 = rdf[rdf["layer"] == 24].iloc[0]
l11 = rdf[rdf["layer"] == 11].iloc[0]

print(f"\nL11 (clean):  Δerr={l11['delta_err']:+.4f}  corr={l11['corr_err_flip']:.4f}")
print(f"L18 (suspect):Δerr={l18['delta_err']:+.4f}  corr={l18['corr_err_flip']:.4f}")
print(f"L24 (noisy):  Δerr={l24['delta_err']:+.4f}  corr={l24['corr_err_flip']:.4f}")

print()
l18_vs_l11_delta = l18["delta_err"] < l11["delta_err"]
l18_vs_l11_corr  = l18["corr_err_flip"] < l11["corr_err_flip"]

if l18_vs_l11_delta and l18_vs_l11_corr:
    print("CONFIRMED: L18 flips occur at smaller errors and error magnitude is a\n"
          "  weaker predictor of flips than L11. Consistent with adversarially\n"
          "  directed error in a task-sensitive subspace.")
elif l18_vs_l11_delta or l18_vs_l11_corr:
    print("PARTIAL: One of the two indicators supports the hypothesis.")
else:
    print("NOT CONFIRMED: L18 behaves similarly to clean layers on these metrics.")
