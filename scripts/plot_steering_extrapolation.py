"""
How hard would we need to push h(p) along w_res to actually flip the answer?

Two complementary methods (no model needed locally):

METHOD 1 — analytical (linear unembed approximation):
  logit_diff(α-β) = <h_final, gbar>
  After steering: <h + c·σ·w_res, gbar> = <h, gbar> + c·σ·<w_res, gbar>
  To flip: c_flip = -<h, gbar> / (σ·<w_res, gbar>)

METHOD 2 — empirical extrapolation from j65 CSV:
  We have dl_steered at c ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 2.0}.
  Fit per-prompt linear slope dl_steered = dl_clean + c·slope,
  extrapolate to dl_steered = 0 → c_flip.

Both should agree (or differ by LayerNorm rescaling).
"""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, pandas as pd

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
h = np.load(ROOT / "h_residual_per_depth.npz")
y, is_train = h["y"], h["is_train"]
cd = np.load(ROOT / "concept_directions.npz")
gbar = cd["gbar"].astype(np.float64)

# Final-layer activations + probe direction
H_final = h["final"].astype(np.float64)
w = np.load(ROOT / "w_res_final.npy").astype(np.float64)

# σ = stdev of projection along w_res on data
projections = H_final @ w
sigma = projections.std()

# baseline logit diff (signed): positive = α preferred, negative = β preferred
# CONVENTION: gbar = W_U[alpha] - W_U[beta], so <h, gbar> > 0 → α
dl_clean = H_final @ gbar          # (538,)

# Euclidean inner product between w_res and gbar (NOT causal cosine)
wg = float(w @ gbar)
print(f"⟨w_res, gbar⟩ (Euclidean) = {wg:.4f}")
print(f"σ (stdev projections)     = {sigma:.3f}")
print(f"σ × ⟨w,gbar⟩              = {sigma*wg:.4f}   ← shift per unit c")

# ── METHOD 1: analytical extrapolation ────────────────────────────────────────
# For each prompt: c_flip = -dl_clean / (σ × <w,gbar>)
c_flip_analytical = -dl_clean / (sigma * wg)

# ── METHOD 2: empirical (from j65 CSV) ────────────────────────────────────────
curve = pd.read_csv(ROOT / "axis_causality_curve.csv")
final = curve[(curve.tap == "final") & (curve.dir == "w_res")].copy()
# delta = how much dl shifted per c
final["delta"] = final["dl_steered"] - final["dl_clean"]

# group by (toward, c): aggregate slope across prompts
slopes_empirical = final.groupby("c")["delta"].mean()
slopes_per_c = slopes_empirical / slopes_empirical.index  # delta / c → slope estimate
slope_avg = float(slopes_per_c.mean())
print(f"\nEmpirical slope (avg Δ/c from CSV) = {slope_avg:.4f}")
print(f"Analytical slope σ×⟨w,gbar⟩         = {sigma*wg:.4f}")
print(f"  → ratio (empirical/analytical) = {slope_avg/(sigma*wg):.3f}")

# Per-prompt empirical c_flip: extrapolate dl_clean / slope
c_flip_empirical = -final.groupby("dl_clean")["delta"].apply(
    lambda s: s.iloc[-1] / 2.0   # take c=2 as anchor, slope = delta/2
).reset_index()
# Simpler: c_flip ≈ -dl_clean / slope_avg
# but slope is signed per-direction; use abs
# Per-prompt: dl_clean is constant across c values for the same prompt.
# Take unique dl_clean values (one per prompt)
dl_clean_unique = final["dl_clean"].drop_duplicates().values
c_flip_emp_unique = -dl_clean_unique / slope_avg

# For panel A: also use those same unique values to be apples-to-apples
# (analytical version uses full 538, but distribution shape is what matters)

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A: distribution of c_flip across prompts (analytical)
ax = axes[0]
c_flip_a = c_flip_analytical[is_train]   # use train+heldout, all
# split by class
m_a = (y == 0); m_b = (y == 1)

bins = np.linspace(-200, 200, 50)
ax.hist(c_flip_analytical[m_a], bins=bins, alpha=0.6, color="tab:blue",
        label=f"α-prompts (median c_flip = {np.median(c_flip_analytical[m_a]):+.1f})",
        edgecolor="black")
ax.hist(c_flip_analytical[m_b], bins=bins, alpha=0.6, color="tab:orange",
        label=f"β-prompts (median c_flip = {np.median(c_flip_analytical[m_b]):+.1f})",
        edgecolor="black")
ax.axvline(2, color="red", linestyle="--", lw=1.5, label="j65 max |c|=2")
ax.axvline(-2, color="red", linestyle="--", lw=1.5)
ax.set_xlabel("required |c| to flip the answer  (analytical: -dl/(σ·⟨w,gbar⟩))")
ax.set_ylabel("count of prompts")
ax.set_title("A — Required steering intensity (analytical)\n"
             f"⟨w,gbar⟩={wg:.3f} (tiny!) → typical c_flip ≈ {np.median(np.abs(c_flip_analytical)):.0f}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# B: empirical extrapolation
ax = axes[1]
bins = np.linspace(-200, 200, 50)
# Split unique values by sign of dl_clean (>0 = α-prompt, <0 = β-prompt)
m_a_emp = dl_clean_unique > 0
m_b_emp = dl_clean_unique < 0
ax.hist(c_flip_emp_unique[m_a_emp], bins=bins, alpha=0.6, color="tab:blue",
        label=f"α-prompts (median = {np.median(c_flip_emp_unique[m_a_emp]):+.1f})",
        edgecolor="black")
ax.hist(c_flip_emp_unique[m_b_emp], bins=bins, alpha=0.6, color="tab:orange",
        label=f"β-prompts (median = {np.median(c_flip_emp_unique[m_b_emp]):+.1f})",
        edgecolor="black")
ax.axvline(2, color="red", linestyle="--", lw=1.5)
ax.axvline(-2, color="red", linestyle="--", lw=1.5)
ax.set_xlabel("required |c| to flip the answer  (empirical from CSV)")
ax.set_ylabel("count of prompts")
ax.set_title(f"B — Required steering intensity (empirical, j65 CSV)\n"
             f"slope = {slope_avg:.4f}/c → median c_flip = "
             f"{np.median(np.abs(c_flip_emp)):.0f}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# C: dose-response extrapolation curve
ax = axes[2]
c_grid = np.linspace(-50, 50, 200)
# average over all β-prompts: dl(c) ≈ dl_clean + c × slope
example_beta = final[final.toward == "alpha"]["dl_clean"].mean()  # β-prompts get pushed toward α (negative c)
example_alpha = final[final.toward == "beta"]["dl_clean"].mean()  # α-prompts get pushed toward β
ax.plot(c_grid, example_beta + c_grid * slope_avg, "tab:orange", lw=2,
        label=f"avg β-prompt: dl(c) ≈ {example_beta:.2f} + c·{slope_avg:.3f}")
ax.plot(c_grid, example_alpha + c_grid * slope_avg, "tab:blue", lw=2,
        label=f"avg α-prompt: dl(c) ≈ {example_alpha:+.2f} + c·{slope_avg:.3f}")
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=0.5)

# mark the c needed to flip each
c_flip_beta = -example_beta / slope_avg
c_flip_alpha = -example_alpha / slope_avg
ax.scatter([c_flip_beta], [0], s=200, c="tab:orange", marker="*",
           edgecolor="black", zorder=5, label=f"β flip @ c≈{c_flip_beta:+.0f}")
ax.scatter([c_flip_alpha], [0], s=200, c="tab:blue", marker="*",
           edgecolor="black", zorder=5, label=f"α flip @ c≈{c_flip_alpha:+.0f}")

# j65 tested range
ax.axvspan(-2, 2, alpha=0.15, color="red", label="j65 tested range |c|≤2")

ax.set_xlabel("steering intensity c (in σ-units)")
ax.set_ylabel("predicted logit_diff after steering")
ax.set_title("C — Linear extrapolation: how far OUTSIDE j65's range\n"
             f"flip would occur → ~{abs(c_flip_beta):.0f}σ push needed")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(-60, 60)

fig.suptitle("How hard would we need to push to flip α↔β? "
             "(j65 stopped at |c|=2; need ~30× more)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = ROOT / "steering_extrapolation.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved: {out}")

# Summary
print(f"\n{'='*65}")
print("SUMMARY: How hard to push to flip?")
print(f"{'='*65}")
print(f"  Method 1 (analytical):  median |c_flip| = {np.median(np.abs(c_flip_analytical)):.0f}σ")
print(f"  Method 2 (empirical):   median |c_flip| = {np.median(np.abs(c_flip_emp)):.0f}σ")
print(f"  j65 tested up to |c|=2  → ~{np.median(np.abs(c_flip_emp))/2:.0f}× short of needed")
print(f"\n  At |c|=2 (j65 max), only {(np.abs(c_flip_emp) <= 2).mean()*100:.1f}% of prompts would flip")
print(f"  At |c|=10, would flip ~{(np.abs(c_flip_emp) <= 10).mean()*100:.1f}%")
print(f"  At |c|=30, would flip ~{(np.abs(c_flip_emp) <= 30).mean()*100:.1f}%")
print(f"  At |c|=50, would flip ~{(np.abs(c_flip_emp) <= 50).mean()*100:.1f}%")
