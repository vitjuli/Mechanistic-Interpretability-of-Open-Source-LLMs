"""
Plot j74 result: carrier collective capture vs matched random-feature null.
Closes "Hole 3" — confirms 0.303 is at the null baseline, not significant.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
d = json.load(open(ROOT / "carrier_capture_null.json"))

# We don't have the full null sample distribution saved, just summary stats.
# Reconstruct an approximate gaussian for visualisation using the stats.
mean = d["null_mean"]
std  = d["null_std"]
p95  = d["null_p95"]
p99  = d["null_p99"]
n_min = d["null_min"]
n_max = d["null_max"]
carrier = d["carrier_capture"]
isotropic = d["isotropic_sqrt_k_over_d"]
percentile = d["percentile_of_carrier"]

fig, ax = plt.subplots(figsize=(11, 6))

# Build approximate distribution: gaussian + samples scattered within range
xs = np.linspace(n_min - 0.005, n_max + 0.005, 500)
gauss = (1 / (std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
ax.fill_between(xs, 0, gauss, alpha=0.3, color="tab:blue",
                label=f"matched random-feature null (n=300)\nmean={mean:.3f} std={std:.3f}")
ax.plot(xs, gauss, color="tab:blue", lw=1.5)

# Mark percentile thresholds
ax.axvline(p95, color="darkblue", linestyle="--", lw=1, alpha=0.6,
           label=f"null p95 = {p95:.3f}")
ax.axvline(p99, color="darkblue", linestyle=":", lw=1, alpha=0.6,
           label=f"null p99 = {p99:.3f}")

# Isotropic baseline
ax.axvline(isotropic, color="gray", linestyle="dashdot", lw=1.5,
           label=f"isotropic √(k/d) = {isotropic:.3f}")

# Carrier value — the main bar
ax.axvline(carrier, color="tab:red", lw=3,
           label=f"co-importance carrier = {carrier:.3f}  (pct = {percentile:.0f}%)")
ax.annotate(f"carrier\n{carrier:.3f}\nat {percentile:.0f}th pct",
            xy=(carrier, max(gauss) * 0.85), xytext=(carrier - 0.02, max(gauss) * 1.05),
            ha="center", color="darkred", fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

ax.set_xlabel("collective capture of w_res by 227-feature subspace", fontsize=11)
ax.set_ylabel("null density", fontsize=11)
ax.set_title(
    "Hole 3 closed — Carrier capture (0.303) sits IN the matched random-feature null\n"
    "The dictionary contains the concept axis no more than random features would",
    fontsize=12
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# Verdict box
ax.text(0.98, 0.02,
        f"VERDICT: NULL-LEVEL\n"
        f"Carrier at pct {percentile:.0f}% (>50% would be specific)\n"
        f"0.30 is a dimensional artefact",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

fig.tight_layout()
out = ROOT / "carrier_capture_null.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print(f"\nSummary:")
print(f"  carrier:           {carrier:.4f}")
print(f"  isotropic √(k/d):  {isotropic:.4f}")
print(f"  random-feat null:  mean={mean:.4f}  p95={p95:.4f}  p99={p99:.4f}")
print(f"  carrier percentile: {percentile:.1f}%")
print(f"  → {d['verdict'][:80]}...")
