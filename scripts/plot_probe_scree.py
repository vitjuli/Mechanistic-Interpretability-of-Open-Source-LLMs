"""
Scree plot of the probe subspace: 13 probes -> 13 PCs.
Shows individual + cumulative explained variance, with 80/90/95% thresholds.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eigh

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
diag = json.load(open(ROOT / "concept_axis_diagnosis.json"))
taps = [r["tap"] for r in diag["D1"]]

W = np.stack([np.load(ROOT / f"w_res_{t}.npy") for t in taps]).astype(np.float64)
Sinv = np.load(ROOT / "concept_directions.npz")["Sigma_inv"].astype(np.float64)

# whitener so causal cos = euclidean cos in whitened space
w, V = eigh(Sinv); w = np.clip(w, 1e-12, None)
A = V @ np.diag(np.sqrt(w)) @ V.T
Ww = W @ A.T
Ww = Ww / np.linalg.norm(Ww, axis=1, keepdims=True)

# SVD -> singular values, explained var
_, S, _ = np.linalg.svd(Ww, full_matrices=False)
expl = (S**2) / (S**2).sum()
cum  = np.cumsum(expl)

n = len(S)
k_80 = int(np.searchsorted(cum, 0.80) + 1)
k_90 = int(np.searchsorted(cum, 0.90) + 1)
k_95 = int(np.searchsorted(cum, 0.95) + 1)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(1, n + 1)

# bars: individual variance
bars = ax.bar(x, expl * 100, color="steelblue", alpha=0.7,
              edgecolor="black", label="individual")
# line: cumulative
ax2 = ax.twinx()
ax2.plot(x, cum * 100, "o-", color="darkorange", lw=2, markersize=8,
         label="cumulative")
ax2.set_ylim(0, 105)
ax2.set_ylabel("cumulative explained variance (%)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# threshold lines on cumulative
for thr, label, k in [(80, "80%", k_80), (90, "90%", k_90), (95, "95%", k_95)]:
    ax2.axhline(thr, color="gray", lw=0.8, linestyle=":")
    ax2.text(n + 0.1, thr, f"{label} → {k} PCs", color="gray",
             va="center", fontsize=9)

# annotate cumulative %
for xi, yi in zip(x, cum * 100):
    ax2.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8, color="darkorange")

# annotate individual %
for xi, yi in zip(x, expl * 100):
    if yi > 1:
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color="steelblue")

ax.set_xticks(x)
ax.set_xlabel("Principal component")
ax.set_ylabel("individual explained variance (%)", color="steelblue")
ax.tick_params(axis="y", labelcolor="steelblue")
ax.set_title("Scree plot — probe subspace {w_res^L14…L25, w_res^final}\n"
             "13 probes → 13 PCs (full subspace dimension)")
ax.grid(alpha=0.3, axis="y")

# legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

fig.tight_layout()
out = ROOT / "probe_subspace_scree.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

print("\nThresholds:")
print(f"  80% variance → {k_80} PCs")
print(f"  90% variance → {k_90} PCs")
print(f"  95% variance → {k_95} PCs")
print(f"  100% variance → {n} PCs (= number of probes, full rank)")
