"""
Visualize the 13 probes {w_res^L14...L25, w_res^final} TOGETHER with gbar
in the same residual stream space.

Two panels:
  A — PCA of [W; gbar] stacked (14 vectors).  gbar participates in choosing
      the basis, so we see its direction explicitly, not collapsed to origin.
  B — A custom 2D plane: x = top-PC of probes only, y = component of gbar
      orthogonal to that PC.  This is the "worst case" view that exposes
      angle between gbar and the dominant probe axis.

All in Σ⁻¹-whitened (causal) metric — angles = causal cosines.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eigh

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
diag = json.load(open(ROOT / "concept_axis_diagnosis.json"))
taps = [r["tap"] for r in diag["D1"]]

W = np.stack([np.load(ROOT / f"w_res_{t}.npy") for t in taps]).astype(np.float64)
cd = np.load(ROOT / "concept_directions.npz")
Sinv = cd["Sigma_inv"].astype(np.float64)
gbar = cd["gbar"].astype(np.float64)

# whitener: cos_C = euclidean cos in whitened space
w, V = eigh(Sinv); w = np.clip(w, 1e-12, None)
A = V @ np.diag(np.sqrt(w)) @ V.T
Ww = W @ A.T
Ww = Ww / np.linalg.norm(Ww, axis=1, keepdims=True)
gw = gbar @ A.T
gw = gw / np.linalg.norm(gw)

short = [t.replace("postL", "L") for t in taps]
cmap = plt.cm.viridis

# ── PANEL A: PCA of [W; gbar] ────────────────────────────────────────────────
stacked = np.vstack([Ww, gw[None, :]])     # (14, d)
_, Sa, Vta = np.linalg.svd(stacked, full_matrices=False)
PC_a = Vta[:2]
expl_a = (Sa**2) / (Sa**2).sum()

P_A = Ww @ PC_a.T
G_A = gw @ PC_a.T

# ── PANEL B: custom basis (probe-PC1, gbar⊥) ─────────────────────────────────
# x = top PC of probes only
_, Sb, Vtb = np.linalg.svd(Ww, full_matrices=False)
e1 = Vtb[0]
# y = component of gbar orthogonal to e1
e2 = gw - (gw @ e1) * e1
e2 = e2 / (np.linalg.norm(e2) + 1e-12)
B = np.stack([e1, e2])                      # (2, d)

P_B = Ww @ B.T
G_B = gw @ B.T

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

def plot_arrows(ax, P, G, xlabel, ylabel, title):
    for i, (x, y) in enumerate(P):
        col = cmap(i / (len(taps) - 1))
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2, alpha=0.9))
        ax.text(x * 1.07, y * 1.07, short[i], color=col, fontsize=10,
                ha="center",
                fontweight="bold" if short[i] in ("L14", "L25", "final") else "normal")
    ax.annotate("", xy=(G[0], G[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5,
                                linestyle="dashed"))
    ax.text(G[0]*1.08, G[1]*1.08, "gbar\n(answer-token)", color="red",
            fontsize=11, ha="center", fontweight="bold")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    lim = max(np.abs(P).max(), np.abs(G).max(), 0.5) * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

plot_arrows(
    axes[0], P_A, G_A,
    f"PC1 ({expl_a[0]:.0%})", f"PC2 ({expl_a[1]:.0%})",
    "A — PCA of [13 probes + gbar]\ngbar participates in basis choice"
)

# annotation on panel A
axes[0].text(0.02, 0.02,
             f"top-2 PCs explain {expl_a[:2].sum():.0%} of stack variance",
             transform=axes[0].transAxes, fontsize=9,
             bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

plot_arrows(
    axes[1], P_B, G_B,
    "probe-PC1 (best probe axis)",
    "gbar⊥  (gbar component ⟂ probe-PC1)",
    "B — Custom plane: probe principal axis vs gbar\n"
    "shows angle between best probe direction and answer axis"
)

# angle annotation on panel B
cos_pc1_gbar = float(e1 @ gw)
angle_deg = np.degrees(np.arccos(np.clip(abs(cos_pc1_gbar), 0, 1)))
axes[1].text(0.02, 0.02,
             f"cos_C(probe-PC1, gbar) = {cos_pc1_gbar:+.3f}\n"
             f"angle ≈ {angle_deg:.1f}°",
             transform=axes[1].transAxes, fontsize=10,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="red"))

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=0, vmax=len(taps)-1))
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cb.set_ticks(range(len(taps))); cb.set_ticklabels(short)
cb.set_label("depth", rotation=270, labelpad=15)

fig.suptitle("13 probe directions + gbar in the SAME residual stream space "
             "(Σ⁻¹-whitened)", fontsize=13)

out = ROOT / "probes_vs_gbar.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

print(f"\nAngle stats:")
print(f"  cos_C(probe-PC1, gbar) = {cos_pc1_gbar:+.3f}  → {angle_deg:.1f}°")
print(f"  Each probe vs gbar:")
for t, ww in zip(short, Ww):
    c = float(ww @ gw)
    print(f"    {t:>6}: cos_C = {c:+.3f}  → angle {np.degrees(np.arccos(np.clip(abs(c),0,1))):.1f}°")
