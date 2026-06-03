"""
Visualize the probe subspace {w_res^L14, ..., w_res^L25, w_res^final}.

Two views (causal metric — using Sigma_inv whitening):
  Top    : 2D PCA — first two principal axes of the stacked probe vectors
           Each probe is an arrow from origin, color = depth (viridis L14→final).
           gbar (answer-token axis) is shown for contrast.
  Bottom : Same plane, but with α/β data clouds (from h_residual_per_depth.npz
           at postL22) projected onto it — confirms that the plane separates
           classes.

The Σ⁻¹ whitening is applied so cosines = standard inner products in this view.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eigh

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
diag = json.load(open(ROOT / "concept_axis_diagnosis.json"))
taps = [r["tap"] for r in diag["D1"]]                # postL14..postL25, final

W = np.stack([np.load(ROOT / f"w_res_{t}.npy") for t in taps]).astype(np.float64)
cd = np.load(ROOT / "concept_directions.npz")
Sinv = cd["Sigma_inv"].astype(np.float64)
gbar = cd["gbar"].astype(np.float64)

# ── whitener A so that <a,b>_M = <Aa, Ab>_eucl (where M = Sigma_inv) ──────────
w, V = eigh(Sinv)
w = np.clip(w, 1e-12, None)
A = V @ np.diag(np.sqrt(w)) @ V.T

Ww = W @ A.T                       # (n_taps, d) in whitened space
gw = gbar @ A.T
# normalise to unit length in whitened (= unit in causal metric)
Ww = Ww / np.linalg.norm(Ww, axis=1, keepdims=True)
gw = gw / np.linalg.norm(gw)

# ── PCA in whitened space (top-2 PCs of the probe stack) ─────────────────────
U, S, Vt = np.linalg.svd(Ww, full_matrices=False)
PC = Vt[:2]                                # (2, d)
explained = (S**2) / (S**2).sum()

# 2D coords for probes + gbar
P = Ww @ PC.T                              # (n_taps, 2)
G = gw @ PC.T                              # (2,)

# ── load alpha/beta cloud at postL22 (peak AUC), project onto same plane ──────
h = np.load(ROOT / "h_residual_per_depth.npz")
H = h["postL22"].astype(np.float64)        # (538, 2560)
y = h["y"]                                 # alpha=0, beta=1 (or 1/0 — re-check)
# whiten data but DON'T unit-normalise — keep natural scale in causal metric
Hw = H @ A.T
# center for cleaner projection
Hw = Hw - Hw.mean(0, keepdims=True)
D = Hw @ PC.T                              # (538, 2) — natural-scale projection
# rescale probes/gbar so they sit at the cloud's outer scale
cloud_scale = np.abs(D).max()
P_show = P * cloud_scale * 0.9
G_show = G * cloud_scale * 0.9

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# left: probe subspace (just arrows)
ax = axes[0]
short = [t.replace("postL", "L") for t in taps]
cmap = plt.cm.viridis
for i, (lab, (x, yy)) in enumerate(zip(short, P)):
    col = cmap(i / (len(taps) - 1))
    ax.annotate("", xy=(x, yy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=2, alpha=0.85))
    ax.text(x * 1.07, yy * 1.07, lab, color=col, fontsize=10, ha="center",
            fontweight="bold" if lab in ("L14", "L25", "final") else "normal")

# gbar reference (red dashed)
ax.annotate("", xy=(G[0], G[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2, linestyle="dashed"))
ax.text(G[0]*1.07, G[1]*1.07, "gbar\n(answer-token)", color="red", fontsize=10,
        ha="center", fontweight="bold")

ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel(f"PC1  ({explained[0]:.0%} var)")
ax.set_ylabel(f"PC2  ({explained[1]:.0%} var)")
ax.set_title(f"A — probe subspace {{w_res^L14...L25, final}} (causal metric)\n"
             f"top-2 PCs explain {explained[:2].sum():.0%} of variance")
lim = max(abs(P).max(), abs(G).max(), 0.5) * 1.25
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect("equal"); ax.grid(alpha=0.3)

# right: data cloud + probes
ax = axes[1]
m_alpha = (y == 0)
m_beta  = (y == 1)
ax.scatter(D[m_alpha, 0], D[m_alpha, 1], s=10, c="tab:blue",   alpha=0.4, label=f"α (n={m_alpha.sum()})")
ax.scatter(D[m_beta,  0], D[m_beta,  1], s=10, c="tab:orange", alpha=0.4, label=f"β (n={m_beta.sum()})")

# overlay probes (rescaled to data cloud)
for i, (x, yy) in enumerate(P_show):
    col = cmap(i / (len(taps) - 1))
    ax.annotate("", xy=(x, yy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.8, alpha=0.95))

ax.annotate("", xy=(G_show[0], G_show[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2, linestyle="dashed"))
ax.text(G_show[0]*1.07, G_show[1]*1.07, "gbar", color="red", fontsize=9,
        ha="center", fontweight="bold")

ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel(f"PC1  ({explained[0]:.0%} var)")
ax.set_ylabel(f"PC2  ({explained[1]:.0%} var)")
ax.set_title("B — α/β cloud at postL22 projected onto the probe plane")
ax.legend(loc="best")
ax.set_aspect("equal"); ax.grid(alpha=0.3)

# colorbar for depth
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(taps)-1))
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cb.set_ticks(range(len(taps)))
cb.set_ticklabels(short)
cb.set_label("depth", rotation=270, labelpad=15)

fig.suptitle("Probe subspace geometry — α vs β decay, Σ⁻¹-whitened coordinates",
             fontsize=13)

out = ROOT / "probe_subspace.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print(f"\ntop-{len(taps)} singular values:")
for i, s in enumerate(S):
    print(f"  PC{i+1}: σ={s:.3f}  explained={explained[i]:.2%}")
print(f"\ngbar PC1, PC2: {G[0]:+.3f}, {G[1]:+.3f}  (|gbar in plane|={np.linalg.norm(G):.3f})")
print(f"=> {np.linalg.norm(G)**2:.1%} of gbar lies in the probe plane (causal metric)")
