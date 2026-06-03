"""
Visualize 37 carrier feature decoder-vectors (L18+L24) + gbar in one plane.

We have the carrier-only Gram matrix (37×37) in causal metric and their cosines
with gbar — but NOT full W_dec vectors locally. We reconstruct relative geometry
via classical MDS from the Gram matrix, then anchor gbar through its known
cosines with the carriers.

w_res cannot be placed yet — cos_C(d_f, w_res) is what script 66 computes on
CSD3 (still pending).
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eigh

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
G = np.load(ROOT / "carrier_gram_ALL.npy").astype(np.float64)   # (37, 37)
C = np.load(ROOT / "carrier_cosine_ALL.npy").astype(np.float64) # cos_C (37,37)
cos_gbar = np.load(ROOT / "concept_cosines_ALL.npy").astype(np.float64)  # (37,)

n = G.shape[0]

# ── classical MDS from COSINE matrix (unit-normalised view) ──────────────────
# carriers are NOT unit-norm in causal metric (||d||²~2676 each, full-rank).
# Use cosine matrix C (off-diag ~0, diag=1) so each carrier sits on unit sphere.
# Double-center and eig:
H_cen = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H_cen @ (1 - C) @ H_cen          # squared-distance form: 2 - 2*cos
# Note: cosine MDS uses 2(1-cos) as squared distance for unit vectors
B = -0.5 * H_cen @ (2 * (1 - C)) @ H_cen
ev, vec = eigh(B)
order = np.argsort(-ev)
ev, vec = ev[order], vec[:, order]
coords_full = vec[:, :2] * np.sqrt(np.clip(ev[:2], 0, None))
expl2 = ev[:2].sum() / ev[ev > 0].sum()
print(f"MDS-from-cosine: top-2 axes capture {expl2:.1%} of metric variance")

# ── place gbar via least-squares from known cos_C(d_i, gbar) ─────────────────
# After unit-normalising d_i (via cosine MDS), coords_full[i] are points on a
# sphere of radius sqrt(1-mean_dist). Find 2D coords g for gbar (unit) such
# that distance²(d_i, gbar) ≈ 2(1 - cos_gbar[i]).
# Easier: lstsq on inner products. We treat coords_full as a basis and solve.
g_2d, *_ = np.linalg.lstsq(coords_full, cos_gbar - cos_gbar.mean(), rcond=None)
residual_norm = float(np.linalg.norm(coords_full @ g_2d - (cos_gbar - cos_gbar.mean())))
gbar_norm_in_plane = float(np.linalg.norm(g_2d))
print(f"gbar placement: in-plane norm = {gbar_norm_in_plane:.3f}  "
      f"residual = {residual_norm:.3f}  "
      f"=> gbar mostly OUTSIDE the carrier plane")

# ── layer info (parse layer from feature_id order isn't here, but params has it) ──
# load list from concept_projection.json
proj = json.load(open(ROOT / "concept_projection.json"))
k18 = proj["layers"]["18"]["k"]    # 17
k24 = proj["layers"]["24"]["k"]    # 20
assert k18 + k24 == n, f"{k18}+{k24} != {n}"
layer_of = np.array([18] * k18 + [24] * k24)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# A — carrier cloud + gbar
ax = axes[0]
m18 = layer_of == 18
m24 = layer_of == 24
ax.scatter(coords_full[m18, 0], coords_full[m18, 1], s=80, c="tab:blue",
           alpha=0.7, label=f"L18 carriers (n={m18.sum()})", edgecolor="black")
ax.scatter(coords_full[m24, 0], coords_full[m24, 1], s=80, c="tab:orange",
           alpha=0.7, label=f"L24 carriers (n={m24.sum()})", edgecolor="black")

# gbar as red arrow from origin
ax.annotate("", xy=(g_2d[0], g_2d[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5,
                            linestyle="dashed"))
ax.text(g_2d[0]*1.15, g_2d[1]*1.15,
        f"gbar\n(in-plane fraction\n= {gbar_norm_in_plane:.3f})",
        color="red", fontsize=10, ha="center", fontweight="bold")

# unit circle (where unit vectors should lie if perfectly in-plane)
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.5, linestyle=":")
ax.text(1.02, 0, "|d|=1\n(unit circle)", fontsize=8, color="gray")

ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("MDS axis 1")
ax.set_ylabel("MDS axis 2")
ax.set_title("A — 37 carrier decoder vectors + gbar\n"
             "MDS from causal Gram matrix (L18 blue, L24 orange)")
ax.legend(loc="upper left")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
lim = max(np.abs(coords_full).max() * 1.3, 1.2)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

# B — histogram of cos_C(d_f, gbar) vs null
ax = axes[1]
null_p95 = 0.020   # from concept_projection.json (null mean |cos| p95)
ax.axvspan(-null_p95, null_p95, alpha=0.2, color="red",
           label=f"null |cos|<p95 ({null_p95:.3f})")
bins = np.linspace(-0.06, 0.06, 25)
ax.hist(cos_gbar[m18], bins=bins, alpha=0.6, color="tab:blue",
        label=f"L18 carriers (μ={cos_gbar[m18].mean():+.4f})", edgecolor="black")
ax.hist(cos_gbar[m24], bins=bins, alpha=0.6, color="tab:orange",
        label=f"L24 carriers (μ={cos_gbar[m24].mean():+.4f})", edgecolor="black")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("cos_C(d_f, gbar)  — signed causal cosine")
ax.set_ylabel("count")
ax.set_title("B — Carrier-vs-gbar cosines\n"
             "all carriers fall inside the null band → orthogonal in causal metric")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Carrier feature decoder vectors vs gbar  "
             "(w_res placement awaiting CSD3 job j66)",
             fontsize=13)

out = ROOT / "carriers_vs_gbar.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print(f"\nSummary:")
print(f"  gbar in-plane norm: {gbar_norm_in_plane:.4f}  "
      f"=> {gbar_norm_in_plane**2*100:.2f}% of gbar lies in carrier 2D plane")
print(f"  mean cos_C(carrier, gbar): {cos_gbar.mean():+.4f}")
print(f"  std:                       {cos_gbar.std():.4f}")
print(f"  abs mean:                  {np.abs(cos_gbar).mean():+.4f}  (null mean ~0.016)")
