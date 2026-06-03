"""
Intuitive figure: 'concept axis' explanation.

For one chosen depth (default postL22, peak AUC):
  * Compute mu_alpha, mu_beta, w_res (Fisher LDA).
  * Project all 538 residual stream activations h(p) into 2D:
      x-axis = projection on w_res  ("concept axis")
      y-axis = projection on top orthogonal PC ("orthogonal variation")
  * Show:
      α-cloud (blue) and β-cloud (orange) — each point = one prompt
      centroids μ_α, μ_β as big markers
      w_res arrow from μ_α to μ_β (this IS the discovered axis)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
h = np.load(ROOT / "h_residual_per_depth.npz")
y, is_train = h["y"], h["is_train"]

# pick L22 (peak heldout AUC)
TAP = "postL22"
H = h[TAP].astype(np.float64)              # (538, 2560)
w = np.load(ROOT / f"w_res_{TAP}.npy").astype(np.float64)   # (2560,) unit

# class means (use train only, as Fisher did)
mu_a = H[(y == 0) & is_train].mean(0)
mu_b = H[(y == 1) & is_train].mean(0)

# project onto w_res (the concept axis itself)
x = H @ w                                  # (538,) — "position along concept axis"

# residual variation orthogonal to w
H_perp = H - np.outer(H @ w, w)            # remove component along w
# top PC of orthogonal residual = "second-most-spread direction"
H_perp_centered = H_perp - H_perp.mean(0)
U, S, Vt = np.linalg.svd(H_perp_centered, full_matrices=False)
e2 = Vt[0]                                 # second axis: top PC of perpendicular
y_coord = H @ e2                           # (538,)

# centroid projections
mu_a_x, mu_a_y = float(mu_a @ w), float(mu_a @ e2)
mu_b_x, mu_b_y = float(mu_b @ w), float(mu_b @ e2)

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))

# clouds
m_a = (y == 0)
m_b = (y == 1)
ax.scatter(x[m_a], y_coord[m_a], s=22, c="tab:blue",   alpha=0.55,
           label=f"α-prompts (n={m_a.sum()})", edgecolor="white", linewidth=0.3)
ax.scatter(x[m_b], y_coord[m_b], s=22, c="tab:orange", alpha=0.55,
           label=f"β-prompts (n={m_b.sum()})", edgecolor="white", linewidth=0.3)

# centroids
ax.scatter([mu_a_x], [mu_a_y], s=300, marker="*", c="tab:blue",
           edgecolor="black", linewidth=1.5, zorder=5, label="μ_α (α-centroid)")
ax.scatter([mu_b_x], [mu_b_y], s=300, marker="*", c="tab:orange",
           edgecolor="black", linewidth=1.5, zorder=5, label="μ_β (β-centroid)")

# w_res arrow: from α-centroid toward β-centroid
ax.annotate("",
            xy=(mu_b_x, mu_b_y),
            xytext=(mu_a_x, mu_a_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=3, alpha=0.95),
            zorder=6)
mid_x = (mu_a_x + mu_b_x) / 2
mid_y = (mu_a_y + mu_b_y) / 2 + 25
ax.text(mid_x, mid_y, "w_res\n(concept axis)", color="red", fontsize=14,
        ha="center", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="red", alpha=0.85))

# threshold line — midpoint orthogonal to w_res
threshold = (mu_a_x + mu_b_x) / 2
ax.axvline(threshold, color="red", linestyle="--", lw=1.2, alpha=0.6,
           label=f"decision threshold")

# distributions in the margin (rug-like)
ax.set_xlabel(r"projection onto $w_{\mathrm{res}}$  =  $\langle h(p), w_{\mathrm{res}}\rangle$    "
              "← α  |  β →", fontsize=12)
ax.set_ylabel("projection onto orthogonal direction\n(other variation in residual stream)",
              fontsize=11)

ax.set_title(f"Concept axis at {TAP} — each point is one of 538 prompts\n"
             f"w_res points from μ_α to μ_β, classes separate along it (heldout AUC=0.995)",
             fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.3)

fig.tight_layout()
out = ROOT / "concept_axis_intuition.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print(f"\nGeometry at {TAP}:")
print(f"  μ_α along w_res: {mu_a_x:+.3f}")
print(f"  μ_β along w_res: {mu_b_x:+.3f}")
print(f"  separation (||μ_β - μ_α|| along w):  {mu_b_x - mu_a_x:.3f}")
print(f"  within-class spread (std along w): "
      f"α={x[m_a].std():.3f}, β={x[m_b].std():.3f}")
