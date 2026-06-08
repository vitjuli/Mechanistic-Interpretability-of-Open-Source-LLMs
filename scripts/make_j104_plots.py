"""
4 panels for j104 — the geometric closure of bypass.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/analysis/runD_v2/sweep104"
OUT = ROOT / "docs/j104_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 180, "savefig.bbox": "tight",
})

geo = pd.read_csv(DATA / "geometry_by_layer.csv")
st = pd.read_csv(DATA / "steering_by_layer.csv")
clouds = pd.read_csv(DATA / "clouds_2d.csv")
vec = pd.read_csv(DATA / "vectors_2d.csv")

# ============================================================================
# Plot 1 — AUC-B@35 profile vs intact-flip per layer × {u, w_res}
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: AUC-B@35 collapse profile
for direction, color in [("u", "#d62728"), ("w_res", "#2ca02c"), ("random", "#888")]:
    sub = st[st.direction == direction].sort_values("layer")
    # AUC-B35 has "--" for L24+ when readout layer; convert
    auc = pd.to_numeric(sub["auc_B35_steered"], errors="coerce")
    ax1.plot(sub["layer"], auc, "-o", color=color, lw=2.5, ms=5,
             label=f"steer along {direction}")
ax1.axhline(0.99, color="k", ls="--", lw=1, alpha=0.5, label="clean AUC@35 = 0.99")
ax1.axhline(0.5, color="orange", ls=":", lw=1, alpha=0.5, label="chance")
ax1.set_xlabel("Steered layer")
ax1.set_ylabel(r"AUC-B@35 (decodability at readout L35)")
ax1.set_title("AUC-B@35 after steering at layer $\\ell$\n"
              "w_res push INVERTS readout decodability at L28+ (AUC → 0)")
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc="lower left", fontsize=8, frameon=False)
ax1.grid(alpha=0.2)

# Right: intact-flip per layer
for direction, color in [("u", "#d62728"), ("w_res", "#2ca02c"), ("random", "#888")]:
    sub = st[st.direction == direction].sort_values("layer")
    ax2.plot(sub["layer"], sub["intact_flip"], "-o", color=color, lw=2.5, ms=5,
             label=f"steer along {direction}")
ax2.axhline(0.1, color="black", ls=":", lw=1, alpha=0.5, label="null threshold")
ax2.set_xlabel("Steered layer")
ax2.set_ylabel("intact-flip rate")
ax2.set_title("intact-flip per layer (same prompts, same c=4)\n"
              "u flips 100% from L24; w_res stays in null band everywhere")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="upper left", fontsize=8, frameon=False)
ax2.grid(alpha=0.2)

plt.suptitle("Plot 1 — Decoded ≠ Used: AUC-B@35 collapses for w_res (inversion), behaviour does not flip",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "j104_plot1_aucB_vs_intact.png")
plt.close()
print(f"saved {OUT}/j104_plot1_aucB_vs_intact.png")


# ============================================================================
# Plot 2 — cos profile with γ̄ (signed)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(geo["layer"], geo["cos_wres_gamma"], "-o", color="#2ca02c", lw=3, ms=6,
        label=r"$\cos(w_{res}, \bar\gamma)$  — readable axis vs answer direction")
ax.plot(geo["layer"], geo["cos_u_gamma"], "-o", color="#d62728", lw=3, ms=6,
        label=r"$\cos(u, \bar\gamma)$  — used direction vs answer direction")
ax.plot(geo["layer"], geo["cos_u_wres"], "-o", color="#9467bd", lw=2, ms=5, dash_capstyle="round",
        ls="--", alpha=0.7, label=r"$|\cos(u, w_{res})|$  — used vs readable")
ax.axhline(0, color="k", ls="-", lw=0.7, alpha=0.4)
ax.axhline(0.039, color="gray", ls=":", lw=1, alpha=0.5,
           label="random p95 in d=2560")
ax.axhline(-0.039, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel(r"signed cosine with $\bar\gamma$ (or with $w_{res}$ for purple)")
ax.set_title(r"Signed cosines per layer — $w_{res}$ stays ⊥ to $\bar\gamma$ always; $u \to \bar\gamma$ at the end"
             "\n"
             r"$\bar\gamma = W_U[\beta] - W_U[\alpha]$ (answer-token contrast)")
ax.set_ylim(-0.1, 1.05)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.grid(alpha=0.2)

# Annotate the convergence
ax.annotate(r"$u \to \bar\gamma$ here", xy=(35, 0.993), xytext=(28, 0.85),
            arrowprops=dict(arrowstyle="->", color="#d62728"), fontsize=10, color="#d62728")
ax.annotate(r"$w_{res} \perp \bar\gamma$ everywhere", xy=(24, 0.002), xytext=(15, 0.18),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"), fontsize=10, color="#2ca02c")

plt.tight_layout()
plt.savefig(OUT / "j104_plot2_cos_profile.png")
plt.close()
print(f"saved {OUT}/j104_plot2_cos_profile.png")


# ============================================================================
# Plot 3 — Cloud in plane P1 = {w_res, γ̄} at a key layer
# ============================================================================
key_layer = 21
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax_idx, layer in enumerate([21, 28]):
    ax = axes[ax_idx]
    sub_p1 = clouds[(clouds.layer == layer) & (clouds.plane == "P1")]
    sub_v1 = vec[(vec.layer == layer) & (vec.plane == "P1")]
    # split by class
    a = sub_p1[sub_p1.label == "alpha"]
    b = sub_p1[sub_p1.label == "beta"]
    ax.scatter(a["x"], a["y"], s=18, color="#2ca02c", alpha=0.5, label="α-prompts", edgecolors="none")
    ax.scatter(b["x"], b["y"], s=18, color="#d62728", alpha=0.5, label="β-prompts", edgecolors="none")
    # Arrows for vectors from origin (0,0 after centering at μ_all)
    arrow_objects = ["w_res", "u", "gamma", "delta_wres", "delta_u"]
    colors_v = {"w_res": "#2ca02c", "u": "#d62728", "gamma": "#000000",
                "delta_wres": "#1f77b4", "delta_u": "#ff7f0e"}
    # Scale arrows for visibility
    sx = max(abs(sub_p1["x"].max()), abs(sub_p1["x"].min())) * 0.8
    sy = max(abs(sub_p1["y"].max()), abs(sub_p1["y"].min())) * 0.8
    scale = min(sx, sy)
    for obj in arrow_objects:
        row = sub_v1[sub_v1.object == obj]
        if len(row) == 0: continue
        x, y = float(row["x"].iloc[0]), float(row["y"].iloc[0])
        nrm = np.sqrt(x*x + y*y) + 1e-9
        # plot at unit-vector scale * radius
        x_s, y_s = x / nrm * scale, y / nrm * scale
        ax.arrow(0, 0, x_s, y_s, color=colors_v[obj], head_width=scale*0.04,
                 head_length=scale*0.06, lw=2.5, length_includes_head=True,
                 alpha=0.85, label=obj if ax_idx == 0 else None)
        ax.text(x_s * 1.08, y_s * 1.08, obj, color=colors_v[obj], fontsize=10,
                fontweight="bold", ha="center", va="center")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"projection onto $w_{res}$")
    ax.set_ylabel(r"projection onto $\bar\gamma$")
    ax.set_title(f"L{layer}: prompts in plane $P_1 = \\{{w_{{res}}, \\bar\\gamma\\}}$")
    if ax_idx == 0:
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

plt.suptitle(r"Plot 3 — Class separation along $w_{res}$, but $\bar\gamma$ (answer) ⊥ $w_{res}$",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "j104_plot3_cloud_P1.png")
plt.close()
print(f"saved {OUT}/j104_plot3_cloud_P1.png")


# ============================================================================
# Plot 4 — Cloud in plane P2 = {u, w_res} at a key layer
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax_idx, layer in enumerate([21, 28]):
    ax = axes[ax_idx]
    sub_p2 = clouds[(clouds.layer == layer) & (clouds.plane == "P2")]
    sub_v2 = vec[(vec.layer == layer) & (vec.plane == "P2")]
    a = sub_p2[sub_p2.label == "alpha"]
    b = sub_p2[sub_p2.label == "beta"]
    ax.scatter(a["x"], a["y"], s=18, color="#2ca02c", alpha=0.5, label="α-prompts", edgecolors="none")
    ax.scatter(b["x"], b["y"], s=18, color="#d62728", alpha=0.5, label="β-prompts", edgecolors="none")
    arrow_objects = ["u", "w_res", "gamma"]
    colors_v = {"u": "#d62728", "w_res": "#2ca02c", "gamma": "#000000"}
    sx = max(abs(sub_p2["x"].max()), abs(sub_p2["x"].min())) * 0.8
    sy = max(abs(sub_p2["y"].max()), abs(sub_p2["y"].min())) * 0.8
    scale = min(sx, sy)
    for obj in arrow_objects:
        row = sub_v2[sub_v2.object == obj]
        if len(row) == 0: continue
        x, y = float(row["x"].iloc[0]), float(row["y"].iloc[0])
        nrm = np.sqrt(x*x + y*y) + 1e-9
        x_s, y_s = x / nrm * scale, y / nrm * scale
        ax.arrow(0, 0, x_s, y_s, color=colors_v[obj], head_width=scale*0.04,
                 head_length=scale*0.06, lw=2.5, length_includes_head=True, alpha=0.85)
        ax.text(x_s * 1.08, y_s * 1.08, obj, color=colors_v[obj], fontsize=10,
                fontweight="bold", ha="center", va="center")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"projection onto $u$")
    ax.set_ylabel(r"projection onto $w_{res}$")
    ax.set_title(f"L{layer}: prompts in plane $P_2 = \\{{u, w_{{res}}\\}}$")
    if ax_idx == 0:
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

plt.suptitle(r"Plot 4 — In $P_2$: $u$ and $w_{res}$ are visually ⊥ but BOTH separate the classes",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "j104_plot4_cloud_P2.png")
plt.close()
print(f"saved {OUT}/j104_plot4_cloud_P2.png")

print("\nAll 4 plots saved to", OUT)
