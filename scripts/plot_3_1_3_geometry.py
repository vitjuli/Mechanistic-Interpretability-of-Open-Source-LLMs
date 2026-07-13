"""
Two figures for Section 3.1.3 (alpha/beta scaffold):
  (1) cosines by layer: cos(w,u), cos(Delta_mu,u), cos(u,gamma), cos(w,gamma), cos(Delta_mu,gamma).
      Shows the reading direction stays orthogonal, while use converges to gamma and writing aligns with use.
  (2) 3D trajectory of the use direction u across layers, projected onto a frame whose first axis is
      gamma. u sweeps from orthogonal to gamma (early) to aligned with gamma (late).

Input : data/analysis/runD_v2/B1_alpha_beta/field_dump/{res,grad}_L*.npy, meta.npz (gamma = wU_diff)
Output: docs/figures/cos_by_layer.png, docs/figures/u_rotation_to_gamma_3d.png
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import scienceplots  # noqa: F401
plt.style.use(["science", "ieee", "grid", "no-latex"])
plt.rcParams.update({
    "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
    "xtick.labelsize": 15, "ytick.labelsize": 20, "legend.fontsize": 16,
    "lines.linewidth": 2.6, "lines.markersize": 6, "axes.linewidth": 1.1,
    "legend.title_fontsize": 14,
})
DD = "data/analysis/runD_v2/B1_alpha_beta/field_dump"


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(H, y, lam=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    d = mu1 - mu0
    H0, H1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (H0.T @ H0 + H1.T @ H1) / max(1, len(H) - 2)
    Sw = (1 - lam) * Sw + lam * (np.trace(Sw) / Sw.shape[0]) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, d))


def load():
    m = np.load(f"{DD}/meta.npz")
    y = m["y"].astype(int)
    gamma = unit(m["wU_diff"].astype(np.float64))
    layers, U, D, W = [], [], [], []
    for L in range(36):
        rf, gf = f"{DD}/res_L{L:02d}.npy", f"{DD}/grad_L{L:02d}.npy"
        if not (os.path.exists(rf) and os.path.exists(gf)):
            continue
        H = np.load(rf).astype(np.float64)
        G = np.load(gf).astype(np.float64)
        layers.append(L)
        D.append(unit(H[y == 1].mean(0) - H[y == 0].mean(0)))
        U.append(unit((G[y == 0].mean(0) + G[y == 1].mean(0)) / 2))
        W.append(fisher(H, y))
    return np.array(layers), np.array(U), np.array(D), np.array(W), gamma


def fig_curves(layers, U, D, W, gamma):
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    L0 = 15
    mask = layers >= L0
    lx = layers[mask]
    # cos area per direction: u -> cos(u,gamma); writing -> cos(delta,u); reading -> cos(w,u)
    area = {"u": np.abs(U @ gamma)[mask], "delta": np.abs((D * U).sum(1))[mask], "w": np.abs((W * U).sum(1))[mask]}
    area_lab = {"u": "cos($u$, $\\gamma$)", "delta": "cos($\\Delta\\mu$, $u$)", "w": "cos($w_{\\mathrm{LDA}}$, $u$)"}
    marg_lab = {"u": "margin $u$", "delta": "margin $\\Delta\\mu$", "w": "margin $w_{\\mathrm{LDA}}$"}
    g = pd.read_csv("data/analysis/27c/axis_geometry_by_layer.csv").set_index("layer")
    marg = {k: g[f"margin_{k}"].reindex(layers).values[mask] for k in ("u", "delta", "w")}
    colors = {"u": "#1f77b4", "delta": "#2ca02c", "w": "#d62728"}

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axvspan(19, 24, color="gray", alpha=0.12)  # decision zone L19-24 (App B)
    for k in ("u", "delta", "w"):
        ax.fill_between(lx, 0, area[k], color=colors[k], alpha=0.18, zorder=1)
        ax.plot(lx, area[k], color=colors[k], lw=1.2, ls="--", alpha=0.7, zorder=2)
    for k in ("u", "delta", "w"):
        ax.plot(lx, marg[k], color=colors[k], lw=2.4, marker="o", ms=3.5, zorder=3)

    leg_m = ax.legend(handles=[Line2D([0], [0], color=colors[k], lw=2.4, marker="o", ms=3.5, label=marg_lab[k])
                               for k in ("u", "delta", "w")],
                      fontsize=11, title_fontsize=11, loc="upper left", title="margin flip")
    ax.add_artist(leg_m)
    ax.legend(handles=[Patch(facecolor=colors[k], alpha=0.45, label=area_lab[k]) for k in ("u", "delta", "w")],
              fontsize=11, title_fontsize=11, loc="lower left", bbox_to_anchor=(0.01, 0.10), title="cosine")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine  /  margin flip")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(L0, layers.max())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/figures/cos_by_layer.png", dpi=200, bbox_inches="tight")
    print("wrote docs/figures/cos_by_layer.png")


def fig_3d(layers, U, gamma):
    # frame: e1 = gamma; e2,e3 = top PCs of the component of u orthogonal to gamma
    R = U - (U @ gamma)[:, None] * gamma[None, :]
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    e2, e3 = Vt[0], Vt[1]
    x = U @ gamma          # alignment with gamma
    yv = U @ e2
    zv = U @ e3
    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, yv, zv, c=layers, cmap="viridis", s=35, zorder=3)
    ax.plot(x, yv, zv, color="gray", alpha=0.5, lw=1)
    # gamma direction marker at (1,0,0)
    ax.scatter([1], [0], [0], color="red", s=90, marker="*", label="$\\gamma$ (output)")
    ax.plot([0, 1], [0, 0], [0, 0], color="red", ls="--", alpha=0.6)
    ax.set_xlabel("along $\\gamma$")
    ax.set_ylabel("orthogonal 1")
    ax.set_zlabel("orthogonal 2")
    ax.set_title("Rotation of the use direction $u$ toward $\\gamma$ across layers")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("layer")
    ax.legend(loc="upper left", fontsize=15)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig("docs/figures/u_rotation_to_gamma_3d.png", dpi=200, bbox_inches="tight")
    print("wrote docs/figures/u_rotation_to_gamma_3d.png")


def main():
    os.makedirs("docs/figures", exist_ok=True)
    layers, U, D, W, gamma = load()
    fig_curves(layers, U, D, W, gamma)
    fig_3d(layers, U, gamma)


if __name__ == "__main__":
    main()
