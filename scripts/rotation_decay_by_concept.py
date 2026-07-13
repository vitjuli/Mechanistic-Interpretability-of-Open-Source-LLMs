"""
Rotation decay rho(Delta k) per axis, across concepts (Appendix C, spinning of directions).
rho(dk) = mean over layers of |cos(v(l), v(l+dk))| for each axis v in {w_res, delta, u}.
The reading axis rotates fastest, the use axis slowest, the writing axis in between, on every concept.

Input : data/analysis/runD_v2/{field_dump, B1_alpha_beta/field_dump, raw_suffix/field_dump,
        particles4_binary/electron_vs_photon/field_dump}
Output: data/analysis/27c/rotation_decay_by_concept.csv
        docs/figures/rotation_decay_curves.png, docs/figures/rotation_decay_bars.png
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
RED, GREEN, BLUE = "#b22222", "#2ca02c", "#1f77b4"
CONCEPTS = [("α/β scaffold", "B1_alpha_beta/field_dump"),
            ("α/β raw", "field_dump"),
            ("α/β raw+suffix", "raw_suffix/field_dump"),
            ("particles", "particles4_binary/electron_vs_photon/field_dump")]
ROOT = "data/analysis/runD_v2"
DKS = list(range(1, 25))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(H, y, lam=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    d = mu1 - mu0
    H0, H1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (H0.T @ H0 + H1.T @ H1) / (len(H) - 2)
    Sw = (1 - lam) * Sw + lam * (np.trace(Sw) / Sw.shape[0]) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, d))


def axes_per_layer(base):
    y = np.load(f"{base}/meta.npz")["y"].astype(int)
    W, D, U = {}, {}, {}
    for L in range(36):
        rf, gf = f"{base}/res_L{L:02d}.npy", f"{base}/grad_L{L:02d}.npy"
        if not (os.path.exists(rf) and os.path.exists(gf)):
            continue
        H = np.load(rf).astype(np.float64)
        G = np.load(gf).astype(np.float64)
        W[L] = fisher(H, y)
        D[L] = unit(H[y == 1].mean(0) - H[y == 0].mean(0))
        U[L] = unit((G[y == 0].mean(0) + G[y == 1].mean(0)) / 2)
    return W, D, U


def rho(ax, dk):
    Ls = sorted(ax)
    pairs = [abs(ax[L] @ ax[L + dk]) for L in Ls if L + dk in ax]
    return np.mean(pairs) if pairs else np.nan


def main():
    rows = []
    curves = {}
    for name, sub in CONCEPTS:
        base = f"{ROOT}/{sub}"
        if not os.path.exists(f"{base}/meta.npz"):
            continue
        W, D, U = axes_per_layer(base)
        for lbl, ax in [("w_res", W), ("delta", D), ("u", U)]:
            r = [rho(ax, dk) for dk in DKS]
            curves[(name, lbl)] = r
            rows.append({"concept": name, "axis": lbl,
                         **{f"rho_dk{dk}": round(r[i], 4) for i, dk in enumerate(DKS)}})
    df = pd.DataFrame(rows)
    os.makedirs("data/analysis/27c", exist_ok=True)
    df.to_csv("data/analysis/27c/rotation_decay_by_concept.csv", index=False)
    print("wrote data/analysis/27c/rotation_decay_by_concept.csv")

    # Figure 1: decay curves for one concept (scaffold)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lbl, col, disp in [("w_res", RED, "reading $w_{\\mathrm{LDA}}$"), ("delta", GREEN, "writing $\\Delta\\mu$"), ("u", BLUE, "use $u$")]:
        ax.plot(DKS, curves[("α/β scaffold", lbl)], "-o", ms=4, color=col, label=disp)
    ax.set_xlabel("layer displacement $\\Delta k$")
    ax.set_ylabel("$\\rho(\\Delta k) = |\\cos(v^{(\\ell)}, v^{(\\ell+\\Delta k)})|$")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("docs/figures/rotation_decay_curves.png", dpi=200)
    print("wrote docs/figures/rotation_decay_curves.png")

    # Figure 2: bars of rho at dk=4 per concept per axis
    names = [n for n, _ in CONCEPTS if (n, "u") in curves]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for i, (lbl, col, disp) in enumerate([("w_res", RED, "reading $w_{\\mathrm{LDA}}$"), ("delta", GREEN, "writing $\\Delta\\mu$"), ("u", BLUE, "use $u$")]):
        vals = [curves[(n, lbl)][3] for n in names]  # dk=4
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=disp)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("rotation decay $\\rho(\\Delta k{=}4)$")
    ax.set_ylim(0, 0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, title="axis")
    fig.tight_layout()
    fig.savefig("docs/figures/rotation_decay_bars.png", dpi=200)
    print("wrote docs/figures/rotation_decay_bars.png")


if __name__ == "__main__":
    main()
