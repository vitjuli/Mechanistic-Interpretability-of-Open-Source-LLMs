"""
Norm-to-stream ratio by depth (Section 3.3): r_v = ||v|| / mean||h|| for the writing direction
Delta_mu (grows into the stream) and the use direction u (shrinks within it), on four concepts.

Input : data/analysis/runD_v2/{B1_alpha_beta,raw_suffix,B1_grammar_number,
        particles4_binary/electron_vs_photon}/field_dump/{res,grad}_L*.npy, meta.npz
Output: docs/figures/norm_ratio_by_layer.png
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scienceplots  # noqa: F401
plt.style.use(["science", "ieee", "grid", "no-latex"])
plt.rcParams.update({
    "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
    "xtick.labelsize": 15, "ytick.labelsize": 20, "legend.fontsize": 16,
    "lines.linewidth": 2.6, "lines.markersize": 6, "axes.linewidth": 1.1,
    "legend.title_fontsize": 14,
})

CONCEPTS = [
    ("α/β scaffold", "B1_alpha_beta/field_dump", "#1f77b4"),
    ("α/β raw+suffix", "raw_suffix/field_dump", "#ff7f0e"),
    ("particles", "particles4_binary/electron_vs_photon/field_dump", "#d62728"),
]
ROOT = "data/analysis/runD_v2"


def ratios(base):
    y = np.load(f"{base}/meta.npz")["y"].astype(int)
    layers, rd, ru = [], [], []
    for L in range(36):
        rf, gf = f"{base}/res_L{L:02d}.npy", f"{base}/grad_L{L:02d}.npy"
        if not (os.path.exists(rf) and os.path.exists(gf)):
            continue
        H = np.load(rf).astype(np.float64)
        G = np.load(gf).astype(np.float64)
        delta = H[y == 1].mean(0) - H[y == 0].mean(0)
        u = (G[y == 0].mean(0) + G[y == 1].mean(0)) / 2
        hbar = np.linalg.norm(H, axis=1).mean()
        layers.append(L)
        rd.append(np.linalg.norm(delta) / hbar)
        ru.append(np.linalg.norm(u) / hbar)
    return np.array(layers), np.array(rd), np.array(ru)


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 4.2))
    for name, sub, col in CONCEPTS:
        base = f"{ROOT}/{sub}"
        if not os.path.exists(f"{base}/meta.npz"):
            print(f"skip {name}")
            continue
        L, rd, ru = ratios(base)
        a1.plot(L, rd, "-o", ms=3, color=col, label=name)
        a2.plot(L, ru, "-o", ms=3, color=col, label=name)
    for ax in (a1, a2):
        ax.axvspan(19, 24, color="gray", alpha=0.12)  # decision zone L19-24 (App B)
        ax.set_xlabel("layer")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 35)
    a1.set_ylabel("$r_{\\Delta\\mu} = \\|\\Delta\\mu\\| \\,/\\, \\overline{\\|h\\|}$")
    a1.legend(fontsize=15, loc="upper left")
    a2.set_ylabel("$r_{u} = \\|u\\| \\,/\\, \\overline{\\|h\\|}$")
    a2.legend(fontsize=15, loc="upper right")
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    out = "docs/figures/norm_ratio_by_layer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
