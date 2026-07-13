"""
Active-feature span (feature-span chapter): can the read / write / use directions be
reconstructed from the directions of features that actually fire? Greedy least-squares capture
per layer for w_res, u, delta vs a random null. Only delta (written) rises above null;
w_res (read) sits at null and is not in the dictionary's reach.

Input : data/analysis/runD_v2/active_span_all/active_span_triad.csv
        (cap_wres, cap_u, cap_delta, cap_random_null, n_active per layer)
Output: docs/figures/feature_span_capture.png
"""
import os, numpy as np, pandas as pd
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
RED, BLUE, GREEN, GRAY = "#b22222", "#1f77b4", "#2ca02c", "#bbbbbb"


def main():
    d = pd.read_csv("data/analysis/runD_v2/active_span_all/active_span_triad.csv").sort_values("layer")
    L = d.layer.values
    x = np.arange(len(L))
    w = 0.2
    fig, ax = plt.subplots(figsize=(15.0, 4.6))
    # decision zone L19-24 (App B)
    zlo, zhi = list(L).index(19), list(L).index(24)
    ax.axvspan(zlo - 0.5, zhi + 0.5, color="gray", alpha=0.12, zorder=0)
    ax.bar(x - 1.5 * w, d.cap_wres, w, color=RED, label="$w_{\\mathrm{res}}$ (read)")
    ax.bar(x - 0.5 * w, d.cap_u, w, color=BLUE, label="$u$ (used)")
    ax.bar(x + 0.5 * w, d.cap_delta, w, color=GREEN, label="$\\delta = \\mu_\\beta - \\mu_\\alpha$ (written)")
    ax.bar(x + 1.5 * w, d.cap_random_null, w, color=GRAY, label="random null")
    for xi, v in zip(x, d.cap_delta):
        ax.text(xi + 0.5 * w, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=15, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l:02d}" for l in L], fontsize=11, rotation=90)
    ax.set_ylabel("capture fraction")
    ax.set_xlabel("layer")
    ax.legend(fontsize=15, loc="upper left", ncol=2)
    ax.set_xlim(-0.6, len(L) - 0.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    out = "docs/figures/feature_span_capture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
