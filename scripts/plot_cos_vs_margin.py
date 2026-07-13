"""
Bar chart (Section 3.1.2): w_LDA margin-flip per concept, ordered by cos(w_LDA, u).
Shows the reading direction is causally inert where it is orthogonal to use (alpha/beta)
and causal where it aligns with use (particles). cos annotated on each bar.
Grammar excluded (dirty null); probe excluded (no direction steering).

Verified values (zone L20-30 cos from per-concept dumps; margin from steering sweeps):
  alpha/beta scaffold cos=0.021, raw+suffix cos=0.032 (real, from raw_suffix dump).

Output: docs/figures/cos_wu_vs_margin.png
"""
import os
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

# (label, cos(w,u), w margin-flip, group)  -- sorted by cos
DATA = [
    ("α/β scaffold", 0.021, 0.075, "ab"),
    ("α/β raw+suffix", 0.032, 0.062, "ab"),
    ("n/p", 0.118, 0.969, "particle"),
    ("n/γ", 0.147, 1.000, "particle"),
    ("e/n", 0.148, 0.984, "particle"),
    ("γ/p", 0.149, 1.000, "particle"),
    ("e/γ", 0.153, 1.000, "particle"),
    ("e/p", 0.164, 1.000, "particle"),
]


def main():
    labels = [d[0] for d in DATA]
    cos = [d[1] for d in DATA]
    margin = [d[2] for d in DATA]
    colors = ["#d62728" if d[3] == "ab" else "#1f77b4" for d in DATA]
    x = range(len(DATA))

    fig, ax = plt.subplots(figsize=(14, 3.8))
    bars = ax.bar(x, margin, color=colors, width=0.7, zorder=3, edgecolor="white", linewidth=0.6)
    for i, (b, c) in enumerate(zip(bars, cos)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"cos={c:.2f}", ha="center", va="bottom", fontsize=19)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=15)
    ax.set_ylabel("$w_{\\mathrm{LDA}}$ margin flip")
    ax.set_xlabel("concept")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d62728"),
               plt.Rectangle((0, 0), 1, 1, color="#1f77b4")]
    ax.legend(handles, ["α/β", "particles"], fontsize=15, loc="upper left",
              bbox_to_anchor=(0.0, 0.97), framealpha=0.9)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    out = "docs/figures/cos_wu_vs_margin.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
