#!/usr/bin/env python3
"""
diag_necessity_density_curve.py — Necessity vs Edge Density curve
==================================================================
Shows how necessity grows as graph density decreases and where
community structure breaks down.

Data sources:
  - data/diagnostics/sparsification/sweep_results.csv  (25 configs, proxy metrics)
  - 3 real necessity anchor points from actual runs

Real necessity anchors:
  B1-v2 (beta proxy method):  density~4.41, necessity=0.1042
  B1-gradient (k=all,t=0.01): density~10.56, necessity=0.3540
  B1-sparsified (k=3,t=0.05): density~1.45,  necessity=0.4375

Community breakdown anchors (VW-only Louvain, actual analysis):
  B1-v2:          86 nodes, 379 edges → 7 communities (GOOD)
  B1-gradient:   137 nodes, 1052 edges →12 communities (4 giants; BAD)
  B1-sparsified:  58 nodes,  84 edges → 17 communities (over-fragmented; BAD)

Usage:
  python scripts/diag_necessity_density_curve.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_DIR = Path("data/diagnostics/necessity_density")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CSV = Path("data/diagnostics/sparsification/sweep_results.csv")

# ── Real necessity anchor points ────────────────────────────────────────────
# (label, density=edges/nodes, necessity, n_nodes, n_edges, n_communities, marker_style)
REAL_POINTS = [
    dict(label="B1-v2\n(β proxy)", density=379/86,   necessity=0.1042,
         n_nodes=86,  n_edges=379,  n_comm=7,  marker="^", color="#888888",
         method="beta"),
    dict(label="B1-gradient\n(k=all, t=0.01)", density=1447/137, necessity=0.3540,
         n_nodes=137, n_edges=1447, n_comm=12, marker="s", color="#e67e22",
         method="gradient"),
    dict(label="B1-sparsified\n(k=3, t=0.05)", density=84/58,   necessity=0.4375,
         n_nodes=58,  n_edges=84,   n_comm=17, marker="*", color="#c0392b",
         method="gradient"),
]

# ── Community breakdown thresholds ───────────────────────────────────────────
GIANT_THRESH   = 30.0   # largest community pct — above = giant cluster problem
SINGLETON_THRESH = 10   # count — above = fragmentation problem
COMM_N_THRESH  = 4      # min meaningful communities

K_COLORS = {
    "all": "#2c3e50",
    "8":   "#2980b9",
    "5":   "#27ae60",
    "3":   "#c0392b",
    "2":   "#8e44ad",
}
K_LABELS = {
    "all": "k=all (137 nodes)",
    "8":   "k=8 (108 nodes)",
    "5":   "k=5 (79 nodes)",
    "3":   "k=3 (48 nodes)",
    "2":   "k=2 (32 nodes)",
}

# ─────────────────────────────────────────────────────────────────────────────
def load_sweep():
    df = pd.read_csv(SWEEP_CSV)
    df["k_per_layer"] = df["k_per_layer"].astype(str)
    df["density"] = df["n_edges"] / df["n_nodes"]
    # community quality score: 0=good, higher=worse
    df["comm_quality"] = (
        (df["largest_pct"] > GIANT_THRESH).astype(int) +
        (df["n_singletons"] > SINGLETON_THRESH).astype(int) +
        (df["n_communities"] < COMM_N_THRESH).astype(int)
    )
    df["comm_ok"] = df["comm_quality"] == 0
    return df


def mean_abs_effect_by_k(df):
    """Return mean_abs_effect per k level (it doesn't vary with threshold)."""
    return df.groupby("k_per_layer")["mean_abs_effect"].first()


def build_figure(df):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Necessity vs Edge Density — Sparsification Sweep\n"
        "multilingual_circuits_b1 (gradient × activation attribution)",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.08, right=0.97, top=0.91, bottom=0.07)

    ax_main   = fig.add_subplot(gs[0, :])   # top full-width: necessity curve
    ax_comm   = fig.add_subplot(gs[1, 0])   # bottom-left: n_communities
    ax_giant  = fig.add_subplot(gs[1, 1])   # bottom-right: largest_pct / singletons

    # ── Panel 1: Necessity (proxy + real points) vs Density ──────────────────
    ax = ax_main

    # Plot mean_abs_effect as horizontal line segment per k level
    for k_str in ["all", "8", "5", "3", "2"]:
        sub = df[df["k_per_layer"] == k_str].sort_values("density")
        eff_val = sub["mean_abs_effect"].iloc[0]
        d_min, d_max = sub["density"].min(), sub["density"].max()
        col = K_COLORS[k_str]
        ax.hlines(eff_val, d_min, d_max, colors=col, linewidths=2.0, alpha=0.7,
                  linestyle="--", zorder=2)
        # Individual dots per (k, threshold) point
        comm_ok_mask = sub["comm_ok"]
        ax.scatter(sub["density"][~comm_ok_mask], sub["mean_abs_effect"][~comm_ok_mask],
                   color=col, s=50, marker="o", alpha=0.5, zorder=3)
        ax.scatter(sub["density"][comm_ok_mask], sub["mean_abs_effect"][comm_ok_mask],
                   color=col, s=80, marker="o", alpha=1.0, zorder=4,
                   edgecolors="black", linewidths=0.8)

    # Overlay real necessity points
    for rp in REAL_POINTS:
        ax.scatter(rp["density"], rp["necessity"],
                   color=rp["color"], marker=rp["marker"],
                   s=200 if rp["marker"]=="*" else 120,
                   zorder=6, edgecolors="black", linewidths=1.2,
                   label=rp["label"])
        ax.annotate(
            f"  {rp['necessity']:.1%}",
            xy=(rp["density"], rp["necessity"]),
            fontsize=8.5, va="center", color=rp["color"],
            fontweight="bold"
        )

    # Fit line through real necessity points (gradient only)
    grad_pts = [(rp["density"], rp["necessity"]) for rp in REAL_POINTS
                if rp["method"] == "gradient"]
    if len(grad_pts) >= 2:
        xg = np.array([p[0] for p in grad_pts])
        yg = np.array([p[1] for p in grad_pts])
        coeffs = np.polyfit(np.log(xg + 1e-6), yg, 1)
        x_fit = np.linspace(0.1, 12, 200)
        y_fit = np.polyval(coeffs, np.log(x_fit + 1e-6))
        ax.plot(x_fit, y_fit, color="#c0392b", linewidth=1.0, linestyle=":",
                alpha=0.6, label="log-linear fit (gradient runs)")

    ax.set_xlabel("Edge density  (VW edges / feature nodes)", fontsize=11)
    ax.set_ylabel("Necessity proxy\n(mean |effect| of circuit features)", fontsize=11)
    ax.set_title("Panel A — Necessity grows as density decreases", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.3, 12.0)
    ax.set_ylim(0.05, 1.15)
    ax.axhline(0.4375, color="#c0392b", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axhline(0.3540, color="#e67e22", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axhline(0.1042, color="#888888", linestyle=":", linewidth=0.8, alpha=0.4)

    # Shade "good community" zone
    ax.axvspan(0.8, 3.5, alpha=0.06, color="green", zorder=0,
               label="good community zone (proxy)")

    # Legend
    k_handles = [
        Line2D([0], [0], color=K_COLORS[k], linewidth=2, linestyle="--",
               label=K_LABELS[k])
        for k in ["all", "8", "5", "3", "2"]
    ]
    real_handles = [
        plt.scatter([], [], color=rp["color"], marker=rp["marker"],
                    s=100, label=rp["label"])
        for rp in REAL_POINTS
    ]
    good_patch = mpatches.Patch(color="green", alpha=0.2, label="good comm zone")
    open_circle = Line2D([0],[0], marker="o", color="#555", linestyle="None",
                         markersize=6, alpha=0.4, label="community: FAIL")
    filled_circle = Line2D([0],[0], marker="o", color="#555", linestyle="None",
                           markersize=7, markeredgecolor="black",
                           label="community: PASS")
    ax.legend(
        handles=k_handles + real_handles + [good_patch, open_circle, filled_circle],
        loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85
    )
    ax.grid(True, alpha=0.3)

    # ── Panel 2: n_communities vs density ────────────────────────────────────
    ax = ax_comm
    for k_str in ["all", "8", "5", "3", "2"]:
        sub = df[df["k_per_layer"] == k_str].sort_values("density")
        col = K_COLORS[k_str]
        ax.plot(sub["density"], sub["n_communities"], color=col, linewidth=1.5,
                marker="o", markersize=5, label=K_LABELS[k_str], alpha=0.85)

    # Real anchor points
    for rp in REAL_POINTS:
        ax.scatter(rp["density"], rp["n_comm"],
                   color=rp["color"], marker=rp["marker"],
                   s=150 if rp["marker"]=="*" else 90,
                   zorder=6, edgecolors="black", linewidths=1.2)
        ax.annotate(f"  {rp['n_comm']}",
                    xy=(rp["density"], rp["n_comm"]),
                    fontsize=8, va="center", color=rp["color"], fontweight="bold")

    # Shade acceptable zone
    ax.axhspan(4, 15, alpha=0.06, color="green", zorder=0)
    ax.axhline(4,  color="green", linewidth=0.8, linestyle="--", alpha=0.5,
               label="min useful (4)")
    ax.axhline(15, color="red",   linewidth=0.8, linestyle="--", alpha=0.5,
               label="over-fragmented (15)")

    ax.set_xlabel("Edge density (VW edges / nodes)", fontsize=10)
    ax.set_ylabel("# Louvain communities", fontsize=10)
    ax.set_title("Panel B — Community count vs density", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.3, 12.0)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Giant cluster % and singleton count vs density ──────────────
    ax = ax_giant
    ax2 = ax.twinx()

    for k_str in ["all", "8", "5", "3", "2"]:
        sub = df[df["k_per_layer"] == k_str].sort_values("density")
        col = K_COLORS[k_str]
        ax.plot(sub["density"], sub["largest_pct"], color=col, linewidth=1.5,
                marker="o", markersize=5, alpha=0.85)
        ax2.plot(sub["density"], sub["n_singletons"], color=col, linewidth=1.0,
                 linestyle=":", marker="x", markersize=4, alpha=0.55)

    # Threshold lines
    ax.axhline(GIANT_THRESH, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(25, color="orange", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(11.5, GIANT_THRESH + 0.5, f"{GIANT_THRESH:.0f}%", fontsize=7.5,
            color="red", ha="right")

    # Real anchor points on largest_pct
    b1v2_pct    = 100 * 19 / 86    # B1-v2: largest community was 19/86 ≈ 22%
    b1grad_pct  = 26.3              # from sweep data
    b1spar_pct  = 100 * 14 / 58    # C1 = 24/58 in UI, but VW-only: largest of 17 comms
    # for VW-only: largest community in 17-comm result — we know from analysis it's hard to infer,
    # use as unknown; mark gradient and b1-v2 anchors only
    for rp, pct in zip(REAL_POINTS, [b1v2_pct, b1grad_pct, None]):
        if pct is None:
            continue
        ax.scatter(rp["density"], pct, color=rp["color"], marker=rp["marker"],
                   s=100, zorder=6, edgecolors="black", linewidths=1.2)

    ax.set_xlabel("Edge density (VW edges / nodes)", fontsize=10)
    ax.set_ylabel("Largest community %", fontsize=10, color="black")
    ax2.set_ylabel("# singletons", fontsize=9, color="#666666")
    ax.set_title("Panel C — Community quality breakdown", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.3, 12.0)
    ax.set_ylim(0, 110)

    # Annotate breakdown regions
    ax.fill_between([-0.3, 1.0], [0, 0], [110, 110], alpha=0.06, color="orange",
                    zorder=0, label="over-fragmented region")
    ax.fill_between([5.0, 12.0], [0, 0], [110, 110], alpha=0.06, color="red",
                    zorder=0, label="giant-cluster region")
    ax.fill_between([1.0, 5.0], [0, 0], [110, 110], alpha=0.06, color="green",
                    zorder=0, label="viable zone")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Summary annotation ─────────────────────────────────────────────────
    summary = (
        "Key finding:\n"
        "• Necessity grows monotonically as density falls\n"
        "• Giant-cluster problem: density > ~5 (k=all)\n"
        "• Fragmentation problem: density < ~1.5 (k=3, t=0.05)\n"
        "• Sweet spot (sweep proxy): density ≈ 1.5–3.5\n"
        "• B1-v2 (β, density=4.4): 7 good VW communities but necessity only 10%\n"
        "• B1-gradient (density=10.6): 12 communities, 4 giants, necessity 35%\n"
        "• B1-sparsified (density=1.45): 17 fragmented comms, necessity 44% (best)\n"
        "• Actual community breakdown threshold lower than sweep predicted"
    )
    fig.text(0.02, 0.01, summary, fontsize=7.5, family="monospace",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="#f8f8f8", alpha=0.8,
                       edgecolor="#cccccc"))

    return fig


def print_table(df):
    print("\n" + "=" * 78)
    print("NECESSITY VS DENSITY SUMMARY TABLE")
    print("=" * 78)
    print(f"{'Config':<20} {'Density':>8} {'Nodes':>6} {'Edges':>6} "
          f"{'Comms':>6} {'Giant%':>7} {'Sings':>6} {'MeanAbs':>8} {'GoodComm':>9}")
    print("-" * 78)
    for _, row in df.sort_values("density", ascending=False).iterrows():
        k = row["k_per_layer"]
        t = row["vw_threshold"]
        tag = "★" if row["both_good"] else (" " if row["comm_ok"] else "✗")
        print(f"k={k:<4} t={t:<5} {tag}  "
              f"{row['density']:>7.2f}  {row['n_nodes']:>5}  {row['n_edges']:>5}  "
              f"{row['n_communities']:>5}  {row['largest_pct']:>6.1f}  "
              f"{row['n_singletons']:>5}  {row['mean_abs_effect']:>8.4f}  "
              f"{'YES' if row['comm_ok'] else 'NO':>9}")

    print("\n--- Real necessity anchors (actual runs) ---")
    for rp in REAL_POINTS:
        print(f"  {rp['label'].replace(chr(10),' '):30s}  "
              f"density={rp['density']:5.2f}  necessity={rp['necessity']:.4f}  "
              f"VW-comms={rp['n_comm']}")

    print("\n--- Breakdown summary ---")
    print("  Giant-cluster problem (largest_pct > 30%):")
    bad = df[df["largest_pct"] > 30]
    print(f"    {len(bad)}/25 configs affected (density range: "
          f"{bad['density'].min():.2f}–{bad['density'].max():.2f})")

    print("  Over-fragmentation (n_singletons > 10% nodes):")
    frag = df[df["n_singletons"] > df["n_nodes"] * 0.10]
    print(f"    {len(frag)}/25 configs affected (density range: "
          f"{frag['density'].min():.2f}–{frag['density'].max():.2f})")

    print("  Both-good configs (sweep proxy):")
    good = df[df["both_good"]]
    for _, row in good.iterrows():
        print(f"    k={row['k_per_layer']}, t={row['vw_threshold']}: "
              f"density={row['density']:.2f}, comms={row['n_communities']}, "
              f"necessity_proxy={row['mean_abs_effect']:.4f}")
    print("=" * 78)


def main():
    df = load_sweep()
    print_table(df)
    fig = build_figure(df)
    out = OUT_DIR / "necessity_density_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
