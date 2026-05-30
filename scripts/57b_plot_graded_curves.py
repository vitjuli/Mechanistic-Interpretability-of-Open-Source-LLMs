"""
57b_plot_graded_curves.py — Visualisation of graded patching IIA(α) curves.

Plots IIA as a function of fraction α (of pool size) patched, separately for
each pool (L24, L18, random control). Includes error bars over random subsets.

Three diagnostic curves expected:
  - L24 (β-carrier, hypothesised redundant): SATURATING curve, plateau at α≈0.1
  - L18 (α-carrier, hypothesised population): near-LINEAR curve
  - random control: near-zero across all α (specificity)

Usage:
  python3 scripts/57b_plot_graded_curves.py \\
    --in_dir data/analysis/runD_v2/graded_patching
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, default=None,
                    help="Output PNG (default: in_dir/graded_patching_curve.png)")
    args = ap.parse_args()

    raw = pd.read_csv(args.in_dir / "graded_patching_raw.csv")
    summary = pd.read_csv(args.in_dir / "graded_patching_summary.csv")

    out_path = args.out_path or (args.in_dir / "graded_patching_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Color/marker per pool
    pool_styles = {
        "L24":         {"color": "#d62728", "marker": "o", "label": "L24 (β-carrier)"},
        "L18":         {"color": "#1f77b4", "marker": "s", "label": "L18 (α-carrier)"},
        "random_l25":  {"color": "#7f7f7f", "marker": "^", "label": "Random L25 control"},
        "random_l11":  {"color": "#999999", "marker": "v", "label": "Random L11 control"},
    }

    for pool in summary["pool"].unique():
        sub = summary[summary["pool"] == pool].sort_values("alpha")
        style = pool_styles.get(pool, {"color": "k", "marker": "x", "label": pool})
        # Scatter individual subsets
        raw_sub = raw[raw["pool"] == pool]
        ax.scatter(raw_sub["alpha"], raw_sub["iia"],
                   c=style["color"], alpha=0.25, s=25, zorder=2)
        # Mean line with error bars
        ax.errorbar(sub["alpha"], sub["mean_iia"], yerr=sub["std_iia"],
                    color=style["color"], marker=style["marker"], markersize=9,
                    linewidth=2.5, capsize=4, label=style["label"], zorder=5)

    # Reference: linear prediction (slope = IIA(α=1) / 1)
    for pool in summary["pool"].unique():
        sub = summary[summary["pool"] == pool]
        full_iia = float(sub[sub["alpha"] == sub["alpha"].max()]["mean_iia"].iloc[0])
        if pool == "L24":  # show linear ref only for L24
            ax.plot([0, 1], [0, full_iia], '--', color=pool_styles[pool]["color"],
                    alpha=0.35, label=f"Linear ref ({pool}): if uniform structure", zorder=1)

    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
    ax.set_xlabel("α: fraction of pool features patched (random subset)", fontsize=12)
    ax.set_ylabel("IIA (interchange intervention accuracy)", fontsize=12)
    ax.set_title("Graded patching: IIA(α) — population vs redundant coding\n"
                 "(scatter = individual subsets; lines = mean±std across 5 subsets)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_xticks([0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0])

    textstr = (
        "Reading the curves:\n"
        "• SATURATION (plateau at low α) → REDUNDANT coding (few features carry most state)\n"
        "• LINEAR rise → POPULATION coding (each feature contributes ~equally)\n"
        "• Near-zero across α → no carrier in this pool (control behaviour)"
    )
    fig.text(0.5, -0.06, textstr, ha='center', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")

    # Also compute & print "saturation diagnostic" per pool
    print("\n=== Saturation diagnostic per pool ===")
    print(f"{'pool':>14} {'IIA(α=0.05)':>12} {'IIA(α=0.10)':>12} {'IIA(α=0.25)':>12} {'IIA(α=1.0)':>12} {'sat_α=0.10':>11}")
    for pool in summary["pool"].unique():
        sub = summary[summary["pool"] == pool].sort_values("alpha")
        full = float(sub.iloc[-1]["mean_iia"])
        def get(a):
            try:
                return float(sub[sub["alpha"].sub(a).abs() < 0.01]["mean_iia"].iloc[0])
            except (IndexError, KeyError):
                return float("nan")
        i05 = get(0.05); i10 = get(0.10); i25 = get(0.25); i100 = full
        sat = i10 / i100 if i100 > 0 else float("nan")
        print(f"{pool:>14} {i05:>12.4f} {i10:>12.4f} {i25:>12.4f} {i100:>12.4f} {sat:>11.2f}")

    print("\nInterpretation:")
    print("  sat_α=0.10 ≈ 0.9 → REDUNDANT coding (10% of features ≈ 90% of effect)")
    print("  sat_α=0.10 ≈ 0.1 → POPULATION coding (10% of features ≈ 10% of effect)")
    print("  sat_α=0.10 between → mixed / threshold structure")


if __name__ == "__main__":
    main()
