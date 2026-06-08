"""
4 panels per experiment for §6 figures:
  j100 — u deep-dive (AUC profile + intact profile + cos + β-rank)
  j101 — head dissociation (heatmap of 4 conditions + head scatter)
  j102 — multi-layer w_res (single-layer profile + composed sweep)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "j100_j101_j102_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 180, "savefig.bbox": "tight",
})

# ───────────────────────────────────────────────────────────────────────────
# j100 — u DEEP-DIVE
# ───────────────────────────────────────────────────────────────────────────
ll = pd.read_csv(ROOT / "data/analysis/runD_v2/u_deepdive/u_logitlens.csv")
sw = pd.read_csv(ROOT / "data/analysis/runD_v2/u_deepdive/usage_deepdive.csv")
# best c per (layer)
best_sw = sw.loc[sw.groupby("layer")["intact_u"].idxmax()].reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# (A) AUC profile: u vs w_res
ax = axes[0, 0]
ax.plot(ll["layer"], ll["auc_wres"], "-o", color="#2ca02c", lw=2, ms=5,
        label="AUC along $w_{res}$ (readable)")
ax.plot(ll["layer"], ll["auc_u"], "-o", color="#d62728", lw=2, ms=5,
        label="AUC along $u$ (used)")
ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)
ax.axvline(21, color="purple", ls=":", lw=1.5, alpha=0.7, label="L21 commitment")
ax.set_xlabel("Layer")
ax.set_ylabel("AUC of α/β separation")
ax.set_title("(A) Concept decodability per layer\n"
             "w_res decodes throughout; u becomes decoder only at L19+ (steep jump)")
ax.set_ylim(0.3, 1.05)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.grid(alpha=0.2)

# (B) Steering intact-flip per layer
ax = axes[0, 1]
ax.plot(best_sw["layer"], best_sw["intact_u"], "-o", color="#d62728", lw=2.5, ms=5,
        label="steering along $u$")
ax.plot(best_sw["layer"], best_sw["intact_wres"], "-o", color="#2ca02c", lw=2, ms=4,
        label="steering along $w_{res}$")
ax.plot(best_sw["layer"], best_sw["intact_random"], "-", color="#888", lw=1.5,
        label="random null")
ax.axvline(21, color="purple", ls=":", lw=1.5, alpha=0.7)
ax.set_xlabel("Layer")
ax.set_ylabel("best intact-flip rate (over c)")
ax.set_title("(B) Behavioural lever per layer\n"
             "u jumps to 1.0 at L23; w_res stays in null band everywhere")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.2)

# (C) cos(u, w_res) per layer
ax = axes[1, 0]
ax.plot(ll["layer"], ll["cos_uw"], "-o", color="#9467bd", lw=2, ms=5,
        label="|cos(u, w_res)|")
ax.axhline(0.016, color="k", ls="--", lw=1, alpha=0.7,
           label=r"random baseline $\mathbb{E}|\cos|$")
ax.axhline(0.039, color="orange", ls=":", lw=1, alpha=0.7,
           label="random p95")
ax.set_xlabel("Layer")
ax.set_ylabel("|cos(u, $w_{res}$)|")
ax.set_title("(C) Direction-level orthogonality\n"
             r"All layers $|\cos| < 0.045 \approx$ random baseline in $d=2560$")
ax.set_ylim(0, 0.06)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.2)

# (D) β-rank along u (logit-lens)
ax = axes[1, 1]
log_rank = np.log10(np.maximum(ll["beta_rank"].values, 1))
ax.plot(ll["layer"], log_rank, "-o", color="#d62728", lw=2, ms=5,
        label="β-rank along $u$")
log_rank_a = np.log10(np.maximum(ll["alpha_rank"].values, 1))
ax.plot(ll["layer"], log_rank_a, "-o", color="#2ca02c", lw=2, ms=5,
        label="α-rank along $u$")
ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
ax.axvline(21, color="purple", ls=":", lw=1.5, alpha=0.7)
ax.set_xlabel("Layer")
ax.set_ylabel(r"$\log_{10}$(rank in vocab)")
ax.set_title("(D) Logit-lens along $u$ — what does $u$ point to?\n"
             "β-rank drops to 0 from L23+ (u = β direction in unembedding)")
ax.set_ylim(-0.3, 6)
ax.legend(loc="center right", fontsize=9, frameon=False)
ax.grid(alpha=0.2)

plt.suptitle("j100 — Usage-direction deep-dive across 36 layers (forced regime)",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "j100_u_deepdive_4panels.png")
plt.close()
print(f"saved {OUT}/j100_u_deepdive_4panels.png")


# ───────────────────────────────────────────────────────────────────────────
# j101 — HEAD DISSOCIATION
# ───────────────────────────────────────────────────────────────────────────
hd = pd.read_csv(ROOT / "data/analysis/runD_v2/head_dissoc/head_dissociation.csv")
hs = pd.read_csv(ROOT / "data/analysis/runD_v2/head_dissoc/head_scores.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (A) 4x3 heatmap-style table
ax = axes[0]
rows = ["global·ablate", "global·negate", "mid[12-28]·ablate", "mid[12-28]·negate"]
cols = ["answer-writers", "axis-writers", "random null"]
data = np.array([
    [hd.iloc[0]["answer_writers"], hd.iloc[0]["axis_writers"], hd.iloc[0]["random_p95"]],
    [hd.iloc[1]["answer_writers"], hd.iloc[1]["axis_writers"], hd.iloc[1]["random_p95"]],
    [hd.iloc[2]["answer_writers"], hd.iloc[2]["axis_writers"], hd.iloc[2]["random_p95"]],
    [hd.iloc[3]["answer_writers"], hd.iloc[3]["axis_writers"], hd.iloc[3]["random_p95"]],
])
# Color cells: orange if above null, grey if at/below
im = ax.imshow(data, cmap="Oranges", vmin=0, vmax=0.5, aspect="auto")
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold" if data[i,j] > 0.3 else "normal",
                color="white" if data[i,j] > 0.3 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=10)
ax.set_yticks(range(4)); ax.set_yticklabels(rows, fontsize=10)
ax.set_title("(A) Head-intervention map (intact-flip, forced n=24)\n"
             "Only mid[12-28]·negate of answer-writers fires above null")

# (B) Where the heads live
ax = axes[1]
top_ans = set((int(r.layer), int(r.head)) for r in hs.nlargest(10, "answer_writer_score").itertuples())
top_axis = set((int(r.layer), int(r.head)) for r in hs.nlargest(10, "axis_writer_score").itertuples())
both = top_ans & top_axis
only_ans = top_ans - top_axis
only_axis = top_axis - top_ans

for L, h in only_ans:
    ax.scatter(h, L, s=120, color="#1f77b4", zorder=3, edgecolors="white", linewidth=1)
for L, h in only_axis:
    ax.scatter(h, L, s=120, color="#ff7f0e", zorder=3, edgecolors="white", linewidth=1)
for L, h in both:
    ax.scatter(h, L, s=120, color="#9467bd", zorder=3, edgecolors="white", linewidth=1)

ax.set_xlabel("Head index")
ax.set_ylabel("Layer")
ax.set_title(f"(B) Where top-10 heads of each type live\n"
             f"answer-only ({len(only_ans)}), axis-only ({len(only_axis)}), "
             f"BOTH ({len(both)}/10 overlap)")
ax.set_xlim(-1, 32)
ax.invert_yaxis()
ax.grid(alpha=0.2)
# Legend
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="answer-writer only"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=10, label="axis-writer only"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#9467bd", markersize=10, label=f"both ({len(both)}/10)"),
], loc="lower left", fontsize=9, frameon=False)

plt.suptitle("j101 — Head dissociation: answer-writers vs axis-writers (forced, intact-flip)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "j101_head_dissociation.png")
plt.close()
print(f"saved {OUT}/j101_head_dissociation.png")


# ───────────────────────────────────────────────────────────────────────────
# j102 — MULTI-LAYER w_res
# ───────────────────────────────────────────────────────────────────────────
sl = pd.read_csv(ROOT / "data/analysis/runD_v2/ml_wres/single_layer_sweep.csv")
cs = pd.read_csv(ROOT / "data/analysis/runD_v2/ml_wres/composed_sweep.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (A) Single-layer profile c=2 and c=8
ax = axes[0]
for c, color in [(2.0, "#2ca02c"), (8.0, "#d62728")]:
    sub = sl[sl.c == c].sort_values("layer")
    ax.plot(sub["layer"], sub["intact_wres"], "-o", color=color, lw=2, ms=4,
            label=f"$w_{{res}}$ steering at c={c:g}")
    ax.plot(sub["layer"], sub["intact_random"], "--", color=color, lw=1, alpha=0.5,
            label=f"random null c={c:g}")
ax.axhline(0.1, color="k", ls=":", lw=1, alpha=0.5, label="null threshold (0.1)")
ax.set_xlabel("Layer (single-layer push)")
ax.set_ylabel("intact-flip rate")
ax.set_title("(A) Single-layer $w_{res}$ steering profile\n"
             "w_res never exceeds null at any layer × any c tested")
ax.set_ylim(0, 0.35)
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.grid(alpha=0.2)

# (B) Composed sweep
ax = axes[1]
ax.plot(cs["c_each"], cs["intact_wres"], "-o", color="#2ca02c", lw=3, ms=8,
        label="composed $w_{res}$ (push at ALL 36 layers)")
ax.plot(cs["c_each"], cs["intact_random"], "--o", color="#888", lw=1.5, ms=5,
        label="composed random direction")
ax.plot(cs["c_each"], cs["intact_shuffled"], ":o", color="#ff7f0e", lw=1.5, ms=5,
        label="composed shuffled-label axes")
ax.axhline(0.1, color="k", ls=":", lw=1, alpha=0.5, label="null threshold")
ax.set_xlabel("$c_{each}$ (amplitude per layer)")
ax.set_ylabel("intact-flip rate")
ax.set_title("(B) Composed across stack — does maintaining $w_{res}$ help?\n"
             "All amplitudes invalid (null > 0.1); no window where composed beats null")
ax.set_ylim(0, 0.25)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.grid(alpha=0.2)

plt.suptitle("j102 — Multi-layer $w_{res}$ steering (single + composed)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "j102_multilayer_wres.png")
plt.close()
print(f"saved {OUT}/j102_multilayer_wres.png")

print("\nAll plots saved to", OUT)
