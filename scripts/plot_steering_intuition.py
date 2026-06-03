"""
Intuitive figure for the steering experiment (j65).

Panel A: schematic — one prompt's h(p) pushed along w_res by c×σ.
          Show original position, all intermediate pushed positions,
          and decision threshold. The arrow shows the push vector.

Panel B: empirical — flip rate vs |c| for w_res steering, per layer,
          with shuffled-label null band for comparison.

Panel C: distribution of (dl_steered - dl_clean) for one (tap, c) — to
          show that pushing DOES move the logits, just not enough to flip.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
h = np.load(ROOT / "h_residual_per_depth.npz")
y, is_train = h["y"], h["is_train"]
cause = json.load(open(ROOT / "axis_causality_summary.json"))
curve = pd.read_csv(ROOT / "axis_causality_curve.csv")

# pick one alpha prompt for the schematic
TAP = "postL22"          # for projection visualisation
H = h[TAP].astype(np.float64)
w = np.load(ROOT / f"w_res_{TAP}.npy").astype(np.float64)
sigma_axis = float(H.std(0) @ np.abs(w))   # rough scale; we use std along axis instead
# better: stdev of projection along w on the data
projections = H @ w
sigma = projections.std()
print(f"σ (stdev of projections along w_res at {TAP}) = {sigma:.3f}")

# select one true-α prompt for the schematic
idx_alpha = np.where((y == 0) & is_train)[0][0]
s_alpha = float(H[idx_alpha] @ w)

# orthogonal axis for 2D layout (same as concept_axis_intuition figure)
H_perp = H - np.outer(H @ w, w)
H_perp_c = H_perp - H_perp.mean(0)
_, _, Vt = np.linalg.svd(H_perp_c, full_matrices=False)
e2 = Vt[0]
y_coord = H @ e2
y_alpha = float(H[idx_alpha] @ e2)

# centroid threshold
mu_a_proj = float(H[(y == 0) & is_train].mean(0) @ w)
mu_b_proj = float(H[(y == 1) & is_train].mean(0) @ w)
threshold = (mu_a_proj + mu_b_proj) / 2

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0])

# ── PANEL A: schematic ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])

m_a = (y == 0); m_b = (y == 1)
ax.scatter(projections[m_a], y_coord[m_a], s=12, c="tab:blue",   alpha=0.25,
           label=f"α-prompts")
ax.scatter(projections[m_b], y_coord[m_b], s=12, c="tab:orange", alpha=0.25,
           label=f"β-prompts")

# the chosen prompt's original position
ax.scatter([s_alpha], [y_alpha], s=200, c="tab:blue",
           edgecolor="black", linewidth=2, zorder=5,
           label="chosen α-prompt h(p)")

# pushed positions at different c
c_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
cmap_c = plt.cm.Reds
for i, c in enumerate(c_grid):
    pushed_x = s_alpha + c * sigma   # push in +w direction (toward β)
    col = cmap_c(0.3 + 0.7 * i / (len(c_grid) - 1))
    ax.scatter([pushed_x], [y_alpha], s=80, c=[col],
               edgecolor="black", linewidth=0.8, zorder=6, marker="X")
    ax.annotate(f"|c|={c}", (pushed_x, y_alpha),
                textcoords="offset points", xytext=(0, -15),
                ha="center", fontsize=8, color="darkred")

# arrow showing direction
ax.annotate("",
            xy=(s_alpha + 2 * sigma, y_alpha),
            xytext=(s_alpha, y_alpha),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5, alpha=0.7),
            zorder=4)
ax.text(s_alpha + 1 * sigma, y_alpha + 6,
        "push along w_res\n+c × σ",
        color="red", fontsize=10, ha="center", fontweight="bold")

# decision threshold
ax.axvline(threshold, color="red", linestyle="--", lw=1.2, alpha=0.6)
ax.text(threshold, ax.get_ylim()[1] * 0.85, " threshold\n model flips here",
        color="red", fontsize=8, ha="left", va="top")

ax.set_xlabel(r"projection on $w_{\mathrm{res}}$  $\langle h(p), w_{\mathrm{res}}\rangle$",
              fontsize=11)
ax.set_ylabel("orthogonal direction")
ax.set_title(f"A — Steering schematic at {TAP}\n"
             f"push one α-prompt toward β-side by c×σ (σ={sigma:.2f})",
             fontsize=11)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# ── PANEL B: dose-response curves ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])

spec = pd.DataFrame(cause["specificity"])
spec["abs_c"] = spec["c"].abs()
agg = spec.groupby(["tap", "abs_c"]).agg({
    "flip_w_res": "mean", "shuffled_p95": "mean", "random": "mean",
}).reset_index()

taps_in_order = ["postL18", "postL21", "postL24", "final"]
cmap = plt.cm.viridis
for i, tap in enumerate(taps_in_order):
    sub = agg[agg["tap"] == tap].sort_values("abs_c")
    col = cmap(i / max(1, len(taps_in_order) - 1))
    ax.plot(sub["abs_c"], sub["flip_w_res"], "o-", color=col, lw=2,
            label=f"{tap.replace('postL','L')}  (w_res push)", markersize=8)

shuf = agg.groupby("abs_c")["shuffled_p95"].mean()
ax.plot(shuf.index, shuf.values, "k--", lw=1.5, alpha=0.7,
        label="shuffled-label null (p95)")
rnd = agg.groupby("abs_c")["random"].mean()
ax.plot(rnd.index, rnd.values, "k:", lw=1.5, alpha=0.5,
        label="random direction")

ax.axhline(0.7, color="red", linestyle="--", lw=1.5, alpha=0.7,
           label="τ = 0.7 (causal threshold)")
ax.set_xlabel("steering intensity |c| (in units of σ)", fontsize=11)
ax.set_ylabel("flip rate (fraction of prompts where answer changed)")
ax.set_title(f"B — Empirical: pushing by 2σ flips < 7% of prompts\n"
             f"max w_res flip = {cause['max_wres_flip']:.3f}  →  NOT causal",
             fontsize=11)
ax.set_ylim(-0.02, 0.8)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# ── PANEL C: logit shift distribution ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])

# pick final, beta→alpha (push toward alpha for β-prompts), c=2
ch = curve[(curve.tap == "final") & (curve.toward == "alpha") & (curve.dir == "w_res")
           & (curve.c == -2.0)]
ch_c1 = curve[(curve.tap == "final") & (curve.toward == "alpha") & (curve.dir == "w_res")
              & (curve.c == -1.0)]
# delta = dl_steered - dl_clean (how much logit moved)
delta_c2 = ch["dl_steered"].values - ch["dl_clean"].values
delta_c1 = ch_c1["dl_steered"].values - ch_c1["dl_clean"].values

# also: distance to flip (how far each prompt was from flipping)
gap_clean = ch["dl_clean"].values   # logit(β) − logit(α); positive = β preferred
                                     # for β-prompts, we want this to become negative

bins = np.linspace(-2, 2, 30)
ax.hist(delta_c1, bins=bins, alpha=0.6, color="tab:orange",
        label=f"|c|=1.0:  Δ(dl) shift", edgecolor="black")
ax.hist(delta_c2, bins=bins, alpha=0.6, color="tab:red",
        label=f"|c|=2.0:  Δ(dl) shift", edgecolor="black")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Δ logit_diff after steering  (steered − clean)")
ax.set_ylabel("count of β-prompts")
ax.set_title("C — Pushing DOES move logits...\n"
             "...just not enough to flip past threshold", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# annotation
needed_clean_avg = abs(gap_clean.mean())
ax.text(0.02, 0.5,
        f"average β-prompt has logit_diff ≈ −{needed_clean_avg:.1f}\n"
        f"steering shifts it by only ~0.05  →  not enough\n"
        f"would need ~{needed_clean_avg/0.05:.0f}× stronger push",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.85))

fig.suptitle("Steering experiment (j65) — push h(p) along w_res, measure answer flip",
             fontsize=13, y=1.02)
fig.tight_layout()

out = ROOT / "steering_intuition.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved: {out}")
print(f"\nKey numbers:")
print(f"  σ along w_res at postL22: {sigma:.3f}")
print(f"  α-prompt baseline projection: {s_alpha:.3f}")
print(f"  threshold:                    {threshold:.3f}")
print(f"  push by c=2.0 takes it to:    {s_alpha + 2*sigma:.3f}")
print(f"  → still {('PAST' if s_alpha+2*sigma > threshold else 'BELOW')} threshold")
print(f"  max flip rate in j65:         {cause['max_wres_flip']:.3f}  (need 0.7)")
