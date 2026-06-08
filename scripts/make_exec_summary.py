"""
Build executive-summary plots for §6 supervisor handout.
4 PNGs at high DPI suitable for print.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "exec_summary"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ───────────────────────────────────────────────────────────────────────────
# Plot 1 — Concept-blind usage trajectory (j86)
# ───────────────────────────────────────────────────────────────────────────
geo = pd.read_csv(ROOT / "data/analysis/runD_v2/usage_direction/usage_geometry.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(geo["layer"], geo["auc_along_wres"], "-o", color="#2ca02c",
        label="AUC along $w_{res}$ (readable axis)", lw=2.5, ms=5)
ax.plot(geo["layer"], geo["auc_along_u"], "-o", color="#d62728",
        label="AUC along $u$ (used axis)", lw=2.5, ms=5)
ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)
ax.text(28, 0.52, "chance", fontsize=9, alpha=0.7)
ax.set_xlabel("Layer")
ax.set_ylabel("AUC of α/β separation")
ax.set_title("Plot 1 — Concept-blind usage direction (j86)\n"
             r"$\cos(u, w_{res}) \approx 0.03$, yet only $w_{res}$ decodes the concept")
ax.set_ylim(0.3, 1.05)
ax.legend(loc="lower center", frameon=False)
ax.grid(alpha=0.2)
plt.savefig(OUT / "01_usage_concept_blind.png")
plt.close()
print(f"saved {OUT}/01_usage_concept_blind.png")

# ───────────────────────────────────────────────────────────────────────────
# Plot 2 — Behavioural asymmetry (j90 intact-flip)
# ───────────────────────────────────────────────────────────────────────────
steer = pd.read_csv(ROOT / "data/analysis/runD_v2/forced_mode/forced_steering.csv")

def dir_kind(d):
    if "usage" in d: return "usage_forced"
    if "w_res" in d: return "w_res_forced"
    return "null"

steer["kind"] = steer["dir"].apply(dir_kind)
agg = (steer.groupby(["layer", "kind"])["intact_flip"].max().unstack().fillna(0).reset_index())
agg["null"] = agg["null"].fillna(0)
layers = agg["layer"].astype(int).tolist()

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(layers))
w = 0.25
ax.bar(x - w, agg["usage_forced"], w, color="#2ca02c", label="push along $u$ (used)")
ax.bar(x, agg["w_res_forced"], w, color="#d62728", label="push along $w_{res}$ (readable)")
ax.bar(x + w, agg["null"], w, color="#bbbbbb", label="null (random/shuffled)")
for i, v in enumerate(agg["usage_forced"]):
    ax.text(i - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#2ca02c", fontweight="bold")
for i, v in enumerate(agg["w_res_forced"]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels([f"L{L}" for L in layers])
ax.set_ylabel("max intact-flip rate")
ax.set_xlabel("Steered layer (forced regime)")
ax.set_title("Plot 2 — Behavioural asymmetry: $u$ flips, $w_{res}$ does not (j90)\n"
             "Forced-regime steering on the same prompts with the same magnitudes")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.2, axis="y")
plt.savefig(OUT / "02_behavioural_asymmetry.png")
plt.close()
print(f"saved {OUT}/02_behavioural_asymmetry.png")

# ───────────────────────────────────────────────────────────────────────────
# Plot 3 — Linear-theory predictability (j89-A) + dim-free lever-cone PR (j89-B)
# (Replaces the s_concept/s_usage plot which depended on the arbitrary k=13 choice)
# ───────────────────────────────────────────────────────────────────────────
pts = pd.read_csv(ROOT / "data/analysis/runD_v2/intervention_calculus/calculus_points.csv")
mfl = pd.read_csv(ROOT / "data/analysis/runD_v2/intervention_calculus/minflip_per_layer.csv")

def r2_grp(g):
    ss_res = float(((g["meas"] - g["pred"]) ** 2).sum())
    ss_tot = float(((g["meas"] - g["meas"].mean()) ** 2).sum()) + 1e-30
    return 1 - ss_res / ss_tot
r2t = pts.groupby(["layer", "c"]).apply(r2_grp).reset_index(name="R2")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: R² vs amplitude per layer
for L in sorted(r2t["layer"].unique()):
    sub = r2t[r2t["layer"] == L].sort_values("c")
    ax1.plot(sub["c"], sub["R2"], "-o", label=f"L{int(L)}", lw=2, ms=5)
ax1.set_xscale("log")
ax1.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax1.axhline(0.8, color="orange", ls=":", lw=1, alpha=0.7)
ax1.set_xlabel("amplitude c (σ units, log)")
ax1.set_ylabel(r"$R^2$(predicted, measured)")
ax1.set_title("(j89-A)  Linear theory is exact at small amplitudes\n"
              r"$R^2 = 1.000$ at $|c| \leq 4\sigma$ on all 6 taps")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", fontsize=8, ncol=2, frameon=False)
ax1.grid(alpha=0.2)

# Right: lever-cone PR (dim-free)
ax2.plot(mfl["layer"], mfl["lever_cone_pr"], "-o", color="#9467bd", lw=3, ms=7)
for i, (L, pr) in enumerate(zip(mfl["layer"], mfl["lever_cone_pr"])):
    ax2.text(L, pr + 0.08, f"{pr:.1f}", ha="center", fontsize=9, color="#9467bd",
             fontweight="bold")
ax2.set_xlabel("Layer")
ax2.set_ylabel("lever-cone PR")
ax2.set_title("(j89-B)  Working interventions live in a low-dim cone\n"
              r"PR $\in$ [1.0, 2.5] across all 6 taps — dim-free")
ax2.set_ylim(0.5, 3.0)
ax2.grid(alpha=0.2)

plt.suptitle("Plot 3 — Intervention calculus (dim-free): predictability + cone dimensionality",
             fontsize=13, y=1.02)
plt.savefig(OUT / "03_calculus_and_cone.png")
plt.close()
print(f"saved {OUT}/03_calculus_and_cone.png")
# Remove old plot
old = OUT / "03_decision_normal_geometry.png"
if old.exists():
    old.unlink()

# ───────────────────────────────────────────────────────────────────────────
# Plot 4 — Healing contest at L20 forced regime (j98)
# ───────────────────────────────────────────────────────────────────────────
hf = pd.read_csv(ROOT / "data/analysis/runD_v2/healing_forced/healing_contest_forced.csv")

# Take valid α=2.0 inject + suppress, all_failed_beta group
inj = hf[(hf["mode"] == "inject") & (hf["alpha"] == 2.0) &
         (hf["prompt_set"] == "all_failed_beta")]
sup = hf[(hf["mode"] == "suppress") & (hf["prompt_set"] == "all_failed_beta") &
         (hf["direction"] == "d_error")]

# Three layers (17, 20, 24) × three interventions (concept, surface, suppress)
layers = [17, 20, 24]
labels = ["L17", "L20", "L24"]
concept_vals = [float(inj[(inj["layer"] == L) & (inj["direction"] == "concept_wres")]["intact_flip"].iloc[0]) for L in layers]
surface_vals = [float(inj[(inj["layer"] == L) & (inj["direction"] == "surface_dsurf")]["intact_flip"].iloc[0]) for L in layers]
suppress_vals = [float(sup[sup["layer"] == L]["intact_flip"].iloc[0]) for L in layers]
null_vals = [float(inj[(inj["layer"] == L) & (inj["direction"] == "null_random")]["intact_flip"].iloc[0]) for L in layers]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(labels))
w = 0.2
ax.bar(x - 1.5*w, concept_vals, w, color="#2ca02c",
       label="inject concept ($w_{res}$ push)")
ax.bar(x - 0.5*w, surface_vals, w, color="#1f77b4",
       label="inject surface ($d_{surf}$ push)")
ax.bar(x + 0.5*w, suppress_vals, w, color="#d62728",
       label="suppress $d_{error}$")
ax.bar(x + 1.5*w, null_vals, w, color="#bbbbbb",
       label="null (random)")
for i, (c, s, sp) in enumerate(zip(concept_vals, surface_vals, suppress_vals)):
    ax.text(i - 1.5*w, c + 0.02, f"{c:.2f}", ha="center", fontsize=8, color="#2ca02c")
    ax.text(i - 0.5*w, s + 0.02, f"{s:.2f}", ha="center", fontsize=8, color="#1f77b4")
    ax.text(i + 0.5*w, sp + 0.02, f"{sp:.2f}", ha="center", fontsize=8, color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("intact-flip rate (n=16 hard failures)")
ax.set_xlabel("Layer (forced regime, α = 2σ)")
ax.set_title("Plot 4 — Bypass partly reversible at L20 (j98 healing contest)\n"
             "Concept-injection: clean partial heal (0.38). Surface-injection dominates (0.81).")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.grid(alpha=0.2, axis="y")
plt.savefig(OUT / "04_healing_contest_forced.png")
plt.close()
print(f"saved {OUT}/04_healing_contest_forced.png")

print(f"\nAll plots saved to {OUT}")
