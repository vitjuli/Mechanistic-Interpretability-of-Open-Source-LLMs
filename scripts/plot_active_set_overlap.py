"""
Plot the active-set overlap (Jaccard) by layer from j79 output.

Two panels:
  A — Jaccard by layer with verdict bands (linear / intermediate / piecewise)
       Also shows core-features and prompt-specific counts as bars.
  B — Per-layer "regime structure": always-on core vs prompt-specific features.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("data/analysis/runD_v2/geometry_stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to load JSON; fallback to hardcoded values from CSD3 log if not synced yet
JSON = Path("data/analysis/active_set_overlap/active_set_overlap.json")
if JSON.exists():
    d = json.load(open(JSON))
    rows = d.get("rows") or d.get("layers") or list(d.values())
    layers = sorted(int(r["layer"]) if "layer" in r else int(k.replace("L","")) for r, k in zip(rows, d.keys()))
else:
    # Hardcoded from j79 CSD3 log
    raw = """L10 0.723 23 87.2 16 50
L11 0.693 22 89.6 13 61
L12 0.673 29 94.6 18 51
L13 0.540 19 94.6  9 57
L14 0.523 26 96.1 10 110
L15 0.426 31 96.5  7 175
L16 0.473 15 93.9  6 94
L17 0.443 34 97.0 11 189
L18 0.485 26 95.9 10 181
L19 0.471 31 96.3 12 150
L20 0.468 32 96.7 12 200
L21 0.548 32 96.1 12 169
L22 0.548 21 96.1 11 85
L23 0.625 29 96.1 15 91
L24 0.628 35 95.9 22 83
L25 0.684 24 93.3 16 78"""
    layers, jaccards, med_sizes, distincts, cores, specifics = [], [], [], [], [], []
    for line in raw.strip().split("\n"):
        p = line.split()
        layers.append(int(p[0][1:]))
        jaccards.append(float(p[1]))
        med_sizes.append(int(p[2]))
        distincts.append(float(p[3]))
        cores.append(int(p[4]))
        specifics.append(int(p[5]))
    layers = np.array(layers)
    jaccards = np.array(jaccards)
    med_sizes = np.array(med_sizes)
    distincts = np.array(distincts)
    cores = np.array(cores)
    specifics = np.array(specifics)

random_baseline = 7e-5
mean_jaccard = float(jaccards.mean())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ── Panel A: Jaccard by layer with regime bands ──────────────────────────────
ax = axes[0]
# regime bands
ax.axhspan(0.8, 1.0,   color="tab:red",    alpha=0.15, label="NEAR-LINEAR (§4.4 weak)")
ax.axhspan(0.4, 0.8,   color="tab:orange", alpha=0.15, label="INTERMEDIATE (§4.4 supported)")
ax.axhspan(0.0, 0.4,   color="tab:green",  alpha=0.15, label="STRONGLY PIECEWISE (§4.4 strong)")

ax.plot(layers, jaccards, "o-", color="black", lw=2, markersize=9,
        markerfacecolor="tab:blue", label="mean Jaccard")

# annotate values
for L, j in zip(layers, jaccards):
    ax.annotate(f"{j:.2f}", (L, j), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)

# mean line
ax.axhline(mean_jaccard, color="darkblue", linestyle=":", lw=1.5,
           label=f"mean across layers = {mean_jaccard:.3f}")

ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise Jaccard of active sets")
ax.set_ylim(0, 1)
ax.set_xticks(layers)
ax.set_xticklabels([str(L) for L in layers], rotation=45, fontsize=8)
ax.set_title("A — Active-set overlap by layer\n"
             "Intermediate regime → §4.4 argument supported quantitatively")
ax.legend(loc="upper center", fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# annotate "concept layers" region
ax.axvspan(15, 20, alpha=0.07, color="purple")
ax.text(17.5, 0.05, "concept layers\n(j67 peak ablation)",
        ha="center", fontsize=8, color="purple", style="italic")

# ── Panel B: regime structure — core vs prompt-specific ──────────────────────
ax = axes[1]
width = 0.35
x = np.arange(len(layers))
ax.bar(x - width/2, cores, width, label="core (always-on, freq≥0.9)",
       color="tab:green", alpha=0.8, edgecolor="black")
ax.bar(x + width/2, specifics, width, label="prompt-specific (freq≤0.1)",
       color="tab:orange", alpha=0.8, edgecolor="black")

ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in layers], rotation=45, fontsize=8)
ax.set_ylabel("# features")
ax.set_title("B — Stable core vs prompt-specific features\n"
             "Core is small (6–22), specifics dominate (50–200) → genuine piecewise behaviour")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ratio annotation
for xi, (c, s) in enumerate(zip(cores, specifics)):
    ratio = s / max(c, 1)
    ax.text(xi, max(c, s) + 10, f"{ratio:.0f}×", ha="center", fontsize=7,
            color="darkorange", style="italic")

fig.suptitle(
    "Active-set overlap (j79) — encoder operates in piecewise-linear regime\n"
    f"Jaccard {mean_jaccard:.2f} >> random {random_baseline:.0e}, but << 1 → many active hyperplanes",
    fontsize=12
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = OUT_DIR / "active_set_overlap.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# print summary
print(f"\n{'='*70}")
print("ACTIVE-SET OVERLAP SUMMARY")
print(f"{'='*70}")
print(f"Mean Jaccard:              {mean_jaccard:.3f}  →  INTERMEDIATE regime")
print(f"Random baseline:           ~{random_baseline:.0e}  (mean is ~{int(mean_jaccard/random_baseline):,}× above)")
print(f"Min (most piecewise):      L{layers[jaccards.argmin()]}  Jaccard = {jaccards.min():.3f}")
print(f"Max (most linear):         L{layers[jaccards.argmax()]}  Jaccard = {jaccards.max():.3f}")
print(f"\nConcept-active layers L15-L20 — Jaccard = {jaccards[(layers>=15)&(layers<=20)].mean():.3f}")
print(f"  → MOST piecewise where concept lives (specifically supports §4.4)")
print(f"\nDistinct active-set patterns: {distincts.mean():.1f}% on average  (almost every prompt unique)")
print(f"\nVerdict supports §4.4:")
print(f"  feature-decodability (AUC 0.97) and w_res orthogonality (capture 0.30) are")
print(f"  two CONSEQUENCES of piecewise encoder nonlinearity, not a contradiction.")
