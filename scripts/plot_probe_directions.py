"""
Plot probe directions across depths.

Loads w_res_post{L14..L25}.npy + w_res_final.npy + concept_directions.npz
and produces:
  panel A: AUC by depth (line)
  panel B: pairwise cos_C(w_i, w_j) heatmap (causal cosine in Sigma_inv metric)
  panel C: cos_C(w_i, w_final) — how each layer aligns to the final probe
  panel D: cos_C(w_i, gbar)    — how each layer aligns to the logit axis

All cos are computed in the causal inner product <a, Sigma_inv b> / sqrt(...).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("data/analysis/runD_v2/geometry_stage1")

# load AUC + verdict
diag = json.load(open(ROOT / "concept_axis_diagnosis.json"))
layers = [r["tap"] for r in diag["D1"]]      # postL14..postL25, final
aucs   = [r["heldout_auc"] for r in diag["D1"]]
cos_w_gbar = [r["cos_C_w_gbar"] for r in diag["D1"]]

# load w_res for each depth
W = np.stack([np.load(ROOT / f"w_res_{tap}.npy") for tap in layers]).astype(np.float64)
n = W.shape[0]

# load Sigma_inv (causal inner product)
cd = np.load(ROOT / "concept_directions.npz")
Sinv = cd["Sigma_inv"].astype(np.float64)
gbar = cd["gbar"].astype(np.float64)
gbar = gbar / np.sqrt(max(gbar @ Sinv @ gbar, 1e-30))   # unit in causal metric

def cos_C(a, b):
    num = float(a @ Sinv @ b)
    na  = float(np.sqrt(max(a @ Sinv @ a, 1e-30)))
    nb  = float(np.sqrt(max(b @ Sinv @ b, 1e-30)))
    return num / (na * nb)

pair = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pair[i, j] = cos_C(W[i], W[j])

cos_final = np.array([cos_C(W[i], W[-1]) for i in range(n)])
cos_gbar_recomputed = np.array([cos_C(W[i], gbar) for i in range(n)])

# nice short labels
short = [t.replace("postL", "L") for t in layers]   # L14...L25, final

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# A — AUC
ax = axes[0, 0]
ax.plot(range(n), aucs, marker="o", linewidth=2)
ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45)
ax.set_ylim(0.95, 1.0)
ax.set_ylabel("held-out AUC")
ax.set_title("A — Linear probe accuracy by depth")
ax.grid(alpha=0.3)
for x, y in zip(range(n), aucs):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8)

# B — pairwise heatmap
ax = axes[0, 1]
im = ax.imshow(pair, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45)
ax.set_yticks(range(n)); ax.set_yticklabels(short)
ax.set_title("B — pairwise cos_C(w_i, w_j) — probe rotation across depths")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# C — alignment to final
ax = axes[1, 0]
ax.plot(range(n), cos_final, marker="o", linewidth=2, color="tab:orange")
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45)
ax.set_ylim(-0.2, 1.05)
ax.set_ylabel("cos_C(w_i, w_final)")
ax.set_title("C — Each layer probe vs FINAL probe (causal cosine)")
ax.grid(alpha=0.3)

# D — alignment to gbar (answer-token contrast)
ax = axes[1, 1]
ax.plot(range(n), cos_gbar_recomputed, marker="o", linewidth=2, color="tab:green",
        label="cos_C(w_res, gbar) — recomputed")
ax.plot(range(n), cos_w_gbar, marker="x", linestyle="--", color="gray",
        label="from diagnosis JSON (sanity check)", alpha=0.5)
ax.axhline(0, color="black", lw=0.5)
# null band p95 ~ 0.04 from diagnosis run
null_p95 = max(r["cos_C_w_gbar_null_p95"] for r in diag["D1"])
ax.axhspan(-null_p95, null_p95, alpha=0.15, color="red", label=f"null |cos|<p95 ({null_p95:.2f})")
ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45)
ax.set_ylabel("cos_C(w_res, gbar)")
ax.set_title("D — Probe direction vs answer-token axis (gbar)")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Probe directions across depths — α vs β decay (538 prompts, family-grouped CV)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out = ROOT / "probe_directions_overview.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# also print quick summary
print("\nSummary:")
print(f"  AUC range:           {min(aucs):.3f} (L14) → {max(aucs):.3f}")
print(f"  cos_C(w_14, w_final):  {cos_final[0]:+.3f}")
print(f"  cos_C(w_25, w_final):  {cos_final[-2]:+.3f}")
print(f"  mean off-diag pair: {pair[~np.eye(n, dtype=bool)].mean():+.3f}")
