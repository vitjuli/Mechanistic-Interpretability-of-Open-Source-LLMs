"""
Use w_res as a binary classifier: project h(p) onto probe direction and threshold.

For each layer:
  s(p) = <h(p), w_res>           # scalar score, scalar per prompt
  predict α if s(p) > threshold, β otherwise

Threshold = midpoint of projected train-class means (optimal under Fisher LDA).

Reports:
  - distribution of s(p) for α vs β (heldout only)
  - classification accuracy at optimal threshold
  - confusion matrix
  - error analysis: which prompts get misclassified
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, json

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
h    = np.load(ROOT / "h_residual_per_depth.npz")
y, is_train = h["y"], h["is_train"]

taps = ["postL14", "postL18", "postL22", "postL25", "final"]
n_panels = len(taps)

fig, axes = plt.subplots(2, n_panels, figsize=(4*n_panels, 8))

print(f"{'tap':>8}  {'thresh':>8}  {'acc_train':>9}  {'acc_heldout':>11}  {'errors':>7}")
print("-" * 60)

results = []
for col, tap in enumerate(taps):
    H = h[tap].astype(np.float64)        # (538, 2560)
    w = np.load(ROOT / f"w_res_{tap}.npy").astype(np.float64)
    # project
    s = H @ w                            # (538,) — scalar score per prompt

    # Fisher-optimal threshold = midpoint of projected train means
    s_a_train = s[(y == 0) & is_train]
    s_b_train = s[(y == 1) & is_train]
    threshold = (s_a_train.mean() + s_b_train.mean()) / 2

    # convention: alpha has higher projection (since w ∝ μ_α - μ_β)
    pred = (s > threshold).astype(int) * 0 + (s <= threshold).astype(int) * 1  # 0=alpha, 1=beta
    # actually: if alpha has HIGHER s (above thresh), then pred=0 when above
    # quick: just sign-match using train accuracy
    pred = np.where(s_a_train.mean() > s_b_train.mean(),
                    (s <= threshold).astype(int),   # alpha=0 above
                    (s > threshold).astype(int))    # alpha=0 below

    acc_train   = (pred[is_train]    == y[is_train]).mean()
    acc_heldout = (pred[~is_train]   == y[~is_train]).mean()
    n_errors    = int((pred[~is_train] != y[~is_train]).sum())

    print(f"{tap:>8}  {threshold:>8.3f}  {acc_train:>9.3f}  {acc_heldout:>11.3f}  {n_errors:>5}/202")
    results.append({"tap": tap, "acc_train": float(acc_train),
                    "acc_heldout": float(acc_heldout), "n_errors": n_errors,
                    "threshold": float(threshold)})

    # ── top row: histogram of projections (heldout only) ─────────────────────
    ax = axes[0, col]
    mask_h = ~is_train
    s_a = s[(y == 0) & mask_h]
    s_b = s[(y == 1) & mask_h]
    bins = np.linspace(s.min(), s.max(), 35)
    ax.hist(s_a, bins=bins, alpha=0.6, color="tab:blue",   label=f"α (n={len(s_a)})", edgecolor="black")
    ax.hist(s_b, bins=bins, alpha=0.6, color="tab:orange", label=f"β (n={len(s_b)})", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", lw=2, label=f"threshold")
    ax.set_title(f"{tap}  |  acc={acc_heldout:.1%}")
    ax.set_xlabel("⟨h(p), w_res⟩")
    if col == 0:
        ax.set_ylabel("count (heldout)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── bottom row: confusion matrix (heldout) ────────────────────────────────
    ax = axes[1, col]
    cm = np.zeros((2, 2), dtype=int)
    for true, p in zip(y[mask_h], pred[mask_h]):
        cm[true, p] += 1
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max()/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred α", "pred β"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true α", "true β"])
    ax.set_title(f"confusion (heldout)")

fig.suptitle("Probe-as-classifier: predict α/β by sign(⟨h(p), w_res⟩ − threshold)\n"
             "538 prompts, family-grouped CV (336 train / 202 heldout)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = ROOT / "probe_as_classifier.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved: {out}")

# also save summary
with open(ROOT / "probe_as_classifier_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"saved: {ROOT}/probe_as_classifier_summary.json")
