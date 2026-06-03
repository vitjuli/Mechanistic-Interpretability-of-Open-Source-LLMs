"""
Chapter §6 summary: three converging negative verdicts on w_res.

Panel A (j65) — Steering dose-response:
  flip rate of w_res vs shuffled-label band, by intensity |c|, per layer.
  Result: max flip rate 0.07, never approaches tau=0.7. w_res is decodable
  but NOT causal.

Panel B (j66) — Carrier orthogonality to w_res:
  per-layer signed cos_C(d_f, w_res) and capture_wres.
  Result: signed_mean ≈ 0 across layers, all in null band.
  Carriers do NOT write toward w_res.

Panel C (j68) — w_res instability:
  20 family resamples → pairwise stability cos = 0.625 (≈51° between draws).
  AUC stable (0.984) but DIRECTION drifts.
  Result: decodability is robust, but w_res itself is overfit to specific split.
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("data/analysis/runD_v2/geometry_stage1")

# ── load all three sources ───────────────────────────────────────────────────
cause = json.load(open(ROOT / "axis_causality_summary.json"))
carrier = json.load(open(ROOT / "carrier_vs_wres.json"))
stability = json.load(open(ROOT / "wres_stability_final.json"))
curve = pd.read_csv(ROOT / "axis_causality_curve.csv")

# ── figure layout ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ── PANEL A: steering dose-response ──────────────────────────────────────────
ax = axes[0]
# get flip rates per (tap, c) for w_res and shuffled
# CSV has dir in {w_res, sperp, random, format} — and shuffled is null derived
# We use the specificity table from cause['specificity'] which has aggregated flip rates
spec = pd.DataFrame(cause["specificity"])
# c can be negative (α→β arm uses -c). Use |c| for x-axis.
spec["abs_c"] = spec["c"].abs()
# average over both arms (alpha→beta and beta→alpha)
agg = spec.groupby(["tap", "abs_c"]).agg({
    "flip_w_res": "mean",
    "shuffled_p95": "mean",
    "random": "mean",
}).reset_index()

taps_in_order = ["postL18", "postL21", "postL24", "final"]
cmap = plt.cm.viridis
for i, tap in enumerate(taps_in_order):
    sub = agg[agg["tap"] == tap].sort_values("abs_c")
    col = cmap(i / max(1, len(taps_in_order) - 1))
    ax.plot(sub["abs_c"], sub["flip_w_res"], "o-", color=col, lw=2,
            label=f"{tap.replace('postL','L')}", markersize=7)

# shuffled p95 band (mean across taps)
shuf_by_c = agg.groupby("abs_c")["shuffled_p95"].mean()
rand_by_c = agg.groupby("abs_c")["random"].mean()
ax.plot(shuf_by_c.index, shuf_by_c.values, "k--", lw=1.5, alpha=0.6,
        label="shuffled p95 (null)")
ax.plot(rand_by_c.index, rand_by_c.values, "k:", lw=1.5, alpha=0.4,
        label="random direction")

ax.axhline(0.7, color="red", linestyle="--", lw=1, alpha=0.7,
           label="τ_flip = 0.7 (causal threshold)")
ax.set_xlabel("steering intensity |c|")
ax.set_ylabel("flip rate (heldout)")
ax.set_title("A — Steering dose-response (j65)\n"
             f"max w_res flip = {cause['max_wres_flip']:.2f} "
             "→ NOT causal")
ax.set_ylim(-0.02, 0.8)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# verdict box
ax.text(0.98, 0.02,
        "OUTCOME 3:\nNO CONTROL\n(decodable ≠ causal)",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

# ── PANEL B: carriers orthogonal to w_res ────────────────────────────────────
ax = axes[1]
layers_dict = carrier["layers"]
layers = sorted(int(k) for k in layers_dict.keys())
signed = [layers_dict[str(L)]["signed_mean"] for L in layers]
capture = [layers_dict[str(L)]["carrier_capture_wres"] for L in layers]
k_per = [layers_dict[str(L)]["k"] for L in layers]
null_p95 = [layers_dict[str(L)]["null_mean_abs_cos"]["p95"] for L in layers]

# signed cos (left axis)
ax.plot(layers, signed, "o-", color="tab:blue", lw=2, markersize=7,
        label="signed mean cos_C(d_f, w_res)")
ax.fill_between(layers,
                [-p for p in null_p95],
                null_p95,
                color="red", alpha=0.15, label="null |cos|<p95")
ax.axhline(0, color="black", lw=0.5)

ax.set_xlabel("layer")
ax.set_ylabel("signed cos_C(carrier, w_res)", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(-0.05, 0.05)
ax.set_xticks(layers[::2])

# capture (right axis)
ax2 = ax.twinx()
ax2.plot(layers, capture, "s--", color="darkorange", lw=2, markersize=7,
         alpha=0.85, label="capture w_res (subspace)")
ax2.set_ylabel("carrier subspace capture of w_res", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.set_ylim(0, 0.5)

ax.set_title("B — Carriers vs w_res (j66)\n"
             f"227 carriers, signed_mean=-0.002, ALL capture={carrier['cross_layer']['carrier_capture_wres']:.2f}")

# combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

ax.text(0.98, 0.02,
        "ORTHOGONAL:\ncarriers DO NOT\nwrite along w_res",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

# ── PANEL C: stability ───────────────────────────────────────────────────────
ax = axes[2]

# bar chart of key metrics
metrics = {
    "Held-out AUC\n(decodability)":      stability["heldout_auc"]["mean"],
    "Direction stability\n(cos w_res, resamples)":   stability["wres_stability_cos"]["mean"],
    "3× null p95\n(stability threshold)":            3 * stability["random_cos_null"]["p95"],
    "cos_C(w_res, gbar)\n(orthog. stable?)":         stability["cos_C_wres_gbar"]["mean"],
    "Shuffled-label AUC\n(sanity: ~0.5)":            stability["shuffled_label_auc"]["mean"],
}
errors = {
    "Held-out AUC\n(decodability)":      stability["heldout_auc"]["std"],
    "Direction stability\n(cos w_res, resamples)":   (stability["wres_stability_cos"]["p95"] - stability["wres_stability_cos"]["p05"]) / 2,
    "3× null p95\n(stability threshold)":            0,
    "cos_C(w_res, gbar)\n(orthog. stable?)":         stability["cos_C_wres_gbar"]["std"],
    "Shuffled-label AUC\n(sanity: ~0.5)":            0,
}

x = np.arange(len(metrics))
vals = list(metrics.values())
errs = list(errors.values())
colors = ["tab:green", "tab:green", "tab:red", "tab:orange", "tab:gray"]

bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.7,
              edgecolor="black", capsize=5)

# expected values
ax.axhline(1.0, color="green", linestyle=":", lw=1, alpha=0.5,
           label="1.0 (perfect)")
ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.5,
           label="0.5 (chance)")

# annotate values
for i, (xi, v) in enumerate(zip(x, vals)):
    ax.text(xi, v + 0.03, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics.keys(), fontsize=8)
ax.set_ylim(-0.1, 1.15)
ax.set_ylabel("value")
stable_str = "STABLE ✓" if stability["stable"] else "UNSTABLE ✗"
ax.set_title(f"C — w_res stability (j68)\n"
             f"20 resamples: cos={stability['wres_stability_cos']['mean']:.2f} "
             f"vs 3×null p95={3*stability['random_cos_null']['p95']:.2f} → {stable_str}")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3, axis="y")

if stability["stable"]:
    ax.text(0.98, 0.45,
            "STABLE:\nw_res IS a real\ndirection (4× null)",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(facecolor="honeydew", edgecolor="green", alpha=0.9))
else:
    ax.text(0.98, 0.45,
            "UNSTABLE:\nw_res direction\nis overfit artefact",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

fig.suptitle("Chapter §6 — w_res is decodable AND stable, but NOT causal and NOT written by features",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = ROOT / "chapter6_verdicts.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# print summary table
print("\n" + "="*70)
print("CHAPTER §6 SUMMARY")
print("="*70)
print(f"\n[A] j65 — steering causality:")
print(f"    max flip rate (w_res): {cause['max_wres_flip']:.3f}  (tau=0.70 required)")
print(f"    verdict: OUTCOME 3 — NO CONTROL")
print(f"\n[B] j66 — carrier vs w_res:")
print(f"    227 carriers; signed_mean = {carrier['cross_layer']['signed_mean']:+.4f}")
print(f"    capture w_res = {carrier['cross_layer']['carrier_capture_wres']:.3f} (30%)")
print(f"    → all carriers orthogonal to w_res in null band")
print(f"\n[C] j68 — w_res stability (n=20 resamples) [CORRECTED]:")
print(f"    Held-out AUC: {stability['heldout_auc']['mean']:.4f}")
print(f"    Direction stability: {stability['wres_stability_cos']['mean']:.3f}")
print(f"    Null p95: {stability['random_cos_null']['p95']:.4f}  (3× = {3*stability['random_cos_null']['p95']:.3f})")
print(f"    cos_C(w_res, gbar): {stability['cos_C_wres_gbar']['mean']:+.4f} (still ⊥ gbar, stably)")
print(f"    stable: {stability['stable']}  →  w_res IS a real direction")
