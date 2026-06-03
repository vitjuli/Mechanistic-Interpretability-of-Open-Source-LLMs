"""
Chapter §6 summary v2: now with FOUR converging verdicts on w_res/carriers.

Panel A (j65) — steering causality
Panel B (j66) — carrier ⊥ w_res
Panel C (j68) — w_res stability (corrected null)
Panel D (j67) — carrier ablation per-layer vs random null  [NEW]
"""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, pandas as pd

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
IIA  = Path("data/analysis/iia_failure_diagnosis")

cause      = json.load(open(ROOT / "axis_causality_summary.json"))
carrier    = json.load(open(ROOT / "carrier_vs_wres.json"))
stability  = json.load(open(ROOT / "wres_stability_final.json"))
ablation   = json.load(open(IIA  / "carrier_ablation_full.json"))
curve      = pd.read_csv(ROOT / "axis_causality_curve.csv")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# ── PANEL A: steering ────────────────────────────────────────────────────────
ax = axes[0, 0]
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
            label=f"{tap.replace('postL','L')}", markersize=7)
shuf = agg.groupby("abs_c")["shuffled_p95"].mean()
ax.plot(shuf.index, shuf.values, "k--", lw=1.5, alpha=0.6, label="shuffled p95")
ax.axhline(0.7, color="red", linestyle="--", lw=1, alpha=0.7, label="τ=0.7")
ax.set_xlabel("steering intensity |c|")
ax.set_ylabel("flip rate")
ax.set_title(f"A — Steering (j65)\nmax flip = {cause['max_wres_flip']:.2f} → NOT causal")
ax.set_ylim(-0.02, 0.8); ax.legend(loc="upper left", fontsize=7)
ax.grid(alpha=0.3)
ax.text(0.98, 0.02, "OUTCOME 3:\nNO CONTROL", transform=ax.transAxes,
        fontsize=9, ha="right", va="bottom",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

# ── PANEL B: carrier ⊥ w_res ─────────────────────────────────────────────────
ax = axes[0, 1]
layers_dict = carrier["layers"]
layers = sorted(int(k) for k in layers_dict.keys())
signed   = [layers_dict[str(L)]["signed_mean"] for L in layers]
capture  = [layers_dict[str(L)]["carrier_capture_wres"] for L in layers]
null_p95 = [layers_dict[str(L)]["null_mean_abs_cos"]["p95"] for L in layers]

ax.plot(layers, signed, "o-", color="tab:blue", lw=2, markersize=7,
        label="signed mean cos_C(d_f, w_res)")
ax.fill_between(layers, [-p for p in null_p95], null_p95, color="red", alpha=0.15,
                label="null |cos|<p95")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("layer"); ax.set_ylabel("signed cos_C", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(-0.05, 0.05); ax.set_xticks(layers[::2])

ax2 = ax.twinx()
ax2.plot(layers, capture, "s--", color="darkorange", lw=2, markersize=7, alpha=0.85,
         label="capture w_res")
ax2.set_ylabel("capture w_res", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.set_ylim(0, 0.5)

ax.set_title(f"B — Carriers vs w_res (j66)\n"
             f"227 carriers, capture ALL = {carrier['cross_layer']['carrier_capture_wres']:.2f}")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
ax.grid(alpha=0.3)
ax.text(0.98, 0.02, "ORTHOGONAL", transform=ax.transAxes, fontsize=9,
        ha="right", va="bottom",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.9))

# ── PANEL C: stability ───────────────────────────────────────────────────────
ax = axes[1, 0]
metrics = {
    "Held-out AUC":         stability["heldout_auc"]["mean"],
    "Direction\nstability": stability["wres_stability_cos"]["mean"],
    "3× null p95\n(threshold)": 3 * stability["random_cos_null"]["p95"],
    "cos_C(w_res,gbar)":    stability["cos_C_wres_gbar"]["mean"],
    "Shuffled AUC\n(sanity)": stability["shuffled_label_auc"]["mean"],
}
errors = {
    "Held-out AUC":         stability["heldout_auc"]["std"],
    "Direction\nstability": (stability["wres_stability_cos"]["p95"] - stability["wres_stability_cos"]["p05"]) / 2,
    "3× null p95\n(threshold)": 0,
    "cos_C(w_res,gbar)":    stability["cos_C_wres_gbar"]["std"],
    "Shuffled AUC\n(sanity)": 0,
}
x = np.arange(len(metrics))
colors = ["tab:green", "tab:green", "tab:red", "tab:orange", "tab:gray"]
ax.bar(x, list(metrics.values()), yerr=list(errors.values()),
       color=colors, alpha=0.7, edgecolor="black", capsize=4)
ax.axhline(1.0, color="green", linestyle=":", lw=1, alpha=0.5)
ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.5)
for xi, v in zip(x, metrics.values()):
    ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics.keys(), fontsize=8)
ax.set_ylim(-0.1, 1.15); ax.set_ylabel("value")
stable_str = "STABLE ✓" if stability["stable"] else "UNSTABLE ✗"
ax.set_title(f"C — w_res stability (j68)\n"
             f"cos={stability['wres_stability_cos']['mean']:.2f} vs "
             f"3×null={3*stability['random_cos_null']['p95']:.2f} → {stable_str}")
ax.grid(alpha=0.3, axis="y")
ax.text(0.98, 0.45, "STABLE\n(real direction)", transform=ax.transAxes,
        fontsize=9, ha="right", va="top",
        bbox=dict(facecolor="honeydew", edgecolor="green", alpha=0.9))

# ── PANEL D: carrier ablation per-layer vs null  [NEW] ───────────────────────
ax = axes[1, 1]
sets = ablation["sets"]
layer_keys = [k for k in sets.keys() if k.startswith("L")]
layer_nums = sorted(int(k[1:]) for k in layer_keys)
all_key = "ALL" if "ALL" in sets else next((k for k in sets if k not in layer_keys), None)

carriers   = [sets[f"L{L}"]["carrier_mean_abs_dlogit"] for L in layer_nums]
ci_lower   = [sets[f"L{L}"]["carrier_CI"][0]            for L in layer_nums]
ci_upper   = [sets[f"L{L}"]["carrier_CI"][1]            for L in layer_nums]
null_mean  = [sets[f"L{L}"]["null_band"]["mean"]        for L in layer_nums]
null_p95   = [sets[f"L{L}"]["null_band"]["p95"]         for L in layer_nums]
pcts       = [sets[f"L{L}"]["carrier_percentile_vs_null"] for L in layer_nums]

ci_err = [[c - lo for c, lo in zip(carriers, ci_lower)],
          [up - c for c, up in zip(carriers, ci_upper)]]

# carrier bars
xs = np.arange(len(layer_nums))
bar_colors = ["tab:green" if p >= 0.95 else "tab:orange" if p >= 0.8 else "tab:red"
              for p in pcts]
ax.bar(xs, carriers, yerr=ci_err, color=bar_colors, alpha=0.7,
       edgecolor="black", capsize=3, label="carrier |Δ logit|")
# null p95 line
ax.plot(xs, null_p95, "k--", lw=1.5, alpha=0.7, label="random null p95")
ax.plot(xs, null_mean, "k:", lw=1, alpha=0.5, label="random null mean")

# annotate percentile on each bar
for xi, c, p in zip(xs, carriers, pcts):
    color = "darkgreen" if p >= 0.95 else "darkorange" if p >= 0.8 else "darkred"
    ax.text(xi, c + 0.02, f"{int(p*100)}", ha="center", va="bottom",
            fontsize=7, color=color, fontweight="bold")

ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in layer_nums], fontsize=7, rotation=45)
ax.set_ylabel("|Δ logit| after ablation")
all_pct = sets[all_key]["carrier_percentile_vs_null"] if all_key else None
ax.set_title(f"D — Carrier ablation per-layer (j67)\n"
             f"ALL 227 carriers: pct = {int(all_pct*100)}% (in null band) → correlates, not mechanism")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3, axis="y")

# inset for ALL set
ax.text(0.98, 0.97,
        f"ALL set: |Δ|={sets[all_key]['carrier_mean_abs_dlogit']:.2f}\n"
        f"null p95: {sets[all_key]['null_band']['p95']:.2f}\n"
        f"percentile: {int(all_pct*100)}%",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.85))

fig.suptitle(
    "Chapter §6 — Four converging negative verdicts on w_res / carriers:  "
    "decodable & stable, but not causal, not written, not ablation-specific",
    fontsize=13
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "chapter6_verdicts_v2.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# summary text
print(f"\n{'='*70}")
print("CHAPTER §6 SUMMARY (4 verdicts)")
print(f"{'='*70}")
print(f"\n[A] j65 — steering causality:  max flip = {cause['max_wres_flip']:.2f}  → NOT causal")
print(f"[B] j66 — carrier vs w_res:    capture = {carrier['cross_layer']['carrier_capture_wres']:.2f}  → orthogonal")
print(f"[C] j68 — stability:           cos={stability['wres_stability_cos']['mean']:.2f}, threshold={3*stability['random_cos_null']['p95']:.2f}  → STABLE")
print(f"[D] j67 — ablation specificity:")
print(f"    L25 (peak):    |Δ|={sets['L25']['carrier_mean_abs_dlogit']:.2f}  pct={int(sets['L25']['carrier_percentile_vs_null']*100)}%  [outlier]")
print(f"    L24 (worst):   |Δ|={sets['L24']['carrier_mean_abs_dlogit']:.2f}  pct={int(sets['L24']['carrier_percentile_vs_null']*100)}%")
print(f"    ALL combined:  |Δ|={sets[all_key]['carrier_mean_abs_dlogit']:.2f}  pct={int(all_pct*100)}%  → in null band")
print(f"\n  per-layer: {sum(1 for p in pcts if p >= 0.95)}/{len(pcts)} layers beat null at p95")
print(f"  but combined 227 carriers behave like 227 random features")
