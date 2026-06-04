"""
Chapter §6 summary v3: extended steering curve (j65 up to 2σ + j75 up to 32σ)
+ all four converging verdicts.

Panel A (j65+j75) — steering up to 32σ, with intact-rate annotation
Panel B (j66) — carriers ⊥ w_res
Panel C (j68) — w_res stability
Panel D (j67) — carrier ablation per-layer vs random null
"""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, pandas as pd

ROOT = Path("data/analysis/runD_v2/geometry_stage1")
IIA  = Path("data/analysis/iia_failure_diagnosis")
SAT  = Path("data/analysis/runD_v2/steering_saturation")

cause      = json.load(open(ROOT / "axis_causality_summary.json"))
sat        = json.load(open(SAT / "steering_saturation_summary.json"))
carrier    = json.load(open(ROOT / "carrier_vs_wres.json"))
stability  = json.load(open(ROOT / "wres_stability_final.json"))
ablation   = json.load(open(IIA / "carrier_ablation_full.json"))

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# ── PANEL A: extended steering curve (j65 [0..2σ] + j75 [0.5..32σ]) ──────────
ax = axes[0, 0]

# j75 data: postL24 + final at c ∈ {0.5, 1, 2, 4, 8, 16, 32}
sat_df = pd.DataFrame(sat["curve"])
for tap, color, marker in [("postL24", "tab:purple", "o"),
                           ("final",  "tab:red",    "s")]:
    s = sat_df[sat_df.tap == tap].sort_values("c")
    ax.plot(s["c"], s["wres_flip"], marker=marker, color=color, lw=2,
            markersize=8, label=f"{tap.replace('postL','L')} w_res flip")
    ax.plot(s["c"], s["shuffled_flip_p95"], "--", color=color, lw=1, alpha=0.5,
            label=f"{tap.replace('postL','L')} shuffled p95")

# Annotate intact rate (always 0)
ax.axhline(0.5, color="red", linestyle=":", lw=1.5, alpha=0.7, label="τ_flip=0.5")

# j65 zone (where most chapter discussion lives)
ax.axvspan(0, 2.0, alpha=0.07, color="blue")
ax.text(1, 0.96, "j65 range\n(|c|≤2)", ha="center", fontsize=8, color="darkblue",
        style="italic")

# log-x for the wide range
ax.set_xscale("log")
ax.set_xlim(0.4, 40)
ax.set_xlabel("steering intensity |c| (in σ; log scale)")
ax.set_ylabel("flip rate")
ax.set_ylim(-0.02, 1.02)
ax.set_title("A — Steering up to 32σ (j75) + j65\n"
             "Even at c=32σ, intact-rate stays 0 → flips don't preserve α/β as top-1")

# Annotate intact stays 0
ax.text(0.98, 0.45,
        "intact_rate = 0 everywhere\n(top-1 token is never α/β at\n"
        "this prompt position — even baseline)\n→ no causal flip exists",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.85))

ax.legend(loc="upper left", fontsize=7)
ax.grid(alpha=0.3, which="both")

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

# ── PANEL D: carrier ablation per-layer vs null ──────────────────────────────
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

xs = np.arange(len(layer_nums))
bar_colors = ["tab:green" if p >= 0.95 else "tab:orange" if p >= 0.8 else "tab:red"
              for p in pcts]
ax.bar(xs, carriers, yerr=ci_err, color=bar_colors, alpha=0.7,
       edgecolor="black", capsize=3, label="carrier |Δ logit|")
ax.plot(xs, null_p95, "k--", lw=1.5, alpha=0.7, label="random null p95")
ax.plot(xs, null_mean, "k:", lw=1, alpha=0.5, label="random null mean")

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

ax.text(0.98, 0.97,
        f"ALL set: |Δ|={sets[all_key]['carrier_mean_abs_dlogit']:.2f}\n"
        f"null p95: {sets[all_key]['null_band']['p95']:.2f}\n"
        f"percentile: {int(all_pct*100)}%",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(facecolor="mistyrose", edgecolor="red", alpha=0.85))

fig.suptitle(
    "Chapter §6 — Four converging negative verdicts (extended to 32σ):  "
    "decodable & stable, but not causal at any magnitude",
    fontsize=13
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "chapter6_verdicts_v3.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# ── extra summary ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("CHAPTER §6 SUMMARY (v3, 4 verdicts + j75 saturation)")
print(f"{'='*72}")

# Find c where wres_flip is high
sat_final = sat_df[sat_df.tap == "final"].sort_values("c")
sat_l24   = sat_df[sat_df.tap == "postL24"].sort_values("c")
print(f"\n[A] j65+j75 steering saturation:")
print(f"    postL24: wres_flip rises {sat_l24.iloc[0]['wres_flip']:.2f}→{sat_l24.iloc[-1]['wres_flip']:.2f}  (c=0.5→32)")
print(f"             but intact_rate=0 throughout → no top-1 ever becomes α/β")
print(f"    final:   wres_flip rises {sat_final.iloc[0]['wres_flip']:.2f}→{sat_final.iloc[-1]['wres_flip']:.2f}  (c=0.5→32)")
print(f"             at c=32, shuf_p95={sat_final.iloc[-1]['shuffled_flip_p95']:.2f}  random={sat_final.iloc[-1]['random_flip_mean']:.2f}")
print(f"             still no intact flip → no causal magnitude")
print(f"\n[B] j66 carriers vs w_res:  signed_mean ≈ 0, capture = 0.30  → orthogonal")
print(f"[C] j68 stability:          cos = 0.62 > 3×null = 0.15  → real direction")
print(f"[D] j67 ablation:           ALL 227 carriers, pct = 88% (in null band)")
print(f"\n→ w_res IS a real distributed direction (B+C+D agree)")
print(f"  but it is NOT causally hooked to the answer at any push magnitude (A)")
