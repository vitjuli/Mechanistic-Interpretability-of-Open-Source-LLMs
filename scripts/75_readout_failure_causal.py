"""
Causal test: do C0/C4 cause cross-domain readout failures?

A readout failure = probe(L34) predicts correct class, but model output is wrong.
We test whether ablating C0/C4 (extensive-supporting clusters) rescues intensive
readout failures, and whether it worsens extensive prompts — confirming C0/C4 as
causal readout-bias clusters, not merely physics-domain correlates.

Protocol:
  1. Load readout failures from old Family D (ie_transfer_readout.csv)
     + clean D-v2 if script 74 has been run (clean_cross_domain_v2/eval_results.csv)
  2. For each readout-failure prompt, run:
       • C0 ablation
       • C4 ablation
       • C0+C4 joint ablation
       • C1 ablation (control — general-scaling cluster)
       • C4 amplification (steer toward extensive, to test directionality)
  3. Also run ablation on extensive prompts (not readout failures) to measure worsening
  4. Report: rescue rate, worsening rate, steering direction
  5. Compare by domain and wording family

Key causal predictions:
  - C4 ablation should RESCUE intensive readout failures (flip incorrect→correct)
  - C4 ablation should WORSEN extensive prompts (correct→incorrect)
  - C4 amplification should WORSEN intensive readout failures further
  - C1 ablation should have no systematic effect

Outputs:
  data/results/abstraction_ie/readout_failure_causal/
    readout_failure_interventions.csv
    extensive_worsening.csv
    summary.json
  docs/readout_failure_causal_results.md
  plots: nd_scatter.png, rescue_worsening.png, steering.png

Usage:
    python scripts/75_readout_failure_causal.py --device cuda
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

OLD_D_CSV    = Path("data/results/abstraction_ie/physics_intensive_extensive_v1/ie_transfer_readout.csv")
D2_CSV       = Path("data/results/abstraction_ie/clean_cross_domain_v2/eval_results.csv")
D2_JSONL     = Path("data/prompts/abstraction/clean_cross_domain_v2.jsonl")
CLUSTER_DIR  = Path("data/results/abstraction_ie/physics_intensive_extensive_v1")
OUT_DIR      = Path("data/results/abstraction_ie/readout_failure_causal")
DOCS_DIR     = Path("docs")
MODEL_NAME   = "Qwen/Qwen3-4B"
MODEL_SIZE   = "4b"
INT_TOKEN    = 36195
EXT_TOKEN    = 16376
LABEL_MAP    = {"intensive": 0, "extensive": 1}

CLUSTERS_TO_TEST = {
    "C0":    0,
    "C4":    4,
    "C0C4":  (0, 4),   # joint
    "C1":    1,        # control
}
AMPLIFY_ALPHA = 2.5   # factor for C4 amplification (steering)


# ── Load cluster definitions ──────────────────────────────────────────────────

def load_cluster_by_layer(k=6, split="train"):
    labels_path = CLUSTER_DIR / f"ie_cluster_labels_k{k}.npy"
    meta_path   = CLUSTER_DIR / "ie_feature_meta.json"

    cluster_labels = np.load(labels_path)

    if meta_path.exists():
        with open(meta_path) as f:
            feat_meta = json.load(f)
    else:
        # rebuild from transcoder_features
        tc_base = Path("data/results/transcoder_features")
        layer_dirs = sorted(tc_base.glob("layer_*"), key=lambda p: int(p.name.split("_")[1]))
        behaviour = "physics_intensive_extensive_v1"
        feat_meta = []
        for ld in layer_dirs:
            idx_path = ld / f"{behaviour}_{split}_top_k_indices.npy"
            if not idx_path.exists(): continue
            layer_idx = int(ld.name.split("_")[1])
            indices = np.load(idx_path)
            for fi in np.unique(indices):
                feat_meta.append({"layer": layer_idx, "feature_idx": int(fi)})

    # build: cluster_id -> {layer -> [feature_indices]}
    clusters = {}
    n = min(len(cluster_labels), len(feat_meta))
    for i in range(n):
        c = int(cluster_labels[i])
        m = feat_meta[i]
        clusters.setdefault(c, {}).setdefault(m["layer"], []).append(m["feature_idx"])

    return clusters


# ── Transcoder / ablation utilities ──────────────────────────────────────────

def _get_norm(model_inner, layer_idx):
    return model_inner.model.layers[layer_idx].post_attention_layernorm


@torch.no_grad()
def capture_last_token_hs(model, inputs, layers):
    captured, handles = {}, []
    for l in layers:
        norm = _get_norm(model.model, l)
        def _h(m, inp, out, _l=l):
            x = out[0] if isinstance(out, tuple) else out
            captured[_l] = x[:, -1, :].detach().clone()
        handles.append(norm.register_forward_hook(_h))
    out = model.model(**inputs, use_cache=False)
    for h in handles: h.remove()
    return captured, out.logits[0, -1, :]


def nd_from_logits(logits):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(lp[INT_TOKEN] - lp[EXT_TOKEN])


@torch.no_grad()
def run_intervention(model, transcoder_set, inputs, by_layer, mode="ablate", alpha=2.5):
    """
    mode="ablate"  → zero out cluster features
    mode="amplify" → multiply cluster features by alpha (steering toward extensive)
    Returns (nd_baseline, nd_modified)
    """
    all_layers = sorted(by_layer.keys())
    base_acts, base_logits = capture_last_token_hs(model, inputs, all_layers)

    modified = {}
    for l, fidxs in by_layer.items():
        tc    = transcoder_set[l]
        act   = base_acts[l]
        feats = tc.encode(act.to(tc.dtype))
        if mode == "ablate":
            feats[:, fidxs] = 0.0
        else:  # amplify
            feats[:, fidxs] *= alpha
        modified[l] = tc.decode(feats).to(act.dtype)

    handles = []
    for l, new_act in sorted(modified.items()):
        norm = _get_norm(model.model, l)
        def _hook(m, inp, out, _new=new_act):
            if isinstance(out, tuple):
                lst = list(out); lst[0][:, -1] = _new; return tuple(lst)
            out[:, -1] = _new; return out
        handles.append(norm.register_forward_hook(_hook))

    mod_out = model.model(**inputs, use_cache=False)
    for h in handles: h.remove()

    return nd_from_logits(base_logits), nd_from_logits(mod_out.logits[0, -1, :])


# ── Load prompts with readout failure info ────────────────────────────────────

def load_rf_prompts():
    """Load readout-failure prompts from old Family D and clean D-v2."""
    rf_prompts  = []   # (prompt_text, true_cls, domain, property, wording, source)
    ext_prompts = []   # extensive non-failures for worsening test

    # Old Family D
    if OLD_D_CSV.exists():
        old = pd.read_csv(OLD_D_CSV)
        for _, r in old.iterrows():
            entry = {
                "prompt": r["prompt"],
                "abstraction_class": r["abstraction_class"],
                "domain":   r.get("domain", "unknown"),
                "property": r.get("property", ""),
                "wording_family": r.get("wording_family", ""),
                "source":   "old_family_d",
                "nd_baseline": r.get("nd_output", 0.0),
            }
            if r["readout_failure"]:
                rf_prompts.append(entry)
            elif r["abstraction_class"] == "extensive" and r["correct_output"]:
                ext_prompts.append(entry)

    # Clean D-v2 (if available)
    if D2_CSV.exists() and D2_JSONL.exists():
        d2_res = pd.read_csv(D2_CSV)
        d2_raw = {i: json.loads(l) for i, l in enumerate(open(D2_JSONL))}
        # sort: d2_raw is shuffled, need to match by index
        d2_raw_list = [json.loads(l) for l in open(D2_JSONL)]
        # match by prompt text
        prompt_to_raw = {r["prompt"]: r for r in d2_raw_list}
        for _, r in d2_res.iterrows():
            raw = prompt_to_raw.get(r.get("prompt", ""), {})
            entry = {
                "prompt": raw.get("prompt", ""),
                "abstraction_class": r["abstraction_class"],
                "domain":   r.get("domain", ""),
                "property": r.get("property", ""),
                "wording_family": r.get("wording_family", ""),
                "source":   "clean_d2",
                "nd_baseline": r.get("nd_output", 0.0),
            }
            if not entry["prompt"]:
                continue
            if r["readout_failure"]:
                rf_prompts.append(entry)
            elif r["abstraction_class"] == "extensive" and r["correct_output"]:
                ext_prompts.append(entry)

    print(f"Readout failures: {len(rf_prompts)} | Extensive non-failures: {len(ext_prompts)}")
    return rf_prompts, ext_prompts


# ── Main intervention loop ────────────────────────────────────────────────────

def run_intervention_sweep(model, tok, transcoder_set, prompts, clusters,
                            device, include_amplify=True):
    rows = []
    for pi, prow in enumerate(prompts):
        if pi % 5 == 0:
            print(f"  {pi+1}/{len(prompts)}", end="\r", flush=True)
        prompt_text = prow["prompt"]
        inputs      = tok(prompt_text, return_tensors="pt")
        inputs      = {k: v.to(device) for k, v in inputs.items()}
        true_cls    = prow["abstraction_class"]
        true_label  = LABEL_MAP[true_cls]

        base_row = {
            "prompt_idx":   pi,
            "domain":       prow.get("domain", ""),
            "property":     prow.get("property", ""),
            "wording_family": prow.get("wording_family", ""),
            "abstraction_class": true_cls,
            "source":       prow.get("source", ""),
        }

        for cluster_name, cluster_spec in CLUSTERS_TO_TEST.items():
            # Build by_layer for this cluster (or joint clusters)
            if isinstance(cluster_spec, tuple):
                by_layer = {}
                for cid in cluster_spec:
                    for l, fidxs in clusters.get(cid, {}).items():
                        by_layer.setdefault(l, []).extend(fidxs)
            else:
                by_layer = dict(clusters.get(cluster_spec, {}))

            if not by_layer:
                continue

            try:
                nd_base, nd_abl = run_intervention(model, transcoder_set, inputs,
                                                    by_layer, mode="ablate")
            except Exception as e:
                print(f"\n    [WARN] ablate {cluster_name}: {e}")
                continue

            pred_base = "intensive" if nd_base > 0 else "extensive"
            pred_abl  = "intensive" if nd_abl  > 0 else "extensive"
            correct_base = (pred_base == true_cls)
            correct_abl  = (pred_abl  == true_cls)

            ablation_row = {**base_row,
                "cluster": cluster_name, "mode": "ablate",
                "nd_baseline": round(nd_base, 4),
                "nd_modified": round(nd_abl, 4),
                "delta_nd":    round(nd_abl - nd_base, 4),
                "pred_baseline": pred_base,
                "pred_modified": pred_abl,
                "correct_baseline": correct_base,
                "correct_modified": correct_abl,
                "rescue":   not correct_base and correct_abl,
                "worsen":   correct_base and not correct_abl,
                "no_change": pred_base == pred_abl,
            }
            rows.append(ablation_row)

            # C4 amplification only
            if include_amplify and cluster_name == "C4":
                try:
                    _, nd_amp = run_intervention(model, transcoder_set, inputs,
                                                  by_layer, mode="amplify",
                                                  alpha=AMPLIFY_ALPHA)
                except Exception as e:
                    print(f"\n    [WARN] amplify C4: {e}")
                    continue

                pred_amp   = "intensive" if nd_amp > 0 else "extensive"
                correct_amp = (pred_amp == true_cls)
                rows.append({**base_row,
                    "cluster": "C4", "mode": "amplify",
                    "nd_baseline": round(nd_base, 4),
                    "nd_modified": round(nd_amp, 4),
                    "delta_nd":    round(nd_amp - nd_base, 4),
                    "pred_baseline": pred_base,
                    "pred_modified": pred_amp,
                    "correct_baseline": correct_base,
                    "correct_modified": correct_amp,
                    "rescue":   not correct_base and correct_amp,
                    "worsen":   correct_base and not correct_amp,
                    "no_change": pred_base == pred_amp,
                })
    print()
    return pd.DataFrame(rows)


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_summary(rf_df, ext_df):
    summary = {}

    # Rescue rates (readout failures)
    for cluster_name in CLUSTERS_TO_TEST:
        for mode in ["ablate", "amplify"]:
            if mode == "amplify" and cluster_name != "C4":
                continue
            sub = rf_df[(rf_df["cluster"] == cluster_name) & (rf_df["mode"] == mode)]
            if sub.empty:
                continue
            key = f"{cluster_name}_{mode}"
            summary[key] = {
                "n_rf_prompts":   len(sub),
                "rescue_rate":    round(float(sub["rescue"].mean()), 4),
                "n_rescued":      int(sub["rescue"].sum()),
                "no_change_rate": round(float(sub["no_change"].mean()), 4),
                "mean_delta_nd":  round(float(sub["delta_nd"].mean()), 4),
                "type": "readout_failures",
            }

    # Worsening rates (extensive non-failures)
    for cluster_name in CLUSTERS_TO_TEST:
        sub = ext_df[(ext_df["cluster"] == cluster_name) & (ext_df["mode"] == "ablate")]
        if sub.empty:
            continue
        summary[f"{cluster_name}_ablate_ext_worsening"] = {
            "n_ext_prompts":  len(sub),
            "worsen_rate":    round(float(sub["worsen"].mean()), 4),
            "n_worsened":     int(sub["worsen"].sum()),
            "mean_delta_nd":  round(float(sub["delta_nd"].mean()), 4),
            "type": "extensive_worsening",
        }

    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(rf_df, ext_df, summary, out_dir):
    if not HAS_MPL:
        return

    # ── Plot 1: nd_baseline vs nd_modified for C0/C4 ablation on RF prompts ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cluster in zip(axes, ["C4", "C0"]):
        sub = rf_df[(rf_df["cluster"] == cluster) & (rf_df["mode"] == "ablate")]
        if sub.empty:
            ax.set_title(f"{cluster} — no data"); continue
        rescued = sub["rescue"].values
        ax.scatter(sub["nd_baseline"], sub["nd_modified"],
                   c=["#22c55e" if r else "#f87171" for r in rescued],
                   s=60, alpha=0.8)
        lim = max(abs(sub["nd_baseline"].max()), abs(sub["nd_modified"].max()), 1) * 1.1
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.plot([-lim, lim], [-lim, lim], "gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("nd baseline"); ax.set_ylabel("nd after ablation")
        ax.set_title(f"{cluster} ablation on readout failures\n"
                     f"(green=rescued, red=still wrong, n={len(sub)})")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle("ND before/after ablation — Readout Failure Prompts", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "nd_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: rescue/worsening bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    clusters = ["C0_ablate", "C4_ablate", "C0C4_ablate", "C1_ablate", "C4_amplify"]
    rescue_rates  = [summary.get(k, {}).get("rescue_rate", 0)  for k in clusters]
    worsen_rates  = [-summary.get(f"{k.split('_')[0]}_ablate_ext_worsening",{}).get("worsen_rate",0)
                     if "ablate" in k else 0 for k in clusters]
    x = np.arange(len(clusters))
    ax.bar(x - 0.2, rescue_rates, 0.35, label="Rescue rate (int RF)", color="#22c55e", alpha=0.8)
    ax.bar(x + 0.2, [-w for w in worsen_rates], 0.35,
           label="Worsening rate (ext prompts)", color="#f87171", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(clusters, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Rate"); ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Rescue rate (readout failures) vs Worsening rate (extensive prompts)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "rescue_worsening.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: C4 ablate vs amplify on RF prompts ────────────────────────────
    abl_sub = rf_df[(rf_df["cluster"]=="C4") & (rf_df["mode"]=="ablate")]
    amp_sub = rf_df[(rf_df["cluster"]=="C4") & (rf_df["mode"]=="amplify")]
    if not abl_sub.empty and not amp_sub.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (sub, title) in zip(axes, [(abl_sub, "C4 Ablate (zero)"),
                                             (amp_sub, f"C4 Amplify (×{AMPLIFY_ALPHA})")]):
            agg = sub.groupby("abstraction_class")["delta_nd"].mean()
            colors = ["#60a5fa" if c=="intensive" else "#f97316" for c in agg.index]
            ax.bar(agg.index, agg.values, color=colors, alpha=0.85)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_title(f"{title}\nMean Δnd by class")
            ax.set_ylabel("Mean Δnd (ablated − baseline)")
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("C4 Ablation vs Amplification Direction", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "steering.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(rf_df, ext_df, summary, out_dir):
    lines = [
        "# Readout Failure Causal Test — C0/C4 Cluster Interventions",
        "",
        "## Setup",
        "",
        "**Readout failures**: cross-domain prompts where L34 linear probe predicts correct class",
        "but model output (logit difference) is wrong.",
        "",
        f"**Intensive readout failures**: {int((rf_df['abstraction_class']=='intensive').sum()//len(CLUSTERS_TO_TEST))} prompts",
        f"**Sources**: old Family D + clean D-v2 (if available)",
        "",
        "## Predictions",
        "",
        "If C0/C4 are causal readout-bias clusters:",
        "- **C4 ablation should rescue** intensive readout failures (flip incorrect→correct)",
        "- **C4 ablation should worsen** extensive prompts (correct→incorrect)",
        "- **C4 amplification should worsen** intensive readout failures further",
        "- **C1 ablation** (control) should have no systematic rescue/worsening",
        "",
        "## Results",
        "",
        "### Rescue rates (readout failure prompts, ablation)",
        "",
        "| Cluster | Mode | n_RF | Rescue rate | n_rescued | Mean Δnd |",
        "|---------|------|------|-------------|-----------|----------|",
    ]

    for cluster_name in ["C0", "C4", "C0C4", "C1"]:
        for mode in ["ablate", "amplify"]:
            if mode == "amplify" and cluster_name != "C4":
                continue
            key = f"{cluster_name}_{mode}"
            s = summary.get(key)
            if s:
                lines.append(
                    f"| {cluster_name} | {mode} | {s['n_rf_prompts']} | "
                    f"{s['rescue_rate']:.3f} | {s['n_rescued']} | {s['mean_delta_nd']:+.3f} |"
                )

    lines += [
        "",
        "### Worsening rates (extensive non-failure prompts, ablation only)",
        "",
        "| Cluster | n_ext | Worsen rate | n_worsened | Mean Δnd |",
        "|---------|-------|-------------|------------|----------|",
    ]
    for cluster_name in ["C0", "C4", "C0C4", "C1"]:
        key = f"{cluster_name}_ablate_ext_worsening"
        s = summary.get(key)
        if s:
            lines.append(
                f"| {cluster_name} | {s['n_ext_prompts']} | {s['worsen_rate']:.3f} | "
                f"{s['n_worsened']} | {s['mean_delta_nd']:+.3f} |"
            )

    # Classification
    c4_rescue  = summary.get("C4_ablate", {}).get("rescue_rate", 0)
    c4_worsen  = summary.get("C4_ablate_ext_worsening", {}).get("worsen_rate", 0)
    c1_rescue  = summary.get("C1_ablate", {}).get("rescue_rate", 0)
    c4_amp     = summary.get("C4_amplify", {}).get("rescue_rate", 0)

    if c4_rescue > 0.40 and c4_worsen > 0.30 and c1_rescue < 0.15:
        verdict = "Strong causal evidence: C4 is a readout-bias cluster cross-domain"
    elif c4_rescue > 0.25 and c4_rescue > c1_rescue + 0.10:
        verdict = "Moderate causal evidence: C4 preferentially rescues intensive failures"
    elif c4_rescue <= 0.15 and c1_rescue <= 0.15:
        verdict = "No causal readout evidence: neither C4 nor C1 systematically rescues failures"
    else:
        verdict = "Weak causal evidence: C4 effect present but not clearly cluster-specific"

    lines += [
        "",
        "## Classification",
        "",
        f"**{verdict}**",
        "",
        f"C4 rescue rate: {c4_rescue:.3f} | C4 worsening: {c4_worsen:.3f} | C1 rescue (control): {c1_rescue:.3f}",
        "",
        "### Directional Test (C4 ablate vs amplify)",
        "",
        f"C4 ablation rescue rate: {c4_rescue:.3f} (expected: pushes toward intensive = rescues int failures)",
        f"C4 amplify rescue rate: {c4_amp:.3f} (expected: pushes toward extensive = fails to rescue)",
        "",
        "*Data: data/results/abstraction_ie/readout_failure_causal/*",
    ]

    path = DOCS_DIR / "readout_failure_causal_results.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",   default="bfloat16", choices=["float32","bfloat16","float16"])
    ap.add_argument("--k",       type=int, default=6)
    ap.add_argument("--no_amplify", action="store_true",
                    help="Skip C4 amplification (faster)")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load cluster definitions ──────────────────────────────────────────────
    print("Loading cluster definitions…")
    clusters = load_cluster_by_layer(k=args.k)
    for cid, by_layer in sorted(clusters.items()):
        n_feats = sum(len(v) for v in by_layer.values())
        print(f"  C{cid}: {n_feats} features, layers {sorted(by_layer.keys())}")

    # ── Load prompts ──────────────────────────────────────────────────────────
    rf_prompts, ext_prompts = load_rf_prompts()
    if not rf_prompts:
        print("No readout failure prompts found. Run script 71 and/or script 74 first.")
        sys.exit(1)

    # Limit extensive prompts to same count as RF for speed
    ext_prompts = ext_prompts[:max(len(rf_prompts) * 2, 30)]
    print(f"Using {len(rf_prompts)} RF prompts + {len(ext_prompts)} extensive prompts")

    # ── Load model + transcoders ──────────────────────────────────────────────
    all_layers = sorted(set(l for by_layer in clusters.values() for l in by_layer))
    print(f"Loading model + transcoders (layers {all_layers[0]}–{all_layers[-1]})…")
    model = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    model.model.eval()
    tok = model.tokenizer

    transcoder_set = load_transcoder_set(
        model_size=MODEL_SIZE, device=device, dtype=dtype,
        lazy_load=True, layers=all_layers,
    )

    # ── Run interventions on RF prompts ───────────────────────────────────────
    print("\nRunning interventions on readout-failure prompts…")
    rf_df = run_intervention_sweep(
        model, tok, transcoder_set, rf_prompts, clusters, device,
        include_amplify=not args.no_amplify
    )
    rf_df.to_csv(OUT_DIR / "readout_failure_interventions.csv", index=False)

    # ── Run ablations on extensive prompts (worsening test) ───────────────────
    print("\nRunning ablations on extensive prompts (worsening test)…")
    ext_df = run_intervention_sweep(
        model, tok, transcoder_set, ext_prompts, clusters, device,
        include_amplify=False
    )
    ext_df.to_csv(OUT_DIR / "extensive_worsening.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = compute_summary(rf_df, ext_df)

    print("\n=== READOUT FAILURE CAUSAL SUMMARY ===")
    for key, s in summary.items():
        print(f"  {key}: rescue={s.get('rescue_rate','—'):.3f}  worsen={s.get('worsen_rate','—') if 'worsen_rate' in s else '—'}  Δnd={s.get('mean_delta_nd',0):+.3f}"
              if 'rescue_rate' in s else
              f"  {key}: worsen={s.get('worsen_rate',0):.3f}  Δnd={s.get('mean_delta_nd',0):+.3f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_plots(rf_df, ext_df, summary, OUT_DIR)

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(rf_df, ext_df, summary, OUT_DIR)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
