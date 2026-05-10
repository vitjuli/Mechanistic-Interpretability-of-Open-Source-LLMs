"""
Evaluate clean cross-domain v2 (Family D-v2) against old Family D.

Steps:
  1. Load physics train prompts (384) and clean D-v2 (124)
  2. Extract hidden states at L34, L35, L36 using Qwen3-4B BASE
  3. Train LogisticRegression probes on physics hidden states at each layer
  4. Apply probe to clean D-v2 → transfer accuracy by layer
  5. Compute behavioural accuracy from logp(' intensive') - logp(' extensive')
  6. Load old Family D results from ie_transfer_readout.csv for comparison
  7. Output CSV, markdown report, comparison plot

Outputs:
  data/results/abstraction_ie/clean_cross_domain_v2/
    eval_results.csv
    probe_by_layer.csv
    summary.json
  docs/clean_cross_domain_v2_results.md
  (plot at data/results/abstraction_ie/clean_cross_domain_v2/comparison_plot.png)

Usage:
    python scripts/74_eval_clean_cross_domain.py --device cuda
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src.model_utils import ModelWrapper

PHYS_PROMPT_PATH = Path("data/prompts/physics_intensive_extensive_v1_train.jsonl")
D2_PROMPT_PATH   = Path("data/prompts/abstraction/clean_cross_domain_v2.jsonl")
OLD_D_CSV        = Path("data/results/abstraction_ie/physics_intensive_extensive_v1/ie_transfer_readout.csv")
OUT_DIR          = Path("data/results/abstraction_ie/clean_cross_domain_v2")
DOCS_DIR         = Path("docs")
MODEL_NAME       = "Qwen/Qwen3-4B"
INT_TOKEN        = 36195
EXT_TOKEN        = 16376
LABEL_MAP        = {"intensive": 0, "extensive": 1}
PROBE_LAYERS     = [31, 32, 33, 34, 35, 36]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def nd_from_logits(logits):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(lp[INT_TOKEN] - lp[EXT_TOKEN])


# ── Hidden-state extraction ───────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts, layers, device):
    """
    Returns dict: layer -> np.ndarray (n_prompts, d_model)
    """
    layer_states = {l: [] for l in layers}
    nd_vals      = []

    for i, row in enumerate(prompts):
        if i % 50 == 0:
            print(f"    {i}/{len(prompts)}", end="\r", flush=True)
        prompt = row["prompt"] if isinstance(row, dict) else row
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out    = model.model(**inputs, output_hidden_states=True, use_cache=False)
        for l in layers:
            h = out.hidden_states[l][0, -1, :].float().cpu().numpy()
            layer_states[l].append(h)
        nd_vals.append(nd_from_logits(out.logits[0, -1]))

    print()
    return {l: np.stack(layer_states[l]) for l in layers}, np.array(nd_vals)


# ── Probe training and evaluation ─────────────────────────────────────────────

def train_probe(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_scaled, y_train)
    return clf, scaler


def apply_probe(clf, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    return clf.predict(X_scaled)


# ── Comparison plot ───────────────────────────────────────────────────────────

def make_comparison_plot(probe_df, old_d_probe, old_d_output,
                         d2_probe_by_layer, d2_output_acc, out_dir):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: probe accuracy by layer — old D vs clean D-v2
    ax = axes[0]
    layers = probe_df["layer"].tolist()
    ax.plot(layers, probe_df["probe_acc"].tolist(),
            "b-o", label="Probe — clean D-v2", linewidth=2, markersize=5)

    # Add old D as a horizontal line (from ie_transfer_readout summary)
    ax.axhline(old_d_probe, color="royalblue", linestyle="--", linewidth=1.5,
               label=f"Probe — old Family D (L34={old_d_probe:.3f})", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.50)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Probe accuracy")
    ax.set_title("Transfer probe accuracy: old Family D vs clean D-v2")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.05)

    # Right: behavioural vs probe at each layer for clean D-v2
    ax = axes[1]
    ax.plot(layers, probe_df["probe_acc"].tolist(),
            "b-o", label="Probe acc (clean D-v2)", linewidth=2, markersize=5)
    ax.axhline(d2_output_acc, color="orange", linestyle="--", linewidth=2,
               label=f"Behavioural acc clean D-v2 ({d2_output_acc:.3f})", alpha=0.9)
    ax.axhline(old_d_probe,  color="royalblue", linestyle=":", linewidth=1.5,
               label=f"Old D probe acc (L34={old_d_probe:.3f})", alpha=0.7)
    ax.axhline(old_d_output, color="tomato", linestyle=":", linewidth=1.5,
               label=f"Old D behavioural acc ({old_d_output:.3f})", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy")
    ax.set_title("Probe vs behavioural: old D vs clean D-v2")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.05)

    fig.suptitle("Family D-v2 (clean) vs old Family D — Transfer Analysis", fontsize=12)
    fig.tight_layout()
    out = out_dir / "comparison_plot.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {out}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(d2_rows, probe_df, old_d_probe, old_d_output,
                 old_d_rf_rate, d2_output_acc, d2_probe_best, d2_rf_rate,
                 by_domain, by_wording, classification):
    lines = [
        "# Clean Cross-Domain v2 (Family D-v2) Results",
        "",
        "## Experiment",
        "",
        "**Family D-v2** replaces vague scaling language (\"scale up\", \"expand\") with formally",
        "matched scaling operations identical across all domains:",
        "- W1_duplicate: exact duplication, local ratios unchanged",
        "- W2_combine: two identical systems merged, per-unit quantities preserved",
        "- W3_split: split into two equal halves, per-unit quantities preserved",
        "- W4_ratio_fixed: double size, all ratios/rates/densities held constant",
        "",
        f"**Dataset**: 124 prompts (64 intensive, 60 extensive) across 4 domains",
        f"**Domains**: economics, biology, statistics, information theory",
        f"**Properties**: 31 (16 intensive + 15 extensive)",
        "",
        "## Results vs Old Family D",
        "",
        "| Metric | Old Family D | Clean D-v2 | Change |",
        "|--------|-------------|------------|--------|",
        f"| Probe acc (best layer) | {old_d_probe:.3f} | {d2_probe_best:.3f} | {d2_probe_best-old_d_probe:+.3f} |",
        f"| Behavioural acc | {old_d_output:.3f} | {d2_output_acc:.3f} | {d2_output_acc-old_d_output:+.3f} |",
        f"| Readout failure rate | {old_d_rf_rate:.3f} | {d2_rf_rate:.3f} | {d2_rf_rate-old_d_rf_rate:+.3f} |",
        "",
        "## Probe Accuracy by Layer (Clean D-v2)",
        "",
        probe_df.to_markdown(index=False),
        "",
        "## Behavioural Accuracy by Domain",
        "",
        pd.DataFrame(by_domain).T.to_markdown(),
        "",
        "## Behavioural Accuracy by Wording Family",
        "",
        pd.DataFrame(by_wording).T.to_markdown(),
        "",
        "## Readout Failure Analysis",
        "",
        "Readout failure = probe correct, output wrong",
        "",
    ]

    rf_rows = [r for r in d2_rows if r.get("readout_failure")]
    if rf_rows:
        lines += [
            f"Total readout failures: {len(rf_rows)}/{len(d2_rows)} ({len(rf_rows)/len(d2_rows):.1%})",
            "",
            "| Domain | Property | Class | Wording | nd_output |",
            "|--------|----------|-------|---------|-----------|",
        ]
        for r in rf_rows[:15]:
            lines.append(
                f"| {r['domain']} | {r['property']} | {r['abstraction_class']} | {r['wording_family']} | {r.get('nd_output',0):.3f} |"
            )

    lines += [
        "",
        f"## Classification",
        "",
        f"**{classification}**",
        "",
        "### Criteria",
        "- **Strong domain-general**: probe > 0.80 AND behavioural > 0.75 for clean D-v2",
        "- **Representation-only**: probe > 0.70 but behavioural ≤ 0.60",
        "- **Partial transfer**: probe 0.60–0.80, behavioural 0.55–0.75",
        "- **Domain-specific**: probe ≤ 0.60 or behav ≤ 0.55 for clean D-v2",
        "",
        "*Data: data/results/abstraction_ie/clean_cross_domain_v2/*",
    ]

    path = DOCS_DIR / "clean_cross_domain_v2_results.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",   default="bfloat16", choices=["float32","bfloat16","float16"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ─────────────────────────────────────────────────────────
    phys_rows = load_jsonl(PHYS_PROMPT_PATH)
    d2_rows   = load_jsonl(D2_PROMPT_PATH)
    print(f"Physics train: {len(phys_rows)} | Clean D-v2: {len(d2_rows)}")

    phys_labels = np.array([LABEL_MAP[r["abstraction_class"]] for r in phys_rows])
    d2_labels   = np.array([LABEL_MAP[r["abstraction_class"]] for r in d2_rows])

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model…")
    model = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    model.model.eval()
    tok = model.tokenizer

    # ── Extract hidden states — physics train ─────────────────────────────────
    print(f"Extracting physics hidden states at layers {PROBE_LAYERS}…")
    phys_hs, phys_nd = extract_hidden_states(model, tok, phys_rows, PROBE_LAYERS, device)
    print(f"  Physics ND: mean={phys_nd.mean():.3f}  sign_acc={float((np.sign(phys_nd)==(1-2*phys_labels)).mean()):.3f}")

    # ── Extract hidden states — clean D-v2 ───────────────────────────────────
    print(f"Extracting clean D-v2 hidden states at layers {PROBE_LAYERS}…")
    d2_hs, d2_nd = extract_hidden_states(model, tok, d2_rows, PROBE_LAYERS, device)

    # Behavioural accuracy: sign(nd) == +1 → intensive (label 0), -1 → extensive (label 1)
    # nd > 0 → predict intensive (label 0), nd < 0 → predict extensive (label 1)
    d2_pred_output = (d2_nd < 0).astype(int)   # 1=extensive, 0=intensive
    d2_output_acc  = float((d2_pred_output == d2_labels).mean())
    print(f"  Clean D-v2 behavioural acc: {d2_output_acc:.4f}")

    # ── Train probes and measure transfer ────────────────────────────────────
    probe_rows = []
    for l in PROBE_LAYERS:
        clf, scaler = train_probe(phys_hs[l], phys_labels)
        d2_pred_probe = apply_probe(clf, scaler, d2_hs[l])
        probe_acc     = float((d2_pred_probe == d2_labels).mean())

        # By domain
        by_domain_probe = {}
        for dom, idxs in [(d, [i for i,r in enumerate(d2_rows) if r["domain"]==d])
                           for d in sorted(set(r["domain"] for r in d2_rows))]:
            if idxs:
                by_domain_probe[dom] = float((d2_pred_probe[idxs] == d2_labels[idxs]).mean())

        probe_rows.append({"layer": l, "probe_acc": round(probe_acc, 4),
                           **{f"probe_{k.replace(' ','_')}": round(v, 4)
                              for k, v in by_domain_probe.items()}})
        print(f"  L{l}: probe_acc={probe_acc:.4f} | by_domain: {by_domain_probe}")

    probe_df  = pd.DataFrame(probe_rows)
    best_probe_row = probe_df.loc[probe_df["probe_acc"].idxmax()]
    best_layer     = int(best_probe_row["layer"])
    best_probe_acc = float(best_probe_row["probe_acc"])

    # ── Per-prompt results using best probe layer ─────────────────────────────
    clf_best, scaler_best = train_probe(phys_hs[best_layer], phys_labels)
    d2_pred_probe_best    = apply_probe(clf_best, scaler_best, d2_hs[best_layer])

    result_rows = []
    for i, row in enumerate(d2_rows):
        true_cls       = row["abstraction_class"]
        true_label     = LABEL_MAP[true_cls]
        pred_probe     = int(d2_pred_probe_best[i])
        pred_output    = int(d2_pred_output[i])
        correct_probe  = pred_probe  == true_label
        correct_output = pred_output == true_label
        rf             = correct_probe and not correct_output

        result_rows.append({
            "domain":           row["domain"],
            "property":         row["property"],
            "wording_family":   row["wording_family"],
            "abstraction_class":true_cls,
            "nd_output":        round(float(d2_nd[i]), 4),
            "pred_output":      "intensive" if pred_output == 0 else "extensive",
            "correct_output":   correct_output,
            "pred_probe":       "intensive" if pred_probe == 0 else "extensive",
            "correct_probe":    correct_probe,
            "readout_failure":  rf,
            "probe_layer":      best_layer,
        })

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(OUT_DIR / "eval_results.csv", index=False)
    probe_df.to_csv(OUT_DIR / "probe_by_layer.csv", index=False)

    # ── By-domain and by-wording breakdown ───────────────────────────────────
    def group_stats(df, col):
        out = {}
        for val, grp in df.groupby(col):
            out[val] = {
                "n":            len(grp),
                "output_acc":   round(float(grp["correct_output"].mean()), 3),
                "probe_acc":    round(float(grp["correct_probe"].mean()), 3),
                "rf_rate":      round(float(grp["readout_failure"].mean()), 3),
            }
        return out

    by_domain  = group_stats(results_df, "domain")
    by_wording = group_stats(results_df, "wording_family")
    by_class   = group_stats(results_df, "abstraction_class")

    d2_rf_rate = float(results_df["readout_failure"].mean())

    print("\n=== CLEAN D-v2 RESULTS ===")
    print(f"Behavioural acc: {d2_output_acc:.4f}")
    print(f"Best probe acc:  {best_probe_acc:.4f} (L{best_layer})")
    print(f"Readout failure: {d2_rf_rate:.4f} ({int(results_df['readout_failure'].sum())}/{len(results_df)})")
    print("\nBy domain (output | probe | RF):")
    for dom, s in by_domain.items():
        print(f"  {dom:20s}: output={s['output_acc']:.3f}  probe={s['probe_acc']:.3f}  RF={s['rf_rate']:.3f}")
    print("\nBy wording (output | probe | RF):")
    for wf, s in by_wording.items():
        print(f"  {wf:20s}: output={s['output_acc']:.3f}  probe={s['probe_acc']:.3f}  RF={s['rf_rate']:.3f}")
    print("\nBy class (output | probe | RF):")
    for cls, s in by_class.items():
        print(f"  {cls:12s}: output={s['output_acc']:.3f}  probe={s['probe_acc']:.3f}  RF={s['rf_rate']:.3f}")

    # ── Load old Family D for comparison ─────────────────────────────────────
    old_d_probe  = 0.636   # from ie_transfer_readout.csv summary
    old_d_output = 0.564
    old_d_rf_rate = 0.309
    if OLD_D_CSV.exists():
        old = pd.read_csv(OLD_D_CSV)
        old_d_probe   = float(old["correct_probe"].mean())
        old_d_output  = float(old["correct_output"].mean())
        old_d_rf_rate = float(old["readout_failure"].mean())

    # ── Classification ───────────────────────────────────────────────────────
    if best_probe_acc > 0.80 and d2_output_acc > 0.75:
        classification = "Strong domain-general abstraction"
    elif best_probe_acc > 0.70 and d2_output_acc <= 0.62:
        classification = "Representation-only abstraction (probe transfers, output fails)"
    elif best_probe_acc >= 0.60 and d2_output_acc >= 0.55:
        classification = "Partial transfer (both probe and output above chance)"
    else:
        classification = "Domain-specific abstraction (poor cross-domain transfer)"

    print(f"\nClassification: {classification}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "n_prompts":          len(d2_rows),
        "n_intensive":        int((d2_labels == 0).sum()),
        "n_extensive":        int((d2_labels == 1).sum()),
        "best_probe_layer":   best_layer,
        "best_probe_acc":     round(best_probe_acc, 4),
        "output_acc":         round(d2_output_acc, 4),
        "rf_rate":            round(d2_rf_rate, 4),
        "rf_n":               int(results_df["readout_failure"].sum()),
        "old_d_probe_acc":    round(old_d_probe, 4),
        "old_d_output_acc":   round(old_d_output, 4),
        "old_d_rf_rate":      round(old_d_rf_rate, 4),
        "probe_delta":        round(best_probe_acc - old_d_probe, 4),
        "output_delta":       round(d2_output_acc - old_d_output, 4),
        "by_domain":          by_domain,
        "by_wording":         by_wording,
        "by_class":           by_class,
        "classification":     classification,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    make_comparison_plot(probe_df, old_d_probe, old_d_output,
                         best_probe_acc, d2_output_acc, OUT_DIR)

    # ── Report ───────────────────────────────────────────────────────────────
    write_report(result_rows, probe_df, old_d_probe, old_d_output,
                 old_d_rf_rate, d2_output_acc, best_probe_acc, d2_rf_rate,
                 by_domain, by_wording, classification)

    print("\nDone.")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
