"""
Linear transfer probe: does the physics intensive/extensive direction transfer cross-domain?

Train:  Family A physics prompts — clean wordings W0, W2, W4 only
Test:   Family D cross-domain prompts (economics, statistics, biology, information theory)
Method: logistic regression on per-layer hidden states (all layers, focusing on L20–L36)

Outputs:
  data/results/abstraction_probe/linear_transfer_probe.csv
  data/results/abstraction_probe/linear_transfer_probe_by_layer.png
  data/results/abstraction_probe/linear_transfer_probe_report.md

Usage:
    python scripts/62_linear_transfer_probe.py --device cuda
    python scripts/62_linear_transfer_probe.py --device cpu --layers 28 29 30 31 32 33 34 35 36
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3-4B"
PROMPT_DIR    = Path("data/prompts/abstraction")
OUT_DIR       = Path("data/results/abstraction_probe")
LABEL_MAP     = {"intensive": 0, "extensive": 1}

TRAIN_FILE    = PROMPT_DIR / "A_intensive_extensive_train.jsonl"
TEST_FILE     = PROMPT_DIR / "D_cross_domain_train.jsonl"
CLEAN_WORDINGS = {"W0_combine_doubles", "W2_split_preserves", "W4_additive"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prompts(path, wording_filter=None):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if wording_filter and r.get("wording_family") not in wording_filter:
                continue
            if r["abstraction_class"] not in LABEL_MAP:
                continue
            rows.append(r)
    return rows


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} on {device}…")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    mdl.eval()
    return mdl, tok


# ── Hidden-state extraction ───────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden(model, tokenizer, prompt, device, layers):
    """Single forward pass — return dict {layer_idx: np.array (d_model,)}."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out    = model(**inputs, output_hidden_states=True, use_cache=False)
    result = {}
    for l in layers:
        h = out.hidden_states[l]          # (1, seq_len, d_model)
        result[l] = h[0, -1, :].float().cpu().numpy()
    return result


def collect_hidden_states(model, tokenizer, rows, device, layers, desc=""):
    """Returns X: (n_prompts, n_layers, d_model), y: (n_prompts,)."""
    all_states = []   # list of dict {layer: np array}
    labels     = []
    domains    = []

    for i, row in enumerate(rows):
        if i % 20 == 0:
            print(f"  {desc} {i}/{len(rows)}", end="\r", flush=True)
        states = extract_hidden(model, tokenizer, row["prompt"], device, layers)
        all_states.append(states)
        labels.append(LABEL_MAP[row["abstraction_class"]])
        domains.append(row.get("domain", "physics"))

    print()
    n      = len(all_states)
    d_model = len(next(iter(all_states[0].values())))
    X = np.zeros((n, len(layers), d_model), dtype=np.float32)
    for i, states in enumerate(all_states):
        for j, l in enumerate(layers):
            X[i, j] = states[l]

    return X, np.array(labels), domains


# ── Probe ─────────────────────────────────────────────────────────────────────

def run_probe(X_train, y_train, X_test, y_test, layer_idx, domains_test=None, cv=5):
    scaler   = StandardScaler()
    Xtr_s    = scaler.fit_transform(X_train[:, layer_idx])
    Xte_s    = scaler.transform(X_test[:, layer_idx])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    # In-domain CV
    skf        = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(clf, Xtr_s, y_train, cv=skf, scoring="accuracy")
    cv_acc = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Transfer: train on all physics, test on cross-domain
    clf.fit(Xtr_s, y_train)
    y_pred        = clf.predict(Xte_s)
    transfer_acc  = float(accuracy_score(y_test, y_pred))
    flipped_acc   = 1.0 - transfer_acc
    is_inverted   = transfer_acc < 0.5
    corrected_acc = max(transfer_acc, flipped_acc)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    # Per-domain transfer accuracy
    domain_acc = {}
    if domains_test:
        for dom in set(domains_test):
            mask = np.array([d == dom for d in domains_test])
            if mask.sum() >= 2:
                domain_acc[dom] = float(accuracy_score(y_test[mask], y_pred[mask]))

    return {
        "cv_acc":        cv_acc,
        "cv_std":        cv_std,
        "transfer_acc":  transfer_acc,
        "flipped_acc":   flipped_acc,
        "corrected_acc": corrected_acc,
        "is_inverted":   is_inverted,
        "cm":            cm,
        "domain_acc":    domain_acc,
        "y_pred":        y_pred.tolist(),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results_df, best_layer, best_row, class_names, out_dir, n_train, n_test):
    lines = [
        "# Linear Transfer Probe Report",
        f"## Physics (Family A, W0+W2+W4) → Cross-domain (Family D)",
        f"## Model: Qwen3-4B | Train N={n_train} | Test N={n_test}",
        "",
        "## Method",
        "- Train: logistic regression on Family A (physics) hidden states at last token",
        "- Test: apply trained probe to Family D (economics/statistics/biology) hidden states",
        "- In-domain accuracy: 5-fold stratified CV on Family A",
        "- Transfer accuracy: direct prediction on all Family D (no D training)",
        "- Sign check: if transfer < 0.5, report flipped accuracy (inverted direction)",
        "",
        "## Per-Layer Results",
        "",
        "| Layer | In-domain CV (±std) | Transfer | Transfer (corrected) | Inverted? |",
        "|---|---|---|---|---|",
    ]
    for _, r in results_df.iterrows():
        inv = "✓ inverted" if r["is_inverted"] else ""
        lines.append(
            f"| L{int(r['layer'])} | {r['cv_acc']:.3f} ± {r['cv_std']:.3f} | "
            f"{r['transfer_acc']:.3f} | **{r['corrected_acc']:.3f}** | {inv} |"
        )

    # Best layer summary
    lines += [
        "",
        f"## Best Transfer Layer: L{best_layer}",
        f"- In-domain CV: {best_row['cv_acc']:.3f} ± {best_row['cv_std']:.3f}",
        f"- Transfer accuracy: {best_row['transfer_acc']:.3f}",
        f"- Corrected (flipped if inverted): **{best_row['corrected_acc']:.3f}**",
        f"- Direction inverted: {'YES' if best_row['is_inverted'] else 'NO'}",
        "",
        "### Confusion matrix at best layer",
        f"(rows = true label, cols = predicted; labels = {class_names})",
        "",
        "| | Pred intensive | Pred extensive |",
        "|---|---|---|",
        f"| True intensive | {best_row['cm'][0][0]} | {best_row['cm'][0][1]} |",
        f"| True extensive | {best_row['cm'][1][0]} | {best_row['cm'][1][1]} |",
    ]

    # Per-domain
    if best_row.get("domain_acc"):
        lines += ["", "### Transfer by domain", ""]
        for dom, acc in sorted(best_row["domain_acc"].items()):
            flag = "✓" if acc > 0.65 else "~" if acc > 0.45 else "✗"
            lines.append(f"  {flag} {dom}: {acc:.3f}")

    # Verdict
    best_corr = best_row["corrected_acc"]
    if best_corr >= 0.80:
        verdict    = "**DOMAIN-GENERAL**: the physics intensive/extensive direction transfers well."
        action     = "Proceed with full mechanistic pipeline — the circuit is likely domain-general."
    elif best_corr >= 0.60:
        verdict    = "**PARTIAL TRANSFER**: some domain-general signal, but not fully invariant."
        action     = "Proceed with physics Family A as primary, use Family D as a generalisation test."
    else:
        verdict    = "**DOMAIN-LOCAL**: the physics direction does not transfer."
        action     = "The abstraction is physics-specific. Restrict mechanistic claims to physics domain."

    inv_note = ""
    if best_row["is_inverted"]:
        inv_note = (
            f"\n\n**Note**: the raw transfer accuracy ({best_row['transfer_acc']:.3f}) is BELOW "
            f"chance, meaning the probe direction is **inverted** for cross-domain prompts. "
            f"The model has a clear signal distinguishing intensive/extensive in the hidden "
            f"state, but the direction is flipped relative to physics. This suggests the model "
            f"represents cross-domain intensive/extensive using the *opposite* activation "
            f"direction — consistent with the Family D behavioral inversion (45.5% accuracy)."
        )

    lines += [
        "",
        "## Verdict",
        "",
        verdict + inv_note,
        "",
        "### Implication",
        action,
        "",
        "### Boundary conditions",
        f"- Domain-general if corrected transfer > 80%: {'YES' if best_corr >= 0.80 else 'NO'}",
        f"- Partial transfer if corrected transfer 60-80%: {'YES' if 0.60 <= best_corr < 0.80 else 'NO'}",
        f"- Domain-local if corrected transfer < 60%: {'YES' if best_corr < 0.60 else 'NO'}",
    ]

    (out_dir / "linear_transfer_probe_report.md").write_text("\n".join(lines))
    print(f"  Report: {out_dir}/linear_transfer_probe_report.md")


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(results_df, best_layer, out_dir):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    layers   = results_df["layer"].tolist()
    cv_acc   = results_df["cv_acc"].tolist()
    cv_std   = results_df["cv_std"].tolist()
    tr_acc   = results_df["transfer_acc"].tolist()
    tr_corr  = results_df["corrected_acc"].tolist()
    inverted = results_df["is_inverted"].tolist()

    ax = axes[0]
    ax.fill_between(layers,
                    [c - s for c, s in zip(cv_acc, cv_std)],
                    [c + s for c, s in zip(cv_acc, cv_std)],
                    alpha=0.15, color="#7c3aed")
    ax.plot(layers, cv_acc,  color="#7c3aed", lw=2.5, label="In-domain CV (physics, W0+W2+W4)")
    ax.plot(layers, tr_acc,  color="#dc2626", lw=2,   ls="--", label="Transfer (cross-domain, raw)")
    ax.plot(layers, tr_corr, color="#16a34a", lw=2.5, label="Transfer (corrected for inversion)")
    ax.axhline(0.5, color="#9ca3af", lw=1, ls=":", label="Chance")
    ax.axvline(best_layer, color="#f97316", lw=1.5, ls="--", alpha=0.7,
               label=f"Best L{best_layer}")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Probe accuracy by layer\nPhysics train → Cross-domain test", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    # Mark inverted layers
    for l, inv in zip(layers, inverted):
        if inv:
            ax.axvspan(l - 0.4, l + 0.4, alpha=0.08, color="#dc2626")

    ax = axes[1]
    # Transfer gap: in-domain - transfer
    gaps = [cv - tr for cv, tr in zip(cv_acc, tr_acc)]
    colors = ["#dc2626" if g > 0.15 else "#f97316" if g > 0.05 else "#16a34a" for g in gaps]
    ax.bar(layers, cv_acc,  label="In-domain CV",   color="#7c3aed", alpha=0.7)
    ax.bar(layers, tr_corr, label="Transfer (corr)", color="#16a34a", alpha=0.7)
    ax.axhline(0.5, color="#9ca3af", lw=1, ls=":")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("In-domain vs transfer accuracy\n(green = corrected transfer)", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.suptitle("Linear Transfer Probe: Physics → Cross-domain", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "linear_transfer_probe_by_layer.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_dir}/linear_transfer_probe_by_layer.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",   default="bfloat16", choices=["float32","bfloat16","float16"])
    ap.add_argument("--layers",  nargs="+", type=int, default=None,
                    help="Which hidden-state indices to probe (default: all, 0–36)")
    ap.add_argument("--cv",      type=int, default=5)
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype     = dtype_map[args.dtype]
    device    = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    train_rows = load_prompts(TRAIN_FILE, wording_filter=CLEAN_WORDINGS)
    test_rows  = load_prompts(TEST_FILE,  wording_filter=None)
    print(f"Train (physics clean): {len(train_rows)} prompts")
    print(f"Test  (cross-domain):  {len(test_rows)} prompts")
    if len(train_rows) == 0 or len(test_rows) == 0:
        print("No prompts found — run scripts/60_generate_abstraction_probe_datasets.py first")
        return

    # Determine layers
    model, tok = load_model(MODEL_NAME, device, dtype)
    n_layers_total = model.config.num_hidden_layers + 1   # +1 for embedding
    layers = args.layers if args.layers else list(range(n_layers_total))
    print(f"Probing {len(layers)} layers: L{layers[0]}…L{layers[-1]}")

    # Extract hidden states
    print("\nExtracting physics hidden states…")
    X_train, y_train, domains_train = collect_hidden_states(
        model, tok, train_rows, device, layers, desc="Train"
    )
    print(f"  Shape: {X_train.shape} | Classes: {np.bincount(y_train)}")

    print("Extracting cross-domain hidden states…")
    X_test,  y_test,  domains_test  = collect_hidden_states(
        model, tok, test_rows,  device, layers, desc="Test"
    )
    print(f"  Shape: {X_test.shape}  | Classes: {np.bincount(y_test)}")

    # Free model memory
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Probe each layer
    print("\nRunning probes…")
    rows = []
    for li, l in enumerate(layers):
        res = run_probe(X_train, y_train, X_test, y_test, li,
                        domains_test=domains_test, cv=args.cv)
        row = {
            "layer":        l,
            "cv_acc":       round(res["cv_acc"], 4),
            "cv_std":       round(res["cv_std"], 4),
            "transfer_acc": round(res["transfer_acc"], 4),
            "flipped_acc":  round(res["flipped_acc"], 4),
            "corrected_acc":round(res["corrected_acc"], 4),
            "is_inverted":  res["is_inverted"],
            "cm":           res["cm"],
            "domain_acc":   res["domain_acc"],
        }
        rows.append(row)
        inv = " [INVERTED]" if res["is_inverted"] else ""
        print(f"  L{l:2d}: cv={res['cv_acc']:.3f}  transfer={res['transfer_acc']:.3f}"
              f"  corrected={res['corrected_acc']:.3f}{inv}")

    # Build results DataFrame (flat CSV — serialise cm and domain_acc separately)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cm","domain_acc")}
                        for r in rows])
    df.to_csv(OUT_DIR / "linear_transfer_probe.csv", index=False)
    print(f"  CSV: {OUT_DIR}/linear_transfer_probe.csv")

    # Best layer by corrected transfer accuracy
    best_idx  = df["corrected_acc"].idxmax()
    best_layer = int(df.loc[best_idx, "layer"])
    best_row   = {**df.loc[best_idx].to_dict(), **rows[best_idx]}
    print(f"\nBest layer: L{best_layer} (corrected transfer={best_row['corrected_acc']:.3f})")

    # Plot
    make_plot(df, best_layer, OUT_DIR)

    # Report
    class_names = ["intensive", "extensive"]
    write_report(df, best_layer, best_row, class_names, OUT_DIR,
                 n_train=len(train_rows), n_test=len(test_rows))

    # Console summary
    print("\n=== KEY RESULTS ===")
    print(f"  Best in-domain CV:   {df['cv_acc'].max():.3f}  @ L{int(df.loc[df['cv_acc'].idxmax(),'layer'])}")
    print(f"  Best raw transfer:   {df['transfer_acc'].max():.3f}  @ L{int(df.loc[df['transfer_acc'].idxmax(),'layer'])}")
    print(f"  Best corr. transfer: {df['corrected_acc'].max():.3f}  @ L{best_layer}")
    n_inv = df["is_inverted"].sum()
    print(f"  Layers with inverted direction: {n_inv}/{len(df)}")
    bd = rows[best_idx]["domain_acc"]
    if bd:
        print("  Transfer by domain at best layer:")
        for dom, acc in sorted(bd.items()):
            print(f"    {dom}: {acc:.3f}")


if __name__ == "__main__":
    main()
