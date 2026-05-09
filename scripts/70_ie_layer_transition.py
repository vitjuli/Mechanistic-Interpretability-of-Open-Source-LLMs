"""
Layer-wise ARI transition analysis for physics_intensive_extensive_v1.

For each layer L0–L36:
  - Extract hidden state at last token position
  - K-means cluster hidden states, compute ARI vs abstraction_class / wording_family / property
  - Fit linear probe (logistic regression) for abstraction_class
  - Repeat for Family D cross-domain prompts → transfer probe accuracy per layer

Tests the hypothesis:
  Early layers (L10–L20): features cluster by wording family / property name
  Late layers  (L21–L34): features cluster by abstraction class (intensive vs extensive)

Usage:
    python scripts/70_ie_layer_transition.py --behaviour physics_intensive_extensive_v1 --device cuda
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

MODEL_NAME  = "Qwen/Qwen3-4B"
PROMPT_DIR  = Path("data/prompts")
OUT_BASE    = Path("data/results/abstraction_ie")
FAMILY_D    = Path("data/prompts/abstraction/D_cross_domain_train.jsonl")
LABEL_MAP   = {"intensive": 0, "extensive": 1}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_behaviour(behaviour, split, wording_filter=None):
    path = PROMPT_DIR / f"{behaviour}_{split}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    if wording_filter:
        rows = [r for r in rows if r.get("wording_family") in wording_filter]
    return rows

def load_cross_domain():
    if not FAMILY_D.exists():
        return []
    rows = [json.loads(l) for l in open(FAMILY_D)]
    return [r for r in rows if r.get("abstraction_class") in LABEL_MAP]


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype,
                                                trust_remote_code=True).to(device)
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def get_hidden_states(model, tokenizer, prompt, device, layers):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out    = model(**inputs, output_hidden_states=True, use_cache=False)
    return {l: out.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}


def collect(model, tok, rows, device, layers, desc=""):
    all_states, labels, wf_labels, prop_labels = [], [], [], []
    for i, r in enumerate(rows):
        if i % 40 == 0:
            print(f"  {desc} {i}/{len(rows)}", end="\r", flush=True)
        states = get_hidden_states(model, tok, r["prompt"], device, layers)
        all_states.append(states)
        labels.append(LABEL_MAP[r["abstraction_class"]])
        wf_labels.append(r.get("wording_family", ""))
        prop_labels.append(r.get("property", ""))
    print()
    n = len(all_states)
    d = len(next(iter(all_states[0].values())))
    X = np.zeros((n, len(layers), d), dtype=np.float32)
    for i, states in enumerate(all_states):
        for j, l in enumerate(layers):
            X[i, j] = states[l]
    return X, np.array(labels), wf_labels, prop_labels


# ── Per-layer probe + ARI ─────────────────────────────────────────────────────

def analyse_layer(X_phys, y_phys, wf_phys, prop_phys,
                  X_cross, y_cross, layer_idx, cv=5):
    feats = X_phys[:, layer_idx]
    feat_std = float(feats.std())
    degen = feat_std < 1.0

    result = {"std": feat_std, "degenerate": degen,
              "probe_cv": float("nan"), "ari_cls": float("nan"),
              "ari_wf": float("nan"), "ari_prop": float("nan"),
              "transfer_acc": float("nan")}

    if degen or len(set(y_phys)) < 2:
        return result

    # Normalise
    norm   = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sc     = StandardScaler()
    Fp     = sc.fit_transform(norm(feats))

    # Linear probe CV
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, Fp, y_phys, cv=skf, scoring="accuracy")
    result["probe_cv"] = float(scores.mean())

    # ARI: abstraction class, wording family, property
    Fn = norm(feats)
    for key, lab in [("ari_cls", y_phys),
                     ("ari_wf",  pd.Categorical(wf_phys).codes),
                     ("ari_prop",pd.Categorical(prop_phys).codes)]:
        n_k = len(set(lab))
        if n_k < 2: continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=n_k, random_state=42, n_init=5).fit(Fn)
        result[key] = float(adjusted_rand_score(lab, km.labels_))

    # Transfer probe: train on physics, test on cross-domain
    if X_cross is not None and len(X_cross) > 0:
        Fc = sc.transform(norm(X_cross[:, layer_idx]))
        clf.fit(Fp, y_phys)
        pred = clf.predict(Fc)
        tr   = float((pred == y_cross).mean())
        result["transfer_acc"] = max(tr, 1 - tr)   # correct for sign inversion

    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df, out_dir):
    if not HAS_MPL:
        return
    valid = df[~df["degenerate"]].copy()
    layers = valid["layer"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(layers, valid["probe_cv"],    color="#7c3aed", lw=2.5, label="Linear probe CV")
    ax.plot(layers, valid["transfer_acc"],color="#16a34a", lw=2,   ls="--", label="Transfer (cross-domain)")
    ax.axhline(0.5, color="#aaa", lw=1, ls=":", label="Chance")
    ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy")
    ax.set_title("Probe accuracy by layer"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(layers, valid["ari_cls"],  color="#2563eb", lw=2.5, label="ARI: abstraction class")
    ax.plot(layers, valid["ari_wf"],   color="#f97316", lw=2,   label="ARI: wording family")
    ax.plot(layers, valid["ari_prop"], color="#dc2626", lw=1.5, ls="--", label="ARI: property name")
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    # Shade layers where abstraction > wording
    for i, (l, ac, aw) in enumerate(zip(layers, valid["ari_cls"], valid["ari_wf"])):
        if not np.isnan(ac) and not np.isnan(aw) and ac > aw:
            ax.axvspan(l - 0.4, l + 0.4, alpha=0.12, color="#2563eb")
    ax.set_xlabel("Layer"); ax.set_ylabel("ARI")
    ax.set_title("Feature clustering ARI\n(blue shading = abstraction > surface form)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Ratio plot
    ax = axes[2]
    ratio = (valid["ari_cls"] / (valid["ari_wf"].abs() + 1e-6)).clip(-5, 5)
    colors = ["#2563eb" if r > 1 else "#9ca3af" for r in ratio]
    ax.bar(layers, ratio, color=colors)
    ax.axhline(1, color="#dc2626", lw=1.5, ls="--", label="ratio=1 (transition)")
    ax.set_xlabel("Layer"); ax.set_ylabel("ARI(cls) / ARI(wording)")
    ax.set_title("Form→Abstraction transition\n(ratio > 1 = abstraction dominates)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("physics_intensive_extensive_v1 — Layer Transition Analysis", fontsize=13)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "ie_layer_transition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_dir}/ie_layer_transition.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default="physics_intensive_extensive_v1")
    ap.add_argument("--split",     default="train")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",     default="bfloat16", choices=["float32","bfloat16","float16"])
    ap.add_argument("--cv",        type=int, default=5)
    ap.add_argument("--wording_filter", nargs="*",
                    default=["W0_combine","W2_split","W4_additive","W6_symbolic"],
                    help="Wording families to include (default: all 4)")
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}
    device = torch.device(args.device)
    dtype  = dtype_map[args.dtype]
    out_dir = OUT_BASE / args.behaviour

    phys_rows  = load_behaviour(args.behaviour, args.split, args.wording_filter)
    cross_rows = load_cross_domain()
    print(f"Physics prompts: {len(phys_rows)} | Cross-domain: {len(cross_rows)}")

    model, tok = load_model(device, dtype)
    n_layers = model.config.num_hidden_layers + 1   # includes embedding layer
    layers   = list(range(n_layers))
    print(f"Extracting {len(layers)} layers for {len(phys_rows)} physics prompts…")

    X_phys,  y_phys,  wf_phys,  prop_phys  = collect(model, tok, phys_rows,  device, layers, "Physics")
    X_cross, y_cross, _,         _          = collect(model, tok, cross_rows, device, layers, "Cross") \
        if cross_rows else (None, None, None, None)

    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nAnalysing layers…")
    rows = []
    for li, l in enumerate(layers):
        r = analyse_layer(X_phys, y_phys, wf_phys, prop_phys,
                          X_cross, y_cross if y_cross is not None else np.array([]), li, args.cv)
        r["layer"] = l
        rows.append(r)
        if not r["degenerate"]:
            print(f"  L{l:2d}: probe={r['probe_cv']:.3f}  ari_cls={r['ari_cls']:.4f}"
                  f"  ari_wf={r['ari_wf']:.4f}  transfer={r['transfer_acc']:.3f}")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ie_layer_transition.csv", index=False)

    valid = df[~df["degenerate"]]
    best_probe = int(valid["probe_cv"].idxmax())
    best_ari   = int(valid["ari_cls"].idxmax())
    best_trans = int(valid["transfer_acc"].idxmax())
    ratio_col  = valid["ari_cls"] / (valid["ari_wf"].abs() + 1e-6)
    transition = valid.index[ratio_col > 1].tolist()

    print(f"\n=== LAYER TRANSITION SUMMARY ===")
    print(f"  Peak probe CV:      L{best_probe} = {valid.loc[best_probe,'probe_cv']:.3f}")
    print(f"  Peak ARI(cls):      L{best_ari}   = {valid.loc[best_ari,'ari_cls']:.4f}")
    print(f"  Peak transfer:      L{best_trans} = {valid.loc[best_trans,'transfer_acc']:.3f}")
    print(f"  Abstraction > Form layers: {[df.loc[i,'layer'] for i in transition]}")

    make_plots(df, out_dir)
    print(f"\nCSV: {out_dir}/ie_layer_transition.csv")


if __name__ == "__main__":
    main()
