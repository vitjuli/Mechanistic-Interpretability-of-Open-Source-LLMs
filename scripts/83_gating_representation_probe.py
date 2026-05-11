"""
Lightweight hidden-state representation probe for gating_probe_v1.

For each non-degenerate layer (std ≥ 1.0):
  1. Extract hidden state h^ℓ(x) at the final token position
  2. Train LogisticRegression to classify gate_label (allow=1, block=0)
  3. Compute: probe CV accuracy, ARI(gate_label), ARI(wording_family)
  4. Identify layer where gate_label first becomes linearly decodable (CV > 0.75)

Cross-family transfer:
  - Train probe on Family A (thermodynamic_spontaneity)
  - Test on Families F, G, H (same binary allow/block gate_label)
  - Determines if "gate direction" is domain-general or physics-specific

Output:
  data/results/gating_probe_v1/
    probe_by_layer.csv
    cross_family_transfer.csv
    probe_summary.json

Usage:
    python scripts/83_gating_representation_probe.py --device cuda
"""

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src.model_utils import ModelWrapper

PROMPT_PATH   = Path("data/prompts/gating/gating_probe_v1.jsonl")
BASELINE_CSV  = Path("data/results/gating_probe_v1/baseline_results.csv")
OUT_DIR       = Path("data/results/gating_probe_v1")
MODEL_NAME    = "Qwen/Qwen3-4B"

PROBE_LAYERS  = list(range(0, 37))
DEGEN_STD_THR = 1.0
TRAIN_FAMILY  = "A"
TRANSFER_FAMILIES = ["F", "G", "H"]


def train_probe(X, y):
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xs, y)
    return clf, sc


def cv_probe(X, y, n_folds=5):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy")
    return float(scores.mean())


def ari_kmeans(X, labels, k=None):
    if k is None:
        k = len(set(labels))
    k = max(2, min(k, len(X) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km   = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
    return float(adjusted_rand_score(labels, km.labels_))


@torch.no_grad()
def extract_hs(model, tok, prompts, device, layers):
    hs = {l: [] for l in layers}
    for i, row in enumerate(prompts):
        if i % 60 == 0:
            print(f"    {i}/{len(prompts)}", end="\r", flush=True)
        inp = tok(row["prompt"], return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        for l in layers:
            hs[l].append(out.hidden_states[l][0, -1, :].float().cpu().numpy())
    print()
    return {l: np.stack(hs[l]) for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",  default="bfloat16",
                    choices=["float32","bfloat16","float16"])
    ap.add_argument("--families", default="A,F,G,H",
                    help="Comma-separated families to probe")
    args = ap.parse_args()

    device  = torch.device(args.device)
    dtype   = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fam_filter = set(args.families.split(","))

    prompts = [json.loads(l) for l in open(PROMPT_PATH)
               if json.loads(l)["family"] in fam_filter]
    print(f"Using {len(prompts)} prompts from families {fam_filter}")

    gate_labels = np.array([1 if r["gate_label"] == "allow" else 0
                             for r in prompts])
    wf_labels   = np.array([r["wording_family"] for r in prompts])
    fam_labels  = np.array([r["family"] for r in prompts])

    print("Loading model…")
    mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    mw.model.eval()

    print(f"Extracting hidden states at layers 0–36…")
    all_hs = extract_hs(mw.model, mw.tokenizer, prompts, device, PROBE_LAYERS)

    # ── Layer-by-layer probe and ARI ──────────────────────────────────────────
    probe_rows = []
    for l in PROBE_LAYERS:
        X = all_hs[l]
        std_val = float(X.std())
        degen   = std_val < DEGEN_STD_THR

        if degen:
            probe_rows.append({
                "layer": l, "std": round(std_val, 4), "degenerate": True,
                "probe_cv": None, "ari_gate": None, "ari_wording": None,
                "ari_family": None,
            })
            continue

        probe_cv  = cv_probe(X, gate_labels)
        ari_gate  = ari_kmeans(X, gate_labels, k=2)
        ari_wf    = ari_kmeans(X, wf_labels, k=len(set(wf_labels)))
        ari_fam   = ari_kmeans(X, fam_labels, k=len(set(fam_labels)))

        probe_rows.append({
            "layer": l, "std": round(std_val, 4), "degenerate": False,
            "probe_cv": round(probe_cv, 4),
            "ari_gate": round(ari_gate, 4),
            "ari_wording": round(ari_wf, 4),
            "ari_family": round(ari_fam, 4),
        })
        print(f"  L{l:2d}: std={std_val:.3f}  probe_cv={probe_cv:.4f}  "
              f"ARI_gate={ari_gate:.4f}  ARI_wf={ari_wf:.4f}")

    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(OUT_DIR / "probe_by_layer.csv", index=False)

    # ── Cross-family transfer ─────────────────────────────────────────────────
    valid = probe_df[~probe_df["degenerate"]]
    best_layer = int(valid.loc[valid["probe_cv"].idxmax(), "layer"])
    print(f"\nBest probe layer: L{best_layer}")

    train_mask    = fam_labels == TRAIN_FAMILY
    transfer_rows = []

    if train_mask.sum() >= 10:
        X_train = all_hs[best_layer][train_mask]
        y_train = gate_labels[train_mask]
        clf, sc = train_probe(X_train, y_train)

        for tgt_fam in TRANSFER_FAMILIES:
            tgt_mask = fam_labels == tgt_fam
            if tgt_mask.sum() < 5:
                continue
            X_tgt  = sc.transform(all_hs[best_layer][tgt_mask])
            y_tgt  = gate_labels[tgt_mask]
            acc    = float((clf.predict(X_tgt) == y_tgt).mean())
            transfer_rows.append({
                "train_family":  TRAIN_FAMILY,
                "test_family":   tgt_fam,
                "probe_layer":   best_layer,
                "transfer_acc":  round(acc, 4),
                "n_train":       int(train_mask.sum()),
                "n_test":        int(tgt_mask.sum()),
            })
            print(f"  Transfer A→{tgt_fam}: acc={acc:.4f} (n={tgt_mask.sum()})")

    xfer_df = pd.DataFrame(transfer_rows)
    xfer_df.to_csv(OUT_DIR / "cross_family_transfer.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_probe = probe_df[~probe_df["degenerate"] & probe_df["probe_cv"].notna()]
    decodable_layer = None
    for _, row in valid_probe.sort_values("layer").iterrows():
        if row["probe_cv"] > 0.75:
            decodable_layer = int(row["layer"])
            break

    summary = {
        "best_probe_layer":   best_layer,
        "best_probe_cv":      round(float(valid_probe["probe_cv"].max()), 4),
        "decodable_from_layer": decodable_layer,
        "n_degenerate_layers":  int(probe_df["degenerate"].sum()),
        "cross_family_transfer": transfer_rows,
        "mean_ari_gate":   round(float(valid_probe["ari_gate"].mean()), 4),
        "mean_ari_wording":round(float(valid_probe["ari_wording"].mean()), 4),
    }
    with open(OUT_DIR / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== PROBE SUMMARY ===")
    print(f"Best probe layer: L{best_layer} (CV={summary['best_probe_cv']:.4f})")
    print(f"Gate decodable from: L{decodable_layer}")
    print(f"Mean ARI(gate_label): {summary['mean_ari_gate']:.4f}")
    print(f"Mean ARI(wording): {summary['mean_ari_wording']:.4f}")
    if transfer_rows:
        for t in transfer_rows:
            print(f"Transfer A→{t['test_family']}: {t['transfer_acc']:.4f}")


if __name__ == "__main__":
    main()
