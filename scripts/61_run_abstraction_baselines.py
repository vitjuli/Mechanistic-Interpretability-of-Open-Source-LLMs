"""
Abstraction probe baseline evaluator.

For each prompt family (A–E) and each model:
  1. Extracts ' Yes' / ' No' logits → accuracy, ND, consistency
  2. Extracts per-layer hidden states at decision token
  3. Runs linear probes (logistic regression) per layer
  4. Computes ARI(abstraction_class) vs ARI(wording_family) per layer
  5. Outputs ranked summary + plots + recommendation report

CPU-capable for 0.6B; GPU required for 4B+.

Usage:
    python scripts/61_run_abstraction_baselines.py --model_size 0.6b --device cpu
    python scripts/61_run_abstraction_baselines.py --model_size 4b   --device cuda
    python scripts/61_run_abstraction_baselines.py --model_size 4b   --families A D E

Requires:
    sbatch jobs/run_abstraction_baseline.sbatch  (CSD3 GPU)
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Optional imports ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import cross_val_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False
    warnings.warn("scikit-learn not found — skipping linear probes and ARI")

# ── Constants ─────────────────────────────────────────────────────────────────────
MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "4b":   "Qwen/Qwen3-4B",
    "8b":   "Qwen/Qwen3-8B",
}

FAMILIES = ["A", "B", "C", "D", "E"]
FAMILY_FILES = {
    "A": "A_intensive_extensive",
    "B": "B_scaling_law",
    "C": "C_representation_equiv",
    "D": "D_cross_domain",
    "E": "E_conservation_law",
}

YES_TOKEN = 7414   # ' Yes' in Qwen3 tokenizer
NO_TOKEN  = 2308   # ' No'  in Qwen3 tokenizer

PROMPT_DIR = Path("data/prompts/abstraction")
OUT_BASE   = Path("data/results/abstraction_probe")


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_family(family_key, split="train"):
    fname = FAMILY_FILES[family_key]
    path  = PROMPT_DIR / f"{fname}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nRun: python scripts/60_generate_abstraction_probe_datasets.py")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ── Model loading ─────────────────────────────────────────────────────────────────

def load_model(model_name, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} on {device}…")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    mdl.eval()
    # Verify Yes/No tokens
    yes_id = tok.encode(" Yes", add_special_tokens=False)
    no_id  = tok.encode(" No",  add_special_tokens=False)
    assert yes_id == [YES_TOKEN], f"Unexpected ' Yes' token: {yes_id}"
    assert no_id  == [NO_TOKEN],  f"Unexpected ' No' token:  {no_id}"
    print(f"  Model loaded. ' Yes'={YES_TOKEN}, ' No'={NO_TOKEN}")
    return mdl, tok


# ── Single-prompt inference ───────────────────────────────────────────────────────

@torch.no_grad()
def get_logits_and_states(mdl, tok, prompt, device, capture_hidden=True):
    """
    Returns:
        logp_yes  float
        logp_no   float
        hidden    list[Tensor shape (d_model,)] — one per layer, or None
    """
    inputs = tok(prompt, return_tensors="pt").to(device)
    out    = mdl(**inputs, output_hidden_states=capture_hidden, use_cache=False)
    logits = out.logits[0, -1, :]      # last token logits
    log_p  = torch.log_softmax(logits, dim=0)
    logp_yes = float(log_p[YES_TOKEN])
    logp_no  = float(log_p[NO_TOKEN])

    if capture_hidden:
        # hidden_states: tuple of (n_layers+1) tensors, shape (1, seq_len, d_model)
        hidden = [h[0, -1, :].float().cpu() for h in out.hidden_states]
    else:
        hidden = None
    return logp_yes, logp_no, hidden


# ── Batch evaluation ─────────────────────────────────────────────────────────────

def evaluate_family(mdl, tok, rows, device, capture_hidden=False, desc=""):
    results = []
    hidden_list = []   # list of [n_layers, d_model] tensors

    for i, row in enumerate(rows):
        if i % 20 == 0:
            print(f"  {desc} {i}/{len(rows)}", end="\r", flush=True)
        try:
            lpy, lpn, hidden = get_logits_and_states(
                mdl, tok, row["prompt"], device, capture_hidden=capture_hidden
            )
        except Exception as exc:
            print(f"\n  [WARN] row {i}: {exc}")
            continue

        nd      = lpy - lpn
        pred    = " Yes" if lpy > lpn else " No"
        correct = (pred == row["correct_answer"])

        results.append({
            **{k: row[k] for k in
               ("family","abstraction_class","property","domain","wording_family",
                "adversarial","difficulty","correct_answer")},
            "logp_yes": lpy,
            "logp_no":  lpn,
            "nd":       nd,
            "pred":     pred,
            "correct":  correct,
        })
        if capture_hidden and hidden:
            hidden_list.append(torch.stack(hidden))  # (n_layers, d_model)

    print()
    df = pd.DataFrame(results)
    hidden_mat = torch.stack(hidden_list) if hidden_list else None  # (n_prompts, n_layers, d_model)
    return df, hidden_mat


# ── Consistency analysis ──────────────────────────────────────────────────────────

def consistency_stats(df):
    """
    For each property, compute whether model is consistent across all wording families.
    Returns a DataFrame with one row per property.
    """
    rows = []
    for prop, grp in df.groupby("property"):
        n_wf   = grp["wording_family"].nunique()
        acc    = grp["correct"].mean()
        n_yes  = (grp["pred"] == " Yes").sum()
        n_no   = (grp["pred"] == " No").sum()
        # Consistency: fraction of prompts where prediction agrees with majority vote
        majority = " Yes" if n_yes >= n_no else " No"
        consist  = float((grp["pred"] == majority).mean())
        rows.append({
            "property":        prop,
            "abstraction_class": grp["abstraction_class"].iloc[0],
            "domain":          grp["domain"].iloc[0],
            "n_wording":       n_wf,
            "accuracy":        float(acc),
            "consistency":     consist,
            "mean_nd":         float(grp["nd"].mean()),
            "nd_std":          float(grp["nd"].std()),
            "adversarial":     bool(grp["adversarial"].any()),
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=True)


# ── Representation analysis ───────────────────────────────────────────────────────

def representation_analysis(df, hidden_mat, family_key, out_dir):
    """
    Per-layer:
      - ARI(abstraction_class) vs ARI(wording_family)
      - Linear probe accuracy (5-fold CV)
    """
    if not HAS_SKL:
        return None

    n_prompts, n_layers, d_model = hidden_mat.shape
    X = hidden_mat.numpy()   # (n_prompts, n_layers, d_model)

    labels_cls = pd.Categorical(df["abstraction_class"]).codes.tolist()
    labels_wf  = pd.Categorical(df["wording_family"]).codes.tolist()

    if len(set(labels_cls)) < 2:
        print("  [SKIP] Only one abstraction class — skipping representation analysis")
        return None

    ari_cls, ari_wf, probe_acc, degenerate = [], [], [], []

    for layer in range(n_layers):
        feats = X[:, layer, :]

        # Skip degenerate layers (embedding layer and first 1-2 transformer layers
        # often collapse all prompts to nearly the same point, causing spurious probe accuracy)
        feat_std = float(feats.std())
        is_degen = feat_std < 1.0   # threshold: meaningful layers have std >> 1
        degenerate.append(is_degen)

        # Normalise
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

        if is_degen:
            probe_acc.append(float("nan"))
            ari_cls.append(float("nan"))
            ari_wf.append(float("nan"))
            continue

        # Linear probe: 5-fold CV
        scaler = StandardScaler()
        feats_s = scaler.fit_transform(feats_norm)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, feats_s, labels_cls, cv=min(5, len(set(labels_cls))+1),
                                     scoring="accuracy")
        probe_acc.append(float(scores.mean()))

        # ARI: cluster by abstraction class vs by wording family
        try:
            from sklearn.cluster import KMeans
            n_cls = len(set(labels_cls))
            n_wf  = len(set(labels_wf))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km_cls = KMeans(n_clusters=n_cls, random_state=42, n_init=5).fit(feats_norm)
                km_wf  = KMeans(n_clusters=n_wf,  random_state=42, n_init=5).fit(feats_norm)
            ari_cls.append(float(adjusted_rand_score(labels_cls, km_cls.labels_)))
            ari_wf.append(float(adjusted_rand_score(labels_wf,  km_wf.labels_)))
        except Exception:
            ari_cls.append(float("nan"))
            ari_wf.append(float("nan"))

    rep_df = pd.DataFrame({
        "layer":      list(range(n_layers)),
        "probe_acc":  probe_acc,
        "ari_cls":    ari_cls,
        "ari_wf":     ari_wf,
        "degenerate": degenerate,
        "ratio":      [a / (b + 1e-6) if not np.isnan(a) and not np.isnan(b) else float("nan")
                       for a, b in zip(ari_cls, ari_wf)],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(out_dir / f"layer_representation_{family_key}.csv", index=False)

    # Plot (mask degenerate layers)
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        layers     = rep_df["layer"].tolist()
        valid_mask = ~rep_df["degenerate"].values
        valid_layers = [l for l, v in zip(layers, valid_mask) if v]

        ax = axes[0]
        probe_valid = rep_df["probe_acc"].where(~rep_df["degenerate"]).tolist()
        ax.plot(layers, probe_valid, color="#7c3aed", lw=2, label="Linear probe acc")
        ax.axhline(0.5, color="#aaa", lw=1, ls="--", label="Chance (binary)")
        # Grey out degenerate layers
        for l, degen in zip(layers, degenerate):
            if degen:
                ax.axvspan(l - 0.5, l + 0.5, alpha=0.15, color="#9ca3af")
        ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy (5-fold CV)")
        ax.set_title(f"Family {family_key}: Linear probe (grey=degenerate layers)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        ari_cls_valid = rep_df["ari_cls"].where(~rep_df["degenerate"]).tolist()
        ari_wf_valid  = rep_df["ari_wf"].where(~rep_df["degenerate"]).tolist()
        ax.plot(layers, ari_cls_valid, color="#2563eb", lw=2, label="ARI: abstraction class")
        ax.plot(layers, ari_wf_valid,  color="#f97316", lw=2, label="ARI: wording family")
        ax.axhline(0, color="#aaa", lw=0.8, ls="--")
        ax.set_xlabel("Layer"); ax.set_ylabel("ARI")
        ax.set_title(f"Family {family_key}: Feature clustering ARI")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Shade layers where abstraction ARI > wording ARI (valid only)
        for i in range(len(layers) - 1):
            if not degenerate[i] and not np.isnan(ari_cls[i]) and not np.isnan(ari_wf[i]):
                if ari_cls[i] > ari_wf[i]:
                    ax.axvspan(layers[i] - 0.5, layers[i] + 0.5, alpha=0.15, color="#2563eb")

        fig.tight_layout()
        fig.savefig(out_dir / f"representation_{family_key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return rep_df


# ── Per-family plots ──────────────────────────────────────────────────────────────

def plot_family_summary(df, model_tag, family_key, out_dir):
    if not HAS_MPL:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Accuracy by wording family
    ax = axes[0]
    wf_acc = df.groupby("wording_family")["correct"].mean().sort_values()
    ax.barh(wf_acc.index, wf_acc.values,
            color=["#16a34a" if v > 0.7 else "#dc2626" if v < 0.5 else "#f97316" for v in wf_acc.values])
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_xlabel("Accuracy"); ax.set_title(f"F{family_key}: Accuracy by wording")
    ax.grid(alpha=0.3, axis="x")

    # 2. ND distribution by abstraction class
    ax = axes[1]
    for cls, grp in df.groupby("abstraction_class"):
        ax.hist(grp["nd"], bins=20, alpha=0.6, label=cls, density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("ND = logp(Yes) − logp(No)")
    ax.set_title(f"F{family_key}: Confidence distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Consistency by property
    ax = axes[2]
    cons = consistency_stats(df)
    colors = ["#16a34a" if r > 0.8 else "#f97316" if r > 0.6 else "#dc2626"
              for r in cons["consistency"]]
    ax.barh(range(len(cons)), cons["consistency"], color=colors)
    ax.set_yticks(range(len(cons)))
    ax.set_yticklabels(cons["property"], fontsize=7)
    ax.axvline(0.8, color="black", lw=1, ls="--")
    ax.set_xlabel("Consistency"); ax.set_title(f"F{family_key}: Cross-wording consistency")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Family {family_key} | {model_tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"summary_{family_key}_{model_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Summary / ranking ─────────────────────────────────────────────────────────────

def rank_families(all_results, model_tag, out_dir):
    """
    Ranks families by:
      1. Accuracy (overall)
      2. Consistency (cross-wording)
      3. ND separability (class-separated margin)
      4. Non-adversarial accuracy (no shortcut)
    """
    summary_rows = []
    for family_key, df in all_results.items():
        acc_all   = df["correct"].mean()
        acc_nonadv = df[~df["adversarial"].astype(bool)]["correct"].mean() if "adversarial" in df else acc_all
        acc_adv    = df[df["adversarial"].astype(bool)]["correct"].mean() if df["adversarial"].any() else float("nan")

        # Separability: AUC-like measure using ND
        from sklearn.metrics import roc_auc_score
        try:
            # For binary families
            classes = df["abstraction_class"].unique()
            if len(classes) == 2:
                pos_cls = sorted(classes)[1]
                y_true  = (df["abstraction_class"] == pos_cls).astype(int)
                auc     = float(roc_auc_score(y_true, df["nd"]))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        # Consistency
        cons_df  = consistency_stats(df)
        mean_cons = cons_df["consistency"].mean()

        # ND gap between abstraction classes (for binary)
        nd_by_cls = df.groupby("abstraction_class")["nd"].mean()
        nd_gap    = float(nd_by_cls.max() - nd_by_cls.min()) if len(nd_by_cls) >= 2 else 0.0

        # Cross-domain generalisation (only for D)
        if "domain" in df.columns and df["domain"].nunique() > 1:
            dom_acc_std = df.groupby("domain")["correct"].mean().std()
        else:
            dom_acc_std = float("nan")

        summary_rows.append({
            "family":           family_key,
            "n_prompts":        len(df),
            "accuracy":         round(acc_all, 4),
            "acc_non_adv":      round(acc_nonadv, 4),
            "acc_adv":          round(acc_adv, 4) if not np.isnan(acc_adv) else float("nan"),
            "consistency":      round(mean_cons, 4),
            "nd_gap":           round(nd_gap, 4),
            "nd_auc":           round(auc, 4) if not np.isnan(auc) else float("nan"),
            "domain_acc_std":   round(dom_acc_std, 4) if not np.isnan(dom_acc_std) else float("nan"),
        })

    summ = pd.DataFrame(summary_rows)
    # Composite score: accuracy + consistency - domain_std (generalisation penalty)
    summ["composite"] = (
        summ["accuracy"].fillna(0) * 0.35 +
        summ["consistency"].fillna(0) * 0.30 +
        summ["nd_auc"].fillna(0.5) * 0.20 +
        (1 - summ["domain_acc_std"].fillna(0)) * 0.15
    )
    summ = summ.sort_values("composite", ascending=False).reset_index(drop=True)
    summ["rank"] = range(1, len(summ) + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / f"family_ranking_{model_tag}.csv", index=False)
    return summ


# ── Report writer ─────────────────────────────────────────────────────────────────

def write_report(summ, all_results, model_tag, model_name, out_dir, rep_results=None):
    lines = [
        "# Abstraction Probe Baseline Report",
        f"## Model: {model_name} | Tag: {model_tag}",
        "",
        "## Family Ranking (composite score = 0.35·acc + 0.30·consistency + 0.20·ND-AUC + 0.15·domain-gen)",
        "",
        "| Rank | Family | Accuracy | Consistency | ND-AUC | ND gap | Composite |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['family']} | {r['accuracy']:.3f} | "
            f"{r['consistency']:.3f} | {r['nd_auc']:.3f} | {r['nd_gap']:.3f} | "
            f"{r['composite']:.3f} |"
        )

    lines += ["", "## Per-family accuracy by wording style", ""]
    for fk, df in all_results.items():
        lines.append(f"### Family {fk}")
        wf_acc = df.groupby("wording_family")["correct"].mean().round(3)
        for wf, acc in wf_acc.items():
            flag = "✓" if acc > 0.7 else "✗" if acc < 0.5 else "~"
            lines.append(f"  {flag} {wf}: {acc:.3f}")
        lines.append("")

    lines += ["## Adversarial vs Non-adversarial", ""]
    for fk, df in all_results.items():
        if df.get("adversarial", pd.Series(False)).any():
            acc_a  = df[df["adversarial"] == True]["correct"].mean()
            acc_na = df[df["adversarial"] == False]["correct"].mean()
            lines.append(f"  Family {fk}: non-adv={acc_na:.3f}, adv={acc_a:.3f}, "
                         f"gap={acc_na - acc_a:.3f}")
    lines.append("")

    # Worst prompts
    lines += ["## 10 Hardest Prompts (lowest ND margin)", ""]
    all_df = pd.concat(all_results.values(), ignore_index=True)
    hard = all_df.nsmallest(10, "nd")[["family","property","wording_family",
                                       "correct_answer","pred","nd","abstraction_class"]]
    lines.append(hard.to_string(index=False))

    # Representation analysis peak layer
    if rep_results:
        lines += ["", "## Layer transition (form → abstraction)", ""]
        for fk, rep_df in rep_results.items():
            if rep_df is None:
                continue
            valid = rep_df[~rep_df["degenerate"]] if "degenerate" in rep_df.columns else rep_df
            valid = valid.dropna(subset=["probe_acc"])
            if len(valid) == 0:
                continue
            peak_probe = int(valid["probe_acc"].idxmax())
            peak_ari   = int(valid["ari_cls"].idxmax()) if not valid["ari_cls"].isna().all() else -1
            peak_ratio = int(valid["ratio"].idxmax()) if not valid["ratio"].isna().all() else -1
            lines.append(
                f"  Family {fk}: peak probe acc = L{peak_probe} ({valid['probe_acc'].max():.3f}) | "
                f"peak ARI(cls) = L{peak_ari} ({valid['ari_cls'].max():.4f}) | "
                f"peak ratio = L{peak_ratio} ({valid['ratio'].max():.2f}×)"
            )

    # Recommendation
    best = summ.iloc[0]
    lines += [
        "",
        "## Recommendation",
        "",
        f"**Best family for mechanistic analysis: {best['family']}** "
        f"(composite={best['composite']:.3f})",
        "",
        "Criteria for full pipeline:",
        "  1. Accuracy ≥ 0.75 (model reliably knows the answer)",
        "  2. Consistency ≥ 0.80 (robust across wording families)",
        "  3. ND-AUC ≥ 0.70 (logit confidence separates classes)",
        "  4. Adversarial accuracy gap < 0.15 (no severe shortcut usage)",
        "",
        "**Pass/fail for top family:**",
    ]
    for thresh, key, label in [
        (0.75, "accuracy",    "Overall accuracy"),
        (0.80, "consistency", "Wording consistency"),
        (0.70, "nd_auc",      "ND-AUC separability"),
    ]:
        val = float(best[key])
        flag = "PASS" if val >= thresh else "FAIL"
        lines.append(f"  [{flag}] {label}: {val:.3f} (threshold {thresh})")

    (out_dir / f"baseline_report_{model_tag}.md").write_text("\n".join(lines))
    print(f"\nReport: {out_dir}/baseline_report_{model_tag}.md")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size",  default="4b",  choices=list(MODELS))
    ap.add_argument("--split",       default="train")
    ap.add_argument("--families",    nargs="+", default=FAMILIES,
                    help="Which families to evaluate (default: all A B C D E)")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rep_analysis",action="store_true",
                    help="Run per-layer representation analysis (slower, needs more VRAM)")
    ap.add_argument("--rep_family",  default="A",
                    help="Which family to use for representation analysis (default: A)")
    ap.add_argument("--dtype",       default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    model_name = MODELS[args.model_size]
    model_tag  = args.model_size.replace(".", "")
    dtype_map  = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype      = dtype_map[args.dtype]
    device     = torch.device(args.device)
    out_dir    = OUT_BASE / model_tag

    # Evaluate
    mdl, tok = load_model(model_name, device, dtype)

    all_results  = {}
    rep_results  = {}

    for fk in args.families:
        print(f"\n── Family {fk} ──────────────────────────────")
        try:
            rows = load_family(fk, args.split)
        except FileNotFoundError as e:
            print(f"  {e}"); continue

        capture = args.rep_analysis and (fk == args.rep_family)
        df, hidden_mat = evaluate_family(
            mdl, tok, rows, device,
            capture_hidden=capture, desc=f"F{fk}"
        )

        # Save raw results
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"results_{fk}_{model_tag}.csv", index=False)

        # Print quick stats
        acc = df["correct"].mean()
        cons_df = consistency_stats(df)
        mean_cons = cons_df["consistency"].mean()
        print(f"  Accuracy: {acc:.3f} | Consistency: {mean_cons:.3f} | N={len(df)}")
        print(f"  By wording:\n" +
              df.groupby("wording_family")["correct"].mean().round(3).to_string())

        # Plots
        plot_family_summary(df, model_tag, fk, out_dir)

        # Representation analysis
        if capture and hidden_mat is not None:
            print(f"\n  Running representation analysis (Family {fk})…")
            rep_df = representation_analysis(df, hidden_mat, fk, out_dir)
            rep_results[fk] = rep_df
            if rep_df is not None:
                valid = rep_df[~rep_df["degenerate"]]
                if len(valid):
                    peak = int(valid["probe_acc"].idxmax())
                    print(f"  Peak probe (non-degenerate): L{peak} = {valid['probe_acc'].max():.3f}")
                    peak_ari = int(valid["ari_cls"].idxmax()) if not valid["ari_cls"].isna().all() else "?"
                    print(f"  Peak ARI(cls): L{peak_ari} = {valid['ari_cls'].max():.4f}")
        else:
            rep_results[fk] = None

        all_results[fk] = df

    # Ranking
    if len(all_results) > 1:
        print("\n── Family ranking ─────────────────────────────")
        summ = rank_families(all_results, model_tag, out_dir)
        print(summ[["rank","family","accuracy","consistency","nd_auc","composite"]].to_string(index=False))
        write_report(summ, all_results, model_tag, model_name, out_dir, rep_results)

    print(f"\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
