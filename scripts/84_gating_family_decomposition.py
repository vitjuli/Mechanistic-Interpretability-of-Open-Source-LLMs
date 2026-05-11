"""
Gating family decomposition — per-family L28 transition analysis.

Extracts hidden states at L19–L36 for families A/F/G/H separately, then asks:
  - Does each family independently show the ARI_gate≈0.60 jump at L28?
  - Or does the mixed-dataset transition disappear when families are isolated?
  - Are the probe directions at L34 shared or family-specific?

Single GPU pass over 220 prompts; all analysis in memory.

Outputs:
  data/results/gating_probe_v1/
    family_transition_metrics.csv
    probe_directions_L34.npz         (for downstream use)
  plots/:
    per_family_probe_cv.png
    per_family_ARI_gate.png
    per_family_ARI_wording.png
    gate_transition_comparison.png
    probe_direction_cosine_heatmap.png
  docs/gating_family_decomposition.md

Usage:
    python scripts/84_gating_family_decomposition.py --device cuda
"""

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src.model_utils import ModelWrapper

PROMPT_PATH = Path("data/prompts/gating/gating_probe_v1.jsonl")
OUT_DIR     = Path("data/results/gating_probe_v1")
PLOT_DIR    = Path("plots")
DOCS_DIR    = Path("docs")
MODEL_NAME  = "Qwen/Qwen3-4B"

FAMILIES    = ["A", "F", "G", "H"]
FAM_NAMES   = {"A": "thermodynamic_spontaneity", "F": "dimensional_analysis",
               "G": "probability_statistics",    "H": "physical_boundary"}
FAM_COLORS  = {"A": "#f97316", "F": "#6c8cff", "G": "#22c55e", "H": "#e879f9"}

PROBE_LAYERS = list(range(19, 37))   # L19–L36 inclusive


# ── Utilities ─────────────────────────────────────────────────────────────────

def cv_probe(X, y, n_folds=5):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy")
    return float(scores.mean())


def ari_kmeans(X, labels, k):
    k = max(2, min(k, len(X) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return float(adjusted_rand_score(labels, km.labels_))


def train_probe_direction(X, y):
    """Return unit-normalised LR coefficient vector."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf.fit(Xs, y)
    w = clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-10)


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts, layers, device):
    """Returns dict layer → ndarray (n_prompts, d_model)."""
    hs = {l: [] for l in layers}
    for i, row in enumerate(prompts):
        if i % 40 == 0:
            print(f"    {i}/{len(prompts)}", end="\r", flush=True)
        inp = tokenizer(row["prompt"], return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        for l in layers:
            hs[l].append(out.hidden_states[l][0, -1, :].float().cpu().numpy())
    print()
    return {l: np.stack(hs[l]) for l in layers}


# ── Per-family layer analysis ─────────────────────────────────────────────────

def analyse_family(fam_prompts, all_hs, layers):
    """
    Returns DataFrame with one row per layer:
      layer, probe_cv, ari_gate, ari_wording
    """
    gate_labels = np.array([1 if r["gate_label"] == "allow" else 0
                             for r in fam_prompts])
    wf_labels   = np.array([r["wording_family"] for r in fam_prompts])
    n_wf        = len(set(wf_labels))
    rows        = []

    for l in layers:
        X = all_hs[l]
        std_val = float(X.std())
        degen   = std_val < 1.0

        if degen or len(X) < 10:
            rows.append({"layer": l, "std": round(std_val,4),
                         "degenerate": True, "probe_cv": None,
                         "ari_gate": None, "ari_wording": None})
            continue

        # Skip CV if insufficient samples per class
        min_cls = min(np.bincount(gate_labels))
        n_folds = min(5, min_cls)
        probe   = cv_probe(X, gate_labels, n_folds=n_folds) if n_folds >= 2 else float("nan")
        ari_g   = ari_kmeans(X, gate_labels, k=2)
        ari_w   = ari_kmeans(X, wf_labels,   k=n_wf)

        rows.append({"layer": l, "std": round(std_val, 4),
                     "degenerate": False,
                     "probe_cv":   round(probe, 4),
                     "ari_gate":   round(ari_g, 4),
                     "ari_wording": round(ari_w, 4)})
    return pd.DataFrame(rows)


# ── Transition metrics ────────────────────────────────────────────────────────

def transition_metrics(df, fam):
    valid   = df[~df["degenerate"]].dropna(subset=["ari_gate"])
    layers  = valid["layer"].values
    ag      = valid["ari_gate"].values
    aw      = valid["ari_wording"].values
    pc      = valid["probe_cv"].values

    peak_ag     = float(ag.max()) if len(ag) else float("nan")
    peak_layer  = int(layers[ag.argmax()]) if len(ag) else -1

    # First layer where ARI_gate > 0.30
    over_thr    = layers[ag > 0.30]
    first_gate  = int(over_thr[0]) if len(over_thr) else -1

    # Area under curve (trapezoid, clipped to [0,1])
    if len(ag) > 1:
        area_ag = float(np.trapz(np.clip(ag, 0, 1), layers))
    else:
        area_ag = float("nan")

    # Transition sharpness: max one-step ARI_gate increase
    if len(ag) > 1:
        deltas     = np.diff(ag)
        sharpness  = float(deltas.max())
        sharp_layer = int(layers[deltas.argmax()])
    else:
        sharpness  = float("nan")
        sharp_layer = -1

    # Mean wording ARI before / after transition layer
    if first_gate > 0:
        before = valid[valid["layer"] < first_gate]["ari_wording"]
        after  = valid[valid["layer"] >= first_gate]["ari_wording"]
    else:
        before = pd.Series([], dtype=float)
        after  = aw

    mean_wf_before = float(before.mean()) if len(before) else float("nan")
    mean_wf_after  = float(after.mean())  if len(after)  else float("nan")
    peak_probe     = float(pc.max())       if len(pc)     else float("nan")

    # Gate dominance: fraction of layers where ARI_gate > ARI_wording
    dominance = float((ag > aw).mean()) if len(ag) else float("nan")

    return {
        "family":           fam,
        "family_name":      FAM_NAMES[fam],
        "n_prompts":        None,   # filled later
        "peak_ari_gate":    round(peak_ag, 4),
        "peak_layer":       peak_layer,
        "first_gate_layer": first_gate,
        "transition_sharp": round(sharpness, 4),
        "sharp_at_layer":   sharp_layer,
        "area_ari_gate":    round(area_ag, 4) if not np.isnan(area_ag) else None,
        "mean_wf_before":   round(mean_wf_before, 4) if not np.isnan(mean_wf_before) else None,
        "mean_wf_after":    round(mean_wf_after, 4)  if not np.isnan(mean_wf_after)  else None,
        "peak_probe_cv":    round(peak_probe, 4),
        "gate_dominance":   round(dominance, 4),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_per_family(family_dfs, metric, ylabel, fname, ylim=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for ax, fam in zip(axes, FAMILIES):
        df   = family_dfs[fam]
        valid = df[~df["degenerate"]].dropna(subset=[metric])
        color = FAM_COLORS[fam]
        ax.plot(valid["layer"], valid[metric], "-o", color=color,
                linewidth=2, markersize=5, label=FAM_NAMES[fam])
        ax.axvline(28, color="gray", linewidth=1, linestyle="--", alpha=0.6, label="L28")
        ax.axhline(0.30, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title(f"Family {fam}: {FAM_NAMES[fam]}", fontsize=10)
        ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
    fig.suptitle(f"Per-family {ylabel} (L19–L36)", fontsize=12)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / fname}")


def plot_comparison(family_dfs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ARI_gate comparison
    ax = axes[0]
    for fam in FAMILIES:
        df    = family_dfs[fam]
        valid = df[~df["degenerate"]].dropna(subset=["ari_gate"])
        ax.plot(valid["layer"], valid["ari_gate"], "-o",
                color=FAM_COLORS[fam], linewidth=2.5, markersize=6,
                label=f"{fam}: {FAM_NAMES[fam]}")
    ax.axvline(28, color="black", linewidth=1.5, linestyle="--", label="L28")
    ax.axhline(0.30, color="red", linewidth=1, linestyle=":", label="thr=0.30")
    ax.set_xlabel("Layer"); ax.set_ylabel("ARI(gate_label)")
    ax.set_title("ARI(gate_label) per family — L28 transition")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Right: probe CV comparison
    ax = axes[1]
    for fam in FAMILIES:
        df    = family_dfs[fam]
        valid = df[~df["degenerate"]].dropna(subset=["probe_cv"])
        ax.plot(valid["layer"], valid["probe_cv"], "-o",
                color=FAM_COLORS[fam], linewidth=2.5, markersize=6,
                label=f"{fam}: {FAM_NAMES[fam]}")
    ax.axvline(28, color="black", linewidth=1.5, linestyle="--", label="L28")
    ax.axhline(0.75, color="red", linewidth=1, linestyle=":", label="thr=0.75")
    ax.set_xlabel("Layer"); ax.set_ylabel("Probe CV accuracy")
    ax.set_title("Probe CV per family")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Gate Transition Comparison across Families", fontsize=13)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "gate_transition_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/gate_transition_comparison.png")


def plot_cosine_heatmap(directions):
    fams = list(directions.keys())
    n    = len(fams)
    C    = np.zeros((n, n))
    for i, fi in enumerate(fams):
        for j, fj in enumerate(fams):
            C[i, j] = float(np.dot(directions[fi], directions[fj]))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{f}\n{FAM_NAMES[f][:12]}" for f in fams], fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{f}: {FAM_NAMES[f][:18]}" for f in fams], fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i,j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if abs(C[i,j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("Probe direction cosine similarity at L34\n(high = shared gate direction)", fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "probe_direction_cosine_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/probe_direction_cosine_heatmap.png")

    return C, fams


# ── Report ────────────────────────────────────────────────────────────────────

def verdict(m, cos_with_A, sign_acc_dict):
    fam   = m["family"]
    acc   = sign_acc_dict[fam]
    peak  = m["peak_ari_gate"]
    first = m["first_gate_layer"]
    cos_a = cos_with_A.get(fam, float("nan"))

    if peak < 0.10:
        v = "❌ No gate transition"
    elif peak < 0.30:
        v = "⚠ Weak gate transition"
    elif acc >= 0.95:
        v = "⚠ Ceiling family — gate trivial"
    elif peak >= 0.40 and first <= 28:
        v = "✅ Strong gate transition at L28"
    elif peak >= 0.30:
        v = "✅ Moderate gate transition"
    else:
        v = "⚠ Marginal"

    if not np.isnan(cos_a) and cos_a > 0.70 and fam != "A":
        v += " (shared direction with A)"
    elif not np.isnan(cos_a) and cos_a < 0.30 and fam != "A":
        v += " (independent direction from A)"

    return v


def write_report(family_dfs, metrics_df, cos_matrix, cos_fams, sign_accs, directions):
    cos_with_A = {cos_fams[j]: float(cos_matrix[cos_fams.index("A"), j])
                  for j in range(len(cos_fams))}

    lines = [
        "# Gating Family Decomposition — L28 Transition Analysis",
        "",
        "## Question",
        "",
        "Is the sharp L28 gate-clustering transition (ARI_gate: 0→0.60) driven by:",
        "- **(A)** thermodynamic gating (Family A)?",
        "- **(B)** all families equally (domain-general gate)?",
        "- **(C)** ceiling families (G/D, trivial validity)?",
        "- **(D)** dimensional/boundary gating only?",
        "",
        "## Summary Table",
        "",
        "| Family | Name | sign_acc | Peak ARI_gate | Transition L | Sharpness | Peak probe | Gate dominance | Verdict |",
        "|--------|------|----------|---------------|-------------|-----------|------------|----------------|---------|",
    ]

    for _, m in metrics_df.iterrows():
        fam   = m["family"]
        v     = verdict(m, cos_with_A, sign_accs)
        acc   = sign_accs[fam]
        lines.append(
            f"| {fam} | {FAM_NAMES[fam]} | {acc:.3f} | {m['peak_ari_gate']:.4f} | "
            f"L{m['first_gate_layer']} | {m['transition_sharp']:.4f} | "
            f"{m['peak_probe_cv']:.4f} | {m['gate_dominance']:.3f} | {v} |"
        )

    lines += [
        "",
        "## Per-Family Layer Curves",
        "",
    ]

    for fam in FAMILIES:
        df    = family_dfs[fam]
        m     = metrics_df[metrics_df["family"]==fam].iloc[0]
        valid = df[~df["degenerate"]].dropna(subset=["ari_gate"])
        lines += [
            f"### Family {fam}: {FAM_NAMES[fam]}",
            "",
            f"- Prompts: {m['n_prompts']} | sign_acc: {sign_accs[fam]:.3f}",
            f"- Peak ARI_gate: {m['peak_ari_gate']:.4f} at L{m['peak_layer']}",
            f"- First gate layer (ARI>0.30): L{m['first_gate_layer']}",
            f"- Transition sharpness: {m['transition_sharp']:.4f} (jump at L{m['sharp_at_layer']})",
            f"- Mean ARI_wording before/after transition: {m['mean_wf_before']} / {m['mean_wf_after']}",
            f"- Gate dominates wording in {m['gate_dominance']:.0%} of layers",
            f"- Peak probe CV: {m['peak_probe_cv']:.4f}",
            f"- Cosine similarity with Family A probe direction: {cos_with_A[fam]:.4f}",
            "",
            "| Layer | probe_cv | ARI_gate | ARI_wording | Gate dominant? |",
            "|-------|----------|----------|-------------|----------------|",
        ]
        for _, row in valid.sort_values("layer").iterrows():
            dom = "✓" if row["ari_gate"] > row["ari_wording"] else ""
            lines.append(
                f"| L{int(row['layer'])} | {row['probe_cv']:.4f} | "
                f"{row['ari_gate']:.4f} | {row['ari_wording']:.4f} | {dom} |"
            )
        lines.append("")

    # Cosine similarity section
    lines += [
        "## Probe Direction Cosine Similarities at L34",
        "",
        "Each family's probe direction w_family is trained at L34 on that family's prompts only.",
        "cosine(w_i, w_j) > 0.70 → shared gate direction.",
        "cosine < 0.30 → independent family-specific directions.",
        "",
        "| | " + " | ".join(FAMILIES) + " |",
        "|---|" + "---|"*len(FAMILIES),
    ]
    for i, fi in enumerate(cos_fams):
        row_str = f"| **{fi}** |"
        for j, fj in enumerate(cos_fams):
            c = cos_matrix[i, j]
            bold = "**" if abs(c) > 0.70 and i != j else ""
            row_str += f" {bold}{c:.3f}{bold} |"
        lines.append(row_str)

    lines += [
        "",
        "## Case Determination",
        "",
    ]

    # Determine which case applies
    peaks   = {fam: metrics_df[metrics_df["family"]==fam].iloc[0]["peak_ari_gate"] for fam in FAMILIES}
    cos_off = {fam: cos_with_A[fam] for fam in FAMILIES if fam != "A"}

    case_a = peaks["A"] >= 0.30 and all(peaks[f] < 0.20 for f in ["F","G","H"])
    case_c = peaks["G"] >= 0.40 and peaks["A"] < 0.20
    case_b_shared = all(peaks[f] >= 0.25 for f in FAMILIES) and all(v > 0.60 for v in cos_off.values())
    case_d_partial = peaks["A"] >= 0.30 and not all(peaks[f] >= 0.25 for f in ["F","G","H"])

    if case_c:
        case_str = "**CASE C: Ceiling families drive the transition.** G/H provide trivially easy valid/invalid separations. Thermodynamic gating is not the primary source."
    elif case_a:
        case_str = "**CASE A: Family A (thermodynamic_spontaneity) primarily drives the transition.** The entropy-direction gate is a genuine mechanistic circuit."
    elif case_b_shared and sum(v > 0.60 for v in cos_off.values()) >= 2:
        case_str = "**CASE B: Domain-general gate representation.** Multiple families show strong transitions with shared probe directions — there exists a generic 'constraint validity' manifold."
    elif case_d_partial:
        case_str = "**CASE D: Partial sharing.** Thermodynamic and some other families show the transition, but not all. The gate is real for A but not universal."
    else:
        case_str = "**MIXED: No single case clearly dominates. See individual family results.**"

    lines += [case_str, ""]

    # Recommendation
    lines += [
        "## Final Recommendation: Primary Full Mechanistic Pipeline",
        "",
        "### Selection criteria",
        "- Non-ceiling accuracy (75–92%)",
        "- Clean L28 gate transition (ARI_gate ≥ 0.30)",
        "- High pair flip rate",
        "- Strong mechanistic narrative",
        "- Intervention potential (minimal pairs for patch/ablation)",
        "- Adversarial controllability (surface-form traps possible)",
        "",
    ]

    # Rank families for mechanistic pipeline
    ranking = []
    for fam in FAMILIES:
        m        = metrics_df[metrics_df["family"]==fam].iloc[0]
        acc      = sign_accs[fam]
        # Penalise ceiling heavily
        ceil_pen = max(0, (acc - 0.92) / 0.08)
        floor_pen = max(0, (0.72 - acc) / 0.08)
        # Score: gate strength × non-ceiling × probe × sharpness
        score = (m["peak_ari_gate"] * (1 - ceil_pen) * (1 - floor_pen)
                 * m["peak_probe_cv"] * m["gate_dominance"])
        ranking.append((fam, score, m, acc))

    ranking.sort(key=lambda x: -x[1])
    top_fam, top_score, top_m, top_acc = ranking[0]

    lines += [
        f"### Rank 1: **Family {top_fam} — {FAM_NAMES[top_fam]}**",
        "",
        f"- sign_acc: {top_acc:.3f}",
        f"- Peak ARI_gate: {top_m['peak_ari_gate']:.4f} (transition at L{top_m['first_gate_layer']})",
        f"- Transition sharpness: {top_m['transition_sharp']:.4f}",
        f"- Peak probe CV: {top_m['peak_probe_cv']:.4f}",
        f"- Gate dominance: {top_m['gate_dominance']:.0%} of layers",
        f"- Mechanistic score: {top_score:.4f}",
        "",
    ]

    if len(ranking) > 1:
        sec_fam, sec_score, sec_m, sec_acc = ranking[1]
        lines += [
            f"### Rank 2: **Family {sec_fam} — {FAM_NAMES[sec_fam]}**",
            "",
            f"- sign_acc: {sec_acc:.3f}",
            f"- Peak ARI_gate: {sec_m['peak_ari_gate']:.4f} (transition at L{sec_m['first_gate_layer']})",
            f"- Mechanistic score: {sec_score:.4f}",
            "",
        ]

    # Justification for top choice
    justification = {
        "A": """
**Why thermodynamic_spontaneity:**
- The entropy-direction gate (hot→cold vs cold→hot) is the most narratively rich mechanism.
- Minimal pairs are physically grounded and intervention-ready: ablate entropy-direction features → lose asymmetry; patch blocked→allowed activations → output flips.
- sign_acc in the ideal 75–90% range — not ceiling, not ambiguous.
- The gate concept (ΔS_universe < 0 → blocked) is a genuine scientific constraint, not a statistical artefact.
- Cross-family transfer shows the gate direction partially generalises, but with family-specific refinements.
- Adversarial controllability: we can construct "sounds-impossible" allowed prompts and "sounds-obvious" blocked prompts using unusual wording.
""",
        "F": """
**Why dimensional_analysis:**
- Requires genuine computation: the model must track unit dimensions, not recall facts.
- sign_acc = 88%, flip_rate = 80% — clean behavioural signal.
- Minimal pairs are perfectly controlled: change only one exponent (ma vs ma²).
- Intervention plan: ablate "unit-mismatch detector" features → model accepts dimensionally inconsistent equations.
- Less narrative richness than thermodynamics, but potentially cleaner mechanistic circuit.
""",
        "G": """
**Why probability_statistics (NOT recommended):**
- Ceiling (98%) — almost certainly memorized knowledge.
- The gate does not require reasoning about physical processes.
""",
        "H": """
**Why physical_boundary:**
- Clean constraint (T < 0K, v > c) — precise and unambiguous.
- Moderate narrative richness.
""",
    }

    lines += [
        f"### Justification",
        justification.get(top_fam, ""),
        "",
        "*Plots: plots/gate_transition_comparison.png, plots/probe_direction_cosine_heatmap.png*",
        "",
        "*Data: data/results/gating_probe_v1/family_transition_metrics.csv*",
    ]

    path = DOCS_DIR / "gating_family_decomposition.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",  default="bfloat16",
                    choices=["float32","bfloat16","float16"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load prompts split by family ──────────────────────────────────────────
    all_prompts = [json.loads(l) for l in open(PROMPT_PATH)]
    fam_prompts = {fam: [r for r in all_prompts if r["family"] == fam]
                   for fam in FAMILIES}
    combined    = [r for r in all_prompts if r["family"] in FAMILIES]
    print(f"Total prompts: {len(combined)}")
    for fam in FAMILIES:
        print(f"  Family {fam}: {len(fam_prompts[fam])}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model…")
    mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    mw.model.eval()
    tok = mw.tokenizer

    # ── Single forward pass over all prompts ──────────────────────────────────
    print(f"\nExtracting hidden states at layers {PROBE_LAYERS[0]}–{PROBE_LAYERS[-1]}…")
    all_hs = extract_hidden_states(mw.model, tok, combined, PROBE_LAYERS, device)

    # Map each prompt in combined to its index within its family
    fam_indices = {fam: [] for fam in FAMILIES}
    for i, r in enumerate(combined):
        if r["family"] in FAMILIES:
            fam_indices[r["family"]].append(i)

    # Build per-family hidden state dicts
    fam_hs = {}
    for fam in FAMILIES:
        idxs = fam_indices[fam]
        fam_hs[fam] = {l: all_hs[l][idxs] for l in PROBE_LAYERS}

    # ── Per-family layer analysis ─────────────────────────────────────────────
    print("\nAnalysing per-family…")
    family_dfs = {}
    metrics_rows = []
    sign_accs = {}

    for fam in FAMILIES:
        print(f"\n  Family {fam}: {FAM_NAMES[fam]}")
        df = analyse_family(fam_prompts[fam], fam_hs[fam], PROBE_LAYERS)
        family_dfs[fam] = df

        # Print layer table
        valid = df[~df["degenerate"]].dropna(subset=["ari_gate"])
        for _, row in valid.sort_values("layer").iterrows():
            marker = " *** JUMP" if row["layer"] == 28 and row["ari_gate"] > 0.25 else ""
            print(f"    L{int(row['layer']):2d}: probe={row['probe_cv']:.4f}  "
                  f"ARI_gate={row['ari_gate']:+.4f}  ARI_wf={row['ari_wording']:.4f}{marker}")

        m = transition_metrics(df, fam)
        m["n_prompts"] = len(fam_prompts[fam])

        # Sign accuracy from baseline CSV
        bl_csv = OUT_DIR / "baseline_results.csv"
        if bl_csv.exists():
            bl = pd.read_csv(bl_csv)
            acc = float(bl[bl["family"]==fam]["correct"].mean())
        else:
            acc = float("nan")
        sign_accs[fam] = acc

        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "family_transition_metrics.csv", index=False)

    # ── Probe direction cosine similarities at L34 ────────────────────────────
    print("\nComputing probe directions at L34…")
    directions = {}
    for fam in FAMILIES:
        X = fam_hs[fam][34]
        y = np.array([1 if r["gate_label"]=="allow" else 0
                      for r in fam_prompts[fam]])
        w = train_probe_direction(X, y)
        directions[fam] = w
        print(f"  {fam}: ||w||={np.linalg.norm(w):.4f} (should be 1.0)")

    # Save directions for downstream use
    np.savez(OUT_DIR / "probe_directions_L34.npz", **directions)

    cos_matrix, cos_fams = plot_cosine_heatmap(directions)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_per_family(family_dfs, "probe_cv",    "Probe CV accuracy", "per_family_probe_cv.png",    ylim=(0.4, 1.05))
    plot_per_family(family_dfs, "ari_gate",    "ARI(gate_label)",   "per_family_ARI_gate.png",    ylim=(-0.1, 0.8))
    plot_per_family(family_dfs, "ari_wording", "ARI(wording)",      "per_family_ARI_wording.png", ylim=(-0.1, 1.0))
    plot_comparison(family_dfs)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== TRANSITION METRICS ===")
    print(metrics_df[["family","peak_ari_gate","first_gate_layer",
                       "transition_sharp","peak_probe_cv","gate_dominance"]].to_string(index=False))

    print("\n=== COSINE SIMILARITY (L34 probe directions) ===")
    for i, fi in enumerate(cos_fams):
        row = "  " + fi + ": "
        for j, fj in enumerate(cos_fams):
            row += f"  {fj}={cos_matrix[i,j]:+.3f}"
        print(row)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\nWriting report…")
    write_report(family_dfs, metrics_df, cos_matrix, cos_fams, sign_accs, directions)

    print("\nDone.")


if __name__ == "__main__":
    main()
