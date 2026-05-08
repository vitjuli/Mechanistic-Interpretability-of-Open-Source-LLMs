"""
Candidate trajectory analysis: how candidate representations evolve L10→L25.

For each prompt and each layer, computes a candidate score for each particle
(electron/proton/neutron/photon). Uses candidate-associated features from
script 41 (candidate_feature_table.csv). Falls back to graph features if
script 41 has not been run.

Per-layer metrics:
  candidate_rank_accuracy   — how often is correct particle rank-1?
  mean_correct_score        — mean score of correct particle
  mean_best_competitor_score — mean score of best non-correct particle
  margin                    — mean(correct - best_competitor)
  entropy                   — mean H(softmax(scores)) over 4 candidates

Key scientific question: does the model activate multiple candidates early
(high entropy, low margin) and converge on one late (low entropy, high margin)?
Expected Anthropic-like pattern: multi-candidate early → single-candidate late.

Usage:
  python scripts/42_candidate_trajectory_analysis.py
  python scripts/42_candidate_trajectory_analysis.py --no_plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BEHAVIOUR = "physics_internal_candidate_selection_v2"
SPLIT     = "train"
LAYERS    = list(range(10, 26))
PARTICLES = ["electron", "proton", "neutron", "photon"]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def get_paths(behaviour: str, split: str) -> dict:
    base = Path("data")
    analysis_dir = base / "results" / "internal_candidate_analysis" / behaviour
    return {
        "prompts":      base / "prompts" / f"{behaviour}_{split}.jsonl",
        "graph_json":   base / "results" / "attribution_graphs" / behaviour
                             / f"attribution_graph_{split}_n120_roleaware.json",
        "circuit_json": base / "results" / f"circuits_{behaviour}_{split}_roleaware.json",
        "feature_dir":  base / "results" / "transcoder_features",
        "candidate_table": analysis_dir / "candidate_feature_table.csv",
        "output_dir":   analysis_dir,
    }


def rsync_cmd(behaviour, split):
    remote = "iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
    local  = "data/results/transcoder_features"
    return (
        f"for L in {{10..25}}; do\n"
        f'  rsync -av "{remote}/data/results/transcoder_features/layer_${{L}}/'
        f'{behaviour}_{split}_top_k_"*.npy "{local}/layer_${{L}}/"\n'
        f"done"
    )


# ─── Loading helpers ──────────────────────────────────────────────────────────

def load_prompts(paths):
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct_stripped"] = r["correct_answer"].strip()
        r["_pool_set"] = set(r["implicit_candidate_pool"])
    return [p for p in rows if not p.get("multi_token_answer", False)]


def load_layer_features(behaviour, split, layer, feature_dir):
    layer_dir = feature_dir / f"layer_{layer}"
    idx_path  = layer_dir / f"{behaviour}_{split}_top_k_indices.npy"
    val_path  = layer_dir / f"{behaviour}_{split}_top_k_values.npy"
    if not idx_path.exists() or not val_path.exists():
        return None
    return np.load(idx_path), np.load(val_path)


def get_activation(indices, values, feat_idx):
    act = np.zeros(indices.shape[0], dtype=np.float32)
    rows, cols = np.where(indices == feat_idx)
    act[rows] = values[rows, cols]
    return act


# ─── Build candidate feature map from script 41 output ───────────────────────

def load_candidate_features(paths, min_specificity=0.0):
    """
    Returns {particle: {layer: [feat_idx, ...]}} for strong candidate features.
    Falls back to top graph features by specific_score if script 41 not run.
    """
    cand_feats: dict[str, dict[int, list]] = {p: {} for p in PARTICLES}

    if paths["candidate_table"].exists():
        df = pd.read_csv(paths["candidate_table"])
        # Use strong features first
        strong = df[df["strong_candidate_feature"] == True]
        if len(strong) == 0:
            print("[WARN] No strong features in candidate_feature_table.csv — using all T>C>B ordering")
            strong = df[df["ordering_T_gt_C_gt_B"] == True]
        if len(strong) == 0:
            print("[WARN] No T>C>B features — falling back to top candidate_specificity")
            strong = df[df["candidate_specificity"] > min_specificity]

        for _, row in strong.iterrows():
            p = row["particle"]
            l = int(row["layer"])
            f = int(row["feature_idx"])
            cand_feats[p].setdefault(l, []).append(f)
        print(f"Loaded candidate features: "
              f"{sum(len(v) for p in cand_feats.values() for v in p.values())} total")
        return cand_feats, "script_41"

    # Fallback: graph features per layer, assigned to particle by highest target_mean
    # (without script 41 results, use specific_score as a rough proxy)
    print("[WARN] candidate_feature_table.csv not found — using graph features as fallback")
    print("Run script 41 first for best results.")

    with open(paths["graph_json"]) as f:
        g = json.load(f)
    feat_nodes = [n for n in g["nodes"] if n.get("type") == "feature"]

    # Sort by specific_score and assign top-5 per layer to all particles
    for n in sorted(feat_nodes, key=lambda x: x.get("specific_score", 0), reverse=True):
        layer = n["layer"]
        fidx  = n["feature_idx"]
        for p in PARTICLES:
            cand_feats[p].setdefault(layer, []).append(fidx)

    return cand_feats, "graph_fallback"


# ─── Candidate score computation ─────────────────────────────────────────────

def compute_scores_at_layer(indices, values, cand_feats, layer, n_prompts):
    """Returns {particle: np.ndarray[n_prompts]} activation scores."""
    scores = {}
    for particle in PARTICLES:
        feat_list = cand_feats[particle].get(layer, [])
        if not feat_list:
            scores[particle] = np.zeros(n_prompts, dtype=np.float32)
            continue
        acts = np.stack([get_activation(indices, values, f) for f in feat_list], axis=1)
        scores[particle] = acts.mean(axis=1)
    return scores


def softmax_entropy(scores_matrix):
    """scores_matrix: (N, 4) → entropy per prompt."""
    # Shift for numerical stability
    shifted = scores_matrix - scores_matrix.max(axis=1, keepdims=True)
    exp_s   = np.exp(shifted)
    probs   = exp_s / exp_s.sum(axis=1, keepdims=True)
    # Clip to avoid log(0)
    probs   = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


# ─── Main trajectory computation ─────────────────────────────────────────────

def run_trajectory(behaviour, split, paths, cand_feats):
    prompts = load_prompts(paths)
    n       = len(prompts)
    correct_labels = [p["_correct_stripped"] for p in prompts]
    print(f"Prompts: {n}")

    # Per-layer metrics
    layer_rows  = []
    # Per-prompt × layer data
    traj_rows   = []
    # Rank evolution: (n_prompts, n_layers) arrays
    rank_matrix       = np.full((n, len(LAYERS)), fill_value=np.nan)
    margin_matrix     = np.full((n, len(LAYERS)), fill_value=np.nan)
    entropy_matrix    = np.full((n, len(LAYERS)), fill_value=np.nan)
    correct_score_mat = np.full((n, len(LAYERS)), fill_value=np.nan)

    for li, layer in enumerate(LAYERS):
        result = load_layer_features(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            print(f"  L{layer}: missing features")
            continue
        indices, values = result
        scores = compute_scores_at_layer(indices, values, cand_feats, layer, n)

        # Stack (N, 4)
        score_matrix = np.stack([scores[p] for p in PARTICLES], axis=1)

        # Rank of correct candidate (0 = best)
        particle_idx = {p: i for i, p in enumerate(PARTICLES)}
        correct_col  = np.array([particle_idx.get(c, 0) for c in correct_labels])
        correct_scores = score_matrix[np.arange(n), correct_col]
        ranks = (score_matrix > correct_scores[:, None]).sum(axis=1)  # 0=rank1, 3=rank4

        best_competitor_scores = score_matrix.copy()
        best_competitor_scores[np.arange(n), correct_col] = -np.inf
        best_comp_scores = best_competitor_scores.max(axis=1)
        margins = correct_scores - best_comp_scores
        entropies = softmax_entropy(score_matrix)

        # Store per-prompt values
        rank_matrix[:, li]       = ranks
        margin_matrix[:, li]     = margins
        entropy_matrix[:, li]    = entropies
        correct_score_mat[:, li] = correct_scores

        # Layer-level stats
        layer_rows.append({
            "layer":                      layer,
            "candidate_rank_accuracy":    float((ranks == 0).mean()),
            "mean_correct_score":         float(correct_scores.mean()),
            "mean_best_competitor_score": float(best_comp_scores.mean()),
            "mean_margin":                float(margins.mean()),
            "std_margin":                 float(margins.std()),
            "mean_entropy":               float(entropies.mean()),
            "pct_rank1":                  float((ranks == 0).mean()),
            "pct_rank2":                  float((ranks == 1).mean()),
            "pct_rank3_or_4":             float((ranks >= 2).mean()),
        })

        # Per-prompt rows for a sample of prompts
        for i in range(n):
            traj_rows.append({
                "prompt_idx":    i,
                "layer":         layer,
                "correct_answer": correct_labels[i],
                "filter_property": prompts[i].get("filter_property", ""),
                "wording_family":  prompts[i].get("wording_family", ""),
                "correct_score": float(correct_scores[i]),
                "competitor_score": float(best_comp_scores[i]),
                "margin":        float(margins[i]),
                "rank":          int(ranks[i]),
                "entropy":       float(entropies[i]),
                "score_electron": float(scores["electron"][i]),
                "score_proton":   float(scores["proton"][i]),
                "score_neutron":  float(scores["neutron"][i]),
                "score_photon":   float(scores["photon"][i]),
            })

        pct_r1 = (ranks == 0).mean()
        mean_m = margins.mean()
        mean_e = entropies.mean()
        print(f"  L{layer}: rank_acc={pct_r1:.3f}, margin={mean_m:.3f}, entropy={mean_e:.3f}")

    layer_df  = pd.DataFrame(layer_rows)
    traj_df   = pd.DataFrame(traj_rows)

    # Rank evolution: per-prompt first layer at which correct reaches rank-1
    first_rank1_layer = []
    for i in range(n):
        rank_vec = rank_matrix[i, :]
        valid    = np.where(np.isfinite(rank_vec) & (rank_vec == 0))[0]
        first_rank1_layer.append(LAYERS[valid[0]] if len(valid) else -1)

    rank_evo_df = pd.DataFrame({
        "prompt_idx":           range(n),
        "correct_answer":       correct_labels,
        "filter_property":      [p.get("filter_property", "") for p in prompts],
        "wording_family":       [p.get("wording_family", "") for p in prompts],
        "first_rank1_layer":    first_rank1_layer,
        "final_rank":           rank_matrix[:, -1].astype(float),
        "final_margin":         margin_matrix[:, -1],
        "final_entropy":        entropy_matrix[:, -1],
        "mean_margin_L10_13":   margin_matrix[:, :4].mean(axis=1),
        "mean_margin_L22_25":   margin_matrix[:, -4:].mean(axis=1),
        "margin_increase":      margin_matrix[:, -4:].mean(axis=1) - margin_matrix[:, :4].mean(axis=1),
    })

    return layer_df, traj_df, rank_evo_df, rank_matrix, margin_matrix, entropy_matrix


# ─── Plotting ─────────────────────────────────────────────────────────────────

def make_plots(layer_df, rank_matrix, margin_matrix, entropy_matrix,
               traj_df, prompts, output_dir):
    if not MATPLOTLIB:
        print("[SKIP] matplotlib not available — skipping plots")
        return

    layer_arr = np.array(LAYERS)

    # ── Fig 1: rank accuracy by layer ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(layer_df["layer"], layer_df["candidate_rank_accuracy"], "o-", color="steelblue", lw=2)
    ax.axhline(0.25, color="gray", linestyle="--", label="chance (k=4)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of prompts where correct is rank-1")
    ax.set_title("Candidate Rank Accuracy by Layer")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(layer_df["layer"], layer_df["mean_margin"], "o-", color="darkorange", lw=2, label="mean margin")
    ax.fill_between(
        layer_df["layer"],
        layer_df["mean_margin"] - layer_df["std_margin"],
        layer_df["mean_margin"] + layer_df["std_margin"],
        alpha=0.2, color="darkorange"
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score margin (correct − best competitor)")
    ax.set_title("Candidate Score Margin by Layer")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(layer_df["layer"], layer_df["mean_entropy"], "o-", color="mediumpurple", lw=2)
    ax.axhline(np.log(4), color="gray", linestyle="--", label="max entropy (k=4)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean entropy H(softmax(scores))")
    ax.set_title("Candidate Score Entropy by Layer")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "candidate_trajectory_overview.png", dpi=150)
    plt.close(fig)

    # ── Fig 2: per-particle rank accuracy ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    particle_idx = {p: i for i, p in enumerate(PARTICLES)}
    colors = {"electron": "#1f77b4", "proton": "#ff7f0e",
               "neutron": "#2ca02c", "photon": "#d62728"}

    correct_labels = traj_df[traj_df["layer"] == LAYERS[0]]["correct_answer"].tolist()
    for particle in PARTICLES:
        mask = np.array([c == particle for c in correct_labels])
        if mask.sum() == 0:
            continue
        # Rank of correct particle across layers for this subset
        rank_sub = rank_matrix[mask, :]
        acc_by_layer = [(rank_sub[:, li] == 0).mean() for li in range(len(LAYERS))]
        ax.plot(LAYERS, acc_by_layer, "o-", label=particle, color=colors.get(particle),
                alpha=0.85, lw=2)

    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank-1 accuracy")
    ax.set_title("Rank-1 Accuracy by Particle and Layer")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_accuracy_by_layer.png", dpi=150)
    plt.close(fig)

    # ── Fig 3: example prompts scorecard ─────────────────────────────────────
    # Pick 4 example prompts (one per particle)
    ex_prompts: dict[str, int] = {}
    for particle in PARTICLES:
        sub = traj_df[(traj_df["correct_answer"] == particle) & (traj_df["layer"] == LAYERS[0])]
        if len(sub):
            # pick a prompt with high final margin
            top = traj_df[
                (traj_df["correct_answer"] == particle) & (traj_df["layer"] == LAYERS[-1])
            ].sort_values("margin", ascending=False)
            if len(top):
                ex_prompts[particle] = int(top.iloc[0]["prompt_idx"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (particle, pidx) in zip(axes.flatten(), ex_prompts.items()):
        prompt_data = traj_df[traj_df["prompt_idx"] == pidx].sort_values("layer")
        for p2 in PARTICLES:
            col = colors.get(p2, "gray")
            ls  = "-" if p2 == particle else "--"
            lw  = 2.5 if p2 == particle else 1.0
            ax.plot(prompt_data["layer"], prompt_data[f"score_{p2}"],
                    label=p2, color=col, lw=lw, ls=ls)
        ax.set_title(f"Correct: {particle} (prompt {pidx})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Candidate activation score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_scores_example_prompts.png", dpi=150)
    plt.close(fig)

    # ── Fig 4: margin histogram early vs late ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    early_layers = [l for l in LAYERS if l <= 14]
    late_layers  = [l for l in LAYERS if l >= 22]

    early_idx = [LAYERS.index(l) for l in early_layers]
    late_idx  = [LAYERS.index(l) for l in late_layers]

    early_margins = margin_matrix[:, early_idx].mean(axis=1)
    late_margins  = margin_matrix[:, late_idx].mean(axis=1)

    for ax, (margins, label) in zip(axes, [(early_margins, "Early (L10-L14)"),
                                            (late_margins, "Late (L22-L25)")]):
        ax.hist(margins, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", lw=1.5, label="margin=0")
        ax.axvline(np.nanmean(margins), color="orange", lw=1.5,
                   linestyle="--", label=f"mean={np.nanmean(margins):.3f}")
        ax.set_title(f"Score Margin Distribution — {label}")
        ax.set_xlabel("margin (correct − best competitor)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_margin_by_layer.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_trajectory_report(layer_df, rank_evo_df, output_dir):
    lines = [
        "# Candidate Trajectory Analysis Report",
        f"## {BEHAVIOUR} | {SPLIT}",
        "",
        "## Key question: does the model activate multiple candidates early and converge late?",
        "",
    ]

    # Early vs late summary
    early = layer_df[layer_df["layer"] <= 14]
    late  = layer_df[layer_df["layer"] >= 22]

    e_acc = early["candidate_rank_accuracy"].mean() if len(early) else float("nan")
    l_acc = late["candidate_rank_accuracy"].mean()  if len(late)  else float("nan")
    e_ent = early["mean_entropy"].mean() if len(early) else float("nan")
    l_ent = late["mean_entropy"].mean()  if len(late)  else float("nan")
    e_mar = early["mean_margin"].mean()  if len(early) else float("nan")
    l_mar = late["mean_margin"].mean()   if len(late)  else float("nan")

    lines += [
        "| Metric | Early (L10-L14) | Late (L22-L25) | Change |",
        "|---|---|---|---|",
        f"| Rank-1 accuracy | {e_acc:.3f} | {l_acc:.3f} | {l_acc-e_acc:+.3f} |",
        f"| Mean entropy | {e_ent:.3f} | {l_ent:.3f} | {l_ent-e_ent:+.3f} |",
        f"| Mean margin | {e_mar:.3f} | {l_mar:.3f} | {l_mar-e_mar:+.3f} |",
        "",
    ]

    if l_acc > e_acc + 0.1 and e_ent > l_ent:
        pattern = "**MULTI-THEN-SINGLE**: consistent with internal candidate pool (Anthropic-like pattern)"
    elif l_acc > e_acc + 0.05:
        pattern = "**GRADUAL IMPROVEMENT**: candidate strengthens across layers"
    elif e_acc > 0.8:
        pattern = "**EARLY SELECTION**: model selects correct candidate already at early layers"
    else:
        pattern = "**UNCLEAR**: pattern does not clearly match either hypothesis"

    lines += [f"### Pattern: {pattern}", ""]

    # First-rank1-layer distribution
    lines += [
        "## First layer at which correct candidate reaches rank-1",
        "",
        "| Layer | n prompts | fraction |",
        "|---|---|---|",
    ]
    first_counts = rank_evo_df["first_rank1_layer"].value_counts().sort_index()
    total = len(rank_evo_df)
    for layer, cnt in first_counts.items():
        lines.append(f"| {'never' if layer == -1 else f'L{int(layer)}'} | {cnt} | {cnt/total:.2f} |")
    lines += [""]

    # Per-layer table
    lines += [
        "## Layer-by-layer trajectory",
        "",
        "| Layer | rank_acc | mean_correct_score | mean_comp_score | margin | entropy |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in layer_df.sort_values("layer").iterrows():
        lines.append(
            f"| L{int(r['layer'])} | {r['candidate_rank_accuracy']:.3f} | "
            f"{r['mean_correct_score']:.3f} | {r['mean_best_competitor_score']:.3f} | "
            f"{r['mean_margin']:.3f} | {r['mean_entropy']:.3f} |"
        )
    lines += [""]

    (output_dir / "report_trajectory.md").write_text("\n".join(lines))
    print(f"Trajectory report written to {output_dir}/report_trajectory.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default=BEHAVIOUR)
    ap.add_argument("--split",     default=SPLIT)
    ap.add_argument("--no_plots",  action="store_true")
    ap.add_argument("--min_specificity", type=float, default=0.0)
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split)

    has_any = any(
        (paths["feature_dir"] / f"layer_{l}" / f"{args.behaviour}_{args.split}_top_k_indices.npy").exists()
        for l in LAYERS
    )
    if not has_any:
        print(f"[ERROR] No transcoder feature files found for {args.behaviour}_{args.split}")
        print("Run:\n" + rsync_cmd(args.behaviour, args.split))
        sys.exit(1)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    cand_feats, source = load_candidate_features(paths, args.min_specificity)
    print(f"Candidate features loaded from: {source}")

    prompts = load_prompts(paths)
    layer_df, traj_df, rank_evo_df, rank_mat, margin_mat, entropy_mat = run_trajectory(
        args.behaviour, args.split, paths, cand_feats
    )

    out = paths["output_dir"]
    layer_df.to_csv(out / "candidate_trajectory_by_layer.csv", index=False)
    traj_df.to_csv(out  / "candidate_trajectory_by_prompt.csv", index=False)
    rank_evo_df.to_csv(out / "candidate_rank_evolution.csv", index=False)
    np.save(out / "candidate_margin_matrix.npy", margin_mat)
    print(f"Saved trajectory tables to {out}/")

    if not args.no_plots:
        make_plots(layer_df, rank_mat, margin_mat, entropy_mat, traj_df, prompts, out)

    write_trajectory_report(layer_df, rank_evo_df, out)

    # Console summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print("\nLayer | rank_acc | margin | entropy")
    for _, r in layer_df.sort_values("layer").iterrows():
        bar = "█" * int(r["candidate_rank_accuracy"] * 20)
        print(f"  L{int(r['layer']):2d} | {r['candidate_rank_accuracy']:.3f} {bar:<20s} | "
              f"margin={r['mean_margin']:.3f} | entropy={r['mean_entropy']:.3f}")


if __name__ == "__main__":
    main()
