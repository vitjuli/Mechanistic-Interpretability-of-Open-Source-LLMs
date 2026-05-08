"""
Internal candidate-state analysis for physics_internal_candidate_selection_v2.

For each transcoder feature at each layer L10-L25, splits all prompts into
three groups per candidate particle:

  Target group    — correct_answer == c
  Competitor group — c in implicit_candidate_pool but not correct
  Background group — c not in implicit_candidate_pool at all

Computes:
  target_activation_mean, competitor_activation_mean, background_activation_mean
  target_attr_mean, competitor_attr_mean, background_attr_mean (proxy)
  candidate_specificity  = target_mean - background_mean
  competitor_presence    = competitor_mean - background_mean
  internal_presence_ratio = competitor_mean / target_mean   (IPR)
  ordering               = target > competitor > background
  Mann-Whitney tests for T>C and C>B (FDR corrected per layer)

Requires transcoder feature files (from script 04):
  data/results/transcoder_features/layer_{L}/{behaviour}_{split}_top_k_indices.npy
  data/results/transcoder_features/layer_{L}/{behaviour}_{split}_top_k_values.npy

Usage:
  python scripts/41_internal_candidate_state_analysis.py
  python scripts/41_internal_candidate_state_analysis.py --behaviour physics_internal_candidate_selection_v2
  python scripts/41_internal_candidate_state_analysis.py --feature_set all_graph circuit_only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─── Configuration ────────────────────────────────────────────────────────────

BEHAVIOUR  = "physics_internal_candidate_selection_v2"
SPLIT      = "train"
LAYERS     = list(range(10, 26))
PARTICLES  = ["electron", "proton", "neutron", "photon"]
MIN_GROUP_SIZE = 5   # skip stat test if group smaller than this

# ─── Paths ────────────────────────────────────────────────────────────────────

def get_paths(behaviour: str, split: str) -> dict:
    base = Path("data")
    return {
        "prompts":       base / "prompts" / f"{behaviour}_{split}.jsonl",
        "baseline_csv":  base / "results" / f"baseline_{behaviour}_{split}.csv",
        "graph_json":    base / "results" / "attribution_graphs" / behaviour
                              / f"attribution_graph_{split}_n120_roleaware.json",
        "circuit_json":  base / "results" / f"circuits_{behaviour}_{split}_roleaware.json",
        "feature_dir":   base / "results" / "transcoder_features",
        "output_dir":    base / "results" / "internal_candidate_analysis" / behaviour,
    }


def rsync_cmd(behaviour: str, split: str) -> str:
    remote = "iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
    local  = "data/results/transcoder_features"
    lines = [
        f"# Sync transcoder feature files for {behaviour}_{split}:",
        f"for L in {{10..25}}; do",
        f'  rsync -av "{remote}/data/results/transcoder_features/layer_${{L}}/'
        f'{behaviour}_{split}_top_k_"*.npy \\\n'
        f'    "{local}/layer_${{L}}/"',
        f"done",
    ]
    return "\n".join(lines)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_prompts(paths: dict) -> list[dict]:
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    # Normalise: pool entries have no leading space; correct_answer has space
    for r in rows:
        r["_correct_stripped"] = r["correct_answer"].strip()
        r["_pool_set"] = set(r["implicit_candidate_pool"])
    return rows


def load_graph_features(paths: dict) -> dict:
    """Returns {(layer, feat_idx): node_dict} for all graph feature nodes."""
    with open(paths["graph_json"]) as f:
        g = json.load(f)
    result = {}
    for n in g["nodes"]:
        if n.get("type") == "feature":
            result[(n["layer"], n["feature_idx"])] = n
    return result


def load_circuit_features(paths: dict) -> set[tuple]:
    """Returns {(layer, feat_idx)} for all circuit feature nodes."""
    with open(paths["circuit_json"]) as f:
        c = json.load(f)
    circ = c.get("circuit", {})
    result = set()
    for node_id in circ.get("feature_nodes", []):
        if node_id.startswith("L") and "_F" in node_id:
            parts = node_id.split("_F")
            layer = int(parts[0][1:])
            fidx  = int(parts[1])
            result.add((layer, fidx))
    return result


def load_layer_features(
    behaviour: str, split: str, layer: int, feature_dir: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads (top_k_indices [N,50], top_k_values [N,50]) or None if missing."""
    layer_dir = feature_dir / f"layer_{layer}"
    idx_path  = layer_dir / f"{behaviour}_{split}_top_k_indices.npy"
    val_path  = layer_dir / f"{behaviour}_{split}_top_k_values.npy"
    if not idx_path.exists() or not val_path.exists():
        return None
    return np.load(idx_path), np.load(val_path)


def get_activation_for_feature(
    indices: np.ndarray, values: np.ndarray, feat_idx: int
) -> np.ndarray:
    """Returns activation vector [N] for feat_idx; 0 if not in top-k."""
    n_prompts = indices.shape[0]
    act = np.zeros(n_prompts, dtype=np.float32)
    match = indices == feat_idx           # shape (N, 50) bool
    rows, cols = np.where(match)
    act[rows] = values[rows, cols]
    return act


# ─── Group assignment ─────────────────────────────────────────────────────────

def assign_groups(prompts: list[dict], particle: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (target_mask, competitor_mask, background_mask) boolean arrays."""
    n = len(prompts)
    target     = np.zeros(n, dtype=bool)
    competitor = np.zeros(n, dtype=bool)
    background = np.zeros(n, dtype=bool)
    for i, r in enumerate(prompts):
        correct = r["_correct_stripped"]
        pool    = r["_pool_set"]
        if correct == particle:
            target[i] = True
        elif particle in pool:
            competitor[i] = True
        else:
            background[i] = True
    return target, competitor, background


# ─── Statistics ───────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def mw_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U (one-sided a>b). Returns (U, p_value)."""
    if len(a) < MIN_GROUP_SIZE or len(b) < MIN_GROUP_SIZE:
        return float("nan"), float("nan")
    try:
        result = stats.mannwhitneyu(a, b, alternative="greater")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def fdr_bh(p_values: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pv = np.array(p_values, dtype=float)
    n  = len(pv)
    if n == 0:
        return pv
    nan_mask = np.isnan(pv)
    valid_pv = pv[~nan_mask]
    if len(valid_pv) == 0:
        return pv
    order   = np.argsort(valid_pv)
    ranked  = np.empty_like(valid_pv)
    ranked[order] = np.arange(1, len(valid_pv) + 1)
    adjusted = np.minimum(1.0, valid_pv * len(valid_pv) / ranked)
    # Ensure monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = pv.copy()
    result[~nan_mask] = adjusted
    return result


# ─── Per-feature analysis ────────────────────────────────────────────────────

def analyse_feature(
    act: np.ndarray,
    masks: dict[str, np.ndarray],
    node: dict | None,
) -> dict:
    """Compute T/C/B group stats and attribution proxy for one feature."""
    row: dict = {}
    for group, mask in masks.items():
        acts_g = act[mask]
        row[f"{group}_mean"]       = float(np.mean(acts_g)) if len(acts_g) else float("nan")
        row[f"{group}_std"]        = float(np.std(acts_g))  if len(acts_g) else float("nan")
        row[f"{group}_n"]          = int(mask.sum())
        row[f"{group}_active_frac"] = float((acts_g > 0).mean()) if len(acts_g) else float("nan")

    t_mean = row["target_mean"]
    c_mean = row["competitor_mean"]
    b_mean = row["background_mean"]
    row["candidate_specificity"] = (t_mean - b_mean) if not np.isnan(t_mean + b_mean) else float("nan")
    row["competitor_presence"]   = (c_mean - b_mean) if not np.isnan(c_mean + b_mean) else float("nan")
    row["internal_presence_ratio"] = (c_mean / t_mean) if (not np.isnan(t_mean) and t_mean > 0) else float("nan")
    row["ordering_T_gt_C_gt_B"]    = bool(t_mean > c_mean > b_mean) if not any(
        np.isnan([t_mean, c_mean, b_mean])
    ) else False

    # Attribution proxy: scale activation by attribution-per-unit-activation from graph
    if node is not None:
        m_act  = node.get("mean_activation_conditional", 0)
        m_attr = node.get("mean_grad_attr_conditional", 0)
        attr_scale = (m_attr / m_act) if m_act > 0 else 0.0
        attr = act * attr_scale
        for group, mask in masks.items():
            attrs_g = attr[mask]
            row[f"{group}_attr_mean"] = float(np.mean(attrs_g)) if len(attrs_g) else float("nan")
        row["graph_mean_attr"]    = float(node.get("mean_grad_attr_conditional", float("nan")))
        row["graph_specific_score"] = float(node.get("specific_score", float("nan")))
        row["graph_frequency"]    = float(node.get("frequency", float("nan")))
        row["graph_position_role"]  = node.get("position_role", "")
        row["graph_causal_status"]  = node.get("causal_status", "")
    else:
        for group in masks:
            row[f"{group}_attr_mean"] = float("nan")
        row["graph_mean_attr"]     = float("nan")
        row["graph_specific_score"] = float("nan")
        row["graph_frequency"]     = float("nan")
        row["graph_position_role"] = ""
        row["graph_causal_status"] = ""

    # Stat tests (before FDR)
    target_acts     = act[masks["target"]]
    competitor_acts = act[masks["competitor"]]
    background_acts = act[masks["background"]]
    row["mw_U_TC"], row["mw_p_TC"] = mw_test(target_acts, competitor_acts)
    row["mw_U_CB"], row["mw_p_CB"] = mw_test(competitor_acts, background_acts)
    row["cohen_d_TC"] = cohen_d(target_acts, competitor_acts)
    row["cohen_d_CB"] = cohen_d(competitor_acts, background_acts)
    return row


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_analysis(
    behaviour: str,
    split: str,
    feature_sets: list[str],
    single_token_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    paths = get_paths(behaviour, split)

    # Load metadata
    prompts = load_prompts(paths)
    if single_token_only:
        prompts = [p for p in prompts if not p.get("multi_token_answer", False)]
    print(f"Prompts: {len(prompts)} ({'single-token only' if single_token_only else 'all'})")

    graph_features  = load_graph_features(paths)
    circuit_features = load_circuit_features(paths)
    print(f"Graph features: {len(graph_features)}, circuit features: {len(circuit_features)}")

    # Group masks per particle
    masks: dict[str, dict] = {}
    for particle in PARTICLES:
        target, competitor, background = assign_groups(prompts, particle)
        masks[particle] = {
            "target": target, "competitor": competitor, "background": background
        }
        print(
            f"  {particle}: target={target.sum()}, "
            f"competitor={competitor.sum()}, background={background.sum()}"
        )

    # Per-layer analysis
    all_rows: list[dict] = []
    missing_layers: list[int] = []

    for layer in LAYERS:
        result = load_layer_features(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            missing_layers.append(layer)
            continue
        indices, values = result
        print(f"\nLayer {layer}: indices={indices.shape}, values={values.shape}")

        # Determine which features to analyse at this layer
        features_to_analyse: list[tuple[int, str, dict | None]] = []

        if "circuit_only" in feature_sets:
            for (l, fidx) in circuit_features:
                if l == layer:
                    node = graph_features.get((layer, fidx))
                    features_to_analyse.append((fidx, "circuit", node))

        if "all_graph" in feature_sets:
            for (l, fidx), node in graph_features.items():
                if l == layer:
                    tag = "circuit" if (layer, fidx) in circuit_features else "graph"
                    if not any(f[0] == fidx for f in features_to_analyse):
                        features_to_analyse.append((fidx, tag, node))

        if "top_active" in feature_sets:
            # Top-50 most frequently active features (union across prompts)
            unique_feats, counts = np.unique(indices.flatten(), return_counts=True)
            top_idx = np.argsort(counts)[-50:]
            for fidx in unique_feats[top_idx]:
                if not any(f[0] == fidx for f in features_to_analyse):
                    features_to_analyse.append((int(fidx), "top_active", graph_features.get((layer, int(fidx)))))

        print(f"  Features to analyse: {len(features_to_analyse)}")

        for feat_idx, feat_set, node in features_to_analyse:
            act = get_activation_for_feature(indices, values, feat_idx)
            for particle in PARTICLES:
                row = analyse_feature(act, masks[particle], node)
                row["layer"]      = layer
                row["feature_idx"] = feat_idx
                row["feature_id"]  = f"L{layer}_F{feat_idx}"
                row["particle"]    = particle
                row["feature_set"] = feat_set
                all_rows.append(row)

    if missing_layers:
        print(f"\n[WARN] Missing layers: {missing_layers}")
        print(rsync_cmd(behaviour, split))

    df = pd.DataFrame(all_rows)

    # FDR correction per (layer, particle) group
    df["mw_p_TC_adj"] = float("nan")
    df["mw_p_CB_adj"] = float("nan")
    for (layer, particle), grp in df.groupby(["layer", "particle"]):
        tc_adj = fdr_bh(grp["mw_p_TC"].tolist())
        cb_adj = fdr_bh(grp["mw_p_CB"].tolist())
        df.loc[grp.index, "mw_p_TC_adj"] = tc_adj
        df.loc[grp.index, "mw_p_CB_adj"] = cb_adj

    # Significant T>C>B: both p_TC_adj < 0.05 and p_CB_adj < 0.05 and ordering correct
    df["sig_TC"] = (df["mw_p_TC_adj"] < 0.05)
    df["sig_CB"] = (df["mw_p_CB_adj"] < 0.05)
    df["strong_candidate_feature"] = df["ordering_T_gt_C_gt_B"] & df["sig_TC"] & df["sig_CB"]
    df["partial_candidate_feature"] = (
        (df["candidate_specificity"] > 0) & df["sig_TC"] & ~df["sig_CB"]
    )

    # ── Summary by layer ─────────────────────────────────────────────────────
    layer_summary_rows = []
    for layer, grp in df.groupby("layer"):
        n_strong   = int(grp["strong_candidate_feature"].sum())
        n_partial  = int(grp["partial_candidate_feature"].sum())
        n_ordering = int(grp["ordering_T_gt_C_gt_B"].sum())
        layer_summary_rows.append({
            "layer":                  layer,
            "n_features_analysed":    int(grp["feature_id"].nunique()),
            "n_particle_tests":       len(grp),
            "n_strong_candidate":     n_strong,
            "n_partial_candidate":    n_partial,
            "n_ordering_T_C_B":       n_ordering,
            "mean_candidate_specificity": float(grp["candidate_specificity"].mean()),
            "mean_competitor_presence":   float(grp["competitor_presence"].mean()),
            "mean_IPR":               float(grp["internal_presence_ratio"].dropna().mean()),
        })
    layer_summary = pd.DataFrame(layer_summary_rows)

    # ── Summary by particle ───────────────────────────────────────────────────
    particle_summary_rows = []
    for particle, grp in df.groupby("particle"):
        n_strong  = int(grp["strong_candidate_feature"].sum())
        n_partial = int(grp["partial_candidate_feature"].sum())
        top5 = (
            grp[grp["strong_candidate_feature"]]
            .sort_values("competitor_presence", ascending=False)
            .head(5)[["feature_id", "layer", "candidate_specificity",
                       "competitor_presence", "internal_presence_ratio"]]
            .to_dict("records")
        )
        particle_summary_rows.append({
            "particle":                   particle,
            "n_strong_candidate_features": n_strong,
            "n_partial_candidate_features": n_partial,
            "mean_candidate_specificity":  float(grp["candidate_specificity"].mean()),
            "mean_competitor_presence":    float(grp["competitor_presence"].mean()),
            "mean_IPR":                    float(grp["internal_presence_ratio"].dropna().mean()),
            "top5_strong_features":        json.dumps(top5),
        })
    particle_summary = pd.DataFrame(particle_summary_rows)

    return df, layer_summary, particle_summary


# ─── Per-prompt candidate scores ─────────────────────────────────────────────

def compute_per_prompt_scores(
    df: pd.DataFrame,
    behaviour: str,
    split: str,
    prompts: list[dict],
) -> pd.DataFrame:
    """
    For each prompt, compute a candidate score per particle = mean activation of
    features identified as strong candidate features for that particle.

    Also computes the predicted_candidate (argmax score) and checks if it matches
    the correct_answer.
    """
    paths = get_paths(behaviour, split)

    # Build feature lookup per particle: {particle: {layer: [feat_idx, ...]}}
    cand_feats: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for _, row_r in df[df["strong_candidate_feature"]].iterrows():
        cand_feats[row_r["particle"]][row_r["layer"]].append(int(row_r["feature_idx"]))

    # Check we have any candidate features
    total_feats = sum(len(fidxs) for p in cand_feats.values() for fidxs in p.values())
    if total_feats == 0:
        print("[WARN] No strong candidate features found — using top candidate_specificity features")
        # Fall back: top-3 by candidate_specificity per particle
        for particle in PARTICLES:
            sub = df[df["particle"] == particle].sort_values("candidate_specificity", ascending=False).head(3)
            for _, r in sub.iterrows():
                cand_feats[particle][int(r["layer"])].append(int(r["feature_idx"]))

    score_rows = []
    for layer in LAYERS:
        result = load_layer_features(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            continue
        indices, values = result

        particle_scores: dict[str, np.ndarray] = {}
        for particle in PARTICLES:
            feat_list = cand_feats[particle].get(layer, [])
            if not feat_list:
                particle_scores[particle] = np.zeros(len(prompts), dtype=np.float32)
                continue
            acts = np.stack([
                get_activation_for_feature(indices, values, fidx) for fidx in feat_list
            ], axis=1)  # (N, n_feats)
            particle_scores[particle] = acts.mean(axis=1)

        # Accumulate: per-prompt, per-particle mean across layers
        for i, prompt in enumerate(prompts):
            row_r: dict = {
                "prompt_idx":      i,
                "layer":           layer,
                "correct_answer":  prompt["correct_answer"].strip(),
                "filter_property": prompt.get("filter_property", ""),
                "wording_family":  prompt.get("wording_family", ""),
                "candidate_pool":  ",".join(sorted(prompt["_pool_set"])),
            }
            best_particle = max(PARTICLES, key=lambda p: particle_scores[p][i])
            for p in PARTICLES:
                row_r[f"score_{p}"] = float(particle_scores[p][i])
            row_r["predicted_candidate"] = best_particle
            row_r["matches_correct"]     = (best_particle == prompt["_correct_stripped"])
            score_rows.append(row_r)

    return pd.DataFrame(score_rows)


# ─── Top candidate features JSON ─────────────────────────────────────────────

def build_top_features_json(df: pd.DataFrame) -> dict:
    result = {}
    for particle in PARTICLES:
        sub = df[df["particle"] == particle].copy()
        # Strong: T>C>B significant
        strong = sub[sub["strong_candidate_feature"]].sort_values(
            "competitor_presence", ascending=False
        )
        # Partial: T>>background only
        partial = sub[sub["partial_candidate_feature"]].sort_values(
            "candidate_specificity", ascending=False
        )
        result[particle] = {
            "strong_candidate_features": strong[
                ["feature_id", "layer", "candidate_specificity", "competitor_presence",
                 "internal_presence_ratio", "target_mean", "competitor_mean", "background_mean",
                 "cohen_d_TC", "cohen_d_CB", "mw_p_TC_adj", "mw_p_CB_adj"]
            ].head(20).to_dict("records"),
            "partial_candidate_features": partial[
                ["feature_id", "layer", "candidate_specificity",
                 "target_mean", "competitor_mean", "background_mean"]
            ].head(20).to_dict("records"),
        }
    return result


# ─── Report ──────────────────────────────────────────────────────────────────

def write_report(
    df: pd.DataFrame,
    layer_summary: pd.DataFrame,
    particle_summary: pd.DataFrame,
    prompts: list[dict],
    output_dir: Path,
) -> None:
    n_prompts = len(prompts)
    n_st = sum(1 for p in prompts if not p.get("multi_token_answer", False))

    lines = [
        "# Internal Candidate-State Analysis Report",
        f"## {BEHAVIOUR} | {SPLIT} split | {n_prompts} prompts",
        "",
        "## 1. Summary of findings",
        "",
    ]

    n_strong_total = int(df["strong_candidate_feature"].sum())
    n_partial_total = int(df["partial_candidate_feature"].sum())
    n_ordering_total = int(df["ordering_T_gt_C_gt_B"].sum())

    lines += [
        f"- Total (feature × particle) tests: {len(df)}",
        f"- Features with **T > C > B** ordering: {n_ordering_total} / {len(df)} "
        f"({n_ordering_total/len(df):.1%})",
        f"- **Strong candidate features** (T>C>B + both MW tests significant, FDR α=0.05): "
        f"{n_strong_total}",
        f"- **Partial candidate features** (T>>B only, C≈B): {n_partial_total}",
        "",
    ]

    # Determine overall evidence category
    if n_strong_total >= 10:
        evidence = "**STRONG** — multiple features show T > C > B with statistical significance"
    elif n_strong_total >= 3:
        evidence = "**MODERATE** — some features show internal candidate-state pattern"
    elif n_partial_total >= 5:
        evidence = "**PARTIAL** — features are candidate-specific but competitor ≈ background (output-selection only)"
    else:
        evidence = "**WEAK / NEGATIVE** — no clear internal candidate representation found"

    lines += [f"### Evidence classification: {evidence}", ""]

    # Per-particle summary
    lines += ["## 2. Per-particle summary", ""]
    for _, pr in particle_summary.iterrows():
        lines += [
            f"### {pr['particle']}",
            f"- Strong features: {int(pr['n_strong_candidate_features'])}  |  "
            f"Partial: {int(pr['n_partial_candidate_features'])}",
            f"- Mean candidate_specificity: {pr['mean_candidate_specificity']:.4f}",
            f"- Mean competitor_presence:   {pr['mean_competitor_presence']:.4f}",
            f"- Mean IPR (competitor/target): {pr['mean_IPR']:.4f}",
            "",
        ]

    # Per-layer summary
    lines += ["## 3. Per-layer summary (candidate feature emergence)", ""]
    lines += ["| Layer | n_features | n_strong | n_partial | mean_spec | mean_comp_presence | mean_IPR |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for _, lr in layer_summary.sort_values("layer").iterrows():
        lines.append(
            f"| L{int(lr['layer'])} | {int(lr['n_features_analysed'])} | "
            f"{int(lr['n_strong_candidate'])} | {int(lr['n_partial_candidate'])} | "
            f"{lr['mean_candidate_specificity']:.4f} | {lr['mean_competitor_presence']:.4f} | "
            f"{lr['mean_IPR']:.4f} |"
        )
    lines += [""]

    # Top strong features overall
    lines += ["## 4. Top strong candidate features (T > C > B, FDR-significant)", ""]
    strong = df[df["strong_candidate_feature"]].sort_values("competitor_presence", ascending=False)
    if len(strong):
        lines += ["| Feature | Particle | Layer | target_μ | competitor_μ | background_μ | IPR | cohend_TC | cohend_CB |"]
        lines += ["|---|---|---|---|---|---|---|---|---|"]
        for _, r in strong.head(30).iterrows():
            lines.append(
                f"| {r['feature_id']} | {r['particle']} | {r['layer']} | "
                f"{r['target_mean']:.3f} | {r['competitor_mean']:.3f} | "
                f"{r['background_mean']:.3f} | {r['internal_presence_ratio']:.3f} | "
                f"{r['cohen_d_TC']:.3f} | {r['cohen_d_CB']:.3f} |"
            )
    else:
        lines += ["No strong features found with FDR correction."]
        lines += [""]
        lines += ["### Top features by competitor_presence (no significance threshold):"]
        top = df.sort_values("competitor_presence", ascending=False).head(20)
        lines += ["| Feature | Particle | Layer | target_μ | competitor_μ | background_μ | IPR |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(
                f"| {r['feature_id']} | {r['particle']} | {r['layer']} | "
                f"{r['target_mean']:.3f} | {r['competitor_mean']:.3f} | "
                f"{r['background_mean']:.3f} | {r['internal_presence_ratio']:.3f} |"
            )
    lines += [""]

    # Key scientific questions
    lines += [
        "## 5. Key scientific questions",
        "",
        "### Q1: Which layers first show candidate presence?",
    ]
    early = layer_summary[layer_summary["layer"] <= 13].sort_values("n_strong_candidate", ascending=False)
    mid   = layer_summary[(layer_summary["layer"] >= 14) & (layer_summary["layer"] <= 18)].sort_values("n_strong_candidate", ascending=False)
    late  = layer_summary[layer_summary["layer"] >= 22].sort_values("n_strong_candidate", ascending=False)
    for label, subset in [("Early (L10-L13)", early), ("Mid (L14-L18)", mid), ("Late (L22-L25)", late)]:
        if len(subset):
            best = subset.iloc[0]
            lines.append(f"- {label}: best layer L{int(best['layer'])} with {int(best['n_strong_candidate'])} strong features, mean_IPR={best['mean_IPR']:.3f}")
    lines += [""]

    # Q2: Are L19/L20 retrieval layers?
    retrieval = layer_summary[layer_summary["layer"].isin([19, 20, 21])]
    lines += ["### Q2: Are L19-L21 candidate retrieval layers?"]
    for _, r in retrieval.sort_values("layer").iterrows():
        lines.append(
            f"- L{int(r['layer'])}: {int(r['n_strong_candidate'])} strong features, "
            f"mean_IPR={r['mean_IPR']:.3f}, mean_comp_presence={r['mean_competitor_presence']:.4f}"
        )
    lines += [""]

    # Q3: Are L24/L25 output-selection?
    output_layers = layer_summary[layer_summary["layer"].isin([24, 25])]
    lines += ["### Q3: Are L24/L25 output-selection (partial > strong)?"]
    for _, r in output_layers.sort_values("layer").iterrows():
        lines.append(
            f"- L{int(r['layer'])}: strong={int(r['n_strong_candidate'])}, "
            f"partial={int(r['n_partial_candidate'])}, mean_IPR={r['mean_IPR']:.3f}"
        )
    lines += [""]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",    default=BEHAVIOUR)
    ap.add_argument("--split",        default=SPLIT)
    ap.add_argument("--feature_set",  nargs="+",
                    default=["circuit_only", "all_graph"],
                    choices=["circuit_only", "all_graph", "top_active"],
                    help="Which features to analyse")
    ap.add_argument("--include_multi_token", action="store_true",
                    help="Include multi-token prompts (default: exclude)")
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split)

    # Pre-flight check
    if not paths["feature_dir"].exists():
        print(f"[ERROR] Feature directory not found: {paths['feature_dir']}")
        print(rsync_cmd(args.behaviour, args.split))
        sys.exit(1)

    # Check at least one layer has features
    has_any = any(
        (paths["feature_dir"] / f"layer_{l}" / f"{args.behaviour}_{args.split}_top_k_indices.npy").exists()
        for l in LAYERS
    )
    if not has_any:
        print(f"[ERROR] No transcoder feature files found for {args.behaviour}_{args.split}")
        print("Run this to sync from CSD3:")
        print(rsync_cmd(args.behaviour, args.split))
        sys.exit(1)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print(f"Behaviour: {args.behaviour}  Split: {args.split}")
    print(f"Feature sets: {args.feature_set}")

    # Main analysis
    df, layer_summary, particle_summary = run_analysis(
        args.behaviour, args.split,
        feature_sets=args.feature_set,
        single_token_only=not args.include_multi_token,
    )

    # Save tables
    out = paths["output_dir"]
    df.to_csv(out / "candidate_feature_table.csv", index=False)
    layer_summary.to_csv(out / "candidate_feature_summary_by_layer.csv", index=False)
    particle_summary.to_csv(out / "candidate_feature_summary_by_particle.csv", index=False)
    print(f"Saved feature table ({len(df)} rows) to {out}/candidate_feature_table.csv")

    # Top candidate features JSON
    top_json = build_top_features_json(df)
    with open(out / "top_candidate_features.json", "w") as f:
        json.dump(top_json, f, indent=2)
    print(f"Saved top candidate features to {out}/top_candidate_features.json")

    # Per-prompt scores
    prompts = load_prompts(paths)
    if not args.include_multi_token:
        prompts = [p for p in prompts if not p.get("multi_token_answer", False)]
    per_prompt = compute_per_prompt_scores(df, args.behaviour, args.split, prompts)
    per_prompt.to_csv(out / "per_prompt_candidate_scores.csv", index=False)
    print(f"Saved per-prompt scores ({len(per_prompt)} rows) to {out}/per_prompt_candidate_scores.csv")

    # Report
    write_report(df, layer_summary, particle_summary, prompts, out)

    # Quick console summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nStrong T>C>B features (FDR α=0.05):")
    strong = df[df["strong_candidate_feature"]]
    if len(strong):
        print(strong.groupby(["particle", "layer"])["feature_id"].count().to_string())
    else:
        print("  NONE found — check per_prompt scores and partial features")
    print(f"\nTop 10 features by competitor_presence:")
    print(
        df[["feature_id", "particle", "layer", "target_mean", "competitor_mean",
            "background_mean", "internal_presence_ratio", "ordering_T_gt_C_gt_B"]]
        .sort_values("competitor_presence", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
