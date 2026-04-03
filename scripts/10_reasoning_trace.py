#!/usr/bin/env python3
"""
scripts/10_reasoning_trace.py
Reasoning-trace engine for multilingual_circuits_b1.

Reconstructs per-prompt decision processes as competing causal trajectories
evolving across layers. CPU-only; reads from existing circuit artifacts.

Inputs:
  data/results/causal_edges/{behaviour}/circuits_{behaviour}_{split}.json
  data/results/causal_edges/{behaviour}/causal_edges_{behaviour}_{split}.json
  data/ui_offline/{run_id}/interventions.csv
  data/prompts/{behaviour}_{split}.jsonl

Outputs (in data/results/reasoning_traces/{behaviour}/):
  reasoning_traces_{split}.jsonl          -- per-prompt structured trace + narrative
  prompt_features_{split}.parquet         -- per-prompt per-feature contributions
  prompt_paths_{split}.parquet            -- per-prompt per-path scored paths
  layerwise_decision_trace_{split}.parquet -- per-prompt per-layer cumulative Δlogit
  error_cases_{split}.json                -- incorrect-prompt error analysis
  figures/layerwise_delta_logit.png
  figures/top_paths.png
  figures/error_analysis.png

Data coverage notes:
  - ablation_zero: 10/27 circuit features × 96 prompts (primary; full coverage)
  - patching:      27/27 circuit features × 48 prompts (supplementary; cross-lang semantics)
  - Layerwise dynamics computed from ablation_zero features only (conservative).
  - Feature table includes all 27 features; patching-only rows flagged with data_source="patching".
  - Sign convention: contribution_to_correct = -effect_size
    (effect_size = intervened_logit_diff - baseline_logit_diff; ablation reduces feature → negative
     effect_size means feature was helping the correct prediction → positive contribution)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Zone definitions ─────────────────────────────────────────────────────────

LAYER_ZONE = {
    10: "early", 11: "early", 12: "early", 13: "early",
    14: "early", 15: "early", 16: "early",
    17: "mid",   18: "mid",   19: "mid",   20: "mid",
    21: "mid",   22: "mid",
    23: "late",  24: "late",  25: "late",
}

ZONE_LAYER_RANGES = {
    "early": (9.5, 16.5),
    "mid":   (16.5, 22.5),
    "late":  (22.5, 25.5),
}

ZONE_COLORS = {"early": "#d4e6f1", "mid": "#d5f5e3", "late": "#fde8d8"}

IO_NODES = {"input", "output_correct", "output_incorrect"}


def get_layer(feature_id: str) -> int:
    return int(feature_id.split("_")[0][1:])


def get_zone(feature_id: str) -> str:
    return LAYER_ZONE.get(get_layer(feature_id), "unknown")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Reasoning trace engine for multilingual circuits")
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--run_id", default=None,
                   help="UI offline run ID (auto-detected from data/ui_offline/)")
    p.add_argument("--supplement_csv", default=None,
                   help="Path to ablation_supplement_{split}.csv from script 11. "
                        "Adds ablation_zero coverage for late-hub circuit features.")
    p.add_argument("--top_k_paths", type=int, default=5,
                   help="Top paths per prompt to include in trace (default: 5)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: data/results/reasoning_traces/{behaviour})")
    p.add_argument("--no_figures", action="store_true",
                   help="Skip matplotlib figure generation")
    p.add_argument("--circuits_json", default=None,
                   help="Override default circuits JSON path "
                        "(default: data/results/causal_edges/<behaviour>/circuits_<behaviour>_<split>.json). "
                        "Useful when running sparsified graph variants.")
    return p.parse_args()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_circuit(behaviour: str, split: str) -> dict:
    path = Path(f"data/results/causal_edges/{behaviour}/circuits_{behaviour}_{split}.json")
    if not path.exists():
        raise FileNotFoundError(f"Circuit JSON not found: {path}\n  Run scripts/08_causal_edges.py first.")
    return json.loads(path.read_text())


def load_causal_edges(behaviour: str, split: str) -> list:
    path = Path(f"data/results/causal_edges/{behaviour}/causal_edges_{behaviour}_{split}.json")
    if not path.exists():
        raise FileNotFoundError(f"Causal edges JSON not found: {path}")
    data = json.loads(path.read_text())
    return data.get("causal_edges", data)


def load_interventions(run_id: str, supplement_csv: str = None) -> pd.DataFrame:
    """
    Load interventions CSV.  If supplement_csv is provided, merge its ablation_zero
    rows on top of the base CSV, so that script 11 late-hub measurements are
    recognised as 'ablation_zero' by build_feature_contributions().

    Merge priority: supplement ablation_zero > original ablation_zero > patching.
    (In practice they don't overlap — supplement covers different features.)
    """
    path = Path(f"data/ui_offline/{run_id}/interventions.csv")
    if not path.exists():
        raise FileNotFoundError(f"Interventions CSV not found: {path}")
    df_base = pd.read_csv(path)

    if supplement_csv is None:
        return df_base

    supp_path = Path(supplement_csv)
    if not supp_path.exists():
        raise FileNotFoundError(
            f"Supplement CSV not found: {supp_path}\n"
            f"  Run scripts/11_ablation_supplement.py first."
        )
    df_supp = pd.read_csv(supp_path)
    # Keep only ablation_zero rows from supplement (should be all of them)
    df_supp = df_supp[df_supp["experiment_type"] == "ablation_zero"].copy()
    print(f"  Supplement: {len(df_supp)} ablation_zero rows "
          f"({df_supp['feature_id'].nunique()} features × "
          f"{df_supp['prompt_idx'].nunique()} prompts)")

    # Concat — supplement rows will be picked up by build_feature_contributions
    # because they have experiment_type='ablation_zero' and the feature_ids that
    # previously fell through to the patching branch.
    df_combined = pd.concat([df_base, df_supp], ignore_index=True)
    return df_combined


def load_prompts(behaviour: str, split: str) -> list:
    path = Path(f"data/prompts/{behaviour}_{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompts JSONL not found: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def find_run_id(behaviour: str, split: str) -> str:
    ui_dir = Path("data/ui_offline")
    # Pattern: YYYYMMDD-HHMMSS_{behaviour}_{split}_n{N}
    matches = sorted(ui_dir.glob(f"*_{behaviour}_{split}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No UI offline run for {behaviour}/{split} in {ui_dir}.\n"
            f"  Run scripts/09_ui_prep.py first."
        )
    run_id = matches[-1].name
    print(f"  Auto-detected run_id: {run_id}")
    return run_id


# ─── Step 1: per-prompt feature contributions ─────────────────────────────────

def build_feature_contributions(
    df: pd.DataFrame,
    circuit_features: list,
    n_prompts: int,
) -> pd.DataFrame:
    """
    Build per-prompt per-feature contribution table.

    Priority:
      1. ablation_zero: full 96-prompt coverage, direct single-feature ablation.
      2. patching: 48-prompt coverage (cross-lang injection), supplementary.
      3. missing: feature not measured for this prompt → contribution=NaN.

    contribution_to_correct = -effect_size
    (ablating a helpful feature → effect_size < 0 → contribution_to_correct > 0)
    """
    az = df[df["experiment_type"] == "ablation_zero"]
    pat = df[df["experiment_type"] == "patching"]
    az_features = set(az["feature_id"].unique())

    records = []
    for fid in circuit_features:
        layer = get_layer(fid)
        zone = get_zone(fid)

        if fid in az_features:
            # Full 96-prompt ablation_zero coverage
            sub = az[az["feature_id"] == fid][
                ["prompt_idx", "effect_size", "baseline_logit_diff"]
            ].copy()
            sub["data_source"] = "ablation_zero"
        else:
            # Patching only (48 prompts); remaining prompts get NaN
            pat_sub = pat[pat["feature_id"] == fid][
                ["prompt_idx", "effect_size", "baseline_logit_diff"]
            ].copy()
            if pat_sub.empty:
                # No data at all
                sub = pd.DataFrame({
                    "prompt_idx": range(n_prompts),
                    "effect_size": np.nan,
                    "baseline_logit_diff": np.nan,
                    "data_source": "none",
                })
            else:
                pat_sub["data_source"] = "patching"
                # Add missing prompts with NaN
                covered = set(pat_sub["prompt_idx"].tolist())
                missing_rows = pd.DataFrame({
                    "prompt_idx": [i for i in range(n_prompts) if i not in covered],
                    "effect_size": np.nan,
                    "baseline_logit_diff": np.nan,
                    "data_source": "missing",
                })
                sub = pd.concat([pat_sub, missing_rows], ignore_index=True)

        sub = sub.copy()
        sub["feature_id"] = fid
        sub["layer"] = layer
        sub["zone"] = zone
        sub["contribution_to_correct"] = -sub["effect_size"]
        records.append(
            sub[["prompt_idx", "feature_id", "layer", "zone",
                 "contribution_to_correct", "effect_size", "data_source"]]
        )

    return pd.concat(records, ignore_index=True).sort_values(
        ["prompt_idx", "layer"], ignore_index=True
    )


# ─── Step 2: layer-wise decision dynamics ─────────────────────────────────────

def build_layerwise_trace(
    feat_df: pd.DataFrame,
    n_prompts: int,
    baseline_by_prompt: dict,
) -> pd.DataFrame:
    """
    Per-prompt per-layer cumulative Δlogit from ablation_zero features only.

    layer_delta = sum of contribution_to_correct for features in this layer.
    cumulative_delta = running sum across layers (L10 → L25).
    flip_layer = first layer where (baseline + cumulative) changes sign vs baseline.
    """
    # Use only ablation_zero for reliable full-coverage dynamics
    az_df = feat_df[feat_df["data_source"] == "ablation_zero"].copy()
    layers = list(range(10, 26))  # All circuit layers; 0-contribution for layers with no ablation_zero features

    records = []
    flip_records = []

    for prompt_idx in range(n_prompts):
        p = az_df[az_df["prompt_idx"] == prompt_idx]
        baseline = baseline_by_prompt.get(prompt_idx, 0.0)
        correct_baseline = baseline > 0
        cumulative = 0.0
        flip_layer = None

        for layer in layers:
            layer_delta = p[p["layer"] == layer]["contribution_to_correct"].sum()
            cumulative += layer_delta
            projected = baseline + cumulative

            records.append({
                "prompt_idx": prompt_idx,
                "layer": layer,
                "zone": LAYER_ZONE.get(layer, "unknown"),
                "layer_delta": float(layer_delta),
                "cumulative_delta": float(cumulative),
                "projected_logit_diff": float(projected),
                "baseline_logit_diff": float(baseline),
            })

            # Detect first flip
            if flip_layer is None and (projected > 0) != correct_baseline:
                flip_layer = layer

        flip_records.append({"prompt_idx": prompt_idx, "flip_layer": flip_layer})

    df = pd.DataFrame(records)
    flip_df = pd.DataFrame(flip_records)
    df = df.merge(flip_df, on="prompt_idx", how="left")
    return df


# ─── Step 3: per-prompt path scoring ─────────────────────────────────────────

def score_paths_per_prompt(
    paths: list,
    feat_df: pd.DataFrame,
    n_prompts: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Score each global circuit path for each prompt by weighting by per-prompt
    feature contributions along the path.

    prompt_score = global_score × sigmoid(mean_feature_contrib)
      where mean_feature_contrib = mean contribution_to_correct over path features
      (using ablation_zero when available; patching if available; else 0).

    sign_agreement = fraction of path features with contribution_to_correct > 0
    path_direction = "correct" | "incorrect" based on terminal node.
    """
    # Build contribution lookup: (prompt_idx, feature_id) -> contribution_to_correct
    # Use ablation_zero preferentially, then patching, else NaN
    az_contrib = feat_df[feat_df["data_source"] == "ablation_zero"].set_index(
        ["prompt_idx", "feature_id"]
    )["contribution_to_correct"]
    all_contrib = feat_df[feat_df["data_source"].isin(["ablation_zero", "patching"])].set_index(
        ["prompt_idx", "feature_id"]
    )["contribution_to_correct"]

    records = []
    for prompt_idx in range(n_prompts):
        for path_dict in paths:
            path_nodes = path_dict["path"]
            global_score = path_dict["score"]
            rank = path_dict["rank"]

            path_features = [n for n in path_nodes if n not in IO_NODES]
            terminal = path_nodes[-1] if path_nodes else "unknown"
            path_direction = (
                "correct" if terminal == "output_correct"
                else "incorrect" if terminal == "output_incorrect"
                else "other"
            )

            # Collect per-prompt contributions for features in this path
            contribs = []
            for fid in path_features:
                key = (prompt_idx, fid)
                c = all_contrib.get(key, np.nan)
                if not np.isnan(c):
                    contribs.append(float(c))

            if contribs:
                mean_contrib = float(np.mean(contribs))
                sign_agreement = float(np.mean([1.0 if c > 0 else 0.0 for c in contribs]))
            else:
                mean_contrib = 0.0
                sign_agreement = 0.5  # uninformative

            # Sigmoid-weighted score: modulates global score by per-prompt alignment
            # sigmoid(0) = 0.5 → neutral; sigmoid(+large) → 1 → full global score
            sigmoid_weight = 1.0 / (1.0 + np.exp(-mean_contrib))
            prompt_score = float(global_score * sigmoid_weight * 2)  # ×2 to center at global_score

            records.append({
                "prompt_idx": prompt_idx,
                "path_rank": int(rank),
                "path_str": " → ".join(path_nodes),
                "global_score": float(global_score),
                "n_edges": int(path_dict["n_edges"]),
                "terminal": terminal,
                "path_direction": path_direction,
                "n_path_features": len(path_features),
                "mean_feature_contrib": mean_contrib,
                "sign_agreement": sign_agreement,
                "prompt_score": prompt_score,
            })

    df = pd.DataFrame(records)
    df["prompt_path_rank"] = (
        df.groupby("prompt_idx")["prompt_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return df.sort_values(["prompt_idx", "prompt_path_rank"], ignore_index=True)


# ─── Step 4: competing trajectories ──────────────────────────────────────────

def compute_trajectories(feat_df: pd.DataFrame, n_prompts: int) -> pd.DataFrame:
    """
    Per-prompt: split features into correct-supporting vs incorrect-supporting trajectories.
    Uses ablation_zero features only (semantically consistent: single-feature ablation).
    Patching has cross-lang semantics and would give misleading trajectory signals.
    """
    records = []
    for prompt_idx in range(n_prompts):
        p = feat_df[
            (feat_df["prompt_idx"] == prompt_idx) &
            (feat_df["data_source"] == "ablation_zero")
        ].dropna(subset=["contribution_to_correct"])

        correct_sup = p[p["contribution_to_correct"] > 0]
        incorrect_sup = p[p["contribution_to_correct"] < 0]
        neutral = p[p["contribution_to_correct"] == 0]

        correct_strength = float(correct_sup["contribution_to_correct"].sum())
        incorrect_strength = float(incorrect_sup["contribution_to_correct"].sum())
        net = correct_strength + incorrect_strength

        records.append({
            "prompt_idx": prompt_idx,
            "n_correct_features": int(len(correct_sup)),
            "n_incorrect_features": int(len(incorrect_sup)),
            "n_neutral_features": int(len(neutral)),
            "correct_trajectory_strength": correct_strength,
            "incorrect_trajectory_strength": incorrect_strength,
            "net_trajectory": net,
            "dominant_trajectory": "correct" if net >= 0 else "incorrect",
        })

    return pd.DataFrame(records)


# ─── Step 5: reasoning trace construction ────────────────────────────────────

def build_trace_for_prompt(
    prompt_idx: int,
    prompt_meta: dict,
    baseline_logit_diff: float,
    feat_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    top_k: int,
    n_measured_features: int,
    n_circuit_features: int,
) -> dict:
    """Build the full reasoning trace for one prompt."""
    p_feat = feat_df[feat_df["prompt_idx"] == prompt_idx]
    p_layer = layer_df[layer_df["prompt_idx"] == prompt_idx].sort_values("layer")
    p_paths = paths_df[paths_df["prompt_idx"] == prompt_idx].sort_values("prompt_path_rank")
    p_traj = traj_df[traj_df["prompt_idx"] == prompt_idx].iloc[0]

    correct = baseline_logit_diff > 0

    # Zone summaries — measured = ablation_zero only (reliable)
    zone_summary = {}
    for zone in ["early", "mid", "late"]:
        z = p_feat[p_feat["zone"] == zone]
        z_az = z[z["data_source"] == "ablation_zero"]
        z_all = z.dropna(subset=["contribution_to_correct"])
        zone_summary[zone] = {
            "n_features": int(len(z)),
            "n_measured_ablation": int(len(z_az)),
            "n_measured_patching": int(len(z[z["data_source"] == "patching"])),
            "measured_contribution": float(z_az["contribution_to_correct"].sum()),
            "total_measured_contribution": float(z_all["contribution_to_correct"].sum()),
        }

    # Top features by contribution magnitude
    p_feat_valid = p_feat.dropna(subset=["contribution_to_correct"])
    top_correct_feats = (
        p_feat_valid.nlargest(3, "contribution_to_correct")
        [["feature_id", "layer", "zone", "contribution_to_correct", "data_source"]]
        .to_dict("records")
    )
    top_incorrect_feats = (
        p_feat_valid.nsmallest(3, "contribution_to_correct")
        [["feature_id", "layer", "zone", "contribution_to_correct", "data_source"]]
        .to_dict("records")
    )

    # Flip layer from layerwise trace
    flip_layer = None
    if len(p_layer) > 0 and not p_layer["flip_layer"].isna().all():
        val = p_layer["flip_layer"].iloc[0]
        if not (isinstance(val, float) and np.isnan(val)):
            flip_layer = int(val)

    # Top paths
    top_paths = (
        p_paths.head(top_k)
        [["path_rank", "path_str", "global_score", "prompt_score",
          "path_direction", "mean_feature_contrib", "sign_agreement"]]
        .to_dict("records")
    )

    # Trajectory
    traj_info = {
        "n_correct_features": int(p_traj["n_correct_features"]),
        "n_incorrect_features": int(p_traj["n_incorrect_features"]),
        "correct_strength": float(p_traj["correct_trajectory_strength"]),
        "incorrect_strength": float(p_traj["incorrect_trajectory_strength"]),
        "net": float(p_traj["net_trajectory"]),
        "dominant": str(p_traj["dominant_trajectory"]),
    }

    narrative = _build_narrative(
        prompt_meta, baseline_logit_diff, zone_summary, traj_info, flip_layer, correct
    )

    return {
        "prompt_idx": prompt_idx,
        "prompt": prompt_meta["prompt"],
        "language": prompt_meta.get("language", "?"),
        "concept_index": int(prompt_meta.get("concept_index", -1)),
        "template_idx": int(prompt_meta.get("template_idx", -1)),
        "correct_answer": prompt_meta.get("correct_answer", ""),
        "incorrect_answer": prompt_meta.get("incorrect_answer", ""),
        "baseline_logit_diff": float(baseline_logit_diff),
        "prediction_correct": bool(correct),
        "data_completeness": {
            "ablation_measured_features": n_measured_features,
            "circuit_features": n_circuit_features,
            "ablation_coverage_fraction": round(n_measured_features / n_circuit_features, 3),
        },
        "zone_summary": zone_summary,
        "trajectories": traj_info,
        "top_correct_features": top_correct_feats,
        "top_incorrect_features": top_incorrect_feats,
        "flip_layer": flip_layer,
        "top_paths": top_paths,
        "narrative": narrative,
    }


def _build_narrative(prompt_meta, baseline_logit_diff, zone_summary, traj, flip_layer, correct):
    lang = prompt_meta.get("language", "?").upper()
    outcome = "CORRECT" if correct else "INCORRECT"

    early_c = zone_summary["early"]["measured_contribution"]
    mid_c = zone_summary["mid"]["measured_contribution"]
    late_total = zone_summary["late"]["total_measured_contribution"]

    parts = [
        (f"[{lang} concept={prompt_meta.get('concept_index')} "
         f"tmpl={prompt_meta.get('template_idx')}]"),
        f"Prompt: '{prompt_meta['prompt']}' → {outcome} (logit_diff={baseline_logit_diff:+.3f})",
        f"Early(L10-L16): contrib={early_c:+.3f} [ablation]",
        f"Mid(L18-L22): contrib={mid_c:+.3f} [ablation]",
        f"Late(L23-L25): contrib={late_total:+.3f} [patching+ablation]",
        (f"Trajectories(ablation): dominant={traj['dominant']}, net={traj['net']:+.3f} "
         f"({traj['n_correct_features']} correct-supporting, "
         f"{traj['n_incorrect_features']} incorrect-supporting)"),
    ]
    if flip_layer is not None:
        parts.append(f"Projection flip detected at L{flip_layer}")
    return " | ".join(parts)


# ─── Step 6: error analysis ────────────────────────────────────────────────────

def build_error_analysis(
    traces: list,
    feat_df: pd.DataFrame,
) -> dict:
    """Systematic analysis of incorrect predictions vs correct ones."""
    incorrect = [t for t in traces if not t["prediction_correct"]]
    correct_tr = [t for t in traces if t["prediction_correct"]]

    def mean_safe(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else None

    def zone_measured_mean(trace_list, zone):
        return mean_safe([t["zone_summary"][zone]["measured_contribution"] for t in trace_list])

    # Zone comparison
    zone_comparison = {}
    for zone in ["early", "mid", "late"]:
        zone_comparison[zone] = {
            "correct_mean": zone_measured_mean(correct_tr, zone),
            "incorrect_mean": zone_measured_mean(incorrect, zone),
        }

    # Concept-level breakdown
    concept_errors = defaultdict(list)
    for t in incorrect:
        concept_errors[t["concept_index"]].append(t["prompt_idx"])

    # Feature-level comparison (ablation_zero only for reliability)
    az_df = feat_df[feat_df["data_source"] == "ablation_zero"]
    incorrect_idx = set(t["prompt_idx"] for t in incorrect)
    correct_idx = set(t["prompt_idx"] for t in correct_tr)

    feat_comp = []
    for fid in az_df["feature_id"].unique():
        f_sub = az_df[az_df["feature_id"] == fid]
        inc_sub = f_sub[f_sub["prompt_idx"].isin(incorrect_idx)]["contribution_to_correct"]
        cor_sub = f_sub[f_sub["prompt_idx"].isin(correct_idx)]["contribution_to_correct"]
        feat_comp.append({
            "feature_id": fid,
            "zone": get_zone(fid),
            "correct_mean_contrib": float(cor_sub.mean()) if len(cor_sub) > 0 else None,
            "incorrect_mean_contrib": float(inc_sub.mean()) if len(inc_sub) > 0 else None,
            "delta": float(inc_sub.mean() - cor_sub.mean()) if len(inc_sub) > 0 and len(cor_sub) > 0 else None,
        })
    feat_comp_sorted = sorted(
        [f for f in feat_comp if f["delta"] is not None],
        key=lambda x: abs(x["delta"]),
        reverse=True,
    )

    # Flip layer distribution
    inc_flips = [t["flip_layer"] for t in incorrect if t["flip_layer"] is not None]
    cor_flips = [t["flip_layer"] for t in correct_tr if t["flip_layer"] is not None]

    # Dominant trajectory accuracy
    correct_dominant = sum(
        1 for t in traces if t["trajectories"]["dominant"] == ("correct" if t["prediction_correct"] else "incorrect")
    )

    return {
        "n_incorrect": len(incorrect),
        "n_correct": len(correct_tr),
        "accuracy": round(len(correct_tr) / len(traces), 4),
        "incorrect_prompt_indices": [t["prompt_idx"] for t in incorrect],
        "zone_comparison": zone_comparison,
        "concept_errors": {str(k): v for k, v in sorted(concept_errors.items())},
        "feature_comparison_ablation_only": feat_comp_sorted,
        "flip_layer_distribution": {
            "incorrect_prompts": {"values": inc_flips, "mean": mean_safe(inc_flips)},
            "correct_prompts": {"values": cor_flips, "mean": mean_safe(cor_flips)},
        },
        "trajectory_prediction_accuracy": round(correct_dominant / len(traces), 4),
        "error_patterns": _identify_error_patterns(incorrect, zone_comparison),
    }


def _identify_error_patterns(incorrect: list, zone_comparison: dict) -> list:
    if not incorrect:
        return ["No incorrect predictions."]

    patterns = []

    # Language concentration
    lang_counts = Counter(t["language"] for t in incorrect)
    if lang_counts.get("fr", 0) == len(incorrect):
        patterns.append("All errors are French prompts — EN performance is perfect.")
    elif lang_counts.get("en", 0) == len(incorrect):
        patterns.append("All errors are English prompts.")

    # Concept concentration
    concept_counts = Counter(t["concept_index"] for t in incorrect)
    top_concept, top_count = concept_counts.most_common(1)[0]
    if top_count / len(incorrect) > 0.35:
        patterns.append(
            f"Errors concentrated on concept {top_concept} "
            f"({top_count}/{len(incorrect)} = {100*top_count/len(incorrect):.0f}%)"
        )

    # Zone contribution differences
    for zone in ["early", "mid"]:
        cor = zone_comparison[zone]["correct_mean"]
        inc = zone_comparison[zone]["incorrect_mean"]
        if cor is not None and inc is not None:
            diff = inc - cor
            if abs(diff) > 0.05:
                direction = "lower" if diff < 0 else "higher"
                patterns.append(
                    f"{zone.capitalize()}-zone ablation contribution is {direction} "
                    f"for incorrect prompts ({inc:+.3f} vs {cor:+.3f}, Δ={diff:+.3f})"
                )

    # Trajectory accuracy
    traj_errors = sum(1 for t in incorrect if t["trajectories"]["dominant"] == "correct")
    if traj_errors > 0:
        patterns.append(
            f"{traj_errors}/{len(incorrect)} incorrect prompts have dominant trajectory = 'correct' "
            f"(late-layer effect not captured by ablation features)"
        )

    return patterns if patterns else ["No strong systematic error patterns identified."]


# ─── Coverage diagnostics ─────────────────────────────────────────────────────

def compute_coverage_report(
    feat_df: pd.DataFrame,
    circuit_features: list,
    supplement_csv: str = None,
) -> dict:
    """
    Report how many circuit features now have ablation_zero measurements,
    and which key late-hub features are covered.  Emitted both to stdout and
    returned as a dict for inclusion in error_cases JSON.
    """
    az_df = feat_df[feat_df["data_source"] == "ablation_zero"]
    az_covered = set(az_df["feature_id"].unique()) & set(circuit_features)
    pat_only = set(f for f in circuit_features if f not in az_covered)

    # Which came from supplement vs original (supplement rows have data_source='ablation_zero'
    # but were loaded from a separate file — we can't distinguish without a marker column,
    # so we just report totals)
    key_late = [
        "L23_F64429", "L24_F136810", "L24_F134204", "L24_F29680",
        "L24_F131457", "L22_F108295", "L25_F138698",
    ]

    print("\n" + "─" * 50)
    print("COVERAGE REPORT")
    print("─" * 50)
    print(f"Circuit features:       {len(circuit_features)}")
    print(f"Ablation-zero covered:  {len(az_covered)}/{len(circuit_features)}"
          + (" [FULL COVERAGE]" if len(az_covered) == len(circuit_features) else ""))
    print(f"Patching-only:          {len(pat_only)}")
    if supplement_csv:
        print(f"Supplement source:      {supplement_csv}")
    print("\nKey late-hub features:")
    for fid in key_late:
        if fid in az_covered:
            print(f"  {fid}: ✓ ablation_zero")
        elif fid in pat_only:
            print(f"  {fid}: ~ patching-only (approximate)")
        elif fid in set(circuit_features):
            print(f"  {fid}: ✗ missing")
        else:
            print(f"  {fid}: not in circuit")
    print("─" * 50)

    return {
        "n_circuit_features": len(circuit_features),
        "n_ablation_covered": len(az_covered),
        "n_patching_only": len(pat_only),
        "ablation_covered_features": sorted(az_covered),
        "patching_only_features": sorted(pat_only),
        "key_late_hub_coverage": {
            fid: ("ablation_zero" if fid in az_covered
                  else "patching_only" if fid in pat_only else "missing")
            for fid in key_late
        },
        "supplement_used": supplement_csv is not None,
    }


def classify_failure_types(traces: list, feat_df: pd.DataFrame) -> dict:
    """
    Classify incorrect predictions into:
      Type A: measured trajectory dominant='correct' but prediction wrong.
              (Circuit measurement says 'should work' but doesn't → late-layer override)
      Type B: measured trajectory dominant='incorrect'.
              (Circuit measurement agrees with prediction failure → early/mid breakdown)

    Returns counts and prompt indices for each type, plus reclassification stats
    (useful for comparing before/after supplement).
    """
    incorrect = [t for t in traces if not t["prediction_correct"]]
    if not incorrect:
        return {"n_incorrect": 0, "type_A": [], "type_B": []}

    type_A = [t for t in incorrect if t["trajectories"]["dominant"] == "correct"]
    type_B = [t for t in incorrect if t["trajectories"]["dominant"] == "incorrect"]

    # Per-feature contribution detail for Type A (the interesting failures)
    type_A_detail = []
    az_df = feat_df[feat_df["data_source"] == "ablation_zero"]
    for t in type_A:
        p_az = az_df[az_df["prompt_idx"] == t["prompt_idx"]]
        neg_feats = p_az[p_az["contribution_to_correct"] < 0]["feature_id"].tolist()
        top_neg = (
            p_az.nsmallest(3, "contribution_to_correct")
            [["feature_id", "contribution_to_correct"]]
            .to_dict("records")
        )
        type_A_detail.append({
            "prompt_idx": t["prompt_idx"],
            "language": t["language"],
            "concept_index": t["concept_index"],
            "template_idx": t["template_idx"],
            "baseline_logit_diff": t["baseline_logit_diff"],
            "net_trajectory": t["trajectories"]["net"],
            "n_neg_features": len(neg_feats),
            "top_neg_features": top_neg,
        })

    print("\n" + "─" * 50)
    print("FAILURE TYPE CLASSIFICATION")
    print("─" * 50)
    print(f"Total incorrect:  {len(incorrect)}")
    print(f"  Type A (trajectory=correct, outcome wrong): {len(type_A)}  "
          f"→ late-hub override")
    print(f"  Type B (trajectory=incorrect):              {len(type_B)}  "
          f"→ early/mid breakdown")
    if type_A:
        print(f"\nType A prompts (late-hub likely responsible):")
        for d in type_A_detail:
            print(f"  p{d['prompt_idx']:>3} [{d['language'].upper()} c{d['concept_index']} t{d['template_idx']}]"
                  f"  bl={d['baseline_logit_diff']:+.3f}  net_traj={d['net_trajectory']:+.3f}")
    print("─" * 50)

    return {
        "n_incorrect": len(incorrect),
        "type_A_count": len(type_A),
        "type_B_count": len(type_B),
        "type_A_prompt_indices": [t["prompt_idx"] for t in type_A],
        "type_B_prompt_indices": [t["prompt_idx"] for t in type_B],
        "type_A_detail": type_A_detail,
    }


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_layerwise_delta(layer_df: pd.DataFrame, out_dir: Path, n_prompts: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    agg = layer_df.groupby("layer")["layer_delta"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(11, 4))

    # Zone shading
    for zone, (lo, hi) in ZONE_LAYER_RANGES.items():
        ax.axvspan(lo, hi, alpha=0.15, color=ZONE_COLORS[zone], label=f"{zone} zone")

    ax.bar(agg["layer"], agg["mean"], color="steelblue", alpha=0.8, label="mean δ")
    ax.errorbar(agg["layer"], agg["mean"], yerr=agg["std"],
                fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer Δlogit (contribution to correct)")
    ax.set_title(
        f"Layerwise Decision Dynamics — {n_prompts} prompts\n"
        f"(ablation_zero features only; error bars = ±1 SD)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks(sorted(layer_df["layer"].unique()))

    plt.tight_layout()
    out = fig_dir / "layerwise_delta_logit.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure: {out}")


def plot_top_paths(paths_df: pd.DataFrame, circuit_paths: list, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    top = (
        paths_df.groupby(["path_rank", "path_str", "path_direction"])["prompt_score"]
        .mean()
        .reset_index()
        .sort_values("prompt_score", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ecc71" if d == "correct" else "#e74c3c" for d in top["path_direction"]]
    labels = [
        f"#{int(r)}: {s[:70]}…" if len(s) > 70 else f"#{int(r)}: {s}"
        for r, s in zip(top["path_rank"], top["path_str"])
    ]
    ax.barh(range(len(top)), top["prompt_score"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean prompt-adjusted score")
    ax.set_title("Top 10 Circuit Paths — mean prompt-adjusted score across all prompts")
    ax.invert_yaxis()
    ax.legend(handles=[
        Patch(color="#2ecc71", label="→ output_correct"),
        Patch(color="#e74c3c", label="→ output_incorrect"),
    ])

    plt.tight_layout()
    out = fig_dir / "top_paths.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure: {out}")


def plot_error_analysis(traces: list, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    correct_tr = [t for t in traces if t["prediction_correct"]]
    incorrect = [t for t in traces if not t["prediction_correct"]]

    if not incorrect:
        print("  No incorrect predictions — skipping error analysis figure.")
        return

    zones = ["early", "mid", "late"]
    cor_means = [
        np.mean([t["zone_summary"][z]["measured_contribution"] for t in correct_tr])
        for z in zones
    ]
    inc_means = [
        np.mean([t["zone_summary"][z]["measured_contribution"] for t in incorrect])
        for z in zones
    ]

    x = np.arange(len(zones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, cor_means, width,
           label=f"Correct (n={len(correct_tr)})", color="#2ecc71", alpha=0.8)
    ax.bar(x + width / 2, inc_means, width,
           label=f"Incorrect (n={len(incorrect)})", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{z.capitalize()} zone" for z in zones])
    ax.set_ylabel("Mean measured contribution to correct\n(ablation_zero features only)")
    ax.set_title("Zone Contributions: Correct vs Incorrect Predictions")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend()

    plt.tight_layout()
    out = fig_dir / "error_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure: {out}")


# ─── Save helpers ─────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
        print(f"  Saved: {path} ({len(df)} rows × {len(df.columns)} cols)")
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"  Parquet failed ({e}); saved CSV: {csv_path} ({len(df)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"Reasoning Trace Engine — {args.behaviour}/{args.split}")
    print("=" * 60)

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"data/results/reasoning_traces/{args.behaviour}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\nLoading inputs...")
    if args.circuits_json:
        _cpath = Path(args.circuits_json)
        if not _cpath.exists():
            raise FileNotFoundError(f"circuits_json not found: {_cpath}")
        circuit = json.loads(_cpath.read_text())
    else:
        circuit = load_circuit(args.behaviour, args.split)
    _causal_edges = load_causal_edges(args.behaviour, args.split)  # loaded for future use

    run_id = args.run_id or find_run_id(args.behaviour, args.split)
    df_interventions = load_interventions(run_id, supplement_csv=args.supplement_csv)
    prompts = load_prompts(args.behaviour, args.split)
    n_prompts = len(prompts)

    feature_nodes = [
        n for n in circuit["circuit"]["feature_nodes"]
        if n not in IO_NODES
    ]
    n_circuit_features = len(feature_nodes)
    circuit_paths = circuit["paths"]

    print(f"  Circuit: {n_circuit_features} features, "
          f"{circuit['circuit']['n_edges']} edges, {len(circuit_paths)} paths")
    print(f"  Prompts: {n_prompts}")
    print(f"  Interventions: {len(df_interventions)} rows"
          + (" (base + supplement)" if args.supplement_csv else ""))

    # Ablation_zero measured features (includes supplement if loaded)
    az = df_interventions[df_interventions["experiment_type"] == "ablation_zero"]
    measured_features = set(az["feature_id"].unique()) & set(feature_nodes)
    n_measured = len(measured_features)
    n_patching_only = n_circuit_features - n_measured
    print(f"  Ablation-measured circuit features: {n_measured}/{n_circuit_features}")
    print(f"  Patching-only features: {n_patching_only}"
          + (" ← REDUCED by supplement" if args.supplement_csv and n_patching_only < 17 else ""))

    # Baseline logit diff per prompt (unique per prompt in ablation_zero)
    baseline_by_prompt = (
        az.drop_duplicates("prompt_idx")
        .set_index("prompt_idx")["baseline_logit_diff"]
        .to_dict()
    )
    n_correct = sum(1 for i, bl in baseline_by_prompt.items() if bl > 0)
    n_incorrect = n_prompts - n_correct
    print(f"  Correct predictions: {n_correct}/{n_prompts} "
          f"({100*n_correct/n_prompts:.1f}%)")

    # ── Step 1: Feature contributions ────────────────────────────────────────
    print("\nStep 1: Feature contributions...")
    feat_df = build_feature_contributions(df_interventions, feature_nodes, n_prompts)
    save_parquet(feat_df, out_dir / f"prompt_features_{args.split}.parquet")

    # ── Step 2: Layerwise decision dynamics ──────────────────────────────────
    print("\nStep 2: Layerwise decision dynamics...")
    layer_df = build_layerwise_trace(feat_df, n_prompts, baseline_by_prompt)
    save_parquet(layer_df, out_dir / f"layerwise_decision_trace_{args.split}.parquet")

    # ── Step 3: Per-prompt path scoring ──────────────────────────────────────
    print("\nStep 3: Per-prompt path scoring...")
    paths_df = score_paths_per_prompt(circuit_paths, feat_df, n_prompts, top_k=args.top_k_paths)
    save_parquet(paths_df, out_dir / f"prompt_paths_{args.split}.parquet")

    # ── Step 4: Competing trajectories ───────────────────────────────────────
    print("\nStep 4: Competing trajectories...")
    traj_df = compute_trajectories(feat_df, n_prompts)

    # ── Step 5: Reasoning traces ─────────────────────────────────────────────
    print("\nStep 5: Building reasoning traces...")
    traces = []
    for i, prompt_meta in enumerate(prompts):
        baseline = baseline_by_prompt.get(i, 0.0)
        trace = build_trace_for_prompt(
            i, prompt_meta, baseline,
            feat_df, layer_df, paths_df, traj_df,
            top_k=args.top_k_paths,
            n_measured_features=n_measured,
            n_circuit_features=n_circuit_features,
        )
        traces.append(trace)

    trace_path = out_dir / f"reasoning_traces_{args.split}.jsonl"
    with open(trace_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"  Saved: {trace_path} ({len(traces)} traces)")

    # ── Step 6: Error analysis + coverage + failure classification ───────────
    print("\nStep 6: Error analysis...")
    error_analysis = build_error_analysis(traces, feat_df)

    # Coverage report (new in patched version)
    coverage = compute_coverage_report(feat_df, feature_nodes, args.supplement_csv)
    error_analysis["coverage"] = coverage

    # Failure type classification (new in patched version)
    failure_types = classify_failure_types(traces, feat_df)
    error_analysis["failure_types"] = failure_types

    err_path = out_dir / f"error_cases_{args.split}.json"
    err_path.write_text(json.dumps(error_analysis, indent=2))
    print(f"  Saved: {err_path}")
    print(f"  Errors: {error_analysis['n_incorrect']}/{n_prompts}")
    for pat in error_analysis["error_patterns"]:
        print(f"    → {pat}")

    # ── Figures ──────────────────────────────────────────────────────────────
    if not args.no_figures:
        print("\nGenerating figures...")
        plot_layerwise_delta(layer_df, out_dir, n_prompts)
        plot_top_paths(paths_df, circuit_paths, out_dir)
        plot_error_analysis(traces, out_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Prompts: {n_prompts} ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"Circuit: {n_circuit_features} features "
          f"({n_measured} ablation [{coverage['n_ablation_covered']}/{n_circuit_features} covered], "
          f"{coverage['n_patching_only']} patching-only)")
    if args.supplement_csv:
        print(f"  [SUPPLEMENT ACTIVE — late-hub features now ablation_zero]")

    for zone in ["early", "mid", "late"]:
        vals = [t["zone_summary"][zone]["measured_contribution"] for t in traces]
        print(f"Zone {zone}: mean ablation contrib = {np.mean(vals):+.4f} ± {np.std(vals):.4f}")

    n_correct_dom = sum(1 for t in traces if t["trajectories"]["dominant"] == "correct")
    print(f"Dominant trajectory = correct: "
          f"{n_correct_dom}/{n_prompts} ({100*n_correct_dom/n_prompts:.1f}%)")
    print(f"Trajectory prediction accuracy: "
          f"{error_analysis['trajectory_prediction_accuracy']:.4f}")

    ft = error_analysis["failure_types"]
    print(f"\nFailure type breakdown ({ft['n_incorrect']} incorrect):")
    print(f"  Type A (trajectory=correct, outcome wrong): {ft['type_A_count']}"
          f"  ← late-hub override candidates")
    print(f"  Type B (trajectory=incorrect):              {ft['type_B_count']}"
          f"  ← early/mid breakdown")

    print(f"\nOutputs in: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
