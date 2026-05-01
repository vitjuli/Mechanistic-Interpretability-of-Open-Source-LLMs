"""
Script 18: Latent-state probing analysis for physics_decay_type_probe.

Uses the ablation CSV (Section C output) to run SE and IC tests.
Full activation-based analysis (cosine similarity of raw transcoder activations)
requires the transcoder feature files from Section B — see --activation_dir.

Tests implemented here:
  0. Baseline breakdown — per-level and per-group accuracy from Section A CSV
  1. SE test   — within-group sign_flip_rate variance across wording variants
  2. IC test   — cross-group effect profile similarity (same concept, different cues)
  3. CP test   — differential feature activation between contrastive pair members
  4. KW test   — comparison of keyword-free vs keyword-present ablation effects

Usage:
  # Baseline-only (no ablation needed, runs immediately after Section A):
  python scripts/18_latent_state_analysis.py \
    --behaviour physics_decay_type_probe --split train --baseline_only

  # Core subset only (L1 + L2 + L3 anchors, n=350):
  python scripts/18_latent_state_analysis.py \
    --behaviour physics_decay_type_probe --split train --baseline_only --core_only

  # Full analysis (requires ablation CSV from step 07):
  python scripts/18_latent_state_analysis.py \
    --behaviour physics_decay_type_probe --split train

  # With activation data (after rsyncing transcoder features from CSD3):
  python scripts/18_latent_state_analysis.py \
    --behaviour physics_decay_type_probe --split train \
    --activation_dir data/results/transcoder_features

Outputs:
  data/results/grouping/probe_baseline_breakdown.csv — per-prompt with metadata
  data/results/grouping/probe_level_summary.csv      — per-level accuracy stats
  data/results/grouping/probe_group_summary.csv      — per-group accuracy stats
  data/results/grouping/probe_se_analysis.csv        — SE test per group
  data/results/grouping/probe_ic_analysis.csv        — IC convergence per group pair
  data/results/grouping/probe_cp_analysis.csv        — CP discriminating features
  data/results/grouping/probe_summary.html           — interactive Plotly summary

Core set definition (--core_only):
  level in [1, 2]  OR  group_id in ['L3-FA', 'L3-FB']
  n=350: L1(160) + L2(174) + anchors(16)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/results/grouping"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ablation(behaviour: str, ui_run_dir: Path | None = None) -> pd.DataFrame:
    """Load ablation CSV — tries UI run raw_sources first, then results dir."""
    candidates = []

    # Try the most recent UI run for this behaviour
    if ui_run_dir is None:
        ui_runs = sorted(
            (ROOT / "data/ui_offline").glob(f"*_{behaviour}_train_n*"),
            reverse=True
        )
        if ui_runs:
            ui_run_dir = ui_runs[0]

    if ui_run_dir:
        raw = ui_run_dir / "raw_sources"
        for stem in [f"intervention_ablation_{behaviour}", f"intervention_ablation_{behaviour}_train"]:
            candidates.append(raw / f"{stem}.csv")

    # Also try results dir
    res = ROOT / f"data/results/interventions/{behaviour}"
    for stem in [f"intervention_ablation_{behaviour}", f"intervention_ablation_{behaviour}_train"]:
        candidates.append(res / f"{stem}.csv")

    for p in candidates:
        if p.exists():
            print(f"  Loading ablation CSV: {p}")
            df = pd.read_csv(p)
            if "sign_flipped" in df.columns:
                df["sign_flipped"] = df["sign_flipped"].astype(bool)
            return df

    raise FileNotFoundError(f"No ablation CSV found for {behaviour}. Tried:\n" +
                            "\n".join(f"  {c}" for c in candidates))


def load_prompts(behaviour: str, split: str) -> list[dict]:
    p = ROOT / f"data/prompts/{behaviour}_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Prompts not found: {p}")
    return [json.loads(l) for l in open(p)]


def load_baseline(behaviour: str, split: str) -> pd.DataFrame | None:
    p = ROOT / f"data/results/baseline_{behaviour}_{split}.csv"
    if p.exists():
        print(f"  Loading baseline: {p}")
        return pd.read_csv(p)
    return None


# Core set filter: level in [1,2] or L3 anchors
CORE_GROUPS = {"L3-FA", "L3-FB"}

def is_core(p: dict) -> bool:
    lv = p.get("level")
    return lv in (1, 2) or p.get("group_id") in CORE_GROUPS


# ─────────────────────────────────────────────────────────────────────────────
# Test 0 — Baseline breakdown
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_breakdown(baseline_df: pd.DataFrame, prompts: list[dict],
                           core_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join baseline CSV with prompt metadata and compute per-group / per-level accuracy.

    Returns:
        detail_df   — per-prompt rows with metadata + accuracy
        level_df    — per-level aggregated stats
        group_df    — per-group aggregated stats
    """
    prompt_by_text = {p["prompt"]: p for p in prompts}

    rows = []
    for _, row in baseline_df.iterrows():
        meta = prompt_by_text.get(row["prompt"], {})
        lv   = meta.get("level")
        lv_key = lv if lv is not None else "AUX"
        diff = float(row["logprob_diff_normalized"])
        rows.append({
            "prompt_id":        meta.get("prompt_id", ""),
            "level":            lv_key,
            "group_id":         meta.get("group_id", "?"),
            "correct_answer":   str(row.get("correct_answer", "")).strip(),
            "logprob_diff":     round(diff, 4),
            "sign_correct":     diff > 0,
            "hard_correct":     diff > 0.5,
            "difficulty":       meta.get("difficulty", "?"),
            "keyword_free":     meta.get("keyword_free", True),
            "is_anchor":        meta.get("is_anchor", False),
            "is_auxiliary":     meta.get("is_auxiliary", False),
            "is_kw_variant":    meta.get("is_kw_variant", False),
            "cue_type":         meta.get("cue_type", ""),
            "relation_type":    meta.get("relation_type", ""),
            "concept_route":    meta.get("concept_route", ""),
            "inference_steps":  meta.get("inference_steps", ""),
            "contrastive_pair": meta.get("contrastive_pair_id", ""),
            "contrastive_role": meta.get("contrastive_role", ""),
            "is_core":          is_core(meta),
        })

    detail_df = pd.DataFrame(rows)
    if core_only:
        detail_df = detail_df[detail_df["is_core"]].copy()

    def agg(sub, key_col, key_val):
        if sub.empty:
            return {}
        return {
            key_col:        key_val,
            "n":            len(sub),
            "sign_acc":     round(sub["sign_correct"].mean(), 4),
            "hard_acc":     round(sub["hard_correct"].mean(), 4),
            "mean_diff":    round(sub["logprob_diff"].mean(), 4),
            "std_diff":     round(sub["logprob_diff"].std(), 4),
            "n_fail_sign":  int((~sub["sign_correct"]).sum()),
            "n_fail_hard":  int((~sub["hard_correct"]).sum()),
        }

    # Per-level
    level_rows = []
    for lv in ["1", "2", "3", "AUX"]:
        sub = detail_df[detail_df["level"].astype(str) == lv]
        r = agg(sub, "level", lv)
        if r:
            level_rows.append(r)
    level_df = pd.DataFrame(level_rows)

    # Per-group
    group_rows = []
    for gid, sub in detail_df.groupby("group_id"):
        meta_row = sub.iloc[0]
        r = agg(sub, "group_id", gid)
        if r:
            r["level"]         = meta_row["level"]
            r["correct_answer"] = meta_row["correct_answer"]
            r["cue_type"]      = meta_row["cue_type"]
            r["relation_type"] = meta_row["relation_type"]
            r["concept_route"] = meta_row["concept_route"]
            r["is_core"]       = meta_row["is_core"]
            group_rows.append(r)
    group_df = pd.DataFrame(group_rows).sort_values(["level", "group_id"])

    return detail_df, level_df, group_df


def print_baseline_report(level_df: pd.DataFrame, group_df: pd.DataFrame,
                           core_only: bool) -> None:
    """Print formatted baseline breakdown to stdout."""
    label = " [CORE ONLY]" if core_only else " [ALL PROMPTS]"
    print(f"\n{'='*70}")
    print(f"BASELINE BREAKDOWN{label}")
    print(f"{'='*70}")

    if not level_df.empty:
        print(f"\n{'Level':<8} {'n':>5} {'sign_acc':>10} {'hard_acc':>10} {'mean_diff':>11} {'n_fail_sign':>12}")
        print("-" * 60)
        total_n = 0; total_sign = 0; total_hard = 0
        for _, r in level_df.iterrows():
            n = int(r['n'])
            total_n += n
            total_sign += int(r['sign_acc'] * n)
            total_hard += int(r['hard_acc'] * n)
            flag = "  ← WARN" if r['sign_acc'] < 0.80 else ""
            print(f"  L{r['level']:<5} {n:>5}  {r['sign_acc']:>9.1%}  {r['hard_acc']:>9.1%}  {r['mean_diff']:>10.3f}{flag}")
        print("-" * 60)
        print(f"  {'TOTAL':<6} {total_n:>5}  {total_sign/total_n:>9.1%}  {total_hard/total_n:>9.1%}")

    if not group_df.empty:
        print(f"\n{'Group':<18} {'Lv':<4} {'n':>4} {'sign_acc':>9} {'hard_acc':>9} {'mean_diff':>10}  cue/route")
        print("-" * 80)
        for _, r in group_df.iterrows():
            flag = " ✗" if r['sign_acc'] < 0.70 else (" △" if r['sign_acc'] < 0.85 else "")
            def _s(v): return v if isinstance(v, str) else ""
            cue = (_s(r['concept_route']) or _s(r['relation_type']) or _s(r['cue_type']))[:22]
            print(f"  {r['group_id']:<16} L{r['level']:<3} {int(r['n']):>4}  {r['sign_acc']:>8.1%}  {r['hard_acc']:>8.1%}  {r['mean_diff']:>9.3f}  {cue}{flag}")

    # Summary verdict
    overall_sign = level_df["sign_acc"].mul(level_df["n"]).sum() / level_df["n"].sum() if not level_df.empty else 0
    print(f"\n{'─'*70}")
    if overall_sign >= 0.90:
        verdict = "PASS (≥90%)"
    elif overall_sign >= 0.85:
        verdict = "PASS (≥85% gate)"
    elif overall_sign >= 0.78:
        verdict = "WARN_PASS (below gate but structured failures)"
    else:
        verdict = "FAIL — review failure pattern"
    print(f"  Overall sign accuracy: {overall_sign:.1%}  →  {verdict}")
    if not core_only:
        core_n = group_df[group_df["is_core"]]["n"].sum() if "is_core" in group_df else 0
        core_sign = (group_df[group_df["is_core"]]["sign_acc"] * group_df[group_df["is_core"]]["n"]).sum() / core_n if core_n > 0 else 0
        print(f"  Core subset (L1+L2+anchors, n={int(core_n)}): {core_sign:.1%}")
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — SE test: within-group ablation effect consistency
# ─────────────────────────────────────────────────────────────────────────────

def run_se_test(df: pd.DataFrame, prompts: list[dict]) -> pd.DataFrame:
    """
    Semantic-equivalence test using ablation effects.

    For each SE group (group_id), compute:
    - mean_sfr: mean sign_flip_rate across all prompts in the group
    - std_sfr: std of per-prompt sfr (over all features), measuring within-group variance
    - mean_abs_effect: mean |effect_size| across all (feature, prompt) pairs in group
    - cv_abs_effect: coefficient of variation of |effect_size| within group

    Low std_sfr + low cv_abs_effect = consistent ablation effects across wording variants
    → suggests the group's prompts are encoded via similar internal features (SE test PASS).
    """
    prompt_meta = {i: p for i, p in enumerate(prompts)}
    df = df.copy()
    df["group_id"]   = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("group_id"))
    df["level"]      = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("level"))
    df["concept"]    = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("physics_concept"))
    df["test_type"]  = df["prompt_idx"].map(lambda i: str(prompt_meta.get(i, {}).get("test_type", [])))
    df["is_aux"]     = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("is_auxiliary", False))

    # Exclude auxiliary prompts from SE analysis
    core = df[~df["is_aux"]]

    records = []
    for gid, grp in core.groupby("group_id"):
        if grp.empty:
            continue
        meta = prompt_meta.get(grp["prompt_idx"].iloc[0], {})
        level = meta.get("level")
        concept = meta.get("physics_concept", "?")
        n_prompts_in_group = grp["prompt_idx"].nunique()

        # Per-prompt sfr (aggregated over all features)
        per_prompt_sfr = grp.groupby("prompt_idx")["sign_flipped"].mean()
        mean_sfr = per_prompt_sfr.mean()
        std_sfr  = per_prompt_sfr.std() if len(per_prompt_sfr) > 1 else 0.0

        # |effect_size| distribution
        abs_effects = grp["abs_effect_size"].dropna()
        mean_abs = abs_effects.mean()
        std_abs  = abs_effects.std()
        cv_abs   = std_abs / (mean_abs + 1e-6)

        # Which features drive sign flips in this group?
        feature_sfr = grp.groupby("feature_id")["sign_flipped"].mean()
        top_features = feature_sfr.nlargest(5).index.tolist()

        records.append({
            "group_id":       gid,
            "level":          level,
            "concept":        concept,
            "n_prompts":      n_prompts_in_group,
            "mean_sfr":       round(mean_sfr, 4),
            "std_sfr":        round(std_sfr,  4),
            "cv_sfr":         round(std_sfr / (mean_sfr + 1e-6), 3),
            "mean_abs_effect": round(mean_abs, 5),
            "cv_abs_effect":  round(cv_abs, 3),
            "top_features":   ",".join(top_features),
        })

    se_df = pd.DataFrame(records).sort_values(["level", "concept", "group_id"])
    return se_df


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — IC test: cross-group effect profile similarity
# ─────────────────────────────────────────────────────────────────────────────

def run_ic_test(df: pd.DataFrame, prompts: list[dict]) -> pd.DataFrame:
    """
    Inferential convergence test.

    For each concept (alpha/beta), compare the ablation effect profile across groups.
    The effect profile for a group = mean effect_size per feature (40-dim vector).

    Cosine similarity between group profiles measures whether different cue types
    drive the same circuit features.

    Groups at Level 3 (concept probes) should be most similar to the anchor (L3-FA/L3-FB)
    if a latent concept state exists.
    """
    prompt_meta = {i: p for i, p in enumerate(prompts)}
    df = df.copy()
    df["group_id"] = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("group_id"))
    df["level"]    = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("level"))
    df["concept"]  = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("physics_concept"))
    df["is_aux"]   = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("is_auxiliary", False))

    core = df[~df["is_aux"]]
    features = sorted(core["feature_id"].unique())
    feat_idx = {f: i for i, f in enumerate(features)}

    # Build per-group effect profile (mean effect_size per feature)
    profiles = {}
    for gid, grp in core.groupby("group_id"):
        vec = np.zeros(len(features))
        for fid, fgrp in grp.groupby("feature_id"):
            if fid in feat_idx:
                vec[feat_idx[fid]] = fgrp["effect_size"].mean()
        profiles[gid] = vec

    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(np.dot(a, b) / denom)

    # For each concept, compute similarity of each group to the anchor
    records = []
    for concept in ["alpha_decay", "beta_decay"]:
        anchor_id = "L3-FA" if concept == "alpha_decay" else "L3-FB"
        if anchor_id not in profiles:
            continue
        anchor_vec = profiles[anchor_id]

        concept_groups = {
            gid: vec for gid, vec in profiles.items()
            if prompt_meta.get(
                next((r["prompt_idx"] for _, r in core[core["group_id"] == gid].iterrows()), 0), {}
            ).get("physics_concept") == concept
        }

        for gid, vec in sorted(concept_groups.items()):
            meta = prompt_meta.get(
                next((r["prompt_idx"] for _, r in core[core["group_id"] == gid].iterrows()), 0), {}
            )
            sim_to_anchor = cosine(vec, anchor_vec)
            records.append({
                "group_id":       gid,
                "concept":        concept,
                "level":          meta.get("level"),
                "sim_to_anchor":  round(sim_to_anchor, 4),
                "anchor_id":      anchor_id,
                "vec_norm":       round(float(np.linalg.norm(vec)), 4),
            })

    ic_df = pd.DataFrame(records).sort_values(["concept", "level", "sim_to_anchor"], ascending=[True, True, False])
    return ic_df


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — CP test: differential features between contrastive pairs
# ─────────────────────────────────────────────────────────────────────────────

def run_cp_test(df: pd.DataFrame, prompts: list[dict]) -> pd.DataFrame:
    """
    Contrastive pair test.

    For each CP-XX pair, compute the feature-wise effect difference:
        diff[feature] = mean_effect(alpha_member) - mean_effect(beta_member)

    Features with largest |diff| are the mechanistic discriminators for that pair's
    physical contrast. If the same features discriminate all pairs, they are
    global concept discriminators. If pair-specific, they are cue-specific.
    """
    prompt_meta = {i: p for i, p in enumerate(prompts)}
    df = df.copy()
    df["cp_id"]   = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("contrastive_pair_id"))
    df["cp_role"] = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("contrastive_role"))

    cp_df = df[df["cp_id"].notna() & df["cp_role"].notna()]
    if cp_df.empty:
        print("  [WARN] No contrastive pair rows found — check metadata.")
        return pd.DataFrame()

    records = []
    for cp_id, grp in cp_df.groupby("cp_id"):
        alpha_rows = grp[grp["cp_role"] == "alpha_member"]
        beta_rows  = grp[grp["cp_role"] == "beta_member"]
        if alpha_rows.empty or beta_rows.empty:
            continue

        alpha_by_feat = alpha_rows.groupby("feature_id")["effect_size"].mean()
        beta_by_feat  = beta_rows.groupby("feature_id")["effect_size"].mean()

        all_feats = sorted(set(alpha_by_feat.index) | set(beta_by_feat.index))
        diffs = {f: alpha_by_feat.get(f, 0.0) - beta_by_feat.get(f, 0.0) for f in all_feats}
        sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

        top_alpha_feats = [f for f, d in sorted_diffs if d > 0][:5]
        top_beta_feats  = [f for f, d in sorted_diffs if d < 0][:5]
        max_diff = abs(sorted_diffs[0][1]) if sorted_diffs else 0.0

        records.append({
            "cp_id":              cp_id,
            "n_alpha_prompts":    alpha_rows["prompt_idx"].nunique(),
            "n_beta_prompts":     beta_rows["prompt_idx"].nunique(),
            "max_abs_diff":       round(max_diff, 5),
            "top_alpha_features": ",".join(top_alpha_feats),
            "top_beta_features":  ",".join(top_beta_feats),
        })

    return pd.DataFrame(records).sort_values("cp_id")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — KW test: keyword-free vs direct-keyword ablation effects
# ─────────────────────────────────────────────────────────────────────────────

def run_kw_test(df: pd.DataFrame, prompts: list[dict]) -> pd.DataFrame:
    """
    Keyword shortcut test.

    For each KW-XX pair, compare ablation effect profiles between:
    - v1 (keyword-free) and v2 (direct keyword)

    If profiles are similar (high cosine similarity), keywords don't change
    the circuit used. If dissimilar, keywords activate a different pathway.
    """
    prompt_meta = {i: p for i, p in enumerate(prompts)}
    df = df.copy()
    df["group_id"]  = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("group_id"))
    df["kw_variant"] = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("wording_variant"))
    df["is_kw"]     = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("is_kw_variant", False))

    kw_df = df[df["is_kw"]]
    if kw_df.empty:
        print("  [WARN] No keyword-variant rows found.")
        return pd.DataFrame()

    records = []
    for gid, grp in kw_df.groupby("group_id"):
        v1 = grp[grp["kw_variant"] == 1]
        v2 = grp[grp["kw_variant"] == 2]
        if v1.empty or v2.empty:
            continue

        feats = sorted(grp["feature_id"].unique())
        def profile(sub):
            return np.array([sub[sub["feature_id"] == f]["effect_size"].mean()
                             if f in sub["feature_id"].values else 0.0
                             for f in feats])

        p1 = profile(v1)
        p2 = profile(v2)
        denom = (np.linalg.norm(p1) * np.linalg.norm(p2)) + 1e-10
        sim = float(np.dot(p1, p2) / denom)

        label = prompt_meta.get(v1["prompt_idx"].iloc[0], {}).get("physics_concept", "?")
        records.append({
            "kw_pair_id":  gid,
            "concept":     label,
            "profile_sim": round(sim, 4),
            "sfr_kw_free": round(v1["sign_flipped"].mean(), 4),
            "sfr_kw_direct": round(v2["sign_flipped"].mean(), 4),
            "sfr_delta":   round(v2["sign_flipped"].mean() - v1["sign_flipped"].mean(), 4),
        })

    return pd.DataFrame(records).sort_values("kw_pair_id")


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_COLORS = {1: "#4a90d9", 2: "#7b68ee", 3: "#e07b39", None: "#555"}
CONCEPT_COLORS = {"alpha_decay": "#e07b39", "beta_decay": "#4a90d9"}


def make_summary_html(se_df, ic_df, cp_df, kw_df) -> str:
    figs = []

    # 1. SE test: cv_sfr by group (lower = more consistent within-group)
    if not se_df.empty:
        se_sorted = se_df.sort_values(["level", "concept", "cv_sfr"])
        fig1 = go.Figure()
        for level in [1, 2, 3]:
            sub = se_sorted[se_sorted["level"] == level]
            if sub.empty:
                continue
            fig1.add_trace(go.Bar(
                x=sub["group_id"],
                y=sub["cv_sfr"],
                name=f"Level {level}",
                marker_color=LEVEL_COLORS[level],
                customdata=sub[["mean_sfr", "n_prompts", "top_features"]].values,
                hovertemplate="<b>%{x}</b><br>CV sfr: %{y:.3f}<br>Mean sfr: %{customdata[0]:.3f}<br>N: %{customdata[1]}<br>Top: %{customdata[2]}<extra></extra>",
            ))
        fig1.update_layout(
            title="SE test — coefficient of variation of sign_flip_rate within group<br><sub>Lower = more consistent across wording variants (SE test PASS)</sub>",
            barmode="group",
            height=450, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#ccc"), xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(title="CV(sfr)", gridcolor="#222"),
            legend=dict(font=dict(size=10)),
        )
        figs.append(fig1.to_html(include_plotlyjs="cdn", full_html=False))

    # 2. IC test: similarity to anchor by group
    if not ic_df.empty:
        fig2 = go.Figure()
        for concept in ["alpha_decay", "beta_decay"]:
            sub = ic_df[ic_df["concept"] == concept].sort_values("level")
            if sub.empty:
                continue
            fig2.add_trace(go.Bar(
                x=sub["group_id"],
                y=sub["sim_to_anchor"],
                name=concept.replace("_", " "),
                marker_color=CONCEPT_COLORS[concept],
                hovertemplate="<b>%{x}</b><br>Sim to anchor: %{y:.4f}<extra></extra>",
            ))
        fig2.update_layout(
            title="IC test — cosine similarity of group effect profile to full-specification anchor<br><sub>Higher = group activates same circuit features as anchor (IC test PASS)</sub>",
            barmode="group",
            height=450, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#ccc"), xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(title="Cosine similarity to anchor", range=[0, 1], gridcolor="#222"),
        )
        figs.append(fig2.to_html(include_plotlyjs=False, full_html=False))

    # 3. KW test: sfr comparison keyword-free vs keyword
    if not kw_df.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=kw_df["kw_pair_id"], y=kw_df["sfr_kw_free"],
                              name="Keyword-free", marker_color="#4a90d9"))
        fig3.add_trace(go.Bar(x=kw_df["kw_pair_id"], y=kw_df["sfr_kw_direct"],
                              name="Direct keyword", marker_color="#f5a623"))
        fig3.update_layout(
            title="KW test — sign_flip_rate: keyword-free vs direct keyword prompts<br><sub>Similar sfr = keyword doesn't change circuit used</sub>",
            barmode="group", height=350,
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#ccc"), xaxis=dict(tickangle=-45),
            yaxis=dict(title="Sign flip rate", gridcolor="#222"),
        )
        figs.append(fig3.to_html(include_plotlyjs=False, full_html=False))

    # 4. CP test: max discriminating feature per pair
    if not cp_df.empty:
        fig4 = go.Figure(go.Bar(
            x=cp_df["cp_id"],
            y=cp_df["max_abs_diff"],
            marker_color="#7b68ee",
            customdata=cp_df[["top_alpha_features", "top_beta_features"]].values,
            hovertemplate="<b>%{x}</b><br>Max |diff|: %{y:.5f}<br>α-features: %{customdata[0]}<br>β-features: %{customdata[1]}<extra></extra>",
        ))
        fig4.update_layout(
            title="CP test — max |effect_size| difference between contrastive pair members<br><sub>The feature with max diff is the discriminating feature for that physical contrast</sub>",
            height=350, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#ccc"), xaxis=dict(tickangle=-45),
            yaxis=dict(title="Max |effect diff|", gridcolor="#222"),
        )
        figs.append(fig4.to_html(include_plotlyjs=False, full_html=False))

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Latent-State Probing Analysis</title>
<style>
  body{background:#0f1117;color:#ccc;font-family:monospace;margin:20px;}
  h1{color:#4a90d9;} h2{color:#7b68ee;margin-top:30px;}
  .section{margin-bottom:30px;border:1px solid #222;padding:15px;border-radius:8px;}
</style>
</head><body>
<h1>Latent-State Probing Analysis — physics_decay_type_probe</h1>
""" + "\n".join(f'<div class="section">{f}</div>' for f in figs) + """
</body></html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Script 18: Latent-state probing analysis")
    parser.add_argument("--behaviour", default="physics_decay_type_probe")
    parser.add_argument("--split",     default="train")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run only the baseline breakdown (no ablation CSV required)")
    parser.add_argument("--core_only", action="store_true",
                        help="Restrict analysis to core subset: L1+L2+L3 anchors (n=350)")
    parser.add_argument("--activation_dir", type=Path, default=None,
                        help="Path to transcoder_features directory (for activation-based analysis)")
    parser.add_argument("--ui_run",    type=Path, default=None,
                        help="Explicit UI run directory to load ablation CSV from")
    args = parser.parse_args()

    print(f"=== Latent-State Probing Analysis: {args.behaviour} ===\n")

    prompts = load_prompts(args.behaviour, args.split)
    print(f"Loaded {len(prompts)} prompts")
    if args.core_only:
        core_n = sum(1 for p in prompts if is_core(p))
        print(f"  Core subset: {core_n} prompts (L1+L2+anchors)")

    # ── Test 0: baseline breakdown (always runs if CSV exists) ────────────────
    baseline = load_baseline(args.behaviour, args.split)
    if baseline is not None:
        detail_df, level_df, group_df = run_baseline_breakdown(baseline, prompts, args.core_only)
        print_baseline_report(level_df, group_df, args.core_only)

        detail_out = OUT_DIR / "probe_baseline_breakdown.csv"
        level_out  = OUT_DIR / "probe_level_summary.csv"
        group_out  = OUT_DIR / "probe_group_summary.csv"
        detail_df.to_csv(detail_out, index=False)
        level_df.to_csv(level_out,  index=False)
        group_df.to_csv(group_out,  index=False)
        print(f"  Saved: {detail_out}")
        print(f"  Saved: {level_out}")
        print(f"  Saved: {group_out}")
    else:
        print("  [NOTE] Baseline CSV not found — run step 02 first, then rsync the CSV.")
        if args.baseline_only:
            print("  Nothing to do in --baseline_only mode without the CSV. Exiting.")
            return

    if args.baseline_only:
        print("\n=== Done (baseline-only mode) ===")
        return

    # ── Full analysis: requires ablation CSV from step 07 ────────────────────
    print()
    try:
        df = load_ablation(args.behaviour, ui_run_dir=args.ui_run)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Full analysis requires ablation CSV from step 07. Run the full pipeline first.")
        return

    print(f"Loaded {len(df)} ablation rows ({df['feature_id'].nunique()} features × {df['prompt_idx'].nunique()} prompts)")
    print()

    # Filter prompts for core_only mode (ablation tests)
    active_prompts = [p for p in prompts if is_core(p)] if args.core_only else prompts
    active_indices = {i for i, p in enumerate(prompts) if not args.core_only or is_core(p)}
    df_active = df[df["prompt_idx"].isin(active_indices)].copy() if args.core_only else df

    # ── Test 1: SE ────────────────────────────────────────────────────────────
    print("--- Test 1: SE (semantic-equivalence) ---")
    se_df = run_se_test(df_active, prompts)
    se_out = OUT_DIR / "probe_se_analysis.csv"
    se_df.to_csv(se_out, index=False)
    print(f"  Groups: {len(se_df)}  Saved: {se_out}")
    print("  Low cv_sfr groups (consistent, SE likely PASS):")
    for _, r in se_df[se_df["cv_sfr"] < 0.5].sort_values("cv_sfr").head(10).iterrows():
        print(f"    {r['group_id']:18s} L{r['level']} cv={r['cv_sfr']:.3f}  sfr={r['mean_sfr']:.4f}")
    print("  High cv_sfr groups (wording-sensitive):")
    for _, r in se_df[se_df["cv_sfr"] >= 0.5].sort_values("cv_sfr", ascending=False).head(5).iterrows():
        print(f"    {r['group_id']:18s} L{r['level']} cv={r['cv_sfr']:.3f}  sfr={r['mean_sfr']:.4f}")

    # ── Test 2: IC ────────────────────────────────────────────────────────────
    print()
    print("--- Test 2: IC (inferential convergence) ---")
    ic_df = run_ic_test(df_active, prompts)
    ic_out = OUT_DIR / "probe_ic_analysis.csv"
    ic_df.to_csv(ic_out, index=False)
    print(f"  Comparisons: {len(ic_df)}  Saved: {ic_out}")
    print("  Top IC convergence (sim to anchor):")
    for _, r in ic_df.sort_values("sim_to_anchor", ascending=False).head(10).iterrows():
        print(f"    {r['group_id']:18s} L{r['level']} {r['concept'][:5]}  sim={r['sim_to_anchor']:.4f}")
    print("  Lowest IC convergence:")
    for _, r in ic_df.sort_values("sim_to_anchor").head(5).iterrows():
        print(f"    {r['group_id']:18s} L{r['level']} {r['concept'][:5]}  sim={r['sim_to_anchor']:.4f}")

    # ── Test 3: CP ────────────────────────────────────────────────────────────
    print()
    print("--- Test 3: CP (contrastive pairs) ---")
    cp_df = run_cp_test(df_active, prompts)
    cp_out = OUT_DIR / "probe_cp_analysis.csv"
    cp_df.to_csv(cp_out, index=False)
    print(f"  Pairs: {len(cp_df)}  Saved: {cp_out}")
    if not cp_df.empty:
        for _, r in cp_df.sort_values("max_abs_diff", ascending=False).head(5).iterrows():
            print(f"    {r['cp_id']:12s}  max_diff={r['max_abs_diff']:.5f}  α-feat: {r['top_alpha_features'][:40]}")

    # ── Test 4: KW ────────────────────────────────────────────────────────────
    print()
    print("--- Test 4: KW (keyword presence) ---")
    kw_df = run_kw_test(df_active, prompts)
    kw_out = OUT_DIR / "probe_kw_analysis.csv"
    kw_df.to_csv(kw_out, index=False)
    print(f"  KW pairs: {len(kw_df)}  Saved: {kw_out}")
    if not kw_df.empty:
        for _, r in kw_df.iterrows():
            print(f"    {r['kw_pair_id']:12s}  sim={r['profile_sim']:.4f}  Δsfr={r['sfr_delta']:+.4f}")

    # ── Summary HTML ──────────────────────────────────────────────────────────
    print()
    print("--- Building summary HTML ---")
    html = make_summary_html(se_df, ic_df, cp_df, kw_df)
    html_out = OUT_DIR / "probe_summary.html"
    html_out.write_text(html)
    print(f"  Saved: {html_out}")

    if args.activation_dir:
        print()
        print(f"  [NOTE] --activation_dir specified: {args.activation_dir}")
        print("         Raw transcoder activation analysis is not yet implemented.")
        print("         The ablation-based SE/IC tests above use effect profiles as proxies.")

    print("\n=== Done ===")
    print(f"Outputs in {OUT_DIR}:")
    print("  probe_baseline_breakdown.csv  probe_level_summary.csv  probe_group_summary.csv")
    print("  probe_se_analysis.csv  probe_ic_analysis.csv  probe_cp_analysis.csv")
    print("  probe_kw_analysis.csv  probe_summary.html")


if __name__ == "__main__":
    main()
