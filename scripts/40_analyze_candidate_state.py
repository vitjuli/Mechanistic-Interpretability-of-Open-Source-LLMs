"""
Candidate-state mechanistic analysis.

Reads the rich CSV from script 39 and produces:
  1. Overall baseline stats (go/no-go for full pipeline)
  2. H1 vs H2 test — counterfactual: with vs without explicit candidate set
  3. Distractor sensitivity curves — does margin decrease with harder distractors?
  4. Set-size effect — does adding more candidates change model confidence?
  5. Candidate rank distribution — what does the model rank second/third?
  6. Competition heatmap — which wrong candidates get elevated probability?
  7. Recommendation memo

Usage:
  python scripts/40_analyze_candidate_state.py
  python scripts/40_analyze_candidate_state.py --results_csv data/results/candidate_set_large/candidate_state_results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/results/candidate_set_large")


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["candidate_set"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
            )
    return df


def pct(x):
    return f"{x:.1%}"


def fmt(x, decimals=3):
    return f"{x:.{decimals}f}"


# ─── 1. Overall baseline ────────────────────────────────────────────────────

def overall_stats(df: pd.DataFrame) -> dict:
    valid = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": len(df),
        "sign_acc": float((valid > 0).mean()) if len(valid) else 0.0,
        "hard_acc": float(df["success"].astype(bool).mean()),
        "mean_nd":  float(valid.mean()) if len(valid) else 0.0,
        "mean_rank": float(df["rank_correct"].dropna().mean()) if "rank_correct" in df.columns else None,
        "mean_margin": float(df["margin"].replace([np.inf,-np.inf],np.nan).dropna().mean())
                       if "margin" in df.columns else None,
    }


# ─── 2. H1 vs H2 test ───────────────────────────────────────────────────────

def h1_vs_h2_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    H1 vs H2 test using natural structure of the dataset:
      - F5_no_set prompts: same filter, NO explicit candidate list ("Which particle X? Answer:")
      - F1_explicit_list prompts: same filter + explicit candidate set
    For each filter_correct_id:
      - no_set_nd  = mean logprob_diff_normalized for F5_no_set prompts
      - with_set_nd = mean logprob_diff_normalized for F1_explicit_list prompts
      - delta = with_set_nd - no_set_nd

    H1: delta ≈ 0 (explicit list doesn't affect the model's answer logprobs)
    H2: delta ≠ 0 (explicit list changes the competitive logprob landscape)

    Also computed: variance of with_set_nd ACROSS candidate sets (within same filter_correct_id)
    If variance is high → different candidate sets produce different confidence → H2 evidence.
    """
    if "wording_family" not in df.columns or "filter_correct_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for fcid, sub in df.groupby("filter_correct_id", dropna=True):
        # No-set baseline (F5)
        nos = sub[sub["wording_family"] == "F5_no_set"]["logprob_diff_normalized"]\
                  .replace([np.inf,-np.inf],np.nan).dropna()
        # With-set (F1 only — explicit list, comparable across sets)
        wit = sub[sub["wording_family"] == "F1_explicit_list"]["logprob_diff_normalized"]\
                  .replace([np.inf,-np.inf],np.nan).dropna()
        if len(nos) == 0 or len(wit) == 0:
            continue
        n_nd = float(nos.mean())
        w_nd = float(wit.mean())
        # Variance across candidate sets (H2: should be non-zero if set matters)
        w_std = float(wit.std()) if len(wit) > 1 else 0.0
        rows.append({
            "filter_correct_id": fcid,
            "no_set_nd":  n_nd,
            "with_set_nd_mean": w_nd,
            "with_set_nd_std":  w_std,
            "delta": w_nd - n_nd,
            "n_no_set": len(nos),
            "n_with_set": len(wit),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── 3. Distractor sensitivity ──────────────────────────────────────────────

def distractor_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each filter_correct_id, compute mean_nd by distractor_difficulty.
    H2 prediction: mean_nd decreases as difficulty increases.
    """
    if "distractor_difficulty" not in df.columns:
        return pd.DataFrame()
    ds = df[df["experiment_type"] == "distractor_sensitivity"] if "experiment_type" in df.columns else df
    ORDER = ["trivial", "easy", "medium", "hard", "hardest"]
    rows = []
    for (fcid, diff), sub in ds.groupby(["filter_correct_id", "distractor_difficulty"], dropna=True):
        sv = sub["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
        rows.append({
            "filter_correct_id": fcid,
            "distractor_difficulty": diff,
            "difficulty_rank": ORDER.index(diff) if diff in ORDER else 99,
            "mean_nd": float(sv.mean()) if len(sv) else np.nan,
            "sign_acc": float((sv > 0).mean()) if len(sv) else np.nan,
            "n": len(sub),
        })
    df_out = pd.DataFrame(rows).sort_values(["filter_correct_id", "difficulty_rank"])
    return df_out


def sensitivity_trend(df_ds: pd.DataFrame) -> dict:
    """
    For each filter_correct_id, fit slope of mean_nd vs difficulty_rank.
    Negative slope → H2 evidence (harder distractors reduce confidence).
    """
    trends = {}
    for fcid, sub in df_ds.groupby("filter_correct_id"):
        sub = sub.dropna(subset=["mean_nd"])
        if len(sub) >= 3:
            x = sub["difficulty_rank"].values
            y = sub["mean_nd"].values
            # Linear fit
            slope = float(np.polyfit(x, y, 1)[0])
            trends[fcid] = slope
    return trends


# ─── 4. Set-size effect ──────────────────────────────────────────────────────

def set_size_effect(df: pd.DataFrame) -> pd.DataFrame:
    if "n_candidates" not in df.columns:
        return pd.DataFrame()
    ss = df[df["experiment_type"] == "set_size"] if "experiment_type" in df.columns else df
    rows = []
    for (fcid, n), sub in ss.groupby(["filter_correct_id", "n_candidates"], dropna=True):
        sv = sub["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
        rows.append({
            "filter_correct_id": fcid,
            "n_candidates": int(n),
            "mean_nd": float(sv.mean()) if len(sv) else np.nan,
            "sign_acc": float((sv > 0).mean()) if len(sv) else np.nan,
            "n": len(sub),
        })
    return pd.DataFrame(rows).sort_values(["filter_correct_id", "n_candidates"])


# ─── 5. Competition analysis ─────────────────────────────────────────────────

def competition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (target_candidate, filter_property), which wrong candidates
    get elevated normalised logprobs?
    """
    pool = ["electron", "proton", "neutron", "photon", "positron", "muon"]
    nlp_cols = {p: f"nlp{p}" for p in pool}
    available = [p for p in pool if nlp_cols[p] in df.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    for (target, fp), sub in df.groupby(["target_candidate", "filter_property"], dropna=True):
        for wrong in available:
            if wrong == target:
                continue
            col = nlp_cols[wrong]
            vals = sub[col].replace([np.inf,-np.inf],np.nan).dropna()
            if len(vals):
                rows.append({
                    "target_candidate": target,
                    "filter_property": fp,
                    "wrong_candidate": wrong,
                    "mean_nlp_wrong": float(vals.mean()),
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default=str(OUT_DIR / "candidate_state_results.csv"))
    parser.add_argument("--output_dir",  default=str(OUT_DIR))
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"[ERROR] Results not found: {csv_path}")
        print("  Run: python scripts/39_evaluate_candidate_state.py first")
        import sys; sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print()

    # 1. Overall
    stats = overall_stats(df)
    print("=== Overall ===")
    print(f"  n={stats['n']}  sign_acc={pct(stats['sign_acc'])}  "
          f"hard_acc={pct(stats['hard_acc'])}  mean_nd={fmt(stats['mean_nd'])}")
    if stats["mean_rank"] is not None:
        print(f"  mean_rank_correct={fmt(stats['mean_rank'], 2)}  "
              f"mean_margin={fmt(stats['mean_margin'], 3)}")
    print()

    # 2. H1 vs H2
    df_h = h1_vs_h2_test(df)
    if len(df_h):
        print("=== H1 vs H2 test ===")
        print("  delta = with_set_nd - no_set_nd")
        print("  with_set_nd_std = spread of logprob_diff across different candidate sets")
        print("  H1: delta≈0, std≈0  |  H2: delta≠0 or std>0")
        print()
        print(df_h.to_string(index=False))
        mean_abs_delta = float(df_h["delta"].abs().mean())
        mean_std       = float(df_h["with_set_nd_std"].mean())
        print(f"\n  Mean |delta| (explicit set vs no-set): {fmt(mean_abs_delta, 3)}")
        print(f"  Mean std (variance ACROSS candidate sets): {fmt(mean_std, 3)}")
        if mean_std > 0.5:
            print("  → H2 EVIDENCE: different candidate sets produce different confidence levels")
        elif mean_abs_delta > 0.5:
            print("  → H2 EVIDENCE: explicit list shifts logprob vs no-list baseline")
        else:
            print("  → INCONCLUSIVE / H1: candidate set has minimal logprob effect")
        df_h.to_csv(out_dir / "h1_vs_h2_results.csv", index=False)
        print()

        # Per-candidate-set breakdown for the most important filter
        print("  Per-candidate-set breakdown for negative_charge__electron:")
        neg_f1 = df[(df.get("filter_correct_id", pd.Series()).eq("negative_charge__electron") if "filter_correct_id" in df.columns else pd.Series(False, index=df.index)) &
                    (df["wording_family"] == "F1_explicit_list")]
        if len(neg_f1) and "candidate_set_str" in neg_f1.columns:
            cs_breakdown = neg_f1.groupby("candidate_set_str")["logprob_diff_normalized"].mean().sort_values(ascending=False)
            for cs_str, nd in cs_breakdown.items():
                print(f"    {cs_str:<40} nd={fmt(nd)}")
        print()

    # 3. Distractor sensitivity
    df_ds = distractor_sensitivity(df)
    if len(df_ds):
        print("=== Distractor sensitivity ===")
        trends = sensitivity_trend(df_ds)
        print()
        print(df_ds[["filter_correct_id","distractor_difficulty","mean_nd","sign_acc","n"]].to_string(index=False))
        print()
        print("  Slopes (negative → harder distractors reduce margin = H2 evidence):")
        for fcid, slope in trends.items():
            h2 = "H2" if slope < -0.5 else ("weak H2" if slope < 0 else "H1")
            print(f"    {fcid}: slope={fmt(slope, 3)} [{h2}]")
        df_ds.to_csv(out_dir / "distractor_sensitivity.csv", index=False)
        print()

    # 4. Set-size
    df_ss = set_size_effect(df)
    if len(df_ss):
        print("=== Set-size effect ===")
        print(df_ss[["filter_correct_id","n_candidates","mean_nd","sign_acc","n"]].to_string(index=False))
        df_ss.to_csv(out_dir / "set_size_effect.csv", index=False)
        print()

    # 5. Competition
    df_comp = competition_analysis(df)
    if len(df_comp):
        # Top competitors per (target, filter)
        top_comp = (df_comp.sort_values("mean_nlp_wrong", ascending=False)
                    .groupby(["target_candidate","filter_property"]).head(2))
        print("=== Competition (top-2 wrong candidates per target/filter) ===")
        print(top_comp[["target_candidate","filter_property","wrong_candidate","mean_nlp_wrong","n"]].to_string(index=False))
        df_comp.to_csv(out_dir / "competition_analysis.csv", index=False)
        print()

    # ── Recommendation memo ──────────────────────────────────────────────────
    sa = stats["sign_acc"]
    ha = stats["hard_acc"]
    nd = stats["mean_nd"]

    lines = [
        "# Candidate-State Analysis Report",
        "",
        "## 1. Go / No-Go",
        "",
        f"- Sign accuracy: {pct(sa)}",
        f"- Hard accuracy: {pct(ha)}",
        f"- Mean normalised logprob diff: {fmt(nd)}",
        "",
    ]
    if sa >= 0.90:
        lines += ["**Verdict: GO — strong enough for full mechanistic pipeline.**",
                  "",
                  "Recommended next step: run feature extraction (script 04) then attribution "
                  "graph (script 06) with `--behaviour physics_particle_candidate_selection`.",
                  ""]
    elif sa >= 0.80:
        lines += ["**Verdict: MARGINAL — proceed with caution.**",
                  "",
                  "Focus mechanistic analysis on the core experiment subset (experiment_type=core) "
                  "which likely has higher accuracy than the combined dataset.",
                  ""]
    else:
        lines += ["**Verdict: FAIL — behaviour too weak for mechanistic analysis.**", ""]

    lines += [
        "## 2. H1 vs H2 Evidence",
        "",
    ]
    if len(df_h):
        abs_d = float(df_h["delta"].abs().mean())
        if abs_d > 1.0:
            verdict_h = "STRONG H2 — explicit candidate set substantially changes logprob."
        elif abs_d > 0.3:
            verdict_h = "WEAK H2 — candidate set has modest effect on logprob."
        else:
            verdict_h = "INCONCLUSIVE / H1 — candidate set has minimal effect."
        lines += [
            f"Mean |delta| (original vs no_set): {fmt(abs_d)}",
            f"Interpretation: **{verdict_h}**",
            "",
        ]

    lines += [
        "## 3. Distractor Sensitivity",
        "",
    ]
    if trends:
        neg_slopes = {k: v for k, v in trends.items() if v < 0}
        if neg_slopes:
            lines += [
                f"Negative slopes for {len(neg_slopes)}/{len(trends)} filter_correct_id pairs:",
                f"  {list(neg_slopes.keys())}",
                "",
                "**Interpretation:** Harder distractors reduce model confidence, consistent with "
                "candidate-set-mediated reasoning (H2). This is the strongest mechanistic evidence "
                "that the model does not purely retrieve 'filter → answer' independently of context.",
                "",
            ]
        else:
            lines += ["No negative slopes found. Marginal evidence for H2 from distractor sensitivity.", ""]

    lines += [
        "## 4. Recommended Full Pipeline Config",
        "",
        "Use `experiment_type=core` prompts only for feature extraction (more stable signal).",
        "If H2 evidence is strong: include distractor_sensitivity prompts for cross-prompt analysis.",
        "",
        "```bash",
        "# Feature extraction",
        "python scripts/04_extract_transcoder_features.py \\",
        "    --behaviour physics_particle_candidate_selection \\",
        "    --split train --device cuda --top_k 50",
        "",
        "# Attribution graph",
        "python scripts/06_build_attribution_graph.py \\",
        "    --behaviour physics_particle_candidate_selection \\",
        "    --split train --activation_weighted --top_k_per_layer 3",
        "```",
        "",
    ]

    report = "\n".join(lines)
    report_path = out_dir / "analysis_report.md"
    report_path.write_text(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
