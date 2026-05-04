"""
Candidate-set behaviour screening: baseline evaluation across all mini behaviours.

Loads the language model once, evaluates all four mini behaviours in sequence,
writes per-behaviour CSVs, and produces a ranked comparison report.

Usage:
  python scripts/33_run_candidate_set_screening.py
  python scripts/33_run_candidate_set_screening.py --behaviours physics_parity_rule_mini physics_spin_statistics_mini
  python scripts/33_run_candidate_set_screening.py --split train --no-viz

Outputs:
  data/results/baseline_<behaviour>_train.csv   — per-behaviour results (compatible with downstream)
  data/results/candidate_set_screening_summary.csv
  data/results/candidate_set_screening_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_utils import ModelWrapper

# ─── configuration ──────────────────────────────────────────────────────────

BEHAVIOURS_DEFAULT = [
    "physics_parity_rule_mini",
    "physics_spin_statistics_mini",
    "physics_approximation_regime_mini",
    "physics_e1_selection_mini",
]

MECHANISTIC_NOTES = {
    "physics_parity_rule_mini": (
        "Rule: (-1)^l → even/odd. Model must map orbital name→l (F1) or apply parity formula directly "
        "(F2/F3). Strong candidate for candidate-set computation: two-step filter "
        "(l value extraction → parity bit)."
    ),
    "physics_spin_statistics_mini": (
        "Rule: integer spin → boson, half-integer → fermion. F1=recall (particle name→class), "
        "F2=rule application (spin number). Mechanistically interesting if F2 accuracy > F1 "
        "(rule generalises beyond memorisation). Type 2 candidate-set."
    ),
    "physics_approximation_regime_mini": (
        "Rule: v/c → 0 = classical; v/c → 1 = relativistic. Threshold-filter on continuous input. "
        "Model must represent v/c magnitude and apply a threshold. Mechanistically analogous to "
        "other Type 2 behaviours (gate on scalar quantity)."
    ),
    "physics_e1_selection_mini": (
        "Rule: |Δl|=1 → allowed. Cross-shell only (no same-n ambiguity). Negative-control candidate: "
        "the full E1 run failed at 48.8%; this mini version tests whether cleaner prompts help. "
        "If still <70%, behaviour is unsuitable for this model."
    ),
}

GO_THRESHOLDS = {
    "sign_accuracy": 0.85,
    "mean_norm_diff": 1.0,
}

SCREENING_SCORE_WEIGHTS = {
    "sign_accuracy": 0.50,
    "mean_norm_diff_clipped": 0.35,
    "balance_score": 0.15,
}


# ─── evaluation ─────────────────────────────────────────────────────────────

def load_prompts(behaviour: str, split: str, prompt_dir: Path) -> list[dict]:
    path = prompt_dir / f"{behaviour}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts not found: {path}\n"
            f"Run: python scripts/32_generate_candidate_set_screening_prompts.py"
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_behaviour(
    model: ModelWrapper,
    prompts: list[dict],
    min_logit_diff: float = 0.5,
) -> pd.DataFrame:
    """Teacher-forced logprob evaluation — same logic as script 02."""
    records = []
    for item in tqdm(prompts, leave=False):
        prompt_text     = item["prompt"]
        correct_answer  = item["correct_answer"]
        incorrect_answer = item["incorrect_answer"]
        try:
            correct_lp, correct_len     = model.get_token_log_prob(prompt_text, correct_answer)
            incorrect_lp, incorrect_len = model.get_token_log_prob(prompt_text, incorrect_answer)

            logprob_diff            = correct_lp - incorrect_lp
            correct_lp_norm         = correct_lp   / max(correct_len,   1)
            incorrect_lp_norm       = incorrect_lp / max(incorrect_len, 1)
            logprob_diff_normalized = correct_lp_norm - incorrect_lp_norm
            success                 = logprob_diff_normalized > min_logit_diff

            records.append({
                **{k: v for k, v in item.items()
                   if k not in ("prompt", "correct_answer", "incorrect_answer")},
                "prompt": prompt_text,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "correct_log_prob": correct_lp,
                "incorrect_log_prob": incorrect_lp,
                "correct_token_len": correct_len,
                "incorrect_token_len": incorrect_len,
                "logprob_diff": logprob_diff,
                "logprob_diff_normalized": logprob_diff_normalized,
                "success": bool(success),
                "sign_correct": bool(logprob_diff_normalized > 0),
                "hard_correct": bool(success),
            })
        except Exception as exc:
            records.append({
                **{k: v for k, v in item.items()
                   if k not in ("prompt", "correct_answer", "incorrect_answer")},
                "prompt": prompt_text,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "correct_log_prob": np.nan,
                "incorrect_log_prob": np.nan,
                "correct_token_len": np.nan,
                "incorrect_token_len": np.nan,
                "logprob_diff": np.nan,
                "logprob_diff_normalized": np.nan,
                "success": False,
                "sign_correct": False,
                "hard_correct": False,
                "error": str(exc),
            })
    return pd.DataFrame(records)


# ─── scoring & reporting ─────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    valid = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    n_valid = len(valid)
    sign_acc = float((valid > 0).mean()) if n_valid else 0.0
    mean_norm = float(valid.mean()) if n_valid else 0.0
    median_norm = float(valid.median()) if n_valid else 0.0
    hard_acc = float(df["success"].mean()) if n else 0.0

    # Balance score: 1 - |2p - 1| where p = fraction answering correctly across the two classes
    if "correct_answer" in df.columns:
        n_pos = (df["correct_answer"].str.strip() == df["correct_answer"].str.strip().value_counts().index[0]).sum()
        balance = 1.0 - abs(2 * n_pos / n - 1) if n else 0.0
    else:
        balance = 1.0

    # Screening score (higher is better)
    norm_score = min(max(mean_norm, -5), 5) / 5.0
    screening_score = (
        SCREENING_SCORE_WEIGHTS["sign_accuracy"]       * sign_acc
        + SCREENING_SCORE_WEIGHTS["mean_norm_diff_clipped"] * max(norm_score, 0)
        + SCREENING_SCORE_WEIGHTS["balance_score"]     * balance
    )

    # Per-family breakdown
    family_accs = {}
    if "wording_family" in df.columns:
        for fam, sub in df.groupby("wording_family"):
            sub_valid = sub["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
            family_accs[fam] = float((sub_valid > 0).mean()) if len(sub_valid) else np.nan

    return {
        "n": n,
        "n_valid": n_valid,
        "sign_accuracy": sign_acc,
        "hard_accuracy": hard_acc,
        "mean_norm_diff": mean_norm,
        "median_norm_diff": median_norm,
        "screening_score": screening_score,
        "balance_score": balance,
        "family_accs": family_accs,
        "go": sign_acc >= GO_THRESHOLDS["sign_accuracy"] and mean_norm >= GO_THRESHOLDS["mean_norm_diff"],
    }


def verdict(metrics: dict) -> str:
    s = metrics["sign_accuracy"]
    if s >= 0.90:
        return "GO (strong)"
    if s >= 0.85:
        return "GO"
    if s >= 0.70:
        return "MARGINAL"
    if s >= 0.55:
        return "WEAK"
    return "FAIL (near chance)"


def build_report(
    all_metrics: dict[str, dict],
    all_dfs: dict[str, pd.DataFrame],
    timestamp: str,
) -> str:
    lines = [
        "# Candidate-Set Behaviour Screening Report",
        f"\nGenerated: {timestamp}",
        "",
        "## Ranking (by screening score)",
        "",
        "| Rank | Behaviour | Sign Acc | Hard Acc | Mean Norm Diff | Score | Verdict |",
        "|------|-----------|----------|----------|----------------|-------|---------|",
    ]

    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["screening_score"], reverse=True)
    for i, (beh, m) in enumerate(ranked, 1):
        v = verdict(m)
        lines.append(
            f"| {i} | `{beh}` | {m['sign_accuracy']:.1%} | {m['hard_accuracy']:.1%} "
            f"| {m['mean_norm_diff']:.3f} | {m['screening_score']:.3f} | **{v}** |"
        )

    lines += ["", "---", "", "## Per-behaviour detail", ""]
    for beh, m in ranked:
        v = verdict(m)
        lines.append(f"### `{beh}`")
        lines.append(f"**Verdict: {v}**  |  Screening score: {m['screening_score']:.3f}")
        lines.append("")
        lines.append(f"- Prompts: {m['n']} (valid: {m['n_valid']})")
        lines.append(f"- Sign accuracy: {m['sign_accuracy']:.1%}")
        lines.append(f"- Hard accuracy (norm diff > 0.5): {m['hard_accuracy']:.1%}")
        lines.append(f"- Mean normalised logprob diff: {m['mean_norm_diff']:.3f}")
        lines.append(f"- Median normalised logprob diff: {m['median_norm_diff']:.3f}")
        if m["family_accs"]:
            lines.append(f"- Family breakdown:")
            for fam, acc in sorted(m["family_accs"].items()):
                flag = " ← FAIL" if not np.isnan(acc) and acc < 0.70 else ""
                lines.append(f"  - {fam}: {acc:.1%}{flag}")
        lines.append("")
        lines.append(f"**Mechanistic notes:** {MECHANISTIC_NOTES.get(beh, '')}")
        lines.append("")

    # Recommendation
    winner = ranked[0][0] if ranked else None
    lines += ["---", "", "## Recommendation", ""]
    if winner and verdict(ranked[0][1]).startswith("GO"):
        lines.append(f"**Recommended for full pipeline: `{winner}`**")
        lines.append("")
        lines.append(
            f"Sign accuracy {ranked[0][1]['sign_accuracy']:.1%} meets the ≥85% threshold. "
            f"Proceed to full feature extraction and mechanistic analysis."
        )
    else:
        best = ranked[0][0] if ranked else "none"
        lines.append(
            f"No behaviour reached the ≥85% sign accuracy threshold. "
            f"Best candidate: `{best}` ({ranked[0][1]['sign_accuracy']:.1%} if ranked else 'N/A')."
        )
        lines.append("")
        lines.append(
            "Consider: (1) reviewing prompt wording, (2) trying a different model (BASE vs Instruct), "
            "(3) redesigning the weakest behaviour family, or (4) choosing a different domain."
        )

    return "\n".join(lines)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run candidate-set screening baseline evaluation",
    )
    parser.add_argument(
        "--behaviours", nargs="+", default=BEHAVIOURS_DEFAULT,
        help="Behaviours to evaluate (default: all 4 mini behaviours)",
    )
    parser.add_argument(
        "--split", default="train", choices=["train"],
        help="Split to evaluate (screening uses train only)",
    )
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--prompt_dir", default="data/prompts",
        help="Directory containing JSONL prompt files",
    )
    parser.add_argument(
        "--output_dir", default="data/results",
        help="Directory for output CSVs",
    )
    parser.add_argument(
        "--min_logit_diff", type=float, default=0.5,
        help="Threshold for hard_correct (default: 0.5)",
    )
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Load config for model settings
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("CANDIDATE-SET BEHAVIOUR SCREENING")
    print("=" * 70)
    print(f"\nBehaviours: {args.behaviours}")
    print(f"Timestamp:  {timestamp}\n")

    # Load model ONCE
    print("Loading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    print("Model loaded.\n")

    # Evaluate each behaviour
    all_metrics = {}
    all_dfs = {}

    for beh in args.behaviours:
        print(f"── {beh} ──────────────────────────────")
        try:
            prompts = load_prompts(beh, args.split, prompt_dir)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}\n")
            continue

        print(f"  Loaded {len(prompts)} prompts")
        df = evaluate_behaviour(model, prompts, min_logit_diff=args.min_logit_diff)

        # Save per-behaviour CSV (compatible with downstream pipeline scripts)
        csv_path = output_dir / f"baseline_{beh}_{args.split}.csv"
        df.to_csv(csv_path, index=False)

        metrics = compute_metrics(df)
        all_metrics[beh] = metrics
        all_dfs[beh] = df

        v = verdict(metrics)
        print(f"  Sign accuracy:  {metrics['sign_accuracy']:.1%}")
        print(f"  Hard accuracy:  {metrics['hard_accuracy']:.1%}")
        print(f"  Mean norm diff: {metrics['mean_norm_diff']:.3f}")
        print(f"  Verdict:        {v}")
        if metrics["family_accs"]:
            for fam, acc in sorted(metrics["family_accs"].items()):
                print(f"    {fam}: {acc:.1%}")
        print()

    if not all_metrics:
        print("[ERROR] No behaviours evaluated — check prompt files exist.")
        sys.exit(1)

    # Summary CSV
    summary_rows = []
    for beh, m in all_metrics.items():
        row = {
            "behaviour": beh,
            "n_prompts": m["n"],
            "sign_accuracy": m["sign_accuracy"],
            "hard_accuracy": m["hard_accuracy"],
            "mean_norm_diff": m["mean_norm_diff"],
            "median_norm_diff": m["median_norm_diff"],
            "screening_score": m["screening_score"],
            "verdict": verdict(m),
        }
        # Add per-family columns
        for fam, acc in m.get("family_accs", {}).items():
            row[f"acc_{fam}"] = acc
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("screening_score", ascending=False)
    summary_path = output_dir / "candidate_set_screening_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Markdown report
    report = build_report(all_metrics, all_dfs, timestamp)
    report_path = output_dir / "candidate_set_screening_report.md"
    report_path.write_text(report)
    print(f"Saved report:  {report_path}")

    # Print ranked table
    print("\n" + "=" * 70)
    print("SCREENING RESULTS (ranked)")
    print("=" * 70)
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["screening_score"], reverse=True)
    print(f"\n{'Rank':<5} {'Behaviour':<40} {'Sign':<8} {'Score':<8} {'Verdict'}")
    print("-" * 72)
    for i, (beh, m) in enumerate(ranked, 1):
        print(f"{i:<5} {beh:<40} {m['sign_accuracy']:>6.1%}  {m['screening_score']:>6.3f}  {verdict(m)}")

    winner = ranked[0][0]
    print(f"\nRecommended next behaviour: {winner}  ({verdict(ranked[0][1])})")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
