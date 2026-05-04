"""
Multi-candidate logprob evaluator for candidate-state analysis.

For each prompt, evaluates logprob of ALL particles in the global pool (not just
correct vs one incorrect). Saves full candidate distributions for competition analysis.

Outputs to data/results/candidate_set_large/:
  candidate_state_results.csv    — per-prompt results with full candidate logprobs
  candidate_state_summary.md     — summary report

Computed per-prompt metrics:
  rank_correct:     rank of correct answer by normalised logprob (1 = highest)
  margin:           normed_logprob(correct) - normed_logprob(best_incorrect)
  h1_signal:        logprob_diff normalised by token length (same as baseline scorer)
  entropy:          entropy over softmax of normed logprobs for in-set candidates
  sign_correct:     whether correct outranks primary incorrect (binary, backward compat.)

Usage:
  python scripts/39_evaluate_candidate_state.py
  python scripts/39_evaluate_candidate_state.py --device cuda --split train
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_utils import ModelWrapper

# ─── Global candidate pool ────────────────────────────────────────────────────
# Evaluate logprob of each token for every prompt.
# Includes all particles that appear as answer candidates in the dataset.
GLOBAL_POOL = [' electron', ' proton', ' neutron', ' photon', ' positron', ' muon']

# Token lengths (populated at runtime from token audit; used for normalization)
TOKEN_LENGTHS: dict[str, int] = {}


def token_audit(model: ModelWrapper) -> dict[str, int]:
    lengths = {}
    print("  Token lengths in global pool:")
    for t in GLOBAL_POOL:
        ids = model.tokenizer.encode(t, add_special_tokens=False)
        lengths[t] = len(ids)
        status = "OK (1-tok)" if len(ids) == 1 else f"MULTI ({len(ids)} toks)"
        print(f"    {repr(t)}: {status}")
    return lengths


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_prompt(
    model: ModelWrapper,
    prompt: str,
    correct: str,
    incorrect: str,
    tok_lengths: dict[str, int],
    min_logit_diff: float = 0.5,
) -> dict:
    """
    Evaluate one prompt:
    1. Compute logprob for all tokens in GLOBAL_POOL
    2. Compute binary metrics (correct vs incorrect) for backward compat
    3. Compute competition metrics over in-set candidates
    """
    # ── Evaluate all pool tokens in one forward pass ──────────────────────────
    log_probs_t, token_lengths_t = model.get_sequence_log_probs(
        [prompt],
        target_sequences=GLOBAL_POOL,
    )
    # log_probs_t shape: (1, len(GLOBAL_POOL))
    raw_lps = {t: float(log_probs_t[0, i]) for i, t in enumerate(GLOBAL_POOL)}
    pool_lengths = {t: int(token_lengths_t[i]) for i, t in enumerate(GLOBAL_POOL)}

    # Normalised logprobs (per token)
    normed_lps = {t: raw_lps[t] / max(pool_lengths[t], 1) for t in GLOBAL_POOL}

    # ── Binary metrics ─────────────────────────────────────────────────────────
    c_lp  = raw_lps.get(correct, np.nan)
    ic_lp = raw_lps.get(incorrect, np.nan)
    c_len  = pool_lengths.get(correct, 1)
    ic_len = pool_lengths.get(incorrect, 1)

    if not np.isnan(c_lp) and not np.isnan(ic_lp):
        logprob_diff = c_lp - ic_lp
        logprob_diff_normalized = c_lp / c_len - ic_lp / ic_len
        sign_correct  = bool(logprob_diff_normalized > 0)
        hard_correct  = bool(logprob_diff_normalized > min_logit_diff)
    else:
        logprob_diff = logprob_diff_normalized = np.nan
        sign_correct = hard_correct = False

    # ── Competition metrics over all pool tokens ───────────────────────────────
    pool_normed = list(normed_lps.values())
    ranked = sorted(GLOBAL_POOL, key=lambda t: normed_lps[t], reverse=True)
    rank_correct = ranked.index(correct) + 1 if correct in ranked else None
    best_incorrect_normed = max(normed_lps[t] for t in GLOBAL_POOL if t != correct)
    margin = normed_lps.get(correct, np.nan) - best_incorrect_normed

    # Entropy of softmax over pool normed logprobs
    lps_arr = np.array(pool_normed, dtype=np.float64)
    lps_arr -= lps_arr.max()  # numerical stability
    probs = np.exp(lps_arr) / np.exp(lps_arr).sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    result = {
        # Binary scoring (backward compat)
        "correct_log_prob": c_lp,
        "incorrect_log_prob": ic_lp,
        "correct_token_len": c_len,
        "incorrect_token_len": ic_len,
        "logprob_diff": logprob_diff,
        "logprob_diff_normalized": logprob_diff_normalized,
        "sign_correct": sign_correct,
        "hard_correct": hard_correct,
        "success": hard_correct,
        # Competition metrics
        "rank_correct": rank_correct,
        "margin": margin,
        "entropy_pool": entropy,
        # Per-token raw logprobs (for analysis)
        **{f"lp{t.strip()}": raw_lps[t] for t in GLOBAL_POOL},
        **{f"nlp{t.strip()}": normed_lps[t] for t in GLOBAL_POOL},
    }
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-candidate logprob evaluator")
    parser.add_argument("--behaviour",   default="physics_particle_candidate_selection")
    parser.add_argument("--split",       default="train", choices=["train"])
    parser.add_argument("--config",      default="configs/experiment_config.yaml")
    parser.add_argument("--prompt_dir",  default="data/prompts")
    parser.add_argument("--output_dir",  default="data/results/candidate_set_large")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--min_logit_diff", type=float, default=0.5)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # ── Load prompts ────────────────────────────────────────────────────────
    prompt_path = Path(args.prompt_dir) / f"{args.behaviour}_{args.split}.jsonl"
    if not prompt_path.exists():
        print(f"[ERROR] Prompts not found: {prompt_path}")
        print(f"  Run: python scripts/38_generate_candidate_state_large.py")
        sys.exit(1)
    prompts = [json.loads(l) for l in prompt_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompt_path}")

    # ── Load model ──────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = config["model"]["device"]
    dtype = "bfloat16" if device == "cuda" else config["model"]["dtype"]

    print(f"\nLoading model ({device}, {dtype})...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=dtype,
        device=device,
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    print("Model loaded.\n")

    # ── Token audit ─────────────────────────────────────────────────────────
    print("Token audit:")
    tok_lengths = token_audit(model)
    print()

    # ── Evaluate ────────────────────────────────────────────────────────────
    records = []
    n_errors = 0
    for item in tqdm(prompts, desc="Evaluating"):
        prompt_text = item["prompt"]
        correct     = item["correct_answer"]
        incorrect   = item["incorrect_answer"]
        try:
            metrics = evaluate_prompt(
                model, prompt_text, correct, incorrect, tok_lengths,
                min_logit_diff=args.min_logit_diff,
            )
            records.append({
                **{k: v for k, v in item.items()
                   if k not in ("correct_answer", "incorrect_answer", "incorrect_answers")},
                "prompt": prompt_text,
                "correct_answer": correct,
                "incorrect_answer": incorrect,
                **metrics,
            })
        except Exception as exc:
            n_errors += 1
            if n_errors <= 3:
                print(f"\n  [ERROR] {exc!r}")
            records.append({
                **{k: v for k, v in item.items()
                   if k not in ("correct_answer", "incorrect_answer", "incorrect_answers")},
                "prompt": prompt_text,
                "correct_answer": correct,
                "incorrect_answer": incorrect,
                "logprob_diff": np.nan, "logprob_diff_normalized": np.nan,
                "sign_correct": False, "hard_correct": False, "success": False,
                "rank_correct": None, "margin": np.nan, "entropy_pool": np.nan,
                "error": str(exc),
            })

    if n_errors:
        print(f"\n[WARN] {n_errors}/{len(prompts)} prompts errored")

    df = pd.DataFrame(records)
    csv_path = out_dir / "candidate_state_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df)} rows)")
    _print_summary(df, out_dir, timestamp)


def _print_summary(df: pd.DataFrame, out_dir: Path, timestamp: str):
    valid = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
    sign_acc = float((valid > 0).mean()) if len(valid) else 0.0
    hard_acc = float(df["success"].astype(bool).mean())
    mean_nd  = float(valid.mean()) if len(valid) else 0.0
    mean_rank = float(df["rank_correct"].dropna().mean()) if "rank_correct" in df.columns else None
    mean_margin = float(df["margin"].replace([np.inf,-np.inf],np.nan).dropna().mean())

    lines = [
        "# Candidate-State Evaluation Summary",
        f"\nGenerated: {timestamp}",
        f"\n## Overall",
        f"- n_prompts: {len(df)}",
        f"- sign_accuracy: {sign_acc:.1%}",
        f"- hard_accuracy: {hard_acc:.1%}",
        f"- mean_norm_diff: {mean_nd:.3f}",
        f"- mean_rank_correct: {mean_rank:.2f}" if mean_rank else "",
        f"- mean_margin: {mean_margin:.3f}",
        "",
    ]

    for col, label in [("experiment_type","Experiment type"),
                        ("wording_family","Wording family"),
                        ("filter_property","Filter"),
                        ("target_candidate","Target"),
                        ("distractor_difficulty","Distractor difficulty"),
                        ("variant_type","Variant type"),
                        ("n_candidates","Set size")]:
        if col not in df.columns:
            continue
        lines.append(f"## By {label}")
        lines.append("")
        lines.append("| Group | Sign Acc | Hard Acc | Mean ND | Mean Rank | N |")
        lines.append("|-------|----------|----------|---------|-----------|---|")
        for key, sub in df.groupby(col):
            sv = sub["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
            sa  = float((sv > 0).mean()) if len(sv) else 0.0
            ha  = float(sub["success"].astype(bool).mean())
            nd  = float(sv.mean()) if len(sv) else 0.0
            rk  = float(sub["rank_correct"].dropna().mean()) if "rank_correct" in sub.columns and sub["rank_correct"].notna().any() else float('nan')
            lines.append(f"| {key} | {sa:.1%} | {ha:.1%} | {nd:.2f} | {rk:.1f} | {len(sub)} |")
        lines.append("")

    # H1 vs H2 test: counterfactual comparison
    if "variant_type" in df.columns and "filter_correct_id" in df.columns:
        lines.append("## H1 vs H2 test (counterfactual)")
        lines.append("")
        lines.append("Comparison: logprob_diff_normalized for 'original' vs 'no_set' variants.")
        lines.append("H1 prediction: same diff. H2 prediction: diff changes when set is removed.")
        lines.append("")
        lines.append("| filter_correct_id | original_nd | no_set_nd | delta | n_pairs |")
        lines.append("|------------------|-------------|-----------|-------|---------|")
        cf = df[df["experiment_type"] == "counterfactual"]
        for fcid, sub in cf.groupby("filter_correct_id"):
            orig = sub[sub["variant_type"] == "original"]["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
            nos  = sub[sub["variant_type"] == "no_set"]["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
            if len(orig) and len(nos):
                o_nd = float(orig.mean())
                n_nd = float(nos.mean())
                delta = o_nd - n_nd
                lines.append(f"| {fcid} | {o_nd:.3f} | {n_nd:.3f} | {delta:+.3f} | {min(len(orig),len(nos))} |")
        lines.append("")

    # Distractor sensitivity
    if "distractor_difficulty" in df.columns and "filter_correct_id" in df.columns:
        lines.append("## Distractor sensitivity")
        lines.append("")
        lines.append("Prediction: mean_norm_diff decreases for harder distractors (H2 signal).")
        lines.append("")
        lines.append("| filter_correct_id | difficulty | mean_nd | sign_acc | n |")
        lines.append("|------------------|------------|---------|----------|---|")
        ds = df[df["experiment_type"] == "distractor_sensitivity"]
        order = ["trivial", "easy", "medium", "hard", "hardest"]
        for fcid, sub_fc in ds.groupby("filter_correct_id"):
            for diff in order:
                sub = sub_fc[sub_fc["distractor_difficulty"] == diff]
                if len(sub) == 0:
                    continue
                sv = sub["logprob_diff_normalized"].replace([np.inf,-np.inf],np.nan).dropna()
                nd = float(sv.mean()) if len(sv) else 0.0
                sa = float((sv > 0).mean()) if len(sv) else 0.0
                lines.append(f"| {fcid} | {diff} | {nd:.3f} | {sa:.1%} | {len(sub)} |")
        lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "candidate_state_summary.md"
    report_path.write_text(report)
    print(f"Saved: {report_path}")

    # Also print key stats
    rank_str   = f"{mean_rank:.2f}"   if mean_rank   is not None else "n/a"
    margin_str = f"{mean_margin:.3f}" if mean_margin is not None else "n/a"
    print(f"\nOverall: sign_acc={sign_acc:.1%}  hard_acc={hard_acc:.1%}  "
          f"mean_nd={mean_nd:.3f}  mean_rank={rank_str}  mean_margin={margin_str}")


if __name__ == "__main__":
    main()
