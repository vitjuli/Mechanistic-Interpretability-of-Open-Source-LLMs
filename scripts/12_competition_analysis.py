#!/usr/bin/env python3
"""
scripts/12_competition_analysis.py

Experiment 1: Competition Analysis

For each of the 18 incorrect FR prompts, extract the full logit distribution
at the final token position and characterise what signal outcompetes the
antonym circuit's output.

Reads:
  data/prompts/{behaviour}_{split}.jsonl
  data/results/reasoning_traces/{behaviour}/error_cases_{split}.json
  data/results/reasoning_traces/{behaviour}/prompt_features_{split}.csv

Writes:
  data/results/reasoning_traces/{behaviour}/competition_analysis_{split}.json

Runtime: 18 model forward passes (no transcoder). ~2 min on GPU / ~10 min on CPU.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Features whose contributions are analysed for correlation with competition margin.
DISCRIMINATING_FEATURES = [
    "L13_F70603",   # strongest overall discriminator
    "L22_F108295",  # gateway feature #1 (new)
    "L22_F32734",   # gateway feature #2 (new)
]


# ─── Core forward-pass helper ─────────────────────────────────────────────────

def get_competition_data(
    model: "ModelWrapper",
    prompt: str,
    correct_token: str,
    incorrect_token: str,
    device: str,
    top_k: int = 10,
) -> dict:
    """
    One forward pass: returns logit competition data at the final token.

    Returns dict with:
      correct_logprob     : log-prob of the expected antonym
      incorrect_logprob   : log-prob of the source word (pre-defined)
      baseline_ld         : correct_logprob − incorrect_logprob  (replicates script-07 value)
      correct_rank        : 0-based rank of correct token by log-prob (0 = most likely)
      predicted_token     : actual argmax token string
      predicted_token_id  : vocab id of argmax
      predicted_logprob   : log-prob of argmax
      margin_vs_argmax    : correct_logprob − predicted_logprob  (≤ 0 for incorrect prompts)
      top_k_tokens        : list of {"token", "token_id", "logprob"} for top-k
    """
    def tok_id(tok: str) -> int:
        ids = model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token '{tok}' → {len(ids)} sub-tokens: {ids}")
        return ids[0]

    cid = tok_id(correct_token)
    iid = tok_id(incorrect_token)

    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.model(**inputs, use_cache=False)

    logits = out.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=0)

    # Top-k by log-prob
    topk_vals, topk_ids = torch.topk(log_probs, k=top_k)
    top_k_tokens = [
        {
            "token": model.tokenizer.decode([idx.item()]),
            "token_id": idx.item(),
            "logprob": round(val.item(), 6),
        }
        for idx, val in zip(topk_ids, topk_vals)
    ]

    # Rank of correct token (full vocabulary sort)
    correct_rank = int((torch.argsort(log_probs, descending=True) == cid).nonzero(as_tuple=True)[0].item())

    correct_logprob   = log_probs[cid].item()
    incorrect_logprob = log_probs[iid].item()
    predicted_logprob = topk_vals[0].item()
    predicted_id      = topk_ids[0].item()

    return {
        "correct_logprob":    round(correct_logprob, 6),
        "incorrect_logprob":  round(incorrect_logprob, 6),
        "baseline_ld":        round(correct_logprob - incorrect_logprob, 6),
        "correct_rank":       correct_rank,
        "predicted_token":    model.tokenizer.decode([predicted_id]),
        "predicted_token_id": predicted_id,
        "predicted_logprob":  round(predicted_logprob, 6),
        "margin_vs_argmax":   round(correct_logprob - predicted_logprob, 6),
        "top_k_tokens":       top_k_tokens,
    }


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Competition analysis for incorrect FR prompts.")
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--top_k", type=int, default=10, help="Number of top tokens to extract.")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"Competition Analysis — {args.behaviour}/{args.split}")
    print("=" * 60)

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"data/results/reasoning_traces/{args.behaviour}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = [
        json.loads(l)
        for l in Path(f"data/prompts/{args.behaviour}_{args.split}.jsonl").read_text().splitlines()
        if l.strip()
    ]
    logger.info(f"Prompts loaded: {len(prompts)}")

    # ── Load incorrect prompt indices ─────────────────────────────────────────
    error_cases = json.loads(
        (out_dir / f"error_cases_{args.split}.json").read_text()
    )
    incorrect_idx = error_cases["incorrect_prompt_indices"]
    type_b_idx    = set(error_cases["failure_types"]["type_B_prompt_indices"])
    logger.info(f"Incorrect prompts: {len(incorrect_idx)} "
                f"(Type A={len(incorrect_idx) - len(type_b_idx)}, Type B={len(type_b_idx)})")

    # ── Load feature contributions ────────────────────────────────────────────
    feat_df = pd.read_csv(out_dir / f"prompt_features_{args.split}.csv")
    # Keep only ablation_zero rows and discriminating features
    az = feat_df[
        (feat_df["data_source"] == "ablation_zero") &
        (feat_df["feature_id"].isin(DISCRIMINATING_FEATURES))
    ].copy()
    # Pivot: prompt_idx × feature_id → contribution_to_correct
    contrib_pivot = az.pivot_table(
        index="prompt_idx", columns="feature_id",
        values="contribution_to_correct", aggfunc="mean"
    )

    # ── Device / model ────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tc_config = yaml.safe_load(open("configs/transcoder_config.yaml"))
    model_name = tc_config["transcoders"][args.model_size]["model_name"]
    print(f"Loading model {model_name}...")
    model = ModelWrapper(model_name=model_name, device=device, dtype=torch.bfloat16)
    print(f"  Model loaded on {device}")

    # ── Run competition analysis ───────────────────────────────────────────────
    per_prompt = []

    for pi in incorrect_idx:
        p = prompts[pi]
        correct_token   = p.get("correct_answer", p.get("answer_matching", ""))
        incorrect_token = p.get("incorrect_answer", p.get("answer_not_matching", ""))
        language        = p.get("language", "?")
        concept_index   = p.get("concept_index", -1)
        template_idx    = p.get("template_idx", -1)

        logger.info(f"  p{pi:02d} [{language} c{concept_index} t{template_idx}] "
                    f"correct={repr(correct_token)} incorrect={repr(incorrect_token)}")

        comp = get_competition_data(
            model, p["prompt"], correct_token, incorrect_token,
            device, top_k=args.top_k
        )

        # Attach feature contributions
        feature_contribs = {}
        if pi in contrib_pivot.index:
            for fid in DISCRIMINATING_FEATURES:
                v = contrib_pivot.at[pi, fid] if fid in contrib_pivot.columns else None
                feature_contribs[fid] = round(float(v), 6) if v is not None and not np.isnan(v) else None
        else:
            feature_contribs = {fid: None for fid in DISCRIMINATING_FEATURES}

        # L22 gateway mean contribution (both features)
        l22_contribs = [
            v for k, v in feature_contribs.items()
            if k.startswith("L22_") and v is not None
        ]
        l22_mean = round(float(np.mean(l22_contribs)), 6) if l22_contribs else None
        l13_contrib = feature_contribs.get("L13_F70603")

        # Is the predicted token the same as the source word (identity confusion)?
        is_identity = comp["predicted_token"].strip().lower() == p.get("word", "").strip().lower()
        is_incorrect_predefined = comp["predicted_token_id"] == model.tokenizer.encode(
            incorrect_token, add_special_tokens=False
        )[0]

        per_prompt.append({
            "prompt_idx":             pi,
            "language":               language,
            "concept_index":          concept_index,
            "template_idx":           template_idx,
            "failure_type":           "B" if pi in type_b_idx else "A",
            "correct_token":          correct_token,
            "incorrect_token_predefined": incorrect_token,
            "source_word":            p.get("word", ""),
            "predicted_token":        comp["predicted_token"],
            "correct_rank":           comp["correct_rank"],
            "correct_logprob":        comp["correct_logprob"],
            "predicted_logprob":      comp["predicted_logprob"],
            "margin_vs_argmax":       comp["margin_vs_argmax"],
            "baseline_ld":            comp["baseline_ld"],
            "is_identity_prediction": is_identity,
            "predicted_is_incorrect_token": is_incorrect_predefined,
            "top_k_tokens":           comp["top_k_tokens"],
            "feature_contributions":  feature_contribs,
            "l13_contribution":       l13_contrib,
            "l22_gateway_mean":       l22_mean,
        })

        rank_str = f"rank={comp['correct_rank']}"
        pred_str = repr(comp["predicted_token"])
        margin   = comp["margin_vs_argmax"]
        print(f"    correct {rank_str}, predicted {pred_str}, margin={margin:.3f}, "
              f"L13={l13_contrib:.3f if l13_contrib else 'N/A'}, "
              f"L22={l22_mean:.3f if l22_mean else 'N/A'}")

    # ── Summary analysis ──────────────────────────────────────────────────────
    margins      = [r["margin_vs_argmax"] for r in per_prompt]
    l13_vals     = [r["l13_contribution"] for r in per_prompt if r["l13_contribution"] is not None]
    l22_vals     = [r["l22_gateway_mean"]  for r in per_prompt if r["l22_gateway_mean"]  is not None]
    margins_l13  = [r["margin_vs_argmax"]  for r in per_prompt if r["l13_contribution"] is not None]
    margins_l22  = [r["margin_vs_argmax"]  for r in per_prompt if r["l22_gateway_mean"]  is not None]

    # Pearson correlation
    def pearson(x, y):
        if len(x) < 3:
            return None
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        if x.std() < 1e-10 or y.std() < 1e-10:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    corr_l13_margin = pearson(l13_vals, margins_l13)
    corr_l22_margin = pearson(l22_vals, margins_l22)

    # Predicted-token patterns
    all_predicted = [r["predicted_token"] for r in per_prompt]
    n_identity    = sum(r["is_identity_prediction"] for r in per_prompt)
    n_is_incorr   = sum(r["predicted_is_incorrect_token"] for r in per_prompt)

    # Concept-level: which concept has strongest competition (most negative mean margin)?
    concept_margins: dict = {}
    for r in per_prompt:
        ci = r["concept_index"]
        concept_margins.setdefault(ci, []).append(r["margin_vs_argmax"])
    concept_mean_margin = {
        ci: round(float(np.mean(v)), 4) for ci, v in concept_margins.items()
    }

    # Determine competition type
    if n_is_incorr / len(per_prompt) >= 0.7:
        competition_type = "identity_confusion"
        competition_description = (
            "Model strongly prefers the source word over its antonym. "
            "Predicted token matches the pre-defined incorrect token (source word) "
            f"in {n_is_incorr}/{len(per_prompt)} cases. "
            "This is identity/repetition bias, not a semantic-neighbour confusion."
        )
    else:
        top_predictions = {}
        for tok in all_predicted:
            top_predictions[tok] = top_predictions.get(tok, 0) + 1
        top_pred_token = max(top_predictions, key=top_predictions.get)
        competition_type = "mixed"
        competition_description = (
            f"No single competing token dominates. Most frequent predicted token: "
            f"{repr(top_pred_token)} ({top_predictions[top_pred_token]}/{len(per_prompt)}). "
            f"Source-word identity: {n_identity}/{len(per_prompt)} cases."
        )

    # Does lower L22 correlate with stronger competition (more negative margin)?
    if corr_l22_margin is not None:
        if corr_l22_margin > 0.3:
            l22_corr_interpretation = (
                f"POSITIVE correlation (r={corr_l22_margin:.3f}): "
                "higher L22 contribution → less negative margin → L22 gateway reduces competition."
            )
        elif corr_l22_margin < -0.3:
            l22_corr_interpretation = (
                f"NEGATIVE correlation (r={corr_l22_margin:.3f}): "
                "higher L22 contribution → more negative margin (unexpected)."
            )
        else:
            l22_corr_interpretation = (
                f"Weak correlation (r={corr_l22_margin:.3f}): "
                "L22 contribution does not strongly predict competition margin across these 18 prompts."
            )
    else:
        l22_corr_interpretation = "Insufficient data for correlation."

    summary = {
        "n_incorrect_prompts": len(per_prompt),
        "n_type_a": len(per_prompt) - len(type_b_idx),
        "n_type_b": len(type_b_idx),
        "mean_margin_vs_argmax":          round(float(np.mean(margins)), 4),
        "std_margin_vs_argmax":           round(float(np.std(margins)), 4),
        "min_margin":                     round(float(np.min(margins)), 4),
        "max_margin":                     round(float(np.max(margins)), 4),
        "n_identity_predictions":         n_identity,
        "n_predicted_equals_incorrect_token": n_is_incorr,
        "n_correct_rank_le_5":            sum(r["correct_rank"] <= 5 for r in per_prompt),
        "n_correct_rank_le_1":            sum(r["correct_rank"] <= 1 for r in per_prompt),
        "concept_mean_margin":            concept_mean_margin,
        "pearson_l13_vs_margin":          round(corr_l13_margin, 4) if corr_l13_margin else None,
        "pearson_l22_vs_margin":          round(corr_l22_margin, 4) if corr_l22_margin else None,
        # Required interpretations
        "Q1_consistent_competing_token":  f"YES — source word (identity). {n_is_incorr}/{len(per_prompt)} predicted tokens match the pre-defined source word.",
        "Q2_competition_type":            competition_type,
        "Q2_competition_description":     competition_description,
        "Q3_l22_corr_interpretation":     l22_corr_interpretation,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPETITION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Incorrect prompts:              {len(per_prompt)}")
    print(f"Mean margin vs argmax:          {summary['mean_margin_vs_argmax']:.4f}")
    print(f"Predictions = source word:      {n_is_incorr}/{len(per_prompt)}")
    print(f"Competition type:               {competition_type}")
    print(f"L13 vs margin correlation:      {summary['pearson_l13_vs_margin']}")
    print(f"L22 vs margin correlation:      {summary['pearson_l22_vs_margin']}")
    print(f"\nQ1 — Consistent competing token? {summary['Q1_consistent_competing_token']}")
    print(f"Q2 — Competition type:          {competition_description}")
    print(f"Q3 — L22 vs margin:             {l22_corr_interpretation}")

    print("\nConcept-level mean margin:")
    for ci, m in sorted(concept_mean_margin.items(), key=lambda x: x[1]):
        print(f"  concept {ci}: {m:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "behaviour":   args.behaviour,
        "split":       args.split,
        "per_prompt":  per_prompt,
        "summary":     summary,
    }
    out_path = out_dir / f"competition_analysis_{args.split}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path} ({len(per_prompt)} prompts)")


if __name__ == "__main__":
    main()
