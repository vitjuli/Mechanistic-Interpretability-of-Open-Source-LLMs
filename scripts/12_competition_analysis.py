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

# Common function words per language (used in competition classification).
_FR_COMMON = {
    "le", "la", "les", "un", "une", "des", "de", "du", "est", "et", "en",
    "dans", "sur", "qui", "que", "pas", "ne", "plus", "très", "aussi",
    "mais", "ou", "donc", "car", "si", "c", "se", "lui", "ils", "elles",
}
_EN_COMMON = {
    "the", "a", "an", "is", "are", "was", "were", "of", "and", "or",
    "not", "in", "on", "at", "to", "for", "with", "it", "its", "that",
}


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (character level)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


def classify_competing_token(
    predicted_token: str,
    source_word: str,
    correct_token: str,
    language: str,
) -> str:
    """
    Classify the token that outcompetes the correct antonym into one of:
      "source_word_copy"           — model predicts the source word itself
      "semantic_neighbor_or_synonym" — semantically close to source/correct (edit dist ≤ 2)
      "morphological_variant"      — shares a 4-char stem with source or correct
      "high_frequency_unrelated"   — common function word, no semantic relation
      "other"                      — punctuation, template artifact, or uncategorised
    """
    pred       = predicted_token.strip()
    pred_lower = pred.lower()
    src_lower  = source_word.strip().lower()
    corr_lower = correct_token.strip().lower()

    # Punctuation / formatting / non-alphabetic (e.g., '"', ':', digits)
    if not any(c.isalpha() for c in pred) or len(pred) == 0:
        return "other"

    # Exact source word copy
    if pred_lower == src_lower:
        return "source_word_copy"

    # High-frequency function word (language-specific)
    common = _FR_COMMON if language == "fr" else _EN_COMMON
    if pred_lower in common:
        return "high_frequency_unrelated"

    # Morphological variant: shares first 4 characters with source or correct
    if len(pred_lower) >= 4 and len(src_lower) >= 4 and pred_lower[:4] == src_lower[:4]:
        return "morphological_variant"
    if len(pred_lower) >= 4 and len(corr_lower) >= 4 and pred_lower[:4] == corr_lower[:4]:
        return "morphological_variant"

    # Semantic neighbor / synonym: small edit distance to source or correct
    if len(pred_lower) > 3:
        if _edit_distance(pred_lower, src_lower) <= 2 or _edit_distance(pred_lower, corr_lower) <= 2:
            return "semantic_neighbor_or_synonym"

    return "other"


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

        # Fix 3: classify the competing token
        competition_class = classify_competing_token(
            comp["predicted_token"],
            p.get("word", ""),
            correct_token,
            language,
        )

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
            "competition_class":      competition_class,
        })

        rank_str = f"rank={comp['correct_rank']}"
        pred_str = repr(comp["predicted_token"])
        margin   = comp["margin_vs_argmax"]
        l13_str  = f"{l13_contrib:.3f}" if l13_contrib is not None else "N/A"
        l22_str  = f"{l22_mean:.3f}"    if l22_mean    is not None else "N/A"
        print(f"    correct {rank_str}, predicted {pred_str}, margin={margin:.3f}, "
              f"L13={l13_str}, L22={l22_str}")

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

    # Fix 3: competition class distribution
    class_counts: dict = {}
    for r in per_prompt:
        cc = r["competition_class"]
        class_counts[cc] = class_counts.get(cc, 0) + 1
    dominant_class = max(class_counts, key=class_counts.get) if class_counts else "unknown"
    # Token frequency for "other" sub-class (detect template artifacts)
    other_tokens = [r["predicted_token"] for r in per_prompt if r["competition_class"] == "other"]
    other_token_freq: dict = {}
    for tok in other_tokens:
        other_token_freq[tok] = other_token_freq.get(tok, 0) + 1

    # Concept-level: which concept has strongest competition (most negative mean margin)?
    concept_margins: dict = {}
    for r in per_prompt:
        ci = r["concept_index"]
        concept_margins.setdefault(ci, []).append(r["margin_vs_argmax"])
    concept_mean_margin = {
        ci: round(float(np.mean(v)), 4) for ci, v in concept_margins.items()
    }

    # Determine competition type from class distribution
    n_total = len(per_prompt)
    dominant_n = class_counts.get(dominant_class, 0)
    # Token frequency for all predicted tokens
    token_freq: dict = {}
    for tok in all_predicted:
        token_freq[tok] = token_freq.get(tok, 0) + 1
    most_common_token = max(token_freq, key=token_freq.get)

    if dominant_class == "source_word_copy" and dominant_n >= int(n_total * 0.7):
        competition_type = "identity_confusion"
        competition_description = (
            f"Source-word copy in {dominant_n}/{n_total} cases. "
            "Model prefers the source word over its antonym (identity/repetition bias)."
        )
    elif dominant_class == "other" and dominant_n >= int(n_total * 0.7):
        # Check if it's driven by a single template-artifact token
        if other_token_freq and max(other_token_freq.values()) >= int(n_total * 0.5):
            top_other = max(other_token_freq, key=other_token_freq.get)
            competition_type = "template_artifact"
            competition_description = (
                f"Dominant class 'other' in {dominant_n}/{n_total} cases, "
                f"driven by single token {repr(top_other)} "
                f"({other_token_freq[top_other]}/{n_total}). "
                "Model is completing a formatting template rather than predicting a semantic token."
            )
        else:
            competition_type = "other_heterogeneous"
            competition_description = (
                f"'Other' class dominates ({dominant_n}/{n_total}) but no single token "
                "drives it — heterogeneous formatting or rare-word predictions."
            )
    elif dominant_class == "high_frequency_unrelated" and dominant_n >= int(n_total * 0.5):
        competition_type = "frequency_bias"
        competition_description = (
            f"High-frequency function words in {dominant_n}/{n_total} cases. "
            "Model defaults to common words rather than computing the antonym."
        )
    elif dominant_class == "semantic_neighbor_or_synonym" and dominant_n >= int(n_total * 0.4):
        competition_type = "semantic_confusion"
        competition_description = (
            f"Semantic neighbors in {dominant_n}/{n_total} cases. "
            "Model retrieves related words instead of the correct antonym."
        )
    else:
        competition_type = "mixed"
        competition_description = (
            f"No dominant class. Distribution: {class_counts}. "
            f"Most common token: {repr(most_common_token)} ({token_freq[most_common_token]}/{n_total})."
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

    # Q1: is there a consistent competing token?
    if token_freq and max(token_freq.values()) >= int(n_total * 0.7):
        top_tok = most_common_token
        q1_answer = (
            f"YES — single consistent token: {repr(top_tok)} in "
            f"{token_freq[top_tok]}/{n_total} cases."
        )
    elif dominant_n >= int(n_total * 0.6):
        q1_answer = (
            f"PARTIAL — dominant class '{dominant_class}' in {dominant_n}/{n_total} cases "
            f"but multiple distinct tokens."
        )
    else:
        q1_answer = f"NO — no consistent pattern. Distribution: {class_counts}."

    summary = {
        "n_incorrect_prompts": n_total,
        "n_type_a": n_total - len(type_b_idx),
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
        # Fix 3: competition class distribution
        "competition_class_distribution": class_counts,
        "dominant_competition_class":     dominant_class,
        "other_token_frequency":          other_token_freq,
        "most_common_predicted_token":    most_common_token,
        "most_common_predicted_token_count": token_freq[most_common_token],
        # Required interpretations
        "Q1_consistent_competing_token":  q1_answer,
        "Q2_competition_type":            competition_type,
        "Q2_competition_description":     competition_description,
        "Q3_l22_corr_interpretation":     l22_corr_interpretation,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPETITION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Incorrect prompts:              {n_total}")
    print(f"Mean margin vs argmax:          {summary['mean_margin_vs_argmax']:.4f}")
    print(f"Predictions = source word:      {n_is_incorr}/{n_total}")
    print(f"Competition type:               {competition_type}")
    print(f"L13 vs margin correlation:      {summary['pearson_l13_vs_margin']}")
    print(f"L22 vs margin correlation:      {summary['pearson_l22_vs_margin']}")
    print(f"\nCompetition class distribution:")
    for cc, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cc:35s}: {cnt}/{n_total}")
    if other_token_freq:
        print(f"  'other' token breakdown: {other_token_freq}")
    print(f"\nQ1 — Consistent competing token? {q1_answer}")
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
