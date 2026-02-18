#!/usr/bin/env python3
"""
Compute baseline margins for all prompts without intervention.

This script computes the logit difference (margin) between correct and incorrect
answer tokens for a given behavior dataset. Results are used to stratify prompts
into low-margin subsets for intervention experiments.

Usage:
    python scripts/07a_compute_baselines.py \
        --behaviour grammar_agreement \
        --split train \
        --n_prompts 80 \
        --output data/results/baselines/
"""

import sys
from pathlib import Path

# Add project root to Python path (HPC compatibility)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm

from src.model_utils import ModelWrapper

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def get_answer_tokens(prompt_data: dict) -> tuple:
    """
    Extract answer tokens from prompt data.
    
    Handles multiple field name variations.
    
    Returns:
        (correct_token, incorrect_token)
    """
    # Try different field names
    if 'correct_answer' in prompt_data and 'incorrect_answer' in prompt_data:
        return prompt_data['correct_answer'], prompt_data['incorrect_answer']
    elif 'answer_matching' in prompt_data and 'answer_not_matching' in prompt_data:
        return prompt_data['answer_matching'], prompt_data['answer_not_matching']
    elif 'correct' in prompt_data and 'incorrect' in prompt_data:
        return prompt_data['correct'], prompt_data['incorrect']
    else:
        raise ValueError(f"Could not find answer tokens in prompt data. Keys: {prompt_data.keys()}")


def load_jsonl(path: Path) -> list:
    """Load JSONL file (one JSON object per line)."""
    import json
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_single_token(tokenizer, token_str: str) -> int:
    """
    Ensure token is single-token and return its ID.
    
    Raises ValueError if not single-token.
    """
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Answer must be single-token: {token_str!r} -> {ids}")
    return ids[0]


def compute_logit_diff(
    model,
    device: torch.device,
    prompt: str,
    correct_token: str,
    incorrect_token: str
) -> float:
    """
    Compute log-prob margin: log_softmax(correct) - log_softmax(incorrect).

    Uses log_softmax to match 07_run_interventions.py exactly.
    Raw logit diff and log-prob diff have different scales; using the same
    metric here ensures stratification thresholds are meaningful in 07.

    Args:
        model: ModelWrapper instance
        device: Device to use
        prompt: Input text
        correct_token: Correct answer token (must be single-token)
        incorrect_token: Incorrect answer token (must be single-token)

    Returns:
        Log-prob margin. Positive = model prefers correct.
    """
    inputs = model.tokenize([prompt])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    correct_id = ensure_single_token(model.tokenizer, correct_token)
    incorrect_id = ensure_single_token(model.tokenizer, incorrect_token)

    with torch.no_grad():
        outputs = model.model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :]  # Last token logits

    # Fix: use log_softmax to match 07_run_interventions.py
    log_probs = torch.log_softmax(logits, dim=0)
    margin = (log_probs[correct_id] - log_probs[incorrect_id]).item()

    return margin


def compute_baselines(args):
    """Main function to compute baseline margins."""
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts (JSONL format!)
    prompts_path = Path(f"data/prompts/{args.behaviour}_{args.split}.jsonl")
    logger.info(f"Loading prompts from {prompts_path}")
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    all_prompts = load_jsonl(prompts_path)
    
    # Subsample if requested, but PRESERVE original indices
    if args.n_prompts and args.n_prompts < len(all_prompts):
        import random
        random.seed(42)
        # Store original index before sampling
        for i, p in enumerate(all_prompts):
            p['orig_idx'] = i
        all_prompts = random.sample(all_prompts, args.n_prompts)
    else:
        # Add orig_idx to all
        for i, p in enumerate(all_prompts):
            p['orig_idx'] = i
    
    logger.info(f"Computing baselines for {len(all_prompts)} prompts")
    
    # Load model
    logger.info("Loading language model...")
    model = ModelWrapper(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code
    )
    model.model.eval()
    
    # Get device
    try:
        device = next(model.model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    logger.info(f"Model loaded on device: {device}")
    
    # Compute margins
    results = []
    
    for prompt_data in tqdm(all_prompts, desc="Computing margins"):
        prompt = prompt_data["prompt"]
        
        # Get answer tokens
        correct, incorrect = get_answer_tokens(prompt_data)
        
        try:
            # Compute margin
            margin = compute_logit_diff(model, device, prompt, correct, incorrect)
            
            # Store result with all important fields
            result = {
                'orig_idx': prompt_data['orig_idx'],  # CRITICAL: preserve original index
                'prompt': prompt,
                'margin': margin,
                'correct_token': correct,
                'incorrect_token': incorrect,
            }
            
            # Use correct field name: 'number' (not 'subject_num')
            if 'number' in prompt_data:
                result['number'] = prompt_data['number']
            elif 'subject_num' in prompt_data:
                result['number'] = prompt_data['subject_num']
            else:
                result['number'] = 'unknown'
            
            # Add any other metadata
            for key in ['label', 'template', 'subject']:
                if key in prompt_data:
                    result[key] = prompt_data[key]
            
            # Reproducibility metadata
            result['baseline_seed'] = 42
            result['baseline_n_requested'] = args.n_prompts if args.n_prompts else len(all_prompts)

            results.append(result)
        
        except ValueError as e:
            logger.warning(f"Skipping prompt due to multi-token answer: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)

    # Generate output filename
    output_file = output_dir / f"baselines_{args.behaviour}_{args.split}_n{len(results)}.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Saved {len(results)} baseline margins to {output_file}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("BASELINE STATISTICS  (metric: log_softmax margin)")
    logger.info("="*60)
    logger.info(f"Total prompts: {len(df)}")
    logger.info(f"Mean margin:   {df['margin'].mean():.4f}")
    logger.info(f"Median margin: {df['margin'].median():.4f}")
    logger.info(f"Std margin:    {df['margin'].std():.4f}")
    logger.info(f"Min margin:    {df['margin'].min():.4f}")
    logger.info(f"Max margin:    {df['margin'].max():.4f}")

    # Quantile summary — use these to set thresholds in 07b!
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_vals = df['margin'].quantile(qs)
    logger.info("\nQuantiles (use for --low_max / --high_min in 07b):")
    for q, v in zip(qs, q_vals):
        logger.info(f"  q{int(q*100):02d}: {v:.4f}")
    logger.info("  Suggested: --low_max q25  --high_min q75")
    logger.info(f"  => --low_max {q_vals[0.25]:.3f} --high_min {q_vals[0.75]:.3f}")

    # Negative / positive split
    negative = df[df['margin'] <= 0]
    positive = df[df['margin'] > 0]
    logger.info(f"\nNegative margin (model wrong): {len(negative)} ({len(negative)/len(df)*100:.1f}%)")
    logger.info(f"Positive margin (model correct): {len(positive)} ({len(positive)/len(df)*100:.1f}%)")

    # Number distribution
    if 'number' in df.columns and df['number'].nunique() > 1:
        logger.info("\nNumber distribution:")
        for num in df['number'].unique():
            if num != 'unknown':
                count = len(df[df['number'] == num])
                logger.info(f"  {num}: {count} ({count/len(df)*100:.1f}%)")

    logger.info("="*60)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Compute baseline margins for prompts")

    parser.add_argument('--behaviour', type=str, default='grammar_agreement',
                        help='Behavior name')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Data split')
    parser.add_argument('--n_prompts', type=int, default=None,
                        help='Number of prompts to use (default: all)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name. If not set, read from configs/transcoder_config.yaml '
                             '(same source as 07_run_interventions.py).')
    parser.add_argument('--model_size', type=str, default='4b',
                        help='Model size key in transcoder_config.yaml (default: 4b)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Model dtype (default: bfloat16)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (default: auto)')
    parser.add_argument('--trust_remote_code', action='store_true',
                        help='Trust remote code for model loading')
    parser.add_argument('--output', type=str, default='data/results/baselines/',
                        help='Output directory')

    args = parser.parse_args()

    # Resolve model name: prefer explicit arg, else read from transcoder_config.yaml
    # (same file used by 07_run_interventions.py — guarantees model consistency)
    if args.model_name is None:
        import yaml
        tc_config_path = project_root / "configs" / "transcoder_config.yaml"
        if not tc_config_path.exists():
            raise FileNotFoundError(
                f"transcoder_config.yaml not found at {tc_config_path}. "
                f"Pass --model_name explicitly to bypass."
            )
        with open(tc_config_path) as fh:
            tc_cfg = yaml.safe_load(fh)
        # Strict guard: raise rather than silently fall back to wrong model
        if "transcoders" not in tc_cfg:
            raise KeyError(
                f"'transcoders' key missing in {tc_config_path}. "
                f"Pass --model_name explicitly."
            )
        if args.model_size not in tc_cfg["transcoders"]:
            available = list(tc_cfg["transcoders"].keys())
            raise KeyError(
                f"model_size '{args.model_size}' not found in transcoder_config.yaml. "
                f"Available: {available}. Use --model_size to select the right one."
            )
        tc_entry = tc_cfg["transcoders"][args.model_size]
        if "model_name" not in tc_entry:
            raise KeyError(
                f"'model_name' missing for size '{args.model_size}' in transcoder_config.yaml. "
                f"Pass --model_name explicitly."
            )
        args.model_name = tc_entry["model_name"]
        repo_id = tc_entry.get("repo_id", "<unknown>")
        logger.info(f"Model name from transcoder_config.yaml [{args.model_size}]: {args.model_name}")
        logger.info(f"Transcoder repo_id: {repo_id}  (verify this matches 07_run_interventions.py)")
    
    logger.info("="*60)
    logger.info("BASELINE MARGIN COMPUTATION")
    logger.info("="*60)
    logger.info(f"Behaviour: {args.behaviour}")
    logger.info(f"Split: {args.split}")
    logger.info(f"N prompts: {args.n_prompts or 'all'}")
    logger.info(f"Model: {args.model_name}")
    logger.info("")
    
    output_file = compute_baselines(args)
    
    logger.info(f"\n✅ SUCCESS! Baseline file: {output_file}")
    logger.info("\nNext steps:")
    logger.info("1. Stratify into low-margin subset using scripts/07b_stratify_prompts.py")
    logger.info("2. Run intervention experiments on low-margin subset")


if __name__ == "__main__":
    main()
