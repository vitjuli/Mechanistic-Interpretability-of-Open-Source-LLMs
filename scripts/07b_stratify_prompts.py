#!/usr/bin/env python3
"""
Stratify prompts into Low-Margin (Targets) and High-Margin (Sources) subsets.

Creates two sets of prompts for causal intervention experiments:
1. Targets: Low margin (unconfident), used as recipients of patching.
2. Sources: High margin (confident), used as donors for patching.

Usage:
    python scripts/07b_stratify_prompts.py \
        --baselines data/results/baselines/baselines_grammar_agreement_train_n80.csv \
        --output_prefix data/prompts/grammar_agreement \
        --low_max 1.5 \
        --high_min 2.0 \
        --n_per_class 20
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def save_subset(df, output_path, label):
    """Save subset to CSV and JSONL."""
    if df.empty:
        logger.warning(f"Empty dataframe for {label}, skipping save.")
        return

    # Verify essential fields
    essential_fields = ['orig_idx', 'prompt', 'margin', 'correct_token', 'incorrect_token']
    missing_fields = [f for f in essential_fields if f not in df.columns]
    if missing_fields:
        logger.error(f"Missing essential fields in {label}: {missing_fields}")
        return

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} {label} prompts to {output_path}")

    # Save JSONL
    jsonl_path = output_path.with_suffix('.jsonl')
    with open(jsonl_path, 'w') as f:
        for _, row in df.iterrows():
            record = {
                'prompt': row['prompt'],
                'correct_answer': row['correct_token'],
                'incorrect_answer': row['incorrect_token'],
                'number': row.get('number', 'unknown'),
            }
            # Optional metadata
            for col in ['orig_idx', 'margin', 'subject', 'template']:
                if col in row:
                    val = row[col]
                    if col == 'margin': val = float(val)
                    if col == 'orig_idx': val = int(val)
                    record[col] = val
            
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved {label} JSONL to {jsonl_path}")


def balance_and_select(df, margin_min, margin_max, n_per_class, label, strict=False):
    """Filter by margin and balance classes."""
    logger.info(f"\nSelecting {label} (Margin: {margin_min} - {margin_max})...")
    
    # Filter by margin (inclusive)
    filtered = df[(df['margin'] >= margin_min) & (df['margin'] <= margin_max)].copy()
    logger.info(f"  Found {len(filtered)} candidates in range")
    
    if len(filtered) == 0:
        logger.warning(f"  No prompts found for {label} in range!")
        return pd.DataFrame()

    # Balance by class
    if 'number' in df.columns and df['number'].nunique() > 1:
        # Check strict class requirement
        if strict:
            need = {"singular", "plural"}
            have = set([c for c in filtered["number"].unique() if c != "unknown"])
            missing = need - have
            if missing:
                raise ValueError(
                    f"{label}: missing classes {missing} in margin range "
                    f"({margin_min}, {margin_max}). Adjust thresholds or baseline size."
                )

        # Deterministic order (fixed preference)
        classes = [c for c in ("singular", "plural") if c in set(filtered["number"])]
        # Add any others (e.g. unknown) just in case
        already = set(classes)
        remaining = sorted([c for c in filtered['number'].unique() if c not in already and c != 'unknown'])
        classes.extend(remaining)
        
        selected_dfs = []
        
        for cls in classes:
            cls_df = filtered[filtered['number'] == cls]
            n_avail = len(cls_df)
            
            if n_avail < n_per_class:
                msg = f"  [WARNING] {cls}: only {n_avail}/{n_per_class} available for {label}"
                logger.warning(msg)
                
                # Strict check: fail if not enough prompts
                if strict:
                     raise ValueError(f"{label}: class {cls} has only {n_avail} < {n_per_class}. Increase baseline size or relax thresholds.")
            
            # For Targets (Low Margin): Pick lowest margins (Hardest)
            # For Sources (High Margin): Pick highest margins (Most confident)
            if label == "TARGETS":
                sampled = cls_df.nsmallest(min(n_per_class, n_avail), 'margin')
            else:
                sampled = cls_df.nlargest(min(n_per_class, n_avail), 'margin')
            
            selected_dfs.append(sampled)
            logger.info(f"  {cls}: selected {len(sampled)} (Mean margin: {sampled['margin'].mean():.3f})")
            
        if not selected_dfs:
            return pd.DataFrame()
            
        result = pd.concat(selected_dfs, ignore_index=True)
    else:
        # No classes
        n_total = min(n_per_class * 2, len(filtered))
        if label == "TARGETS":
            result = filtered.nsmallest(n_total, 'margin')
        else:
            result = filtered.nlargest(n_total, 'margin')
            
    return result


def stratify_prompts(args):
    # Load baselines
    df = pd.read_csv(args.baselines)
    logger.info(f"Loaded {len(df)} baselines")
    
    # 1. Select Targets (Low Margin)
    targets = balance_and_select(
        df, 
        margin_min=args.low_min, 
        margin_max=args.low_max, 
        n_per_class=args.n_low_per_class, 
        label="TARGETS",
        strict=args.strict_classes
    )
    
    # 2. Select Sources (High Margin)
    sources = balance_and_select(
        df, 
        margin_min=args.high_min, 
        margin_max=args.high_max, 
        n_per_class=args.n_high_per_class, 
        label="SOURCES",
        strict=args.strict_classes
    )
    
    # Save files
    out_prefix = Path(args.output_prefix)
    
    if not targets.empty:
        t_path = out_prefix.with_name(f"{out_prefix.name}_targets.csv")
        save_subset(targets, t_path, "TARGETS")
        logger.info(f"Targets file: {t_path.with_suffix('.jsonl')}")
        
    if not sources.empty:
        s_path = out_prefix.with_name(f"{out_prefix.name}_sources.csv")
        save_subset(sources, s_path, "SOURCES")
        logger.info(f"Sources file: {s_path.with_suffix('.jsonl')}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Stratify prompts into Source/Target subsets")
    
    parser.add_argument('--baselines', type=str, required=True, help='Path to baselines CSV')
    parser.add_argument('--output_prefix', type=str, default=None, help='Output path prefix (e.g. data/prompts/grammar)')
    
    # Ranges
    parser.add_argument('--low_min', type=float, default=0.0)
    parser.add_argument('--low_max', type=float, default=1.5)
    parser.add_argument('--high_min', type=float, default=2.0)
    parser.add_argument('--high_max', type=float, default=10.0)
    
    # Counts
    parser.add_argument('--n_low_per_class', type=int, default=20)
    parser.add_argument('--n_high_per_class', type=int, default=20)
    
    # Flags
    parser.add_argument('--strict_classes', action='store_true',
                       help='Require both singular and plural in each subset (targets/sources)')
    
    # Backwards compatibility args (ignored but accepted to prevent crash if old scripts run)
    parser.add_argument('--output', type=str, help='DEPRECATED: Use --output_prefix')
    parser.add_argument('--min_margin', type=float, help='DEPRECATED')
    parser.add_argument('--max_margin', type=float, help='DEPRECATED')
    parser.add_argument('--n_per_class', type=int, help='DEPRECATED')

    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.output_prefix is None:
        if args.output:
            p = Path(args.output)
            args.output_prefix = str(p.with_suffix('')) # Strip extension
            logger.warning(f"Mapping legacy --output to --output_prefix={args.output_prefix}")
        else:
             parser.error("At least one of --output_prefix or --output is required.")
        
    if args.n_per_class:
        args.n_low_per_class = args.n_per_class
        args.n_high_per_class = args.n_per_class
        
    if args.max_margin:
        args.low_max = args.max_margin
        
    stratify_prompts(args)


if __name__ == "__main__":
    main()
