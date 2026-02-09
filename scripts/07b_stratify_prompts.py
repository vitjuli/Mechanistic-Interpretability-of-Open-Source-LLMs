#!/usr/bin/env python3
"""
Stratify prompts into low-margin subset for intervention experiments.

Takes baseline margins CSV and creates a balanced subset of low-margin prompts
(0 < margin < 1.5) for causal intervention experiments.

Usage:
    python scripts/07b_stratify_prompts.py \
        --baselines data/results/baselines/baselines_grammar_agreement_train_n80.csv \
        --output data/prompts/low_margin_subset_n40.csv \
        --min_margin 0.0 \
        --max_margin 1.5 \
        --n_per_class 20
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def stratify_prompts(args):
    """Stratify prompts into low-margin subset."""
    
    # Load baselines
    df = pd.read_csv(args.baselines)
    logger.info(f"Loaded {len(df)} baseline margins from {args.baselines}")
    
    # Filter by margin range
    filtered = df[(df['margin'] > args.min_margin) & (df['margin'] < args.max_margin)].copy()
    logger.info(f"Found {len(filtered)} prompts in margin range ({args.min_margin}, {args.max_margin})")
    
    if len(filtered) == 0:
        logger.error("No prompts in specified margin range!")
        logger.error("Try increasing --max_margin or decreasing --min_margin")
        return None
    
    # Balance by class if 'number' column exists
    if 'number' in df.columns and df['number'].nunique() > 1:
        logger.info("\nBalancing by grammatical number...")
        
        classes = [c for c in filtered['number'].unique() if c != 'unknown']
        balanced_dfs = []
        
        for cls in classes:
            cls_df = filtered[filtered['number'] == cls]
            n_available = len(cls_df)
            n_to_sample = min(args.n_per_class, n_available)
            
            if n_available < args.n_per_class:
                logger.warning(f"  {cls}: only {n_available} available (requested {args.n_per_class})")
                logger.warning(f"  → Taking all {n_available} prompts for {cls}")
                
                # If significantly short, suggest expanding margin range
                if n_available < args.n_per_class // 2:
                    logger.warning(f"  → Consider increasing --max_margin to get more {cls} prompts")
            
            # Sample (or take all if n_to_sample == n_available)
            if n_to_sample < n_available:
                # Sort by margin (lowest first) and take top-n_to_sample
                sampled = cls_df.nsmallest(n_to_sample, 'margin')
            else:
                sampled = cls_df
            
            balanced_dfs.append(sampled)
            
            logger.info(f"  {cls}: selected {len(sampled)} prompts (mean margin: {sampled['margin'].mean():.3f})")
        
        result = pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        # No balancing, just take top-n by lowest margin
        logger.info("No class balancing (number field not available)")
        n_total = min(args.n_per_class * 2, len(filtered))  # Assume 2 classes
        result = filtered.nsmallest(n_total, 'margin')
    
    # Shuffle for randomness in presentation order
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify all essential fields are present
    essential_fields = ['orig_idx', 'prompt', 'margin', 'correct_token', 'incorrect_token']
    missing_fields = [f for f in essential_fields if f not in result.columns]
    if missing_fields:
        logger.error(f"Missing essential fields: {missing_fields}")
        logger.error("Baseline CSV must contain: orig_idx, prompt, margin, correct_token, incorrect_token")
        return None
    
    # Save CSV (for human readability and analysis)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    
    logger.info(f"\nSaved {len(result)} prompts to {output_path}")
    
    # Save JSONL (for intervention script compatibility)
    jsonl_path = output_path.with_suffix('.jsonl')
    
    import json
    with open(jsonl_path, 'w') as f:
        for _, row in result.iterrows():
            # Create JSONL record with correct field names for interventions
            record = {
                'prompt': row['prompt'],
                'correct_answer': row['correct_token'],  # Rename for intervention compatibility
                'incorrect_answer': row['incorrect_token'],
                'number': row.get('number', 'unknown'),
            }
            
            # Add optional metadata
            if 'orig_idx' in row:
                record['orig_idx'] = int(row['orig_idx'])
            if 'margin' in row:
                record['margin'] = float(row['margin'])
            if 'subject' in row:
                record['subject'] = row['subject']
            if 'template' in row:
                record['template'] = row['template']
            
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved JSONL subset to {jsonl_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("LOW-MARGIN SUBSET STATISTICS")
    logger.info("="*60)
    logger.info(f"Total prompts: {len(result)}")
    logger.info(f"Mean margin: {result['margin'].mean():.3f}")
    logger.info(f"Median margin: {result['margin'].median():.3f}")
    logger.info(f"Std margin: {result['margin'].std():.3f}")
    logger.info(f"Min margin: {result['margin'].min():.3f}")
    logger.info(f"Max margin: {result['margin'].max():.3f}")
    
    if 'number' in result.columns:
        logger.info(f"\nClass distribution:")
        for cls in result['number'].unique():
            if cls != 'unknown':
                count = len(result[result['number'] == cls])
                mean_margin = result[result['number'] == cls]['margin'].mean()
                logger.info(f"  {cls}: {count} ({count/len(result)*100:.1f}%), mean margin: {mean_margin:.3f}")
    
    logger.info(f"\nEssential fields verified:")
    logger.info(f"  orig_idx: {len(result['orig_idx'].unique())} unique values")
    logger.info(f"  correct_token: {result['correct_token'].nunique()} unique tokens")
    logger.info(f"  incorrect_token: {result['incorrect_token'].nunique()} unique tokens")
    
    logger.info("="*60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Stratify prompts into low-margin subset")
    
    parser.add_argument('--baselines', type=str, required=True,
                        help='Path to baseline margins CSV')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for stratified subset CSV')
    parser.add_argument('--min_margin', type=float, default=0.0,
                        help='Minimum margin (default: 0.0)')
    parser.add_argument('--max_margin', type=float, default=1.5,
                        help='Maximum margin (default: 1.5)')
    parser.add_argument('--n_per_class', type=int, default=20,
                        help='Number of prompts per class (default: 20)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PROMPT STRATIFICATION")
    logger.info("="*60)
    logger.info(f"Baselines: {args.baselines}")
    logger.info(f"Margin range: ({args.min_margin}, {args.max_margin})")
    logger.info(f"N per class: {args.n_per_class}")
    logger.info("")
    
    output_path = stratify_prompts(args)
    
    if output_path:
        logger.info(f"\n✅ SUCCESS! Low-margin subset: {output_path}")
        logger.info("\nNext steps:")
        logger.info("1. Run Phase 0.5: Activation audit")
        logger.info("2. Run Phase 1: Feature importance analysis")
        logger.info("3. Run Phase 2: Ablation experiments")


if __name__ == "__main__":
    main()
