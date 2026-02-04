"""
Run baseline performance measurements on Qwen3-4B-Instruct-2507.

Validates that the model can perform the chosen behaviours with sufficient
clarity (log-probability differences) to proceed with circuit analysis.

Generates visualizations and saves comprehensive results for reproducibility.

Usage:
    python scripts/02_run_baseline.py --behaviour grammar_agreement
    python scripts/02_run_baseline.py --behaviour factual_recall
    python scripts/02_run_baseline.py --behaviour sentiment_continuation
    python scripts/02_run_baseline.py --behaviour arithmetic
    python scripts/02_run_baseline.py --all
"""

import json
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper

# Set style for figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load config from YAML file
def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load prompts from JSONL file
def load_prompts(
    prompt_path: Path,
    behaviour: str,
    split: str = "train",
) -> List[Dict]:
    """Load prompts from JSONL file."""
    file_path = prompt_path / f"{behaviour}_{split}.jsonl"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {file_path}\n"
            f"Run 'python scripts/01_generate_prompts.py' first."
        )

    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    return prompts

# Evaluate model performance on behaviour prompts
def evaluate_behaviour(
    model: ModelWrapper,
    prompts: List[Dict],
    behaviour_name: str,
    min_score_diff: float = 2.0,  # Renamed: this is normalized logprob diff threshold
) -> pd.DataFrame:
    """
    Evaluate model performance on behaviour prompts.

    Args:
        model: ModelWrapper instance
        prompts: List of prompt dictionaries
        behaviour_name: Name of behaviour
        min_score_diff: Minimum normalized logprob difference for "success"
                        (normalized = mean per token, avoids length bias)

    Returns:
        DataFrame with per-prompt results
    """
    results = []

    print(f"Evaluating {len(prompts)} prompts...")

    for prompt_data in tqdm(prompts):
        prompt_text = prompt_data["prompt"]
        correct_answer = prompt_data["correct_answer"]
        incorrect_answer = prompt_data["incorrect_answer"]

        try:
            # Get log probabilities for both answers (handles multi-token)
            log_probs, token_lengths = model.get_sequence_log_probs(
                [prompt_text],
                target_sequences=[correct_answer, incorrect_answer],
            )

            # Extract log probabilities (batch=1)
            correct_log_prob = log_probs[0, 0].item()
            incorrect_log_prob = log_probs[0, 1].item()
            
            # Extract token lengths (robust to different shapes)
            # Could be (num_sequences,) or (batch_size, num_sequences)
            if token_lengths.ndim == 1:
                # Shape: (num_sequences,) - expected for batch=1
                correct_token_len = int(token_lengths[0])
                incorrect_token_len = int(token_lengths[1])
            elif token_lengths.ndim == 2:
                # Shape: (batch_size, num_sequences)
                correct_token_len = int(token_lengths[0, 0])
                incorrect_token_len = int(token_lengths[0, 1])
            else:
                raise ValueError(f"Unexpected token_lengths shape: {token_lengths.shape}")
            
            # DEFENSIVE: Check for zero-length or negative tokens (should never happen, but be safe)
            if correct_token_len <= 0 or incorrect_token_len <= 0:
                raise ValueError(
                    f"Invalid token sequence length: "
                    f"correct={correct_token_len}, incorrect={incorrect_token_len}. "
                    f"Answers: '{correct_answer}', '{incorrect_answer}'"
                )
            
            # Raw score difference (sum of log probs)
            logprob_diff = correct_log_prob - incorrect_log_prob
            
            # Normalized score (mean log prob per token) - USED FOR SUCCESS
            correct_log_prob_normalized = correct_log_prob / correct_token_len
            incorrect_log_prob_normalized = incorrect_log_prob / incorrect_token_len
            logprob_diff_normalized = correct_log_prob_normalized - incorrect_log_prob_normalized

            # Determine success using NORMALIZED metric (avoids length bias)
            # min_score_diff is threshold on normalized (per-token) difference
            success = logprob_diff_normalized > min_score_diff

            # Get actual generation
            gen_output = model.generate(
                [prompt_text],
                max_new_tokens=5,
                temperature=0.0,
            )
            generated_ids = gen_output.sequences[0]

            # Extract only new tokens (not the prompt tokens)
            # Must match tokenization in generate() which uses tokenize() with default add_special_tokens=True
            input_token_ids = model.tokenizer.encode(
                prompt_text,
                add_special_tokens=True,  # Matches ModelWrapper.generate()->tokenize() default
            )
            input_length = len(input_token_ids)
            generated_text = model.tokenizer.decode(
                generated_ids[input_length:],
                skip_special_tokens=True,
            )

            results.append({
                "prompt": prompt_text,
                "correct_answer": correct_answer,  # Keep leading space
                "incorrect_answer": incorrect_answer,  # Keep leading space
                "correct_log_prob": correct_log_prob,
                "incorrect_log_prob": incorrect_log_prob,
                "correct_token_len": correct_token_len,
                "incorrect_token_len": incorrect_token_len,
                "logprob_diff": logprob_diff,
                "logprob_diff_normalized": logprob_diff_normalized,
                "success": success,
                "generated": generated_text.strip(),
                "score_method": "teacher_forced_logprob",  # How computed
                "success_metric": "normalized_per_token",  # What used for success
                "threshold": min_score_diff,  # Threshold used
                "threshold_metric": "logprob_diff_normalized",  # Metric threshold applies to
                **{k: v for k, v in prompt_data.items() if k not in ["prompt", "correct_answer", "incorrect_answer"]},
            })

        except Exception as e:
            print(f"\nError processing prompt: {prompt_text}")
            print(f"Error: {e}")
            results.append({
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
                "generated": "ERROR",
                "score_method": "teacher_forced_logprob",
                "success_metric": "normalized_per_token",
                "error": str(e),
            })

    return pd.DataFrame(results)


def create_visualizations(
    df: pd.DataFrame,
    behaviour_name: str,
    output_dir: Path,
    min_score_diff: float,
    split: str,
) -> Dict[str, Path]:
    """
    Create and save publication-quality visualizations for baseline results.

    Args:
        df: Results DataFrame
        behaviour_name: Name of the behaviour
        output_dir: Directory to save figures
        min_score_diff: Threshold for success
        split: Data split (train/test)

    Returns:
        Dictionary mapping figure names to file paths
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}

    # Figure 1: NORMALIZED log prob difference distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter out NaN/inf for statistics and plotting
    valid_data = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data for distribution plots in {behaviour_name} (all NaN/inf)")
        plt.close(fig)  # Prevent figure leak
        return {}

    # Histogram - NORMALIZED metric
    ax1 = axes[0]
    ax1.hist(valid_data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=min_score_diff, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({min_score_diff})')
    ax1.axvline(x=valid_data.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean ({valid_data.mean():.2f})')
    ax1.set_xlabel("Normalized Log Prob Diff (per token)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Normalized Log Prob Diff Distribution\n{behaviour_name} ({split})", fontsize=14)
    ax1.legend()

    # Box plot - NORMALIZED metric
    ax2 = axes[1]
    # Filter NaN/inf for boxplot
    valid_df = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_df) > 0:
        # Reconstruct with success labels from valid indices
        valid_indices = valid_df.index
        success_labels = df.loc[valid_indices, "success"].map({True: "Success", False: "Failure"})
        sns.boxplot(x=success_labels, y=valid_df, ax=ax2, palette=["coral", "lightgreen"])
        ax2.axhline(y=min_score_diff, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel("Classification", fontsize=12)
        ax2.set_ylabel("Normalized Log Prob Diff", fontsize=12)
        ax2.set_title("Normalized Log Prob Diff by Success/Failure", fontsize=14)
    else:
        # Empty plot with message
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel("Classification", fontsize=12)
        ax2.set_ylabel("Normalized Log Prob Diff", fontsize=12)
        ax2.set_title("Normalized Log Prob Diff by Success/Failure (No Data)", fontsize=14)

    plt.tight_layout()
    fig_path = figures_dir / f"baseline_{behaviour_name}_{split}_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_figures["distribution"] = fig_path

    # Figure 2: Normalized log prob scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = df["success"].map({True: "green", False: "red"})
    # Plot NORMALIZED values
    # Filter out NaN values for visualization
    norm_incorrect = df["incorrect_log_prob"] / df["incorrect_token_len"]
    norm_correct = df["correct_log_prob"] / df["correct_token_len"]
    
    # Remove NaN/inf for plotting
    valid_mask = np.isfinite(norm_incorrect) & np.isfinite(norm_correct)
    if valid_mask.sum() == 0:
        print(f"Warning: No valid data points for scatter plot in {behaviour_name} (all NaN/inf)")
        plt.close(fig)
        return saved_figures
    
    ax.scatter(norm_incorrect[valid_mask], 
               norm_correct[valid_mask], 
               c=colors[valid_mask], alpha=0.6, s=50)

    # Add diagonal line (equal normalized log probs)
    # Compute limits from valid data (simplified - already finite after masking)
    lo = min(norm_incorrect[valid_mask].min(), norm_correct[valid_mask].min()) - 1
    hi = max(norm_incorrect[valid_mask].max(), norm_correct[valid_mask].max()) + 1
    lims = [lo, hi]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal (normalized)')
    ax.plot(lims, [l + min_score_diff for l in lims], 'r--', alpha=0.5,
            label=f'Threshold (+{min_score_diff})')

    ax.set_xlabel("Incorrect Answer (normalized log prob)", fontsize=12)
    ax.set_ylabel("Correct Answer (normalized log prob)", fontsize=12)
    ax.set_title(f"Normalized Log Prob Comparison: {behaviour_name}\nGreen=Success, Red=Failure", fontsize=14)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig_path = figures_dir / f"baseline_{behaviour_name}_{split}_scatter.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_figures["scatter"] = fig_path

    # Figure 3: Per-category breakdown (if applicable)
    category_cols = []
    if "number" in df.columns:  # Grammar agreement
        category_cols.append(("number", "Singular vs Plural"))
    if "sentiment" in df.columns:  # Sentiment
        category_cols.append(("sentiment", "Sentiment Type"))
    if "error_type" in df.columns:  # Arithmetic
        category_cols.append(("error_type", "Error Type"))

    if category_cols:
        fig, axes = plt.subplots(1, len(category_cols), figsize=(6 * len(category_cols), 5))
        if len(category_cols) == 1:
            axes = [axes]

        for ax, (col, title) in zip(axes, category_cols):
            category_acc = df.groupby(col)["success"].mean()
            category_acc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel(title, fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Accuracy by {title}", fontsize=14)
            ax.set_ylim(0, 1)
            ax.axhline(y=df["success"].mean(), color='red', linestyle='--',
                       label=f'Overall: {df["success"].mean():.1%}')
            ax.legend()
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fig_path = figures_dir / f"baseline_{behaviour_name}_{split}_categories.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_figures["categories"] = fig_path

    return saved_figures


def print_summary(
    df: pd.DataFrame,
    behaviour_name: str,
    min_score_diff: float,
    success_threshold: float,
) -> Dict[str, float]:
    """Print summary statistics and return metrics dict."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {behaviour_name.upper()}")
    
    # Identify valid vs error rows
    valid_mask = df["logprob_diff_normalized"].replace([np.inf, -np.inf], np.nan).notna()
    df_valid = df[valid_mask]
    df_errors = df[~valid_mask]
    
    n_total = len(df)
    n_valid = len(df_valid)
    n_errors = len(df_errors)
    
    # CRITICAL: accuracy based on TOTAL (errors count as failures)
    n_success = int(df["success"].astype(bool).sum())
    accuracy = n_success / n_total if n_total > 0 else 0.0
    
    if n_valid == 0:
        print(f"\nWarning: No valid data points for {behaviour_name}!")
        print(f"  Total: {n_total}, Errors: {n_errors}")
        return {
            "accuracy": 0.0,
            "accuracy_valid_only": 0.0,
            "accuracy_sign": 0.0,
            "mean_logprob_diff": np.nan,
            "mean_logprob_diff_normalized": np.nan,
            "median_logprob_diff_normalized": np.nan,
            "std_logprob_diff_normalized": np.nan,
            "mean_correct_token_len": np.nan,
            "mean_incorrect_token_len": np.nan,
            "length_mismatch_rate": np.nan,
            "n_success": 0,
            "n_total": n_total,
            "n_valid": 0,
            "n_errors": n_errors,
            "passed": False,
            "score_method": "teacher_forced_logprob",
            "success_metric": "normalized_per_token",
            "threshold": min_score_diff,
        }
    
    # Calculate statistics on VALID data only (for reporting quality)
    # But accuracy uses total!
    accuracy_valid_only = df_valid["success"].mean()  # What would it be without errors
    mean_logprob_diff = df_valid["logprob_diff"].mean()
    mean_logprob_diff_norm = df_valid["logprob_diff_normalized"].mean()
    median_logprob_diff_norm = df_valid["logprob_diff_normalized"].median()
    std_logprob_diff_norm = df_valid["logprob_diff_normalized"].std()
    
    # Token length statistics (valid only)
    mean_correct_len = df_valid["correct_token_len"].mean()
    mean_incorrect_len = df_valid["incorrect_token_len"].mean()
    
    # Length mismatch rate (valid only - need both lengths to compare)
    length_mismatch_rate = (df_valid["correct_token_len"] != df_valid["incorrect_token_len"]).mean()
    
    # Sign accuracy (how often correct > incorrect, on valid data)
    acc_sign = (df_valid["logprob_diff_normalized"] > 0).mean()

    print(f"\nOverall Performance:")
    print(f"  Total samples: {n_total} (valid: {n_valid}, errors: {n_errors})")
    print(f"  Accuracy (errors=fail): {accuracy:.1%} ({n_success}/{n_total})")
    if n_errors > 0:
        print(f"  Accuracy (valid only): {accuracy_valid_only:.1%} ({df_valid['success'].sum()}/{n_valid})")
    print(f"  Sign accuracy (correct > incorrect): {acc_sign:.1%}")
    print(f"  Mean normalized logprob_diff: {mean_logprob_diff_norm:.3f}")
    print(f"  Median normalized logprob_diff: {median_logprob_diff_norm:.3f}")
    print(f"  Std normalized logprob_diff: {std_logprob_diff_norm:.3f}")
    print(f"  (Raw sum mean: {mean_logprob_diff:.3f})")
    print(f"\nToken Length Statistics (valid samples):")
    print(f"  Mean correct answer length: {mean_correct_len:.2f} tokens")
    print(f"  Mean incorrect answer length: {mean_incorrect_len:.2f} tokens")
    print(f"  Length mismatch rate: {length_mismatch_rate:.1%}")

    # Decision
    print(f"\nDecision Criterion:")
    print(f"  Required accuracy: ≥{success_threshold:.0%}")
    print(f"  Achieved accuracy: {accuracy:.1%}")
    print(f"  Success metric: normalized logprob_diff > {min_score_diff}")
    print(f"    (Per-token metric avoids length bias)")

    passed = accuracy >= success_threshold
    if passed:
        print(f"  PASS - Proceed to activation capture")
    else:
        print(f"  FAIL - Behaviour too weak for circuit analysis")
        print(f"  Consider different behaviour or model")

    # Show some examples
    print(f"\nExample Predictions (first 5):")
    for idx, row in df.head(5).iterrows():
        status = "PASS" if row["success"] else "FAIL"
        print(f"\n  [{status}] Prompt: '{row['prompt']}'")
        print(f"    Correct: '{row['correct_answer']}' (log_prob: {row['correct_log_prob']:.2f}, {row['correct_token_len']} tok)")
        print(f"    Incorrect: '{row['incorrect_answer']}' (log_prob: {row['incorrect_log_prob']:.2f}, {row['incorrect_token_len']} tok)")
        print(f"    Diff: {row['logprob_diff']:.2f} | Generated: '{row['generated']}'")

    # Show examples of failure cases
    if not passed:
        print(f"\nExample failures (showing worst normalized diffs):")
        failures = df[~df["success"]].nsmallest(5, "logprob_diff_normalized")
        for _, row in failures.iterrows():
            print(f"  - '{row['prompt']}' (normalized diff: {row['logprob_diff_normalized']:.2f})")

    return {
        "accuracy": accuracy,  # errors count as failures
        "accuracy_valid_only": accuracy_valid_only,  # what it would be if no errors
        "accuracy_sign": acc_sign,
        "mean_logprob_diff": mean_logprob_diff,
        "mean_logprob_diff_normalized": mean_logprob_diff_norm,
        "median_logprob_diff_normalized": median_logprob_diff_norm,
        "std_logprob_diff_normalized": std_logprob_diff_norm,
        "mean_correct_token_len": mean_correct_len,
        "mean_incorrect_token_len": mean_incorrect_len,
        "length_mismatch_rate": length_mismatch_rate,
        "n_success": int(n_success),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_errors": n_errors,
        "passed": passed,
        "score_method": "teacher_forced_logprob",
        "success_metric": "normalized_per_token",
        "threshold": min_score_diff,
    }


def create_summary_visualization(
    all_metrics: Dict[str, Dict],
    output_dir: Path,
    split: str,
):
    """Create a summary visualization comparing all behaviours."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    behaviours = list(all_metrics.keys())
    accuracies = [all_metrics[b]["accuracy"] for b in behaviours]
    # Use NORMALIZED metric (not raw)
    mean_diffs_norm = [all_metrics[b]["mean_logprob_diff_normalized"] for b in behaviours]
    passed = [all_metrics[b]["passed"] for b in behaviours]
    # Extract actual thresholds (may differ per behaviour)
    thresholds = [all_metrics[b].get("threshold", 2.0) for b in behaviours]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    ax1 = axes[0]
    colors = ['green' if p else 'red' for p in passed]
    bars = ax1.bar(behaviours, accuracies, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Threshold (80%)')
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"Baseline Accuracy by Behaviour ({split})", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

    # Normalized log prob difference comparison
    ax2 = axes[1]
    bars = ax2.bar(behaviours, mean_diffs_norm, color='steelblue', edgecolor='black', alpha=0.7)
    # Show threshold range if they differ
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)
    if min_threshold == max_threshold:
        ax2.axhline(y=min_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold: {min_threshold}')
    else:
        ax2.axhspan(min_threshold, max_threshold, alpha=0.2, color='red', 
                    label=f'Threshold range: [{min_threshold}, {max_threshold}]')
    ax2.set_ylabel("Mean Normalized Log Prob Diff", fontsize=12)
    ax2.set_title(f"Mean Normalized Log Prob Diff by Behaviour ({split})", fontsize=14)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, diff in zip(bars, mean_diffs_norm):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{diff:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig_path = figures_dir / f"baseline_summary_{split}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Run baseline performance measurements")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        choices=["grammar_agreement", "factual_recall", "sentiment_continuation", "arithmetic"],
        help="Which behaviour to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all behaviours",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    torch.manual_seed(config["seeds"]["torch_seed"])

    # Determine behaviours to evaluate
    if args.all:
        behaviours = ["grammar_agreement", "factual_recall", "sentiment_continuation", "arithmetic"]
    elif args.behaviour:
        behaviours = [args.behaviour]
    else:
        print("Error: Must specify --behaviour or --all")
        return

    print("=" * 70)
    print("BASELINE PERFORMANCE MEASUREMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Device: {config['model']['device']}")
    print(f"  Split: {args.split}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load model
    print(f"\nLoading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    # Evaluate each behaviour
    all_results = {}
    all_metrics = {}

    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)

        # Load prompts
        prompt_path = Path(config["paths"]["prompts"])
        try:
            prompts = load_prompts(prompt_path, behaviour, args.split)
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            print(f"Skipping {behaviour}. Run 01_generate_prompts.py first.")
            continue

        print(f"Loaded {len(prompts)} prompts")

        # Get behaviour config
        behaviour_config = config["behaviours"][behaviour]
        # IMPORTANT: Config uses name "min_logit_diff" for backward compatibility,
        # but we interpret it as NORMALIZED logprob diff threshold (per-token scale)
        # This is NOT the same as raw logit difference!
        # Typical values: 1.5-2.5 (per-token log prob difference)
        min_score_diff = behaviour_config["min_logit_diff"]
        success_threshold = behaviour_config["success_threshold"]

        # Evaluate
        results_df = evaluate_behaviour(
            model,
            prompts,
            behaviour,
            min_score_diff,
        )

        # Save results
        output_path = Path(config["paths"]["results"])
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"baseline_{behaviour}_{args.split}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")

        # Create visualizations
        if not args.no_viz:
            print("\nGenerating visualizations...")
            fig_paths = create_visualizations(
                results_df,
                behaviour,
                output_path,
                min_score_diff,
                args.split,
            )
            for name, path in fig_paths.items():
                print(f"  Saved {name} figure: {path.name}")

        # Print summary and get metrics
        metrics = print_summary(
            results_df,
            behaviour,
            min_score_diff,
            success_threshold,
        )

        all_results[behaviour] = results_df
        all_metrics[behaviour] = metrics

    # Skip summary if no behaviours were evaluated
    if not all_results:
        print("\nNo behaviours were evaluated. Check prompt files exist.")
        return

    # Create summary visualization
    if not args.no_viz and len(all_metrics) > 1:
        print("\nGenerating summary visualization...")
        summary_path = create_summary_visualization(
            all_metrics,
            Path(config["paths"]["results"]),
            args.split,
        )
        print(f"  Saved summary: {summary_path.name}")

    # Save metrics summary as JSON
    output_path = Path(config["paths"]["results"])
    metrics_file = output_path / f"baseline_metrics_{args.split}.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": config["model"]["name"],
            "split": args.split,
            "behaviours": all_metrics,
        }, f, indent=2)
    print(f"\nSaved metrics summary: {metrics_file}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_passed = True
    for behaviour, metrics in all_metrics.items():
        status = "PASS" if metrics["passed"] else "FAIL"
        if not metrics["passed"]:
            all_passed = False
        print(f"\n{behaviour}:")
        print(f"  Accuracy: {metrics['accuracy']:.1%} | "
              f"Mean norm diff: {metrics['mean_logprob_diff_normalized']:.2f} | {status}")

    print("\n" + "=" * 70)
    if all_passed:
        print("All behaviours PASSED. Ready for circuit analysis.")
        print("\nNext steps:")
        print("  1. Review visualizations in data/results/figures/")
        print("  2. Proceed to: python scripts/03_capture_activations.py --all")
    else:
        print("Some behaviours FAILED threshold.")
        print("\nRecommendations:")
        print("  1. Review failed behaviours' visualizations")
        print("  2. Consider adjusting prompts or thresholds")
        print("  3. Partial analysis possible with passing behaviours")
    print("=" * 70)


if __name__ == "__main__":
    main()
