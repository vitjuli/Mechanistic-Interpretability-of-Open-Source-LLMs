"""
Generate synthetic prompts for grammar agreement behaviour.

Single behaviour for end-to-end pipeline testing on CSD3.

Usage:
    python scripts/01_generate_prompts.py
"""

import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_grammar_agreement_prompts(
    n_train: int = 80,
    n_test: int = 20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate subject-verb agreement prompts.

    Structure:
        "The [subject] [verb] ..." where verb must agree with subject number.

    Returns:
        (train_prompts, test_prompts)
    """
    random.seed(seed)

    # Define subjects (singular, plural)
    subjects = {
        "singular": [
            "cat", "dog", "student", "teacher", "doctor",
            "scientist", "engineer", "artist", "musician", "writer",
            "computer", "building", "tree", "flower", "book",
            "car", "phone", "table", "chair", "lamp",
        ],
        "plural": [
            "cats", "dogs", "students", "teachers", "doctors",
            "scientists", "engineers", "artists", "musicians", "writers",
            "computers", "buildings", "trees", "flowers", "books",
            "cars", "phones", "tables", "chairs", "lamps",
        ],
    }

    # Define verb pairs (singular, plural)
    verb_pairs = [
        ("is", "are"),
        ("was", "were"),
        ("sits", "sit"),
        ("stands", "stand"),
        ("runs", "run"),
        ("walks", "walk"),
        ("works", "work"),
        ("plays", "play"),
        ("eats", "eat"),
        ("sleeps", "sleep"),
        ("reads", "read"),
        ("writes", "write"),
        ("thinks", "think"),
        ("knows", "know"),
        ("seems", "seem"),
    ]

    # Define continuations
    continuations = [
        "in the room",
        "near the door",
        "by the window",
        "on the table",
        "in the garden",
        "at the park",
        "every day",
        "very well",
        "quickly",
        "carefully",
    ]

    all_prompts = []

    # Generate prompts
    for number in ["singular", "plural"]:
        for subject in subjects[number]:
            for verb_sing, verb_plur in verb_pairs:
                for continuation in random.sample(continuations, 2):
                    # Choose correct verb form
                    correct_verb = verb_sing if number == "singular" else verb_plur
                    incorrect_verb = verb_plur if number == "singular" else verb_sing

                    prompt_text = f"The {subject}"
                    answer_correct = f" {correct_verb}"
                    answer_incorrect = f" {incorrect_verb}"

                    all_prompts.append({
                        "prompt": prompt_text,
                        "correct_answer": answer_correct,
                        "incorrect_answer": answer_incorrect,
                        "continuation": continuation,
                        "subject": subject,
                        "number": number,
                        "verb_pair": f"{verb_sing}/{verb_plur}",
                        "full_sentence": f"{prompt_text}{answer_correct} {continuation}.",
                    })

    # Shuffle and split
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:n_train]
    test_prompts = all_prompts[n_train:n_train + n_test]

    return train_prompts, test_prompts


def save_prompts(
    prompts: List[Dict],
    output_path: Path,
    behaviour_name: str,
    split: str,
):
    """Save prompts to JSONL file."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{behaviour_name}_{split}.jsonl"

    with open(output_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"Saved {len(prompts)} {split} prompts to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic prompts")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    seed = config["seeds"]["prompt_generation"]
    output_path = Path(config["paths"]["prompts"])

    print("=" * 70)
    print("GENERATING PROMPTS: GRAMMAR AGREEMENT")
    print("=" * 70)
    print(f"\nModel: {config['model']['name']}")
    print(f"Output: {output_path}")

    # =========================================================================
    # Generate Grammar Agreement Prompts
    # =========================================================================
    print("\n[1/1] Grammatical Number Agreement")
    print("-" * 70)

    behaviour_config = config["behaviours"]["grammar_agreement"]
    train_prompts, test_prompts = generate_grammar_agreement_prompts(
        n_train=behaviour_config["train_size"],
        n_test=behaviour_config["test_size"],
        seed=seed,
    )

    save_prompts(train_prompts, output_path, "grammar_agreement", "train")
    save_prompts(test_prompts, output_path, "grammar_agreement", "test")

    print("\nExample prompts:")
    for i, prompt in enumerate(train_prompts[:5], 1):
        print(f"  {i}. Prompt: '{prompt['prompt']}'")
        print(f"     Correct: '{prompt['correct_answer']}' | Incorrect: '{prompt['incorrect_answer']}'")
        print(f"     Full: {prompt['full_sentence']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROMPT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Files created:")
    print(f"  - grammar_agreement_train.jsonl ({len(train_prompts)} prompts)")
    print(f"  - grammar_agreement_test.jsonl ({len(test_prompts)} prompts)")
    print("\nBehaviour: Subject-verb number agreement (singular/plural)")
    print("Target tokens: ' is' vs ' are', ' was' vs ' were', etc.")
    print("\nNext step: python scripts/02_run_baseline.py")


if __name__ == "__main__":
    main()
