"""
Generate synthetic prompts for behaviours.

Supported behaviours:
  - grammar_agreement: Subject-verb number agreement (singular/plural)
  - physics_scalar_vector_operator: Classify operators/quantities as scalar vs vector

Usage:
    python scripts/01_generate_prompts.py
    python scripts/01_generate_prompts.py --behaviour physics_scalar_vector_operator
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


def generate_physics_scalar_vector_operator_prompts(
    n_train: int = 80,
    n_test: int = 20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate prompts for scalar vs vector operator/quantity classification.

    Type 3 (Abstraction) from behaviour_analysis.pdf:
      surface input -> abstract category {scalar, vector} -> surface output

    The model must decide whether a described physical operator or quantity
    is scalar-valued or vector-valued. Answer tokens: " scalar" / " vector".

    Includes:
      - Surface-form variants (reorder, whitespace, context sentences, notation)
      - Semantic swaps (scalar<->vector flip correct answer)
      - Deterministic via seed

    Returns:
        (train_prompts, test_prompts)
    """
    rng = random.Random(seed)

    # ── Core knowledge base ──────────────────────────────────────────────
    # Each entry: (description_template, correct_class, metadata_dict)
    # Templates use {field_name}, {symbol}, {context} placeholders.

    # Scalar operators/quantities
    scalar_items = [
        {
            "templates": [
                "The {name} of a {field_desc} produces a",
                "Computing the {name} of {field_desc} yields a",
                "The result of applying the {name} to a {field_desc} is a",
                "In physics, the {name} of a {field_desc} is classified as a",
            ],
            "name_variants": ["divergence", "div"],
            "field_desc_variants": ["vector field", "velocity field", "force field",
                                     "magnetic field", "electric field"],
            "class": "scalar",
            "physics_concept": "divergence",
        },
        {
            "templates": [
                "The {name} of a {field_desc} produces a",
                "Taking the {name} of {field_desc} gives a",
                "The result of applying {name} to a {field_desc} is a",
                "In vector calculus, the {name} of a {field_desc} is a",
            ],
            "name_variants": ["Laplacian"],
            "field_desc_variants": ["scalar field", "temperature field",
                                     "pressure field", "potential field",
                                     "scalar potential"],
            "class": "scalar",
            "physics_concept": "laplacian_scalar",
        },
        {
            "templates": [
                "The {name} between two vectors is a",
                "Computing the {name} of two vectors yields a",
                "The result of the {name} of two vectors is a",
                "In linear algebra, the {name} of two vectors is a",
            ],
            "name_variants": ["dot product", "scalar product", "inner product"],
            "field_desc_variants": ["force vectors", "displacement vectors",
                                     "velocity vectors", "unit vectors"],
            "class": "scalar",
            "physics_concept": "dot_product",
        },
        {
            "templates": [
                "The {name} in electrostatics is a",
                "{name} is classified as a",
                "In physics, {name} is a",
                "The quantity {name} is a",
            ],
            "name_variants": ["electric potential", "voltage",
                              "scalar potential", "electrostatic potential"],
            "field_desc_variants": [""],
            "class": "scalar",
            "physics_concept": "electric_potential",
        },
        {
            "templates": [
                "The {name} of a charged particle in an electric field is a",
                "{name} is classified as a",
                "In mechanics, {name} is a",
                "The quantity called {name} is a",
            ],
            "name_variants": ["kinetic energy", "potential energy",
                              "work done", "power"],
            "field_desc_variants": [""],
            "class": "scalar",
            "physics_concept": "energy_quantity",
        },
        {
            "templates": [
                "In thermodynamics, {name} is a",
                "The quantity {name} is a",
                "{name} is classified as a",
                "The physical quantity {name} is a",
            ],
            "name_variants": ["temperature", "pressure", "mass", "charge",
                              "density", "entropy"],
            "field_desc_variants": [""],
            "class": "scalar",
            "physics_concept": "thermodynamic_scalar",
        },
    ]

    # Vector operators/quantities
    vector_items = [
        {
            "templates": [
                "The {name} of a {field_desc} produces a",
                "Computing the {name} of {field_desc} yields a",
                "The result of applying the {name} to a {field_desc} is a",
                "In physics, the {name} of a {field_desc} is classified as a",
            ],
            "name_variants": ["curl", "rot"],
            "field_desc_variants": ["vector field", "velocity field", "force field",
                                     "magnetic field", "vector potential"],
            "class": "vector",
            "physics_concept": "curl",
        },
        {
            "templates": [
                "The {name} of a {field_desc} produces a",
                "Taking the {name} of {field_desc} gives a",
                "The result of computing the {name} of a {field_desc} is a",
                "In vector calculus, the {name} of a {field_desc} is a",
            ],
            "name_variants": ["gradient", "grad"],
            "field_desc_variants": ["scalar field", "temperature field",
                                     "pressure field", "potential field",
                                     "scalar potential"],
            "class": "vector",
            "physics_concept": "gradient",
        },
        {
            "templates": [
                "The {name} of two vectors is a",
                "Computing the {name} of two vectors yields a",
                "The result of the {name} of two vectors is a",
                "In linear algebra, the {name} of two vectors is a",
            ],
            "name_variants": ["cross product", "vector product"],
            "field_desc_variants": ["force vectors", "displacement vectors",
                                     "velocity vectors", "unit vectors"],
            "class": "vector",
            "physics_concept": "cross_product",
        },
        {
            "templates": [
                "The {name} at a point in space is a",
                "{name} is classified as a",
                "In electromagnetism, {name} is a",
                "The quantity {name} is a",
            ],
            "name_variants": ["electric field", "magnetic field",
                              "gravitational field", "force field"],
            "field_desc_variants": [""],
            "class": "vector",
            "physics_concept": "field_vector",
        },
        {
            "templates": [
                "In classical mechanics, {name} is a",
                "The quantity {name} is a",
                "{name} is classified as a",
                "The physical quantity {name} is a",
            ],
            "name_variants": ["velocity", "acceleration", "momentum",
                              "force", "torque", "angular momentum"],
            "field_desc_variants": [""],
            "class": "vector",
            "physics_concept": "mechanics_vector",
        },
        {
            "templates": [
                "The {name} of a {field_desc} is a",
                "In electromagnetism, the {name} is classified as a",
                "The quantity called {name} is a",
                "In gauge theory, {name} is a",
            ],
            "name_variants": ["vector potential", "magnetic vector potential"],
            "field_desc_variants": ["magnetic field", ""],
            "class": "vector",
            "physics_concept": "vector_potential",
        },
    ]

    # ── Context sentences (surface variation) ─────────────────────────────
    context_prefixes = [
        "",
        "Consider a smooth field in three dimensions. ",
        "In classical electromagnetism, ",
        "Given a well-behaved function, ",
        "For a physical system in equilibrium, ",
        "In the study of fluid dynamics, ",
        "Recall from vector calculus that ",
        "In the framework of classical field theory, ",
    ]

    # ── Generate all candidate prompts ────────────────────────────────────
    all_items = scalar_items + vector_items
    all_prompts = []

    for item in all_items:
        for name in item["name_variants"]:
            for template in item["templates"]:
                field_descs = item["field_desc_variants"]
                for fd in field_descs:
                    # Pick a random context prefix
                    ctx = rng.choice(context_prefixes)

                    # Fill template
                    if fd:
                        text = template.format(name=name, field_desc=fd)
                    else:
                        text = template.format(name=name, field_desc="").rstrip()

                    prompt_text = ctx + text
                    correct_class = item["class"]
                    incorrect_class = "vector" if correct_class == "scalar" else "scalar"

                    all_prompts.append({
                        "prompt": prompt_text,
                        "correct_answer": f" {correct_class}",
                        "incorrect_answer": f" {incorrect_class}",
                        "operator_name": name,
                        "field_type": correct_class,
                        "physics_concept": item["physics_concept"],
                        "context_prefix": ctx.strip(),
                        "full_sentence": f"{prompt_text} {correct_class} quantity.",
                    })

    # Shuffle deterministically then split
    rng.shuffle(all_prompts)

    # Ensure balanced classes in train and test
    scalars = [p for p in all_prompts if p["field_type"] == "scalar"]
    vectors = [p for p in all_prompts if p["field_type"] == "vector"]

    n_train_half = n_train // 2
    n_test_half = n_test // 2

    train_prompts = scalars[:n_train_half] + vectors[:n_train_half]
    test_prompts = (scalars[n_train_half:n_train_half + n_test_half] +
                    vectors[n_train_half:n_train_half + n_test_half])

    # Shuffle within splits
    rng.shuffle(train_prompts)
    rng.shuffle(test_prompts)

    logger.info(
        f"physics_scalar_vector_operator: generated {len(all_prompts)} candidates, "
        f"train={len(train_prompts)} (s={n_train_half}, v={n_train_half}), "
        f"test={len(test_prompts)} (s={n_test_half}, v={n_test_half})"
    )

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


GENERATORS = {
    "grammar_agreement": generate_grammar_agreement_prompts,
    "physics_scalar_vector_operator": generate_physics_scalar_vector_operator_prompts,
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic prompts")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        default=None,
        choices=list(GENERATORS.keys()),
        help="Generate prompts for a specific behaviour (default: all in config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    seed = config["seeds"]["prompt_generation"]
    output_path = Path(config["paths"]["prompts"])

    # Determine which behaviours to generate
    if args.behaviour:
        behaviours_to_gen = [args.behaviour]
    else:
        behaviours_to_gen = list(config.get("behaviours", {}).keys())

    print("=" * 70)
    print("GENERATING PROMPTS")
    print("=" * 70)
    print(f"\nModel: {config['model']['name']}")
    print(f"Output: {output_path}")
    print(f"Behaviours: {behaviours_to_gen}")

    all_files = []

    for idx, behaviour in enumerate(behaviours_to_gen, 1):
        print(f"\n[{idx}/{len(behaviours_to_gen)}] {behaviour}")
        print("-" * 70)

        if behaviour not in GENERATORS:
            print(f"  WARNING: No generator for '{behaviour}', skipping")
            continue

        behaviour_config = config.get("behaviours", {}).get(behaviour, {})
        n_train = behaviour_config.get("train_size", 80)
        n_test = behaviour_config.get("test_size", 20)

        generator = GENERATORS[behaviour]
        train_prompts, test_prompts = generator(
            n_train=n_train,
            n_test=n_test,
            seed=seed,
        )

        save_prompts(train_prompts, output_path, behaviour, "train")
        save_prompts(test_prompts, output_path, behaviour, "test")
        all_files.append((behaviour, len(train_prompts), len(test_prompts)))

        print("\nExample prompts:")
        for i, prompt in enumerate(train_prompts[:3], 1):
            print(f"  {i}. Prompt: '{prompt['prompt']}'")
            print(f"     Correct: '{prompt['correct_answer']}' | Incorrect: '{prompt['incorrect_answer']}'")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROMPT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Files created:")
    for beh, n_tr, n_te in all_files:
        print(f"  - {beh}_train.jsonl ({n_tr} prompts)")
        print(f"  - {beh}_test.jsonl ({n_te} prompts)")
    print("\nNext step: python scripts/02_run_baseline.py")


if __name__ == "__main__":
    main()
