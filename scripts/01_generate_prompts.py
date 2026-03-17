"""
Generate synthetic prompts for behaviours.

Supported behaviours:
  - grammar_agreement: Subject-verb number agreement (singular/plural)
  - physics_scalar_vector_operator: Classify operators/quantities as scalar vs vector
  - antonym_operation: Predict the antonym of an adjective (EN only, v0)
  - multilingual_antonym: EN+FR antonym + EN synonym prompts for Anthropic circuits
                          reproduction (operation/operand/language swap experiments)
  - multilingual_circuits: Antonym-only EN+FR prompts, Anthropic reproduction (4 templates)
  - multilingual_circuits_b1: B1 extended set — 8 templates per concept (T0-T7), 96 train

Usage:
    python scripts/01_generate_prompts.py
    python scripts/01_generate_prompts.py --behaviour physics_scalar_vector_operator
    python scripts/01_generate_prompts.py --behaviour antonym_operation
    python scripts/01_generate_prompts.py --behaviour multilingual_antonym
    python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1
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


# ---------------------------------------------------------------------------
# Antonym operation: tokenization-verified (word, antonym) pairs per language.
# These passed the b_tokenize_audit_antonyms.py audit against Qwen3-4B-Instruct-2507.
# Each word and its antonym tokenise to exactly ONE BPE token (with leading space).
# EN: 25/25 candidates valid.  FR: 8/20 candidates valid.
# Cross-language pairs (same concept valid in both EN+FR): 8 pairs.
# ---------------------------------------------------------------------------
_ANTONYM_PAIRS = {
    "en": [
        ("small",  "large"),   ("hot",    "cold"),    ("fast",   "slow"),
        ("dark",   "light"),   ("hard",   "soft"),    ("rich",   "poor"),
        ("new",    "old"),     ("good",   "bad"),     ("strong", "weak"),
        ("happy",  "sad"),     ("open",   "closed"),  ("empty",  "full"),
        ("early",  "late"),    ("high",   "low"),     ("long",   "short"),
        ("wide",   "narrow"),  ("clean",  "dirty"),   ("loud",   "quiet"),
        ("cheap",  "expensive"), ("easy", "hard"),    ("bright", "dark"),
        ("deep",   "shallow"), ("heavy",  "light"),   ("sharp",  "dull"),
        ("thick",  "thin"),
    ],
    "fr": [
        ("petit",   "grand"),    ("vite",    "lent"),
        ("nouveau", "vieux"),    ("vide",    "plein"),
        ("haut",    "bas"),      ("long",    "court"),
        ("propre",  "sale"),     ("facile",  "difficile"),
    ],
}

# Concepts valid in BOTH en and fr (aligned by index in the en candidate list).
# Source of truth: b_tokenize_audit_antonyms.py run on 2026-03-01.
_ANTONYM_CROSS_LANG: List[Tuple[str, str, str, str]] = [
    # (en_word, en_ant, fr_word, fr_ant)
    ("small",  "large",  "petit",   "grand"),
    ("fast",   "slow",   "vite",    "lent"),
    ("new",    "old",    "nouveau", "vieux"),
    ("empty",  "full",   "vide",    "plein"),
    ("high",   "low",    "haut",    "bas"),
    ("long",   "short",  "long",    "court"),
    ("clean",  "dirty",  "propre",  "sale"),
    ("easy",   "hard",   "facile",  "difficile"),
]

_ANTONYM_TEMPLATES = {
    # All templates end with a complete word (no trailing space/quote).
    # Model predicts ' {antonym}' with leading space — matches physics/grammar convention.
    # 4 variants per language; forward-only direction (word → antonym) eliminates
    # duplicate prompt texts that arise from bidirectional generation.
    "en": [
        'The opposite of "{word}" is',     # T1
        'The antonym of "{word}" is',      # T2
        '"{word}" is the opposite of',     # T3
        '"{word}" is the antonym of',      # T4
    ],
    "fr": [
        'Le contraire de "{word}" est',                 # T1
        "L'antonyme de \"{word}\" est",                 # T2
        '"{word}" est le contraire de',                 # T3
        '"{word}" est l\'antonyme de',                  # T4
    ],
}


def generate_antonym_operation_prompts(
    n_train: int = 20,
    n_test: int = 5,
    seed: int = 42,
    languages: List[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate antonym-prediction prompts from tokenization-audited pairs.

    Each prompt dict contains:
      prompt          : text up to (not including) the predicted token
      correct_answer  : the antonym (stripped, no leading space)
      incorrect_answer: the original word (identity, no leading space)
      word            : source word
      antonym         : ground-truth antonym
      language        : ISO language code
      concept_index   : index into the language's pair list (for cross-lang alignment)
      cross_lang_valid: True iff this concept has a valid translation in both en+fr

    The generator creates BOTH directions for each pair:
      word -> antonym  (forward)
      antonym -> word  (reverse)
    This doubles the dataset and makes ablation bidirectionally meaningful.

    Returns:
        (train_prompts, test_prompts)
    """
    if languages is None:
        languages = ["en"]

    rng = random.Random(seed)
    all_prompts: List[Dict] = []

    cross_lang_en_words = {row[0] for row in _ANTONYM_CROSS_LANG}
    cross_lang_fr_words = {row[2] for row in _ANTONYM_CROSS_LANG}

    for lang in languages:
        pairs  = _ANTONYM_PAIRS.get(lang, [])
        tmpls  = _ANTONYM_TEMPLATES[lang]  # list of 4 template strings

        for concept_idx, (word, antonym) in enumerate(pairs):
            if lang == "en":
                xvalid = word in cross_lang_en_words
            elif lang == "fr":
                xvalid = word in cross_lang_fr_words
            else:
                xvalid = False

            base = dict(
                language=lang,
                concept_index=concept_idx,
                cross_lang_valid=xvalid,
                direction="forward",   # forward-only; avoids duplicate prompt texts
            )

            # One prompt per template variant (word → antonym, forward only).
            # Using bidirectional generation would create duplicate prompt texts
            # for the 3 words that appear as both word and antonym across pairs
            # ('dark', 'hard', 'light'), producing conflicting correct_answer labels.
            for tmpl_idx, tmpl in enumerate(tmpls):
                all_prompts.append({
                    **base,
                    "prompt":           tmpl.format(word=word),
                    "correct_answer":   f" {antonym}",  # leading space — standard BPE convention
                    "incorrect_answer": f" {word}",
                    "word":             word,
                    "antonym":          antonym,
                    "template_idx":     tmpl_idx,
                })

    rng.shuffle(all_prompts)
    train_prompts = all_prompts[:n_train]
    test_prompts  = all_prompts[n_train : n_train + n_test]
    return train_prompts, test_prompts


GENERATORS = {
    "grammar_agreement": generate_grammar_agreement_prompts,
    "physics_scalar_vector_operator": generate_physics_scalar_vector_operator_prompts,
    "antonym_operation": generate_antonym_operation_prompts,
    # multilingual_antonym registered below after its definition
}


# ===========================================================================
# multilingual_antonym
#
# Reproduces the Anthropic "Multilingual Circuits" case study structure.
# Three independently-intervene-able axes:
#   (1) operation : antonym vs synonym  (C1 swap)
#   (2) operand   : small vs hot        (C2 swap)
#   (3) language  : EN vs FR            (C3 swap)
#
# Data sources:
#   - EN antonym pairs: 9 concepts (audited subset of antonym_operation's 25)
#   - FR antonym pairs: 8 concepts (concept_idx=1 "hot/cold" fails: froid=2 tokens)
#   - EN synonym pairs: 9/9 pass tokenization audit (b_tokenize_audit_multilingual.py)
#   - FR synonym pairs: 0 valid — FR synonyms are NOT single-token
#
# Tokenization audit: 2026-03-04, Qwen3-4B-Instruct-2507 tokenizer.
# ---------------------------------------------------------------------------

# 9 cross-language concepts used in this behaviour.
# Schema: (concept_index, en_word, en_antonym, fr_word, fr_antonym, has_fr)
# has_fr=False means the FR antonym pair failed the tokenization audit.
_ML_CONCEPTS: List[Tuple] = [
    (0, "small",  "large",   "petit",    "grand",      True),
    (1, "hot",    "cold",    None,        None,         False),  # froid=2 tokens
    (2, "fast",   "slow",    "rapide",   "lent",        True),  # was "vite" (adverb); "rapide" is adj, parallel to EN "fast"
    (3, "new",    "old",     "nouveau",  "vieux",       True),
    (4, "empty",  "full",    "vide",     "plein",       True),
    (5, "high",   "low",     "haut",     "bas",         True),
    (6, "long",   "short",   "long",     "court",       True),
    (7, "clean",  "dirty",   "propre",   "sale",        True),
    (8, "easy",   "hard",    "facile",   "difficile",   True),
]

# EN synonym pairs — all 9/9 passed audit (2026-03-04).
# Schema: concept_index → (synonym_of_word, synonym_of_antonym)
_ML_SYNONYMS_EN: Dict[int, Tuple[str, str]] = {
    0: ("tiny",    "big"),       # small→tiny,    large→big
    1: ("warm",    "cool"),      # hot→warm,      cold→cool
    2: ("quick",   "gradual"),   # fast→quick,    slow→gradual
    3: ("fresh",   "aged"),      # new→fresh,     old→aged
    4: ("bare",    "packed"),    # empty→bare,    full→packed
    5: ("tall",    "short"),     # high→tall,     low→short
    6: ("lengthy", "brief"),     # long→lengthy,  short→brief
    7: ("neat",    "messy"),     # clean→neat,    dirty→messy
    8: ("simple",  "tough"),     # easy→simple,   hard→tough
}

# Antonym prompt templates — same 4 variants as antonym_operation.
_ML_ANT_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        'The opposite of "{word}" is',      # T0
        'The antonym of "{word}" is',       # T1
        '"{word}" is the opposite of',      # T2
        '"{word}" is the antonym of',       # T3
    ],
    "fr": [
        'Le contraire de "{word}" est',                 # T0
        "L'antonyme de \"{word}\" est",                 # T1
        '"{word}" est le contraire de',                 # T2
        '"{word}" est l\'antonyme de',                  # T3
    ],
}

# Synonym prompt templates — 4 variants, EN only (FR synonyms failed audit).
_ML_SYN_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        'A word similar in meaning to "{word}" is',  # T0
        'A synonym of "{word}" is',                  # T1
        '"{word}" means roughly the same as',        # T2
        'The synonym of "{word}" is',                # T3
    ],
}


def generate_multilingual_antonym_prompts(
    n_train: int = 80,
    n_test: int = 24,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate multilingual antonym + synonym prompts for Anthropic circuits reproduction.

    Prompt pool (104 total, before train/test split):
      - EN antonym : 9 concepts × 4 templates = 36
      - FR antonym : 8 concepts × 4 templates = 32  (concept_idx=1 excluded, froid fails)
      - EN synonym : 9 concepts × 4 templates = 36
      - FR synonym : 0                               (no valid single-token FR synonyms)

    Fields in each prompt dict:
      prompt          : text fed to the model (excludes the predicted token)
      correct_answer  : target token with leading space (e.g. ' large')
      incorrect_answer: word itself with leading space  (e.g. ' small')
      word            : source adjective
      target          : word the model should predict (= antonym or synonym)
      antonym         : ground-truth antonym for this concept
      synonym_word    : ground-truth synonym for this concept (EN only; None for FR)
      synonym_antonym : synonym of the antonym (EN only; None for FR)
      language        : "en" | "fr"
      operation       : "antonym" | "synonym"
      concept_index   : 0-8 (shared across languages and operations)
      template_idx    : 0-3
      cross_lang_valid: True iff concept has a valid FR antonym pair

    Patching pair modes (used by script 07 --patch_mode):
      C1 (operation swap): source=antonym prompt, target=synonym prompt;
                           same concept_index + language
      C2 (operand swap)  : source=hot antonym (idx=1), target=small antonym (idx=0);
                           same language (EN only, FR hot/cold unavailable)
      C3 (language swap) : source=EN antonym, target=FR antonym;
                           same concept_index (8 concepts with has_fr=True)
    """
    rng = random.Random(seed)
    all_prompts: List[Dict] = []

    for cidx, en_word, en_ant, fr_word, fr_ant, has_fr in _ML_CONCEPTS:
        syn_pair = _ML_SYNONYMS_EN.get(cidx)      # (syn_word, syn_ant) or None
        syn_word = syn_pair[0] if syn_pair else None
        syn_ant  = syn_pair[1] if syn_pair else None

        # --- EN antonym prompts ---
        for tidx, tmpl in enumerate(_ML_ANT_TEMPLATES["en"]):
            all_prompts.append({
                "prompt":           tmpl.format(word=en_word),
                "correct_answer":   f" {en_ant}",
                "incorrect_answer": f" {en_word}",
                "word":             en_word,
                "target":           en_ant,
                "antonym":          en_ant,
                "synonym_word":     syn_word,
                "synonym_antonym":  syn_ant,
                "language":         "en",
                "operation":        "antonym",
                "concept_index":    cidx,
                "template_idx":     tidx,
                "cross_lang_valid": has_fr,
            })

        # --- FR antonym prompts (only when has_fr=True) ---
        if has_fr:
            for tidx, tmpl in enumerate(_ML_ANT_TEMPLATES["fr"]):
                all_prompts.append({
                    "prompt":           tmpl.format(word=fr_word),
                    "correct_answer":   f" {fr_ant}",
                    "incorrect_answer": f" {fr_word}",
                    "word":             fr_word,
                    "target":           fr_ant,
                    "antonym":          fr_ant,
                    "synonym_word":     None,   # FR synonyms unavailable
                    "synonym_antonym":  None,
                    "language":         "fr",
                    "operation":        "antonym",
                    "concept_index":    cidx,
                    "template_idx":     tidx,
                    "cross_lang_valid": True,
                })

        # --- EN synonym prompts (all 9 concepts pass audit) ---
        if syn_pair is not None:
            for tidx, tmpl in enumerate(_ML_SYN_TEMPLATES["en"]):
                all_prompts.append({
                    "prompt":           tmpl.format(word=en_word),
                    "correct_answer":   f" {syn_word}",
                    "incorrect_answer": f" {en_word}",
                    "word":             en_word,
                    "target":           syn_word,
                    "antonym":          en_ant,
                    "synonym_word":     syn_word,
                    "synonym_antonym":  syn_ant,
                    "language":         "en",
                    "operation":        "synonym",
                    "concept_index":    cidx,
                    "template_idx":     tidx,
                    "cross_lang_valid": has_fr,
                })

    rng.shuffle(all_prompts)
    train_prompts = all_prompts[:n_train]
    test_prompts  = all_prompts[n_train : n_train + n_test]
    return train_prompts, test_prompts


GENERATORS["multilingual_antonym"] = generate_multilingual_antonym_prompts


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour: multilingual_circuits
#   Antonym-only EN+FR prompts for a 1-to-1 reproduction of Anthropic's
#   "Multilingual Circuits" case study (no synonyms, no operand-swap).
#   8 cross-language concepts (concept_index in {0,2,3,4,5,6,7,8}).
#   Stratified split: for each (concept_index, language) group, 1 template
#   goes to test and 3 go to train → train=48, test=16.
# ─────────────────────────────────────────────────────────────────────────────

def generate_multilingual_circuits_prompts(
    n_train: int = 48,
    n_test: int = 16,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate multilingual antonym-only prompts for Anthropic circuits reproduction.

    8 cross-language concepts (concept_idx=1 excluded: 'froid' = 2 tokens):
      indices 0, 2, 3, 4, 5, 6, 7, 8.
    4 templates × 2 languages = 8 prompts per concept → 64 total.

    Stratified split: for each of 16 (concept_index, language) groups, shuffle
    [0,1,2,3] template indices; assign first shuffled index to test and the
    remaining 3 to train.
    → train = 24 EN + 24 FR = 48, test = 8 EN + 8 FR = 16.

    Fields in each prompt dict:
      prompt          : text fed to the model
      correct_answer  : target token with leading space (e.g. ' large')
      incorrect_answer: word itself with leading space  (e.g. ' small')
      word            : source adjective
      antonym         : ground-truth antonym for this concept
      language        : "en" | "fr"
      concept_index   : 0–8 (only cross-lang valid indices are included)
      template_idx    : 0–3
      cross_lang_valid: always True (all included concepts pass FR tokenization audit)

    Intervention axis (script 07 --patch_mode):
      C3 (language swap): source=EN antonym, target=FR antonym, same concept_index.
    """
    rng = random.Random(seed)
    train_prompts: List[Dict] = []
    test_prompts:  List[Dict] = []

    # 8 cross-language concepts (has_fr=True from _ML_CONCEPTS)
    mc_concepts = [
        (cidx, ew, ea, fw, fa)
        for cidx, ew, ea, fw, fa, hf in _ML_CONCEPTS
        if hf
    ]

    # Draw test_tidx ONCE per concept (shared between EN and FR) so that
    # C3 patching pairs always have matching template_idx in source and target.
    for cidx, en_word, en_ant, fr_word, fr_ant in mc_concepts:
        tidxs = list(range(4))
        rng.shuffle(tidxs)
        test_tidx = tidxs[0]  # same for EN and FR

        for lang in ["en", "fr"]:
            templates = _ML_ANT_TEMPLATES[lang]
            word = en_word if lang == "en" else fr_word
            ant  = en_ant  if lang == "en" else fr_ant

            for tidx, tmpl in enumerate(templates):
                p = {
                    "prompt":           tmpl.format(word=word),
                    "correct_answer":   f" {ant}",
                    "incorrect_answer": f" {word}",
                    "word":             word,
                    "antonym":          ant,
                    "language":         lang,
                    "concept_index":    cidx,
                    "template_idx":     tidx,
                    "cross_lang_valid": True,
                }
                if tidx == test_tidx:
                    test_prompts.append(p)
                else:
                    train_prompts.append(p)

    assert len(train_prompts) == 48, f"Expected 48 train, got {len(train_prompts)}"
    assert len(test_prompts)  == 16, f"Expected 16 test, got {len(test_prompts)}"

    return train_prompts, test_prompts


GENERATORS["multilingual_circuits"] = generate_multilingual_circuits_prompts


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour: multilingual_circuits_b1
#   B1 extended template set: 8 templates per (concept, language) group.
#   T0-T3 are identical to multilingual_circuits; T4-T7 add new surface variants.
#   Stratified split: 2 test_tidxs per concept (shared EN/FR), 6 train_tidxs.
#   → train = 48 EN + 48 FR = 96, test = 16 EN + 16 FR = 32.
# ─────────────────────────────────────────────────────────────────────────────

_ML_ANT_TEMPLATES_B1: Dict[str, List[str]] = {
    "en": [
        'The opposite of "{word}" is',        # T0  (same as MC T0)
        'The antonym of "{word}" is',         # T1  (same as MC T1)
        '"{word}" is the opposite of',        # T2  (same as MC T2)
        '"{word}" is the antonym of',         # T3  (same as MC T3)
        'The contrary of "{word}" is',        # T4  (new)
        '"{word}" is the contrary of',        # T5  (new)
        'The word opposite to "{word}" is',   # T6  (new)
        '"{word}" is an antonym of',          # T7  (new; "an antonym" vs "the antonym")
    ],
    "fr": [
        'Le contraire de "{word}" est',       # T0  (same as MC T0)
        "L'antonyme de \"{word}\" est",       # T1  (same as MC T1)
        '"{word}" est le contraire de',       # T2  (same as MC T2)
        '"{word}" est l\'antonyme de',        # T3  (same as MC T3)
        "L'opposé de \"{word}\" est",         # T4  (new)
        '"{word}" est l\'opposé de',          # T5  (new)
        'Le mot opposé à "{word}" est',       # T6  (new; uses "à")
        '"{word}" est un antonyme de',        # T7  (new; "un antonyme" vs "l'antonyme")
    ],
}


def generate_multilingual_circuits_b1_prompts(
    n_train: int = 96,
    n_test: int = 32,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate B1 multilingual antonym prompts: 8 templates per (concept, language) group.

    Extends multilingual_circuits (4 templates, T0-T3) with 4 new surface variants
    (T4-T7), providing 8 templates total per (concept, language) group.

    8 cross-language concepts (concept_idx=1 excluded: 'froid' = 2 tokens):
      indices 0, 2, 3, 4, 5, 6, 7, 8.
    8 templates × 2 languages = 16 prompts per concept → 128 total.

    Stratified split: for each of 16 (concept_index, language) groups, shuffle
    [0..7] template indices; assign first 2 shuffled indices to test and the
    remaining 6 to train. Test indices are drawn ONCE per concept (shared EN/FR)
    to preserve C3 patching alignment (source=EN, target=FR, same template_idx).
    → train = 48 EN + 48 FR = 96, test = 16 EN + 16 FR = 32.

    Fields: identical schema to multilingual_circuits (prompt, correct_answer,
    incorrect_answer, word, antonym, language, concept_index, template_idx,
    cross_lang_valid).

    Intervention axis: C3 (language swap) — same concept_index; test_tidxs are
    shared between EN and FR per concept so patching pairs are template-matched.
    """
    rng = random.Random(seed)
    train_prompts: List[Dict] = []
    test_prompts:  List[Dict] = []

    # 8 cross-language concepts (has_fr=True from _ML_CONCEPTS)
    mc_concepts = [
        (cidx, ew, ea, fw, fa)
        for cidx, ew, ea, fw, fa, hf in _ML_CONCEPTS
        if hf
    ]

    # Draw 2 test_tidxs ONCE per concept (shared between EN and FR) so that
    # C3 patching pairs always have matching template_idx in source and target.
    for cidx, en_word, en_ant, fr_word, fr_ant in mc_concepts:
        tidxs = list(range(8))
        rng.shuffle(tidxs)
        test_tidxs = set(tidxs[:2])  # 2 test templates per concept

        for lang in ["en", "fr"]:
            templates = _ML_ANT_TEMPLATES_B1[lang]
            word = en_word if lang == "en" else fr_word
            ant  = en_ant  if lang == "en" else fr_ant

            for tidx, tmpl in enumerate(templates):
                p = {
                    "prompt":           tmpl.format(word=word),
                    "correct_answer":   f" {ant}",
                    "incorrect_answer": f" {word}",
                    "word":             word,
                    "antonym":          ant,
                    "language":         lang,
                    "concept_index":    cidx,
                    "template_idx":     tidx,
                    "cross_lang_valid": True,
                }
                if tidx in test_tidxs:
                    test_prompts.append(p)
                else:
                    train_prompts.append(p)

    assert len(train_prompts) == 96, f"Expected 96 train, got {len(train_prompts)}"
    assert len(test_prompts)  == 32, f"Expected 32 test, got {len(test_prompts)}"

    return train_prompts, test_prompts


GENERATORS["multilingual_circuits_b1"] = generate_multilingual_circuits_b1_prompts


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
