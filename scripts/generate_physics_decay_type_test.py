#!/usr/bin/env python3
"""
Generate the test split for physics_decay_type (36 prompts, 18 alpha + 18 beta).

All prompts are NEW — no overlap with train. Same 5 families, same format.
Balanced by concept and family.

  F0_test (8):  Direct physical properties (new phrasings)
  F1_test (10): Keyword-free (new, no alpha/beta/helium/electron in body)
  F2_test (8):  Named isotope pairs (different from train set)
  F3_test (6):  Experimental observables (new scenarios)
  F4_test (4):  Theoretical mechanisms (new angles)

Usage:
    python scripts/generate_physics_decay_type_test.py
"""

import json
from pathlib import Path

BEHAVIOUR = "physics_decay_type"
BEHAVIOUR_TYPE = "latent_state"
ANSWER_FORMAT = "x_or_y"
PROMPTS_DIR = Path("data/prompts")
OUT_PATH = PROMPTS_DIR / f"{BEHAVIOUR}_test.jsonl"


def make_prompt(text, concept, concept_index, template_idx, surface_family,
                keyword_free, difficulty="standard"):
    correct   = " alpha" if concept == "alpha_decay" else " beta"
    incorrect = " beta"  if concept == "alpha_decay" else " alpha"
    return {
        "prompt": text,
        "correct_answer": correct,
        "incorrect_answer": incorrect,
        "behaviour": BEHAVIOUR,
        "behaviour_type": BEHAVIOUR_TYPE,
        "physics_concept": concept,
        "template_idx": template_idx,
        "concept_index": concept_index,
        "difficulty": difficulty,
        "answer_format": ANSWER_FORMAT,
        "surface_family": surface_family,
        "keyword_free": keyword_free,
    }


# ── F0: Direct physical properties (4 alpha + 4 beta) ────────────────────────

F0_TEST_ALPHA = [
    "A particle with two protons and two neutrons is ejected from a heavy nucleus. Is the decay type alpha or beta?",
    "The parent nucleus loses four nucleons and two units of charge in a single emission. Is the decay type alpha or beta?",
    "The emitted cluster has the same composition as a fully ionised helium atom. Is the decay type alpha or beta?",
    "In the decay, the mass number drops by 4 and the proton number drops by 2. Is the decay type alpha or beta?",
]

F0_TEST_BETA = [
    "A high-energy electron is emitted from the nucleus alongside an antineutrino. Is the decay type alpha or beta?",
    "The atomic number increases by one unit while the mass number stays constant. Is the decay type alpha or beta?",
    "A down quark inside the nucleus converts to an up quark, releasing a lepton. Is the decay type alpha or beta?",
    "The daughter nucleus is an isobar of the parent with one extra proton. Is the decay type alpha or beta?",
]

# ── F1: Keyword-free (5 alpha + 5 beta) ──────────────────────────────────────
# No "alpha", "beta", "helium", "electron" in the description before the question tail.

F1_TEST_ALPHA = [
    "A bound cluster of two protons and two neutrons is ejected spontaneously from a heavy parent nucleus. Is the decay type alpha or beta?",
    "The emitted particle has charge +2e and nucleon number 4. Is the decay type alpha or beta?",
    "The daughter nucleus has atomic number Z−2 and mass number A−4 compared to the parent. Is the decay type alpha or beta?",
    "A nucleus spontaneously emits a tightly bound light nucleus with two units of positive charge. Is the decay type alpha or beta?",
    "The emitted particle has the same nucleon composition as an ordinary noble gas nucleus of mass 4. Is the decay type alpha or beta?",
]

F1_TEST_BETA = [
    "A nucleus emits a negatively charged lepton with negligible rest mass compared to a nucleon. Is the decay type alpha or beta?",
    "The daughter nucleus differs from the parent only in having one additional unit of charge. Is the decay type alpha or beta?",
    "A nuclear neutron transforms into a proton, and a light negatively charged particle is emitted. Is the decay type alpha or beta?",
    "The emitted particle is a first-generation lepton with charge −e. Is the decay type alpha or beta?",
    "The mass number is conserved but the charge number increases by 1 in the transition. Is the decay type alpha or beta?",
]

# ── F2: Named isotope pairs (4 alpha + 4 beta) ───────────────────────────────
# Different isotopes from train (train used: U-238, Th-230, Ra-226, Po-210, Am-241, Pu-239, Bi-214 etc.)

F2_TEST_ALPHA = [
    "Curium-244 decays to plutonium-240. Is the decay type alpha or beta?",
    "Bismuth-212 undergoes decay to thallium-208. Is the decay type alpha or beta?",
    "Fermium-257 decays to californium-253. Is the decay type alpha or beta?",
    "Neptunium-237 transforms into protactinium-233. Is the decay type alpha or beta?",
]

F2_TEST_BETA = [
    "Cobalt-60 decays to nickel-60. Is the decay type alpha or beta?",
    "Tritium decays to helium-3. Is the decay type alpha or beta?",
    "Potassium-40 undergoes decay to calcium-40. Is the decay type alpha or beta?",
    "Nickel-63 decays to copper-63. Is the decay type alpha or beta?",
]

# ── F3: Experimental observables (3 alpha + 3 beta) ──────────────────────────

F3_TEST_ALPHA = [
    "A particle emitted in the decay is completely stopped by a few centimetres of air and cannot penetrate even thin foil. Is the decay type alpha or beta?",
    "The emitted radiation produces a dense, straight track of fixed length in a cloud chamber. Is the decay type alpha or beta?",
    "A detector placed behind a sheet of paper registers no counts from this radiation. Is the decay type alpha or beta?",
]

F3_TEST_BETA = [
    "The emitted particle has a continuous range of kinetic energies up to a well-defined maximum. Is the decay type alpha or beta?",
    "The radiation is deflected by a magnetic field toward the positive plate in an electric field experiment. Is the decay type alpha or beta?",
    "The track in a cloud chamber is long, wispy, and curved — typical of a low-ionisation, high-penetration particle. Is the decay type alpha or beta?",
]

# ── F4: Theoretical mechanisms (2 alpha + 2 beta) ─────────────────────────────

F4_TEST_ALPHA = [
    "The Geiger–Nuttall law relates the half-life to the energy of the emitted particle via quantum tunnelling probability. Is the decay type alpha or beta?",
    "The preformation factor and barrier penetrability both appear in the theoretical rate formula for this decay mode. Is the decay type alpha or beta?",
]

F4_TEST_BETA = [
    "The Fermi theory of this decay treats the process as a four-fermion point interaction with coupling constant G_F. Is the decay type alpha or beta?",
    "The Q-value of this transition is shared among three particles due to the simultaneous emission of a neutrino. Is the decay type alpha or beta?",
]


def build_test_dataset():
    records = []
    for i, text in enumerate(F0_TEST_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F0", False))
    for i, text in enumerate(F0_TEST_BETA):
        records.append(make_prompt(text, "beta_decay",  1, i, "F0", False))
    for i, text in enumerate(F1_TEST_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F1", True))
    for i, text in enumerate(F1_TEST_BETA):
        records.append(make_prompt(text, "beta_decay",  1, i, "F1", True))
    for i, text in enumerate(F2_TEST_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F2", False))
    for i, text in enumerate(F2_TEST_BETA):
        records.append(make_prompt(text, "beta_decay",  1, i, "F2", False))
    for i, text in enumerate(F3_TEST_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F3", False, "indirect"))
    for i, text in enumerate(F3_TEST_BETA):
        records.append(make_prompt(text, "beta_decay",  1, i, "F3", False, "indirect"))
    for i, text in enumerate(F4_TEST_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F4", False, "hard"))
    for i, text in enumerate(F4_TEST_BETA):
        records.append(make_prompt(text, "beta_decay",  1, i, "F4", False, "hard"))
    return records


def sanity_check(records):
    from collections import Counter
    n_alpha = sum(1 for r in records if r["physics_concept"] == "alpha_decay")
    n_beta  = sum(1 for r in records if r["physics_concept"] == "beta_decay")
    n_kf    = sum(1 for r in records if r["keyword_free"])
    fams    = dict(sorted(Counter(r["surface_family"] for r in records).items()))

    print(f"Total prompts  : {len(records)}")
    print(f"  alpha_decay  : {n_alpha}")
    print(f"  beta_decay   : {n_beta}")
    print(f"  keyword_free : {n_kf}")
    print(f"  by family    : {fams}")

    # Keyword-free integrity
    tail = "is the decay type alpha or beta?"
    kw_forbidden = ["alpha", "beta", "helium", "electron"]
    violations = []
    for r in records:
        if r["keyword_free"]:
            body = r["prompt"].lower()
            idx = body.find(tail)
            if idx >= 0:
                body = body[:idx]
            for kw in kw_forbidden:
                if kw in body:
                    violations.append((r["surface_family"], r["physics_concept"],
                                       r["template_idx"], kw, r["prompt"]))
    if violations:
        print("\nWARNING — keyword_free violations:")
        for v in violations:
            print(f"  [{v[0]} t{v[2]} {v[1]}] found '{v[3]}': {v[4]}")
    else:
        print("  keyword_free check: OK")

    # Train overlap check
    train_path = PROMPTS_DIR / f"{BEHAVIOUR}_train.jsonl"
    if train_path.exists():
        train_texts = {
            json.loads(l)["prompt"].lower().strip()
            for l in train_path.read_text().splitlines() if l.strip()
        }
        overlaps = [r for r in records if r["prompt"].lower().strip() in train_texts]
        if overlaps:
            print(f"\nWARNING — {len(overlaps)} test prompts overlap with train:")
            for r in overlaps:
                print(f"  {r['prompt'][:80]}")
        else:
            print("  train/test overlap check: OK (0 overlaps)")


def main():
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    records = build_test_dataset()
    sanity_check(records)

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} prompts → {OUT_PATH}")


if __name__ == "__main__":
    main()
