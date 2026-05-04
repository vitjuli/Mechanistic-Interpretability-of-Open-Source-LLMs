"""
Generate candidate-state pilot prompts.

Two behaviours where the model must SELECT a candidate from an explicit set
using a filter/condition — not binary label classification.

Structure: candidate_set + filter_property → selected_candidate

  physics_particle_candidate_selection_mini  (32 prompts)
    Prompt gives 3 particles explicitly; model selects the one satisfying a filter.
    Candidate sets: {proton, neutron, electron}, {proton, electron, photon},
                    {electron, positron, photon}
    Filters: electric charge, particle family (lepton/boson), rest mass, decay role
    Answer tokens: electron, proton, neutron, photon, positron (all expected 1-token)

  physics_decay_product_selection_mini  (32 prompts)
    Prompt describes a decay process; model selects the product satisfying a filter.
    Processes: beta-minus (n→p+e+ν̄), beta-plus (p→n+e⁺+ν)
    Filters: charge, lepton, nuclear role, rest mass
    Answer tokens: electron, proton, positron, neutron (all expected 1-token)
    Note: antineutrino/neutrino appear in prompt text but never as answer tokens
          (they are multi-token: anti+neutrino / neut+rino)

Token audit (confirmed on CSD3 — see logs/cand-r3-* output):
  ' electron':  1 token  ✓
  ' proton':    1 token  ✓
  ' neutron':   1 token  ✓
  ' photon':    1 token  ✓
  ' positron':  1 token  expected ✓ (confirm at runtime)
  ' antineutrino': MULTI — used in prompt text only, never as answer token

Candidate ordering across families (positional balance):
  F1 (explicit list):     correct LAST  → others + [correct]
  F2 (contextual setup):  correct FIRST → [correct] + others
  F3 (filter-first):      correct MIDDLE → [other0, correct, other1]
  F4 (physics context):   original definition order
"""

import json
from pathlib import Path

OUT_DIR = Path("data/prompts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    pos_ans = records[0]["correct_answer"].strip()
    n_pos = sum(1 for r in records if r["correct_answer"].strip() == pos_ans)
    n_dup = len(records) - len({r["prompt"] for r in records})
    families = sorted({r.get("wording_family", "?") for r in records})
    targets = sorted({r.get("target_candidate", "?") for r in records})
    filters = sorted({r.get("filter_property", "?") for r in records})
    print(f"  Wrote {len(records)} prompts → {path.name}  |  duplicates: {n_dup}")
    print(f"    Target candidates: {targets}")
    print(f"    Filter properties: {filters}")
    print(f"    Families: {families}")


def _ordered(cs: list[str], correct: str, family_idx: int) -> tuple[str, str, str]:
    """Return (p1, p2, p3) with correct candidate at different positions per family."""
    others = [c for c in cs if c != correct]
    assert len(others) == 2, f"Expected 2 others, got {others}"
    o0, o1 = others[0], others[1]
    if family_idx == 0:   # F1: correct last
        return o0, o1, correct
    elif family_idx == 1:  # F2: correct first
        return correct, o0, o1
    elif family_idx == 2:  # F3: correct middle
        return o0, correct, o1
    else:                  # F4: original definition order
        return cs[0], cs[1], cs[2]


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour 1: physics_particle_candidate_selection_mini
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
PARTICLE_CASES = [
    # (candidate_set, filter_property, correct, incorrect_primary, filter_question, physics_context)
    # incorrect_primary = hardest distractor from the candidate set
    (["proton", "neutron", "electron"], "negative_charge", "electron", "proton",
     "has negative electric charge",
     "Three particles are central to atomic structure: the proton, the neutron, and the electron"),

    (["proton", "neutron", "electron"], "positive_charge", "proton", "electron",
     "carries positive electric charge",
     "An atomic nucleus is built from two types of nucleons, plus an orbiting electron"),

    (["proton", "neutron", "electron"], "neutral_charge", "neutron", "proton",
     "has zero (neutral) electric charge",
     "In the Standard Model, protons and electrons carry charge while one nucleon does not"),

    (["proton", "neutron", "electron"], "lepton", "electron", "proton",
     "belongs to the lepton family in the Standard Model",
     "Protons and neutrons are composite hadrons (made of quarks), while a third particle is a fundamental lepton"),

    (["proton", "electron", "photon"], "boson", "photon", "proton",
     "is a boson (obeys Bose-Einstein statistics)",
     "The electromagnetic force is mediated by a massless boson; protons and electrons are fermions"),

    (["proton", "electron", "photon"], "zero_rest_mass", "photon", "electron",
     "has zero rest mass",
     "One of these three particles is massless and always travels at exactly c"),

    (["electron", "positron", "photon"], "negative_charge", "electron", "positron",
     "has negative electric charge",
     "In pair production, a photon creates two particles with opposite charges"),

    (["electron", "positron", "photon"], "antiparticle_of_electron", "positron", "electron",
     "is the antiparticle of the electron",
     "Pair annihilation destroys an electron and its corresponding antiparticle to produce photons"),
]
# fmt: on

PARTICLE_FAMILY_NAMES = [
    "F1_explicit_list",
    "F2_contextual",
    "F3_filter_first",
    "F4_physics_context",
]


def particle_candidate_prompts() -> list[dict]:
    records = []
    for cs, fp, correct, incorrect, filter_q, context in PARTICLE_CASES:
        wrong_all = [c for c in cs if c != correct]
        for fi, fam in enumerate(PARTICLE_FAMILY_NAMES):
            p1, p2, p3 = _ordered(cs, correct, fi)

            if fam == "F1_explicit_list":
                prompt = (
                    f"The options are {p1}, {p2}, and {p3}. "
                    f"Which of these particles {filter_q}? Answer:"
                )
            elif fam == "F2_contextual":
                prompt = (
                    f"Consider three particles: the {p1}, the {p2}, and the {p3}. "
                    f"Among these, which one {filter_q}? Answer:"
                )
            elif fam == "F3_filter_first":
                prompt = (
                    f"Which particle {filter_q} — "
                    f"the {p1}, the {p2}, or the {p3}? Answer:"
                )
            else:  # F4
                prompt = (
                    f"{context}. "
                    f"Among the {p1}, {p2}, and {p3}, which one {filter_q}? Answer:"
                )

            records.append({
                "prompt": prompt,
                "correct_answer":   f" {correct}",
                "incorrect_answer": f" {incorrect}",
                "incorrect_answers": [f" {c}" for c in cs if c != correct],
                "candidate_set":    cs,
                "filter_property":  fp,
                "wording_family":   fam,
                "target_candidate": correct,
                "candidate_type":   "particle",
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour 2: physics_decay_product_selection_mini
# ─────────────────────────────────────────────────────────────────────────────
#
# The full product list may include multi-token items (antineutrino, neutrino)
# which appear only in the prompt TEXT, never as answer tokens.
# The answer candidates (for logprob comparison) are always single-token.
#
# fmt: off
DECAY_CASES = [
    # (process, full_products_text, answer_c1, answer_c2, filter_prop, correct, incorrect,
    #  filter_q, process_brief)
    # --- beta-minus: n → p + e⁻ + ν̄ ---
    ("beta-minus", "a proton, an electron, and an antineutrino",
     "proton", "electron", "negative_charge", "electron", "proton",
     "has negative electric charge",
     "a neutron decays into a proton, an electron, and an antineutrino (β-)"),

    ("beta-minus", "a proton, an electron, and an antineutrino",
     "proton", "electron", "lepton", "electron", "proton",
     "is a lepton (not a hadron)",
     "one of the charged products is a lepton, the other is a hadron"),

    ("beta-minus", "a proton, an electron, and an antineutrino",
     "proton", "electron", "remains_in_nucleus", "proton", "electron",
     "remains in the daughter nucleus after the decay",
     "one charged product is absorbed into the new nucleus, the other is emitted"),

    ("beta-minus", "a proton, an electron, and an antineutrino",
     "proton", "electron", "smaller_rest_mass", "electron", "proton",
     "has smaller rest mass (lighter particle)",
     "the proton (938 MeV/c²) and electron (0.511 MeV/c²) differ greatly in mass"),

    # --- beta-plus: p → n + e⁺ + ν ---
    ("beta-plus", "a neutron, a positron, and a neutrino",
     "neutron", "positron", "positive_charge", "positron", "neutron",
     "has positive electric charge",
     "a proton converts into a neutron, a positron, and a neutrino (β+)"),

    ("beta-plus", "a neutron, a positron, and a neutrino",
     "neutron", "positron", "lepton", "positron", "neutron",
     "is a lepton",
     "one of the two charged decay products is a lepton"),

    ("beta-plus", "a neutron, a positron, and a neutrino",
     "neutron", "positron", "remains_in_nucleus", "neutron", "positron",
     "remains in the daughter nucleus",
     "one product becomes the new nucleus while the positron and neutrino are emitted"),

    ("beta-plus", "a neutron, a positron, and a neutrino",
     "neutron", "positron", "smaller_rest_mass", "positron", "neutron",
     "has smaller rest mass",
     "the neutron (939 MeV/c²) and positron (0.511 MeV/c²) differ greatly in mass"),
]
# fmt: on

DECAY_FAMILY_NAMES = [
    "F1_explicit_products",
    "F2_contextual_process",
    "F3_filter_first",
    "F4_two_candidate",
]


def decay_product_prompts() -> list[dict]:
    records = []
    for (proc, full_prod, ac1, ac2, fp, correct, incorrect,
         filter_q, proc_brief) in DECAY_CASES:
        answer_cs = [ac1, ac2]
        wrong_all = [c for c in answer_cs if c != correct]

        for fi, fam in enumerate(DECAY_FAMILY_NAMES):
            # Vary which answer candidate appears first in the answer part
            if fi == 0:   # F1: correct last in answer clause
                qa1, qa2 = (wrong_all[0], correct)
            elif fi == 1: # F2: correct first
                qa1, qa2 = (correct, wrong_all[0])
            elif fi == 2: # F3: correct in "or" position (second)
                qa1, qa2 = (wrong_all[0], correct)
            else:         # F4: original definition order
                qa1, qa2 = ac1, ac2

            if fam == "F1_explicit_products":
                prompt = (
                    f"In {proc} decay, the products are {full_prod}. "
                    f"Among the {qa1} and the {qa2} (the charged products), "
                    f"which one {filter_q}? Answer:"
                )
            elif fam == "F2_contextual_process":
                prompt = (
                    f"In {proc} decay, {proc_brief}. "
                    f"Between the {qa1} and the {qa2}, "
                    f"which {filter_q}? Answer:"
                )
            elif fam == "F3_filter_first":
                prompt = (
                    f"In {proc} decay (products: {full_prod}), "
                    f"which charged product {filter_q} — "
                    f"the {qa1} or the {qa2}? Answer:"
                )
            else:  # F4: two-candidate, no extra context
                prompt = (
                    f"Consider the two main charged products of {proc} decay: "
                    f"the {qa1} and the {qa2}. "
                    f"Which {filter_q}? Answer:"
                )

            records.append({
                "prompt": prompt,
                "correct_answer":   f" {correct}",
                "incorrect_answer": f" {incorrect}",
                "incorrect_answers": [f" {c}" for c in answer_cs if c != correct],
                "candidate_set":    answer_cs,
                "filter_property":  fp,
                "wording_family":   fam,
                "target_candidate": correct,
                "candidate_type":   "decay_product",
                "process":          proc,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating candidate-state pilot prompts...")
    print()

    behaviours = {
        "physics_particle_candidate_selection_mini": particle_candidate_prompts(),
        "physics_decay_product_selection_mini":      decay_product_prompts(),
    }

    for beh, records in behaviours.items():
        path = OUT_DIR / f"{beh}_train.jsonl"
        write_jsonl(path, records)
        print()

    print("Done.")
    print()
    print("Token audit (confirm on CSD3):")
    print("  Expect ALL of the following to be 1-token:")
    for t in [" electron", " proton", " neutron", " photon", " positron"]:
        print(f"    {repr(t)}")
    print("  NEVER used as answer token (multi-token):")
    print("    ' antineutrino'  ' neutrino'")


if __name__ == "__main__":
    main()
