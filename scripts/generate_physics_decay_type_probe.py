"""
Generate the physics_decay_type_probe latent-state probing corpus.

Writes: data/prompts/physics_decay_type_probe_train.jsonl

Design: docs/latent_probing_corpus_phase2.md
Framework: docs/latent_probing_framework_v2.md

Three-level structure:
  Level 1 — atomic static cues (12 groups, 8–12 variants each)
  Level 2 — relation / process descriptions (16 groups, 6–10 variants)
  Level 3 — concept / latent-state probes (8 groups, 6–8 variants)
  Auxiliary — world-knowledge, conflict, isotope, partial cues (56 prompts)

Usage:
  python scripts/generate_physics_decay_type_probe.py [--include_auxiliary]
  python scripts/generate_physics_decay_type_probe.py --level 1   # Level 1 only
"""

import argparse
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/prompts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOUR      = "physics_decay_type_probe"
ANCHOR_SENT    = "A nucleus undergoes a decay process."
QUESTION       = "Is the decay type alpha or beta?"
CORRECT_ALPHA  = " alpha"
CORRECT_BETA   = " beta"

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt(description: str) -> str:
    return f"{ANCHOR_SENT} {description} {QUESTION}"


def record(
    group_id: str,
    variant: int,
    description: str,
    label: str,            # "alpha" or "beta"
    level: int,
    level_label: str,
    cue_type: str | None          = None,
    cue_set: list[str]            = (),
    relation_type: str | None     = None,
    concept_route: str | None     = None,
    test_type: list[str]          = ("SE", "IC"),
    abstraction_level: int        = 1,
    inference_steps: int          = 1,
    has_alpha_keyword: bool       = False,
    has_beta_keyword: bool        = False,
    has_helium_ref: bool          = False,
    has_electron_ref: bool        = False,
    has_isotope: bool             = False,
    keyword_type: str | None      = None,
    contrastive_pair_id: str | None = None,
    contrastive_role: str | None  = None,
    is_anchor: bool               = False,
    is_kw_variant: bool           = False,
    is_auxiliary: bool            = False,
    prompt_format: str            = "emission",
    evidence_completeness: str    = "single",
    is_uniquely_determining: bool = True,
    difficulty: str               = "easy",
    notes: str                    = "",
) -> dict:
    correct   = CORRECT_ALPHA if label == "alpha" else CORRECT_BETA
    incorrect = CORRECT_BETA  if label == "alpha" else CORRECT_ALPHA
    concept   = "alpha_decay" if label == "alpha" else "beta_decay"
    concept_idx = 0 if label == "alpha" else 1

    return {
        "prompt_id":       f"{group_id}-v{variant:02d}",
        "prompt":          make_prompt(description),
        "correct_answer":  correct,
        "incorrect_answer": incorrect,
        "behaviour":       BEHAVIOUR,
        "behaviour_type":  "latent_state_probing",
        "physics_concept": concept,
        "concept_index":   concept_idx,
        "template_idx":    variant,
        "difficulty":      difficulty,
        "answer_format":   "x_or_y",
        # pipeline-compatible
        "surface_family":  group_id,
        "keyword_free":    not (has_alpha_keyword or has_beta_keyword or has_helium_ref or has_electron_ref),
        # framework fields
        "level":           level,
        "level_label":     level_label,
        "group_id":        group_id,
        "level_group":     group_id,
        "cue_type":        cue_type,
        "cue_set":         list(cue_set) if cue_set else ([cue_type] if cue_type else []),
        "relation_type":   relation_type,
        "concept_route":   concept_route,
        "wording_variant": variant,
        "test_type":       list(test_type),
        "abstraction_level": abstraction_level,
        "inference_steps": inference_steps,
        "has_alpha_keyword": has_alpha_keyword,
        "has_beta_keyword":  has_beta_keyword,
        "has_helium_ref":    has_helium_ref,
        "has_electron_ref":  has_electron_ref,
        "has_isotope":       has_isotope,
        "keyword_type":      keyword_type,
        "semantic_equivalence_group": group_id,
        "ic_concept_group":  label,
        "contrastive_pair_id": contrastive_pair_id,
        "contrastive_role":    contrastive_role,
        "is_anchor":           is_anchor,
        "is_kw_variant":       is_kw_variant,
        "is_auxiliary":        is_auxiliary,
        "prompt_format":       prompt_format,
        "evidence_completeness": evidence_completeness,
        "is_uniquely_determining": is_uniquely_determining,
        "notes": notes,
    }


def group(group_id, label, level, level_label, cue_type=None, relation_type=None,
          concept_route=None, descriptions=(), **kwargs):
    """Generate all variant records for one group."""
    records = []
    for i, desc in enumerate(descriptions, start=1):
        records.append(record(
            group_id=group_id,
            variant=i,
            description=desc,
            label=label,
            level=level,
            level_label=level_label,
            cue_type=cue_type,
            relation_type=relation_type,
            concept_route=concept_route,
            **kwargs,
        ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  LEVEL 1 — Atomic cues
# ─────────────────────────────────────────────────────────────────────────────

def level1_alpha() -> list[dict]:
    prompts = []

    # L1-A1 — emitted charge +2e (12 variants)
    prompts += group("L1-A1", "alpha", 1, "atomic_cue",
        cue_type="emitted_charge_plus2", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle carries a charge of +2e.",
            "The particle released has an electric charge of positive 2 elementary units.",
            "The emitted particle is doubly positively charged.",
            "Two units of positive elementary charge are carried away by the emitted particle.",
            "The emitted particle has an electric charge equal in magnitude to twice that of a proton, and the same sign.",
            "The emitted particle has a charge of plus two in units of the elementary charge.",
            "The emitted particle carries a positive electric charge of 2e.",
            "A doubly positively charged particle is emitted.",
            "The emitted particle's electric charge is +2e.",
            "The emitted particle bears a charge of twice the proton charge, positive.",
            "The electric charge carried by the emitted particle is positive and equal to two elementary units.",
            "The emitted particle has a net positive charge of two elementary units.",
        ])

    # L1-A2 — emitted mass number = 4 (12 variants)
    prompts += group("L1-A2", "alpha", 1, "atomic_cue",
        cue_type="emitted_mass4", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has a mass number of 4.",
            "The particle released has a mass number equal to four.",
            "The emitted particle contains a total of 4 nucleons.",
            "The emitted particle has atomic mass unit 4.",
            "The mass number of the emitted particle is 4.",
            "The emitted particle contains exactly 4 nucleons.",
            "The total number of nucleons in the emitted particle is 4.",
            "The emitted particle has a nucleon number of 4.",
            "The emitted particle consists of 4 nucleons in total.",
            "The number of nucleons in the emitted particle is exactly four.",
            "The emitted particle has a mass equivalent to 4 nucleon masses.",
            "The emitted particle contains four nucleons total.",
        ])

    # L1-A3 — daughter Z decreases by 2 (12 variants)
    prompts += group("L1-A3", "alpha", 1, "atomic_cue",
        cue_type="daughter_z_minus2", abstraction_level=2, inference_steps=2,
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has an atomic number 2 less than the parent nucleus.",
            "After the decay, the product nucleus has Z reduced by 2.",
            "The atomic number of the daughter nucleus is 2 fewer than that of the parent.",
            "The daughter element is 2 positions to the left of the parent on the periodic table.",
            "The resulting nucleus has 2 fewer protons than the original nucleus.",
            "The product of this decay has a proton count decreased by 2.",
            "The daughter nucleus has a proton number 2 lower than the parent.",
            "After this decay, the element moves 2 positions to the left in the periodic table.",
            "The number of protons in the product nucleus is 2 less than in the original.",
            "The atomic number decreases by 2 in this decay.",
            "The product nucleus contains 2 fewer protons than the parent nucleus.",
            "The daughter's atomic number is reduced by 2 compared to the parent's.",
        ])

    # L1-A4 — daughter A decreases by 4 (10 variants)
    prompts += group("L1-A4", "alpha", 1, "atomic_cue",
        cue_type="daughter_a_minus4", abstraction_level=2, inference_steps=2,
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has a mass number 4 less than the parent.",
            "The mass number decreases by 4 during this decay.",
            "After the decay, the product nucleus has 4 fewer nucleons than the parent.",
            "The total nucleon count of the daughter nucleus is 4 less than that of the parent.",
            "The daughter nucleus has lost 4 nucleons compared to the parent.",
            "The mass number of the product nucleus is 4 lower than that of the original.",
            "The decay results in a daughter nucleus with 4 fewer nucleons.",
            "The mass number is reduced by 4 in this process.",
            "The product nucleus has a nucleon number 4 less than the parent.",
            "The daughter has 4 fewer nucleons than the parent nucleus.",
        ])

    # L1-A5 — emitted particle proton count = 2 [NEW] (8 variants)
    prompts += group("L1-A5", "alpha", 1, "atomic_cue",
        cue_type="emitted_2protons", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        notes="Tests whether 'proton count=2' activates same feature as 'charge=+2e'.",
        descriptions=[
            "The emitted particle contains 2 protons.",
            "The particle released includes two protons in its composition.",
            "The emitted particle has a proton count of 2.",
            "There are 2 protons in the emitted particle.",
            "The emitted particle contains exactly two protons.",
            "The proton number of the emitted particle is 2.",
            "Two protons are contained within the emitted particle.",
            "The emitted particle is composed of, among other nucleons, exactly 2 protons.",
        ])

    # L1-A6 — emitted particle neutron count = 2 [NEW] (8 variants)
    prompts += group("L1-A6", "alpha", 1, "atomic_cue",
        cue_type="emitted_2neutrons", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        notes="Tests neutron count as an independent atomic cue.",
        descriptions=[
            "The emitted particle contains 2 neutrons.",
            "The particle released includes two neutrons in its composition.",
            "The emitted particle has a neutron count of 2.",
            "There are 2 neutrons in the emitted particle.",
            "The emitted particle contains exactly two neutrons.",
            "The neutron number of the emitted particle is 2.",
            "Two neutrons are contained within the emitted particle.",
            "The emitted particle is composed of two neutrons, along with other nucleons.",
        ])

    return prompts


def level1_beta() -> list[dict]:
    prompts = []

    # L1-B1 — emitted charge -1e (12 variants)
    prompts += group("L1-B1", "beta", 1, "atomic_cue",
        cue_type="emitted_charge_minus1", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has a charge of −1e.",
            "The emitted particle carries a single unit of negative electric charge.",
            "The particle released has an electric charge of minus one elementary unit.",
            "The emitted particle is negatively charged with a magnitude equal to the proton charge.",
            "The emitted particle has charge −e, where e is the elementary charge unit.",
            "One unit of negative charge leaves the nucleus carried by the emitted particle.",
            "The emitted particle carries a charge of −1 in units of the elementary charge.",
            "The emitted particle is singly negatively charged.",
            "The emitted particle has an electric charge of negative one elementary unit.",
            "The emitted particle carries a charge of minus one proton charge.",
            "The emitted particle has a charge of −e.",
            "A singly negatively charged particle is emitted.",
        ])

    # L1-B2 — daughter Z+1 (12 variants)
    prompts += group("L1-B2", "beta", 1, "atomic_cue",
        cue_type="daughter_z_plus1", abstraction_level=2, inference_steps=2,
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has an atomic number 1 greater than the parent.",
            "After the decay, the product nucleus has Z increased by 1.",
            "The resulting nucleus has one more proton than the original nucleus.",
            "The daughter element is one position to the right of the parent on the periodic table.",
            "The atomic number of the daughter is exactly 1 greater than that of the parent.",
            "The decay increases the proton count of the nucleus by exactly 1.",
            "The daughter nucleus has a proton number 1 higher than the parent.",
            "After this decay, the element moves 1 position to the right in the periodic table.",
            "The number of protons in the product nucleus is 1 more than in the original.",
            "The atomic number increases by 1 in this decay.",
            "The product nucleus contains 1 more proton than the parent nucleus.",
            "The daughter's atomic number is higher by 1 than the parent's.",
        ])

    # L1-B3 — daughter A unchanged (12 variants)
    prompts += group("L1-B3", "beta", 1, "atomic_cue",
        cue_type="daughter_a_unchanged", abstraction_level=1, inference_steps=1,
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has the same mass number as the parent.",
            "The mass number is conserved during this decay.",
            "After the decay, the product nucleus has the same total number of nucleons as the parent.",
            "The total nucleon count does not change during this decay.",
            "The daughter nucleus retains the same mass number as the original.",
            "No nucleons are lost from the nucleus during this process; only the proton-neutron balance changes.",
            "The mass number of the daughter is identical to that of the parent.",
            "The number of nucleons in the daughter nucleus equals that of the parent nucleus.",
            "The mass number is unchanged by this decay.",
            "The product nucleus has the same nucleon count as the original.",
            "The daughter's mass number is the same as the parent's mass number.",
            "The total number of nucleons is preserved in this decay.",
        ])

    # L1-B4 — antineutrino emitted (10 variants)
    prompts += group("L1-B4", "beta", 1, "atomic_cue",
        cue_type="antineutrino_emitted", abstraction_level=2, inference_steps=2,
        prompt_format="property", difficulty="medium",
        notes="Requires 1 inference step: antineutrino -> beta-minus.",
        descriptions=[
            "An antineutrino is emitted alongside the main particle during this decay.",
            "The decay produces an antineutrino as one of its products.",
            "An electron antineutrino is released during this nuclear process.",
            "Among the particles emitted, there is an antineutrino carrying away energy and momentum.",
            "The decay products include an antineutrino with very small rest mass.",
            "The decay is accompanied by the emission of an antineutrino.",
            "An antineutrino is one of the particles produced in this decay.",
            "The antineutrino is emitted alongside the charged particle in this decay.",
            "One of the decay products is an antineutrino.",
            "A neutral particle with very small mass and opposite lepton number to a neutrino is emitted.",
        ])

    # L1-B5 — emitted particle has negligible rest mass [NEW] (8 variants)
    prompts += group("L1-B5", "beta", 1, "atomic_cue",
        cue_type="emitted_mass_negligible", abstraction_level=1, inference_steps=1,
        prompt_format="emission", difficulty="easy",
        notes="Atomic mass fact for beta; pairs with L1-A2 (mass=4 for alpha).",
        descriptions=[
            "The emitted particle has negligible rest mass.",
            "The rest mass of the emitted particle is effectively zero.",
            "The emitted particle has near-zero rest mass.",
            "The emitted particle is essentially massless.",
            "The rest mass of the emitted particle is negligible compared to a nucleon.",
            "The emitted particle carries essentially no rest mass.",
            "The mass of the emitted particle is negligible in nuclear physics terms.",
            "The emitted particle has a rest mass that is vanishingly small.",
        ])

    # L1-B6 — daughter has one fewer neutron [NEW, derived] (8 variants)
    prompts += group("L1-B6", "beta", 1, "atomic_cue",
        cue_type="daughter_n_minus1", abstraction_level=3, inference_steps=3,
        prompt_format="daughter", difficulty="medium",
        notes="Derived cue: ΔN=-1 follows from Z+1 and A-unchanged. Tests shortcut.",
        descriptions=[
            "The daughter nucleus has one fewer neutron than the parent.",
            "After the decay, the neutron count of the nucleus decreases by 1.",
            "The product nucleus contains one fewer neutron than the original.",
            "The number of neutrons in the daughter nucleus is 1 less than in the parent.",
            "The daughter loses exactly one neutron compared to the parent nucleus.",
            "The neutron number of the daughter is 1 lower than that of the parent.",
            "The neutron count decreases by 1 during this decay.",
            "There is one fewer neutron in the product nucleus than in the original.",
        ])

    return prompts


def level1_keyword_pairs() -> list[dict]:
    """12 keyword-presence pairs — keyword-free v1 + direct-keyword v2."""
    prompts = []

    pairs = [
        # (pair_id, label, group_id, kw_free_desc, kw_direct_desc, kw_type)
        ("KW-A1", "alpha", "L1-A1",
            "The emitted particle carries a charge of +2e.",
            "An alpha particle with charge +2e is emitted.",
            "alpha_particle"),
        ("KW-A2", "alpha", "L1-A2",
            "The emitted particle has a mass number of 4.",
            "An alpha particle with mass number 4 is emitted.",
            "alpha_particle"),
        ("KW-A3", "alpha", "L1-A3",
            "The daughter nucleus has an atomic number 2 less than the parent nucleus.",
            "An alpha decay gives a daughter nucleus with Z reduced by 2.",
            "alpha_decay"),
        ("KW-A4", "alpha", "L1-A5",
            "The emitted particle contains 2 protons.",
            "The emitted alpha particle contains 2 protons.",
            "alpha_particle"),
        ("KW-A5", "alpha", "L1-A6",
            "The emitted particle contains 2 neutrons.",
            "The emitted alpha particle contains 2 neutrons.",
            "alpha_particle"),
        ("KW-A6", "alpha", "L1-A4",
            "The daughter nucleus has a mass number 4 less than the parent.",
            "An alpha particle is emitted; the daughter has A reduced by 4.",
            "alpha_particle"),
        ("KW-B1", "beta", "L1-B1",
            "The emitted particle has a charge of −1e.",
            "An electron with charge −1e is emitted from the nucleus.",
            "electron"),
        ("KW-B2", "beta", "L1-B2",
            "The daughter nucleus has an atomic number 1 greater than the parent.",
            "A beta decay gives a daughter nucleus with Z increased by 1.",
            "beta_decay"),
        ("KW-B3", "beta", "L1-B3",
            "The daughter nucleus has the same mass number as the parent.",
            "A beta particle is emitted; the daughter has the same mass number as the parent.",
            "beta_particle"),
        ("KW-B4", "beta", "L1-B4",
            "An antineutrino is emitted alongside the main particle.",
            "A beta particle and antineutrino are both emitted.",
            "beta_particle"),
        ("KW-B5", "beta", "L1-B5",
            "The emitted particle has negligible rest mass.",
            "A beta particle with negligible rest mass is emitted.",
            "beta_particle"),
        ("KW-B6", "beta", "L1-B6",
            "The daughter nucleus has one fewer neutron than the parent.",
            "A beta decay results in the daughter having one fewer neutron.",
            "beta_decay"),
    ]

    for pair_id, label, src_group, kw_free, kw_direct, kw_type in pairs:
        # v1: keyword-free
        prompts.append(record(
            group_id=f"{pair_id}", variant=1, description=kw_free,
            label=label, level=1, level_label="atomic_cue",
            cue_type=None, test_type=["KW"],
            abstraction_level=1, inference_steps=1,
            has_alpha_keyword=False, has_beta_keyword=False,
            keyword_type=None, is_kw_variant=True,
            prompt_format="emission", difficulty="easy",
            notes=f"KW pair v1 (keyword-free) for {src_group}",
        ))
        # v2: direct keyword
        has_alpha = "alpha" in kw_type
        has_beta  = "beta" in kw_type or "electron" in kw_type
        has_elec  = "electron" in kw_type
        prompts.append(record(
            group_id=f"{pair_id}", variant=2, description=kw_direct,
            label=label, level=1, level_label="atomic_cue",
            cue_type=None, test_type=["KW"],
            abstraction_level=1, inference_steps=1,
            has_alpha_keyword=has_alpha, has_beta_keyword=has_beta,
            has_electron_ref=has_elec,
            keyword_type="direct", is_kw_variant=True,
            prompt_format="emission", difficulty="easy",
            notes=f"KW pair v2 (direct keyword) for {src_group}",
        ))

    return prompts


def level1_contrastive_pairs() -> list[dict]:
    """6 Level-1 contrastive pairs."""
    prompts = []

    pairs = [
        ("CP-L1-01",
            "alpha", "The emitted particle has a charge of +2e. The particle is not a free proton or neutron.",
            "beta",  "The emitted particle has a charge of −1e. The particle is not a free proton or neutron.",
            "charge sign and magnitude"),
        ("CP-L1-02",
            "alpha", "The daughter nucleus has an atomic number 2 less than the parent.",
            "beta",  "The daughter nucleus has an atomic number 1 greater than the parent.",
            "direction and magnitude of Z change"),
        ("CP-L1-03",
            "alpha", "The daughter nucleus has a mass number 4 less than the parent.",
            "beta",  "The daughter nucleus has the same mass number as the parent.",
            "whether A changes"),
        ("CP-L1-04",
            "alpha", "The emitted particle has a mass number of 4.",
            "beta",  "The emitted particle has negligible rest mass.",
            "emitted particle mass"),
        ("CP-L1-05",
            "alpha", "The emitted particle contains 2 protons.",
            "beta",  "The emitted particle contains no protons.",
            "proton content of emitted particle"),
        ("CP-L1-06",
            "alpha", "The emitted particle contains 2 neutrons.",
            "beta",  "The emitted particle contains no neutrons.",
            "neutron content of emitted particle"),
    ]

    for pair_id, al, ad, bl, bd, discriminant in pairs:
        prompts.append(record(
            group_id=pair_id, variant=1, description=ad,
            label=al, level=1, level_label="atomic_cue",
            test_type=["CP"], abstraction_level=1, inference_steps=1,
            contrastive_pair_id=pair_id, contrastive_role="alpha_member",
            prompt_format="emission", difficulty="easy",
            notes=f"CP discriminant: {discriminant}",
        ))
        prompts.append(record(
            group_id=pair_id, variant=2, description=bd,
            label=bl, level=1, level_label="atomic_cue",
            test_type=["CP"], abstraction_level=1, inference_steps=1,
            contrastive_pair_id=pair_id, contrastive_role="beta_member",
            prompt_format="emission", difficulty="easy",
            notes=f"CP discriminant: {discriminant}",
        ))

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
#  LEVEL 2 — Relations and processes
# ─────────────────────────────────────────────────────────────────────────────

def level2_alpha() -> list[dict]:
    prompts = []

    # L2-AR1 — composition: 2 protons + 2 neutrons (10 variants)
    prompts += group("L2-AR1", "alpha", 2, "relation_process",
        relation_type="composition_2p2n", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_2protons", "emitted_2neutrons"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle contains 2 protons and 2 neutrons.",
            "The particle released from the nucleus is composed of two protons and two neutrons.",
            "The emitted object has proton number 2 and neutron number 2.",
            "The emitted particle contains equal numbers of protons and neutrons, with 2 of each.",
            "The emitted cluster consists of two positively charged nucleons and two neutral nucleons.",
            "The emitted particle has mass number 4 and is made up of exactly 2 protons and 2 neutrons bound together.",
            "The emitted particle is a cluster of 2 protons and 2 neutrons.",
            "The particle emitted consists of two protons and two neutrons.",
            "The emitted particle is composed of 2 protons and 2 neutrons, with no other nucleons.",
            "Two protons and two neutrons together form the emitted particle.",
        ])

    # L2-AR2 — charge + mass (10 variants)
    prompts += group("L2-AR2", "alpha", 2, "relation_process",
        relation_type="charge_plus_mass", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_charge_plus2", "emitted_mass4"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has charge +2e and mass number 4.",
            "The emitted particle carries two units of positive charge and contains 4 nucleons.",
            "The particle released is doubly positively charged and has a mass number of 4.",
            "The emitted particle carries +2 units of electric charge and has atomic mass 4.",
            "The emitted particle has charge +2e and contains 4 nucleons in total.",
            "A particle with charge +2e and mass number 4 is emitted.",
            "The emitted particle has +2e charge and a nucleon number of 4.",
            "The emitted particle is doubly positive with mass number 4.",
            "The emitted particle carries +2 elementary units of charge and has 4 nucleons.",
            "The emitted particle has two units of positive charge and exactly 4 nucleons.",
        ])

    # L2-AR3 — daughter Z-2 AND A-4 (10 variants)
    prompts += group("L2-AR3", "alpha", 2, "relation_process",
        relation_type="daughter_z_and_a_alpha", abstraction_level=2, inference_steps=2,
        cue_set=["daughter_z_minus2", "daughter_a_minus4"],
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has atomic number 2 less and mass number 4 less than the parent.",
            "After the decay, the product nucleus has Z reduced by 2 and A reduced by 4.",
            "The resulting nucleus has 2 fewer protons and 4 fewer nucleons than the original.",
            "The daughter loses 2 protons and 4 total nucleons relative to the parent nucleus.",
            "The decay reduces the atomic number by 2 and the mass number by 4.",
            "The product nucleus is 2 atomic numbers lower and 4 mass units lighter than the parent.",
            "The daughter has Z−2 and A−4 relative to the parent.",
            "The daughter nucleus has both 2 fewer protons and 4 fewer nucleons than the parent.",
            "After this decay, Z decreases by 2 and A decreases by 4.",
            "The product nucleus has a proton number 2 lower and a nucleon number 4 lower than the original.",
        ])

    # L2-AR4 — composition + daughter change (8 variants)
    prompts += group("L2-AR4", "alpha", 2, "relation_process",
        relation_type="composition_plus_daughter", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_2protons", "emitted_2neutrons", "daughter_z_minus2", "daughter_a_minus4"],
        prompt_format="emission", difficulty="easy",
        evidence_completeness="combination",
        descriptions=[
            "The emitted particle contains 2 protons and 2 neutrons, and the daughter nucleus has Z−2 and A−4.",
            "A cluster of 2 protons and 2 neutrons is emitted, reducing the parent's Z by 2 and A by 4.",
            "The emitted particle is a bound system of 2 protons and 2 neutrons, leaving a daughter with 2 fewer protons and 4 fewer nucleons.",
            "The particle emitted consists of 2 protons and 2 neutrons, producing a daughter nucleus 2 atomic numbers and 4 mass units lighter.",
            "The emitted cluster of 2 protons and 2 neutrons results in a daughter with Z−2 and A−4.",
            "A 2p+2n cluster is released, shifting the element 2 places left and reducing the mass number by 4.",
            "The emitted 4-nucleon cluster (2p, 2n) leaves a daughter with Z−2 and A−4.",
            "Two protons and two neutrons are emitted as a bound cluster; the daughter has 2 fewer protons and 4 fewer nucleons.",
        ])

    # L2-AR5 — charge + daughter Z-change [NEW] (8 variants)
    prompts += group("L2-AR5", "alpha", 2, "relation_process",
        relation_type="charge_plus_daughter_z", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_charge_plus2", "daughter_z_minus2"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has charge +2e and the daughter nucleus has atomic number 2 less than the parent.",
            "The emitted particle carries +2e of charge and the product nucleus has Z reduced by 2.",
            "The particle emitted is doubly positively charged and the resulting element moves 2 positions to the left in the periodic table.",
            "The emitted particle has +2 units of electric charge, and the daughter nucleus has 2 fewer protons than the parent.",
            "A +2e particle is emitted, and the daughter has Z−2.",
            "The emitted particle carries charge +2e, leaving a daughter nucleus with atomic number decreased by 2.",
            "The decay releases a +2e particle, moving the element 2 places to the left.",
            "Charge +2e is carried away by the emitted particle; the daughter has Z reduced by 2.",
        ])

    # L2-AR6 — mass + daughter A-change [NEW] (8 variants)
    prompts += group("L2-AR6", "alpha", 2, "relation_process",
        relation_type="mass_plus_daughter_a", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_mass4", "daughter_a_minus4"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has mass number 4 and the daughter nucleus has mass number 4 less than the parent.",
            "The particle released has 4 nucleons, and the daughter has 4 fewer nucleons than the parent.",
            "The emitted particle has atomic mass 4, and the decay reduces A by 4.",
            "A particle with 4 nucleons is emitted, leaving a daughter with nucleon count decreased by 4.",
            "Mass number 4 is carried away; the daughter has A−4.",
            "The emitted particle contains 4 nucleons, and the daughter nucleus loses 4 nucleons.",
            "The decay emits a 4-nucleon particle, reducing the mass number of the daughter by 4.",
            "The emitted particle has mass number 4; accordingly, the daughter has 4 fewer nucleons.",
        ])

    # L2-AR7 — narrative ejection framing [NEW, no numbers] (6 variants)
    prompts += group("L2-AR7", "alpha", 2, "relation_process",
        relation_type="narrative_ejection", abstraction_level=3, inference_steps=3,
        cue_set=[], prompt_format="process", difficulty="medium",
        notes="Bridge L2→L3: describes alpha via narrative without numeric cues.",
        descriptions=[
            "A small, tightly bound cluster of nuclear matter is ejected from the nucleus.",
            "The nucleus ejects a compact fragment consisting entirely of nucleons.",
            "A small nuclear cluster is released from the parent nucleus during this decay.",
            "The decay proceeds by the emission of a small, tightly bound nuclear fragment.",
            "A bound nuclear cluster leaves the nucleus, carrying away part of its mass and charge.",
            "The nucleus expels a compact cluster of nucleons as a single unit.",
        ])

    # L2-AR8 — full emitted-particle specification [NEW] (8 variants)
    prompts += group("L2-AR8", "alpha", 2, "relation_process",
        relation_type="full_emitted_alpha_spec", abstraction_level=2, inference_steps=1,
        cue_set=["emitted_charge_plus2", "emitted_mass4", "emitted_2protons", "emitted_2neutrons"],
        prompt_format="emission", difficulty="easy",
        evidence_completeness="combination",
        descriptions=[
            "The emitted particle has charge +2e, mass number 4, and contains 2 protons and 2 neutrons.",
            "A particle with +2e charge and 4 nucleons (2 protons, 2 neutrons) is emitted.",
            "The emitted cluster carries +2 units of electric charge, consists of 2 protons and 2 neutrons.",
            "The emitted particle has +2e charge, 4 nucleons total, and a proton-neutron composition of 2:2.",
            "A doubly positively charged particle of mass 4, composed of 2 protons and 2 neutrons, is emitted.",
            "The emitted particle is a 4-nucleon cluster (2p + 2n) with charge +2e.",
            "The particle emitted: charge +2e, mass number 4, nucleon composition 2 protons + 2 neutrons.",
            "A +2e, mass-4 particle with 2 protons and 2 neutrons is released.",
        ])

    return prompts


def level2_beta() -> list[dict]:
    prompts = []

    # L2-BR1 — Z+1 AND A unchanged (10 variants)
    prompts += group("L2-BR1", "beta", 2, "relation_process",
        relation_type="z_plus1_a_unchanged", abstraction_level=2, inference_steps=2,
        cue_set=["daughter_z_plus1", "daughter_a_unchanged"],
        prompt_format="daughter", difficulty="easy",
        descriptions=[
            "The daughter nucleus has atomic number 1 greater and the same mass number as the parent.",
            "After the decay, Z increases by 1 and A remains unchanged.",
            "The product nucleus has one more proton than the parent but the same total nucleon count.",
            "The resulting nucleus is one element later in the periodic table but with identical mass number.",
            "The atomic number increases by 1 and the mass number stays the same.",
            "The decay shifts the element one place to the right in the periodic table while conserving mass number.",
            "The daughter has Z+1 and the same A as the parent.",
            "After this decay, the element advances by 1 position and the mass number is unchanged.",
            "Z increases by 1; A is conserved. The daughter is one element higher in the periodic table.",
            "The product nucleus has one more proton and the same number of nucleons as the original.",
        ])

    # L2-BR2 — neutron converts to proton (10 variants)
    prompts += group("L2-BR2", "beta", 2, "relation_process",
        relation_type="neutron_to_proton", abstraction_level=2, inference_steps=2,
        cue_set=[], prompt_format="process", difficulty="easy",
        notes="Process statement — moved from Level 1 (not a static property).",
        descriptions=[
            "A neutron inside the nucleus transforms into a proton.",
            "One neutron within the nucleus converts into a proton as part of this decay.",
            "The number of neutrons decreases by 1 and the number of protons increases by 1 during this decay.",
            "A neutron in the parent nucleus becomes a proton in the daughter nucleus.",
            "One unit of neutron number is converted to one unit of proton number during this decay.",
            "A neutron changes into a proton within the nucleus during this process.",
            "The decay involves a neutron converting into a proton.",
            "One neutron undergoes a transformation into a proton inside the nucleus.",
            "The nucleus loses one neutron and gains one proton during this decay.",
            "A neutron-to-proton conversion occurs within the nucleus.",
        ])

    # L2-BR3 — n→p + antineutrino (8 variants)
    prompts += group("L2-BR3", "beta", 2, "relation_process",
        relation_type="n_to_p_with_antineutrino", abstraction_level=2, inference_steps=2,
        cue_set=["antineutrino_emitted"],
        prompt_format="process", difficulty="easy",
        evidence_completeness="combination",
        descriptions=[
            "A neutron transforms into a proton, and an antineutrino is emitted during this process.",
            "The decay involves a neutron converting to a proton and the simultaneous emission of an antineutrino.",
            "One neutron becomes a proton, producing an antineutrino among the decay products.",
            "The process converts a neutron into a proton while releasing an electron antineutrino.",
            "A neutron changes into a proton and, alongside the charged particle emitted, an antineutrino exits the nucleus.",
            "A neutron-to-proton conversion occurs, accompanied by antineutrino emission.",
            "The neutron-to-proton process within the nucleus releases an antineutrino.",
            "During this decay, a neutron becomes a proton and an antineutrino is produced.",
        ])

    # L2-BR4 — charge + daughter Z-change (8 variants)
    prompts += group("L2-BR4", "beta", 2, "relation_process",
        relation_type="charge_plus_z_change", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_charge_minus1", "daughter_z_plus1"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has charge −1e and the daughter nucleus has atomic number 1 greater than the parent.",
            "The emitted particle carries a single unit of negative charge and the product nucleus has Z increased by 1.",
            "A negatively charged particle is emitted and the daughter element is one position to the right in the periodic table.",
            "The emitted particle has charge −1e and the number of protons in the nucleus increases by 1.",
            "A −1e particle is emitted, and the daughter has Z+1.",
            "The emitted particle carries charge −1e, leaving a daughter nucleus with atomic number increased by 1.",
            "The decay releases a −1e particle, moving the element 1 place to the right.",
            "Charge −1e leaves the nucleus; the daughter has Z+1.",
        ])

    # L2-BR5 — charge + A unchanged [NEW] (6 variants)
    prompts += group("L2-BR5", "beta", 2, "relation_process",
        relation_type="charge_plus_a_unchanged", abstraction_level=2, inference_steps=2,
        cue_set=["emitted_charge_minus1", "daughter_a_unchanged"],
        prompt_format="emission", difficulty="easy",
        descriptions=[
            "The emitted particle has charge −1e and the daughter nucleus has the same mass number as the parent.",
            "The emitted particle carries −1e of charge and A is conserved.",
            "A −1e particle is emitted; the daughter retains the same mass number.",
            "The particle released carries −1e of charge, and the nucleon count is unchanged.",
            "The emitted particle has charge −1e; no nucleons are gained or lost.",
            "A singly negatively charged particle is emitted, and the mass number does not change.",
        ])

    # L2-BR6 — daughter Z+1 + antineutrino [NEW] (8 variants)
    prompts += group("L2-BR6", "beta", 2, "relation_process",
        relation_type="z_plus1_with_antineutrino", abstraction_level=2, inference_steps=2,
        cue_set=["daughter_z_plus1", "antineutrino_emitted"],
        prompt_format="daughter", difficulty="medium",
        descriptions=[
            "The daughter nucleus has atomic number 1 greater than the parent, and an antineutrino is emitted.",
            "After the decay, Z increases by 1 and an antineutrino is produced.",
            "The product nucleus has one more proton, and an antineutrino is released.",
            "The daughter has Z+1 relative to the parent; the decay also produces an antineutrino.",
            "Z increases by 1 in the daughter nucleus, accompanied by antineutrino emission.",
            "The daughter is one atomic number higher, and an antineutrino is emitted alongside the decay.",
            "The nucleus gains one proton (Z+1) and releases an antineutrino.",
            "The decay increases Z by 1 and emits an antineutrino.",
        ])

    # L2-BR7 — n→p + daughter Z+1 [NEW] (8 variants)
    prompts += group("L2-BR7", "beta", 2, "relation_process",
        relation_type="n_to_p_plus_z_plus1", abstraction_level=2, inference_steps=2,
        cue_set=["daughter_z_plus1"],
        prompt_format="process", difficulty="easy",
        notes="Process+outcome: links n→p transformation explicitly to Z+1 result.",
        descriptions=[
            "A neutron converts into a proton, and the daughter nucleus has atomic number 1 greater than the parent.",
            "A neutron transforms into a proton within the nucleus; as a result, Z increases by 1.",
            "The neutron-to-proton conversion leads to a daughter with one more proton than the parent.",
            "One neutron becomes one proton, producing a daughter nucleus with Z+1.",
            "The conversion of a neutron into a proton results in the daughter having one more proton.",
            "A neutron-to-proton change raises Z by 1 in the daughter nucleus.",
            "The decay: a neutron becomes a proton, increasing the atomic number by 1.",
            "A neutron inside the nucleus converts to a proton, giving the daughter Z+1.",
        ])

    # L2-BR8 — full process specification [NEW] (8 variants)
    prompts += group("L2-BR8", "beta", 2, "relation_process",
        relation_type="full_beta_process_spec", abstraction_level=2, inference_steps=1,
        cue_set=["emitted_charge_minus1", "daughter_z_plus1", "daughter_a_unchanged", "antineutrino_emitted"],
        prompt_format="process", difficulty="easy",
        evidence_completeness="combination",
        descriptions=[
            "The emitted particle has charge −1e and the daughter nucleus has Z+1 and unchanged mass number, with an antineutrino also emitted.",
            "A particle with charge −1e is emitted, a neutron converts to a proton, Z increases by 1, A is conserved, and an antineutrino is released.",
            "The decay emits a particle of charge −1e accompanied by an antineutrino, increasing Z by 1 while A stays the same.",
            "The emitted particle carries −1e, the atomic number of the daughter is 1 greater than the parent, the mass number is unchanged, and an antineutrino exits.",
            "The decay: emitted particle charge −1e, daughter Z+1, A unchanged, antineutrino co-emitted.",
            "A −1e particle and antineutrino are released; Z increases by 1, A is conserved.",
            "The emitted −1e particle is accompanied by an antineutrino; the daughter has Z+1 and A unchanged.",
            "Z+1, A conserved, charge −1e emitted, antineutrino released — this is the full description of this decay.",
        ])

    return prompts


def level2_contrastive_pairs() -> list[dict]:
    """14 Level-2 contrastive pairs (6 from Phase 1 + 8 new)."""
    prompts = []

    pairs = [
        # Phase 1 pairs
        ("CP-L2-01", "alpha", "The emitted particle has charge +2e and mass number 4.",
                     "beta",  "The emitted particle has charge −1e and negligible mass.",
                     "charge sign/magnitude and mass magnitude"),
        ("CP-L2-02", "alpha", "The daughter nucleus has atomic number 2 lower and mass number 4 lower than the parent.",
                     "beta",  "The daughter nucleus has atomic number 1 higher and the same mass number as the parent.",
                     "direction and magnitude of Z; whether A changes"),
        ("CP-L2-03", "alpha", "The emitted particle contains 2 protons and 2 neutrons.",
                     "beta",  "The emitted particle contains no protons and belongs to a non-nuclear category.",
                     "nucleon content of emitted particle"),
        ("CP-L2-04", "alpha", "Z decreases by 2 and A decreases by 4.",
                     "beta",  "Z increases by 1 and A is conserved.",
                     "direction of Z change; whether A changes"),
        ("CP-L2-05", "alpha", "The emitted particle has charge +2e and the daughter has Z−2.",
                     "beta",  "The emitted particle has charge −1e and the daughter has Z+1.",
                     "charge and Z-change direction combined"),
        ("CP-L2-06", "alpha", "The emitted particle is hadronic with charge +2e.",
                     "beta",  "The emitted particle is leptonic with charge −1e.",
                     "particle family and charge"),
        # Phase 2 new pairs
        ("CP-L2-07", "alpha", "The emitted particle has mass number 4 and the daughter has A−4.",
                     "beta",  "The emitted particle has negligible mass and the daughter has A unchanged.",
                     "mass of emitted particle and A-change together"),
        ("CP-L2-08", "alpha", "A bound cluster of nucleons is ejected, reducing Z and A.",
                     "beta",  "A neutron within the nucleus converts into a proton, conserving A.",
                     "ejection vs. in-nucleus conversion process type"),
        ("CP-L2-09", "alpha", "The emitted particle has 2 protons and 2 neutrons, leaving Z−2 and A−4.",
                     "beta",  "The emitted particle has no nucleons; Z+1 and A unchanged.",
                     "full nucleon accounting comparison"),
        ("CP-L2-10", "alpha", "The daughter has Z−2 and A−4 relative to the parent.",
                     "beta",  "The daughter has Z+1 and A unchanged relative to the parent.",
                     "full daughter-change comparison"),
        ("CP-L2-11", "alpha", "The emitted particle has charge +2e and contains 2 protons.",
                     "beta",  "The emitted particle has charge −1e and contains no protons.",
                     "charge and proton content combined"),
        ("CP-L2-12", "alpha", "A neutron is NOT converted; a nuclear cluster is instead ejected.",
                     "beta",  "A neutron inside the nucleus converts into a proton.",
                     "process type: ejection vs. conversion"),
        ("CP-L2-13", "alpha", "The emitted particle has charge +2e, mass 4, 2 protons, 2 neutrons; daughter Z−2, A−4.",
                     "beta",  "The emitted particle has charge −1e, negligible mass; daughter Z+1, A unchanged; antineutrino emitted.",
                     "full specification contrast"),
        ("CP-L2-14", "alpha", "The emitted particle has charge +2e and the mass number decreases by 4.",
                     "beta",  "The emitted particle has charge −1e and the mass number is conserved.",
                     "charge and A-change together"),
    ]

    for pair_id, al, ad, bl, bd, discriminant in pairs:
        prompts.append(record(
            group_id=pair_id, variant=1, description=ad,
            label=al, level=2, level_label="relation_process",
            test_type=["CP"], abstraction_level=2, inference_steps=2,
            contrastive_pair_id=pair_id, contrastive_role="alpha_member",
            evidence_completeness="combination", difficulty="easy",
            notes=f"CP discriminant: {discriminant}",
        ))
        prompts.append(record(
            group_id=pair_id, variant=2, description=bd,
            label=bl, level=2, level_label="relation_process",
            test_type=["CP"], abstraction_level=2, inference_steps=2,
            contrastive_pair_id=pair_id, contrastive_role="beta_member",
            evidence_completeness="combination", difficulty="easy",
            notes=f"CP discriminant: {discriminant}",
        ))

    return prompts


def level2_gradient_probes() -> list[dict]:
    """12 abstraction-gradient probes bridging Level 2 and Level 3."""
    descriptions_alpha = [
        "The emitted particle is identical to a helium-4 nucleus, which has charge +2e and mass number 4.",
        "The emitted particle, like helium-4, has 2 protons and 2 neutrons.",
        "The emitted particle is a He-4-like cluster whose emission reduces Z by 2 and A by 4.",
        "A helium-like cluster with +2e charge and 4 nucleons is released.",
        "The emitted particle is compositionally identical to He-4: 2 protons, 2 neutrons, charge +2e.",
        "A nuclear cluster equivalent to a helium-4 nucleus, carrying +2e, is emitted.",
    ]
    descriptions_beta = [
        "The emitted particle, like an electron, has charge −1e and belongs to the lepton family.",
        "An electron-type lepton with charge −1e is emitted as a neutron converts to a proton.",
        "A lepton with charge −1e is released, and the daughter nucleus has Z+1.",
        "The emitted particle is a lepton carrying −1e; the daughter has Z+1 and A unchanged.",
        "The emitted particle belongs to the lepton family and carries charge −1e, produced as a neutron becomes a proton.",
        "The emitted particle is an electron-type particle with −1e charge; an antineutrino is co-emitted.",
    ]

    prompts = []
    for i, desc in enumerate(descriptions_alpha, 1):
        prompts.append(record(
            group_id="L2-GRAD-A", variant=i, description=desc,
            label="alpha", level=2, level_label="relation_process",
            test_type=["IC"], abstraction_level=3, inference_steps=3,
            has_helium_ref=True, prompt_format="equivalence",
            evidence_completeness="combination", difficulty="medium",
            notes="Abstraction gradient probe: bridges L2 and L3 by combining cues with He-4 equivalence.",
        ))
    for i, desc in enumerate(descriptions_beta, 1):
        prompts.append(record(
            group_id="L2-GRAD-B", variant=i, description=desc,
            label="beta", level=2, level_label="relation_process",
            test_type=["IC"], abstraction_level=3, inference_steps=3,
            has_electron_ref=True, prompt_format="equivalence",
            evidence_completeness="combination", difficulty="medium",
            notes="Abstraction gradient probe: bridges L2 and L3 by combining cues with lepton family concept.",
        ))

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
#  LEVEL 3 — Concept / latent-state probes
# ─────────────────────────────────────────────────────────────────────────────

def level3_alpha() -> list[dict]:
    prompts = []

    # L3-A1 — He-4 equivalence (8 variants)
    prompts += group("L3-A1", "alpha", 3, "latent_concept",
        concept_route="helium4_equivalence", abstraction_level=4, inference_steps=3,
        prompt_format="equivalence", difficulty="medium", has_helium_ref=True,
        descriptions=[
            "The emitted particle is identical in composition to the nucleus of a helium atom.",
            "The emitted particle is the same type as a helium-4 nucleus.",
            "The emitted particle has the same nuclear structure as He-4.",
            "The emitted particle is a tightly bound nuclear cluster with the same composition as the helium nucleus.",
            "The particle released is the same type of cluster found at the core of a helium-4 atom.",
            "The emitted particle is equivalent to the helium-4 nucleus in every nuclear respect.",
            "The emitted particle is of the same type as the nucleus of the lightest element that is an inert gas.",
            "The emitted particle has the same nuclear composition as helium-4.",
        ])

    # L3-A2 — composite nuclear object (8 variants)
    prompts += group("L3-A2", "alpha", 3, "latent_concept",
        concept_route="composite_nuclear_object", abstraction_level=4, inference_steps=3,
        prompt_format="property", difficulty="medium",
        descriptions=[
            "The emitted particle is a composite nuclear object, not an elementary particle.",
            "The emitted particle is a bound nucleus rather than a fundamental particle.",
            "The emitted particle is itself a small nucleus rather than an isolated nucleon or lepton.",
            "The decay ejects a small, tightly bound nuclear fragment from the nucleus.",
            "The emitted particle is a multi-nucleon nuclear cluster.",
            "The emitted object is a composite of nucleons, not an elementary constituent.",
            "The emitted particle is a mini-nucleus — a bound system of multiple nucleons.",
            "The decay produces a small nuclear cluster as the emitted particle, not a fundamental particle.",
        ])

    # L3-A3 — heavy nuclear fragment [NEW] (6 variants)
    prompts += group("L3-A3", "alpha", 3, "latent_concept",
        concept_route="heavy_nuclear_fragment", abstraction_level=4, inference_steps=3,
        prompt_format="property", difficulty="medium",
        descriptions=[
            "The emitted particle is a relatively heavy nuclear fragment compared to a lepton.",
            "The emitted particle is substantially heavier than any lepton.",
            "The emitted particle is heavy in the nuclear scale, unlike a lepton.",
            "The emitted particle has hadronic mass, not leptonic mass.",
            "The emitted particle is much heavier than a beta particle.",
            "The emitted particle is a heavy, strongly interacting nuclear fragment.",
        ])

    # L3-A4 — baryon number 4 [NEW] (6 variants)
    prompts += group("L3-A4", "alpha", 3, "latent_concept",
        concept_route="baryon_number_4", abstraction_level=4, inference_steps=4,
        prompt_format="property", difficulty="hard",
        notes="Requires knowing baryon number counting; abstraction_level=4.",
        descriptions=[
            "The emitted particle has baryon number 4.",
            "The emitted particle carries a baryon number of 4.",
            "The total baryon number of the emitted particle is 4.",
            "The emitted particle has four units of baryon number.",
            "The baryon number carried away by the emitted particle is 4.",
            "The emitted particle has baryon number equal to 4.",
        ])

    return prompts


def level3_beta() -> list[dict]:
    prompts = []

    # L3-B1 — lepton family (8 variants)
    prompts += group("L3-B1", "beta", 3, "latent_concept",
        concept_route="lepton_family", abstraction_level=4, inference_steps=3,
        prompt_format="property", difficulty="medium",
        descriptions=[
            "The emitted particle belongs to the lepton family.",
            "The particle released is a lepton.",
            "The emitted particle is classified as a lepton.",
            "The emitted particle is a member of the lepton family of fundamental particles.",
            "The emitted particle is a lepton, not a hadron.",
            "The emitted particle falls into the lepton category of particles.",
            "The emitted particle is leptonic in nature.",
            "The emitted particle is a lepton — it does not participate in the strong interaction.",
        ])

    # L3-B2 — same type as electron (8 variants)
    prompts += group("L3-B2", "beta", 3, "latent_concept",
        concept_route="electron_equivalence", abstraction_level=4, inference_steps=3,
        prompt_format="property", difficulty="medium",
        has_electron_ref=True,
        descriptions=[
            "The emitted particle is the same type of particle as an electron.",
            "The emitted particle is identical in type to an electron.",
            "The emitted particle belongs to the same particle family as the electron.",
            "The emitted particle is an electron or electron-equivalent in its fundamental nature.",
            "The particle emitted is of the same fundamental kind as the electron.",
            "The emitted particle is in the same class of particles as an electron.",
            "The emitted particle and the electron share the same fundamental particle classification.",
            "The emitted particle is the same species of particle as an electron.",
        ])

    # L3-B3 — not a nuclear fragment [NEW] (6 variants)
    prompts += group("L3-B3", "beta", 3, "latent_concept",
        concept_route="not_nuclear_fragment", abstraction_level=4, inference_steps=3,
        prompt_format="property", difficulty="medium",
        descriptions=[
            "The emitted particle is not a nuclear fragment; it contains no nucleons.",
            "The emitted particle contains no protons or neutrons.",
            "The emitted particle is not composed of quarks or nucleons.",
            "The emitted particle is not a hadron; it carries no hadronic matter.",
            "Unlike alpha emission, the emitted particle here contains no nucleons at all.",
            "The emitted particle is not a nuclear cluster — it has no hadronic content.",
        ])

    # L3-B4 — same family as muon [NEW] (6 variants)
    prompts += group("L3-B4", "beta", 3, "latent_concept",
        concept_route="muon_family", abstraction_level=4, inference_steps=4,
        prompt_format="property", difficulty="hard",
        notes="2-step: muon -> lepton family -> beta. Tests indirect concept activation.",
        descriptions=[
            "The emitted particle belongs to the same family of particles as the muon.",
            "The emitted particle is in the same fundamental class as the muon.",
            "The emitted particle and the muon belong to the same particle category.",
            "The emitted particle is of the same type as the muon, differing only in mass.",
            "The emitted particle shares its particle family classification with the muon.",
            "The emitted particle is the same class of particle as the muon.",
        ])

    return prompts


def level3_anchors() -> list[dict]:
    """Full-specification anchors: 8 alpha + 8 beta."""
    prompts = []

    alpha_descs = [
        "The emitted particle has charge +2e, mass number 4, and contains 2 protons and 2 neutrons. The daughter nucleus has Z−2 and A−4.",
        "A particle with +2e charge and 4 nucleons (2 protons, 2 neutrons) is emitted. The product nucleus has 2 fewer protons and 4 fewer nucleons.",
        "The emitted cluster carries +2 units of electric charge, consists of 2 protons and 2 neutrons, and the daughter nucleus is 2 atomic numbers and 4 mass units lighter.",
        "The decay emits a doubly positively charged particle of mass number 4 composed of 2 protons and 2 neutrons, leaving a daughter with Z−2 and A−4.",
        "The emitted particle: charge +2e, mass 4, composition 2p+2n. Daughter nucleus: Z−2, A−4.",
        "A +2e particle with 4 nucleons (proton count 2, neutron count 2) is released; the daughter has 2 fewer protons and 4 fewer nucleons.",
        "The emitted particle has +2e charge and consists of 2 protons and 2 neutrons; the daughter loses 2 protons and 4 nucleons.",
        "Charge +2e, 4 nucleons, 2 protons, 2 neutrons: that is the emitted particle. The daughter has Z−2 and A−4.",
    ]
    for i, desc in enumerate(alpha_descs, 1):
        prompts.append(record(
            group_id="L3-FA", variant=i, description=desc,
            label="alpha", level=3, level_label="latent_concept",
            test_type=["ANC", "IC"],
            cue_set=["emitted_charge_plus2", "emitted_mass4", "emitted_2protons", "emitted_2neutrons",
                     "daughter_z_minus2", "daughter_a_minus4"],
            abstraction_level=1, inference_steps=1,
            is_anchor=True, evidence_completeness="full",
            prompt_format="emission", difficulty="easy",
        ))

    beta_descs = [
        "The emitted particle has charge −1e. The daughter nucleus has Z+1 and unchanged mass number. A neutron converts to a proton, and an antineutrino is also emitted.",
        "A particle with charge −1e is emitted. A neutron transforms into a proton, the atomic number increases by 1, the mass number is conserved, and an antineutrino is released.",
        "The decay emits a particle of charge −1e accompanied by an antineutrino, a neutron converts to a proton, Z increases by 1, and A remains the same.",
        "The emitted particle carries −1e, the daughter has Z+1 and A unchanged, a neutron has become a proton within the nucleus, and an antineutrino exits the system.",
        "Charge −1e, Z+1, A unchanged, n→p, antineutrino: the complete description of this decay.",
        "A −1e particle and antineutrino are emitted as a neutron converts to a proton; the daughter has Z+1 and A unchanged.",
        "The emitted particle has charge −1e; the daughter has one more proton and the same mass number; a neutron becomes a proton; an antineutrino is released.",
        "Emitted particle: charge −1e. Daughter: Z+1, A unchanged. Process: neutron→proton. Also emitted: antineutrino.",
    ]
    for i, desc in enumerate(beta_descs, 1):
        prompts.append(record(
            group_id="L3-FB", variant=i, description=desc,
            label="beta", level=3, level_label="latent_concept",
            test_type=["ANC", "IC"],
            cue_set=["emitted_charge_minus1", "daughter_z_plus1", "daughter_a_unchanged",
                     "antineutrino_emitted"],
            abstraction_level=1, inference_steps=1,
            is_anchor=True, evidence_completeness="full",
            prompt_format="process", difficulty="easy",
        ))

    return prompts


def level3_contrastive_pairs() -> list[dict]:
    """4 Level-3 contrastive pairs."""
    prompts = []

    pairs = [
        ("CP-L3-01",
            "alpha", "The emitted particle is identical in composition to the nucleus of a helium atom.",
            "beta",  "The emitted particle belongs to the lepton family.",
            "He-4 equivalence vs lepton family"),
        ("CP-L3-02",
            "alpha", "The emitted particle is a composite nuclear object, not an elementary particle.",
            "beta",  "The emitted particle is the same type of particle as an electron.",
            "composite nuclear vs elementary lepton"),
        ("CP-L3-03",
            "alpha", "The emitted particle is a relatively heavy nuclear fragment.",
            "beta",  "The emitted particle is not a nuclear fragment; it contains no nucleons.",
            "heavy nuclear vs non-nuclear"),
        ("CP-L3-04",
            "alpha", "The emitted particle has baryon number 4.",
            "beta",  "The emitted particle belongs to the same family of particles as the muon.",
            "baryon number 4 vs muon family"),
    ]

    for pair_id, al, ad, bl, bd, discriminant in pairs:
        prompts.append(record(
            group_id=pair_id, variant=1, description=ad,
            label=al, level=3, level_label="latent_concept",
            test_type=["CP"], abstraction_level=4, inference_steps=3,
            contrastive_pair_id=pair_id, contrastive_role="alpha_member",
            prompt_format="property", difficulty="medium",
            notes=f"L3 CP discriminant: {discriminant}",
        ))
        prompts.append(record(
            group_id=pair_id, variant=2, description=bd,
            label=bl, level=3, level_label="latent_concept",
            test_type=["CP"], abstraction_level=4, inference_steps=3,
            contrastive_pair_id=pair_id, contrastive_role="beta_member",
            prompt_format="property", difficulty="medium",
            notes=f"L3 CP discriminant: {discriminant}",
        ))

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
#  AUXILIARY — world-knowledge, conflict, isotope, partial
# ─────────────────────────────────────────────────────────────────────────────

def auxiliary() -> list[dict]:
    """56 auxiliary prompts — excluded from primary SE/IC analysis."""
    prompts = []

    # Hard world-knowledge probes (20)
    wk = [
        ("AUX-A1","alpha","The emitted radiation can be stopped by a sheet of paper.",5,"functional"),
        ("AUX-A2","alpha","The emitted particle has the same composition as the nucleus of the lightest noble gas.",5,"equivalence"),
        ("AUX-A3","alpha","The emitted particle is the nucleus of a doubly ionised helium atom.",4,"equivalence"),
        ("AUX-A4","alpha","The emitted particle has integer spin and is a boson.",5,"property"),
        ("AUX-A5","alpha","The emitted particle is the same type as the particles that constitute cosmic ray primaries.",5,"property"),
        ("AUX-A6","alpha","The emitted particle has zero electric dipole moment and integer spin.",5,"property"),
        ("AUX-A7","alpha","The emitted particle carries the largest charge of any commonly emitted nuclear decay product.",4,"property"),
        ("AUX-A8","alpha","The emitted particle is the nucleus stripped of its electron cloud.",4,"equivalence"),
        ("AUX-A9","alpha","The emitted particle is more ionising than any other nuclear decay emission.",5,"functional"),
        ("AUX-A10","alpha","The emitted particle leaves a short, thick track in a cloud chamber.",5,"functional"),
        ("AUX-B1","beta","The decay is mediated by the weak nuclear force.",5,"process"),
        ("AUX-B2","beta","The decay is driven by the same fundamental force responsible for free neutron decay outside the nucleus.",5,"process"),
        ("AUX-B3","beta","The decay involves a change in quark flavour within a nucleon.",5,"process"),
        ("AUX-B4","beta","The emitted radiation is more penetrating than alpha radiation but less than gamma radiation.",5,"functional"),
        ("AUX-B5","beta","The decay involves a W boson as an intermediate particle.",5,"process"),
        ("AUX-B6","beta","The emitted particle has the same charge as the down quark's contribution to beta emission.",5,"property"),
        ("AUX-B7","beta","The emitted particle is the same class as the particle involved in muon decay.",5,"equivalence"),
        ("AUX-B8","beta","The decay occurs via the electroweak interaction, changing quark flavour inside a nucleon.",5,"process"),
        ("AUX-B9","beta","The emitted particle leaves a longer, thinner track in a cloud chamber than alpha radiation.",5,"functional"),
        ("AUX-B10","beta","The emitted particle participates in electromagnetic and weak interactions but not the strong interaction.",5,"property"),
    ]
    for pid, label, desc, abst, fmt in wk:
        prompts.append(record(
            group_id=pid, variant=1, description=desc,
            label=label, level=None, level_label="auxiliary",
            test_type=["AUX"], abstraction_level=abst, inference_steps=abst,
            is_auxiliary=True, prompt_format=fmt,
            difficulty="hard", is_uniquely_determining=True,
        ))

    # Conflict / contradiction prompts (12)
    conflicts = [
        ("AUX-CF1","alpha","The emitted particle has charge +2e but the mass number is unchanged.",
         "Conflicting: charge implies alpha; A-unchanged implies beta."),
        ("AUX-CF2","beta","The emitted particle has charge −1e but the mass number decreases by 4.",
         "Conflicting: charge implies beta; A−4 implies alpha."),
        ("AUX-CF3","alpha","The daughter nucleus has Z−2 but A is conserved.",
         "Conflicting: Z−2 implies alpha; A-unchanged implies beta."),
        ("AUX-CF4","beta","The daughter nucleus has Z+1 but the mass number decreases by 4.",
         "Conflicting: Z+1 implies beta; A−4 implies alpha."),
        ("AUX-CF5","alpha","The emitted particle has charge +2e but the daughter has Z+1.",
         "Conflicting Z direction with emitted charge."),
        ("AUX-CF6","beta","The emitted particle has charge −1e but the daughter has Z−2.",
         "Conflicting Z direction with emitted charge."),
        ("AUX-CF7","alpha","The emitted particle contains 2 protons but the mass number is unchanged.",
         "Conflicting: 2 protons implies alpha; A unchanged implies beta."),
        ("AUX-CF8","beta","A neutron converts to a proton but the atomic number decreases by 2.",
         "Conflicting: n→p implies beta; Z−2 implies alpha."),
        ("AUX-CF9","alpha","The emitted particle is identical to a helium-4 nucleus but the mass number is unchanged.",
         "He-4 concept conflicts with A-conservation."),
        ("AUX-CF10","beta","The emitted particle belongs to the lepton family but the daughter has Z−2.",
         "Lepton concept conflicts with Z-direction."),
        ("AUX-CF11","alpha","The emitted particle has charge +2e but the daughter has one more proton than the parent.",
         "Charge +2e conflicts with Z+1 in daughter."),
        ("AUX-CF12","beta","The emitted particle has charge −1e but the daughter has the same atomic number as the parent.",
         "Charge −1e conflicts with Z unchanged."),
    ]
    for pid, label, desc, note in conflicts:
        prompts.append(record(
            group_id=pid, variant=1, description=desc,
            label=label, level=None, level_label="auxiliary",
            test_type=["AUX"], abstraction_level=2, inference_steps=2,
            is_auxiliary=True, prompt_format="property",
            is_uniquely_determining=False, difficulty="hard",
            notes=note,
        ))

    # Isotope-specific prompts (12)
    isotopes = [
        ("AUX-ISO1","alpha","Uranium-238 undergoes this decay, producing thorium-234.","Ra→Rn-like alpha",True),
        ("AUX-ISO2","alpha","Radium-226 decays to radon-222 in this process.","classic alpha",True),
        ("AUX-ISO3","alpha","Polonium-210 decays to lead-206 in this decay.","classic alpha",True),
        ("AUX-ISO4","alpha","Thorium-232 decays to radium-228 via this process.","alpha",True),
        ("AUX-ISO5","alpha","Americium-241 decays to neptunium-237 in this manner.","used in smoke detectors",True),
        ("AUX-ISO6","alpha","Bismuth-212 undergoes this decay to thallium-208.","alpha",True),
        ("AUX-ISO7","beta","Carbon-14 undergoes this decay to nitrogen-14.","classic beta",True),
        ("AUX-ISO8","beta","Tritium (hydrogen-3) decays to helium-3 via this process.","classic beta",True),
        ("AUX-ISO9","beta","Strontium-90 decays to yttrium-90 in this decay.","classic beta",True),
        ("AUX-ISO10","beta","Cobalt-60 undergoes this decay to nickel-60.","classic beta",True),
        ("AUX-ISO11","beta","Iodine-131 decays to xenon-131 via this process.","medical isotope",True),
        ("AUX-ISO12","beta","Cesium-137 decays to barium-137 in this manner.","classic beta",True),
    ]
    for pid, label, desc, note, has_iso in isotopes:
        prompts.append(record(
            group_id=pid, variant=1, description=desc,
            label=label, level=None, level_label="auxiliary",
            test_type=["AUX"], abstraction_level=4, inference_steps=3,
            has_isotope=has_iso, is_auxiliary=True,
            prompt_format="property", difficulty="medium",
            notes=f"Isotope-specific: {note}",
        ))

    # Partial / underspecified cues (12)
    partial = [
        ("AUX-P1","alpha","The emitted particle carries a positive electric charge.",
         "Sign only, not magnitude — still implies alpha in binary."),
        ("AUX-P2","alpha","The emitted particle is heavier than a single nucleon.",
         "Partial mass cue — rules out leptons, consistent with alpha."),
        ("AUX-P3","alpha","The atomic number of the daughter nucleus is lower than that of the parent.",
         "Direction only (not magnitude) — Z decreases."),
        ("AUX-P4","alpha","The mass number of the daughter nucleus decreases.",
         "Direction only — A decreases."),
        ("AUX-P5","alpha","The emitted particle contains multiple nucleons.",
         "Partial composition — consistent with alpha, excludes beta."),
        ("AUX-P6","alpha","The emitted particle has more than 1 unit of charge.",
         "Partial charge magnitude — implies alpha (only +2e common)."),
        ("AUX-P7","beta","The emitted particle carries a negative electric charge.",
         "Sign only, not magnitude — implies beta."),
        ("AUX-P8","beta","The emitted particle has less mass than a nucleon.",
         "Partial mass — rules out alpha; consistent with beta lepton."),
        ("AUX-P9","beta","The atomic number of the daughter nucleus is higher than that of the parent.",
         "Direction only — Z increases, consistent with beta."),
        ("AUX-P10","beta","The mass number of the daughter nucleus does not change.",
         "A unchanged — consistent with beta."),
        ("AUX-P11","beta","The emitted particle contains no nucleons.",
         "Nucleon-free — rules out alpha, consistent with beta lepton."),
        ("AUX-P12","beta","The emitted particle has a single unit of electric charge.",
         "Magnitude 1 charge, sign unspecified — partial."),
    ]
    for pid, label, desc, note in partial:
        prompts.append(record(
            group_id=pid, variant=1, description=desc,
            label=label, level=None, level_label="auxiliary",
            test_type=["AUX"], abstraction_level=2, inference_steps=2,
            is_auxiliary=True, prompt_format="property",
            is_uniquely_determining=False, difficulty="medium",
            notes=note,
        ))

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(include_auxiliary: bool = True, level: int | None = None) -> list[dict]:
    all_prompts = []

    if level is None or level == 1:
        all_prompts += level1_alpha()
        all_prompts += level1_beta()
        all_prompts += level1_keyword_pairs()
        all_prompts += level1_contrastive_pairs()

    if level is None or level == 2:
        all_prompts += level2_alpha()
        all_prompts += level2_beta()
        all_prompts += level2_contrastive_pairs()
        all_prompts += level2_gradient_probes()

    if level is None or level == 3:
        all_prompts += level3_alpha()
        all_prompts += level3_beta()
        all_prompts += level3_anchors()
        all_prompts += level3_contrastive_pairs()

    if include_auxiliary and level is None:
        all_prompts += auxiliary()

    return all_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate physics_decay_type_probe corpus")
    parser.add_argument("--include_auxiliary", action="store_true", default=True,
                        help="Include auxiliary prompts (default: True)")
    parser.add_argument("--no_auxiliary", dest="include_auxiliary", action="store_false")
    parser.add_argument("--level", type=int, default=None,
                        choices=[1, 2, 3], help="Generate only one level (default: all)")
    args = parser.parse_args()

    prompts = build_corpus(include_auxiliary=args.include_auxiliary, level=args.level)

    suffix = f"_L{args.level}" if args.level else ""
    out = OUT_DIR / f"{BEHAVIOUR}_train{suffix}.jsonl"
    with open(out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    n_alpha = sum(1 for p in prompts if p["physics_concept"] == "alpha_decay")
    n_beta  = sum(1 for p in prompts if p["physics_concept"] == "beta_decay")
    n_core  = sum(1 for p in prompts if not p.get("is_auxiliary"))
    n_aux   = sum(1 for p in prompts if p.get("is_auxiliary"))
    levels  = {1: 0, 2: 0, 3: 0}
    for p in prompts:
        if p.get("level") in levels:
            levels[p["level"]] += 1

    print(f"Written {len(prompts)} prompts to {out}")
    print(f"  Alpha: {n_alpha}  Beta: {n_beta}")
    print(f"  Core: {n_core}  Auxiliary: {n_aux}")
    print(f"  Level 1: {levels[1]}  Level 2: {levels[2]}  Level 3: {levels[3]}")

    # per-group count
    from collections import Counter
    gc = Counter(p["group_id"] for p in prompts)
    print("\nGroup breakdown:")
    for gid in sorted(gc):
        print(f"  {gid}: {gc[gid]}")


if __name__ == "__main__":
    main()
