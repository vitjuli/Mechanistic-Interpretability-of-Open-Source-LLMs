"""
Generate round-2 candidate-set screening prompts.

Three behaviours designed from the round-1 failure analysis:
  physics_spin_statistics_v2_mini  — 32 prompts; ' boson'/' fermion' (2-token each)
  physics_relativistic_regime_v2_mini — 24 prompts; ' Yes'/' No' (1-token each)
  physics_conductor_insulator_mini — 24 prompts; ' Yes'/' No' (1-token each)

Key design improvements over round-1:
  - Regime: Yes/No framing removes systematic ' classical' token bias
  - Spin: F2_novel_spin uses hypothetical particles (s=3/2, s=2 etc.) — cannot be
    bypassed by particle-name memorisation, tests pure rule application
  - Spin: F4_hybrid gives explicit "integer"/"half-integer" label alongside the value,
    tests whether the bottleneck is numeral→classification or classification→class-label
  - Conductor: new behaviour with strong expected accuracy via named-material recall
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
    neg_ans = records[0]["incorrect_answer"].strip()
    n_pos = sum(1 for r in records if r["correct_answer"].strip() == pos_ans)
    n_neg = len(records) - n_pos
    families = sorted(set(r.get("wording_family", "?") for r in records))
    print(f"  Wrote {len(records)} prompts → {path.name}")
    print(f"    Balance: {n_pos} '{pos_ans}' / {n_neg} '{neg_ans}'")
    print(f"    Families: {families}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  physics_spin_statistics_v2_mini
#
# Four families:
#   F1_particle_name — recall (named particles → class)
#   F2_novel_spin    — symbolic rule (s=3/2, s=2 etc.; not common particle names)
#   F3_spin_descriptor — verbal (textbook phrase "integer spin → boson")
#   F4_hybrid        — hybrid (spin value + explicit integer/half-integer label)
#
# Scientific value: F2 vs F4 comparison directly localises the mechanism failure.
#   If F4 >> F2: model knows "integer → boson" but cannot compute "2 ∈ ℤ"
#   If F4 ≈ F2: even the verbal rule is context-dependent
# ─────────────────────────────────────────────────────────────────────────────

# F1 particles (familiar enough for recall)
F1_PARTICLES = [
    ("photon",      1.0,  "1",   True,  "boson"),
    ("Z boson",     1.0,  "1",   True,  "boson"),
    ("Higgs boson", 0.0,  "0",   True,  "boson"),
    ("pion",        0.0,  "0",   True,  "boson"),
    ("electron",    0.5,  "1/2", False, "fermion"),
    ("proton",      0.5,  "1/2", False, "fermion"),
    ("muon",        0.5,  "1/2", False, "fermion"),
    ("neutron",     0.5,  "1/2", False, "fermion"),
]

# F2 novel spin values — hypothetical/unfamiliar so model must apply the rule
F2_NOVEL = [
    # (spin_value, spin_str, is_integer, stat_class, context_note)
    (0.0, "0",   True,  "boson",   "a hypothetical scalar particle"),
    (1.0, "1",   True,  "boson",   "a hypothetical vector particle"),
    (2.0, "2",   True,  "boson",   "a hypothetical tensor particle"),
    (3.0, "3",   True,  "boson",   "a hypothetical higher-spin particle"),
    (0.5, "1/2", False, "fermion", "a hypothetical spin-1/2 particle"),
    (1.5, "3/2", False, "fermion", "a hypothetical spin-3/2 particle"),
    (2.5, "5/2", False, "fermion", "a hypothetical spin-5/2 particle"),
    (3.5, "7/2", False, "fermion", "a hypothetical spin-7/2 particle"),
]


def spin_statistics_v2_prompts() -> list[dict]:
    records = []

    # F1: particle name recall
    for name, spin_val, spin_str, is_int, stat_class in F1_PARTICLES:
        wrong = "fermion" if stat_class == "boson" else "boson"
        records.append({
            "prompt": f"Is a {name} a boson or a fermion? Answer:",
            "wording_family": "F1_particle_name",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": name,
            "spin_value": spin_str,
            "is_integer_spin": is_int,
            "stat_class": stat_class,
        })

    # F2: novel spin value — hypothetical particle, pure rule application
    for spin_val, spin_str, is_int, stat_class, context in F2_NOVEL:
        wrong = "fermion" if stat_class == "boson" else "boson"
        int_label = "integer" if is_int else "half-integer"
        records.append({
            "prompt": (
                f"Consider {context} with spin quantum number s = {spin_str}. "
                f"Based on the spin-statistics theorem, is it a boson or a fermion? Answer:"
            ),
            "wording_family": "F2_novel_spin",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": f"hypothetical s={spin_str}",
            "spin_value": spin_str,
            "is_integer_spin": is_int,
            "stat_class": stat_class,
        })

    # F3: spin descriptor — textbook verbal phrases (round-1 F3 got 100%)
    # Four distinct phrasings per spin type to avoid identical prompts
    F3_TEMPLATES_INT = [
        "A particle carries integer spin. According to the spin-statistics theorem, is it a boson or a fermion?",
        "A quantum particle possesses integer angular momentum (integer spin). Is it a boson or a fermion?",
        "A particle whose spin is an integer. By the spin-statistics theorem, is it a boson or a fermion?",
        "Consider a particle with an integer-valued spin quantum number. Is it a boson or a fermion?",
    ]
    F3_TEMPLATES_HALF = [
        "A particle carries half-integer spin. According to the spin-statistics theorem, is it a boson or a fermion?",
        "A quantum particle possesses half-integer angular momentum. Is it a boson or a fermion?",
        "A particle whose spin is a half-integer value. By the spin-statistics theorem, is it a boson or a fermion?",
        "Consider a particle with a half-integer-valued spin quantum number. Is it a boson or a fermion?",
    ]
    boson_idx, fermion_idx = 0, 0
    for name, spin_val, spin_str, is_int, stat_class in F1_PARTICLES:
        wrong = "fermion" if stat_class == "boson" else "boson"
        if is_int:
            template = F3_TEMPLATES_INT[boson_idx % len(F3_TEMPLATES_INT)]
            boson_idx += 1
        else:
            template = F3_TEMPLATES_HALF[fermion_idx % len(F3_TEMPLATES_HALF)]
            fermion_idx += 1
        records.append({
            "prompt": f"{template} Answer:",
            "wording_family": "F3_spin_descriptor",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": name,
            "spin_value": spin_str,
            "is_integer_spin": is_int,
            "stat_class": stat_class,
        })

    # F4: hybrid — same novel spin values as F2 + explicit integer/half-integer label
    # Direct comparison with F2: if F4 >> F2, the bottleneck is numeral→classification,
    # not classification→class-label.
    for spin_val, spin_str, is_int, stat_class, context in F2_NOVEL:
        wrong = "fermion" if stat_class == "boson" else "boson"
        int_label = "an integer" if is_int else "a half-integer"
        records.append({
            "prompt": (
                f"Consider {context} with spin quantum number s = {spin_str} ({int_label} value). "
                f"Based on the spin-statistics theorem, is it a boson or a fermion? Answer:"
            ),
            "wording_family": "F4_hybrid",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": f"hypothetical s={spin_str} (labelled)",
            "spin_value": spin_str,
            "is_integer_spin": is_int,
            "stat_class": stat_class,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  physics_relativistic_regime_v2_mini
#
# Reformulation of round-1 failure.
# Round-1 problem: model had systematic ' classical' prior → 50% across all families.
# Fix: Yes/No framing ("Does X require relativistic treatment?")
#   Both tokens single-token and symmetric in prior.
#
# Velocity extremes chosen to avoid ambiguous middle ground:
#   Classical (No):     v/c ∈ {0.001, 0.005, 0.01, 0.03}  (γ < 1.001)
#   Relativistic (Yes): v/c ∈ {0.70,  0.85,  0.95, 0.99}  (γ > 1.40)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_DATA = [
    # (v_over_c, requires_rel, gamma_str, verbal_desc)
    # Classical: γ values use enough precision to be distinct from each other
    (0.001, False, "≈1.0000005", "one tenth of a percent of the speed of light"),
    (0.005, False, "≈1.0000125", "half a percent of the speed of light"),
    (0.010, False, "≈1.0000500", "one percent of the speed of light"),
    (0.030, False, "≈1.000450",  "three percent of the speed of light"),
    # Relativistic: γ values are already distinct
    (0.700, True,  "≈1.400", "seventy percent of the speed of light"),
    (0.850, True,  "≈1.898", "eighty-five percent of the speed of light"),
    (0.950, True,  "≈3.203", "ninety-five percent of the speed of light"),
    (0.990, True,  "≈7.089", "ninety-nine percent of the speed of light"),
]


def relativistic_regime_v2_prompts() -> list[dict]:
    records = []
    for v_c, req_rel, gamma, verbal in REGIME_DATA:
        ans   = "Yes" if req_rel else "No"
        wrong = "No"  if req_rel else "Yes"
        meta = {
            "correct_answer":  f" {ans}",
            "incorrect_answer": f" {wrong}",
            "v_over_c": v_c,
            "requires_relativistic": req_rel,
        }

        # F1: decimal fraction of c
        records.append({
            "prompt": (
                f"A particle moves at v = {v_c}c. "
                f"Does its motion require relativistic treatment for an accurate mechanical description? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F1_decimal_fraction",
            **meta,
        })

        # F2: verbal description of speed
        records.append({
            "prompt": (
                f"A particle is moving at {verbal}. "
                f"Does its motion require relativistic treatment? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F2_verbal_speed",
            **meta,
        })

        # F3: Lorentz factor
        records.append({
            "prompt": (
                f"A particle has Lorentz factor γ {gamma}. "
                f"Does this particle's dynamics require relativistic treatment? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F3_lorentz_factor",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3.  physics_conductor_insulator_mini  (NEW)
#
# Rule: high conductivity / no band gap → conductor (Yes); otherwise insulator (No).
# Yes/No single-token answers.
#
# Three families:
#   F1_named_material   — retrieval (copper → Yes, rubber → No)
#   F2_conductivity     — numerical threshold (σ > ~10^6 S/m → Yes)
#   F3_band_structure   — verbal rule (no band gap → Yes; large band gap → No)
#
# Mechanistic hypothesis: F1 and F2 should activate same intermediate "conductivity"
# features if the model has a unified representation. Cross-family convergence test.
# ─────────────────────────────────────────────────────────────────────────────

CONDUCTOR_MATERIALS = [
    # (name, sigma_str, band_desc, is_conductor)
    ("copper",   "5.96×10^7 S/m", "overlapping valence and conduction bands (metallic)",  True),
    ("aluminum", "3.77×10^7 S/m", "partially filled conduction band at room temperature", True),
    ("silver",   "6.30×10^7 S/m", "no band gap between valence and conduction bands",     True),
    ("gold",     "4.10×10^7 S/m", "free electrons in a partially filled d-band",          True),
    ("rubber",   "10^{-15} S/m",  "a band gap of approximately 9 eV",  False),
    ("glass",    "10^{-12} S/m",  "a band gap of approximately 8 eV",  False),
    ("wood",     "10^{-16} S/m",  "a band gap of approximately 7 eV",  False),
    ("ceramic",  "10^{-13} S/m",  "a band gap of approximately 5 eV",  False),
]


def conductor_insulator_prompts() -> list[dict]:
    records = []
    for name, sigma, band_desc, is_cond in CONDUCTOR_MATERIALS:
        ans   = "Yes" if is_cond else "No"
        wrong = "No"  if is_cond else "Yes"
        meta = {
            "correct_answer":  f" {ans}",
            "incorrect_answer": f" {wrong}",
            "material": name,
            "is_conductor": is_cond,
        }

        # F1: named material
        records.append({
            "prompt": (
                f"Is {name} a good electrical conductor? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F1_named_material",
            **meta,
        })

        # F2: conductivity value
        records.append({
            "prompt": (
                f"A material has electrical conductivity σ = {sigma}. "
                f"Is it a good electrical conductor? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F2_conductivity_value",
            **meta,
        })

        # F3: band structure description
        records.append({
            "prompt": (
                f"A material has {band_desc}. "
                f"Is it a good electrical conductor? "
                f"Yes or No? Answer:"
            ),
            "wording_family": "F3_band_structure",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating round-2 candidate-set screening prompts...")
    print()

    behaviours = {
        "physics_spin_statistics_v2_mini":      spin_statistics_v2_prompts(),
        "physics_relativistic_regime_v2_mini":  relativistic_regime_v2_prompts(),
        "physics_conductor_insulator_mini":      conductor_insulator_prompts(),
    }

    for beh, records in behaviours.items():
        path = OUT_DIR / f"{beh}_train.jsonl"
        write_jsonl(path, records)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
