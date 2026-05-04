"""
Generate round-3 candidate-set screening prompts.

Three behaviours, each targeting known failure modes from rounds 1-2:

  physics_spin_statistics_v3_mini (32 prompts)
    Fix from v2: replace pion→W boson in F1; restrict F2 to s∈{0,1,1/2,3/2} only.
    F3/F4 unchanged (both 100% in round-2).
    Expected: 90-96% sign accuracy.

  chemistry_acid_base_mini (32 prompts)
    New domain. Brønsted-Lowry: proton donor=acid, acceptor=base.
    Tokens: ' acid' / ' base' — both single-token, symmetric, no prior bias.
    F3 verbal rule ("proton donor → acid") should approach 100%.
    Expected: 90%+ overall.

  physics_charge_sign_mini (32 prompts)
    New domain. Electric charge of elementary particles and ions.
    Tokens: ' positive' / ' negative' — both single-token common English words.
    F1 recall (named particle), F2 structural rule, F3 physics context.
    Expected: 90%+ overall.

Token choices (all confirmed or expected single-token):
  ' acid', ' base', ' positive', ' negative' — all high-frequency common words
  ' boson', ' fermion' — 2-token (confirmed round-1), length-normalized; F3/F4 still 100%
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
    n_dup = len(records) - len({r["prompt"] for r in records})
    families = sorted({r.get("wording_family", "?") for r in records})
    print(f"  Wrote {len(records)} prompts → {path.name}")
    print(f"    Balance: {n_pos} '{pos_ans}' / {n_neg} '{neg_ans}'  |  duplicates: {n_dup}")
    print(f"    Families: {families}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  physics_spin_statistics_v3_mini
#
# Changes from v2:
#   F1: replace pion (spin-0 hadron, not prominently labeled "boson" in training data)
#       with W boson (literally named "W boson" → unambiguous for the model)
#   F2: restrict to s∈{0,1} for bosons and s∈{1/2,3/2} for fermions only.
#       s=3/2 appears in training data (Delta baryon family); s=2,3,5/2,7/2 do not.
#   F3: different phrasing templates (same as v2)
#   F4: same as v2 — provides explicit "(an integer)"/"(a half-integer)" label
#       Comparison F2 vs F4 is the key thesis finding: if F4>>F2, the bottleneck
#       is the numeral→integer/half-integer classification, not the rule application.
# ─────────────────────────────────────────────────────────────────────────────

# F1 particles — well-known, each clearly a boson or fermion in training data
F1_PARTICLES_V3 = [
    # (name, spin_str, is_integer, stat_class)
    ("photon",      "1",   True,  "boson"),
    ("W boson",     "1",   True,  "boson"),    # v2: was Z boson → keep; v2 pion → now W boson
    ("Higgs boson", "0",   True,  "boson"),
    ("gluon",       "1",   True,  "boson"),    # v2: pion removed; gluon added (force carrier, clearly a boson)
    ("electron",    "1/2", False, "fermion"),
    ("proton",      "1/2", False, "fermion"),
    ("muon",        "1/2", False, "fermion"),
    ("neutron",     "1/2", False, "fermion"),
]

# F2 novel spin values — restricted to s∈{0,1} bosons and s∈{1/2,3/2} fermions
F2_NOVEL_V3 = [
    # (spin_str, is_integer, stat_class, context_note)
    ("0",   True,  "boson",   "a hypothetical spin-0 scalar boson"),
    ("0",   True,  "boson",   "a hypothetical neutral spin-0 meson"),
    ("1",   True,  "boson",   "a hypothetical spin-1 gauge boson"),
    ("1",   True,  "boson",   "a hypothetical massive spin-1 carrier particle"),
    ("1/2", False, "fermion", "a hypothetical spin-1/2 particle"),
    ("1/2", False, "fermion", "a hypothetical fundamental spin-1/2 lepton"),
    ("3/2", False, "fermion", "a hypothetical spin-3/2 baryon"),
    ("3/2", False, "fermion", "a hypothetical excited baryon resonance with spin 3/2"),
]

F3_TEMPLATES_INT_V3 = [
    "A particle carries integer spin. According to the spin-statistics theorem, is it a boson or a fermion?",
    "A quantum particle possesses integer angular momentum (integer spin). Is it a boson or a fermion?",
    "A particle whose spin is an integer. By the spin-statistics theorem, is it a boson or a fermion?",
    "Consider a particle with an integer-valued spin quantum number. Is it a boson or a fermion?",
]
F3_TEMPLATES_HALF_V3 = [
    "A particle carries half-integer spin. According to the spin-statistics theorem, is it a boson or a fermion?",
    "A quantum particle possesses half-integer angular momentum. Is it a boson or a fermion?",
    "A particle whose spin is a half-integer value. By the spin-statistics theorem, is it a boson or a fermion?",
    "Consider a particle with a half-integer-valued spin quantum number. Is it a boson or a fermion?",
]


def spin_statistics_v3_prompts() -> list[dict]:
    records = []

    # F1: named particle recall
    for name, spin_str, is_int, stat_class in F1_PARTICLES_V3:
        wrong = "fermion" if stat_class == "boson" else "boson"
        article = "an" if name[0].lower() in "aeiou" else "a"
        records.append({
            "prompt": f"Is {article} {name} a boson or a fermion? Answer:",
            "wording_family": "F1_particle_name",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": name, "spin_value": spin_str, "stat_class": stat_class,
        })

    # F2: novel spin values (restricted, no s>3/2)
    for spin_str, is_int, stat_class, context in F2_NOVEL_V3:
        wrong = "fermion" if stat_class == "boson" else "boson"
        records.append({
            "prompt": (
                f"Consider {context} with spin quantum number s = {spin_str}. "
                f"Based on the spin-statistics theorem, is it a boson or a fermion? Answer:"
            ),
            "wording_family": "F2_novel_spin",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": f"hypothetical s={spin_str}", "spin_value": spin_str, "stat_class": stat_class,
        })

    # F3: spin descriptor — four distinct phrasings per type
    boson_idx, fermion_idx = 0, 0
    for name, spin_str, is_int, stat_class in F1_PARTICLES_V3:
        wrong = "fermion" if stat_class == "boson" else "boson"
        if is_int:
            tmpl = F3_TEMPLATES_INT_V3[boson_idx % len(F3_TEMPLATES_INT_V3)]
            boson_idx += 1
        else:
            tmpl = F3_TEMPLATES_HALF_V3[fermion_idx % len(F3_TEMPLATES_HALF_V3)]
            fermion_idx += 1
        records.append({
            "prompt": f"{tmpl} Answer:",
            "wording_family": "F3_spin_descriptor",
            "correct_answer":  f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": name, "spin_value": spin_str, "stat_class": stat_class,
        })

    # F4: hybrid (same novel values as F2 + explicit integer/half-integer label)
    # F2 vs F4 comparison: if F4>>F2, bottleneck is numeral→integer classification
    for spin_str, is_int, stat_class, context in F2_NOVEL_V3:
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
            "particle_name": f"hypothetical s={spin_str} (labelled)", "spin_value": spin_str, "stat_class": stat_class,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  chemistry_acid_base_mini
#
# Brønsted-Lowry: proton (H+) donor = acid; proton acceptor = base.
# Answer tokens: ' acid' / ' base' — both common single-token English words.
# No systematic prior bias expected (unlike ' classical' vs ' relativistic').
#
# F1: named chemical (recall)
# F2: pH value (threshold rule: pH < 7 → acid, pH > 7 → base)
# F3: proton-transfer description (verbal rule — directly matches textbook definition)
# F4: reaction context (what species forms in water)
#
# 8 acids + 8 bases × 4 families = 32 prompts.
# ─────────────────────────────────────────────────────────────────────────────

# 4 acids + 4 bases = 8 chemicals × 4 families = 32 prompts
# pH values chosen to be unique across the full set
ACIDS = [
    # (name, ph_unique, proton_desc, water_product)
    ("hydrochloric acid (HCl)",    "1",  "donates a proton (H+) to water",                    "produces H3O+ ions when dissolved in water"),
    ("sulfuric acid (H2SO4)",      "0",  "donates a proton to a solvent molecule",             "increases the H+ (hydronium) concentration in water"),
    ("acetic acid (CH3COOH)",      "3",  "acts as a proton donor in Brønsted-Lowry theory",    "partially ionises to release H+ ions in water"),
    ("carbonic acid (H2CO3)",      "5",  "donates H+ to water forming bicarbonate",            "reacts with water to release hydrogen ions"),
]

BASES = [
    # (name, ph_unique, proton_desc, water_product)
    ("sodium hydroxide (NaOH)",    "13", "accepts a proton from water (acts as H+ acceptor)", "releases OH- ions when dissolved in water"),
    ("ammonia (NH3)",              "11", "accepts a proton in Brønsted-Lowry theory",          "reacts with water to produce OH- ions"),
    ("sodium bicarbonate (NaHCO3)","9",  "accepts H+ ions from the solvent",                   "hydrolyses in water to produce a mildly alkaline OH- solution"),
    ("calcium hydroxide (Ca(OH)2)","12", "acts as a proton acceptor in aqueous solution",     "dissociates to release hydroxide (OH-) ions in water"),
]


def acid_base_prompts() -> list[dict]:
    records = []

    all_chemicals = [(name, ph, desc, prod, "acid") for name, ph, desc, prod in ACIDS] + \
                    [(name, ph, desc, prod, "base") for name, ph, desc, prod in BASES]

    for name, ph, proton_desc, water_prod, chem_class in all_chemicals:
        wrong = "base" if chem_class == "acid" else "acid"
        meta = {
            "correct_answer":  f" {chem_class}",
            "incorrect_answer": f" {wrong}",
            "chemical_name": name,
            "chem_class": chem_class,
        }

        # F1: named chemical (recall)
        records.append({
            "prompt": f"Is {name} an acid or a base? Answer:",
            "wording_family": "F1_named_chemical",
            **meta,
        })

        # F2: pH value (threshold rule)
        direction = "below" if chem_class == "acid" else "above"
        records.append({
            "prompt": (
                f"A solution has pH = {ph}. "
                f"Is this solution an acid or a base? Answer:"
            ),
            "wording_family": "F2_ph_value",
            **meta,
        })

        # F3: proton-transfer description (direct verbal rule)
        records.append({
            "prompt": (
                f"A substance that {proton_desc}. "
                f"According to the Brønsted-Lowry definition, is it an acid or a base? Answer:"
            ),
            "wording_family": "F3_proton_transfer",
            **meta,
        })

        # F4: reaction-product context
        records.append({
            "prompt": (
                f"A compound that {water_prod}. "
                f"Is this compound an acid or a base? Answer:"
            ),
            "wording_family": "F4_reaction_context",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3.  physics_charge_sign_mini
#
# "Is the electric charge of [X] positive or negative?"
# Answer tokens: ' positive' / ' negative' — both common single-token words.
# No systematic prior bias expected.
#
# F1: named particle or ion (recall)
# F2: structural description (atomic structure rule)
# F3: physics-context description (interaction, decay context)
# F4: charge algebra (sum of constituent charges)
#
# 8 positive + 8 negative × 4 families = 32 prompts.
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_ENTITIES = [
    # (name, structural_desc, context_desc, algebra_desc)
    ("proton",
     "a particle with one more proton than electron count (a bare proton, no electrons)",
     "the particle that is not emitted but remains after beta-minus decay of a neutron",
     "a baryon composed of two up quarks (charge +2/3 each) and one down quark (charge -1/3): net charge = +2/3 + 2/3 - 1/3"),
    ("positron",
     "a particle with the same mass as an electron but opposite charge",
     "the antiparticle of the electron, emitted in beta-plus radioactive decay",
     "the antiparticle of the electron; charge = -(charge of electron) = +e"),
    ("alpha particle",
     "a helium-4 nucleus: two protons and two neutrons, no electrons",
     "the particle emitted in alpha radioactive decay from heavy nuclei",
     "contains two protons (each +e) and two neutrons (neutral): net charge = +2e"),
    ("lithium ion (Li+)",
     "a lithium atom that has lost one electron, leaving three protons and two electrons",
     "the ion formed when lithium metal dissolves in an electrolyte solution",
     "3 protons (3e+) and 2 electrons (2e-): net charge = +e"),
]

NEGATIVE_ENTITIES = [
    # (name, structural_desc, context_desc, algebra_desc)
    ("electron",
     "a fundamental lepton with no internal structure and a fixed unit negative charge",
     "the particle emitted in beta-minus radioactive decay of a neutron",
     "a fundamental lepton; by convention, charge = -e (the elementary charge unit)"),
    ("muon",
     "a second-generation lepton with mass ~207 times the electron mass and the same sign of charge",
     "a particle produced in cosmic-ray showers that decays to an electron and two neutrinos",
     "a second-generation lepton; charge = -e (same sign as the electron)"),
    ("antiproton",
     "the antiparticle of the proton, with the same mass but opposite charge",
     "a particle that annihilates with a proton to produce energy and mesons",
     "the antiparticle of the proton; charge = -(+e) = -e"),
    ("fluoride ion (F-)",
     "a fluorine atom that has gained one extra electron, giving it nine protons and ten electrons",
     "the ion formed when fluorine accepts an electron in an ionic bond",
     "9 protons (9e+) and 10 electrons (10e-): net charge = -e"),
]


def charge_sign_prompts() -> list[dict]:
    records = []

    all_entities = [(name, s, c, a, "positive") for name, s, c, a in POSITIVE_ENTITIES] + \
                   [(name, s, c, a, "negative") for name, s, c, a in NEGATIVE_ENTITIES]

    for name, struct_desc, ctx_desc, alg_desc, charge_sign in all_entities:
        wrong = "negative" if charge_sign == "positive" else "positive"
        meta = {
            "correct_answer":  f" {charge_sign}",
            "incorrect_answer": f" {wrong}",
            "entity_name": name,
            "charge_sign": charge_sign,
        }

        # F1: named particle / ion
        records.append({
            "prompt": f"Is the electric charge of a {name} positive or negative? Answer:",
            "wording_family": "F1_named_entity",
            **meta,
        })

        # F2: structural description (atomic structure rule)
        records.append({
            "prompt": (
                f"Consider {struct_desc}. "
                f"Is its net electric charge positive or negative? Answer:"
            ),
            "wording_family": "F2_structural",
            **meta,
        })

        # F3: physics context (interaction / decay context)
        records.append({
            "prompt": (
                f"{ctx_desc}. "
                f"Is the electric charge of this particle positive or negative? Answer:"
            ),
            "wording_family": "F3_physics_context",
            **meta,
        })

        # F4: charge algebra (explicit sum)
        records.append({
            "prompt": (
                f"{alg_desc}. "
                f"What is the sign of the net electric charge — positive or negative? Answer:"
            ),
            "wording_family": "F4_charge_algebra",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating round-3 candidate-set screening prompts...")
    print()

    behaviours = {
        "physics_spin_statistics_v3_mini": spin_statistics_v3_prompts(),
        "chemistry_acid_base_mini":         acid_base_prompts(),
        "physics_charge_sign_mini":         charge_sign_prompts(),
    }

    for beh, records in behaviours.items():
        path = OUT_DIR / f"{beh}_train.jsonl"
        write_jsonl(path, records)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
