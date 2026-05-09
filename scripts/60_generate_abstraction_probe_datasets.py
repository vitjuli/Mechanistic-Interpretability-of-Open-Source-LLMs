"""
Abstraction probe dataset generator — Exploratory Phase.

Generates JSONL prompt families testing whether Qwen3 forms an invariant
abstract representation (x_surface → z_abstract → y_surface) independent of
surface wording, domain, or notation.

Five families:
  A. IntensiveExtensive  — scale invariance: does quantity change with system size?
  B. ScalingLaw          — mathematical power-law scaling of geometric/physical quantities
  C. RepresentationEq    — symbolic ≡ natural-language equivalence of math expressions
  D. CrossDomain         — same intensive/extensive structure across physics/economics/statistics
  E. ConservationLaw     — temporal invariance (conserved vs non-conserved quantities)

All prompts have binary ' Yes' / ' No' answers (single Qwen3 tokens 7414 / 2308).

Usage:
    python scripts/60_generate_abstraction_probe_datasets.py
    python scripts/60_generate_abstraction_probe_datasets.py --out_dir data/prompts/abstraction
"""

import argparse
import json
import random
from pathlib import Path

# ── Family A: Intensive vs Extensive ────────────────────────────────────────────

PROPERTIES_IE = [
    # ── Intensive (does NOT scale with system size) ──
    # name, domain, article, adversarial_flag, difficulty
    ("temperature",          "thermodynamics",   "the", False, "easy"),
    ("pressure",             "thermodynamics",   "the", False, "easy"),
    ("density",              "thermodynamics",   "the", False, "easy"),
    ("concentration",        "chemistry",        "the", False, "easy"),
    ("molar mass",           "chemistry",        "the", False, "medium"),
    ("viscosity",            "fluid dynamics",   "the", False, "medium"),
    ("refractive index",     "optics",           "the", False, "medium"),
    ("specific heat capacity","thermodynamics",  "the", False, "hard"),
    ("boiling point",        "thermodynamics",   "the", False, "medium"),
    ("electrical resistivity","electromagnetism","the", False, "hard"),
    ("pH",                   "chemistry",        "the", False, "easy"),
    ("melting point",        "thermodynamics",   "the", False, "medium"),
    # Adversarial intensive (sounds like it should scale but doesn't)
    ("voltage",              "electromagnetism", "the", True,  "hard"),
    ("specific gravity",     "fluid dynamics",   "the", True,  "hard"),
    # ── Extensive (DOES scale with system size) ──
    ("mass",                 "mechanics",        "the", False, "easy"),
    ("volume",               "geometry",         "the", False, "easy"),
    ("energy",               "thermodynamics",   "the", False, "easy"),
    ("entropy",              "thermodynamics",   "the", False, "hard"),
    ("electric charge",      "electromagnetism", "the", False, "medium"),
    ("momentum",             "mechanics",        "the", False, "medium"),
    ("heat capacity",        "thermodynamics",   "the", True,  "hard"),  # vs specific heat (adv)
    ("internal energy",      "thermodynamics",   "the", False, "medium"),
    ("amount of substance",  "chemistry",        "the", False, "medium"),
    ("total enthalpy",       "thermodynamics",   "the", False, "hard"),
    ("number of particles",  "statistical mechanics","the", False, "easy"),
]

INTENSIVE_NAMES = {r[0] for r in PROPERTIES_IE[:14]}   # first 14 are intensive

def _ie_class(name):
    return "intensive" if name in INTENSIVE_NAMES else "extensive"

def _ie_answer(name, wording_key):
    """Return ' Yes' or ' No' for each (property, wording) combination."""
    is_ext = _ie_class(name) == "extensive"
    # Wording families that expect "Yes" for extensive, "No" for intensive:
    yes_for_ext = {
        "W0_combine_doubles", "W1_scale_amount", "W3_sample_size",
        "W4_additive",        "W5_expert_claim", "W6_symbolic",
        "W7_adv_claim",
    }
    # Wording families that expect "Yes" for intensive, "No" for extensive:
    yes_for_int = {"W2_split_preserves"}

    if wording_key in yes_for_ext:
        return " Yes" if is_ext else " No"
    elif wording_key in yes_for_int:
        return " No" if is_ext else " Yes"
    raise ValueError(f"Unknown wording key: {wording_key}")

IE_WORDINGS = {
    "W0_combine_doubles": lambda n, art: (
        f"Two identical containers hold the same material with the same {n}. "
        f"If you combine the contents of both containers, does {art} {n} of the combined system double?"
        f" Answer:",
    ),
    "W1_scale_amount": lambda n, art: (
        f"A scientist has a sample with {art} {n} of X. "
        f"She then takes twice as much of the same material. "
        f"Does {art} {n} of the larger sample equal 2X?"
        f" Answer:",
    ),
    "W2_split_preserves": lambda n, art: (
        f"A container holds a material with {art} {n} of X. "
        f"If the material is divided into two equal halves, does each half have {art} {n} equal to X?"
        f" Answer:",
    ),
    "W3_sample_size": lambda n, art: (
        f"Does {art} {n} of a pure substance depend on how large a sample you measure?"
        f" Answer:",
    ),
    "W4_additive": lambda n, art: (
        f"Is {art} {n} an additive quantity — that is, does the {n} of a combined system "
        f"equal the sum of the {n} values of its parts?"
        f" Answer:",
    ),
    "W5_expert_claim": lambda n, art: (
        f"A physicist states: '{n.capitalize()} doubles when the amount of material doubles.' "
        f"Is this statement correct?"
        f" Answer:",
    ),
    "W6_symbolic": lambda n, art: (
        f"Let Q(S) denote {art} {n} of system S. "
        f"Does the relation Q(2S) = 2·Q(S) hold for {n}?"
        f" Answer:",
    ),
    "W7_adv_claim": lambda n, art: (
        f"Since larger systems contain more material, their {n} must be larger. "
        f"Is this reasoning correct?"
        f" Answer:",
    ),
}

def generate_family_A(seed=42):
    rng = random.Random(seed)
    rows = []
    for (name, domain, art, adv, diff) in PROPERTIES_IE:
        cls = _ie_class(name)
        for wk, tmpl_fn in IE_WORDINGS.items():
            prompt_parts = tmpl_fn(name, art)
            prompt = prompt_parts[0]
            answer = _ie_answer(name, wk)
            rows.append({
                "family":            "A_intensive_extensive",
                "prompt":            prompt,
                "correct_answer":    answer,
                "incorrect_answer":  " No" if answer == " Yes" else " Yes",
                "abstraction_class": cls,
                "property":          name,
                "domain":            domain,
                "wording_family":    wk,
                "adversarial":       adv,
                "difficulty":        diff,
            })
    rng.shuffle(rows)
    return rows


# ── Family B: Mathematical Scaling Laws ─────────────────────────────────────────

SCALING_QUANTITIES = [
    # (quantity, scaling_exponent, domain, description)
    # exponent relative to uniform length scaling: x → k·x
    ("the perimeter of a square",       1, "geometry",    "scales linearly with length"),
    ("the area of a square",            2, "geometry",    "scales as length squared"),
    ("the volume of a cube",            3, "geometry",    "scales as length cubed"),
    ("the surface area of a sphere",    2, "geometry",    "scales as length squared"),
    ("the volume of a sphere",          3, "geometry",    "scales as length cubed"),
    ("an angle in a triangle",          0, "geometry",    "is dimensionless and scale-invariant"),
    ("the ratio of two lengths",        0, "geometry",    "is dimensionless and scale-invariant"),
    ("the density of a material",       0, "physics",     "is intensive and scale-invariant"),  # mass/volume both scale as k³
    ("the mass of an object",           3, "physics",     "scales as volume"),
    ("the gravitational potential energy of an object at height h",
                                        4, "physics",     "scales as mass × height = k^4"),  # tricky
    ("the aspect ratio of a rectangle", 0, "geometry",    "is dimensionless and scale-invariant"),
    ("the circumference-to-diameter ratio of a circle",
                                        0, "geometry",    "equals π regardless of size"),
    ("the frequency of oscillation of a pendulum", -1, "physics",
                                                        "scales as 1/√length"),  # tricky
]

SL_WORDINGS = {
    "SB0_double_invariant": lambda q, exp: (
        f"If all lengths in a geometric system are doubled, does {q} remain unchanged?"
        f" Answer:",
        " Yes" if exp == 0 else " No",
    ),
    "SB1_scale_k": lambda q, exp: (
        f"If all lengths in a system are scaled by a factor k, does {q} also scale by exactly k?"
        f" Answer:",
        " Yes" if exp == 1 else " No",
    ),
    "SB2_scale_k2": lambda q, exp: (
        f"If all lengths in a system are scaled by a factor k, does {q} scale by k squared?"
        f" Answer:",
        " Yes" if exp == 2 else " No",
    ),
    "SB3_double_eight": lambda q, exp: (
        f"If all lengths in a system are doubled, does {q} increase by a factor of 8?"
        f" Answer:",
        " Yes" if exp == 3 else " No",
    ),
}

def generate_family_B(seed=43):
    rng = random.Random(seed)
    rows = []
    for (q, exp, domain, note) in SCALING_QUANTITIES:
        for wk, tmpl_fn in SL_WORDINGS.items():
            prompt, answer = tmpl_fn(q, exp)
            rows.append({
                "family":            "B_scaling_law",
                "prompt":            prompt,
                "correct_answer":    answer,
                "incorrect_answer":  " No" if answer == " Yes" else " Yes",
                "abstraction_class": f"exp_{exp}",
                "property":          q,
                "domain":            domain,
                "wording_family":    wk,
                "adversarial":       False,
                "difficulty":        "medium",
                "scaling_exponent":  exp,
                "note":              note,
            })
    rng.shuffle(rows)
    return rows


# ── Family C: Representation Equivalence ────────────────────────────────────────

REP_EQ_PAIRS = [
    # (concept_name, expr_A, expr_B, are_equivalent, domain)
    ("square",           "x²",           "x · x",                  True,  "algebra"),
    ("square",           "x²",           "x squared",              True,  "algebra"),
    ("square",           "x²",           "x to the power of 2",    True,  "algebra"),
    ("square_root",      "√x",           "x^(1/2)",                True,  "algebra"),
    ("square_root",      "√x",           "the square root of x",   True,  "algebra"),
    ("absolute_value",   "|x|",          "abs(x)",                 True,  "algebra"),
    ("absolute_value",   "|x|",          "the magnitude of x",     True,  "algebra"),
    ("derivative",       "dy/dx",        "the derivative of y with respect to x",
                                                                    True,  "calculus"),
    ("derivative",       "dy/dx",        "d/dx of y",              True,  "calculus"),
    ("infinity",         "1/0",          "undefined",              False, "algebra"),  # 1/0 ≠ ∞ in standard math
    ("not_equal",        "x²",           "2x",                     False, "algebra"),
    ("not_equal",        "√(x+y)",       "√x + √y",                False, "algebra"),  # common error
    ("not_equal",        "(x+y)²",       "x² + y²",                False, "algebra"),  # common error
    ("integral",         "∫f(x)dx",      "the antiderivative of f(x)",
                                                                    True,  "calculus"),
    ("log_product",      "log(x·y)",     "log(x) + log(y)",        True,  "algebra"),
    ("log_power",        "log(x²)",      "2·log(x)",               True,  "algebra"),
]

RC_WORDINGS = {
    "RC0_direct": lambda a, b, eq: (
        f"Is '{a}' the same mathematical expression as '{b}'? Answer:",
        " Yes" if eq else " No",
    ),
    "RC1_function": lambda a, b, eq: (
        f"If f(x) = {a} and g(x) = {b}, are f and g identical functions for all valid x?"
        f" Answer:",
        " Yes" if eq else " No",
    ),
    "RC2_student": lambda a, b, eq: (
        f"A student writes {a} = {b}. Is the student correct? Answer:",
        " Yes" if eq else " No",
    ),
    "RC3_notation": lambda a, b, eq: (
        f"Do the expressions '{a}' and '{b}' represent the same mathematical object?"
        f" Answer:",
        " Yes" if eq else " No",
    ),
}

def generate_family_C(seed=44):
    rng = random.Random(seed)
    rows = []
    for (concept, a, b, eq, domain) in REP_EQ_PAIRS:
        for wk, tmpl_fn in RC_WORDINGS.items():
            prompt, answer = tmpl_fn(a, b, eq)
            rows.append({
                "family":            "C_representation_equivalence",
                "prompt":            prompt,
                "correct_answer":    answer,
                "incorrect_answer":  " No" if answer == " Yes" else " Yes",
                "abstraction_class": "equivalent" if eq else "not_equivalent",
                "property":          concept,
                "domain":            domain,
                "wording_family":    wk,
                "adversarial":       not eq,  # non-equivalent pairs test for errors
                "difficulty":        "hard" if not eq else "medium",
                "expr_a":            a,
                "expr_b":            b,
            })
    rng.shuffle(rows)
    return rows


# ── Family D: Cross-domain Abstraction ──────────────────────────────────────────
# Tests whether intensive/extensive structure generalises across domains.
# Same question structure as Family A but in non-physics domains.

CROSS_DOMAIN_PROPERTIES = [
    # (name, domain, abstraction_class, article, description)
    # Economics — intensive analogs
    ("price per unit",    "economics",         "intensive", "the",
     "cost of one item, independent of quantity"),
    ("profit margin",     "economics",         "intensive", "the",
     "percentage, independent of total volume"),
    ("interest rate",     "economics",         "intensive", "the",
     "percentage, does not scale with principal"),
    # Economics — extensive analogs
    ("total revenue",     "economics",         "extensive", "the",
     "sum across all units sold"),
    ("total cost",        "economics",         "extensive", "the",
     "sum across all inputs"),
    # Information theory — intensive analogs
    ("bits per symbol",   "information theory","intensive", "the",
     "entropy rate, per-symbol quantity"),
    ("compression ratio", "information theory","intensive", "the",
     "ratio, independent of file size"),
    # Information theory — extensive analogs
    ("total bits",        "information theory","extensive", "the",
     "total size of encoded message"),
    ("file size",         "information theory","extensive", "the",
     "scales with content length"),
    # Statistics — intensive analogs
    ("mean",              "statistics",        "intensive", "the",
     "average, does not scale with sample size"),
    ("variance",          "statistics",        "intensive", "the",
     "spread measure, does not scale with sample size"),
    # Statistics — extensive analogs
    ("sum",               "statistics",        "extensive", "the",
     "total, scales with sample size"),
    ("total squared deviation", "statistics",  "extensive", "the",
     "sum of squared errors, scales with n"),
    # Biology — intensive analogs
    ("population density","biology",           "intensive", "the",
     "individuals per area, scale-invariant"),
    ("metabolic rate per kilogram", "biology", "intensive", "the",
     "specific metabolic rate, intensive"),
    # Biology — extensive analogs
    ("total population",  "biology",           "extensive", "the",
     "total count, extensive"),
    ("total metabolic rate", "biology",        "extensive", "the",
     "scales with body mass"),
]

CD_WORDINGS = {
    "CD0_scale_up": lambda n, art, cls: (
        f"In {'{}'}, if you double the scale of a system, does {art} {n} double?"
        .format("the relevant domain"),
        " Yes" if cls == "extensive" else " No",
    ),
    "CD1_combine": lambda n, art, cls: (
        f"If two identical systems are combined, does {art} {n} of the combined system "
        f"equal twice {art} {n} of each individual system?"
        f" Answer:",
        " Yes" if cls == "extensive" else " No",
    ),
    "CD2_subsystem": lambda n, art, cls: (
        f"Consider a large system and a subsystem half its size. "
        f"Is {art} {n} of the subsystem equal to {art} {n} of the full system?"
        f" Answer:",
        " Yes" if cls == "intensive" else " No",
    ),
    "CD3_proportional": lambda n, art, cls: (
        f"Is {art} {n} proportional to the size of the system? Answer:",
        " Yes" if cls == "extensive" else " No",
    ),
}

def generate_family_D(seed=45):
    rng = random.Random(seed)
    rows = []
    for (name, domain, cls, art, desc) in CROSS_DOMAIN_PROPERTIES:
        for wk, tmpl_fn in CD_WORDINGS.items():
            # Fix the CD0 template which has a placeholder
            if wk == "CD0_scale_up":
                if cls == "extensive":
                    prompt = (f"In {domain}, if you scale up a system by a factor of two, "
                              f"does {art} {name} also double? Answer:")
                else:
                    prompt = (f"In {domain}, if you scale up a system by a factor of two, "
                              f"does {art} {name} remain the same? Answer:")
                answer = " Yes"
            else:
                prompt, answer = tmpl_fn(name, art, cls)
            rows.append({
                "family":            "D_cross_domain",
                "prompt":            prompt,
                "correct_answer":    answer,
                "incorrect_answer":  " No" if answer == " Yes" else " Yes",
                "abstraction_class": cls,
                "property":          name,
                "domain":            domain,
                "wording_family":    wk,
                "adversarial":       False,
                "difficulty":        "medium",
                "description":       desc,
            })
    rng.shuffle(rows)
    return rows


# ── Family E: Conservation Laws ─────────────────────────────────────────────────
# Tests temporal invariance: is this quantity conserved in an isolated system?

CONSERVATION_QUANTITIES = [
    # (name, conserved, domain, article, context)
    # Conserved
    ("total energy",              True,  "mechanics",       "the", "isolated system"),
    ("total linear momentum",     True,  "mechanics",       "the", "no external forces"),
    ("total angular momentum",    True,  "mechanics",       "the", "no external torques"),
    ("total electric charge",     True,  "electromagnetism","the", "isolated system"),
    ("total mass-energy",         True,  "relativity",      "the", "isolated system"),
    ("total baryon number",       True,  "particle physics","the", "any interaction"),
    # Non-conserved (or only approximately conserved)
    ("temperature",               False, "thermodynamics",  "the", "irreversible processes"),
    ("entropy",                   False, "thermodynamics",  "the", "can only increase"),
    ("kinetic energy",            False, "mechanics",       "the", "inelastic collisions"),
    ("mechanical energy",         False, "mechanics",       "the", "friction dissipates it"),
    ("order of a system",         False, "thermodynamics",  "the", "tends to decrease"),
    ("velocity of each particle", False, "mechanics",       "the", "collisions change it"),
]

CL_WORDINGS = {
    "CL0_isolated": lambda n, art, cons: (
        f"In a perfectly isolated system with no external interactions, "
        f"does {art} {n} remain constant over time? Answer:",
        " Yes" if cons else " No",
    ),
    "CL1_process": lambda n, art, cons: (
        f"During any physical process in a closed system, "
        f"is {art} {n} always the same before and after the process? Answer:",
        " Yes" if cons else " No",
    ),
    "CL2_symmetry": lambda n, art, cons: (
        f"Is {art} {n} a conserved quantity — meaning its total value cannot change "
        f"due to internal interactions? Answer:",
        " Yes" if cons else " No",
    ),
    "CL3_collision": lambda n, art, cons: (
        f"Two objects collide in empty space with no external forces. "
        f"Is {art} {n} of the two-object system the same before and after the collision?"
        f" Answer:",
        " Yes" if cons else " No",
    ),
}

def generate_family_E(seed=46):
    rng = random.Random(seed)
    rows = []
    for (name, cons, domain, art, ctx) in CONSERVATION_QUANTITIES:
        for wk, tmpl_fn in CL_WORDINGS.items():
            prompt, answer = tmpl_fn(name, art, cons)
            rows.append({
                "family":            "E_conservation_law",
                "prompt":            prompt,
                "correct_answer":    answer,
                "incorrect_answer":  " No" if answer == " Yes" else " Yes",
                "abstraction_class": "conserved" if cons else "non_conserved",
                "property":          name,
                "domain":            domain,
                "wording_family":    wk,
                "adversarial":       False,
                "difficulty":        "medium",
                "context":           ctx,
            })
    rng.shuffle(rows)
    return rows


# ── Dataset splitting and saving ─────────────────────────────────────────────────

def stratified_split(rows, test_frac=0.2, seed=99):
    """Split preserving abstraction_class balance."""
    rng = random.Random(seed)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["abstraction_class"], []).append(r)
    train, test = [], []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        n_test = max(1, int(len(cls_rows) * test_frac))
        test.extend(cls_rows[:n_test])
        train.extend(cls_rows[n_test:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(rows):>4d} rows → {path}")


def print_stats(family_name, rows):
    from collections import Counter
    cls_counts = Counter(r["abstraction_class"] for r in rows)
    wf_counts  = Counter(r["wording_family"]    for r in rows)
    dom_counts = Counter(r["domain"]            for r in rows)
    adv_n      = sum(1 for r in rows if r.get("adversarial"))
    print(f"  {family_name}: {len(rows)} total | "
          f"classes={dict(cls_counts)} | "
          f"wording_families={len(wf_counts)} | "
          f"domains={len(dom_counts)} | "
          f"adversarial={adv_n}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/prompts/abstraction")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)

    print("Generating abstraction probe datasets…")
    generators = [
        ("A_intensive_extensive",      generate_family_A),
        ("B_scaling_law",              generate_family_B),
        ("C_representation_equiv",     generate_family_C),
        ("D_cross_domain",             generate_family_D),
        ("E_conservation_law",         generate_family_E),
    ]

    all_train, all_test = [], []
    for fname, gen_fn in generators:
        rows = gen_fn(seed=args.seed)
        print_stats(fname, rows)
        train, test = stratified_split(rows, args.test_frac, args.seed)
        save_jsonl(train, out / f"{fname}_train.jsonl")
        save_jsonl(test,  out / f"{fname}_test.jsonl")
        all_train.extend(train)
        all_test.extend(test)

    save_jsonl(all_train, out / "abstraction_all_train.jsonl")
    save_jsonl(all_test,  out / "abstraction_all_test.jsonl")

    total = len(all_train) + len(all_test)
    print(f"\nTotal: {total} prompts ({len(all_train)} train, {len(all_test)} test)")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
