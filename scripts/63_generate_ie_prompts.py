"""
Generate clean prompt set for physics_intensive_extensive_v1.

Answer format: ' intensive' (token 36195) / ' extensive' (token 16376) — both single Qwen3 tokens.
Using the class name directly as the answer is cleaner than Yes/No because:
  - Same correct_answer for all wordings about a given property (no wording-dependent flipping)
  - ND = logp(intensive) - logp(extensive) directly measures abstraction confidence
  - Simpler attribution analysis

Properties: 20 intensive + 20 extensive (clear-cut; excludes voltage, specific gravity, raw heat capacity)
Wording families: W0 (combine), W2 (split), W4 (additive), W6 (symbolic) — 3 variants each
Total: 40 × 4 × 3 = 480 prompts → 384 train / 96 test (stratified by class)

Usage:
    python scripts/63_generate_ie_prompts.py
    python scripts/63_generate_ie_prompts.py --out_dir data/prompts --seed 42
"""

import argparse
import json
import random
from pathlib import Path

# ── Properties ────────────────────────────────────────────────────────────────
# (name, domain_tag, difficulty)  — all physics domain

INTENSIVE = [
    ("temperature",            "thermodynamics",   "easy"),
    ("pressure",               "thermodynamics",   "easy"),
    ("density",                "thermodynamics",   "easy"),
    ("molar concentration",    "chemistry",        "easy"),
    ("molar mass",             "chemistry",        "medium"),
    ("viscosity",              "fluid dynamics",   "medium"),
    ("refractive index",       "optics",           "medium"),
    ("specific heat capacity", "thermodynamics",   "medium"),
    ("boiling point",          "thermodynamics",   "medium"),
    ("electrical resistivity", "electromagnetism", "medium"),
    ("pH",                     "chemistry",        "easy"),
    ("melting point",          "thermodynamics",   "medium"),
    ("surface tension",        "fluid dynamics",   "medium"),
    ("thermal conductivity",   "thermodynamics",   "medium"),
    ("dielectric constant",    "electromagnetism", "hard"),
    ("chemical potential",     "thermodynamics",   "hard"),
    ("specific volume",        "thermodynamics",   "hard"),
    ("magnetic permeability",  "electromagnetism", "hard"),
    ("vapor pressure",         "thermodynamics",   "medium"),
    ("hardness",               "materials",        "medium"),
]

EXTENSIVE = [
    ("mass",                       "mechanics",       "easy"),
    ("volume",                     "geometry",        "easy"),
    ("internal energy",            "thermodynamics",  "easy"),
    ("entropy",                    "thermodynamics",  "medium"),
    ("electric charge",            "electromagnetism","easy"),
    ("linear momentum",            "mechanics",       "medium"),
    ("amount of substance",        "chemistry",       "easy"),
    ("number of particles",        "statistical mechanics","easy"),
    ("total heat capacity",        "thermodynamics",  "medium"),
    ("enthalpy",                   "thermodynamics",  "medium"),
    ("Gibbs free energy",          "thermodynamics",  "hard"),
    ("angular momentum",           "mechanics",       "medium"),
    ("Helmholtz free energy",      "thermodynamics",  "hard"),
    ("total kinetic energy",       "mechanics",       "medium"),
    ("gravitational potential energy","mechanics",    "medium"),
    ("total electric flux",        "electromagnetism","hard"),
    ("total magnetic flux",        "electromagnetism","hard"),
    ("electric dipole moment",     "electromagnetism","hard"),
    ("total work done",            "mechanics",       "medium"),
    ("magnetic moment",            "electromagnetism","hard"),
]

INTENSIVE_NAMES = {p[0] for p in INTENSIVE}


# ── Wording templates ────────────────────────────────────────────────────────
# Each returns a prompt string ending with "Answer:"
# Correct answer is always " intensive" or " extensive" (leading space, single token)

WORDINGS = {
    "W0_combine": [
        lambda p: (
            f"Two identical systems are combined into one. "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"Doubling the amount of material by merging two identical samples: "
            f"is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"A physicist combines two equal portions of a substance and measures {p} "
            f"before and after. Is {p} intensive or extensive? Answer:"
        ),
    ],
    "W2_split": [
        lambda p: (
            f"A material sample is cut into two equal halves. "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"A system is partitioned into N identical subsystems, "
            f"each inheriting some {p} from the whole. "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"Consider dividing a container of material into several equal portions "
            f"and measuring {p} of each portion. "
            f"Is {p} intensive or extensive? Answer:"
        ),
    ],
    "W4_additive": [
        lambda p: (
            f"System A has {p} = X; system B has {p} = Y. "
            f"When combined, the total {p} equals X + Y (extensive) or equals X = Y (intensive). "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"Is {p} additive — does the whole system's {p} equal "
            f"the sum of its parts' {p} values? "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"For two subsystems A and B: if {p}(A∪B) = {p}(A) + {p}(B), "
            f"it is extensive; if {p}(A∪B) = {p}(A) = {p}(B), it is intensive. "
            f"Is {p} intensive or extensive? Answer:"
        ),
    ],
    "W6_symbolic": [
        lambda p: (
            f"Let Q(S) denote {p} of system S. "
            f"Does Q(2S) = 2·Q(S) hold for {p}? "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"For {p} denoted Q: does Q scale as Q(λS) = λ·Q(S) for all λ > 0? "
            f"Is {p} intensive or extensive? Answer:"
        ),
        lambda p: (
            f"Scaling law: Q(kS) = k·Q(S) defines an extensive quantity; "
            f"Q(kS) = Q(S) defines an intensive quantity. "
            f"Which applies to {p}? Is {p} intensive or extensive? Answer:"
        ),
    ],
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_all(seed=42):
    rng = random.Random(seed)
    rows = []
    prop_id = 0
    for (name, domain, diff) in INTENSIVE + EXTENSIVE:
        cls = "intensive" if name in INTENSIVE_NAMES else "extensive"
        correct   = " intensive" if cls == "intensive" else " extensive"
        incorrect = " extensive" if cls == "intensive" else " intensive"

        for wf, variants in WORDINGS.items():
            for vi, tmpl in enumerate(variants):
                prompt = tmpl(name)
                rows.append({
                    "prompt_id":        f"ie_{name.replace(' ','_')}_{wf}_v{vi}",
                    "prompt":           prompt,
                    "correct_answer":   correct,
                    "incorrect_answer": incorrect,
                    "abstraction_class":cls,
                    "property":         name,
                    "wording_family":   wf,
                    "wording_variant":  vi,
                    "domain":           "physics",
                    "sub_domain":       domain,
                    "difficulty":       diff,
                    "experiment_type":  "core",
                    "multi_token_answer": False,
                })
        prop_id += 1

    rng.shuffle(rows)
    return rows


def stratified_split(rows, test_frac=0.2, seed=99):
    """Split keeping abstraction_class × wording_family balance."""
    rng = random.Random(seed)
    by_key = {}
    for r in rows:
        key = (r["abstraction_class"], r["wording_family"])
        by_key.setdefault(key, []).append(r)
    train, test = [], []
    for cls_rows in by_key.values():
        rng.shuffle(cls_rows)
        n_test = max(1, round(len(cls_rows) * test_frac))
        test.extend(cls_rows[:n_test])
        train.extend(cls_rows[n_test:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(rows):>4d} rows → {path}")


def print_stats(rows, label=""):
    from collections import Counter
    print(f"\n{label} ({len(rows)} prompts):")
    cls_c = Counter(r["abstraction_class"] for r in rows)
    wf_c  = Counter(r["wording_family"]    for r in rows)
    diff_c = Counter(r["difficulty"]       for r in rows)
    print(f"  Classes:  {dict(cls_c)}")
    print(f"  Wordings: {dict(wf_c)}")
    print(f"  Difficulty: {dict(diff_c)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",   default="data/prompts")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    BEHAVIOUR = "physics_intensive_extensive_v1"
    out = Path(args.out_dir)

    rows = generate_all(seed=args.seed)
    train, test = stratified_split(rows, args.test_frac, args.seed)

    print_stats(train, "Train")
    print_stats(test,  "Test")

    save_jsonl(train, out / f"{BEHAVIOUR}_train.jsonl")
    save_jsonl(test,  out / f"{BEHAVIOUR}_test.jsonl")

    # Verify token audit
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        for ans in [" intensive", " extensive"]:
            ids = tok.encode(ans, add_special_tokens=False)
            status = "✓ single token" if len(ids) == 1 else f"✗ MULTI-TOKEN ({ids})"
            print(f"  {repr(ans)} = {ids} {status}")
    except Exception:
        print("  (token audit skipped — transformers not available locally)")

    print(f"\nTotal: {len(rows)} prompts ({len(train)} train + {len(test)} test)")
    print(f"Answer tokens: ' intensive' (36195) / ' extensive' (16376)")


if __name__ == "__main__":
    main()
