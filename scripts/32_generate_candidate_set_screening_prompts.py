"""
Generate mini prompt sets for candidate-set behaviour screening.

Four behaviours, ~16-32 prompts each, designed for fast baseline evaluation:
  physics_parity_rule_mini       — orbital parity (even/odd) by (-1)^l
  physics_spin_statistics_mini   — boson vs fermion classification
  physics_approximation_regime_mini — classical vs relativistic from v/c
  physics_e1_selection_mini      — E1 selection rule (cross-shell only)

Output: data/prompts/<behaviour>_train.jsonl
"""

import json
import os
from pathlib import Path

OUT_DIR = Path("data/prompts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} prompts → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. physics_parity_rule_mini
# Rule: parity = (-1)^l;  l even → "even";  l odd → "odd"
# 8 orbital instances × 4 families = 32 prompts
# ─────────────────────────────────────────────────────────────────────────────

PARITY_ORBITALS = [
    # (spectroscopic_label, orbital_name, l_value, parity)
    ("1s", "s", 0, "even"),
    ("3s", "s", 0, "even"),
    ("2p", "p", 1, "odd"),
    ("4p", "p", 1, "odd"),
    ("3d", "d", 2, "even"),
    ("5d", "d", 2, "even"),
    ("4f", "f", 3, "odd"),
    ("5f", "f", 3, "odd"),
]


def parity_prompts() -> list[dict]:
    records = []
    for spec_label, orb_name, l_val, parity in PARITY_ORBITALS:
        wrong = "odd" if parity == "even" else "even"
        meta = {
            "correct_answer": f" {parity}",
            "incorrect_answer": f" {wrong}",
            "spectroscopic_label": spec_label,
            "orbital_name": orb_name,
            "l_value": l_val,
            "parity": parity,
        }

        # F1: orbital name
        records.append({
            "prompt": f"What is the parity of an {orb_name} orbital under spatial inversion? Even or odd?",
            "wording_family": "F1_orbital_name",
            **meta,
        })

        # F2: l quantum number
        records.append({
            "prompt": f"A quantum state has orbital angular momentum quantum number l={l_val}. "
                      f"Is the parity of this state even or odd?",
            "wording_family": "F2_l_value",
            **meta,
        })

        # F3: spectroscopic notation
        records.append({
            "prompt": f"Consider the {spec_label} orbital. Is its parity even or odd?",
            "wording_family": "F3_spectroscopic",
            **meta,
        })

        # F4: context sentence
        parity_sign = "+1" if parity == "even" else "-1"
        records.append({
            "prompt": f"An electron occupies the {spec_label} orbital (l={l_val}). "
                      f"Applying the parity operator yields a factor of (-1)^l. "
                      f"Is the parity of this orbital even or odd?",
            "wording_family": "F4_context",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. physics_spin_statistics_mini
# Rule: integer spin → boson;  half-integer spin → fermion
# 8 particles × 3 families = 24 prompts
# ─────────────────────────────────────────────────────────────────────────────

SPIN_PARTICLES = [
    # (name, spin_str, spin_is_integer, stat_class)
    ("photon",   "1",   True,  "boson"),
    ("W boson",  "1",   True,  "boson"),
    ("gluon",    "1",   True,  "boson"),
    ("pion",     "0",   True,  "boson"),
    ("electron", "1/2", False, "fermion"),
    ("proton",   "1/2", False, "fermion"),
    ("neutron",  "1/2", False, "fermion"),
    ("muon",     "1/2", False, "fermion"),
]


def spin_statistics_prompts() -> list[dict]:
    records = []
    for name, spin_str, is_integer, stat_class in SPIN_PARTICLES:
        wrong = "fermion" if stat_class == "boson" else "boson"
        meta = {
            "correct_answer": f" {stat_class}",
            "incorrect_answer": f" {wrong}",
            "particle_name": name,
            "spin_value": spin_str,
            "stat_class": stat_class,
        }

        # F1: particle name (recall)
        records.append({
            "prompt": f"Is a {name} a boson or a fermion?",
            "wording_family": "F1_particle_name",
            **meta,
        })

        # F2: explicit spin value (rule application)
        records.append({
            "prompt": f"A particle has spin quantum number s = {spin_str}. "
                      f"Is it a boson or a fermion?",
            "wording_family": "F2_spin_value",
            **meta,
        })

        # F3: integer/half-integer descriptor (conceptual)
        if is_integer:
            descriptor = "integer spin"
        else:
            descriptor = "half-integer spin"
        records.append({
            "prompt": f"A particle carries {descriptor}. "
                      f"According to the spin-statistics theorem, is it a boson or a fermion?",
            "wording_family": "F3_spin_descriptor",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. physics_approximation_regime_mini
# Rule: v/c << 1 → classical;  v/c → 1 → relativistic
# 4 classical + 4 relativistic velocities × 3 families = 24 prompts
# ─────────────────────────────────────────────────────────────────────────────

REGIME_VELOCITIES = [
    # (v_over_c, regime, gamma_approx, pct_str, verbal_desc)
    (0.001, "classical",    "≈1.000",  "0.1%",  "moving very slowly compared to light"),
    (0.005, "classical",    "≈1.000",  "0.5%",  "moving at a very small fraction of the speed of light"),
    (0.01,  "classical",    "≈1.000",  "1%",    "moving at one percent of the speed of light"),
    (0.02,  "classical",    "≈1.000",  "2%",    "moving at two percent of the speed of light"),
    (0.70,  "relativistic", "≈1.400",  "70%",   "moving at seventy percent of the speed of light"),
    (0.85,  "relativistic", "≈1.898",  "85%",   "moving at eighty-five percent of the speed of light"),
    (0.95,  "relativistic", "≈3.203",  "95%",   "moving at ninety-five percent of the speed of light"),
    (0.99,  "relativistic", "≈7.089",  "99%",   "moving at ninety-nine percent of the speed of light"),
]


def approximation_regime_prompts() -> list[dict]:
    records = []
    for v_c, regime, gamma, pct_str, verbal in REGIME_VELOCITIES:
        wrong = "relativistic" if regime == "classical" else "classical"
        meta = {
            "correct_answer": f" {regime}",
            "incorrect_answer": f" {wrong}",
            "v_over_c": v_c,
            "regime": regime,
        }

        # F1: decimal fraction of c
        records.append({
            "prompt": f"A particle moves at v = {v_c}c. "
                      f"Should its motion be treated as classical or relativistic?",
            "wording_family": "F1_decimal_fraction",
            **meta,
        })

        # F2: percentage
        records.append({
            "prompt": f"A particle is {verbal}. "
                      f"Is this a classical or relativistic regime?",
            "wording_family": "F2_percentage",
            **meta,
        })

        # F3: Lorentz factor
        records.append({
            "prompt": f"A particle has Lorentz factor γ {gamma}. "
                      f"Is its dynamics classical or relativistic?",
            "wording_family": "F3_lorentz_factor",
            **meta,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. physics_e1_selection_mini
# Rule: |Δl| = 1 → allowed;  otherwise → forbidden
# Cross-shell only (no same-n) to remove energy-level ambiguity
# 8 transitions × 2 families = 16 prompts
# ─────────────────────────────────────────────────────────────────────────────

E1_TRANSITIONS = [
    # (initial, final, allowed, delta_l, note)
    ("1s", "2p", True,   1, "Δl=+1 cross-shell"),
    ("2p", "3d", True,   1, "Δl=+1 cross-shell"),
    ("2s", "4p", True,   1, "Δl=+1 cross-shell"),
    ("3p", "4d", True,   1, "Δl=+1 cross-shell"),
    ("1s", "2s", False,  0, "Δl=0 cross-shell"),
    ("2p", "3p", False,  0, "Δl=0 cross-shell"),
    ("1s", "3d", False,  2, "Δl=+2 cross-shell"),
    ("3p", "4f", False,  2, "Δl=+2 cross-shell"),
]


def e1_mini_prompts() -> list[dict]:
    records = []
    for init, final, is_allowed, delta_l, note in E1_TRANSITIONS:
        ans = "allowed" if is_allowed else "forbidden"
        wrong = "forbidden" if is_allowed else "allowed"
        meta = {
            "correct_answer": f" {ans}",
            "incorrect_answer": f" {wrong}",
            "initial_state": init,
            "final_state": final,
            "delta_l": delta_l,
            "is_allowed": is_allowed,
            "note": note,
        }

        # F3: spectroscopic notation (cross-shell, no ambiguity)
        records.append({
            "prompt": f"An electron undergoes an electric dipole (E1) transition from {init} to {final}. "
                      f"Is this transition allowed or forbidden by the E1 selection rule?",
            "wording_family": "F3_spectroscopic",
            **meta,
        })

        # F4: explicit Δl
        if delta_l == 1:
            dl_str = f"Δl = +1 (from l={_l(init)} to l={_l(final)})"
        elif delta_l == 0:
            dl_str = f"Δl = 0 (both states have l={_l(init)})"
        else:
            dl_str = f"Δl = {delta_l} (from l={_l(init)} to l={_l(final)})"
        records.append({
            "prompt": f"In an electric dipole (E1) transition, the orbital angular momentum changes by "
                      f"{dl_str}. Is this E1 transition allowed or forbidden?",
            "wording_family": "F4_explicit_delta_l",
            **meta,
        })

    return records


def _l(state: str) -> int:
    """Return l-value from spectroscopic label (e.g. '2p' → 1)."""
    orbital_l = {"s": 0, "p": 1, "d": 2, "f": 3}
    return orbital_l[state[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating candidate-set screening prompts...")
    print()

    behaviours = {
        "physics_parity_rule_mini":         parity_prompts(),
        "physics_spin_statistics_mini":     spin_statistics_prompts(),
        "physics_approximation_regime_mini": approximation_regime_prompts(),
        "physics_e1_selection_mini":        e1_mini_prompts(),
    }

    for beh, records in behaviours.items():
        path = OUT_DIR / f"{beh}_train.jsonl"
        write_jsonl(path, records)

        # Quick balance check
        n_correct = sum(1 for r in records if "allowed" in r["correct_answer"] or
                        "even" in r["correct_answer"] or "boson" in r["correct_answer"] or
                        "classical" in r["correct_answer"])
        n_pos = sum(1 for r in records if r["correct_answer"].strip() ==
                    records[0]["correct_answer"].strip())
        pos_ans = records[0]["correct_answer"].strip()
        neg_ans = records[0]["incorrect_answer"].strip()
        n_pos = sum(1 for r in records if r["correct_answer"].strip() == pos_ans)
        n_neg = len(records) - n_pos
        print(f"    Balance: {n_pos} '{pos_ans}' / {n_neg} '{neg_ans}'")

        families = sorted(set(r.get("wording_family", "?") for r in records))
        print(f"    Families: {families}")
        print()

    print("Done.")
    print()
    print("Token audit (run on CSD3 after env activation):")
    print("  python3 -c \"")
    print("  from transformers import AutoTokenizer")
    print("  tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')")
    answers = [" even", " odd", " boson", " fermion", " classical", " relativistic", " allowed", " forbidden"]
    for a in answers:
        print(f"  print(len(tok.encode('{a}', add_special_tokens=False)), repr('{a}'))")
    print("  \"")


if __name__ == "__main__":
    main()
