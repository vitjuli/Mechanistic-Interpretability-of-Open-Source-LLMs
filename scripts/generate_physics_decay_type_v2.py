#!/usr/bin/env python3
"""
Generate the v2 expanded dataset for physics_decay_type.

108 prompts across 5 surface families:
  F0 (24): original pilot prompts — standard phrasing
  F1 (28): keyword-free — no alpha/beta/helium/electron in text
  F2 (20): extended nuclide pairs — named isotopes, less common
  F3 (20): phenomenological/observational — detector signatures, energy spectra
  F4 (16): mechanism/theory — Gamow tunnelling, Fermi theory, difficulty="hard"

Concept index: 0 = alpha_decay, 1 = beta_decay
Answer format:  correct/incorrect answers have a leading space (" alpha", " beta")
"""

import json
import shutil
from pathlib import Path

BEHAVIOUR = "physics_decay_type"
BEHAVIOUR_TYPE = "latent_state"
ANSWER_FORMAT = "x_or_y"
PROMPTS_DIR = Path("data/prompts")
OUT_PATH = PROMPTS_DIR / f"{BEHAVIOUR}_train.jsonl"
PILOT_ARCHIVE = PROMPTS_DIR / f"{BEHAVIOUR}_train_n24_pilot.jsonl"


def make_prompt(
    text: str,
    concept: str,           # "alpha_decay" | "beta_decay"
    concept_index: int,     # 0 | 1
    template_idx: int,
    surface_family: str,    # "F0" | "F1" | "F2" | "F3" | "F4"
    keyword_free: bool,
    difficulty: str = "standard",
) -> dict:
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


# ---------------------------------------------------------------------------
# F0 — original 24 pilot prompts (enriched with surface_family / keyword_free)
# ---------------------------------------------------------------------------

F0_ALPHA = [
    "A nucleus emits a particle with charge +2e and mass number 4. Is the decay type alpha or beta?",
    "In a radioactive decay, the atomic number decreases by 2 and the mass number decreases by 4. Is the decay type alpha or beta?",
    "A helium-4 nucleus is emitted during a nuclear decay event. Is the decay type alpha or beta?",
    "The emitted particle carries two protons and two neutrons bound together. Is the decay type alpha or beta?",
    "A uranium-238 nucleus decays to thorium-234. Is the decay type alpha or beta?",
    "Radium-226 emits a particle and becomes radon-222, with atomic number decreasing from 88 to 86. Is the decay type alpha or beta?",
    "The daughter nucleus has atomic number Z-2 and mass number A-4 relative to the parent. Is the decay type alpha or beta?",
    "A tightly bound cluster of 2 protons and 2 neutrons is emitted from a heavy unstable nucleus. Is the decay type alpha or beta?",
    "Polonium-210 loses 2 protons and 2 neutrons in a single emission event, becoming lead-206. Is the decay type alpha or beta?",
    "The emitted particle is identical in composition to a helium-4 nucleus. Is the decay type alpha or beta?",
    "In a nuclear decay, the mass number decreases by 4 and the atomic number decreases by 2. Is the decay type alpha or beta?",
    "The emitted radiation is stopped by a sheet of paper and travels only a few centimetres in air. Is the decay type alpha or beta?",
]

F0_BETA = [
    "A nucleus emits a high-energy electron, with the atomic number increasing by 1 and the mass number unchanged. Is the decay type alpha or beta?",
    "In a radioactive decay, the proton number increases by 1 while the nucleon number stays the same. Is the decay type alpha or beta?",
    "A neutron inside the nucleus transforms into a proton, releasing an electron and an antineutrino. Is the decay type alpha or beta?",
    "The emitted particle has charge -1 and mass much smaller than a proton. Is the decay type alpha or beta?",
    "Carbon-14 decays to nitrogen-14 by emitting a fast electron. Is the decay type alpha or beta?",
    "The daughter nucleus is an isobar of the parent: same mass number, atomic number increased by 1. Is the decay type alpha or beta?",
    "Tritium decays to helium-3, with atomic number changing from 1 to 2 and mass number remaining 3. Is the decay type alpha or beta?",
    "A weak-force process changes a down quark to an up quark inside the nucleus, emitting a lepton. Is the decay type alpha or beta?",
    "Phosphorus-32 emits a high-energy electron and transforms into sulfur-32. Is the decay type alpha or beta?",
    "In this decay, Z increases by 1 but A is conserved, and the emitted lepton has negligible rest mass relative to nucleons. Is the decay type alpha or beta?",
    "Iodine-131 transforms into xenon-131 by emitting a fast electron. Is the decay type alpha or beta?",
    "The emitted radiation penetrates a few millimetres of aluminium but is stopped by centimetres of lead. Is the decay type alpha or beta?",
]

# F0 difficulties (matching original pilot)
F0_ALPHA_DIFF = ["standard"] * 11 + ["indirect"]
F0_BETA_DIFF  = ["standard"] * 9 + ["indirect", "standard", "indirect"]


# ---------------------------------------------------------------------------
# F1 — keyword-free (no "alpha", "beta", "helium", "electron" in prompt text)
# ---------------------------------------------------------------------------

F1_ALPHA = [
    "A nucleus ejects a bound cluster with charge +2e and atomic mass 4 u. Is the decay type alpha or beta?",
    "The parent nucleus loses 2 units of charge and 4 units of mass number in a single emission. Is the decay type alpha or beta?",
    "The ejected particle is indistinguishable from the nucleus of the lightest inert gas. Is the decay type alpha or beta?",
    "The daughter nucleus sits two places to the left of the parent on the periodic table and four mass units lighter. Is the decay type alpha or beta?",
    "Thorium-230 decays to radium-226, with the daughter having two fewer protons. Is the decay type alpha or beta?",
    "Americium-241 undergoes decay, emitting a particle of charge +2e and producing neptunium-237. Is the decay type alpha or beta?",
    "The emitted particle is a stable nuclide with Z=2, ejected from a heavy parent. Is the decay type alpha or beta?",
    "The nucleus loses a doubly positive, mass-4 fragment in a single decay step. Is the decay type alpha or beta?",
    "Curium-244 transforms into plutonium-240 via emission of a mass-4 particle with charge +2e. Is the decay type alpha or beta?",
    "The ejected fragment has the same nucleon composition as the nucleus of element 2. Is the decay type alpha or beta?",
    "Thorium-232 transforms into radium-228, with atomic number dropping by 2 and mass number by 4. Is the decay type alpha or beta?",
    "The product nucleus has 2 fewer protons and 2 fewer neutrons than the original nucleus. Is the decay type alpha or beta?",
    "A doubly charged particle of mass number 4 is emitted, reducing the parent's Z by 2. Is the decay type alpha or beta?",
    "The emitted cluster has the same proton-to-neutron ratio as the lightest stable nuclide of element 2. Is the decay type alpha or beta?",
]

F1_BETA = [
    "A nucleus gains one unit of charge without any change in mass number. Is the decay type alpha or beta?",
    "The daughter nucleus has the same mass number as the parent but one more proton. Is the decay type alpha or beta?",
    "A lepton with charge -1e is emitted, and the daughter is an isobar of the parent. Is the decay type alpha or beta?",
    "The nucleon count is unchanged, but the proton count increases by 1 after the decay. Is the decay type alpha or beta?",
    "Strontium-90 decays to yttrium-90, with the daughter having Z = 39. Is the decay type alpha or beta?",
    "The emitted particle has charge -e and a rest mass about 1/1836 of a proton mass. Is the decay type alpha or beta?",
    "Cesium-137 transforms into barium-137, with atomic number increasing by one and mass number unchanged. Is the decay type alpha or beta?",
    "The daughter sits one place to the right of the parent on the periodic table, with identical mass number. Is the decay type alpha or beta?",
    "Cobalt-60 transforms into nickel-60 without any change in nucleon count. Is the decay type alpha or beta?",
    "The emitted particle has the same charge as an orbital lepton but originates from within the nucleus. Is the decay type alpha or beta?",
    "Thallium-208 decays to lead-208, with atomic number increasing by 1 and mass number unchanged. Is the decay type alpha or beta?",
    "A nucleus undergoes a weak decay producing an isobar: same A, Z increased by 1. Is the decay type alpha or beta?",
    "Nickel-63 transforms into copper-63, with the mass number remaining constant. Is the decay type alpha or beta?",
    "The transformation increases the proton count by 1 and leaves the nucleon count unchanged. Is the decay type alpha or beta?",
]


# ---------------------------------------------------------------------------
# F2 — extended nuclide pairs (less common isotopes, no qualitative cues)
# ---------------------------------------------------------------------------

F2_ALPHA = [
    "Thorium-230 decays to radium-226. Is the decay type alpha or beta?",
    "Curium-244 transforms into plutonium-240. Is the decay type alpha or beta?",
    "Americium-241 decays to neptunium-237. Is the decay type alpha or beta?",
    "Radon-222 decays to polonium-218. Is the decay type alpha or beta?",
    "Bismuth-212 decays to thallium-208. Is the decay type alpha or beta?",
    "Francium-223 decays to astatine-219. Is the decay type alpha or beta?",
    "Plutonium-239 decays to uranium-235. Is the decay type alpha or beta?",
    "Thorium-232 decays to radium-228. Is the decay type alpha or beta?",
    "Neptunium-237 decays to protactinium-233. Is the decay type alpha or beta?",
    "Astatine-217 decays to bismuth-213. Is the decay type alpha or beta?",
]

F2_BETA = [
    "Strontium-90 decays to yttrium-90. Is the decay type alpha or beta?",
    "Cesium-137 transforms into barium-137. Is the decay type alpha or beta?",
    "Cobalt-60 transforms into nickel-60. Is the decay type alpha or beta?",
    "Nickel-63 transforms into copper-63. Is the decay type alpha or beta?",
    "Thallium-208 decays to lead-208. Is the decay type alpha or beta?",
    "Bismuth-210 decays to polonium-210. Is the decay type alpha or beta?",
    "Lead-210 decays to bismuth-210. Is the decay type alpha or beta?",
    "Actinium-228 decays to thorium-228. Is the decay type alpha or beta?",
    "Selenium-79 decays to bromine-79. Is the decay type alpha or beta?",
    "Technetium-99 decays to ruthenium-99. Is the decay type alpha or beta?",
]


# ---------------------------------------------------------------------------
# F3 — phenomenological / observational (detector signatures, measurable effects)
# ---------------------------------------------------------------------------

F3_ALPHA = [
    "A 5 MeV particle emitted in the decay creates a dense, short, straight track in a cloud chamber. Is the decay type alpha or beta?",
    "The emitted particle cannot penetrate a thin sheet of paper placed between source and detector. Is the decay type alpha or beta?",
    "The emitted particle travels only 3 cm in air before losing all kinetic energy. Is the decay type alpha or beta?",
    "The Geiger-Nuttall law applies: the half-life correlates exponentially with the particle's kinetic energy. Is the decay type alpha or beta?",
    "The emitted particle is doubly ionising and travels in a straight, dense track in photographic emulsion. Is the decay type alpha or beta?",
    "The emitted particle has a discrete, well-defined kinetic energy rather than a continuous spectrum. Is the decay type alpha or beta?",
    "The emitted particle is deflected toward the negative plate in an electric field, indicating positive charge. Is the decay type alpha or beta?",
    "The emitted particle creates approximately 10,000 ion pairs per centimetre in air, indicating high ionisation density. Is the decay type alpha or beta?",
    "The particle is strongly deflected by a thin gold foil due to Coulomb repulsion, consistent with high charge and mass. Is the decay type alpha or beta?",
    "The track in the cloud chamber curves toward negative voltage and shows a doubly positive charge. Is the decay type alpha or beta?",
]

F3_BETA = [
    "The emitted particle produces a long, wispy, curving track in a cloud chamber. Is the decay type alpha or beta?",
    "The observed spectrum of particle energies is continuous from zero up to a maximum endpoint energy. Is the decay type alpha or beta?",
    "The radiation passes through a sheet of paper but is stopped by a few millimetres of aluminium. Is the decay type alpha or beta?",
    "The emitted particle is deflected toward the positive plate in an electric field, indicating negative charge. Is the decay type alpha or beta?",
    "The emitted particle creates sparse ionisation tracks that curve in a magnetic field, indicating charge −e. Is the decay type alpha or beta?",
    "The emitted particle can travel tens of centimetres in air before being stopped. Is the decay type alpha or beta?",
    "The continuous energy distribution of emitted particles led Pauli to predict the existence of the neutrino. Is the decay type alpha or beta?",
    "The emitted radiation penetrates 3 mm of aluminium but not 5 cm of lead. Is the decay type alpha or beta?",
    "The particle's track in a magnetic field bends in the same direction as a singly negative charge. Is the decay type alpha or beta?",
    "The particle causes sparse ionisation compared to a particle of charge +2e travelling at the same speed. Is the decay type alpha or beta?",
]


# ---------------------------------------------------------------------------
# F4 — mechanism / theory (difficulty = "hard")
# ---------------------------------------------------------------------------

F4_ALPHA = [
    "The decay proceeds by quantum tunnelling through the Coulomb barrier, modelled by the Gamow factor. Is the decay type alpha or beta?",
    "The half-life is correlated with the decay energy via the Geiger-Nuttall relation, reflecting barrier penetration probability. Is the decay type alpha or beta?",
    "The pre-formed cluster model treats the emitted particle as pre-existing inside the parent nucleus before tunnelling out. Is the decay type alpha or beta?",
    "The extraordinarily high binding energy per nucleon of the emitted particle (7.07 MeV/nucleon) provides the thermodynamic driving force for this decay. Is the decay type alpha or beta?",
    "The strong nuclear force confines the emitted cluster inside the nucleus, but it has a finite probability of tunnelling through the Coulomb potential barrier. Is the decay type alpha or beta?",
    "The decay is energetically favoured when the Q-value defined as M_parent − M_daughter − M_cluster is positive. Is the decay type alpha or beta?",
    "The observed discrete energy spectrum of the emitted particle reflects the quantised energy levels of the daughter nucleus. Is the decay type alpha or beta?",
    "This decay mode dominates the instability of heavy nuclei with Z > 82, because Coulomb repulsion overwhelms the nuclear force over large distances. Is the decay type alpha or beta?",
]

F4_BETA = [
    "The decay is mediated by W-boson exchange, with the effective coupling set by the Fermi constant G_F. Is the decay type alpha or beta?",
    "Fermi theory describes the decay rate as proportional to the fifth power of the endpoint energy (Sargent rule). Is the decay type alpha or beta?",
    "The CKM matrix element V_ud governs the quark-level transition responsible for this nuclear decay. Is the decay type alpha or beta?",
    "The continuous energy spectrum of the emitted lepton arises because energy and momentum are shared among three products. Is the decay type alpha or beta?",
    "Parity violation was first observed in this type of decay in the Wu experiment of 1957. Is the decay type alpha or beta?",
    "The Kurie (Fermi) plot of the emitted particle's momentum spectrum gives a straight line that extrapolates to the endpoint energy. Is the decay type alpha or beta?",
    "The allowed transition selection rules require ΔJ = 0 or 1 (no parity change) for Fermi and Gamow-Teller transitions. Is the decay type alpha or beta?",
    "The inverse process — neutrino capture on a proton — was used by Reines and Cowan to confirm the existence of the particle emitted in this decay. Is the decay type alpha or beta?",
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dataset() -> list[dict]:
    records = []

    # F0 alpha
    for i, (text, diff) in enumerate(zip(F0_ALPHA, F0_ALPHA_DIFF)):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F0", False, diff))

    # F0 beta
    for i, (text, diff) in enumerate(zip(F0_BETA, F0_BETA_DIFF)):
        records.append(make_prompt(text, "beta_decay", 1, i, "F0", False, diff))

    # F1 alpha
    for i, text in enumerate(F1_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F1", True))

    # F1 beta
    for i, text in enumerate(F1_BETA):
        records.append(make_prompt(text, "beta_decay", 1, i, "F1", True))

    # F2 alpha
    for i, text in enumerate(F2_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F2", False))

    # F2 beta
    for i, text in enumerate(F2_BETA):
        records.append(make_prompt(text, "beta_decay", 1, i, "F2", False))

    # F3 alpha
    for i, text in enumerate(F3_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F3", False, "indirect"))

    # F3 beta
    for i, text in enumerate(F3_BETA):
        records.append(make_prompt(text, "beta_decay", 1, i, "F3", False, "indirect"))

    # F4 alpha
    for i, text in enumerate(F4_ALPHA):
        records.append(make_prompt(text, "alpha_decay", 0, i, "F4", False, "hard"))

    # F4 beta
    for i, text in enumerate(F4_BETA):
        records.append(make_prompt(text, "beta_decay", 1, i, "F4", False, "hard"))

    return records


def sanity_check(records: list[dict]) -> None:
    n_alpha = sum(1 for r in records if r["physics_concept"] == "alpha_decay")
    n_beta  = sum(1 for r in records if r["physics_concept"] == "beta_decay")
    n_kf    = sum(1 for r in records if r["keyword_free"])
    fams    = {}
    for r in records:
        fams[r["surface_family"]] = fams.get(r["surface_family"], 0) + 1

    print(f"Total prompts  : {len(records)}")
    print(f"  alpha_decay  : {n_alpha}")
    print(f"  beta_decay   : {n_beta}")
    print(f"  keyword_free : {n_kf}")
    print(f"  by family    : {dict(sorted(fams.items()))}")

    # Verify no alpha/beta/helium/electron in the description part of keyword-free prompts.
    # The question tail "Is the decay type alpha or beta?" always contains the keywords,
    # so we only check the text before that suffix.
    kw_forbidden = ["alpha", "beta", "helium", "electron"]
    tail = "is the decay type alpha or beta?"
    violations = []
    for r in records:
        if r["keyword_free"]:
            # Strip the shared question tail before checking
            description = r["prompt"].lower()
            idx = description.find(tail)
            if idx >= 0:
                description = description[:idx]
            for kw in kw_forbidden:
                if kw in description:
                    violations.append((r["surface_family"], r["physics_concept"],
                                   r["template_idx"], kw, r["prompt"]))
    if violations:
        print("\nWARNING — keyword_free violations:")
        for v in violations:
            print(f"  [{v[0]} t{v[2]} {v[1]}] found '{v[3]}': {v[4]}")
    else:
        print("  keyword_free check: OK (no forbidden keywords in F1 prompts)")


def main() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Archive pilot if it exists and is the 24-prompt version
    if OUT_PATH.exists():
        lines = OUT_PATH.read_text().strip().splitlines()
        if len(lines) <= 30:  # pilot was 24 lines
            print(f"Archiving pilot ({len(lines)} prompts) → {PILOT_ARCHIVE}")
            shutil.copy(OUT_PATH, PILOT_ARCHIVE)
        else:
            print(f"Existing file has {len(lines)} lines — not archiving (not pilot).")

    records = build_dataset()
    sanity_check(records)

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} prompts → {OUT_PATH}")


if __name__ == "__main__":
    main()
