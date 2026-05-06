"""
Generate physics_internal_candidate_selection_mini (~150–200 prompts).

Design: NO explicit candidate lists in prompts.
The model must internally generate candidate states from context alone.
Anthropic-like: prompt → internal candidate activation → filter selects one → output.

Filters and targets:
  1-token: negative_charge_atomic→electron, positive_charge_atomic→proton,
           neutral_charge_atomic→neutron, massless_em_mediator→photon,
           beta_minus_charged_product→electron, em_quantum→photon,
           atomic_number_carrier→proton, nuclear_stability_provider→neutron
  2-token: antielectron→positron, heavy_negative_lepton→muon

Token audit:
  electron   OK (1-tok)
  proton     OK (1-tok)
  neutron    OK (1-tok)
  photon     OK (1-tok)
  positron   MULTI (2 toks) — will be skipped by script 07
  muon       MULTI (2 toks) — will be skipped by script 07
  neutrino   MULTI → EXCLUDED
  antineutrino MULTI (4 toks) → EXCLUDED
"""

import json
import random
from pathlib import Path

random.seed(42)

BEHAVIOUR = "physics_internal_candidate_selection_mini"
OUT_PATH = Path("data/prompts") / f"{BEHAVIOUR}_train.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

PRIO = {
    "electron":  [" positron", " muon",    " proton",   " neutron", " photon"],
    "photon":    [" electron", " muon",    " positron", " proton",  " neutron"],
    "neutron":   [" proton",   " positron"," electron", " muon",    " photon"],
    "positron":  [" electron", " muon",    " proton",   " neutron", " photon"],
    "muon":      [" positron", " electron"," proton",   " neutron", " photon"],
    "proton":    [" neutron",  " electron"," positron", " muon",    " photon"],
}

def s(p): return p.strip()

def pinc(pool, correct):
    wrong = [p for p in pool if p != correct]
    for pref in PRIO.get(s(correct), []):
        if pref in wrong:
            return pref
    return wrong[0]

def R(prompt, correct, pool, fp, wf, exp="core", exp_internal=None, multi_token=False):
    wrong = [p for p in pool if p != correct]
    inc   = pinc(pool, correct)
    return {
        "prompt": prompt,
        "correct_answer": correct,
        "incorrect_answer": inc,
        "incorrect_answers": wrong,
        "implicit_candidate_pool": [s(p) for p in pool],
        "expected_internal_candidates": exp_internal or [s(p) for p in pool],
        "target_candidate": s(correct),
        "filter_property": fp,
        "wording_family": wf,
        "candidate_type": "internal_particle_candidate",
        "explicit_candidate_set": False,
        "experiment_type": exp,
        "multi_token_answer": multi_token,
        "filter_correct_id": f"{fp}__{s(correct)}",
    }

# ── filter / target definitions ───────────────────────────────────────────────
# Pool = particles the model is expected to consider internally.
# incorrect_answer will be selected from pool via priority ordering.

CFGS = [

    # ── 1. negative_charge_atomic → electron ──────────────────────────────────
    dict(
        fp="negative_charge_atomic", correct=" electron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False,
        f1=[
            "Which particle in an atom has negative electric charge? Answer:",
            "What is the negatively charged particle that orbits the atomic nucleus? Answer:",
            "Which atomic particle carries a single unit of negative charge? Answer:",
            "Which particle found outside the nucleus is negatively charged? Answer:",
            "Which particle determines the chemical properties of an atom through its negative charge? Answer:",
            "Which subatomic particle has a charge of −1 elementary unit and orbits the nucleus? Answer:",
        ],
        f2=[
            "In atomic physics, atoms consist of a dense nucleus surrounded by a cloud of particles that carry negative charge. What is this orbiting particle? Answer:",
            "The electromagnetic interactions that govern chemistry involve a light particle with negative charge orbiting the nucleus. Which particle is this? Answer:",
            "Atoms contain a central nucleus surrounded by charged particles that determine ionic and covalent bonding; these particles each carry negative charge. What are they called? Answer:",
            "In the Bohr model of the atom, small negatively charged particles orbit the nucleus in quantized energy levels. Which particle is this? Answer:",
        ],
        f3=[
            "In an ionic bond, one atom transfers its outermost negatively charged particles to another. What are these particles? Answer:",
            "When an atom gains additional negatively charged particles, it becomes a negative ion. What particle is added? Answer:",
            "Electrostatic attraction between the nucleus and negatively charged orbiting particles holds the atom together. What are these orbiting particles? Answer:",
        ],
        f4=[
            "Unlike the proton, which particle in an atom carries negative electric charge? Answer:",
            "Unlike the neutron, which particle in the atom carries negative electric charge? Answer:",
            "Unlike the uncharged neutron, which atomic particle has a charge of −1? Answer:",
        ],
    ),

    # ── 2. positive_charge_atomic → proton ────────────────────────────────────
    dict(
        fp="positive_charge_atomic", correct=" proton",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False,
        f1=[
            "Which particle in the atomic nucleus has positive electric charge? Answer:",
            "Which positively charged particle is found in the nucleus of every atom? Answer:",
            "Which nuclear particle carries a single unit of positive charge? Answer:",
            "Which particle in the nucleus gives the atom its positive charge? Answer:",
            "Which nuclear particle has a charge of +1 elementary unit? Answer:",
            "Which particle, found in every atomic nucleus, is positively charged? Answer:",
        ],
        f2=[
            "The atomic nucleus contains two types of particles: one positively charged and one electrically neutral. Which particle carries the positive charge? Answer:",
            "In nuclear physics, the number of positively charged particles in the nucleus determines which chemical element it is. What is this particle? Answer:",
            "Nuclear charge is measured by counting the positively charged particles in the nucleus. What is the name of these particles? Answer:",
            "Rutherford's gold foil experiment revealed that atoms have a small, dense, positively charged core. The positive charge comes from which particle? Answer:",
        ],
        f3=[
            "In beta-minus decay, a neutron transforms into a positively charged nuclear particle while emitting an electron. What is the positively charged particle produced? Answer:",
            "When a nucleus undergoes beta-minus decay, its atomic number increases because a neutral particle becomes a positively charged one. What is this positively charged particle? Answer:",
            "Nuclear fusion in stars combines light nuclei by merging their positively charged particles. What are these positively charged nuclear particles? Answer:",
        ],
        f4=[
            "Unlike the neutron, which particle in the atomic nucleus carries electric charge (specifically positive charge)? Answer:",
            "Unlike the electron, which nuclear particle carries positive electric charge? Answer:",
            "Unlike the electrically neutral neutron, which nuclear particle has charge +1? Answer:",
        ],
    ),

    # ── 3. neutral_charge_atomic → neutron ────────────────────────────────────
    dict(
        fp="neutral_charge_atomic", correct=" neutron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False,
        f1=[
            "Which particle in the atomic nucleus has no electric charge? Answer:",
            "Which electrically neutral particle is found in the atomic nucleus? Answer:",
            "Which nuclear particle carries zero electric charge? Answer:",
            "Which particle in the nucleus contributes to atomic mass without contributing to atomic charge? Answer:",
            "Which uncharged particle, together with protons, makes up the atomic nucleus? Answer:",
            "Which nuclear particle is neither positive nor negative in charge? Answer:",
        ],
        f2=[
            "The atomic nucleus contains charged and uncharged particles. The uncharged particle contributes to the mass of the nucleus but not to its charge. What is this particle? Answer:",
            "In nuclear physics, two types of particles make up the nucleus: one positively charged, the other carrying no charge at all. What is the neutral one called? Answer:",
            "Isotopes of an element have the same number of protons but different numbers of a neutral nuclear particle. What is this neutral particle? Answer:",
            "Nuclear stability depends on the ratio of charged particles to neutral particles in the nucleus. Which neutral nuclear particle is being referred to? Answer:",
        ],
        f3=[
            "In beta-minus decay, a neutral nuclear particle transforms into a charged one. What was the original neutral particle before the decay? Answer:",
            "Isotopes of an element differ only in their number of neutral particles in the nucleus. What are these neutral particles? Answer:",
            "In nuclear fusion, the strong force binds both protons and neutral particles together. What is the neutral particle in the nucleus? Answer:",
        ],
        f4=[
            "Unlike the proton, which particle in the nucleus carries no electric charge? Answer:",
            "Unlike the electron, which nuclear particle has zero electric charge? Answer:",
            "Unlike the charged proton, which nuclear particle has zero net charge? Answer:",
        ],
    ),

    # ── 4. massless_em_mediator → photon ──────────────────────────────────────
    dict(
        fp="massless_em_mediator", correct=" photon",
        pool=[" photon", " electron", " proton", " neutron"],
        exp_internal=["photon", "electron", "proton"],
        multi_token=False,
        f1=[
            "Which particle is massless and mediates the electromagnetic force? Answer:",
            "Which particle carries electromagnetic energy and has zero rest mass? Answer:",
            "Which quantum of the electromagnetic field travels at the speed of light? Answer:",
            "Which fundamental particle mediates electromagnetic interactions between charged particles? Answer:",
            "Which massless boson is the force carrier of electromagnetism? Answer:",
        ],
        f2=[
            "The electromagnetic force between charged particles is mediated by an exchange particle that travels at the speed of light and has no rest mass. What is this particle? Answer:",
            "In quantum electrodynamics, the electromagnetic force is transmitted by a massless boson. What is this force-carrying particle? Answer:",
            "Light is composed of discrete packets of energy that have no rest mass and travel at c. Each such packet is a quantum of the electromagnetic field. What is it called? Answer:",
            "The Standard Model predicts that certain force-carrying particles are massless. The carrier of electromagnetism is one of these. What is it? Answer:",
        ],
        f3=[
            "When an electron transitions to a lower energy level in an atom, it releases energy in the form of a massless particle. What particle is emitted? Answer:",
            "In the photoelectric effect, electromagnetic energy is delivered to electrons in discrete units carried by massless particles. What are these particles? Answer:",
            "Pair production occurs when a high-energy massless particle converts into a particle-antiparticle pair near a nucleus. What is this massless particle? Answer:",
        ],
        f4=[
            "Unlike the electron, which particle carries electromagnetic energy but has zero rest mass? Answer:",
            "Unlike the proton, which force-carrying particle of electromagnetism has no rest mass? Answer:",
        ],
    ),

    # ── 5. beta_minus_charged_product → electron ──────────────────────────────
    dict(
        fp="beta_minus_charged_product", correct=" electron",
        pool=[" electron", " proton", " neutron", " positron"],
        exp_internal=["electron", "positron", "proton", "neutron"],
        multi_token=False,
        f1=[
            "In beta-minus decay, which charged particle is emitted from the nucleus? Answer:",
            "Which negatively charged particle is emitted during beta-minus radioactive decay? Answer:",
            "During beta-minus decay, a neutron transforms into a proton and emits which charged lepton? Answer:",
            "Which particle is the beta particle in beta-minus decay? Answer:",
            "In beta-minus radioactive decay, what charged particle is released? Answer:",
        ],
        f2=[
            "Beta-minus decay is a radioactive process in which a neutron converts to a proton, accompanied by the emission of a negatively charged lepton. What is this particle? Answer:",
            "In nuclear beta decay, the nucleus emits a fast-moving negatively charged particle — the same as the one that orbits atoms. What is it? Answer:",
            "Radioactive isotopes such as carbon-14 undergo beta-minus decay, emitting a negatively charged particle to stabilize the nucleus. What particle is emitted? Answer:",
            "During beta-minus decay, the atomic number of the nucleus increases by one, because a negatively charged lepton is emitted while a neutron becomes a proton. What is this emitted particle? Answer:",
        ],
        f3=[
            "A radioactive nucleus decays by emitting a negatively charged lepton and an antineutrino. Which charged particle is emitted? Answer:",
            "In the reaction n → p + lepton + antineutrino, which negatively charged lepton appears in this beta-minus decay equation? Answer:",
        ],
        f4=[
            "Unlike beta-plus decay which emits a positron, beta-minus decay emits which negatively charged particle? Answer:",
            "Unlike the positron emitted in beta-plus decay, which negatively charged lepton is emitted in beta-minus decay? Answer:",
        ],
    ),

    # ── 6. em_quantum → photon ────────────────────────────────────────────────
    dict(
        fp="em_quantum", correct=" photon",
        pool=[" photon", " electron", " proton"],
        exp_internal=["photon", "electron"],
        multi_token=False,
        f1=[
            "Which particle is the quantum of visible light? Answer:",
            "Which discrete unit of electromagnetic radiation carries energy proportional to its frequency? Answer:",
            "Which massless particle is absorbed when an atom jumps to a higher energy level? Answer:",
            "Which particle, emitted when electrons change energy levels, gives atoms their spectral lines? Answer:",
        ],
        f2=[
            "Atomic spectroscopy shows that atoms emit or absorb electromagnetic radiation in discrete packets. Each such packet is a quantum of light. What is this particle called? Answer:",
            "When an excited atom releases energy, it does so by emitting a particle whose energy equals the transition energy between two electron levels. What is this emitted particle? Answer:",
            "Solar cells work by absorbing packets of electromagnetic energy that then free electrons in a semiconductor. What are these packets of electromagnetic energy called? Answer:",
        ],
        f3=[
            "In stimulated emission, an incoming particle triggers an atom to release an identical particle, both having the same frequency. What is this particle? Answer:",
            "The photoelectric effect occurs when electromagnetic energy packets with sufficient energy eject electrons from a metal surface. What are these energy packets? Answer:",
        ],
        f4=[
            "Unlike the electron, which massless particle is emitted when an atom undergoes an energy transition? Answer:",
            "Unlike the proton, which particle of electromagnetic radiation has zero rest mass? Answer:",
        ],
    ),

    # ── 7. atomic_number_carrier → proton ─────────────────────────────────────
    dict(
        fp="atomic_number_carrier", correct=" proton",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["proton", "neutron", "electron"],
        multi_token=False,
        f1=[
            "Which particle defines an element's position in the periodic table? Answer:",
            "Which nuclear particle, when counted, gives the atomic number of an element? Answer:",
            "Which particle in the nucleus determines which chemical element the atom belongs to? Answer:",
            "Which particle, if its count changes in the nucleus, changes the element entirely? Answer:",
        ],
        f2=[
            "The periodic table is organized by atomic number, which counts a specific particle in the nucleus. Adding or removing this particle changes the element. What particle is being counted? Answer:",
            "In nuclear chemistry, adding one of these positively charged particles to a nucleus transforms it into the next element on the periodic table. What particle is this? Answer:",
        ],
        f3=[
            "In nuclear fusion reactions in stars, the number of positively charged nuclear particles in the product determines which element is formed. What particle is being counted? Answer:",
        ],
        f4=[
            "Unlike the neutron, which particle in the nucleus determines the chemical identity of the element? Answer:",
            "Unlike adding a neutron (which creates an isotope), adding which particle changes the element itself? Answer:",
        ],
    ),

    # ── 8. nuclear_stability_provider → neutron ───────────────────────────────
    dict(
        fp="nuclear_stability_provider", correct=" neutron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["proton", "neutron"],
        multi_token=False,
        f1=[
            "Which neutral nuclear particle helps stabilize the nucleus against proton-proton electromagnetic repulsion? Answer:",
            "Which particle, when added to a nucleus, can change isotopes without changing the element? Answer:",
            "Which nuclear particle differs between isotopes of the same element? Answer:",
            "Which uncharged nuclear particle contributes to nuclear binding through the strong force? Answer:",
        ],
        f2=[
            "Heavy atomic nuclei require a neutral particle alongside protons to counteract electromagnetic repulsion via the strong force. What is this neutral particle? Answer:",
            "Isotopes of an element differ in the number of neutral nuclear particles. These particles contribute to nuclear binding without changing the element's chemistry. What is this neutral particle? Answer:",
        ],
        f3=[
            "In a nucleus, the strong nuclear force acts between protons and neutral particles to provide stability. What are these neutral particles that help bind the nucleus? Answer:",
        ],
        f4=[
            "Unlike the proton, which neutral nuclear particle can be added to a nucleus without changing the element's chemical identity? Answer:",
            "Unlike protons, which neutral nuclear particles determine the mass number without affecting the atomic number? Answer:",
        ],
    ),

    # ── 9. antielectron → positron (MULTI-TOKEN, 2 toks) ─────────────────────
    dict(
        fp="antielectron", correct=" positron",
        pool=[" positron", " electron", " proton", " photon"],
        exp_internal=["positron", "electron", "proton"],
        multi_token=True,
        f1=[
            "Which particle is the antimatter counterpart of the electron? Answer:",
            "Which antiparticle has the same mass as an electron but opposite (positive) charge? Answer:",
            "Which positively charged particle has exactly the same mass as an electron? Answer:",
            "Which particle annihilates with an electron to produce two photons? Answer:",
        ],
        f2=[
            "Dirac's equations predicted the existence of a particle identical to the electron but with positive charge. What is this antiparticle? Answer:",
            "When a high-energy photon passes near a nucleus, it can produce a particle-antiparticle pair: an electron and its antimatter partner. What is this antimatter partner? Answer:",
            "PET scans work by detecting pairs of photons produced when electrons annihilate with their antiparticles. What is the antiparticle of the electron? Answer:",
        ],
        f3=[
            "In beta-plus decay, the nucleus emits an antiparticle of the electron. What is this antiparticle? Answer:",
            "Pair production creates two particles from a photon: an electron and its antimatter equivalent. What is the antimatter equivalent of an electron? Answer:",
        ],
        f4=[
            "Unlike the electron which has negative charge, which antimatter particle has positive charge but the same mass as an electron? Answer:",
            "Unlike the proton which is a baryon, which lighter antimatter particle has the same mass as an electron but opposite charge? Answer:",
        ],
    ),

    # ── 10. heavy_negative_lepton → muon (MULTI-TOKEN, 2 toks) ───────────────
    dict(
        fp="heavy_negative_lepton", correct=" muon",
        pool=[" muon", " electron", " proton", " neutron"],
        exp_internal=["muon", "electron"],
        multi_token=True,
        f1=[
            "Which particle is often called a heavy electron because it has the same charge but much greater mass? Answer:",
            "Which second-generation lepton has the same charge as an electron but is about 207 times heavier? Answer:",
            "Which unstable lepton is produced abundantly in cosmic ray showers in Earth's atmosphere? Answer:",
        ],
        f2=[
            "In particle physics, the second-generation charged lepton has the same charge as an electron but is approximately 207 times more massive. What is this particle? Answer:",
            "Cosmic rays produce a particle in the upper atmosphere that can penetrate deep into the Earth. It has the same charge as an electron but far greater mass. What is this particle? Answer:",
        ],
        f3=[
            "In the decay of a charged pion, a second-generation lepton is produced alongside a neutrino. This lepton has the same charge as an electron but greater mass. What is it? Answer:",
        ],
        f4=[
            "Unlike the electron, which heavier lepton of the same charge is produced in cosmic ray showers and has mass about 207 times greater? Answer:",
        ],
    ),
]

# ── build records ─────────────────────────────────────────────────────────────

def make_records(cfg):
    recs = []
    correct   = cfg["correct"]
    pool      = cfg["pool"]
    fp        = cfg["fp"]
    exp_int   = cfg["exp_internal"]
    mt        = cfg.get("multi_token", False)

    for prompt in cfg.get("f1", []):
        recs.append(R(prompt, correct, pool, fp, "F1_direct_implicit",
                      exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f2", []):
        recs.append(R(prompt, correct, pool, fp, "F2_contextual_implicit",
                      exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f3", []):
        recs.append(R(prompt, correct, pool, fp, "F3_process_implicit",
                      exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f4", []):
        recs.append(R(prompt, correct, pool, fp, "F4_contrast_implicit",
                      exp_internal=exp_int, multi_token=mt))
    return recs


all_records = []
for cfg in CFGS:
    all_records.extend(make_records(cfg))

# Dedup on (prompt, fp, target)
seen = set()
unique = []
for r in all_records:
    key = (r["prompt"], r["filter_property"], r["target_candidate"])
    if key not in seen:
        seen.add(key)
        unique.append(r)

random.shuffle(unique)

with open(OUT_PATH, "w") as f:
    for r in unique:
        f.write(json.dumps(r) + "\n")

# ── summary ───────────────────────────────────────────────────────────────────
from collections import Counter

print(f"Written {len(unique)} prompts to {OUT_PATH}")
print()
print("By filter_correct_id:")
for k, v in sorted(Counter(r["filter_correct_id"] for r in unique).items()):
    mt = "  [MULTI-TOKEN]" if any(r["multi_token_answer"] for r in unique if r["filter_correct_id"]==k) else ""
    print(f"  {k}: {v}{mt}")
print()
print("By wording_family:")
for k, v in sorted(Counter(r["wording_family"] for r in unique).items()):
    print(f"  {k}: {v}")
print()
mt_count = sum(1 for r in unique if r["multi_token_answer"])
print(f"Multi-token answer prompts: {mt_count}/{len(unique)} = {mt_count/len(unique):.1%}")
print()
# Implicit pool diversity
pools = Counter(tuple(r["implicit_candidate_pool"]) for r in unique)
print("Implicit candidate pools:")
for pool, count in sorted(pools.items(), key=lambda x: -x[1]):
    print(f"  {list(pool)}: {count}")
