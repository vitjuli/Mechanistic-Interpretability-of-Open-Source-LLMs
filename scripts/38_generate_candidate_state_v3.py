"""
Generate physics_particle_candidate_selection_v3 (~500 prompts).

Scaled from v2 (224 → ~505):
  - 14 core families per set (7 F1 + 3 F2 + 4 F3)
  - More candidate set variety where valid
  - Expanded distractor sensitivity (4 families × all 7 filter/target pairs)
  - Set-size variants for strong filters only (negative_charge, lepton)
  - Multi-token targets (positron, muon) capped at ~30-40%
"""

import json
import random
from pathlib import Path

random.seed(42)

BEHAVIOUR = "physics_particle_candidate_selection_v3"
OUT_PATH = Path("data/prompts") / f"{BEHAVIOUR}_train.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def s(p): return p.strip()


def tl(ps):
    ns = [s(p) for p in ps]
    if len(ns) == 2:
        return f"the {ns[0]} and the {ns[1]}"
    parts = [f"the {n}" for n in ns]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def ol(ps):
    ns = [s(p) for p in ps]
    if len(ns) == 2:
        return f"{ns[0]} or {ns[1]}"
    return ", ".join(ns[:-1]) + f", or {ns[-1]}"


PRIO = {
    "electron":  [" positron", " muon",    " proton",   " neutron", " photon"],
    "photon":    [" electron", " muon",    " positron", " proton",  " neutron"],
    "neutron":   [" proton",   " positron"," electron", " muon",    " photon"],
    "positron":  [" electron", " muon",    " proton",   " neutron", " photon"],
    "muon":      [" positron", " electron"," proton",   " neutron", " photon"],
}


def pinc(cs, c):
    wrong = [p for p in cs if p != c]
    for pref in PRIO.get(s(c), []):
        if pref in wrong:
            return pref
    return wrong[0]


def R(prompt, correct, cs, fp, wf, exp="core", diff=None, vt=None):
    return {
        "prompt": prompt,
        "correct_answer": correct,
        "incorrect_answer": pinc(cs, correct),
        "incorrect_answers": [p for p in cs if p != correct],
        "candidate_set": list(cs),
        "filter_property": fp,
        "wording_family": wf,
        "target_candidate": s(correct),
        "candidate_type": "particle",
        "experiment_type": exp,
        "n_candidates": len(cs),
        "filter_correct_id": f"{fp}__{s(correct)}",
        **({"distractor_difficulty": diff} if diff else {}),
        **({"variant_type": vt} if vt else {}),
    }


# ── filter configurations ─────────────────────────────────────────────────────

CFGS = [
    # ── negative_charge → electron ──────────────────────────────────────────
    # muon excluded as distractor (also negatively charged)
    dict(
        fp="negative_charge", correct=" electron",
        qp="has negative electric charge",
        ctx="Electric charge determines how particles interact electromagnetically.",
        ctx2="Charged particles respond to electric fields; the electron carries exactly one unit of negative charge.",
        ff="Identify the negatively charged particle",
        no_set="Which particle has negative electric charge? Answer:",
        ext="The electron carries exactly one unit of negative charge, making it the lightest negatively charged particle.",
        core=[
            [" electron", " proton",   " neutron"],
            [" electron", " proton",   " photon"],
            [" electron", " positron", " proton"],
            [" electron", " positron", " neutron"],
            [" electron", " positron", " photon"],
        ],
        dsens={
            "trivial": [" electron", " photon",   " neutron"],
            "easy":    [" electron", " proton",   " neutron"],
            "hard":    [" electron", " positron", " proton"],
            "hardest": [" electron", " positron", " neutron"],
        },
        sz={
            2: [[" electron", " proton"], [" electron", " positron"]],
            4: [[" electron", " positron", " proton", " neutron"]],
            5: [[" electron", " positron", " proton", " neutron", " photon"]],
        },
    ),

    # ── lepton → electron ───────────────────────────────────────────────────
    # muon/positron excluded as distractors (both are leptons)
    dict(
        fp="lepton", correct=" electron",
        qp="is a lepton",
        ctx="Leptons are fundamental particles that do not experience the strong nuclear force.",
        ctx2="The lepton family includes the electron and muon; they are not composed of quarks.",
        ff="Identify the lepton",
        no_set="Which particle is a lepton? Answer:",
        ext="The electron is the lightest lepton and carries lepton number +1.",
        core=[
            [" electron", " proton",  " neutron"],
            [" electron", " proton",  " photon"],
            [" electron", " neutron", " photon"],
        ],
        dsens={
            "easy":   [" electron", " proton",  " neutron"],
            "medium": [" electron", " proton",  " photon"],
            "hard":   [" electron", " neutron", " photon"],
        },
        sz={
            2: [[" electron", " proton"], [" electron", " photon"]],
            4: [[" electron", " proton", " neutron", " photon"]],
        },
    ),

    # ── lepton → muon ────────────────────────────────────────────────────────
    # electron/positron excluded as distractors (also leptons)
    dict(
        fp="lepton", correct=" muon",
        qp="is a lepton",
        ctx="Leptons are fundamental particles that do not experience the strong nuclear force.",
        ctx2="The muon is a second-generation lepton, heavier than the electron but sharing its charge and lepton properties.",
        ff="Identify the lepton",
        no_set=None,  # ambiguous without set (electron equally valid)
        ext="The muon is an unstable second-generation lepton with mass about 207 times that of the electron.",
        core=[
            [" muon", " proton",  " neutron"],
            [" muon", " proton",  " photon"],
            [" muon", " neutron", " photon"],
        ],
        dsens={
            "easy":   [" muon", " proton",  " neutron"],
            "medium": [" muon", " proton",  " photon"],
            "hard":   [" muon", " neutron", " photon"],
        },
        sz={
            2: [[" muon", " proton"], [" muon", " neutron"]],
            4: [[" muon", " proton", " neutron", " photon"]],
        },
    ),

    # ── antimatter → positron ────────────────────────────────────────────────
    # photon excluded (its own antiparticle — ambiguous)
    dict(
        fp="antimatter", correct=" positron",
        qp="is antimatter (an antiparticle)",
        ctx="Antimatter consists of antiparticles that annihilate with their matter counterparts.",
        ctx2="Every matter particle has a corresponding antiparticle; the positron is the antielectron.",
        ff="Identify the antiparticle",
        no_set="Which particle is an antiparticle? Answer:",
        ext="The positron, predicted by Dirac in 1928, is the antiparticle of the electron with positive charge.",
        core=[
            [" positron", " electron", " proton"],
            [" positron", " electron", " neutron"],
            [" positron", " proton",   " neutron"],
            [" positron", " muon",     " proton"],
            [" positron", " muon",     " electron"],
        ],
        dsens={
            "easy":   [" positron", " proton",   " neutron"],
            "medium": [" positron", " electron", " proton"],
            "hard":   [" positron", " muon",     " electron"],
        },
        sz={},  # not a strong filter for set-size analysis
    ),

    # ── neutral_charge → neutron ─────────────────────────────────────────────
    # photon excluded (also electrically neutral — violates unique_satisfier)
    dict(
        fp="neutral_charge", correct=" neutron",
        qp="has no electric charge (is electrically neutral)",
        ctx="Some particles carry zero net electric charge despite containing charged constituents.",
        ctx2="Electric charge is quantized; a particle is electrically neutral if its net charge is exactly zero.",
        ff="Identify the electrically neutral particle",
        no_set="Which particle has no electric charge? Answer:",
        ext="The neutron has zero net electric charge but contains quarks with fractional charges that cancel out.",
        core=[
            [" neutron", " proton",   " electron"],
            [" neutron", " proton",   " positron"],
            [" neutron", " electron", " positron"],
        ],
        dsens={
            "easy":   [" neutron", " proton",   " electron"],
            "medium": [" neutron", " proton",   " positron"],
            "hard":   [" neutron", " electron", " positron"],
        },
        sz={},
    ),

    # ── lightest → electron ──────────────────────────────────────────────────
    # photon excluded (massless, lighter than electron)
    # positron excluded (same mass as electron — ambiguous)
    dict(
        fp="lightest", correct=" electron",
        qp="has the smallest mass",
        ctx="Particle masses span many orders of magnitude; the electron is far lighter than any hadron.",
        ctx2="The electron is the lightest stable massive particle, roughly 1836 times lighter than the proton.",
        ff="Identify the particle with the smallest mass",
        no_set="Which particle has the smallest mass? Answer:",
        ext="The electron mass is 0.511 MeV/c², roughly 1836 times lighter than the proton.",
        core=[
            [" electron", " proton",  " neutron"],
            [" electron", " muon",    " proton"],
            [" electron", " muon",    " neutron"],
        ],
        dsens={
            "trivial": [" electron", " proton",  " neutron"],
            "hard":    [" electron", " muon",    " proton"],
            "hardest": [" electron", " muon",    " neutron"],
        },
        sz={},
    ),

    # ── massless → photon ────────────────────────────────────────────────────
    # only allowed sets: {photon,proton,neutron}, {photon,electron,proton}, {photon,electron,neutron}
    dict(
        fp="massless", correct=" photon",
        qp="is massless (has zero rest mass)",
        ctx="The Standard Model predicts that the photon has exactly zero rest mass.",
        ctx2="Massless particles travel at the speed of light and carry energy without rest mass.",
        ff="Identify the massless particle",
        no_set="Which particle is massless? Answer:",
        ext="The photon carries the electromagnetic force and travels at c with exactly zero rest mass.",
        core=[
            [" photon", " proton",   " neutron"],
            [" photon", " electron", " proton"],
            [" photon", " electron", " neutron"],
        ],
        dsens={
            "easy":   [" photon", " proton",   " neutron"],
            "medium": [" photon", " electron", " proton"],
            "hard":   [" photon", " electron", " neutron"],
        },
        sz={},
    ),
]


# ── prompt generation ─────────────────────────────────────────────────────────

def make_core_prompts(cfg, cs):
    """14 families: 7×F1 + 3×F2 + 4×F3."""
    fp, correct = cfg["fp"], cfg["correct"]
    qp  = cfg["qp"]
    ctx = cfg["ctx"]
    cx2 = cfg["ctx2"]
    ff  = cfg["ff"]
    TL  = tl(cs)
    OL  = ol(cs)

    recs = []

    def add(prompt, wf):
        recs.append(R(prompt, correct, cs, fp, wf))

    # F1 — 7 variants
    add(f"The options are {TL}. Which {qp}? Answer:", "F1_explicit_list")
    add(f"Consider these particles: {TL}. Among them, which one {qp}? Answer:", "F1_explicit_list")
    add(f"Given the particles {TL}, which {qp}? Answer:", "F1_explicit_list")
    add(f"Which of {OL} {qp}? Answer:", "F1_explicit_list")
    add(f"From the following particles — {TL} — which {qp}? Answer:", "F1_explicit_list")
    add(f"A physicist considers these particles: {TL}. Which one {qp}? Answer:", "F1_explicit_list")
    add(f"These particles are present: {TL}. Which {qp}? Answer:", "F1_explicit_list")

    # F2 — 3 variants
    add(f"{ctx} Among {TL}, which {qp}? Answer:", "F2_contextual")
    add(f"{ctx} Of the particles {TL}, which {qp}? Answer:", "F2_contextual")
    add(f"{cx2} From {TL}, select the one that {qp}. Answer:", "F2_contextual")

    # F3 — 4 variants  (target ~57% of F1 count = 4/7)
    add(f"{ff} from the following: {TL}. Which is it? Answer:", "F3_filter_first")
    add(f"Identify the particle that {qp} from these candidates: {TL}. Answer:", "F3_filter_first")
    add(f"Which particle {qp}? Options: {TL}. Answer:", "F3_filter_first")
    add(f"{ff}. The candidate particles are: {TL}. Answer:", "F3_filter_first")

    return recs


def make_sub_prompts(cfg, cs, exp, diff=None, vt=None):
    """4 families for distractor_sensitivity / set_size / counterfactual variants."""
    fp, correct = cfg["fp"], cfg["correct"]
    qp  = cfg["qp"]
    ctx = cfg["ctx"]
    cx2 = cfg["ctx2"]
    TL  = tl(cs)

    recs = []

    def add(prompt, wf):
        recs.append(R(prompt, correct, cs, fp, wf, exp=exp, diff=diff, vt=vt))

    add(f"The options are {TL}. Which {qp}? Answer:", "F1_explicit_list")
    add(f"Consider these particles: {TL}. Among them, which one {qp}? Answer:", "F1_explicit_list")
    add(f"{ctx} Among {TL}, which {qp}? Answer:", "F2_contextual")
    add(f"{cx2} From {TL}, select the one that {qp}. Answer:", "F2_contextual")

    return recs


def gen_counterfactual(cfg):
    fp, correct = cfg["fp"], cfg["correct"]
    qp  = cfg["qp"]
    ctx = cfg["ctx"]
    ext = cfg["ext"]
    rep = cfg["core"][0]
    TL  = tl(rep)
    OL  = ol(rep)
    rev = list(reversed(rep))

    recs = []

    if cfg["no_set"]:
        recs.append(R(cfg["no_set"], correct, rep, fp, "F5_no_set",
                      exp="counterfactual", vt="no_set"))

    recs.append(R(f"{OL} — which {qp}? Answer:", correct, rep, fp, "F1_minimal",
                  exp="counterfactual", vt="minimal"))

    recs.append(R(f"The options are {tl(rev)}. Which {qp}? Answer:", correct, rev, fp,
                  "F1_reordered", exp="counterfactual", vt="reordered"))

    recs.append(R(f"{ext} Among {TL}, which {qp}? Answer:", correct, rep, fp,
                  "F1_extended", exp="counterfactual", vt="extended"))

    return recs


# ── build dataset ─────────────────────────────────────────────────────────────

all_records = []

for cfg in CFGS:
    # Core
    for cs in cfg["core"]:
        all_records.extend(make_core_prompts(cfg, cs))

    # Counterfactual
    all_records.extend(gen_counterfactual(cfg))

    # Distractor sensitivity
    for level, cs in cfg["dsens"].items():
        all_records.extend(make_sub_prompts(cfg, cs, "distractor_sensitivity", diff=level))

    # Set-size variants
    for size, sets in cfg["sz"].items():
        for cs in sets:
            all_records.extend(make_sub_prompts(cfg, cs, "set_size"))

# Deduplicate: same (prompt, filter, target, experiment_type)
seen = set()
unique = []
for r in all_records:
    key = (r["prompt"], r["filter_property"], r["target_candidate"], r["experiment_type"])
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

print("By experiment_type:")
for k, v in sorted(Counter(r["experiment_type"] for r in unique).items()):
    print(f"  {k}: {v}")
print()

print("By filter_correct_id:")
for k, v in sorted(Counter(r["filter_correct_id"] for r in unique).items()):
    print(f"  {k}: {v}")
print()

print("By wording_family:")
for k, v in sorted(Counter(r["wording_family"] for r in unique).items()):
    print(f"  {k}: {v}")
print()

multi_token = [r for r in unique if r["target_candidate"] in ("positron", "muon")]
print(f"Multi-token targets (positron+muon): {len(multi_token)}/{len(unique)} = {len(multi_token)/len(unique):.1%}")

f1_count = sum(1 for r in unique if r["wording_family"] == "F1_explicit_list")
f3_count = sum(1 for r in unique if r["wording_family"] == "F3_filter_first")
print(f"F3/F1 ratio: {f3_count}/{f1_count} = {f3_count/f1_count:.1%}")
