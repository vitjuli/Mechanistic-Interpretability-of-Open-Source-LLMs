"""
Generate physics_particle_candidate_selection_v2 prompts (~220–240 total).

Cleaned v2 design:
  - Removed: boson, heaviest, positive_charge
  - Restricted: massless (no muon/positron distractors)
               neutral_charge (no photon distractor — photon is also neutral)
  - lightest: no photon (massless, lighter than electron) or positron (same mass)
  - Strong H2: distractor_sensitivity + set_size + F5_no_set preserved
"""

import json
import random
from pathlib import Path

random.seed(42)

BEHAVIOUR = "physics_particle_candidate_selection_v2"
OUT_PATH = Path("data/prompts") / f"{BEHAVIOUR}_train.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def s(p): return p.strip()


def tl(ps):
    """'the electron, the proton, and the neutron'"""
    ns = [s(p) for p in ps]
    if len(ns) == 2:
        return f"the {ns[0]} and the {ns[1]}"
    parts = [f"the {n}" for n in ns]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def ol(ps):
    """'electron, proton, or neutron'"""
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
# Each dict fully specifies one (filter, target) pair.
# Rules enforced in set design:
#   lepton: sets contain at most one lepton (no muon/positron as distractors for electron target)
#   neutral_charge: no photon (also neutral) as distractor
#   lightest: no photon (massless < electron) or positron (same mass as electron)
#   massless: only {photon, proton, neutron}, {photon, electron, proton}, {photon, electron, neutron}

CFGS = [
    # ── negative_charge → electron ──────────────────────────────────────────
    dict(
        fp="negative_charge", correct=" electron",
        qp="has negative electric charge",
        ctx="Electric charge determines how particles interact electromagnetically.",
        ff="Identify the negatively charged particle",
        no_set="Which particle has negative electric charge? Answer:",
        extended_ctx="The electron carries exactly one unit of negative charge, making it the lightest negatively charged particle.",
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
        sz2=[[" electron", " proton"], [" electron", " positron"]],
        sz4=[[" electron", " positron", " proton", " neutron"]],
    ),

    # ── lepton → electron ───────────────────────────────────────────────────
    # Only sets with electron as the sole lepton (no muon/positron as distractors)
    dict(
        fp="lepton", correct=" electron",
        qp="is a lepton",
        ctx="Leptons are fundamental particles that do not experience the strong nuclear force.",
        ff="Identify the lepton",
        no_set="Which particle is a lepton? Answer:",
        extended_ctx="The electron is the lightest lepton and carries lepton number +1.",
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
        sz2=[[" electron", " proton"], [" electron", " photon"]],
        sz4=[[" electron", " proton", " neutron", " photon"]],
    ),

    # ── lepton → muon ────────────────────────────────────────────────────────
    # Only sets with muon as the sole lepton (no electron/positron as distractors)
    # F5_no_set omitted: "which is a lepton?" is ambiguous (electron equally valid)
    dict(
        fp="lepton", correct=" muon",
        qp="is a lepton",
        ctx="Leptons are fundamental particles that do not experience the strong nuclear force.",
        ff="Identify the lepton",
        no_set=None,
        extended_ctx="The muon is a second-generation lepton, heavier than the electron but with the same charge.",
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
        sz2=[[" muon", " proton"], [" muon", " neutron"]],
        sz4=[[" muon", " proton", " neutron", " photon"]],
    ),

    # ── antimatter → positron ────────────────────────────────────────────────
    # Photon excluded (it is its own antiparticle — ambiguous for antimatter filter)
    dict(
        fp="antimatter", correct=" positron",
        qp="is antimatter (an antiparticle)",
        ctx="Antimatter consists of antiparticles, which annihilate with their matter counterparts.",
        ff="Identify the antiparticle",
        no_set="Which particle is an antiparticle? Answer:",
        extended_ctx="The positron, predicted by Dirac in 1928, is the antielectron — the first antiparticle discovered.",
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
        sz2=[[" positron", " electron"], [" positron", " proton"]],
        sz4=[[" positron", " electron", " proton", " neutron"]],
    ),

    # ── neutral_charge → neutron ─────────────────────────────────────────────
    # Photon excluded as distractor (photon is also electrically neutral)
    dict(
        fp="neutral_charge", correct=" neutron",
        qp="has no electric charge (is electrically neutral)",
        ctx="Some particles carry zero net electric charge despite containing charged constituents.",
        ff="Identify the electrically neutral particle",
        no_set="Which particle has no electric charge? Answer:",
        extended_ctx="The neutron has zero net charge but contains quarks with fractional charges that sum to zero.",
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
        sz2=[[" neutron", " proton"], [" neutron", " electron"]],
        sz4=[[" neutron", " proton", " electron", " positron"]],
    ),

    # ── lightest → electron ──────────────────────────────────────────────────
    # Photon excluded (massless, lighter than electron)
    # Positron excluded (same mass as electron — ambiguous)
    dict(
        fp="lightest", correct=" electron",
        qp="has the smallest mass",
        ctx="Particle masses span many orders of magnitude; the electron is far lighter than any hadron.",
        ff="Identify the particle with the smallest mass",
        no_set="Which particle has the smallest mass? Answer:",
        extended_ctx="The electron mass is 0.511 MeV/c², roughly 1836 times lighter than the proton.",
        core=[
            [" electron", " proton",  " neutron"],
            [" electron", " muon",    " proton"],
            [" electron", " muon",    " neutron"],
        ],
        dsens={
            "trivial": [" electron", " proton", " neutron"],
            "hard":    [" electron", " muon",   " proton"],
            "hardest": [" electron", " muon",   " neutron"],
        },
        sz2=[[" electron", " proton"], [" electron", " muon"]],
        sz4=[[" electron", " muon", " proton", " neutron"]],
    ),

    # ── massless → photon ────────────────────────────────────────────────────
    # Strictly restricted sets: no muon/positron (could confuse model)
    dict(
        fp="massless", correct=" photon",
        qp="is massless (has zero rest mass)",
        ctx="Gauge bosons that mediate long-range forces are predicted by the Standard Model to be massless.",
        ff="Identify the massless particle",
        no_set="Which particle is massless? Answer:",
        extended_ctx="The photon travels at the speed of light in vacuum because it has exactly zero rest mass.",
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
        sz2=[[" photon", " proton"], [" photon", " electron"]],
        sz4=[[" photon", " electron", " proton", " neutron"]],
    ),
]


# ── prompt generators ─────────────────────────────────────────────────────────

def gen_core(cfg):
    records = []
    fp, correct = cfg["fp"], cfg["correct"]
    qp, ctx, ff = cfg["qp"], cfg["ctx"], cfg["ff"]
    for cs in cfg["core"]:
        the_str = tl(cs)
        # F1a
        records.append(R(
            f"The options are {tl(cs)}. Which {qp}? Answer:",
            correct, cs, fp, "F1_explicit_list",
        ))
        # F1b
        records.append(R(
            f"Consider these particles: {tl(cs)}. Among them, which one {qp}? Answer:",
            correct, cs, fp, "F1_explicit_list",
        ))
        # F1c
        records.append(R(
            f"Given the particles {tl(cs)}, which {qp}? Answer:",
            correct, cs, fp, "F1_explicit_list",
        ))
        # F2
        records.append(R(
            f"{ctx} Among {tl(cs)}, which {qp}? Answer:",
            correct, cs, fp, "F2_contextual",
        ))
        # F3
        records.append(R(
            f"{ff} from the following: {tl(cs)}. Which is it? Answer:",
            correct, cs, fp, "F3_filter_first",
        ))
    return records


def gen_counterfactual(cfg):
    records = []
    fp, correct = cfg["fp"], cfg["correct"]
    qp, ctx = cfg["qp"], cfg["ctx"]
    ext = cfg["extended_ctx"]
    rep_set = cfg["core"][0]

    # F5_no_set
    if cfg["no_set"]:
        records.append(R(
            cfg["no_set"], correct, rep_set, fp, "F5_no_set",
            exp="counterfactual", vt="no_set",
        ))

    # F1_minimal
    records.append(R(
        f"{ol(rep_set)} — which {qp}? Answer:",
        correct, rep_set, fp, "F1_minimal",
        exp="counterfactual", vt="minimal",
    ))

    # F1_reordered (reverse the representative set)
    rev = list(reversed(rep_set))
    records.append(R(
        f"The options are {tl(rev)}. Which {qp}? Answer:",
        correct, rev, fp, "F1_reordered",
        exp="counterfactual", vt="reordered",
    ))

    # F1_extended
    records.append(R(
        f"{ext} Among {tl(rep_set)}, which {qp}? Answer:",
        correct, rep_set, fp, "F1_extended",
        exp="counterfactual", vt="extended",
    ))

    return records


def gen_distractor_sensitivity(cfg):
    records = []
    fp, correct = cfg["fp"], cfg["correct"]
    qp, ctx, ff = cfg["qp"], cfg["ctx"], cfg["ff"]
    for level, cs in cfg["dsens"].items():
        # F1
        records.append(R(
            f"The options are {tl(cs)}. Which {qp}? Answer:",
            correct, cs, fp, "F1_explicit_list",
            exp="distractor_sensitivity", diff=level,
        ))
        # F2
        records.append(R(
            f"{ctx} Among {tl(cs)}, which {qp}? Answer:",
            correct, cs, fp, "F2_contextual",
            exp="distractor_sensitivity", diff=level,
        ))
    return records


def gen_set_size(cfg):
    records = []
    fp, correct = cfg["fp"], cfg["correct"]
    qp, ctx = cfg["qp"], cfg["ctx"]

    for sz_sets in [cfg["sz2"], cfg["sz4"]]:
        cs = sz_sets[0]  # use first set of each size
        records.append(R(
            f"The options are {tl(cs)}. Which {qp}? Answer:",
            correct, cs, fp, "F1_explicit_list",
            exp="set_size",
        ))
        records.append(R(
            f"{ctx} Among {tl(cs)}, which {qp}? Answer:",
            correct, cs, fp, "F2_contextual",
            exp="set_size",
        ))
    return records


# ── build dataset ─────────────────────────────────────────────────────────────

all_records = []
for cfg in CFGS:
    all_records.extend(gen_core(cfg))
    all_records.extend(gen_counterfactual(cfg))
    all_records.extend(gen_distractor_sensitivity(cfg))
    all_records.extend(gen_set_size(cfg))

# Deduplicate on prompt text (edge case: some dsens sets equal core sets)
seen = set()
unique = []
for r in all_records:
    # Include experiment_type so distractor_sensitivity / set_size records with
    # the same prompt text as core records are preserved as distinct entries.
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

exp_counts = Counter(r["experiment_type"] for r in unique)
print("By experiment_type:")
for k, v in sorted(exp_counts.items()):
    print(f"  {k}: {v}")
print()

fc_counts = Counter(r["filter_correct_id"] for r in unique)
print("By filter_correct_id:")
for k, v in sorted(fc_counts.items()):
    print(f"  {k}: {v}")
print()

wf_counts = Counter(r["wording_family"] for r in unique)
print("By wording_family:")
for k, v in sorted(wf_counts.items()):
    print(f"  {k}: {v}")
