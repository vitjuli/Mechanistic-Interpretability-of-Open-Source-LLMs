"""
Generate large candidate-state dataset (~400 prompts) for mechanistic analysis.

Structure:
  Core:                 31 base (set, filter) pairs × 4 families = 124 prompts
  Distractor sensitivity: same (filter, correct) with varying distractor sets = ~60 prompts
  Counterfactual:       with / without explicit candidate list = ~60 prompts
  Set-size variation:   2 to 5 candidates, same filter = ~45 prompts

Key metadata for analysis:
  experiment_type:      core | distractor_sensitivity | counterfactual | set_size
  base_case_id:         stable ID for matching variants of same (filter, correct)
  distractor_difficulty: easy | medium | hard | hardest (for distractor_sensitivity)
  variant_type:         original | no_set | reordered | extended | minimal (counterfactual)
  n_candidates:         2 | 3 | 4 | 5

H1 vs H2 test design:
  H1 (direct retrieval):     logprob_diff(correct - incorrect) constant across different
                              candidate sets with the same filter + correct answer
  H2 (set-mediated):         logprob_diff changes as distractor set composition changes

The counterfactual "no_set" variant (F5_no_set family) is the sharpest test:
  F5: "Which particle {filter_q}? Answer:"  (no candidate list given)
  If diff(original) >> diff(no_set) → model uses the explicit set
  If diff(original) ≈ diff(no_set)  → model retrieves directly, ignoring the set

Pilot neutron fix:
  The pilot decay-product prompts incorrectly labeled neutron as a "charged product".
  The large dataset avoids this by using precise language about nuclear vs emitted particles.

Token status (confirmed from CSD3):
  ' electron': 1 tok  ' proton': 1 tok  ' neutron': 1 tok
  ' photon':   1 tok  ' positron': 2 tok (length-normalized)
  ' muon':     1 tok (expected; confirm in audit)
"""

import json
import hashlib
from itertools import combinations
from pathlib import Path

OUT_DIR = Path("data/prompts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Particle database ────────────────────────────────────────────────────────
# Properties: charge, family, mass_MeV, spin_x2 (2×spin), is_antimatter, is_composite
_P = {
    'electron':  (-1, 'lepton',  0.511,   1, False, False),
    'proton':    (+1, 'baryon',  938.3,   1, False, True),
    'neutron':   ( 0, 'baryon',  939.6,   1, False, True),
    'photon':    ( 0, 'boson',   0.0,     2, False, False),
    'positron':  (+1, 'lepton',  0.511,   1, True,  False),
    'muon':      (-1, 'lepton',  105.7,   1, False, False),
}

def charge(p):   return _P[p][0]
def family(p):   return _P[p][1]
def mass(p):     return _P[p][2]
def spin_x2(p):  return _P[p][3]
def is_anti(p):  return _P[p][4]
def is_comp(p):  return _P[p][5]


# ─── Filter definitions ───────────────────────────────────────────────────────
FILTERS = {
    'negative_charge': {
        'question': 'has negative electric charge',
        'pred': lambda p, cs: charge(p) < 0,
    },
    'positive_charge': {
        'question': 'has positive electric charge',
        'pred': lambda p, cs: charge(p) > 0,
    },
    'neutral_charge': {
        'question': 'has zero (neutral) electric charge',
        'pred': lambda p, cs: charge(p) == 0,
    },
    'lepton': {
        'question': 'is a lepton (a member of the lepton family)',
        'pred': lambda p, cs: family(p) == 'lepton',
    },
    'baryon': {
        'question': 'is a baryon',
        'pred': lambda p, cs: family(p) == 'baryon',
    },
    'boson': {
        'question': 'is a boson (has integer spin and obeys Bose-Einstein statistics)',
        'pred': lambda p, cs: spin_x2(p) % 2 == 0,
    },
    'massless': {
        'question': 'has zero rest mass',
        'pred': lambda p, cs: mass(p) == 0.0,
    },
    'antimatter': {
        'question': 'is an antiparticle (antimatter counterpart of ordinary matter)',
        'pred': lambda p, cs: is_anti(p),
    },
    'heaviest': {
        'question': 'has the largest rest mass',
        'pred': lambda p, cs: p == max(cs, key=mass),
    },
    'lightest': {
        'question': 'has the smallest rest mass (or zero)',
        'pred': lambda p, cs: p == min(cs, key=mass),
    },
}


def unique_satisfier(cs, fp):
    """Return (correct, others) if exactly one particle in cs satisfies filter fp, else None."""
    q = [p for p in cs if FILTERS[fp]['pred'](p, cs)]
    if len(q) == 1:
        correct = q[0]
        others = [p for p in cs if p != correct]
        return correct, others
    return None


def hardest_incorrect(cs, correct, fp):
    """
    Choose the hardest distractor: the non-correct particle most similar to correct.
    Similarity heuristics (in order of priority):
      1. Same sign of charge
      2. Same particle family
      3. Similar mass (within 100×)
    """
    others = [p for p in cs if p != correct]
    def similarity(p):
        score = 0
        if (charge(p) > 0) == (charge(correct) > 0) and charge(p) != 0 and charge(correct) != 0:
            score += 3  # same sign of charge
        if charge(p) != 0 and charge(correct) != 0:
            score += 1  # both charged (vs one neutral)
        if family(p) == family(correct):
            score += 2  # same particle family
        if mass(correct) > 0 and 0.01 < mass(p) / mass(correct) < 100:
            score += 1  # similar mass order of magnitude
        return score
    return max(others, key=similarity)


def base_case_id(cs, fp):
    """Stable short ID for a (candidate_set, filter) pair."""
    key = fp + "__" + "_".join(sorted(cs))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def filter_correct_id(fp, correct):
    """ID for matching all variants of same (filter, correct_answer) pair."""
    return f"{fp}__{correct}"


# ─── Physics context strings ─────────────────────────────────────────────────
# Used in F4_physics_context family.  Key = frozenset of candidate names.
CONTEXTS = {
    frozenset({'electron','proton','neutron'}):
        "An atom consists of a nucleus containing protons and neutrons, with electrons orbiting outside",
    frozenset({'positron','proton','neutron'}):
        "In nuclear physics, positrons can be emitted alongside protons and neutrons in radioactive processes",
    frozenset({'muon','proton','neutron'}):
        "In cosmic-ray interactions, muons are produced alongside protons and neutrons from nuclear targets",
    frozenset({'electron','proton','photon'}):
        "In atomic transitions, electrons, protons, and photons are the key players in absorption and emission",
    frozenset({'electron','neutron','photon'}):
        "In neutron scattering experiments, the incident neutron, scattered electrons, and emitted photons are detected",
    frozenset({'electron','positron','photon'}):
        "In pair production and annihilation, electrons, positrons, and photons interconvert",
    frozenset({'muon','electron','photon'}):
        "In quantum electrodynamics, muons and electrons both couple to the photon field",
    frozenset({'muon','positron','neutron'}):
        "In muon-catalyzed fusion research, muons, positrons, and neutrons play distinct roles",
    frozenset({'electron','muon','proton'}):
        "The electron, muon, and proton are among the most stable particles studied in accelerator physics",
    frozenset({'proton','neutron','photon'}):
        "In nuclear gamma transitions, a nucleus may emit a photon while the proton and neutron numbers are conserved",
    frozenset({'electron','positron','proton'}):
        "In high-energy collisions, electrons, positrons, and protons may all appear as beam or target particles",
}

def get_context(cs):
    key = frozenset(cs)
    return CONTEXTS.get(key, f"Three particles are present: the {cs[0]}, the {cs[1]}, and the {cs[2]}")


# ─── Prompt family templates ──────────────────────────────────────────────────
# Candidates are reordered across families for positional balance.
# family_idx 0→3 correspond to F1_explicit_list, F2_contextual, F3_filter_first, F4_physics_context

def ordered(cs, correct, fi):
    """Return (p1, p2, p3) with correct at different positions depending on family index."""
    others = [c for c in cs if c != correct]
    assert len(others) == len(cs) - 1
    if fi == 0:   # F1: correct last
        return others + [correct]
    elif fi == 1:  # F2: correct first
        return [correct] + others
    elif fi == 2:  # F3: correct middle (if 3-candidate)
        if len(cs) == 3:
            return [others[0], correct, others[1]]
        else:
            return [correct] + others  # fallback for other sizes
    else:          # F4: original order
        return list(cs)


def make_candidate_phrase(parts, connector="or"):
    """'a, b, and c' or 'a or b' etc."""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"the {parts[0]} {connector} the {parts[1]}"
    inner = ", ".join(f"the {p}" for p in parts[:-1])
    return f"{inner}, {connector} the {parts[-1]}"


def make_options_phrase(parts):
    """'proton, neutron, and electron'"""
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


FAMILY_NAMES = ["F1_explicit_list", "F2_contextual", "F3_filter_first", "F4_physics_context",
                "F5_no_set"]


def make_core_prompt(cs, fp, fi):
    """Generate one prompt for a (candidate_set, filter, family_index)."""
    ord_cs = ordered(cs, unique_satisfier(cs, fp)[0], fi)
    fq = FILTERS[fp]['question']

    if fi == 0:  # F1: explicit list
        opts = make_options_phrase(ord_cs)
        prompt = f"The options are {opts}. Which particle {fq}? Answer:"

    elif fi == 1:  # F2: contextual
        opts = ", ".join(f"the {p}" for p in ord_cs)
        prompt = f"Consider these particles: {opts}. Among them, which one {fq}? Answer:"

    elif fi == 2:  # F3: filter-first
        q_phrase = make_candidate_phrase(ord_cs)
        prompt = f"Which particle {fq} — {q_phrase}? Answer:"

    elif fi == 3:  # F4: physics context
        context = get_context(cs)
        opts = make_options_phrase(ord_cs)
        prompt = f"{context}. Among the {opts}, which one {fq}? Answer:"

    else:  # F5: no explicit candidate set (counterfactual ablation)
        prompt = make_no_set_prompt(fp)

    return prompt


def make_no_set_prompt(fp):
    """Counterfactual: same filter question but no candidate list given."""
    fq = FILTERS[fp]['question']
    return f"Which particle {fq}? Answer:"


def make_reordered_prompt(cs, fp, original_cs):
    """Reverse the order of candidates vs F1."""
    fq = FILTERS[fp]['question']
    reversed_cs = list(reversed(original_cs))
    opts = make_options_phrase(reversed_cs)
    return f"The options are {opts}. Which particle {fq}? Answer:"


def make_extended_prompt(cs, extra, fp):
    """Add one extra neutral/easy candidate to the set."""
    fq = FILTERS[fp]['question']
    ext_cs = list(cs) + [extra]
    opts = make_options_phrase(ext_cs)
    return f"The options are {opts}. Among these {len(ext_cs)} particles, which one {fq}? Answer:"


def make_minimal_prompt(correct, incorrect, fp):
    """Minimal: only correct + hardest distractor."""
    fq = FILTERS[fp]['question']
    return (f"The options are {correct} and {incorrect}. "
            f"Which of these two particles {fq}? Answer:")


# ─── Build records ────────────────────────────────────────────────────────────

def make_record(prompt, cs, fp, correct, incorrect, extra_meta=None):
    wrong_all = [p for p in cs if p != correct]
    r = {
        "prompt": prompt,
        "correct_answer":   f" {correct}",
        "incorrect_answer": f" {incorrect}",
        "incorrect_answers": [f" {p}" for p in wrong_all],
        "candidate_set": list(cs),
        "candidate_set_str": "|".join(sorted(cs)),
        "filter_property": fp,
        "target_candidate": correct,
        "n_candidates": len(cs),
        "base_case_id": base_case_id(cs, fp),
        "filter_correct_id": filter_correct_id(fp, correct),
        "candidate_type": "particle",
    }
    if extra_meta:
        r.update(extra_meta)
    return r


# ─── Core dataset (31 valid base cases × 4 families) ─────────────────────────

BASE_CASES = [
    # (candidate_set, filter_property)
    # ── {electron, proton, neutron}
    (('electron','proton','neutron'),  'negative_charge'),
    (('electron','proton','neutron'),  'positive_charge'),
    (('electron','proton','neutron'),  'neutral_charge'),
    (('electron','proton','neutron'),  'lepton'),
    (('electron','proton','neutron'),  'heaviest'),
    (('electron','proton','neutron'),  'lightest'),
    # ── {positron, proton, neutron}
    (('positron','proton','neutron'),  'neutral_charge'),
    (('positron','proton','neutron'),  'lepton'),
    (('positron','proton','neutron'),  'antimatter'),
    # ── {muon, proton, neutron}
    (('muon','proton','neutron'),      'negative_charge'),
    (('muon','proton','neutron'),      'lepton'),
    (('muon','proton','neutron'),      'neutral_charge'),
    (('muon','proton','neutron'),      'heaviest'),
    # ── {electron, proton, photon}
    (('electron','proton','photon'),   'negative_charge'),
    (('electron','proton','photon'),   'positive_charge'),
    (('electron','proton','photon'),   'neutral_charge'),
    (('electron','proton','photon'),   'boson'),
    (('electron','proton','photon'),   'massless'),
    (('electron','proton','photon'),   'lightest'),
    # ── {electron, neutron, photon}
    (('electron','neutron','photon'),  'negative_charge'),
    (('electron','neutron','photon'),  'boson'),
    (('electron','neutron','photon'),  'massless'),
    (('electron','neutron','photon'),  'lightest'),
    # ── {electron, positron, photon}  — key "hard distractor" set
    (('electron','positron','photon'), 'negative_charge'),
    (('electron','positron','photon'), 'positive_charge'),
    (('electron','positron','photon'), 'neutral_charge'),
    (('electron','positron','photon'), 'antimatter'),
    (('electron','positron','photon'), 'boson'),
    (('electron','positron','photon'), 'lightest'),
    # ── {muon, electron, photon}
    (('muon','electron','photon'),     'boson'),
    (('muon','electron','photon'),     'massless'),
    (('muon','electron','photon'),     'lightest'),
    # ── {muon, positron, neutron}
    (('muon','positron','neutron'),    'negative_charge'),
    (('muon','positron','neutron'),    'positive_charge'),
    (('muon','positron','neutron'),    'neutral_charge'),
    (('muon','positron','neutron'),    'antimatter'),
    (('muon','positron','neutron'),    'lightest'),
    # ── {electron, positron, neutron}
    (('electron','positron','neutron'),'negative_charge'),
    (('electron','positron','neutron'),'positive_charge'),
    (('electron','positron','neutron'),'neutral_charge'),
    (('electron','positron','neutron'),'antimatter'),
    (('electron','positron','neutron'),'lightest'),
    # ── {muon, neutron, photon}
    (('muon','neutron','photon'),      'negative_charge'),
    (('muon','neutron','photon'),      'boson'),
    (('muon','neutron','photon'),      'massless'),
    (('muon','neutron','photon'),      'lepton'),
    # ── {muon, positron, photon}
    (('muon','positron','photon'),     'negative_charge'),
    (('muon','positron','photon'),     'positive_charge'),
    (('muon','positron','photon'),     'neutral_charge'),
    (('muon','positron','photon'),     'antimatter'),
    (('muon','positron','photon'),     'boson'),
    (('muon','positron','photon'),     'massless'),
    # ── {muon, proton, photon}
    (('muon','proton','photon'),       'negative_charge'),
    (('muon','proton','photon'),       'positive_charge'),
    (('muon','proton','photon'),       'neutral_charge'),
    (('muon','proton','photon'),       'boson'),
    (('muon','proton','photon'),       'massless'),
    (('muon','proton','photon'),       'lepton'),
    # ── {proton, neutron, photon}
    (('proton','neutron','photon'),    'boson'),
    (('proton','neutron','photon'),    'massless'),
    (('proton','neutron','photon'),    'lightest'),
]


def build_core() -> list[dict]:
    records = []
    for cs_tuple, fp in BASE_CASES:
        cs = list(cs_tuple)
        result = unique_satisfier(cs, fp)
        assert result is not None, f"Non-unique: {cs} / {fp}"
        correct, _ = result
        incorrect = hardest_incorrect(cs, correct, fp)
        for fi, fam in enumerate(FAMILY_NAMES):
            prompt = make_core_prompt(cs, fp, fi)
            r = make_record(prompt, cs, fp, correct, incorrect,
                            {"wording_family": fam, "experiment_type": "core"})
            records.append(r)
    return records


# ─── Distractor sensitivity ───────────────────────────────────────────────────
# For 3 key (filter, correct) pairs, vary which particles appear as distractors.
# This tests whether logprob_diff changes when distractors are harder.
# Under H2 (candidate-set-mediated): diff should decrease for harder distractors.
# Under H1 (direct retrieval): diff should be roughly constant.

DISTRACTOR_SETS = {
    # (filter_correct_id, distractor_difficulty, candidate_set, correct, incorrect)
    # Filter: negative_charge / correct: electron
    ('negative_charge__electron', 'trivial',  ('electron','photon','neutron'),  'electron', 'neutron'),
    ('negative_charge__electron', 'easy',     ('electron','proton','neutron'),  'electron', 'proton'),
    ('negative_charge__electron', 'medium',   ('electron','proton','photon'),   'electron', 'proton'),
    ('negative_charge__electron', 'hard',     ('electron','positron','photon'), 'electron', 'positron'),
    ('negative_charge__electron', 'hardest',  ('electron','positron','proton'), 'electron', 'positron'),
    # Filter: boson / correct: photon
    ('boson__photon', 'trivial',  ('photon','proton','neutron'),  'photon', 'proton'),
    ('boson__photon', 'easy',     ('photon','electron','neutron'),'photon', 'electron'),
    ('boson__photon', 'medium',   ('photon','electron','proton'), 'photon', 'electron'),
    ('boson__photon', 'hard',     ('photon','electron','positron'),'photon','positron'),
    # Filter: lepton / correct: electron
    ('lepton__electron', 'trivial', ('electron','proton','neutron'),  'electron', 'proton'),
    ('lepton__electron', 'easy',    ('electron','proton','photon'),   'electron', 'proton'),
    ('lepton__electron', 'medium',  ('electron','neutron','photon'),  'electron', 'neutron'),
    ('lepton__electron', 'hard',    ('electron','positron','photon'), 'electron', 'positron'),
    # Filter: neutral_charge / correct: photon
    ('neutral_charge__photon', 'trivial', ('photon','proton','positron'), 'photon', 'proton'),
    ('neutral_charge__photon', 'easy',    ('photon','electron','proton'), 'photon', 'electron'),
    ('neutral_charge__photon', 'medium',  ('photon','electron','muon'),   'photon', 'electron'),
    ('neutral_charge__photon', 'hard',    ('photon','electron','positron'),'photon','electron'),
}

DS_FAMILIES = ["F1_explicit_list", "F2_contextual", "F3_filter_first"]  # 3 families


def build_distractor_sensitivity() -> list[dict]:
    records = []
    for fc_id, diff_level, cs_tuple, correct, incorrect in DISTRACTOR_SETS:
        cs = list(cs_tuple)
        fp = fc_id.split("__")[0]
        for fi, fam in enumerate(DS_FAMILIES):
            ord_cs = ordered(cs, correct, fi)
            fq = FILTERS[fp]['question']
            if fi == 0:
                opts = make_options_phrase(ord_cs)
                prompt = f"The options are {opts}. Which particle {fq}? Answer:"
            elif fi == 1:
                opts = ", ".join(f"the {p}" for p in ord_cs)
                prompt = f"Consider these particles: {opts}. Among them, which one {fq}? Answer:"
            else:
                q_phrase = make_candidate_phrase(ord_cs)
                prompt = f"Which particle {fq} — {q_phrase}? Answer:"

            r = make_record(prompt, cs, fp, correct, incorrect, {
                "wording_family": fam,
                "experiment_type": "distractor_sensitivity",
                "distractor_difficulty": diff_level,
                "filter_correct_id": fc_id,
            })
            records.append(r)
    return records


# ─── Counterfactual experiment ────────────────────────────────────────────────
# H1 vs H2 test: does logprob_diff change when the candidate set is explicit vs absent?
# Five variant types:
#   original: standard F1 with candidate set
#   no_set:   same filter question, no candidates listed
#   reordered: same set, reversed order
#   extended:  original set + one extra neutral candidate
#   minimal:   only correct + hardest distractor (no third candidate)

CF_BASE_CASES = [
    # (cs, filter_property): base cases used for counterfactual variants
    (('electron','proton','neutron'),  'negative_charge'),
    (('electron','proton','photon'),   'boson'),
    (('electron','positron','photon'), 'negative_charge'),
    (('muon','proton','neutron'),      'lepton'),
    (('electron','proton','neutron'),  'lepton'),
]
CF_FAMILIES = ["F1_explicit_list", "F2_contextual"]
EXTRA_NEUTRAL = {'photon': 'photon', 'neutron': 'neutron', 'electron': 'electron'}


def build_counterfactual() -> list[dict]:
    records = []
    for cs_tuple, fp in CF_BASE_CASES:
        cs = list(cs_tuple)
        correct, _ = unique_satisfier(cs, fp)
        incorrect = hardest_incorrect(cs, correct, fp)
        fq = FILTERS[fp]['question']
        bcid = base_case_id(cs, fp)
        fcid = filter_correct_id(fp, correct)

        for fi, fam in enumerate(CF_FAMILIES):
            ord_cs = ordered(cs, correct, fi)

            # original
            prompt_orig = make_core_prompt(cs, fp, fi)
            records.append(make_record(prompt_orig, cs, fp, correct, incorrect, {
                "wording_family": fam, "experiment_type": "counterfactual",
                "variant_type": "original", "filter_correct_id": fcid,
            }))

            # no_set (F5)
            records.append(make_record(make_no_set_prompt(fp), cs, fp, correct, incorrect, {
                "wording_family": f"F5_no_set",
                "experiment_type": "counterfactual",
                "variant_type": "no_set", "filter_correct_id": fcid,
                "base_case_id": bcid,
            }))

            # reordered (only once per base case, not per family)
            if fi == 0:
                rev_cs = list(reversed(ord_cs))
                opts_rev = make_options_phrase(rev_cs)
                prompt_rev = f"The options are {opts_rev}. Which particle {fq}? Answer:"
                records.append(make_record(prompt_rev, cs, fp, correct, incorrect, {
                    "wording_family": "F1_reordered",
                    "experiment_type": "counterfactual",
                    "variant_type": "reordered", "filter_correct_id": fcid,
                }))

                # minimal: 2 candidates only
                prompt_min = make_minimal_prompt(correct, incorrect, fp)
                records.append(make_record(prompt_min, [correct, incorrect], fp, correct, incorrect, {
                    "wording_family": "F1_minimal",
                    "experiment_type": "counterfactual",
                    "variant_type": "minimal", "filter_correct_id": fcid,
                    "n_candidates": 2,
                }))

                # extended: 4 candidates — add one extra particle not in original set
                all_p = list(_P.keys())
                extras = [p for p in all_p if p not in cs and not unique_satisfier(cs + [p], fp) is None]
                # Filter to extras that don't change the unique satisfier
                safe_extras = []
                for ext in extras:
                    new_cs = cs + [ext]
                    r = unique_satisfier(new_cs, fp)
                    if r is not None and r[0] == correct:
                        safe_extras.append(ext)
                if safe_extras:
                    ext = safe_extras[0]
                    ext_cs = cs + [ext]
                    opts_ext = make_options_phrase(ext_cs)
                    prompt_ext = f"The options are {opts_ext}. Which particle {fq}? Answer:"
                    records.append(make_record(prompt_ext, ext_cs, fp, correct, incorrect, {
                        "wording_family": "F1_extended",
                        "experiment_type": "counterfactual",
                        "variant_type": "extended", "filter_correct_id": fcid,
                        "n_candidates": len(ext_cs),
                    }))

    # Deduplicate no_set prompts (same prompt appears for multiple base_cases / families)
    seen = set()
    deduped = []
    for r in records:
        key = (r["prompt"], r.get("variant_type",""), r.get("filter_property",""))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


# ─── Set-size variation ───────────────────────────────────────────────────────
# Same filter, same correct answer, but candidate set size varies (2 to 5).
# Tests whether the number of options changes logprob_diff.

SET_SIZE_FILTERS = [
    # (filter_property, correct, size_2, size_3, size_4, size_5)
    # Each entry gives the candidate sets at each size
    ('negative_charge', 'electron',
     ('electron','proton'),
     ('electron','proton','neutron'),
     ('electron','proton','neutron','photon'),
     ('electron','proton','neutron','photon','positron')),
    ('boson', 'photon',
     ('photon','electron'),
     ('photon','electron','proton'),
     ('photon','electron','proton','neutron'),
     ('photon','electron','proton','neutron','positron')),
    ('lepton', 'electron',
     ('electron','proton'),
     ('electron','proton','neutron'),
     ('electron','proton','neutron','photon'),
     ('electron','proton','neutron','photon','muon')),
]
# Note: for size_5 with filter=lepton and correct=electron: {e,p,n,γ,μ} has TWO leptons → INVALID!
# Fix: use {e,p,n,γ, proton_extra} or just exclude invalid ones.

SS_FAMILIES = ["F1_explicit_list", "F2_contextual", "F3_filter_first"]


def build_set_size() -> list[dict]:
    records = []
    for fp, correct, s2, s3, s4, s5 in SET_SIZE_FILTERS:
        for cs_tuple in [s2, s3, s4, s5]:
            cs = list(cs_tuple)
            result = unique_satisfier(cs, fp)
            if result is None or result[0] != correct:
                continue  # skip invalid (e.g., lepton size-5 with muon)
            incorrect = hardest_incorrect(cs, correct, fp)
            fcid = filter_correct_id(fp, correct)
            for fi, fam in enumerate(SS_FAMILIES):
                if len(cs) > 3:
                    # For larger sets use simpler templates
                    fq = FILTERS[fp]['question']
                    opts = make_options_phrase(cs)
                    if fi == 0:
                        prompt = f"The options are {opts}. Which particle {fq}? Answer:"
                    elif fi == 1:
                        opts2 = ", ".join(f"the {p}" for p in cs)
                        prompt = f"Consider these {len(cs)} particles: {opts2}. Which one {fq}? Answer:"
                    else:
                        q_phrase = make_candidate_phrase(cs)
                        prompt = f"Which particle {fq} — {q_phrase}? Answer:"
                else:
                    prompt = make_core_prompt(cs, fp, fi)
                records.append(make_record(prompt, cs, fp, correct, incorrect, {
                    "wording_family": fam,
                    "experiment_type": "set_size",
                    "filter_correct_id": fcid,
                }))
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    prompts = [r["prompt"] for r in records]
    n_dup = len(prompts) - len(set(prompts))
    exps = sorted({r.get("experiment_type","?") for r in records})
    targets = sorted({r.get("target_candidate","?") for r in records})
    filters = sorted({r.get("filter_property","?") for r in records})
    print(f"  Wrote {len(records)} prompts → {path.name}  |  duplicates: {n_dup}")
    print(f"    Experiments: {exps}")
    print(f"    Targets:     {targets}")
    print(f"    Filters:     {filters}")


def main():
    print("Generating candidate-state large dataset...")
    print()

    core  = build_core()
    ds    = build_distractor_sensitivity()
    cf    = build_counterfactual()
    ss    = build_set_size()

    all_records = core + ds + cf + ss
    # Remove global duplicates (same prompt text)
    seen, deduped = set(), []
    for r in all_records:
        k = r["prompt"]
        if k not in seen:
            seen.add(k)
            deduped.append(r)

    print(f"  core:                  {len(core)}")
    print(f"  distractor_sensitivity:{len(ds)}")
    print(f"  counterfactual:        {len(cf)}")
    print(f"  set_size:              {len(ss)}")
    print(f"  total (before dedup):  {len(all_records)}")
    print(f"  total (after dedup):   {len(deduped)}")
    print()

    path = OUT_DIR / "physics_particle_candidate_selection_train.jsonl"
    write_jsonl(path, deduped)
    print()
    print(f"Prompt size breakdown:")
    from collections import Counter
    size_counts = Counter(r["n_candidates"] for r in deduped)
    for k in sorted(size_counts):
        print(f"  {k}-candidate sets: {size_counts[k]}")

    print()
    print("Token audit (confirm on CSD3):")
    for t in [" electron", " proton", " neutron", " photon", " positron", " muon"]:
        print(f"  {repr(t)}")


if __name__ == "__main__":
    main()
