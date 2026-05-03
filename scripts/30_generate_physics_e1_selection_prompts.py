"""
30_generate_physics_e1_selection_prompts.py

Generates train and test JSONL for the physics_e1_selection behaviour.

Task: "Is this electric dipole (E1) transition allowed or forbidden?"
Rule: Δl = ±1 → allowed; all other Δl → forbidden

Design:
  - 8 transition groups (4 allowed: sp,ps,pd,dp / 4 forbidden: ss,pp,dd,sd)
  - 4 wording families: F1=orbital_name, F2=quantum_number, F3=notation, F4=delta_l_explicit
  - Contrastive pairs built in metadata
  - Train: F1 v1-v3 + F2 v1-v3 + F3 v1-v3 + F4 all = 96 + 12 = 108 ... see below
  - Train: 84 prompts, Test: 24 prompts (F1-F3 variant 4 held out)

Outputs:
  data/prompts/physics_e1_selection_train.jsonl
  data/prompts/physics_e1_selection_test.jsonl

Usage:
  python scripts/30_generate_physics_e1_selection_prompts.py
"""

import json
import itertools
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/prompts"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Core transition groups ──────────────────────────────────────────────────
# (group_id, l_init, l_final, result)
TRANSITIONS = {
    "sp": dict(init="s", final="p", l_init=0, l_final=1, delta_l=1,  result="allowed"),
    "ps": dict(init="p", final="s", l_init=1, l_final=0, delta_l=-1, result="allowed"),
    "pd": dict(init="p", final="d", l_init=1, l_final=2, delta_l=1,  result="allowed"),
    "dp": dict(init="d", final="p", l_init=2, l_final=1, delta_l=-1, result="allowed"),
    "ss": dict(init="s", final="s", l_init=0, l_final=0, delta_l=0,  result="forbidden"),
    "pp": dict(init="p", final="p", l_init=1, l_final=1, delta_l=0,  result="forbidden"),
    "dd": dict(init="d", final="d", l_init=2, l_final=2, delta_l=0,  result="forbidden"),
    "sd": dict(init="s", final="d", l_init=0, l_final=2, delta_l=2,  result="forbidden"),
}

# Contrastive pair assignments (group_id → pair_id, role)
CONTRASTIVE = {
    "sp": ("sp_vs_ss", "allowed"),
    "ss": ("sp_vs_ss", "forbidden"),
    "pd": ("pd_vs_pp", "allowed"),
    "pp": ("pd_vs_pp", "forbidden"),
    "dp": ("dp_vs_dd", "allowed"),
    "dd": ("dp_vs_dd", "forbidden"),
    "ps": ("sp_vs_sd", "allowed"),  # reuse sp_vs_sd pair with ps as "same Δl, downward"
    "sd": ("sp_vs_sd", "forbidden"),
}

# Standard notation n-values per group (4 variants)
NOTATION = {
    "sp": [("1s", "2p"), ("2s", "3p"), ("2s", "2p"), ("3s", "4p")],
    "ps": [("2p", "1s"), ("3p", "2s"), ("2p", "2s"), ("4p", "3s")],
    "pd": [("2p", "3d"), ("3p", "4d"), ("3p", "3d"), ("4p", "5d")],
    "dp": [("3d", "2p"), ("4d", "3p"), ("3d", "3p"), ("5d", "4p")],
    "ss": [("2s", "1s"), ("3s", "2s"), ("4s", "3s"), ("2s", "3s")],
    "pp": [("2p", "3p"), ("3p", "4p"), ("4p", "5p"), ("3p", "2p")],
    "dd": [("3d", "4d"), ("4d", "5d"), ("5d", "4d"), ("3d", "5d")],
    "sd": [("1s", "3d"), ("2s", "4d"), ("3s", "5d"), ("2s", "3d")],
}

def article(orbital: str) -> str:
    """Return 'an' for s (vowel start) and 'a' for p, d."""
    return "an" if orbital == "s" else "a"

def an(orbital: str) -> str:
    return article(orbital) + " " + orbital

# ── F1: Orbital name templates (4 wording variants per group) ─────────────
def f1_prompts(gid: str, t: dict) -> list[dict]:
    init, final = t["init"], t["final"]
    ai, af = article(init), article(final)
    same = "another " if init == final else ""

    templates = [
        # v1: direct question with E1 explicit
        f"Is an electric dipole (E1) transition from {an(init)} orbital to {same}{an(final)} orbital "
        f"allowed or forbidden by the selection rules? Answer:",

        # v2: rephrase with "dipole-allowed"
        f"For electric dipole (E1) radiation, is the atomic transition from {an(init)} state to "
        f"{same}{an(final)} state dipole-allowed? Answer:",

        # v3: single-photon framing (keyword-free variant — no explicit "E1" or "electric dipole")
        f"An electron in {an(init)} orbital emits a single photon and transitions to "
        f"{same}{an(final)} orbital. Is this transition allowed or forbidden by angular momentum "
        f"selection rules? Answer:",

        # v4 (TEST): rule-framing without answer cue words in body
        f"The orbital angular momentum selection rule for single-photon emission requires |Δl| = 1. "
        f"Does the transition from {an(init)} orbital to {same}{an(final)} orbital satisfy this? "
        f"Answer:",
    ]
    out = []
    for v, tmpl in enumerate(templates, start=1):
        level  = "L1" if v == 2 else "L2"  # v2 is most direct
        kw_free = (v == 3)  # v3 has no "E1" or "electric dipole"
        out.append(dict(
            wording_family="F1", wording_family_label="orbital_name",
            wording_variant=v, level=level, level_label="orbital_name",
            difficulty="easy" if v in (1, 2) else "medium",
            inference_steps=2 if v in (1, 2) else 3,
            keyword_free=kw_free,
            prompt=tmpl,
            prompt_short=f"E1 {init}→{final}? (F1v{v})",
            has_e1_keyword=(not kw_free),
        ))
    return out

# ── F2: Quantum number templates (3 wording variants per group) ────────────
def f2_prompts(gid: str, t: dict) -> list[dict]:
    l1, l2 = t["l_init"], t["l_final"]
    templates = [
        # v1: direct quantum number statement
        f"An atom with orbital angular momentum quantum number l={l1} undergoes an electric dipole "
        f"(E1) transition to a state with l={l2}. Is this transition allowed or forbidden? Answer:",

        # v2: rephrase as emission
        f"A photon is emitted via electric dipole (E1) radiation. The initial state has l={l1} and "
        f"the final state has l={l2}. Is this E1 transition allowed or forbidden? Answer:",

        # v3: delta-l computed but not stated (harder — inference_steps=2)
        f"In a single-photon emission event, an electron transitions from a state with orbital "
        f"angular momentum quantum number l={l1} to a state with l={l2}. Applying the dipole "
        f"selection rule, is this transition allowed or forbidden? Answer:",
    ]
    out = []
    for v, tmpl in enumerate(templates, start=1):
        out.append(dict(
            wording_family="F2", wording_family_label="quantum_number",
            wording_variant=v, level="L1", level_label="quantum_number",
            difficulty="easy",
            inference_steps=1,
            keyword_free=False,
            prompt=tmpl,
            prompt_short=f"E1 l={l1}→l={l2}? (F2v{v})",
            has_e1_keyword=True,
        ))
    return out

# ── F3: Standard spectroscopic notation (4 variants per group) ─────────────
def f3_prompts(gid: str, t: dict) -> list[dict]:
    notations = NOTATION[gid]
    out = []
    for v, (n1_orb, n2_orb) in enumerate(notations, start=1):
        # Extract orbital letter and principal quantum number
        n1, orb1 = n1_orb[:-1], n1_orb[-1]
        n2, orb2 = n2_orb[:-1], n2_orb[-1]

        templates = {
            1: f"Is the atomic transition {n1_orb} → {n2_orb} allowed by the electric dipole "
               f"(E1) selection rule? Answer:",
            2: f"Consider the single-photon transition {n1_orb} → {n2_orb} in a hydrogen-like "
               f"atom. Is this an E1-allowed transition? Answer:",
            3: f"An electron in the {n1_orb} state emits a photon and transitions to the "
               f"{n2_orb} state. Is this electric dipole transition allowed or forbidden? Answer:",
            4: f"The spectroscopic transition {n1_orb} → {n2_orb}. Does it satisfy the electric "
               f"dipole (E1) selection rule? Answer:",
        }
        tmpl = templates[v]
        out.append(dict(
            wording_family="F3", wording_family_label="standard_notation",
            wording_variant=v, level="L2", level_label="standard_notation",
            difficulty="medium",
            inference_steps=2,
            keyword_free=False,
            prompt=tmpl,
            prompt_short=f"E1 {n1_orb}→{n2_orb}? (F3v{v})",
            has_e1_keyword=True,
            spectroscopic_notation=f"{n1_orb}→{n2_orb}",
        ))
    return out

# ── F4: Explicit Δl (3 variants per Δl value) ──────────────────────────────
# Groups: dl0 (Δl=0, forbidden), dlp1 (Δl=+1, allowed),
#         dlm1 (Δl=−1, allowed), dlp2 (Δl=+2, forbidden)
F4_CASES = {
    "dl0":  dict(delta_l=0,  result="allowed" if False else "forbidden",
                 l_init=None, l_final=None, init=None, final=None),
    "dlp1": dict(delta_l=1,  result="allowed",
                 l_init=None, l_final=None, init=None, final=None),
    "dlm1": dict(delta_l=-1, result="allowed",
                 l_init=None, l_final=None, init=None, final=None),
    "dlp2": dict(delta_l=2,  result="forbidden",
                 l_init=None, l_final=None, init=None, final=None),
}
# Fix result for dl0
F4_CASES["dl0"]["result"] = "forbidden"

def f4_prompts(gid: str, dl: int, result: str) -> list[dict]:
    dl_str = f"+{dl}" if dl > 0 else str(dl)
    templates = [
        # v1: direct rule-application
        f"In an electric dipole (E1) transition, the orbital angular momentum quantum number "
        f"changes by Δl = {dl_str}. Is this transition allowed or forbidden? Answer:",

        # v2: passive phrasing
        f"A photon is emitted via electric dipole (E1) radiation. The change in orbital angular "
        f"momentum quantum number is Δl = {dl_str}. Is this transition allowed or forbidden? Answer:",

        # v3: abstract framing
        f"For the E1 selection rule, a transition with Δl = {dl_str} is: Answer:",
    ]
    out = []
    for v, tmpl in enumerate(templates, start=1):
        out.append(dict(
            wording_family="F4", wording_family_label="delta_l_explicit",
            wording_variant=v, level="L1", level_label="delta_l_explicit",
            difficulty="easy",
            inference_steps=1,
            keyword_free=False,
            prompt=tmpl,
            prompt_short=f"E1 Δl={dl_str}? (F4v{v})",
            has_e1_keyword=True,
            # F4 group-specific overrides:
            group_id=gid,
            initial_l=None, final_l=None,
            initial_orbital=None, final_orbital=None,
            delta_l=dl,
            abs_delta_l=abs(dl),
            transition_direction=("up" if dl > 0 else ("down" if dl < 0 else "same")),
        ))
    return out

# ── Build all prompts ──────────────────────────────────────────────────────
all_prompts = []
idx = 0

# F1, F2, F3 prompts for 8 transition groups
for gid, t in TRANSITIONS.items():
    cp_id, cp_role = CONTRASTIVE.get(gid, (None, None))
    direction = "up" if t["delta_l"] > 0 else ("down" if t["delta_l"] < 0 else "same")
    base_meta = dict(
        behaviour="physics_e1_selection",
        behaviour_type="selection_rule",
        group_id=gid,
        initial_orbital=t["init"],
        final_orbital=t["final"],
        initial_l=t["l_init"],
        final_l=t["l_final"],
        delta_l=t["delta_l"],
        abs_delta_l=abs(t["delta_l"]),
        transition_direction=direction,
        selection_result=t["result"],
        correct_answer=" " + t["result"],
        incorrect_answer=" allowed" if t["result"] == "forbidden" else " forbidden",
        latent_state_target=t["result"],
        physics_concept=f"e1_transition_{t['result']}",
        ic_concept_group=t["result"],
        contrastive_pair_id=cp_id,
        contrastive_role=cp_role,
        is_anchor=(gid in ("sp", "ss")),     # sp/ss are anchor groups
        is_kw_variant=False,
        is_auxiliary=False,
        has_alpha_keyword=False,
        has_beta_keyword=False,
        has_allowed_keyword=False,
        has_forbidden_keyword=False,
        evidence_completeness="single",
        is_uniquely_determining=True,
        semantic_equivalence_group=gid,
        relation_type="delta_l_equals_pm1" if t["result"] == "allowed" else "delta_l_neq_pm1",
        concept_route="selection_rule_filter",
        cue_type=f"delta_l_{'+' if t['delta_l'] >= 0 else ''}{t['delta_l']}",
        cue_label=f"delta_l_{'+' if t['delta_l'] >= 0 else ''}{t['delta_l']}",
        cue_set=[f"delta_l_{'+' if t['delta_l'] >= 0 else ''}{t['delta_l']}"],
    )

    for fam_prompts in [f1_prompts(gid, t), f2_prompts(gid, t), f3_prompts(gid, t)]:
        for p in fam_prompts:
            rec = {**base_meta, **p}
            rec["prompt_idx"] = idx
            rec["prompt_id"] = (
                f"{p['wording_family']}-{gid}-v{p['wording_variant']:02d}"
            )
            # is_test: use variant 4 of F1 and F3 as test; all F2 variants in train
            rec["split"] = (
                "test"
                if (p["wording_family"] in ("F1", "F3") and p["wording_variant"] == 4)
                else "train"
            )
            all_prompts.append(rec)
            idx += 1

# F4 prompts (all in train)
for gid, fc in F4_CASES.items():
    dl = fc["delta_l"]
    result = fc["result"]
    for p in f4_prompts(gid, dl, result):
        rec = {**p}
        rec["behaviour"] = "physics_e1_selection"
        rec["behaviour_type"] = "selection_rule"
        rec["selection_result"] = result
        rec["correct_answer"] = " " + result
        rec["incorrect_answer"] = " allowed" if result == "forbidden" else " forbidden"
        rec["latent_state_target"] = result
        rec["physics_concept"] = f"e1_transition_{result}"
        rec["ic_concept_group"] = result
        rec["contrastive_pair_id"] = None
        rec["contrastive_role"] = None
        rec["is_anchor"] = (gid == "dlp1")
        rec["is_kw_variant"] = False
        rec["is_auxiliary"] = True      # F4 is auxiliary: explicit Δl given
        rec["has_alpha_keyword"] = False
        rec["has_beta_keyword"] = False
        rec["has_allowed_keyword"] = False
        rec["has_forbidden_keyword"] = False
        rec["evidence_completeness"] = "single"
        rec["is_uniquely_determining"] = True
        rec["semantic_equivalence_group"] = gid
        rec["relation_type"] = "delta_l_equals_pm1" if result == "allowed" else "delta_l_neq_pm1"
        rec["concept_route"] = "direct_delta_l"
        rec["cue_type"] = f"explicit_delta_l_{gid}"
        rec["cue_label"] = f"explicit_delta_l_{gid}"
        rec["initial_orbital"] = None
        rec["final_orbital"] = None
        rec["split"] = "train"
        rec["prompt_idx"] = idx
        rec["prompt_id"] = f"{rec['wording_family']}-{gid}-v{rec['wording_variant']:02d}"
        all_prompts.append(rec)
        idx += 1

# ── Split and write ──────────────────────────────────────────────────────────
train = [p for p in all_prompts if p["split"] == "train"]
test  = [p for p in all_prompts if p["split"] == "test"]

# Re-index within each split
for i, p in enumerate(train):
    p["prompt_idx"] = i
for i, p in enumerate(test):
    p["prompt_idx"] = i

def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

write_jsonl(train, OUTDIR / "physics_e1_selection_train.jsonl")
write_jsonl(test,  OUTDIR / "physics_e1_selection_test.jsonl")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Train: {len(train)} prompts")
print(f"Test:  {len(test)} prompts")
print(f"Total: {len(all_prompts)} prompts")
print()

from collections import Counter
for split_name, recs in [("TRAIN", train), ("TEST", test)]:
    print(f"=== {split_name} ===")
    fams = Counter(p["wording_family"] for p in recs)
    groups = Counter(p["group_id"] for p in recs)
    answers = Counter(p["correct_answer"].strip() for p in recs)
    print(f"  Families: {dict(fams)}")
    print(f"  Groups:   {dict(groups)}")
    print(f"  Answers:  {dict(answers)}")
    print()
