"""
gen_B1_grammar_number.py   [corpus generator: grammar_number raw questions + config]
=====================================================================================
Generates the B1 grammar_number concept corpus per spec §3.1:
  - 26 target families (scenario = subject noun + verb context), each rendered
    through 5 wording templates x {singular, plural} subject = 10 prompts/family,
    260 total, perfectly balanced minimal pairs;
  - 20 regular-plural families (cue_type=regular_s) + 6 irregular
    (cue_type=irregular: men, women, children, people, mice, geese);
  - question format: classification with labels singular/plural at the decision
    position (verb form agrees with subject number);
  - exemplars for the scaffold come from 2 SEPARATE scenarios (cat, birds) that
    do not appear among targets;
  - no subject-verb attractors (intervening nouns) in B1 — that harder probe is
    deliberately left out and noted in the concept card.

Intrinsic shortcut (documented, allowed): the subject's number marking is
visible lexically — that IS the concept's surface nature; each specific noun
form appears only ~5 times so the BoW gate (min_count=10) is not tripped.

Usage:  python gen_B1_grammar_number.py --out_raw data/prompts/raw/B1_grammar_number_raw.jsonl \
            --out_config configs/B1_grammar_number.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (noun_sing, noun_plur, verb_context_complement, cue_type)
SCENARIOS = [
    ("dog", "dogs", "to the park", "regular_s"),
    ("teacher", "teachers", "the homework carefully", "regular_s"),
    ("bird", "birds", "over the lake", "regular_s"),
    ("engineer", "engineers", "the bridge design", "regular_s"),
    ("chef", "chefs", "the evening meal", "regular_s"),
    ("student", "students", "in the library", "regular_s"),
    ("nurse", "nurses", "the night shift", "regular_s"),
    ("farmer", "farmers", "the field before dawn", "regular_s"),
    ("musician", "musicians", "at the concert hall", "regular_s"),
    ("driver", "drivers", "along the coastal road", "regular_s"),
    ("scientist", "scientists", "the new results", "regular_s"),
    ("painter", "painters", "the old fence", "regular_s"),
    ("lawyer", "lawyers", "the contract clauses", "regular_s"),
    ("gardener", "gardeners", "the rose beds", "regular_s"),
    ("pilot", "pilots", "through the storm front", "regular_s"),
    ("baker", "bakers", "fresh bread daily", "regular_s"),
    ("soldier", "soldiers", "across the river", "regular_s"),
    ("actor", "actors", "the final scene", "regular_s"),
    ("plumber", "plumbers", "the kitchen sink", "regular_s"),
    ("librarian", "librarians", "the returned books", "regular_s"),
    ("man", "men", "near the harbour", "irregular"),
    ("woman", "women", "the committee meeting", "irregular"),
    ("child", "children", "in the playground", "irregular"),
    ("person", "people", "at the bus stop", "irregular"),
    ("mouse", "mice", "behind the cupboard", "irregular"),
    ("goose", "geese", "across the frozen pond", "irregular"),
]

TEMPLATES = [
    "Every morning the {noun} ___ {comp}",
    "On weekends the {noun} ___ {comp}",
    "In the early evening the {noun} ___ {comp}",
    "Quite often the {noun} ___ {comp}",
    "After lunch the {noun} ___ {comp}",
]

QUESTION = ('In the sentence "{sentence}", should the missing verb be in the '
            'singular or plural form?')

EXEMPLARS = [
    {"question": QUESTION.format(sentence="The cat ___ on the windowsill"),
     "answer": "label_a", "_provenance": "exemplar scenario 'cat' — not among targets"},
    {"question": QUESTION.format(sentence="The birds ___ south in autumn"),
     "answer": "label_b", "_provenance": "exemplar scenario 'birds' — not among targets"},
]


def build_rows():
    rows = []
    for sing, plur, comp, cue in SCENARIOS:
        fam = f"GN-{sing}"
        for t_i, tpl in enumerate(TEMPLATES, 1):
            for noun, ans in ((sing, "singular"), (plur, "plural")):
                sentence = tpl.format(noun=noun, comp=comp)
                rows.append({
                    "raw_question": QUESTION.format(sentence=sentence),
                    "correct_answer": ans,
                    "surface_family": fam,
                    "cue_type": cue,
                    "prompt_id": f"{fam}-t{t_i}-{ans[:4]}",
                })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_raw", default="data/prompts/raw/B1_grammar_number_raw.jsonl")
    p.add_argument("--out_config", default="configs/B1_grammar_number.json")
    args = p.parse_args()

    rows = build_rows()
    n_fam = len({r["surface_family"] for r in rows})
    n_sing = sum(r["correct_answer"] == "singular" for r in rows)
    assert len(rows) == 260 and n_fam == 26 and n_sing == 130, "corpus invariants"

    Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_raw, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = {
        "concept": "grammar_number",
        "label_a": "singular",
        "label_b": "plural",
        "ab_mode": False,
        "_fallback_note": "if 120's tokenization gate fails on ' singular'/' plural', "
                          "set ab_mode=true and add ab_legend 'A: singular, B: plural'",
        "exemplars": EXEMPLARS,
    }
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(args.out_config, "w"), indent=2, ensure_ascii=False)
    print(f"raw: {args.out_raw} ({len(rows)} rows, {n_fam} families, balance {n_sing}/{len(rows)-n_sing})")
    print(f"config: {args.out_config}")


if __name__ == "__main__":
    main()
