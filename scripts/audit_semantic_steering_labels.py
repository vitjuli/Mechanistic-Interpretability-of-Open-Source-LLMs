#!/usr/bin/env python
"""
audit_semantic_steering_labels.py  [what did the semantic-steering bundle actually measure?]

The overnight bundle (colab/run_semantic_steering.sh -> steering_decode_check.py) recorded, for
each prompt, a margin against the token in the prompt's `incorrect_answer` field. In the v2
particle corpora that field is an arbitrary distractor, not the partner class of the pair, so
for many rows `margin`/`dmargin`/`flipped` answer a question nobody asked (e.g. electron vs
proton inside an electron/photon sweep).

This script rebuilds, from the corpora themselves, what the contrast SHOULD have been and
compares it with what was recorded, so App G can quote only the rows that survive.

Row is usable  <=>  recorded contrast == the pair partner of the prompt's own class.

USAGE
  python scripts/audit_semantic_steering_labels.py \
      --csv  data/analysis/steering_named/semantic_L22_L24_L35/semantic_steering_all.csv \
      --out  data/analysis/steering_named/semantic_L22_L24_L35/label_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# tag in the CSV -> corpus that produced it (from colab/run_semantic_steering.sh)
TAG_TO_CORPUS = {
    "electron_vs_photon":  "data/prompts/particle_pairs/particles_electron_vs_photon.jsonl",
    "electron_vs_proton":  "data/prompts/particle_pairs/particles_electron_vs_proton.jsonl",
    "electron_vs_neutron": "data/prompts/particle_pairs/particles_electron_vs_neutron.jsonl",
    "neutron_vs_photon":   "data/prompts/particle_pairs/particles_neutron_vs_photon.jsonl",
    "neutron_vs_proton":   "data/prompts/particle_pairs/particles_neutron_vs_proton.jsonl",
    "photon_vs_proton":    "data/prompts/particle_pairs/particles_photon_vs_proton.jsonl",
    "alpha_beta":          "data/prompts/B1_alpha_beta.jsonl",
    "grammar":             "data/prompts/B1_grammar_number.jsonl",
}
K_PER_CLASS = 8   # run_semantic_steering.sh: first K prompts of each of the first two classes


def balanced_subset(rows, k=K_PER_CLASS):
    """Reproduce the /tmp/sub.jsonl selection of run_semantic_steering.sh exactly."""
    answers = sorted({r["correct_answer"] for r in rows if r.get("correct_answer")})
    sub = []
    for a in answers[:2]:
        sub += [r for r in rows if r.get("correct_answer") == a][:k]
    return sub, answers[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root)

    audit = {}
    for tag, corpus in TAG_TO_CORPUS.items():
        path = root / corpus
        if not path.exists():
            print(f"[skip] {tag}: corpus not found at {corpus}")
            continue
        rows = [json.loads(l) for l in open(path)]
        sub, classes = balanced_subset(rows)
        for i, p in enumerate(sub):
            own = str(p.get("correct_answer", "")).strip()
            partner = [str(c).strip() for c in classes if str(c).strip() != own]
            audit[(tag, i)] = {
                "prompt_class": own,
                "expected_contrast": partner[0] if partner else "",
                "corpus_incorrect_field": str(p.get("incorrect_answer", "")).strip(),
            }

    seen, out_rows = set(), []
    with open(root / a.csv) as f:
        for r in csv.DictReader(f):
            key = (r["tag"], int(r["prompt_idx"]))
            if key in seen:
                continue
            seen.add(key)
            ref = audit.get(key)
            if ref is None:
                out_rows.append({"tag": key[0], "prompt_idx": key[1], "prompt_class": "?",
                                 "recorded_contrast": r["incorrect"], "expected_contrast": "?",
                                 "usable": 0, "reason": "no corpus mapping"})
                continue
            ok = r["incorrect"].strip() == ref["expected_contrast"]
            out_rows.append({"tag": key[0], "prompt_idx": key[1],
                             "prompt_class": ref["prompt_class"],
                             "recorded_contrast": r["incorrect"].strip(),
                             "expected_contrast": ref["expected_contrast"],
                             "usable": int(ok),
                             "reason": "" if ok else "contrast is not the pair partner"})

    out_rows.sort(key=lambda r: (r["tag"], r["prompt_idx"]))
    outp = root / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "prompt_idx", "prompt_class",
                                          "recorded_contrast", "expected_contrast",
                                          "usable", "reason"])
        w.writeheader(); [w.writerow(r) for r in out_rows]

    per_tag = defaultdict(lambda: [0, 0])
    for r in out_rows:
        per_tag[r["tag"]][0] += r["usable"]; per_tag[r["tag"]][1] += 1
    print(f"\nlabel audit -> {outp}  ({len(out_rows)} prompt rows)\n")
    print(f"{'tag':22s} {'usable':>10s}   verdict")
    for tag in sorted(per_tag):
        ok, n = per_tag[tag]
        verdict = "OK — App G may quote it" if ok == n else (
            "UNUSABLE — re-run required" if ok == 0 else "PARTIAL — quote only usable rows")
        print(f"{tag:22s} {ok:>4d}/{n:<5d}   {verdict}")
    tot_ok = sum(v[0] for v in per_tag.values()); tot = sum(v[1] for v in per_tag.values())
    print(f"\ntotal usable: {tot_ok}/{tot}")
    print("\nNOTE: this audits the CONTRAST axis only. The 2026-07-11 bundle also ran on\n"
          "Qwen/Qwen3-4B-Base (the old steering_decode_check.py default) while the dumps that\n"
          "supplied the directions and sigma are Qwen/Qwen3-4B — that defect hits every row,\n"
          "including the ones marked usable here.")


if __name__ == "__main__":
    main()
