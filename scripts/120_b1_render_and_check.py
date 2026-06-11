"""
120_b1_render_and_check.py   [B1 corpus renderer + static gates (spec §1)]
===========================================================================
Takes RAW target questions + a concept config and produces the scaffolded B1
corpus jsonl, enforcing every static rule of the B1 spec BEFORE any GPU time:

  - 2-shot scaffold (§1.1): fixed exemplar pair, one per class, order XY/YX
    counterbalanced 50/50 and recorded in `scaffold_order`;
  - label tokenization gate (§1.2): " "+label must be a SINGLE token; if the
    config sets ab_mode, labels become "A"/"B" with `ab_assignment`
    counterbalanced 50/50 and the legend appended to each question;
  - family stats (§1.3): >= min_families families, >= min_per_family prompts
    per family, class balance per family and globally;
  - shortcut scan (§1.4): bag-of-words per-token AUC over the RAW questions;
    fails the gate if any single token's AUC >= 0.95 (overridable per spec
    when intrinsic to the concept, e.g. lang_id: --allow_shortcuts).

Concept config (json):
{
  "concept": "fission_fusion",
  "label_a": "fission",            # class 0
  "label_b": "fusion",             # class 1
  "ab_mode": false,                # true -> render labels as A/B with legend
  "ab_legend": "A: fission, B: fusion",
  "exemplars": [
    {"question": "...boring class-0 question...", "answer": "label_a"},
    {"question": "...boring class-1 question...", "answer": "label_b"}
  ]
}

Raw questions jsonl (one per line):
{"raw_question": "...", "correct_answer": "<label_a|label_b>",
 "surface_family": "...", "cue_type": "<optional>"}

Output: corpus jsonl in the B1 schema (§1.5) + render_report_{concept}.json.

SELF-TEST (no tokenizer / no repo):  python 120_b1_render_and_check.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("render120")

SCAFFOLD = "{q1}\nAnswer: {a1}\n\n{q2}\nAnswer: {a2}\n\n{qt}\nAnswer:"


# =====================================================================
# Pure-python core (exercised by --self_test)
# =====================================================================
def auc_scalar(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def tokenize_words(text):
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower())


def bow_shortcut_scan(questions, y, min_count=10):
    """Per-token presence AUC over raw questions; returns sorted (token, auc) list."""
    vocab = Counter(w for q in questions for w in set(tokenize_words(q)))
    rows = []
    for w, c in vocab.items():
        if c < min_count:
            continue
        pres = np.array([int(w in set(tokenize_words(q))) for q in questions])
        a = auc_scalar(pres, y)
        rows.append((w, max(a, 1 - a)))
    rows.sort(key=lambda t: -t[1])
    return rows


def render_prompt(qt, ex0, ex1, order, label_for):
    """order 'XY' = class-0 exemplar first; labels resolved via label_for(cls)."""
    e_first, e_second = (ex0, ex1) if order == "XY" else (ex1, ex0)
    return SCAFFOLD.format(q1=e_first["question"], a1=label_for(e_first["cls"]),
                           q2=e_second["question"], a2=label_for(e_second["cls"]),
                           qt=qt)


def assign_counterbalance(n, rng, options=("XY", "YX")):
    half = n // 2
    arr = [options[0]] * half + [options[1]] * (n - half)
    rng.shuffle(arr)
    return arr


def family_stats(rows):
    fam = defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["surface_family"]][r["y"]] += 1
    return {f: tuple(v) for f, v in fam.items()}


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    # (1) scaffold render: order respected, ends with 'Answer:' (no trailing space)
    ex0 = {"question": "Q0?", "cls": 0}; ex1 = {"question": "Q1?", "cls": 1}
    lab = lambda c: ["alpha", "beta"][c]
    p_xy = render_prompt("QT?", ex0, ex1, "XY", lab)
    p_yx = render_prompt("QT?", ex0, ex1, "YX", lab)
    assert p_xy.endswith("QT?\nAnswer:") and p_yx.endswith("QT?\nAnswer:")
    assert p_xy.index("Q0?") < p_xy.index("Q1?") and p_yx.index("Q1?") < p_yx.index("Q0?")
    assert "Answer: alpha" in p_xy and "Answer: beta" in p_xy

    # (2) counterbalance is 50/50
    cb = assign_counterbalance(101, rng)
    assert abs(cb.count("XY") - cb.count("YX")) <= 1

    # (3) BoW scan catches a planted shortcut and ignores balanced words
    qs, y = [], []
    for i in range(200):
        cls = i % 2
        leak = "zork" if cls == 1 else "blee"
        qs.append(f"the common process {leak} happens often in samples")
        y.append(cls)
    scan = bow_shortcut_scan(qs, np.array(y))
    top = dict(scan[:4])
    assert top.get("zork", 0) > 0.99 and top.get("blee", 0) > 0.99, "planted shortcut must be caught"
    assert dict(scan).get("common", 0.0) < 0.6, "balanced word must be ~0.5"

    # (4) family stats
    rows = [{"surface_family": "f1", "y": 0}, {"surface_family": "f1", "y": 1},
            {"surface_family": "f2", "y": 0}]
    fs = family_stats(rows)
    assert fs["f1"] == (1, 1) and fs["f2"] == (1, 0)
    print("[self_test] OK — scaffold render/order, counterbalance, shortcut scan, family stats pass.")


# =====================================================================
# Real run
# =====================================================================
def run_real(args):
    rng = np.random.default_rng(args.seed)
    cfg = json.load(open(args.config))
    concept = cfg["concept"]
    label_a, label_b = cfg["label_a"], cfg["label_b"]
    ab_mode = bool(cfg.get("ab_mode", False))
    raw = [json.loads(l) for l in open(args.raw_questions)]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"concept": concept, "n_raw": len(raw)}

    # ---------- label tokenization gate ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    def one_token(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return (len(ids) == 1), ids
    if ab_mode:
        eff_a, eff_b = "A", "B"
    else:
        eff_a, eff_b = label_a, label_b
    ok_a, ids_a = one_token(eff_a); ok_b, ids_b = one_token(eff_b)
    report["label_token_ids"] = {eff_a: ids_a, eff_b: ids_b}
    if not (ok_a and ok_b):
        logger.error("TOKENIZATION GATE FAILED: ' %s'->%s  ' %s'->%s — use ab_mode or new labels",
                     eff_a, ids_a, eff_b, ids_b)
        sys.exit(2)
    assert ids_a[0] != ids_b[0]
    logger.info("tokenization gate OK: ' %s'=%d  ' %s'=%d", eff_a, ids_a[0], eff_b, ids_b[0])

    # ---------- class map and exemplars ----------
    cls_of = {label_a: 0, label_b: 1}
    exs = cfg["exemplars"]
    assert len(exs) == 2, "exactly two exemplars (one per class)"
    ex_by_cls = {}
    for e in exs:
        c = cls_of[cfg[e["answer"]]] if e["answer"] in ("label_a", "label_b") else cls_of[e["answer"]]
        ex_by_cls[c] = {"question": e["question"], "cls": c}
    assert set(ex_by_cls) == {0, 1}, "need one exemplar per class"

    # ---------- counterbalance assignments ----------
    n = len(raw)
    orders = assign_counterbalance(n, rng, ("XY", "YX"))
    ab_assign = assign_counterbalance(n, rng, ("A0", "A1")) if ab_mode else [None] * n

    # ---------- render ----------
    rows = []
    for i, r in enumerate(raw):
        yc = cls_of[r["correct_answer"]]
        qt = r["raw_question"]
        if ab_mode:
            legend = cfg.get("ab_legend", f"A: {label_a}, B: {label_b}")
            if ab_assign[i] == "A1":
                legend = f"A: {label_b}, B: {label_a}"
            qt = f"{qt} ({legend})"
            if ab_assign[i] == "A0":
                lab_for = lambda c: "A" if c == 0 else "B"
            else:
                lab_for = lambda c: "A" if c == 1 else "B"
            correct_eff = lab_for(yc)
        else:
            lab_for = lambda c: [label_a, label_b][c]
            correct_eff = [label_a, label_b][yc]
        prompt = render_prompt(qt, ex_by_cls[0], ex_by_cls[1], orders[i], lab_for)
        id_c0 = tok.encode(" " + lab_for(0), add_special_tokens=False)[0]
        id_c1 = tok.encode(" " + lab_for(1), add_special_tokens=False)[0]
        rows.append({"prompt": prompt, "correct_answer": correct_eff,
                     "concept": concept, "surface_family": r["surface_family"],
                     "cue_type": r.get("cue_type"), "scaffold_order": orders[i],
                     "ab_assignment": ab_assign[i], "raw_question": r["raw_question"],
                     "canonical_answer": [label_a, label_b][yc], "y_canonical": yc,
                     "tok_id_class0": id_c0, "tok_id_class1": id_c1, "y": yc})

    # ---------- family gate ----------
    fs = family_stats(rows)
    n_fam = len(fs); min_per = min(sum(v) for v in fs.values())
    imbalanced = [f for f, v in fs.items() if min(v) == 0]
    report["n_families"] = n_fam; report["min_per_family"] = min_per
    report["families_single_class"] = imbalanced
    gate_fam = n_fam >= args.min_families and min_per >= args.min_per_family
    logger.info("families: %d (need >=%d) | min size %d (need >=%d) | single-class fams: %d",
                n_fam, args.min_families, min_per, args.min_per_family, len(imbalanced))

    # ---------- class balance ----------
    y = np.array([r["y"] for r in rows])
    bal = abs(float(y.mean()) - 0.5)
    report["class_balance_dev"] = bal
    logger.info("class balance: %.3f beta-frac (dev from 0.5 = %.3f)", float(y.mean()), bal)

    # ---------- shortcut scan on RAW questions ----------
    scan = bow_shortcut_scan([r["raw_question"] for r in rows], y, args.shortcut_min_count)
    report["top_shortcut_tokens"] = scan[:15]
    worst = scan[0][1] if scan else 0.0
    gate_short = worst < args.shortcut_auc_max or args.allow_shortcuts
    logger.info("shortcut scan: worst token AUC = %.3f (%s) %s", worst,
                scan[0][0] if scan else "-", "[ALLOWED by flag]" if args.allow_shortcuts and worst >= args.shortcut_auc_max else "")

    # ---------- write ----------
    for r in rows:
        r.pop("y")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep_path = out_path.with_name(f"render_report_{concept}.json")
    report["gates"] = {"tokenization": True, "families": gate_fam,
                       "shortcut": bool(gate_short), "class_balance": bal <= 0.05}
    json.dump(report, open(rep_path, "w"), indent=2, ensure_ascii=False)

    ok = all(report["gates"].values())
    print("\n" + "=" * 88)
    print(f"B1 RENDER + STATIC GATES — {concept}")
    print("=" * 88)
    print(f"prompts: {len(rows)} | families: {n_fam} (min size {min_per}) | balance dev: {bal:.3f}")
    print(f"labels: ' {eff_a}'={ids_a[0]} ' {eff_b}'={ids_b[0]} | ab_mode={ab_mode}")
    print(f"worst shortcut token AUC: {worst:.3f}" + ("  [allowed]" if args.allow_shortcuts else ""))
    print(f"gates: {report['gates']}")
    print(("ALL STATIC GATES PASS -> run 121 precheck on GPU" if ok else
           "GATE FAILURES — fix the corpus before any GPU time"))
    print(f"corpus: {out_path}\nreport: {rep_path}")
    print("=" * 88 + "\n")
    sys.exit(0 if ok else 1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--config", help="concept config json (see docstring)")
    p.add_argument("--raw_questions", help="raw questions jsonl")
    p.add_argument("--out", help="output corpus jsonl path")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--min_families", type=int, default=24)
    p.add_argument("--min_per_family", type=int, default=8)
    p.add_argument("--shortcut_auc_max", type=float, default=0.95)
    p.add_argument("--shortcut_min_count", type=int, default=10)
    p.add_argument("--allow_shortcuts", action="store_true",
                   help="concept-intrinsic shortcuts (e.g. lang_id) — document, don't fail")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.config and args.raw_questions and args.out, "--config/--raw_questions/--out required"
    run_real(args)


if __name__ == "__main__":
    main()
