"""
121_b1_precheck.py   [B1 precheck gate (spec §2) — forward-only, ~5 GPU-min]
=============================================================================
One forward pass per prompt on the RENDERED corpus (output of 120). Computes
and gates, per spec §2:

  intact-baseline   top-1 token in the prompt's two class tokens     >= 0.90
  margin accuracy   sign(logit[correct] - logit[other]) correct      >= 0.65
  per-class recall  each canonical class                             >= 0.40
  order effects     |acc(XY) - acc(YX)| and |acc(A0) - acc(A1)|      <= 0.10

Writes precheck_{concept}.json next to the corpus and exits non-zero on any
gate failure, so the sbatch chain (121 && 119 && ...) stops before paying for
capture. Uses per-prompt class token ids (tok_id_class0/1) — A/B-fallback safe.

SELF-TEST (no torch / no repo):  python 121_b1_precheck.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precheck121")


# =====================================================================
# Pure-numpy metric core (exercised by --self_test)
# =====================================================================
def precheck_metrics(top1, logit_c0, logit_c1, id0, id1, y, groups):
    """top1: (n,) argmax ids; logit_c0/1: logits of the two class tokens;
    id0/id1: per-prompt class token ids; y: canonical class; groups: dict
    name -> (n,) labels for order-effect splits. Returns metrics dict."""
    top1 = np.asarray(top1); y = np.asarray(y, int)
    intact = ((top1 == np.asarray(id0)) | (top1 == np.asarray(id1))).astype(float)
    margin_correct = np.where(y == 1, logit_c1 - logit_c0, logit_c0 - logit_c1)
    correct = (margin_correct > 0).astype(float)
    out = {"intact_baseline": float(intact.mean()),
           "margin_accuracy": float(correct.mean()),
           "recall_class0": float(correct[y == 0].mean()) if (y == 0).any() else float("nan"),
           "recall_class1": float(correct[y == 1].mean()) if (y == 1).any() else float("nan"),
           "intact_and_correct": float((intact * correct).mean())}
    for name, g in groups.items():
        g = np.asarray(g)
        vals = sorted(set(g[g != None].tolist())) if g.dtype == object else sorted(set(g.tolist()))
        if len(vals) == 2:
            a = float(correct[g == vals[0]].mean()); b = float(correct[g == vals[1]].mean())
            out[f"order_effect_{name}"] = abs(a - b)
            out[f"acc_{name}_{vals[0]}"] = a; out[f"acc_{name}_{vals[1]}"] = b
    return out


def gate(metrics, thr_intact=0.90, thr_acc=0.65, thr_recall=0.40, thr_order=0.10):
    g = {"intact": metrics["intact_baseline"] >= thr_intact,
         "accuracy": metrics["margin_accuracy"] >= thr_acc,
         "recall": min(metrics["recall_class0"], metrics["recall_class1"]) >= thr_recall}
    for k in list(metrics):
        if k.startswith("order_effect_"):
            g[k] = metrics[k] <= thr_order
    return g


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    n = 400
    y = (np.arange(n) % 2)
    id0 = np.full(n, 11); id1 = np.full(n, 22)
    # 90% of prompts answer with the correct class token, 8% wrong class, 2% junk
    top1 = np.where(y == 1, 22, 11).astype(int)
    wrong = rng.choice(n, 32, replace=False); top1[wrong] = np.where(y[wrong] == 1, 11, 22)
    junk = rng.choice(np.setdiff1d(np.arange(n), wrong), 8, replace=False); top1[junk] = 99
    lc1 = np.where(top1 == 22, 2.0, -2.0) + 0.01 * rng.standard_normal(n)
    lc0 = -lc1
    groups = {"scaffold": np.array(["XY", "YX"] * (n // 2))}
    m = precheck_metrics(top1, lc0, lc1, id0, id1, y, groups)
    assert abs(m["intact_baseline"] - (n - 8) / n) < 1e-9
    assert 0.85 < m["margin_accuracy"] < 0.95
    assert m["order_effect_scaffold"] < 0.1
    g = gate(m)
    assert g["intact"] and g["accuracy"] and g["recall"] and g["order_effect_scaffold"]
    # failing corpus: one class never correct
    lc1_bad = np.where(y == 1, -2.0, 2.0)
    m2 = precheck_metrics(top1, -lc1_bad, lc1_bad, id0, id1, y, groups)
    assert not gate(m2)["recall"], "broken class must fail the recall gate"
    print("[self_test] OK — metric core, gates, order effects, failure detection pass.")


# =====================================================================
# Real run
# =====================================================================
def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [json.loads(l) for l in open(args.corpus)]
    nP = len(prompts)
    assert all("tok_id_class0" in p for p in prompts), "corpus must come from 120 (per-prompt ids)"
    concept = prompts[0].get("concept", "unknown")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()

    id0 = np.array([int(p["tok_id_class0"]) for p in prompts])
    id1 = np.array([int(p["tok_id_class1"]) for p in prompts])
    y = np.array([int(p["y_canonical"]) for p in prompts])
    top1 = np.zeros(nP, int); lc0 = np.zeros(nP); lc1 = np.zeros(nP)
    logger.info("precheck %s: %d prompts, forward-only", concept, nP)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
            row = model(**inp, use_cache=False).logits[0, -1, :].float()
            top1[i] = int(row.argmax().item())
            lc0[i] = float(row[id0[i]]); lc1[i] = float(row[id1[i]])
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d", i + 1, nP)

    groups = {"scaffold": np.array([p.get("scaffold_order") for p in prompts], dtype=object)}
    if any(p.get("ab_assignment") for p in prompts):
        groups["ab"] = np.array([p.get("ab_assignment") for p in prompts], dtype=object)
    m = precheck_metrics(top1, lc0, lc1, id0, id1, y, groups)
    g = gate(m, args.thr_intact, args.thr_acc, args.thr_recall, args.thr_order)

    # top-1 vocabulary report (catches scaffold failures immediately)
    from collections import Counter
    cnt = Counter(int(t) for t in top1)
    top_tokens = [(tid, tok.decode([tid]), c) for tid, c in cnt.most_common(8)]

    out = Path(args.corpus).with_name(f"precheck_{concept}.json")
    json.dump({"concept": concept, "n": nP, "metrics": m, "gates": g,
               "top1_tokens": [{"id": t, "text": s, "count": c} for t, s, c in top_tokens],
               "thresholds": {"intact": args.thr_intact, "acc": args.thr_acc,
                              "recall": args.thr_recall, "order": args.thr_order}},
              open(out, "w"), indent=2, ensure_ascii=False)

    ok = all(g.values())
    print("\n" + "=" * 88)
    print(f"B1 PRECHECK — {concept}")
    print("=" * 88)
    print(f"intact-baseline: {m['intact_baseline']:.3f} | margin acc: {m['margin_accuracy']:.3f} | "
          f"recall c0/c1: {m['recall_class0']:.3f}/{m['recall_class1']:.3f}")
    for k in m:
        if k.startswith("order_effect_"):
            print(f"{k}: {m[k]:.3f}")
    print("top-1 tokens:", ", ".join(f"{repr(s)}x{c}" for _, s, c in top_tokens))
    print(f"gates: {g}")
    print(("PRECHECK PASS -> run 119 capture" if ok else "PRECHECK FAIL — fix the corpus first"))
    print(f"report: {out}")
    print("=" * 88 + "\n")
    sys.exit(0 if ok else 1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", help="rendered corpus jsonl from 120")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--thr_intact", type=float, default=0.90)
    p.add_argument("--thr_acc", type=float, default=0.65)
    p.add_argument("--thr_recall", type=float, default=0.40)
    p.add_argument("--thr_order", type=float, default=0.10)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.corpus, "--corpus required"
    run_real(args)


if __name__ == "__main__":
    main()
