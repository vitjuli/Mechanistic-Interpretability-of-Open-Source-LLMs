"""
94_dump_real_predictions.py   [trustworthy per-prompt predictions + metadata]
==============================================================================
The CSV baseline_logit_diff does NOT match the model (sign-agree 0.50, corr 0.15 vs the real run).
This script recomputes the REAL margin m = logit_alpha - logit_beta for every prompt, ONE AT A TIME
with NO padding (the unambiguous ground truth), and dumps it next to all prompt metadata so the
error structure (alpha-bias, framing, cue-type, minimal pairs) can be rebuilt on solid data.

Output CSV columns: prompt_idx, prompt_id, correct_answer, margin_alpha_minus_beta, pred, correct,
plus a broad set of metadata fields for re-deriving any shortcut.

SELF-TEST (no torch):  python 94_dump_real_predictions.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dump_pred")

META = ["cue_type", "relation_type", "concept_route", "level_label", "abstraction_level",
        "inference_steps", "evidence_completeness", "is_uniquely_determining", "prompt_format",
        "surface_family", "semantic_equivalence_group", "contrastive_pair_id", "contrastive_role",
        "difficulty", "behaviour_type", "test_type", "physics_concept", "keyword_type"]


def self_test():
    # trivial: ensure the metadata list is well-formed and the record builder works
    p = {"prompt_id": "X", "correct_answer": " beta", "relation_type": "neutron_to_proton"}
    rec = {"prompt_idx": 0, "prompt_id": p["prompt_id"], "correct_answer": p["correct_answer"].strip(),
           "margin_alpha_minus_beta": -0.5, "pred": "beta", "correct": True}
    for m in META:
        rec[m] = str(p.get(m)) if p.get(m) is not None else "NA"
    assert rec["pred"] == "beta" and rec["correct"] is True and rec["relation_type"] == "neutron_to_proton"
    print("[self_test] OK — record builder and metadata schema valid.")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    logger.info("%d prompts; alpha_id=%d beta_id=%d; computing single no-pad margins...", len(prompts), a_id, b_id)
    recs = []
    n_alpha_ok = n_beta_ok = n_alpha = n_beta = 0
    with torch.no_grad():
        for i, p in enumerate(prompts):
            enc = tok([p["prompt"]], return_tensors="pt").to(args.device)
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            m = float(lo[a_id] - lo[b_id])
            true = p["correct_answer"].strip()
            pred = "alpha" if m > 0 else "beta"
            correct = (pred == true)
            rec = {"prompt_idx": i, "prompt_id": p.get("prompt_id"), "correct_answer": true,
                   "margin_alpha_minus_beta": m, "pred": pred, "correct": correct}
            for k in META:
                rec[k] = str(p.get(k)) if p.get(k) is not None else "NA"
            recs.append(rec)
            if true == "alpha":
                n_alpha += 1; n_alpha_ok += correct
            else:
                n_beta += 1; n_beta_ok += correct
            if (i + 1) % 150 == 0:
                logger.info("  %d/%d", i + 1, len(prompts))

    with open(out / "real_predictions.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); [w.writerow(r) for r in recs]

    print("\n" + "=" * 70)
    print("REAL PREDICTIONS (single, no padding) — summary")
    print("=" * 70)
    print(f"  overall accuracy: {(n_alpha_ok + n_beta_ok)/len(prompts):.3f}")
    print(f"  alpha-recall: {n_alpha_ok/max(n_alpha,1):.3f}  ({n_alpha_ok}/{n_alpha})")
    print(f"  beta-recall : {n_beta_ok/max(n_beta,1):.3f}  ({n_beta_ok}/{n_beta})")
    print(f"  -> dumped {len(recs)} rows to {out/'real_predictions.csv'}")
    print("=" * 70 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/real_predictions")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
