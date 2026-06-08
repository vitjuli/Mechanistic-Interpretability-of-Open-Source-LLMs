"""
93_prediction_sanity.py   [which alpha/beta predictions are real? resolve CSV vs script-92 mismatch]
=====================================================================================================
Problem. The CSV baseline (baseline_logit_diff) implies ~113 beta-errors (beta-recall 0.58) with a
nucleon-framed failure mode, but script 92 (left-padded batched forward) found 149 beta-errors
(beta-recall 0.446) with a lepton/other-framed failure mode. Same prompts, same convention
(alpha iff logit_alpha > logit_beta). Before building any intervention we must know which prediction
is correct. Prime suspect: left-padding without corrected position_ids corrupts RoPE for the real
tokens, scrambling predictions in batched runs.

This script computes, for every prompt, the answer margin  m = logit_alpha - logit_beta  THREE ways:
  (1) single   : batch size 1, NO padding                         <- ground truth
  (2) leftpad  : left-padded batch, position_ids NOT supplied      <- replicates script 92
  (3) leftpad_posfix : left-padded batch, position_ids = cumsum(attn_mask)-1  <- the fix
and reports, for each:
  * beta-recall and overall accuracy
  * the nucleon/lepton/other framing split among failed-beta
  * per-prompt sign-agreement and correlation vs the single (truth) and vs the CSV baseline
  * how many prompts flip class between setups
This pins down (i) whether padding changed predictions, (ii) whether the CSV matches the clean
single-prompt run (=> our 0.13/0.97 framing result is robust), and (iii) the canonical setup to use.

SELF-TEST (no torch):  python 93_prediction_sanity.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pred_sanity")


def S(x):
    return str(x) if x is not None else "NA"


def framing_of(p):
    rt = S(p.get("relation_type")).lower(); cr = S(p.get("concept_route")).lower()
    if any(k in rt for k in ["neutron_to_proton", "n_to_p", "quark", "antineutrino", "full_beta"]):
        return "nucleon"
    if any(k in cr for k in ["lepton", "electron", "muon"]) or "weak_force" in rt:
        return "lepton"
    return "other"


def posfix_from_mask(mask):
    """position_ids for left-padded sequences: cumsum(mask)-1, pads set to 1 (numpy ref for self-test)."""
    pos = np.cumsum(mask, axis=-1) - 1
    pos[mask == 0] = 1
    return pos


def sign_agree(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.sign(a) == np.sign(b)))


def self_test():
    # position_ids logic for left padding
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    pos = posfix_from_mask(mask)
    assert pos[0].tolist() == [1, 1, 0, 1, 2], pos[0].tolist()
    assert pos[1].tolist() == [1, 0, 1, 2, 3], pos[1].tolist()
    # framing classifier
    assert framing_of({"relation_type": "neutron_to_proton"}) == "nucleon"
    assert framing_of({"concept_route": "lepton_family"}) == "lepton"
    assert framing_of({"relation_type": "charge_plus_z_change"}) == "other"
    # sign agreement
    assert sign_agree([1, -1, 1], [1, -1, -1]) == 2 / 3
    print("[self_test] OK — position_ids fix, framing classifier, sign-agreement all pass.")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y_true = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])  # beta=1
    fr = np.array([framing_of(p) for p in prompts])
    logger.info("%d prompts; alpha_id=%d beta_id=%d", nP, a_id, b_id)

    def margin_single():
        m = np.zeros(nP)
        with torch.no_grad():
            for i, p in enumerate(prompts):
                enc = tok([p["prompt"]], return_tensors="pt").to(args.device)  # no padding, bs=1
                lo = model(**enc, use_cache=False).logits[0, -1, :]
                m[i] = float(lo[a_id] - lo[b_id])
                if (i + 1) % 150 == 0:
                    logger.info("  single %d/%d", i + 1, nP)
        return m

    def margin_batched(posfix):
        tok.padding_side = "left"
        m = np.zeros(nP); bs = args.batch_size
        with torch.no_grad():
            for s in range(0, nP, bs):
                chunk = [p["prompt"] for p in prompts[s: s + bs]]
                enc = tok(chunk, return_tensors="pt", padding=True).to(args.device)
                kw = {}
                if posfix:
                    am = enc["attention_mask"]
                    pos = am.long().cumsum(-1) - 1
                    pos = pos.masked_fill(am == 0, 1)
                    kw["position_ids"] = pos
                lo = model(**enc, use_cache=False, **kw).logits[:, -1, :]
                m[s: s + len(chunk)] = (lo[:, a_id] - lo[:, b_id]).float().cpu().numpy()
        return m

    logger.info("(1) single no-pad ...");           m_single = margin_single()
    logger.info("(2) left-pad batched (no posfix; replicates 92) ..."); m_lp = margin_batched(False)
    logger.info("(3) left-pad batched + posfix ..."); m_lpf = margin_batched(True)

    # optional CSV comparison
    csv_m = None
    if args.csv and Path(args.csv).exists():
        cr = list(_csv.DictReader(open(args.csv)))
        if len(cr) == nP:
            csv_m = np.array([float(r["baseline_logit_diff"]) for r in cr])
            logger.info("loaded CSV baseline (%d rows) for comparison", len(cr))

    def report(name, m):
        pred_alpha = m > 0
        failed = (y_true == 1) & pred_alpha
        brecall = (~pred_alpha & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
        acc = np.mean(pred_alpha == (y_true == 0))
        split = {f: int(((fr == f) & failed).sum()) for f in ["nucleon", "lepton", "other"]}
        return dict(name=name, acc=float(acc), beta_recall=float(brecall),
                    n_failed_beta=int(failed.sum()),
                    failed_nucleon=split["nucleon"], failed_lepton=split["lepton"], failed_other=split["other"],
                    agree_vs_single=sign_agree(m, m_single),
                    corr_vs_single=float(np.corrcoef(m, m_single)[0, 1]),
                    agree_vs_csv=(sign_agree(m, csv_m) if csv_m is not None else float("nan")),
                    corr_vs_csv=(float(np.corrcoef(m, csv_m)[0, 1]) if csv_m is not None else float("nan")))

    recs = [report("single_nopad", m_single), report("leftpad_naive(=92)", m_lp), report("leftpad_posfix", m_lpf)]
    if csv_m is not None:
        recs.append(report("CSV_baseline", csv_m))

    with open(out / "prediction_sanity.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); [w.writerow(r) for r in recs]

    # flips between single and leftpad-naive
    flip = int(np.sum(np.sign(m_single) != np.sign(m_lp)))
    flip_f = int(np.sum(np.sign(m_lpf) != np.sign(m_single)))

    print("\n" + "=" * 104)
    print("PREDICTION SANITY — does padding change the model's answer? which run matches the CSV?")
    print("=" * 104)
    print(f"{'setup':>20} | {'acc':>5} {'β-recall':>8} {'failedβ':>7} | "
          f"{'fail:nuc/lep/oth':>16} | {'agree vs single':>15} {'corr':>5} | {'agree vs CSV':>12} {'corr':>5}")
    for r in recs:
        print(f"{r['name']:>20} | {r['acc']:>5.3f} {r['beta_recall']:>8.3f} {r['n_failed_beta']:>7d} | "
              f"{r['failed_nucleon']:>4d}/{r['failed_lepton']:>3d}/{r['failed_other']:>3d}    | "
              f"{r['agree_vs_single']:>15.3f} {r['corr_vs_single']:>5.2f} | "
              f"{r['agree_vs_csv']:>12.3f} {r['corr_vs_csv']:>5.2f}")
    print("-" * 104)
    print(f"class flips: single vs leftpad_naive = {flip}/{nP};  single vs leftpad_posfix = {flip_f}/{nP}")
    truth = recs[0]
    lp = recs[1]
    if flip > 0.05 * nP and flip_f < flip:
        print(f"=> LEFT-PADDING CORRUPTED predictions ({flip} flips), and corrected position_ids fixes it "
              f"({flip_f} flips). Script 92's groups are contaminated; use the SINGLE no-pad run as canonical.")
    elif flip <= 0.05 * nP:
        print("=> Padding did NOT materially change predictions; the CSV/script-92 mismatch is from a different "
              "cause (different prompt construction or measurement in the CSV).")
    if csv_m is not None:
        print(f"=> CSV vs single: sign-agreement {truth['name']}->{recs[-1]['agree_vs_single']:.3f}, corr {recs[-1]['corr_vs_single']:.2f}. "
              f"{'CSV matches the clean run -> the 0.13/0.97 framing result is on solid ground.' if recs[-1]['agree_vs_single']>0.9 else 'CSV does NOT match the clean run -> the framing split must be recomputed on the canonical run before use.'}")
    print(f"Canonical failed-β framing split (single no-pad): nucleon={truth['failed_nucleon']} "
          f"lepton={truth['failed_lepton']} other={truth['failed_other']}.")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--csv", default="per_prompt_cluster_effect_matrix.csv", help="CSV with baseline_logit_diff (optional)")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/prediction_sanity")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--batch_size", type=int, default=16)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
