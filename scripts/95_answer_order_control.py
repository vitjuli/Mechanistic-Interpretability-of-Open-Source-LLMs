"""
95_answer_order_control.py    [is the alpha-default a first-option bias, or content?]
=====================================================================================
Every prompt asks "...Is the decay type alpha or beta?" -- alpha is always listed FIRST, and the
model defaults to alpha (alpha-recall 0.996, beta-recall 0.446). Before attributing this to a
content shortcut we must rule out a first-option / primacy bias. This script recomputes the REAL
margin (single, no padding) with the option order swapped to "beta or alpha" and compares.

For each prompt: original ("alpha or beta") vs swapped ("beta or alpha"). Report alpha/beta recall
under each, prediction flips, and the mean shift of (logit_alpha - logit_beta). Convention >0 = alpha.

Interpretation:
  - if under "beta or alpha" the bias FLIPS (beta-recall jumps, alpha-recall drops, margins shift
    negative) -> the alpha-default is largely POSITIONAL (primacy), not a physics shortcut.
  - if recalls and margins are ~unchanged -> the alpha-default is CONTENT/token-prior driven
    (alpha is the model's default decay regardless of option order). Either way it informs the
    intervention target.

SELF-TEST (no torch):  python 95_answer_order_control.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("order_ctrl")

SWAP = re.compile(r"alpha or beta", re.IGNORECASE)


def swap_order(text):
    """Swap 'alpha or beta' -> 'beta or alpha'. Returns (new_text, did_swap)."""
    new, n = SWAP.subn("beta or alpha", text)
    return new, n > 0


def self_test():
    t = "A nucleus undergoes a decay process. An electron is emitted. Is the decay type alpha or beta?"
    s, did = swap_order(t)
    assert did and "beta or alpha" in s and "alpha or beta" not in s.lower(), s
    t2 = "no options here"
    s2, did2 = swap_order(t2)
    assert (not did2) and s2 == t2
    print("[self_test] OK — order swap works and no-op when absent.")


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
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])  # beta=1

    def margin(text):
        enc = tok([text], return_tensors="pt").to(args.device)
        lo = model(**enc, use_cache=False).logits[0, -1, :]
        return float(lo[a_id] - lo[b_id])

    m_orig = np.zeros(nP); m_swap = np.zeros(nP); swapped = np.zeros(nP, bool)
    logger.info("computing original vs order-swapped margins over %d prompts...", nP)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            m_orig[i] = margin(p["prompt"])
            sw, did = swap_order(p["prompt"]); swapped[i] = did
            m_swap[i] = margin(sw) if did else m_orig[i]
            if (i + 1) % 150 == 0:
                logger.info("  %d/%d", i + 1, nP)

    def recalls(m):
        pa = m > 0
        return (float(np.mean(pa[y == 0])),                       # alpha-recall
                float(np.mean(~pa[y == 1])),                      # beta-recall
                float(np.mean(pa == (y == 0))))                   # overall acc

    a_o, b_o, acc_o = recalls(m_orig)
    a_s, b_s, acc_s = recalls(m_swap)
    flips = int(np.sum((m_orig > 0) != (m_swap > 0)))
    mean_shift = float(np.mean(m_swap - m_orig))                  # negative => swap pushed toward beta
    # how many prompts that were predicted alpha now flip to beta after swap?
    was_alpha = m_orig > 0
    alpha_to_beta = int(np.sum(was_alpha & (m_swap <= 0)))
    beta_to_alpha = int(np.sum((~was_alpha) & (m_swap > 0)))

    rec = dict(n=nP, n_swapped=int(swapped.sum()),
               acc_orig=acc_o, alpha_recall_orig=a_o, beta_recall_orig=b_o,
               acc_swap=acc_s, alpha_recall_swap=a_s, beta_recall_swap=b_s,
               pred_flips=flips, alpha_to_beta=alpha_to_beta, beta_to_alpha=beta_to_alpha,
               mean_margin_shift=mean_shift)
    with open(out / "answer_order_control.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rec.keys())); w.writeheader(); w.writerow(rec)

    print("\n" + "=" * 80)
    print("ANSWER-ORDER CONTROL — is the alpha-default positional?")
    print("=" * 80)
    print(f"  prompts swapped: {rec['n_swapped']}/{nP}")
    print(f"  {'phrasing':>16} | {'acc':>5} | {'alpha-recall':>12} | {'beta-recall':>11}")
    print(f"  {'alpha or beta':>16} | {acc_o:>5.3f} | {a_o:>12.3f} | {b_o:>11.3f}")
    print(f"  {'beta or alpha':>16} | {acc_s:>5.3f} | {a_s:>12.3f} | {b_s:>11.3f}")
    print(f"  prediction flips on swap: {flips}/{nP}  (alpha->beta {alpha_to_beta}, beta->alpha {beta_to_alpha})")
    print(f"  mean shift of (logit_alpha - logit_beta) on swap: {mean_shift:+.3f}  (negative => swap favors beta)")
    print("-" * 80)
    db = b_s - b_o
    if db > 0.2 and mean_shift < -0.2:
        print(f"=> The alpha-default is LARGELY POSITIONAL: swapping order raised beta-recall by {db:+.2f} and "
              f"shifted margins toward beta. The 'alpha-default shortcut' is (partly) a first-option bias -> the "
              f"intervention target and narrative must account for this; report order-balanced accuracy.")
    elif abs(db) < 0.1 and abs(mean_shift) < 0.15:
        print("=> The alpha-default is NOT positional: order swap barely changes anything -> alpha is the model's "
              "content/token-prior default. The surface-phrasing shortcut story stands; proceed (order-robust).")
    else:
        print(f"=> PARTIAL positional effect (Δbeta-recall {db:+.2f}, margin shift {mean_shift:+.2f}): some primacy "
              f"plus content. Use order-balanced prompts (average over both orders) as canonical from here on.")
    print("=" * 80 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/answer_order")
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
