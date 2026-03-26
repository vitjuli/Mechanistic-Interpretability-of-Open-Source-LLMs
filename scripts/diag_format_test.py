"""
diag_format_test.py — Format diagnostics for physics_conservation behaviour.

For each test concept × template variant, shows:
  - Top-10 next tokens with probabilities at the prediction position
  - P(" True") and P(" False") explicitly
  - logprob_diff = log P(correct) - log P(incorrect)

Goal: identify which prompt format reliably puts True/False in the top
probability mass for both conservative AND non-conservative concepts.

Usage:
    python scripts/diag_format_test.py
    python scripts/diag_format_test.py --top_k 20 --model_size 4b
"""
import argparse
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Test cases: (label, concept_family, description, note)
# ---------------------------------------------------------------------------
TEST_CONCEPTS = [
    # Conservative
    (True,  "conservative_named",    "a uniform gravitational field",
     "easy — should be strongly True"),
    (True,  "conservative_abstract", "a static force field with zero curl everywhere",
     "medium — should be True"),
    # Non-conservative named  ← failing in pilot
    (False, "nonconservative_named", "kinetic friction",
     "easy — should be strongly False; fails T0/T3/T5"),
    (False, "nonconservative_named", "air resistance",
     "easy — should be False; fails T0/T3/T5"),
    # Non-conservative abstract
    (False, "nonconservative_abstract", "a force field with nonzero curl",
     "medium — should be False"),
    # Adversarial  ← expected to be hard
    (False, "adversarial_divergence", "a force field with zero divergence everywhere",
     "hard adversarial — correct=False but model may say True"),
]

def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s

# ---------------------------------------------------------------------------
# Format variants to test (name → template function)
# Each function takes (desc, tidx) where tidx in {0, 3, 4, 5}
# ---------------------------------------------------------------------------
FORMATS = {
    "v1_colon":     lambda d, t: {
        0: f"The work done by {d} is path-independent:",
        3: f"{_cap(d)} can be expressed as the negative gradient of a scalar function:",
        4: f"{_cap(d)} is a conservative force:",
        5: f"The circulation integral of {d} around any closed loop is zero:",
    }[t],

    "v2_TF_period": lambda d, t: {
        0: f"True or False? The work done by {d} is path-independent.",
        3: f"True or False? {_cap(d)} can be expressed as the negative gradient of a scalar function.",
        4: f"True or False? {_cap(d)} is a conservative force.",
        5: f"True or False? The circulation integral of {d} around any closed loop is zero.",
    }[t],

    "v3_TF_answer": lambda d, t: {
        0: f"True or False? The work done by {d} is path-independent. Answer:",
        3: f"True or False? {_cap(d)} can be expressed as the negative gradient of a scalar function. Answer:",
        4: f"True or False? {_cap(d)} is a conservative force. Answer:",
        5: f"True or False? The circulation integral of {d} around any closed loop is zero. Answer:",
    }[t],

    "v4_stmt_TF":   lambda d, t: {
        0: f"The work done by {d} is path-independent. True or False?",
        3: f"{_cap(d)} can be expressed as the negative gradient of a scalar function. True or False?",
        4: f"{_cap(d)} is a conservative force. True or False?",
        5: f"The circulation integral of {d} around any closed loop is zero. True or False?",
    }[t],

    "v5_this_stmt": lambda d, t: {
        0: f"The work done by {d} is path-independent. This statement is",
        3: f"{_cap(d)} can be expressed as the negative gradient of a scalar function. This statement is",
        4: f"{_cap(d)} is a conservative force. This statement is",
        5: f"The circulation integral of {d} around any closed loop is zero. This statement is",
    }[t],

    "v6_yn_answer": lambda d, t: {
        0: f"Is the work done by {d} path-independent? Answer:",
        3: f"Can {d} be expressed as the negative gradient of a scalar function? Answer:",
        4: f"Is {d} a conservative force? Answer:",
        5: f"Is the circulation integral of {d} around any closed loop equal to zero? Answer:",
    }[t],
}

# Templates to test
TEMPLATE_INDICES = [0, 3, 4, 5]

TEMPLATE_NAMES = {
    0: "T0 path-indep",
    3: "T3 gradient",
    4: "T4 conservative-label",
    5: "T5 circulation",
}


def load_model(model_name: str, device: str):
    print(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Loading model on {device}...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    print("Model loaded.\n")
    return tok, model


@torch.no_grad()
def get_next_token_probs(prompt: str, tok, model, device: str, top_k: int = 10):
    """Return top-k next token probabilities and specific token logprobs."""
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    out = model(ids)
    logits = out.logits[0, -1, :]   # shape: (vocab,)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Top-k
    topk = torch.topk(log_probs, k=top_k)
    top_tokens = [(tok.decode([i.item()]), lp.item())
                  for i, lp in zip(topk.indices, topk.values)]

    # Specific tokens
    true_id  = tok.encode(" True",  add_special_tokens=False)
    false_id = tok.encode(" False", add_special_tokens=False)

    def _lp(ids_list):
        if len(ids_list) == 1:
            return log_probs[ids_list[0]].item()
        return None  # multi-token — can't measure at single position

    lp_true  = _lp(true_id)
    lp_false = _lp(false_id)

    return top_tokens, lp_true, lp_false


def run_diagnostics(tok, model, device: str, top_k: int, formats_to_run: list):
    sep = "=" * 80

    for label, family, desc, note in TEST_CONCEPTS:
        correct_tok   = " True"  if label else " False"
        incorrect_tok = " False" if label else " True"

        print(f"\n{sep}")
        print(f"CONCEPT: {desc}")
        print(f"Family:  {family}  |  Label: {label}  |  Correct: '{correct_tok}'")
        print(f"Note:    {note}")
        print(sep)

        for tidx in TEMPLATE_INDICES:
            tname = TEMPLATE_NAMES[tidx]
            print(f"\n  ── {tname} ──")

            # Header row
            hdr = f"  {'FORMAT':<16}  {'lp_correct':>10}  {'lp_incorr':>10}  {'diff':>7}  {'rank_correct':>12}  TOP-5 tokens"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            for fmt_name in formats_to_run:
                fmt_fn = FORMATS[fmt_name]
                prompt = fmt_fn(desc, tidx)
                top_tokens, lp_true, lp_false = get_next_token_probs(
                    prompt, tok, model, device, top_k=top_k
                )

                lp_correct   = lp_true  if label else lp_false
                lp_incorrect = lp_false if label else lp_true

                if lp_correct is not None and lp_incorrect is not None:
                    diff = lp_correct - lp_incorrect
                    diff_str = f"{diff:+.3f}"
                    pass_fail = "✓" if diff > 0.5 else ("~" if diff > 0 else "✗")
                else:
                    diff_str = "  n/a "
                    pass_fail = "?"

                # Rank of correct token in top-k
                correct_str = correct_tok
                rank = next((i+1 for i, (t, _) in enumerate(top_tokens)
                             if t == correct_str), f">{top_k}")

                lp_c_str = f"{lp_correct:.3f}" if lp_correct is not None else "  n/a"
                lp_i_str = f"{lp_incorrect:.3f}" if lp_incorrect is not None else "  n/a"

                top5 = "  ".join(f"'{t}'({lp:.2f})" for t, lp in top_tokens[:5])

                print(f"  {fmt_name:<16}  {lp_c_str:>10}  {lp_i_str:>10}  "
                      f"{diff_str:>7} {pass_fail}  rank={rank:<6}  {top5}")

                # Print the prompt itself for the first concept so user can verify
                if desc == TEST_CONCEPTS[0][2]:
                    print(f"    prompt: {repr(prompt)}")

    print(f"\n{sep}")
    print("LEGEND: ✓=diff>0.5 (gate pass)  ~=diff>0 (right dir)  ✗=diff<=0 (wrong)")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="4b", choices=["4b"])
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top tokens to display")
    parser.add_argument("--formats", nargs="+",
                        default=list(FORMATS.keys()),
                        choices=list(FORMATS.keys()),
                        help="Which format variants to test (default: all)")
    parser.add_argument("--device", default=None,
                        help="Force device (cuda/cpu). Default: auto-detect.")
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok, model = load_model(model_name, device)

    # Verify token IDs
    true_ids  = tok.encode(" True",  add_special_tokens=False)
    false_ids = tok.encode(" False", add_special_tokens=False)
    print(f"Token check:  ' True' → {true_ids}  ' False' → {false_ids}")
    if len(true_ids) != 1 or len(false_ids) != 1:
        print("WARNING: True/False are not single tokens — measurements invalid!")
        sys.exit(1)
    print()

    run_diagnostics(tok, model, device, args.top_k, args.formats)


if __name__ == "__main__":
    main()
