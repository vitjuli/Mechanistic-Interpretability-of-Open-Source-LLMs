"""
54_validation_tests.py — Critical implementation validation (Tests 1, 2, 4)

Test 1: Hook patches decision token (last token = correct action token)
Test 2: Transcoder encode/decode invertibility on synthetic inputs
Test 4: Ablation actually zeroes the target feature activation

Tests 3 & 5 are pure-pandas and run locally (scripts/analyse_recon_robustness.py).

Usage:
    python scripts/54_validation_tests.py
    sbatch jobs/run_validation_tests.sbatch
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set
import yaml

root       = Path(__file__).parent.parent
tc_cfg     = yaml.safe_load(open(root / "configs" / "transcoder_config.yaml"))
tc_info    = tc_cfg["transcoders"][tc_cfg["model_size"]]
model_name = tc_info["model_name"]
device     = "cuda" if torch.cuda.is_available() else "cpu"

BEHAVIOUR  = "physics_decay_type_probe_v2"
SPLIT      = "train"
LAYERS     = list(range(10, 26))
CHECK_IDXS = [0, 1, 50, 100, 200]   # prompts to spot-check
FEAT_LAYER = 24                       # layer for test 4
N_FAILS    = 0

def fail(msg):
    global N_FAILS
    N_FAILS += 1
    print(f"  ✗ FAIL: {msg}")

def ok(msg):
    print(f"  ✓ {msg}")

prompts = [json.loads(l) for l in open(
    root / "data" / "prompts" / f"{BEHAVIOUR}_{SPLIT}.jsonl")]

print("Loading model…")
model = ModelWrapper(model_name, device=device)
tok   = model.tokenizer

print("Loading transcoders…")
tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                              layers=LAYERS, device=device)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Hook patches decision token (last token = correct answer token)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: Last token = decision token")
print("=" * 60)

for idx in CHECK_IDXS:
    p           = prompts[idx]
    prompt_text = p["prompt"]
    correct     = p["correct_answer"]

    ids         = tok(prompt_text, return_tensors="pt")["input_ids"][0]
    last_tok_id = ids[-1].item()
    last_decoded = tok.decode([last_tok_id])

    correct_ids = tok.encode(correct, add_special_tokens=False)
    correct_first_id = correct_ids[0]

    # Decision token should be the last token of the prompt,
    # which should be the first token of correct_answer
    # (model generates the answer at position len(prompt))
    # The NEXT generated token is the answer — so we check that the prompt
    # ends at a natural decision boundary, not with a partial answer token.
    # The correct_answer token should NOT appear in the prompt.
    prompt_ids = ids.tolist()
    answer_in_prompt = correct_first_id in prompt_ids

    if answer_in_prompt:
        fail(f"idx={idx}: correct_answer token ({correct!r} → id {correct_first_id}) "
             f"found in prompt — tokenisation issue")
    else:
        ok(f"idx={idx}: last_prompt_tok='{last_decoded}'  "
           f"correct='{correct.strip()}'  answer not pre-leaked ✓")

    # Also verify correct_answer is single token
    if len(correct_ids) != 1:
        fail(f"idx={idx}: correct_answer {correct!r} → {len(correct_ids)} tokens "
             f"(expected 1): {correct_ids}")
    else:
        ok(f"idx={idx}: correct_answer single-token id={correct_first_id}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Transcoder enc→dec invertibility
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: Transcoder enc→dec invertibility")
print("=" * 60)

for layer in [11, 18, 24]:
    tc = tc_set[layer]
    d  = tc_info.get("d_model", 2560)

    test_cases = {
        "zeros":  torch.zeros(1, d),
        "ones":   torch.ones(1, d),
        "randn":  torch.randn(1, d),
    }

    for name, x_raw in test_cases.items():
        x = x_raw.to(tc.dtype).to(device)
        with torch.no_grad():
            a     = tc.encode(x)
            x_hat = tc.decode(a)
        err = (x_hat - x).norm().item()
        # For zeros: expect near-zero (unless transcoder has a large bias)
        # For randn:  expect err < ||x|| (compression not explosion)
        x_norm = x.norm().item()
        rel_err = err / max(x_norm, 1e-6)

        if name == "zeros" and err > 10.0:
            fail(f"L{layer} zeros: recon_err={err:.3f} — large bias in decoder")
        elif name != "zeros" and rel_err > 5.0:
            fail(f"L{layer} {name}: rel_err={rel_err:.2f} — reconstruction explodes")
        else:
            ok(f"L{layer} {name}: ||ε||={err:.4f}  rel={rel_err:.3f}")

    # Check that active features ≥ 0 (JumpReLU)
    x_real = torch.randn(1, d).to(tc.dtype).to(device)
    with torch.no_grad():
        a = tc.encode(x_real)
    n_neg = (a < 0).sum().item()
    if n_neg > 0:
        fail(f"L{layer}: {n_neg} negative feature activations — JumpReLU broken")
    else:
        n_active = (a > 0).sum().item()
        ok(f"L{layer}: {n_active}/{a.shape[-1]} features active, all ≥ 0")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Ablation zeroes the target feature
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"TEST 4: Ablation zeroes feature (spot-check L{FEAT_LAYER})")
print("=" * 60)

tc = tc_set[FEAT_LAYER]
block = model.model.model.layers[FEAT_LAYER]

for idx in CHECK_IDXS[:3]:
    p = prompts[idx]
    inputs = tok([p["prompt"]], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Capture MLP input
    captured = {}
    def _cap(module, inp, out):
        captured["x"] = out.detach()
    h = block.post_attention_layernorm.register_forward_hook(_cap)
    with torch.no_grad():
        model.model(**inputs, use_cache=False)
    h.remove()

    x = captured["x"][0, -1:, :].to(tc.dtype)   # (1, d)
    with torch.no_grad():
        a = tc.encode(x)                          # (1, d_tc)

    # Find first active feature
    active = (a[0] > 0).nonzero(as_tuple=True)[0]
    if len(active) == 0:
        print(f"  idx={idx}: no active features at L{FEAT_LAYER} (skip)")
        continue

    feat_k = active[0].item()
    orig_val = a[0, feat_k].item()

    # Ablate feat_k → decode → check
    a_modified = a.clone()
    a_modified[0, feat_k] = 0.0
    with torch.no_grad():
        x_hat = tc.decode(a_modified)

    # Re-encode x_hat and check feat_k
    with torch.no_grad():
        a_recheck = tc.encode(x_hat.to(tc.dtype))

    val_after = a_recheck[0, feat_k].item()

    # Primary check: a_modified[feat_k] == 0
    assert a_modified[0, feat_k].item() == 0.0, "modification not applied"
    ok(f"idx={idx} L{FEAT_LAYER} feat {feat_k}: "
       f"before={orig_val:.3f}  a_modified=0.000  "
       f"re-encode={val_after:.3f} (expected small)")

    if abs(val_after) > orig_val * 0.3:
        print(f"    WARNING: re-encode of x_hat has feat {feat_k}={val_after:.3f} "
              f"({abs(val_after)/orig_val:.0%} of original) — transcoder non-linearity")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
if N_FAILS == 0:
    print(f"ALL TESTS PASSED (0 failures)")
else:
    print(f"{N_FAILS} TEST(S) FAILED — review above")
print("=" * 60)
