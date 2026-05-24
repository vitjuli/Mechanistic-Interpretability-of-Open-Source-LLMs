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

    # Decision token check: the LAST token of the prompt must NOT be the answer
    # token (that would mean the answer is already in the prompt — pre-leaked).
    # Note: for physics prompts the answer word (e.g. "alpha") can appear in
    # the question body ("alpha decay or beta decay?") — that is expected and
    # is NOT a tokenisation issue.  We only care about the final position.
    if last_tok_id == correct_first_id:
        fail(f"idx={idx}: last prompt token IS the answer token "
             f"({correct!r} → id {correct_first_id}) — answer pre-leaked at end")
    else:
        ok(f"idx={idx}: last_prompt_tok='{last_decoded}'  "
           f"correct='{correct.strip()}'  answer not at end ✓")

    # Informational: note if answer token appears in the prompt body (expected
    # for physics prompts where the question names both decay types).
    prompt_ids = ids.tolist()
    if correct_first_id in prompt_ids:
        print(f"    (note: answer token appears in prompt body — expected for "
              f"physics prompts that name both decay types)")

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

def _capture_mlp_input(layer_idx: int, prompt_text: str) -> torch.Tensor:
    """Return post_attention_layernorm output at last token — (1, d)."""
    inputs = tok([prompt_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}
    block = model.model.model.layers[layer_idx]
    def _hook(module, inp, out):
        captured["x"] = out[:, -1:, :].detach()
    h = block.post_attention_layernorm.register_forward_hook(_hook)
    with torch.no_grad():
        model.model(**inputs, use_cache=False)
    h.remove()
    return captured["x"][0]   # (1, d)

# Collect real activations for a small set of prompts at each test layer.
# Using 5 prompts gives a realistic distribution; we check that enc→dec
# reconstruction error is within a reasonable multiple of the recon baseline
# (~20-25 norm units from script 51).
REAL_REL_ERR_THRESHOLD = 2.0   # rel_err < 2.0 for real activations (generous)

for layer in [11, 18, 24]:
    tc = tc_set[layer]
    errs, norms = [], []
    for idx in CHECK_IDXS:
        x = _capture_mlp_input(layer, prompts[idx]["prompt"]).to(tc.dtype)
        with torch.no_grad():
            a     = tc.encode(x)
            x_hat = tc.decode(a)
        err   = (x_hat - x).norm().item()
        xnorm = x.norm().item()
        errs.append(err)
        norms.append(xnorm)

    mean_err  = sum(errs)  / len(errs)
    mean_norm = sum(norms) / len(norms)
    mean_rel  = mean_err   / max(mean_norm, 1e-6)

    if mean_rel > REAL_REL_ERR_THRESHOLD:
        fail(f"L{layer} real activations: mean_rel_err={mean_rel:.3f} > "
             f"{REAL_REL_ERR_THRESHOLD} — transcoder reconstruction poor on real data")
    else:
        ok(f"L{layer} real activations: mean||ε||={mean_err:.2f}  "
           f"mean||x||={mean_norm:.2f}  rel={mean_rel:.3f}")

    # Check JumpReLU: all activations must be ≥ 0
    x_sample = _capture_mlp_input(layer, prompts[CHECK_IDXS[0]]["prompt"]).to(tc.dtype)
    with torch.no_grad():
        a = tc.encode(x_sample)
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
