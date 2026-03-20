#!/usr/bin/env python3
"""
scripts/13_l22_intervention.py

Experiment 2: L22 Causal Intervention

Tests whether L22_F108295 and L22_F32734 are a causal bottleneck for incorrect
FR antonym predictions by scaling their activations and measuring the effect
on logit_diff and predicted token.

Intervention method (consistent with script 11 / script 07):
  1. Get MLP input at L22 for the decision token (token_pos = -1).
  2. Encode through the L22 transcoder → sparse activation vector.
  3. Scale target feature activations by factor alpha:
       feats[:, feature_idx] *= alpha
     (vs ablation_zero which sets to 0.0)
  4. Decode modified activations → new MLP output.
  5. Patch post_attention_layernorm output at L22 → run forward pass.
  6. Measure intervened logit_diff and predicted token.

Alpha values tested: 2.0, 3.0
  alpha=1 → no change (baseline)
  alpha=2 → double the feature activations
  alpha=3 → triple the feature activations

For Type-A failures (circuit tries correct but baseline is negative):
  Expected: delta_logit > 0 (scaling helps), flipped_to_correct for large alpha.
For Type-B failures (p43, p45, circuit actively pushing wrong):
  Expected: delta_logit < 0 (scaling worsens), confirming L22 is causal driver.

Reads:
  data/prompts/{behaviour}_{split}.jsonl
  data/results/reasoning_traces/{behaviour}/error_cases_{split}.json

Writes:
  data/results/reasoning_traces/{behaviour}/l22_intervention_results_{split}.json

Runtime: 18 prompts × (1 baseline + 2 alpha) × 2 forward-passes per condition ≈ 108 passes.
         ~5 min on GPU.
"""

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# L22 gateway features — the primary intervention targets.
TARGET_FEATURES = {
    "L22_F108295": 108295,
    "L22_F32734":  32734,
}

# Alpha values to test (baseline = 1.0 implicitly).
ALPHAS = [2.0, 3.0]


# ─── Hook utilities (identical to script 11) ─────────────────────────────────

@contextmanager
def _patch_mlp_input(model_hf, layer_idx: int, token_pos: int, new_mlp_input: torch.Tensor):
    """Replace post_attention_layernorm output at token_pos for one forward pass."""
    try:
        block = model_hf.model.layers[layer_idx]
    except AttributeError:
        block = model_hf.transformer.h[layer_idx]

    if hasattr(block, "post_attention_layernorm"):
        hook_module = block.post_attention_layernorm
    elif hasattr(block, "ln_2"):
        hook_module = block.ln_2
    else:
        raise RuntimeError(f"Cannot find MLP-input norm in layer {layer_idx}")

    hook_called = {"count": 0}

    def hook(module, inp, out):
        hook_called["count"] += 1
        h = out[0].clone() if isinstance(out, tuple) else out.clone()
        h[:, token_pos, :] = new_mlp_input.to(h.dtype).to(h.device)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = hook_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        assert hook_called["count"] > 0, (
            f"MLP hook did not fire at layer {layer_idx}."
        )


def _get_mlp_input(model: "ModelWrapper", inputs: dict, layer_idx: int, token_pos: int = -1) -> torch.Tensor:
    """Extract MLP-input activation (post_attention_layernorm) at token_pos."""
    try:
        block = model.model.model.layers[layer_idx]
    except AttributeError:
        block = model.model.transformer.h[layer_idx]

    if hasattr(block, "post_attention_layernorm"):
        hook_module = block.post_attention_layernorm
    elif hasattr(block, "ln_2"):
        hook_module = block.ln_2
    else:
        raise RuntimeError(f"Cannot find MLP-input norm in layer {layer_idx}")

    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["act"] = h[:, token_pos, :].detach().float().cpu()

    handle = hook_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert "act" in captured, f"MLP-input hook did not fire at layer {layer_idx}"
    return captured["act"]


# ─── Baseline ─────────────────────────────────────────────────────────────────

def compute_baseline(
    model: "ModelWrapper",
    prompt: str,
    correct_token: str,
    incorrect_token: str,
    device: str,
) -> dict:
    """One unmodified forward pass. Returns baseline logit_diff and top-1 prediction."""
    def tok_id(tok: str) -> int:
        ids = model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token '{tok}' → {len(ids)} sub-tokens")
        return ids[0]

    cid = tok_id(correct_token)
    iid = tok_id(incorrect_token)

    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.model(**inputs, use_cache=False)

    logits    = out.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=0)

    baseline_ld  = (log_probs[cid] - log_probs[iid]).item()
    pred_id      = torch.argmax(log_probs).item()
    pred_token   = model.tokenizer.decode([pred_id])
    correct_logp = log_probs[cid].item()

    return {
        "baseline_logit_diff":    round(baseline_ld, 6),
        "baseline_correct_logp":  round(correct_logp, 6),
        "baseline_prediction":    pred_token,
        "correct_id":             cid,
        "incorrect_id":           iid,
        "inputs":                 inputs,
    }


# ─── L22 boost intervention ───────────────────────────────────────────────────

def run_l22_boost(
    model: "ModelWrapper",
    tc_l22,
    inputs: dict,
    correct_id: int,
    incorrect_id: int,
    target_feature_indices: List[int],
    alpha: float,
    device: str,
) -> dict:
    """
    Scale target L22 feature activations by alpha.

    Method:
      feats = tc.encode(mlp_input)        # [1, d_tc] dense activation vector
      feats[:, feat_idx] *= alpha          # scale target features in-place
      modified = tc.decode(feats)          # reconstruct MLP output
      forward pass with modified L22 MLP output

    Returns per-condition result dict.
    """
    # 1. Get current MLP input at L22 (decision token, pos=-1)
    mlp_input = _get_mlp_input(model, inputs, layer_idx=22, token_pos=-1)
    mlp_input = mlp_input.to(device)

    # 2. Encode → scale → decode
    with torch.no_grad():
        feats = tc_l22.encode(mlp_input.to(tc_l22.dtype))    # [1, d_tc]

        # Record pre-intervention activation values for the target features
        pre_activations = {
            fidx: float(feats[0, fidx].item())
            for fidx in target_feature_indices
        }

        for fidx in target_feature_indices:
            feats[:, fidx] *= alpha

        post_activations = {
            fidx: float(feats[0, fidx].item())
            for fidx in target_feature_indices
        }

        modified = tc_l22.decode(feats).to(mlp_input.dtype)

    # 3. Intervened forward pass
    with torch.no_grad():
        with _patch_mlp_input(model.model, layer_idx=22, token_pos=-1, new_mlp_input=modified):
            out_i     = model.model(**inputs, use_cache=False)
            logits_i  = out_i.logits[0, -1, :].float()
    log_probs_i      = torch.log_softmax(logits_i, dim=0)
    intervened_ld    = (log_probs_i[correct_id] - log_probs_i[incorrect_id]).item()
    intervened_logp  = log_probs_i[correct_id].item()
    new_pred_id      = torch.argmax(log_probs_i).item()
    new_pred_token   = model.tokenizer.decode([new_pred_id])

    return {
        "alpha":                 alpha,
        "intervention_logit_diff": round(intervened_ld, 6),
        "intervention_correct_logp": round(intervened_logp, 6),
        "new_prediction":        new_pred_token,
        "pre_activations":       {f"L22_F{k}": round(v, 6) for k, v in pre_activations.items()},
        "post_activations":      {f"L22_F{k}": round(v, 6) for k, v in post_activations.items()},
    }


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="L22 causal intervention on incorrect FR prompts.")
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS,
                   help="Alpha scale factors to test (default: 2.0 3.0)")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"L22 Causal Intervention — {args.behaviour}/{args.split}")
    print(f"Target features: {list(TARGET_FEATURES.keys())}")
    print(f"Alpha values:    {args.alphas}")
    print("=" * 60)

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"data/results/reasoning_traces/{args.behaviour}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = [
        json.loads(l)
        for l in Path(f"data/prompts/{args.behaviour}_{args.split}.jsonl").read_text().splitlines()
        if l.strip()
    ]
    logger.info(f"Prompts loaded: {len(prompts)}")

    # ── Load incorrect prompt indices ─────────────────────────────────────────
    error_cases = json.loads(
        (out_dir / f"error_cases_{args.split}.json").read_text()
    )
    incorrect_idx = error_cases["incorrect_prompt_indices"]
    type_b_idx    = set(error_cases["failure_types"]["type_B_prompt_indices"])
    logger.info(f"Incorrect prompts: {len(incorrect_idx)} "
                f"(Type A={len(incorrect_idx) - len(type_b_idx)}, Type B={len(type_b_idx)})")

    # ── Device / model ────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tc_config  = yaml.safe_load(open("configs/transcoder_config.yaml"))
    model_name = tc_config["transcoders"][args.model_size]["model_name"]
    print(f"Loading model {model_name}...")
    model = ModelWrapper(model_name=model_name, device=device, dtype=torch.bfloat16)
    print(f"  Model loaded on {device}")

    # ── Load L22 transcoder only ──────────────────────────────────────────────
    print("Loading L22 transcoder...")
    transcoder_set = load_transcoder_set(
        model_size=args.model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=False,
        layers=[22],
    )
    tc_l22 = transcoder_set[22]
    print(f"  L22 transcoder loaded")

    target_indices = list(TARGET_FEATURES.values())
    print(f"  Target feature indices: {target_indices}")

    # ── Run interventions ─────────────────────────────────────────────────────
    per_prompt = []

    for pi in incorrect_idx:
        p = prompts[pi]
        correct_token   = p.get("correct_answer", p.get("answer_matching", ""))
        incorrect_token = p.get("incorrect_answer", p.get("answer_not_matching", ""))
        language        = p.get("language", "?")
        concept_index   = p.get("concept_index", -1)
        template_idx    = p.get("template_idx", -1)
        failure_type    = "B" if pi in type_b_idx else "A"

        logger.info(f"p{pi:02d} [Type {failure_type}, {language} c{concept_index} t{template_idx}]")

        # Baseline
        bl = compute_baseline(model, p["prompt"], correct_token, incorrect_token, device)
        baseline_ld   = bl["baseline_logit_diff"]
        baseline_logp = bl["baseline_correct_logp"]
        cid, iid      = bl["correct_id"], bl["incorrect_id"]
        inputs        = bl["inputs"]

        print(f"  p{pi:02d} baseline_ld={baseline_ld:.4f}, pred={repr(bl['baseline_prediction'])}")

        # Interventions at each alpha
        conditions = []
        for alpha in args.alphas:
            cond = run_l22_boost(
                model, tc_l22, inputs, cid, iid,
                target_indices, alpha, device,
            )
            delta      = cond["intervention_logit_diff"] - baseline_ld
            delta_logp = cond["intervention_correct_logp"] - baseline_logp
            flipped    = (baseline_ld < 0) and (cond["intervention_logit_diff"] > 0)
            cond.update({
                "delta_logit":     round(delta, 6),
                "delta_correct_logp": round(delta_logp, 6),
                "flipped_to_correct": flipped,
            })
            conditions.append(cond)
            flip_str = " *** FLIPPED ***" if flipped else ""
            print(f"    alpha={alpha:.1f}: ld={cond['intervention_logit_diff']:.4f}, "
                  f"Δ={delta:+.4f}, pred={repr(cond['new_prediction'])}{flip_str}")

        per_prompt.append({
            "prompt_idx":           pi,
            "language":             language,
            "concept_index":        concept_index,
            "template_idx":         template_idx,
            "failure_type":         failure_type,
            "correct_token":        correct_token,
            "incorrect_token":      incorrect_token,
            "baseline_logit_diff":  baseline_ld,
            "baseline_correct_logp": baseline_logp,
            "baseline_prediction":  bl["baseline_prediction"],
            "conditions":           conditions,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_by_alpha: List[dict] = []
    for alpha in args.alphas:
        conds = [
            next(c for c in r["conditions"] if c["alpha"] == alpha)
            for r in per_prompt
        ]
        deltas        = [c["delta_logit"] for c in conds]
        n_improved    = sum(d > 0 for d in deltas)
        n_flipped     = sum(c["flipped_to_correct"] for c in conds)
        # Type A only
        type_a_rows   = [r for r in per_prompt if r["failure_type"] == "A"]
        type_a_conds  = [
            next(c for c in r["conditions"] if c["alpha"] == alpha)
            for r in type_a_rows
        ]
        n_a_improved  = sum(c["delta_logit"] > 0 for c in type_a_conds)
        n_a_flipped   = sum(c["flipped_to_correct"] for c in type_a_conds)

        summary_by_alpha.append({
            "alpha":                      alpha,
            "n_prompts":                  len(per_prompt),
            "n_improved":                 n_improved,
            "n_flipped_to_correct":       n_flipped,
            "mean_delta_logit":           round(float(np.mean(deltas)), 4),
            "std_delta_logit":            round(float(np.std(deltas)), 4),
            "min_delta":                  round(float(np.min(deltas)), 4),
            "max_delta":                  round(float(np.max(deltas)), 4),
            "n_type_a":                   len(type_a_rows),
            "n_type_a_improved":          n_a_improved,
            "n_type_a_flipped":           n_a_flipped,
            "n_type_b":                   len(type_b_idx),
            "n_type_b_worsened":          sum(
                c["delta_logit"] < 0
                for r in per_prompt if r["failure_type"] == "B"
                for c in r["conditions"] if c["alpha"] == alpha
            ),
        })

    # ── Critical interpretations ──────────────────────────────────────────────
    # Use alpha=2.0 as primary reference
    s_primary  = summary_by_alpha[0]
    n_total    = s_primary["n_prompts"]
    n_improved = s_primary["n_improved"]
    n_flipped  = s_primary["n_flipped_to_correct"]
    n_a_flipped = s_primary["n_type_a_flipped"]
    mean_delta = s_primary["mean_delta_logit"]

    if n_improved >= n_total * 0.7 and mean_delta > 0:
        q1_answer = (
            f"YES — scaling L22 features improves logit_diff in {n_improved}/{n_total} prompts "
            f"(alpha=2.0, mean Δ={mean_delta:+.4f}). L22 gateway activation systematically "
            "benefits correct-antonym prediction."
        )
    elif n_improved >= n_total * 0.4:
        q1_answer = (
            f"PARTIAL — {n_improved}/{n_total} prompts improved (alpha=2.0, mean Δ={mean_delta:+.4f}). "
            "L22 contributes but other factors also limit performance."
        )
    else:
        q1_answer = (
            f"NO — only {n_improved}/{n_total} prompts improved (alpha=2.0, mean Δ={mean_delta:+.4f}). "
            "L22 scaling does not reliably help."
        )

    if n_flipped > 0 or mean_delta > 0.1:
        q2_answer = (
            f"CAUSAL — L22 scaling flips {n_flipped}/{n_total} predictions to correct and "
            f"improves {n_improved}/{n_total} prompts (alpha=2.0). "
            "L22 gateway is a genuine causal bottleneck, not just a correlational marker."
        )
    else:
        q2_answer = (
            "CORRELATIONAL ONLY — scaling L22 does not reliably flip predictions. "
            "L22 may be a downstream reporter of the circuit state rather than the bottleneck."
        )

    if n_a_flipped > 0:
        q3_answer = (
            f"YES — {n_a_flipped}/{s_primary['n_type_a']} Type-A failures flipped to correct "
            f"at alpha=2.0. L22 boost reduces Type-A count."
        )
    else:
        q3_answer = (
            f"PARTIAL/NO — 0/{s_primary['n_type_a']} Type-A failures flipped at alpha=2.0. "
            "Larger alpha or other bottlenecks may explain residual Type-A failures."
        )

    interpretations = {
        "Q1_l22_systematically_improves": q1_answer,
        "Q2_l22_causal_vs_correlational":  q2_answer,
        "Q3_reduces_type_a_failures":      q3_answer,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("L22 INTERVENTION SUMMARY")
    print("=" * 60)
    for s in summary_by_alpha:
        print(f"\nalpha={s['alpha']:.1f}:")
        print(f"  Improved:        {s['n_improved']}/{s['n_prompts']}")
        print(f"  Flipped:         {s['n_flipped_to_correct']}/{s['n_prompts']}")
        print(f"  Mean Δlogit:     {s['mean_delta_logit']:+.4f} ± {s['std_delta_logit']:.4f}")
        print(f"  Type-A improved: {s['n_type_a_improved']}/{s['n_type_a']}, flipped: {s['n_type_a_flipped']}")
        print(f"  Type-B worsened: {s['n_type_b_worsened']}/{s['n_type_b']} (expected for causal confirmation)")

    print("\nCritical interpretations:")
    print(f"  Q1: {q1_answer}")
    print(f"  Q2: {q2_answer}")
    print(f"  Q3: {q3_answer}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "behaviour":       args.behaviour,
        "split":           args.split,
        "target_features": list(TARGET_FEATURES.keys()),
        "alphas_tested":   args.alphas,
        "per_prompt":      per_prompt,
        "summary_by_alpha": summary_by_alpha,
        "interpretations": interpretations,
    }
    out_path = out_dir / f"l22_intervention_results_{args.split}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
