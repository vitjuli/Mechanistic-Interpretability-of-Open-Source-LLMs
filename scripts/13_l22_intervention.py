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

Fix 1 — Dose-response: alpha ∈ {1.5, 2.0, 3.0} to distinguish gradual causal
         effect from extreme-forcing artifact.

Fix 2 — Negative control: a non-target L22 feature is selected dynamically
         (highest mean activation across the 18 incorrect prompts, excluding
         target features and circuit features). Identical scaling applied;
         L22 effect compared against control effect to rule out "any scaling helps".

Fix 4 — Activation sanity: per-prompt baseline and post-scaling activations
         recorded for both target and control features; % active and mean
         activation reported in summary.

Reads:
  data/prompts/{behaviour}_{split}.jsonl
  data/results/reasoning_traces/{behaviour}/error_cases_{split}.json

Writes:
  data/results/reasoning_traces/{behaviour}/l22_intervention_results_{split}.json

Runtime: ~90 + 36 (control select) forward passes ≈ 8 min on GPU.
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

# L22 gateway features — primary intervention targets.
TARGET_FEATURES = {
    "L22_F108295": 108295,
    "L22_F32734":  32734,
}

# Fix 1: dose-response alpha values.
ALPHAS = [1.5, 2.0, 3.0]

# Fix 4: threshold below which a feature is considered inactive.
ACTIVATION_THRESHOLD = 0.01

# All L22 circuit features (exclude these from control selection).
_L22_CIRCUIT_FEATURE_INDICES = {108295, 32734}


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
        assert hook_called["count"] > 0, f"MLP hook did not fire at layer {layer_idx}."


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


# ─── Fix 2: Control feature selection ─────────────────────────────────────────

def select_control_feature(
    model: "ModelWrapper",
    tc_l22,
    prompts: list,
    incorrect_idx: list,
    excluded_indices: set,
    device: str,
) -> Tuple[int, float]:
    """
    Select the highest mean-activated L22 feature across the incorrect prompts
    that is NOT in excluded_indices (target features and known circuit features).

    This provides an objective negative control: same layer, same transcoder,
    reliably active, but not part of the causal circuit.

    Returns: (feature_index, mean_activation_across_prompts)
    """
    cumsum = None
    for pi in incorrect_idx:
        p = prompts[pi]
        inputs = model.tokenizer(p["prompt"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mlp_input = _get_mlp_input(model, inputs, layer_idx=22, token_pos=-1)
        mlp_input = mlp_input.to(device)
        with torch.no_grad():
            feats = tc_l22.encode(mlp_input.to(tc_l22.dtype))
            acts = feats[0].float().cpu()
        cumsum = acts.clone() if cumsum is None else cumsum + acts

    mean_acts = cumsum / len(incorrect_idx)
    # Zero out excluded features so they cannot be selected
    for fidx in excluded_indices:
        mean_acts[fidx] = -1.0
    control_idx = int(torch.argmax(mean_acts).item())
    control_mean_act = float(mean_acts[control_idx].item())
    return control_idx, control_mean_act


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
        "baseline_logit_diff":   round(baseline_ld, 6),
        "baseline_correct_logp": round(correct_logp, 6),
        "baseline_prediction":   pred_token,
        "correct_id":            cid,
        "incorrect_id":          iid,
        "inputs":                inputs,
    }


# ─── Feature scaling intervention ─────────────────────────────────────────────

def run_feature_boost(
    model: "ModelWrapper",
    tc_l22,
    inputs: dict,
    correct_id: int,
    incorrect_id: int,
    feature_indices: List[int],
    alpha: float,
    device: str,
) -> dict:
    """
    Scale the specified L22 feature activations by alpha.

    Method:
      feats = tc.encode(mlp_input)         # [1, d_tc] dense activation vector
      feats[:, feat_idx] *= alpha           # scale in-place
      modified = tc.decode(feats)           # reconstruct MLP output
      forward pass with patched L22 MLP output

    Fix 4: Records pre/post activations and active-flag for each feature.

    Returns per-condition result dict compatible with original output schema.
    """
    mlp_input = _get_mlp_input(model, inputs, layer_idx=22, token_pos=-1)
    mlp_input = mlp_input.to(device)

    with torch.no_grad():
        feats = tc_l22.encode(mlp_input.to(tc_l22.dtype))

        # Fix 4: record activations before intervention
        pre_activations = {
            f"L22_F{fidx}": round(float(feats[0, fidx].item()), 6)
            for fidx in feature_indices
        }
        feature_active_pre = {
            f"L22_F{fidx}": bool(float(feats[0, fidx].item()) > ACTIVATION_THRESHOLD)
            for fidx in feature_indices
        }

        for fidx in feature_indices:
            feats[:, fidx] *= alpha

        # Fix 4: record activations after scaling
        post_activations = {
            f"L22_F{fidx}": round(float(feats[0, fidx].item()), 6)
            for fidx in feature_indices
        }

        modified = tc_l22.decode(feats).to(mlp_input.dtype)

    with torch.no_grad():
        with _patch_mlp_input(model.model, layer_idx=22, token_pos=-1, new_mlp_input=modified):
            out_i    = model.model(**inputs, use_cache=False)
            logits_i = out_i.logits[0, -1, :].float()
    log_probs_i     = torch.log_softmax(logits_i, dim=0)
    intervened_ld   = (log_probs_i[correct_id] - log_probs_i[incorrect_id]).item()
    intervened_logp = log_probs_i[correct_id].item()
    new_pred_id     = torch.argmax(log_probs_i).item()
    new_pred_token  = model.tokenizer.decode([new_pred_id])

    return {
        "alpha":                       alpha,
        "intervention_logit_diff":     round(intervened_ld, 6),
        "intervention_correct_logp":   round(intervened_logp, 6),
        "new_prediction":              new_pred_token,
        # Fix 4: activation fields
        "pre_activations":             pre_activations,
        "post_activations":            post_activations,
        "feature_active_pre":          feature_active_pre,
    }


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="L22 causal intervention on incorrect FR prompts.")
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS,
                   help="Alpha scale factors (Fix 1: default 1.5 2.0 3.0)")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"L22 Causal Intervention — {args.behaviour}/{args.split}")
    print(f"Target features: {list(TARGET_FEATURES.keys())}")
    print(f"Alpha values:    {args.alphas}  [Fix 1: dose-response]")
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

    # ── Load L22 transcoder ───────────────────────────────────────────────────
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

    # ── Fix 2: Select negative control feature ────────────────────────────────
    print(f"\n[Fix 2] Selecting negative control feature from L22...")
    print(f"  Running encode on {len(incorrect_idx)} incorrect prompts to find "
          f"highest-activated non-target feature...")
    control_idx, control_mean_act = select_control_feature(
        model, tc_l22, prompts, incorrect_idx, _L22_CIRCUIT_FEATURE_INDICES, device
    )
    control_feature_id = f"L22_F{control_idx}"
    print(f"  Control feature selected: {control_feature_id} "
          f"(mean activation across incorrect prompts = {control_mean_act:.4f})")
    print(f"  Control is NOT in circuit and NOT a target: sanity OK")

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

        # Target interventions (Fix 1: three alpha values)
        conditions = []
        for alpha in args.alphas:
            cond = run_feature_boost(
                model, tc_l22, inputs, cid, iid, target_indices, alpha, device,
            )
            delta      = cond["intervention_logit_diff"] - baseline_ld
            delta_logp = cond["intervention_correct_logp"] - baseline_logp
            flipped    = (baseline_ld < 0) and (cond["intervention_logit_diff"] > 0)
            cond.update({
                "delta_logit":        round(delta, 6),
                "delta_correct_logp": round(delta_logp, 6),
                "flipped_to_correct": flipped,
            })
            conditions.append(cond)
            flip_str = " *** FLIPPED ***" if flipped else ""
            print(f"    [target] α={alpha:.1f}: ld={cond['intervention_logit_diff']:.4f}, "
                  f"Δ={delta:+.4f}, pred={repr(cond['new_prediction'])}{flip_str}")

        # Fix 2: Control feature interventions (same alphas, same logic)
        control_conditions = []
        for alpha in args.alphas:
            ctrl = run_feature_boost(
                model, tc_l22, inputs, cid, iid, [control_idx], alpha, device,
            )
            ctrl_delta = ctrl["intervention_logit_diff"] - baseline_ld
            ctrl["delta_logit"]        = round(ctrl_delta, 6)
            ctrl["delta_correct_logp"] = round(ctrl["intervention_correct_logp"] - baseline_logp, 6)
            ctrl["flipped_to_correct"] = (baseline_ld < 0) and (ctrl["intervention_logit_diff"] > 0)
            control_conditions.append(ctrl)
            print(f"    [ctrl  ] α={alpha:.1f}: ld={ctrl['intervention_logit_diff']:.4f}, "
                  f"Δ={ctrl_delta:+.4f}")

        per_prompt.append({
            "prompt_idx":            pi,
            "language":              language,
            "concept_index":         concept_index,
            "template_idx":          template_idx,
            "failure_type":          failure_type,
            "correct_token":         correct_token,
            "incorrect_token":       incorrect_token,
            "baseline_logit_diff":   baseline_ld,
            "baseline_correct_logp": baseline_logp,
            "baseline_prediction":   bl["baseline_prediction"],
            # Target interventions (Fix 1: three alphas)
            "conditions":            conditions,
            # Fix 2: Control interventions
            "control_feature_id":    control_feature_id,
            "control_conditions":    control_conditions,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_by_alpha: List[dict] = []
    for alpha in args.alphas:
        # Target metrics
        t_conds   = [next(c for c in r["conditions"] if c["alpha"] == alpha) for r in per_prompt]
        t_deltas  = [c["delta_logit"] for c in t_conds]
        # Control metrics (Fix 2)
        c_conds   = [next(c for c in r["control_conditions"] if c["alpha"] == alpha) for r in per_prompt]
        c_deltas  = [c["delta_logit"] for c in c_conds]

        n_t_improved  = sum(d > 0 for d in t_deltas)
        n_c_improved  = sum(d > 0 for d in c_deltas)
        n_t_flipped   = sum(c["flipped_to_correct"] for c in t_conds)
        n_c_flipped   = sum(c["flipped_to_correct"] for c in c_conds)

        type_a_rows   = [r for r in per_prompt if r["failure_type"] == "A"]
        ta_t_conds    = [next(c for c in r["conditions"] if c["alpha"] == alpha) for r in type_a_rows]
        ta_c_conds    = [next(c for c in r["control_conditions"] if c["alpha"] == alpha) for r in type_a_rows]
        n_ta_t_imp    = sum(c["delta_logit"] > 0 for c in ta_t_conds)
        n_ta_t_flip   = sum(c["flipped_to_correct"] for c in ta_t_conds)

        mean_t = float(np.mean(t_deltas))
        mean_c = float(np.mean(c_deltas))
        # Effect superiority: is L22 improvement larger than control?
        l22_vs_ctrl_delta = round(mean_t - mean_c, 4)

        # Fix 2: explicit comparison
        if mean_t > mean_c + 0.1 and n_t_improved > n_c_improved:
            ctrl_comparison = (
                f"L22 target BETTER than control: mean_target={mean_t:+.4f}, "
                f"mean_control={mean_c:+.4f}, Δ={l22_vs_ctrl_delta:+.4f}. "
                "L22 effect is specific, not generic scaling artifact."
            )
        elif abs(mean_t - mean_c) <= 0.1:
            ctrl_comparison = (
                f"L22 target SIMILAR to control: mean_target={mean_t:+.4f}, "
                f"mean_control={mean_c:+.4f}, Δ={l22_vs_ctrl_delta:+.4f}. "
                "Cannot rule out generic L22 scaling effect."
            )
        else:
            ctrl_comparison = (
                f"L22 target NOT better than control: mean_target={mean_t:+.4f}, "
                f"mean_control={mean_c:+.4f}, Δ={l22_vs_ctrl_delta:+.4f}. "
                "Generic L22 scaling provides similar or larger effect."
            )

        summary_by_alpha.append({
            "alpha":                         alpha,
            "n_prompts":                     len(per_prompt),
            # Target
            "n_target_improved":             n_t_improved,
            "n_target_flipped":              n_t_flipped,
            "mean_target_delta_logit":       round(mean_t, 4),
            "std_target_delta_logit":        round(float(np.std(t_deltas)), 4),
            # Control (Fix 2)
            "n_control_improved":            n_c_improved,
            "n_control_flipped":             n_c_flipped,
            "mean_control_delta_logit":      round(mean_c, 4),
            "std_control_delta_logit":       round(float(np.std(c_deltas)), 4),
            # Comparison
            "l22_vs_control_mean_delta":     l22_vs_ctrl_delta,
            "l22_is_better_than_control":    mean_t > mean_c + 0.05,
            "control_comparison":            ctrl_comparison,
            # Type A breakdown
            "n_type_a":                      len(type_a_rows),
            "n_type_a_improved":             n_ta_t_imp,
            "n_type_a_flipped":              n_ta_t_flip,
            "n_type_b":                      len(type_b_idx),
            "n_type_b_worsened":             sum(
                c["delta_logit"] < 0
                for r in per_prompt if r["failure_type"] == "B"
                for c in r["conditions"] if c["alpha"] == alpha
            ),
        })

    # ── Fix 4: Activation sanity summary ─────────────────────────────────────
    # Collect activations for target features and control across all prompts
    activation_summary = {}
    for fkey in [f"L22_F{fidx}" for fidx in target_indices] + [control_feature_id]:
        pre_vals = []
        post_vals_by_alpha: Dict[float, list] = {a: [] for a in args.alphas}
        n_active = 0
        for r in per_prompt:
            # Use first condition to get baseline activation (alpha doesn't affect pre)
            is_ctrl = fkey == control_feature_id
            first_cond = (r["control_conditions"] if is_ctrl else r["conditions"])[0]
            pre = first_cond["pre_activations"].get(fkey)
            if pre is not None:
                pre_vals.append(pre)
                if pre > ACTIVATION_THRESHOLD:
                    n_active += 1
            for alpha in args.alphas:
                conds = r["control_conditions"] if is_ctrl else r["conditions"]
                cond = next((c for c in conds if c["alpha"] == alpha), None)
                if cond:
                    post = cond["post_activations"].get(fkey)
                    if post is not None:
                        post_vals_by_alpha[alpha].append(post)
        activation_summary[fkey] = {
            "mean_baseline_activation": round(float(np.mean(pre_vals)), 6) if pre_vals else None,
            "pct_active_at_baseline":   round(100.0 * n_active / len(per_prompt), 1) if per_prompt else None,
            "mean_post_activation":     {
                str(a): round(float(np.mean(v)), 6) if v else None
                for a, v in post_vals_by_alpha.items()
            },
        }

    print(f"\n[Fix 4] Feature Activation Sanity Check:")
    for fkey, stats in activation_summary.items():
        label = "(TARGET)" if fkey != control_feature_id else "(CONTROL)"
        print(f"  {fkey} {label}:")
        print(f"    Baseline mean activation: {stats['mean_baseline_activation']:.6f}")
        print(f"    Active (>{ACTIVATION_THRESHOLD}) in: {stats['pct_active_at_baseline']:.1f}% of prompts")
        for a_str, post_mean in stats["mean_post_activation"].items():
            print(f"    Post α={a_str}: {post_mean:.6f}")

    # ── Critical interpretations ──────────────────────────────────────────────
    # Use alpha=2.0 as primary; fall back to first available
    s_primary = next((s for s in summary_by_alpha if s["alpha"] == 2.0), summary_by_alpha[0])
    s_max     = max(summary_by_alpha, key=lambda s: s["n_target_improved"])
    n_total   = s_primary["n_prompts"]

    # Dose-response check (Fix 1): is effect monotone across alphas?
    mean_deltas = [s["mean_target_delta_logit"] for s in summary_by_alpha]
    is_monotone = all(mean_deltas[i] <= mean_deltas[i+1] for i in range(len(mean_deltas)-1)) or \
                  all(mean_deltas[i] >= mean_deltas[i+1] for i in range(len(mean_deltas)-1))
    dose_response_str = (
        f"Monotone: {mean_deltas} — "
        + ("gradual causal effect confirmed." if is_monotone else "non-monotone; possible non-linear regime.")
    )

    # Q1: Does increasing L22 systematically improve outcomes?
    n_imp_primary = s_primary["n_target_improved"]
    mean_d_primary = s_primary["mean_target_delta_logit"]
    if n_imp_primary >= int(n_total * 0.7) and mean_d_primary > 0:
        q1_answer = (
            f"YES — {n_imp_primary}/{n_total} prompts improve at α=2.0 "
            f"(mean Δ={mean_d_primary:+.4f}). Dose-response: {dose_response_str}"
        )
    elif n_imp_primary >= int(n_total * 0.4):
        q1_answer = (
            f"PARTIAL — {n_imp_primary}/{n_total} improve at α=2.0 "
            f"(mean Δ={mean_d_primary:+.4f}). Dose-response: {dose_response_str}"
        )
    else:
        q1_answer = (
            f"NO — only {n_imp_primary}/{n_total} improve at α=2.0 "
            f"(mean Δ={mean_d_primary:+.4f}). L22 scaling does not reliably help. "
            f"Dose-response: {dose_response_str}"
        )

    # Q2: Causal vs correlational (Fix 2: includes control comparison)
    n_flip_max = s_max["n_target_flipped"]
    ctrl_comp  = s_primary["control_comparison"]
    if n_flip_max > 0 and s_primary["l22_is_better_than_control"]:
        q2_answer = (
            f"CAUSAL — {n_flip_max} predictions flipped to correct at α={s_max['alpha']:.1f}. "
            f"L22 effect exceeds control: {ctrl_comp}"
        )
    elif n_flip_max > 0 and not s_primary["l22_is_better_than_control"]:
        q2_answer = (
            f"AMBIGUOUS — {n_flip_max} predictions flipped at α={s_max['alpha']:.1f} but "
            f"control comparison inconclusive: {ctrl_comp}"
        )
    elif s_primary["l22_is_better_than_control"] and mean_d_primary > 0.05:
        q2_answer = (
            f"LIKELY CAUSAL — no flips at α=2.0 but L22 outperforms control: {ctrl_comp}"
        )
    else:
        q2_answer = (
            f"CORRELATIONAL — no consistent improvement or control comparison fails: {ctrl_comp}"
        )

    # Q3: Does intervention reduce Type A failures?
    n_a_flip_max = s_max["n_type_a_flipped"]
    q3_answer = (
        f"{'YES' if n_a_flip_max > 0 else 'NO'} — "
        f"{n_a_flip_max}/{s_primary['n_type_a']} Type-A failures flipped at "
        f"α={s_max['alpha']:.1f}."
    )

    interpretations = {
        "Q1_l22_systematically_improves":  q1_answer,
        "Q2_l22_causal_vs_correlational":  q2_answer,
        "Q3_reduces_type_a_failures":      q3_answer,
        "dose_response_alphas":            args.alphas,
        "dose_response_mean_deltas":       mean_deltas,
        "dose_response_monotone":          is_monotone,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("L22 INTERVENTION SUMMARY")
    print("=" * 60)
    print(f"\nControl feature: {control_feature_id}")
    for s in summary_by_alpha:
        print(f"\nalpha={s['alpha']:.1f}:")
        print(f"  Target improved:   {s['n_target_improved']}/{n_total}  (flipped: {s['n_target_flipped']})")
        print(f"  Control improved:  {s['n_control_improved']}/{n_total}  (flipped: {s['n_control_flipped']})")
        print(f"  Mean Δlogit  target: {s['mean_target_delta_logit']:+.4f} ± {s['std_target_delta_logit']:.4f}")
        print(f"  Mean Δlogit control: {s['mean_control_delta_logit']:+.4f} ± {s['std_control_delta_logit']:.4f}")
        print(f"  L22 vs control Δ:  {s['l22_vs_control_mean_delta']:+.4f} "
              f"({'BETTER' if s['l22_is_better_than_control'] else 'NOT BETTER'})")
        print(f"  Type-A improved:   {s['n_type_a_improved']}/{s['n_type_a']}, flipped: {s['n_type_a_flipped']}")
        print(f"  Type-B worsened:   {s['n_type_b_worsened']}/{s['n_type_b']}")

    print(f"\nDose-response: {' → '.join(f'{a:.1f}:{d:+.4f}' for a, d in zip(args.alphas, mean_deltas))}")
    print(f"Monotone: {is_monotone}")
    print(f"\nCritical interpretations:")
    print(f"  Q1: {q1_answer}")
    print(f"  Q2: {q2_answer}")
    print(f"  Q3: {q3_answer}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "behaviour":            args.behaviour,
        "split":                args.split,
        "target_features":      list(TARGET_FEATURES.keys()),
        "control_feature_id":   control_feature_id,
        "control_mean_baseline_activation": round(control_mean_act, 6),
        "alphas_tested":        args.alphas,
        "activation_threshold": ACTIVATION_THRESHOLD,
        "per_prompt":           per_prompt,
        "summary_by_alpha":     summary_by_alpha,
        "activation_summary":   activation_summary,
        "interpretations":      interpretations,
    }
    out_path = out_dir / f"l22_intervention_results_{args.split}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
