"""
51_recon_iia_baseline.py

Computes the reconstruction baseline for the transcoder intervention pipeline.

For each prompt p and each layer ℓ, measures the effect of the trivial
"no-op" intervention:

    x̃ = dec(enc(x))    [encode → decode without modifying any features]

then patches x̃ back as the MLP input via hook and measures:
    Δ_recon(p, ℓ) = Δ(model with x̃) − Δ(model with x)

Key outputs:
    - mean |Δ_recon|   : reconstruction noise level per layer
    - sign_flip_rate   : fraction of prompts where sign flips from enc→dec alone
    - recon_iia        : fraction of prompts where sign(Δ_recon) = sign(Δ_orig)
                         (expected ≈ 1.0 if reconstruction is faithful)

This establishes the noise floor for ablation/patching experiments.
Expected result if transcoder is faithful: mean |Δ_recon| ≪ mean |Δ_ablation|
and sign_flip_rate ≈ 0.

Usage:
    python scripts/51_recon_iia_baseline.py \
        --behaviour physics_decay_type_probe \
        --split train \
        --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
        --max_prompts 100 \
        --output_dir data/analysis/recon_baseline

On CSD3 (GPU):
    python scripts/51_recon_iia_baseline.py --behaviour physics_decay_type_probe \
        --split train --layers 10 25 --max_prompts 470
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers copied / adapted from script 07 ──────────────────────────────────

def ensure_single_token(model: ModelWrapper, tok: str) -> int:
    ids = model.tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) != 1:
        logger.warning(f"Token '{tok}' encodes to {len(ids)} sub-tokens: {ids}. Using first.")
    return ids[0]


def get_mlp_input(model: ModelWrapper, prompt: str, layer_idx: int,
                  device: str) -> torch.Tensor:
    """Capture post_attention_layernorm output at decision token (last position)."""
    inputs = model.tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    captured = {}

    block = model.model.model.layers[layer_idx]
    hook_mod = block.post_attention_layernorm

    def _hook(module, inp, out):
        captured["x"] = out.detach()          # (1, seq, d)

    handle = hook_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert "x" in captured, f"Hook didn't fire at layer {layer_idx}"
    return captured["x"][0, -1:, :]           # (1, d) — decision token only


def compute_logit_diff(model: ModelWrapper, prompt: str,
                       correct_tok: str, incorrect_tok: str,
                       device: str) -> float:
    """Clean (no hook) forward pass → log P(correct) − log P(incorrect)."""
    inputs = model.tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.model(**inputs, use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = ensure_single_token(model, correct_tok)
    iid = ensure_single_token(model, incorrect_tok)
    return (lp[cid] - lp[iid]).item()


def compute_logit_diff_with_patch(model: ModelWrapper, prompt: str,
                                  correct_tok: str, incorrect_tok: str,
                                  layer_idx: int, x_hat: torch.Tensor,
                                  device: str) -> float:
    """Forward pass with x_hat patched as MLP input at layer_idx → logit diff.

    x_hat: shape (1, d_model) — decoded reconstruction at decision token.
    """
    inputs = model.tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    block = model.model.model.layers[layer_idx]
    hook_mod = block.post_attention_layernorm
    called = {"n": 0}

    # x_hat is (1, d); out is (1, seq, d) — replace last token in-place
    _x = x_hat.to(device)

    def _hook(module, inp, out):
        called["n"] += 1
        out_mod = out.clone()
        out_mod[0, -1, :] = _x[0, :]
        return out_mod

    handle = hook_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = model.model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert called["n"] > 0, f"Patch hook didn't fire at layer {layer_idx}"
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = ensure_single_token(model, correct_tok)
    iid = ensure_single_token(model, incorrect_tok)
    return (lp[cid] - lp[iid]).item()


# ── main computation ──────────────────────────────────────────────────────────

def run_recon_baseline(
    model: ModelWrapper,
    transcoder_set,
    prompts: List[Dict],
    layers: List[int],
    device: str,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each (prompt, layer) pair, compute the reconstruction baseline:
        x̃ = dec(enc(x))   [no feature modification]
    and measure Δ_recon vs Δ_orig.

    Returns a DataFrame with one row per (prompt_idx, layer).
    """
    if max_prompts:
        prompts = prompts[:max_prompts]

    rows = []

    for pi, p in enumerate(prompts):
        prompt_text = p["prompt"]
        correct     = p["correct_answer"]
        incorrect   = p["incorrect_answer"]
        prompt_id   = p.get("prompt_id", str(pi))
        group_id    = p.get("group_id", "")
        level_label = p.get("level_label", "")
        correct_ans = correct.strip()

        # Baseline logit diff (clean, no hooks)
        try:
            delta_orig = compute_logit_diff(model, prompt_text, correct, incorrect, device)
        except Exception as e:
            logger.warning(f"Prompt {prompt_id}: baseline failed — {e}")
            continue

        if pi % 20 == 0:
            logger.info(f"  prompt {pi+1}/{len(prompts)}  delta_orig={delta_orig:.3f}")

        for layer in layers:
            if layer not in transcoder_set._transcoders:
                continue

            tc = transcoder_set[layer]

            try:
                x = get_mlp_input(model, prompt_text, layer, device)   # (1, d)
                x_tc = x.to(tc.dtype)

                # ── enc → dec, no modification ─────────────────────────────
                with torch.no_grad():
                    a     = tc.encode(x_tc)          # (1, d_tc)
                    x_hat = tc.decode(a)             # (1, d)   ← x̃

                # Reconstruction error in MLP-input space
                recon_err = (x_hat - x_tc).norm().item()

                # Logit diff with x̃ patched in  — x_hat is (1, d)
                delta_recon = compute_logit_diff_with_patch(
                    model, prompt_text, correct, incorrect,
                    layer, x_hat.to(x.dtype), device
                )

            except Exception as e:
                logger.debug(f"Layer {layer}, prompt {prompt_id}: {e}")
                continue

            effect = delta_recon - delta_orig          # Δ_recon − Δ_orig

            rows.append({
                "prompt_idx":    pi,
                "prompt_id":     prompt_id,
                "group_id":      group_id,
                "level_label":   level_label,
                "correct_answer": correct_ans,
                "layer":         layer,
                "delta_orig":    delta_orig,
                "delta_recon":   delta_recon,
                "effect":        effect,           # Δ_recon − Δ_orig
                "abs_effect":    abs(effect),
                "sign_flipped":  int(np.sign(delta_recon) != np.sign(delta_orig)),
                "recon_err_norm": recon_err,
            })

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> Dict:
    """Per-layer summary statistics."""
    summary = {}
    for layer, grp in df.groupby("layer"):
        summary[int(layer)] = {
            "n_prompts":         int(len(grp)),
            "mean_abs_effect":   float(grp["abs_effect"].mean()),
            "median_abs_effect": float(grp["abs_effect"].median()),
            "std_effect":        float(grp["effect"].std()),
            "sign_flip_rate":    float(grp["sign_flipped"].mean()),
            "mean_recon_err":    float(grp["recon_err_norm"].mean()),
            # recon_iia: fraction where sign(Δ_recon) == sign(Δ_orig)
            # (the reconstruction preserved the original answer direction)
            "recon_iia":         float(1.0 - grp["sign_flipped"].mean()),
        }

    # Also report per answer class
    for (layer, ans), grp in df.groupby(["layer", "correct_answer"]):
        key = f"L{layer}_{ans}"
        summary[key] = {
            "n":              int(len(grp)),
            "sign_flip_rate": float(grp["sign_flipped"].mean()),
            "mean_abs_effect":float(grp["abs_effect"].mean()),
        }

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Reconstruction IIA baseline")
    p.add_argument("--behaviour",   default="physics_decay_type_probe")
    p.add_argument("--split",       default="train")
    p.add_argument("--prompts_file", default=None,
                   help="Override default JSONL path")
    p.add_argument("--layers", nargs="+", type=int,
                   default=list(range(10, 26)),
                   help="Layers to evaluate (default: 10-25)")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap number of prompts (for quick testing)")
    p.add_argument("--output_dir", default="data/analysis/recon_baseline")
    p.add_argument("--model_name",  default=None,
                   help="Override model from transcoder config")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--v2_corpus",   action="store_true",
                   help="Use _v2_train suffix (new symmetric corpus)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve paths ────────────────────────────────────────────────────────
    root = Path(__file__).parent.parent
    out_dir = root / args.output_dir / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    # prompt file
    if args.prompts_file:
        prompt_path = Path(args.prompts_file)
    else:
        suffix = "_v2_train" if args.v2_corpus else f"_{args.split}"
        prompt_path = root / "data" / "prompts" / f"{args.behaviour}{suffix}.jsonl"
    assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"

    prompts = []
    with open(prompt_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")

    # ── load model & transcoders ─────────────────────────────────────────────
    import yaml
    tc_cfg_path = root / "configs" / "transcoder_config.yaml"
    with open(tc_cfg_path) as f:
        tc_cfg = yaml.safe_load(f)

    model_size  = tc_cfg["model_size"]
    tc_info     = tc_cfg["transcoders"][model_size]
    model_name  = args.model_name or tc_info["model_name"]

    logger.info(f"Loading model: {model_name}")
    model = ModelWrapper(model_name, device=args.device)

    logger.info(f"Loading transcoders for layers {args.layers}")
    transcoder_set = load_transcoder_set(
        repo_id   = tc_info["repo_id"],
        layers    = args.layers,
        device    = args.device,
    )

    # ── run ──────────────────────────────────────────────────────────────────
    logger.info("Computing reconstruction baselines…")
    df = run_recon_baseline(
        model          = model,
        transcoder_set = transcoder_set,
        prompts        = prompts,
        layers         = args.layers,
        device         = args.device,
        max_prompts    = args.max_prompts,
    )

    if df.empty:
        logger.error("No results — check model/transcoder setup")
        sys.exit(1)

    # ── save ─────────────────────────────────────────────────────────────────
    csv_path  = out_dir / f"recon_baseline_{args.behaviour}_{args.split}.csv"
    json_path = out_dir / f"recon_baseline_{args.behaviour}_{args.split}_summary.json"

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw results → {csv_path}  ({len(df)} rows)")

    summary = summarise(df)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary      → {json_path}")

    # ── print quick summary ──────────────────────────────────────────────────
    print("\n=== RECONSTRUCTION BASELINE SUMMARY ===")
    print(f"{'Layer':>6}  {'n':>5}  {'mean|Δ|':>9}  {'sign_flip%':>11}  {'recon_iia':>10}  {'recon_err':>10}")
    print("-" * 62)
    layer_keys = [k for k in summary if isinstance(k, int)]
    for layer in sorted(layer_keys):
        s = summary[layer]
        print(f"  L{layer:>2d}   {s['n_prompts']:>5}  "
              f"{s['mean_abs_effect']:>9.4f}  "
              f"{s['sign_flip_rate']*100:>10.1f}%  "
              f"{s['recon_iia']:>10.4f}  "
              f"{s['mean_recon_err']:>10.4f}")

    overall_flip = df["sign_flipped"].mean()
    overall_eff  = df["abs_effect"].mean()
    print(f"\nOverall sign_flip_rate : {overall_flip*100:.2f}%")
    print(f"Overall mean |Δ_recon| : {overall_eff:.4f}")
    print(f"\nIf sign_flip_rate ≈ 0 and mean|Δ| is small → transcoder reconstruction is faithful.")
    print(f"If sign_flip_rate > 5% → reconstruction noise inflates causal estimates.")


if __name__ == "__main__":
    main()
