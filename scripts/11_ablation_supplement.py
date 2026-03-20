#!/usr/bin/env python3
"""
scripts/11_ablation_supplement.py

Supplement ablation-zero measurements for circuit features that script 07 did not
cover because they were not in the star-graph top-k used by that run.

This closes the late-layer hub blindspot in the reasoning-trace analysis.

Background:
  - Script 07 ran ablation_zero against the STAR graph (91 nodes), producing
    per-prompt measurements for 10/27 circuit features.
  - Script 08 discovered 17 additional features via causal-edge detection AFTER
    script 07 ran.  These appear only as "patching" in interventions.csv.
  - Patching has cross-language semantics incompatible with per-prompt trajectory
    computation, so those 17 features are currently "blind" in script 10.
  - This script runs ablation_zero for exactly those 17 features × all prompts.

Ablation method (identical to script 07 run_ablation_experiment):
  1. Baseline logit_diff cached once per prompt (96 forward passes total).
  2. For each (layer, feature_idx): encode MLP input → zero feature → decode →
     patch post_attention_layernorm output → run intervened forward pass.
  3. effect_size = intervened_logit_diff - baseline_logit_diff  (SIGNED, same convention).

Reads:
  data/results/causal_edges/{behaviour}/circuits_{behaviour}_{split}.json
  data/ui_offline/{run_id}/interventions.csv   (to identify which features are already measured)
  data/prompts/{behaviour}_{split}.jsonl
  configs/transcoder_config.yaml

Writes:
  data/results/reasoning_traces/{behaviour}/ablation_supplement_{split}.csv

Requires GPU (model + transcoders for layers 10,11,22,23,24,25).
Estimated: 17 features × 96 prompts = 1,632 intervened passes + 96 baseline passes.
           ~15 min on Ampere A100 (30-min SLURM slot is safe).
"""

import argparse
import dataclasses
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IO_NODES = {"input", "output_correct", "output_incorrect"}


# ─── Core ablation utilities (mirror of script 07) ────────────────────────────
# These replicate the exact same hook + encode/zero/decode pattern as
# TranscoderInterventionExperiment.run_ablation_experiment() in script 07.
# Kept self-contained to avoid fragile importlib from a numbered filename.

@contextmanager
def _patch_mlp_input(model_hf, layer_idx: int, token_pos: int, new_mlp_input: torch.Tensor):
    """
    Hook post_attention_layernorm at layer_idx to replace its output at token_pos.
    Identical to patch_mlp_input() in script 07.
    """
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
    exc = None
    try:
        yield
    except Exception as e:
        exc = e
        raise
    finally:
        handle.remove()
        if exc is None:
            assert hook_called["count"] > 0, (
                f"MLP hook did not fire at layer {layer_idx}. "
                f"Check model architecture or use_cache settings."
            )


def _get_mlp_input(model: ModelWrapper, inputs: Dict, layer_idx: int, token_pos: int = -1) -> torch.Tensor:
    """Extract MLP-input activation (post_attention_layernorm output) at token_pos."""
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


def _compute_logit_diff(
    model: ModelWrapper,
    prompt: str,
    correct_token: str,
    incorrect_token: str,
    device: str,
) -> Tuple[float, int, int]:
    """
    Returns (logit_diff, correct_id, incorrect_id).
    logit_diff = log_softmax[correct] - log_softmax[incorrect] at last token.
    """
    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=0)

    def token_id(tok: str) -> int:
        ids = model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token '{tok}' encodes to {len(ids)} sub-tokens: {ids}")
        return ids[0]

    cid = token_id(correct_token)
    iid = token_id(incorrect_token)
    diff = (log_probs[cid] - log_probs[iid]).item()
    return diff, cid, iid


def run_ablation_zero(
    model: ModelWrapper,
    transcoder_set,
    prompt: str,
    prompt_idx: int,
    correct_token: str,
    incorrect_token: str,
    layer: int,
    feature_idx: int,
    baseline_cache: Dict[int, Tuple[float, int, int]],
    device: str,
) -> Dict:
    """
    Run single-feature ablation_zero for (layer, feature_idx) on one prompt.
    Caches the baseline in baseline_cache[prompt_idx] to avoid redundant passes.

    Returns a dict matching the interventions.csv schema.
    """
    # Baseline (cached)
    if prompt_idx not in baseline_cache:
        bld, cid, iid = _compute_logit_diff(model, prompt, correct_token, incorrect_token, device)
        baseline_cache[prompt_idx] = (bld, cid, iid)
    baseline_ld, cid, iid = baseline_cache[prompt_idx]

    # Get MLP input at this layer
    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    mlp_input = _get_mlp_input(model, inputs, layer_idx=layer, token_pos=-1)
    mlp_input = mlp_input.to(device)

    # Encode → zero feature → decode
    tc = transcoder_set[layer]
    with torch.no_grad():
        feats = tc.encode(mlp_input.to(tc.dtype))
        feats[:, feature_idx] = 0.0
        modified_mlp_input = tc.decode(feats).to(mlp_input.dtype)

    # Intervened forward pass
    with torch.no_grad():
        with _patch_mlp_input(model.model, layer_idx=layer, token_pos=-1,
                               new_mlp_input=modified_mlp_input):
            out = model.model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :]

    log_probs = torch.log_softmax(logits, dim=0)
    intervened_ld = (log_probs[cid] - log_probs[iid]).item()
    effect_size = intervened_ld - baseline_ld

    eps = 1e-6
    bs = 1 if baseline_ld > eps else (-1 if baseline_ld < -eps else 0)
    is_ = 1 if intervened_ld > eps else (-1 if intervened_ld < -eps else 0)
    sign_flipped = bool(bs != 0 and is_ != 0 and bs != is_)

    return {
        "experiment_type": "ablation_zero",
        "prompt_idx": prompt_idx,
        "layer": layer,
        "feature_indices": f"[{feature_idx}]",
        "baseline_logit_diff": baseline_ld,
        "intervened_logit_diff": intervened_ld,
        "effect_size": effect_size,
        "abs_effect_size": abs(effect_size),
        "relative_effect": abs(effect_size) / (abs(baseline_ld) + 1e-8),
        "sign_flipped": sign_flipped,
        "feature_source": "graph",
        "skipped_reason": None,
        "layer_has_graph_features": True,
        "feature_id": f"L{layer}_F{feature_idx}",
        "concept_index": -1,   # filled below from prompt metadata
        "meta.prompt": prompt[:120],
        "meta.correct_token": correct_token,
        "meta.incorrect_token": incorrect_token,
        "meta.mode": "zero",
        "source_script": "11_ablation_supplement",
    }


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Supplement ablation-zero for circuit features missing from script 07 run."
    )
    p.add_argument("--behaviour", default="multilingual_circuits_b1")
    p.add_argument("--split", default="train")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--run_id", default=None,
                   help="UI offline run_id to read existing coverage from (auto-detected)")
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: data/results/reasoning_traces/{behaviour})")
    p.add_argument("--device", default=None,
                   help="Device (auto-detect cuda if not specified)")
    p.add_argument("--target_features", nargs="*", default=None,
                   help="Restrict to specific feature IDs e.g. L23_F64429 L24_F136810. "
                        "Default: all unmeasured circuit features.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan and exit without running any forward passes.")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_run_id(behaviour: str, split: str) -> str:
    ui_dir = Path("data/ui_offline")
    matches = sorted(ui_dir.glob(f"*_{behaviour}_{split}_*"))
    if not matches:
        raise FileNotFoundError(f"No UI offline run for {behaviour}/{split} in {ui_dir}")
    run_id = matches[-1].name
    logger.info(f"Auto-detected run_id: {run_id}")
    return run_id


def get_unmeasured_features(circuit_json: dict, df_existing: pd.DataFrame, target_features=None) -> List[str]:
    """
    Return circuit features that do NOT yet have ablation_zero rows in df_existing.
    Excludes I/O nodes.
    """
    circuit_features = [
        n for n in circuit_json["circuit"]["feature_nodes"]
        if n not in IO_NODES
    ]
    az_covered = set(
        df_existing.loc[df_existing["experiment_type"] == "ablation_zero", "feature_id"].unique()
    )
    unmeasured = [f for f in circuit_features if f not in az_covered]

    if target_features is not None:
        target_set = set(target_features)
        unmeasured = [f for f in unmeasured if f in target_set]
        extra = target_set - set(unmeasured)
        if extra:
            logger.warning(
                f"Some --target_features are already measured or not in circuit: {extra}"
            )
    return unmeasured


def parse_feature_id(fid: str) -> Tuple[int, int]:
    """'L23_F64429' → (23, 64429)"""
    layer_part, feat_part = fid.split("_")
    return int(layer_part[1:]), int(feat_part[1:])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"Ablation Supplement — {args.behaviour}/{args.split}")
    print("=" * 60)

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"data/results/reasoning_traces/{args.behaviour}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load circuit ─────────────────────────────────────────────────────────
    circuit_path = Path(
        f"data/results/causal_edges/{args.behaviour}/"
        f"circuits_{args.behaviour}_{args.split}.json"
    )
    circuit = json.loads(circuit_path.read_text())
    logger.info(f"Circuit: {circuit['circuit']['n_features']} features, "
                f"{circuit['circuit']['n_edges']} edges")

    # ── Load existing interventions to find coverage ──────────────────────────
    run_id = args.run_id or find_run_id(args.behaviour, args.split)
    existing_csv = Path(f"data/ui_offline/{run_id}/interventions.csv")
    df_existing = pd.read_csv(existing_csv)
    logger.info(f"Existing interventions: {len(df_existing)} rows from {run_id}")

    # ── Determine which features to measure ──────────────────────────────────
    unmeasured = get_unmeasured_features(circuit, df_existing, args.target_features)
    if not unmeasured:
        print("All circuit features already have ablation_zero coverage. Nothing to do.")
        return

    # Group by layer
    by_layer: Dict[int, List[int]] = {}
    for fid in unmeasured:
        layer, feat_idx = parse_feature_id(fid)
        by_layer.setdefault(layer, []).append(feat_idx)

    n_total = sum(len(v) for v in by_layer.values())
    print(f"\nFeatures to supplement: {n_total} ({len(unmeasured)} features across {len(by_layer)} layers)")
    for layer in sorted(by_layer):
        print(f"  L{layer}: {sorted(by_layer[layer])}")

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = [
        json.loads(l)
        for l in Path(f"data/prompts/{args.behaviour}_{args.split}.jsonl").read_text().splitlines()
        if l.strip()
    ]
    n_prompts = len(prompts)
    print(f"\nPrompts: {n_prompts}")
    print(f"Total forward passes: {n_total * n_prompts} intervened + {n_prompts} baseline = "
          f"{n_total * n_prompts + n_prompts} total")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running forward passes.")
        return

    # ── Device ────────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    tc_config = yaml.safe_load(open("configs/transcoder_config.yaml"))
    model_name = tc_config["transcoders"][args.model_size]["model_name"]
    print(f"Loading model {model_name}...")
    model = ModelWrapper(
        model_name=model_name,
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"  Model loaded on {device}")

    # ── Load transcoders (only required layers) ───────────────────────────────
    layers_needed = sorted(by_layer.keys())
    print(f"Loading transcoders for layers {layers_needed}...")
    transcoder_set = load_transcoder_set(
        model_size=args.model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=False,
        layers=layers_needed,
    )
    print(f"  Transcoders loaded for {len(layers_needed)} layers")

    # ── Run ablations ─────────────────────────────────────────────────────────
    baseline_cache: Dict[int, Tuple[float, int, int]] = {}
    results = []
    timestamp = datetime.now().isoformat()

    for layer in sorted(by_layer.keys()):
        feat_indices = sorted(by_layer[layer])
        print(f"\n── Layer {layer}: {len(feat_indices)} features ──")
        for feat_idx in feat_indices:
            fid = f"L{layer}_F{feat_idx}"
            print(f"  Ablating {fid} across {n_prompts} prompts...")
            for prompt_idx, p in enumerate(tqdm(prompts, desc=fid, leave=False)):
                correct_token = p.get("correct_answer", p.get("answer_matching", ""))
                incorrect_token = p.get("incorrect_answer", p.get("answer_not_matching", ""))
                row = run_ablation_zero(
                    model=model,
                    transcoder_set=transcoder_set,
                    prompt=p["prompt"],
                    prompt_idx=prompt_idx,
                    correct_token=correct_token,
                    incorrect_token=incorrect_token,
                    layer=layer,
                    feature_idx=feat_idx,
                    baseline_cache=baseline_cache,
                    device=device,
                )
                row["concept_index"] = p.get("concept_index", -1)
                row["meta.template_idx"] = p.get("template_idx", -1)
                row["meta.language"] = p.get("language", "?")
                row["behaviour"] = args.behaviour
                row["split"] = args.split
                row["model_size"] = args.model_size
                row["run_id"] = run_id
                row["prep_timestamp"] = timestamp
                results.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = out_dir / f"ablation_supplement_{args.split}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    # ── Coverage report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)

    circuit_features = [
        n for n in circuit["circuit"]["feature_nodes"] if n not in IO_NODES
    ]
    n_circuit = len(circuit_features)
    az_original = set(
        df_existing.loc[df_existing["experiment_type"] == "ablation_zero", "feature_id"].unique()
    ) & set(circuit_features)
    az_new = set(df["feature_id"].unique()) & set(circuit_features)
    az_total = az_original | az_new

    print(f"Circuit features total:      {n_circuit}")
    print(f"Ablation-zero original:      {len(az_original)}/{n_circuit}  (script 07)")
    print(f"Ablation-zero supplement:    {len(az_new)}/{n_circuit}  (script 11, this run)")
    print(f"Ablation-zero combined:      {len(az_total)}/{n_circuit}")

    # Check key features
    key_late = ["L23_F64429", "L24_F136810", "L24_F134204", "L24_F29680"]
    print("\nKey late-hub features:")
    for fid in key_late:
        if fid in az_total:
            src = "original" if fid in az_original else "SUPPLEMENT (NEW)"
            print(f"  {fid}: measured [{src}]")
        else:
            print(f"  {fid}: STILL MISSING")

    # Effect size summary for new features
    print("\nEffect size summary (supplement features, mean |effect_size| per feature):")
    fid_stats = df.groupby("feature_id")["abs_effect_size"].mean().sort_values(ascending=False)
    for fid, mean_abs in fid_stats.items():
        neg_frac = (df.loc[df["feature_id"] == fid, "effect_size"] < 0).mean()
        print(f"  {fid}: mean |effect|={mean_abs:.4f}, "
              f"neg_frac={neg_frac:.2%} (neg=helps correct)")

    print(f"\nDone. Supplement CSV: {out_path}")
    print("Next: run script 10 with --supplement_csv to get updated reasoning traces.")
    print(f"  python scripts/10_reasoning_trace.py \\")
    print(f"    --behaviour {args.behaviour} --split {args.split} \\")
    print(f"    --supplement_csv {out_path}")


if __name__ == "__main__":
    main()
