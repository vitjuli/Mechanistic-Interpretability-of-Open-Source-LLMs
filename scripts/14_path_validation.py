#!/usr/bin/env python3
"""
Path-centric causal validation (FIX 3).

For each top path in the circuit, ablate the path head (first intermediate
feature A) and measure:
  1. mean_delta_act_B   — mean |Δactivation| at path node B (next feature)
  2. mean_delta_logit   — mean |Δlogit_correct - Δlogit_incorrect| at output
  3. propagation_consistency — fraction of prompts where |Δact_B| > epsilon

Uses a two-hook setup:
  Hook 1: post_attention_layernorm at layer L_A → ablates feature A (write)
  Hook 2: post_attention_layernorm at layer L_B → captures MLP input (read-only)

Transcoder encoder is applied to the captured MLP input to get feature B
activation. Activation is taken at the decision token position.

Usage:
    python scripts/14_path_validation.py \
        --behaviour multilingual_circuits_b1 \
        --split train \
        --graph_n_prompts 96 \
        --n_paths 10 \
        --epsilon 0.01

    python scripts/14_path_validation.py \
        --behaviour multilingual_circuits_b1 \
        --split train \
        --circuits_json data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json \
        --n_paths 10
"""

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str = "configs/experiment_config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_transcoder_config(path: str = "configs/transcoder_config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_circuit(behaviour: str, split: str, config: Dict) -> Dict:
    results_dir = Path(config["paths"]["results"])
    path = results_dir / "causal_edges" / behaviour / f"circuits_{behaviour}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Circuit JSON not found: {path}\n  Run scripts/08_causal_edges.py first."
        )
    return json.loads(path.read_text())


def load_prompts(config: Dict, behaviour: str, split: str) -> List[Dict]:
    prompt_path = Path(config["paths"]["prompts"]) / f"{behaviour}_{split}.jsonl"
    prompts = []
    with open(prompt_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


# ── Node ID parsing ───────────────────────────────────────────────────────────

def parse_node_id(node_id: str) -> Optional[Tuple[int, int]]:
    """Parse 'L{layer}_F{feat}' → (layer, feat). Returns None for I/O nodes."""
    if not node_id.startswith("L"):
        return None
    try:
        layer_part, feat_part = node_id.split("_F")
        layer = int(layer_part[1:])
        feat = int(feat_part)
        return layer, feat
    except (ValueError, AttributeError):
        return None


# ── Block / hook helpers ──────────────────────────────────────────────────────

def get_block(model_hf, layer_idx: int):
    try:
        return model_hf.model.layers[layer_idx]
    except AttributeError:
        try:
            return model_hf.transformer.h[layer_idx]
        except AttributeError:
            raise RuntimeError(f"Cannot locate transformer block {layer_idx}")


def get_layernorm(block, layer_idx: int):
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    if hasattr(block, "ln_2"):
        return block.ln_2
    raise RuntimeError(
        f"Cannot find post_attention_layernorm in layer {layer_idx}. "
        "Check model architecture."
    )


@contextmanager
def ablate_feature_hook(model_hf, layer_idx: int, feat_idx: int,
                        transcoder, token_pos: int = -1):
    """
    Hook 1: ablate feature feat_idx at layer_idx by zeroing its contribution
    to the MLP input reconstruction.

    Mechanism:
      - Capture MLP input x from post_attention_layernorm
      - Run transcoder encode: acts = relu(x @ W_enc.T + b_enc)
      - Zero acts[:, :, feat_idx]
      - Reconstruct modified MLP input: x_mod = acts @ W_dec + b_dec + x  (transcoder residual)
      - Replace hook output with x_mod

    Note: uses the transcoder's full encode→zero→decode pipeline so the
    ablation is in transcoder feature space, consistent with script 07.
    """
    block = get_block(model_hf, layer_idx)
    hook_mod = get_layernorm(block, layer_idx)
    hook_fired = {"n": 0}

    def _hook(module, inp, out):
        hook_fired["n"] += 1
        h = out[0] if isinstance(out, tuple) else out
        # Decision token only — cast to transcoder dtype for matmul compatibility
        tc_dtype = transcoder.W_enc.dtype
        tc_device = transcoder.W_enc.device
        x = h[:, token_pos, :].to(dtype=tc_dtype, device=tc_device)  # (1, d_model)
        with torch.no_grad():
            acts = torch.relu(x @ transcoder.W_enc.T + transcoder.b_enc)  # (1, d_tc)
            acts[:, feat_idx] = 0.0
            x_rec = acts @ transcoder.W_dec + transcoder.b_dec  # (1, d_model)
            # Transcoder is a replacement MLP: output = reconstruction
            # Ablation: patch the transcoder output back, not residual
            # We replace the full MLP-input position so the MLP sees ablated x_rec.
            x_mod = h.clone()
            x_mod[:, token_pos, :] = x_rec.to(h.dtype)
        if isinstance(out, tuple):
            return (x_mod,) + out[1:]
        return x_mod

    handle = hook_mod.register_forward_hook(_hook)
    try:
        yield hook_fired
    finally:
        handle.remove()
        if hook_fired["n"] == 0:
            logger.warning(
                f"Ablation hook at layer {layer_idx} did not fire! "
                "Check token_pos and model architecture."
            )


@contextmanager
def capture_activation_hook(model_hf, layer_idx: int, token_pos: int = -1):
    """
    Hook 2: capture MLP input (post_attention_layernorm output) at layer_idx.
    Read-only — does NOT modify the forward pass.

    Returns a shared dict with key 'activation' set after the context exits.
    """
    block = get_block(model_hf, layer_idx)
    hook_mod = get_layernorm(block, layer_idx)
    captured = {"activation": None}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["activation"] = h[:, token_pos, :].detach().float().cpu()

    handle = hook_mod.register_forward_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()


# ── Core validation ───────────────────────────────────────────────────────────

def get_feature_activation(mlp_input: torch.Tensor, transcoder,
                            feat_idx: int) -> float:
    """Run transcoder encoder on mlp_input, return activation of feat_idx."""
    tc_dtype = transcoder.W_enc.dtype
    x = mlp_input.to(dtype=tc_dtype, device=transcoder.W_enc.device)
    with torch.no_grad():
        acts = torch.relu(x @ transcoder.W_enc.T + transcoder.b_enc)
    return float(acts[0, feat_idx])


def run_forward(model: ModelWrapper, inputs: Dict) -> Tuple[float, float]:
    """Run forward pass, return (logit_correct, logit_incorrect)."""
    with torch.no_grad():
        out = model.model(**inputs)
    logits = out.logits[0, -1, :]  # last token
    return float(logits[0]), float(logits[1])  # placeholder — see below


def get_logit_diff(model: ModelWrapper, inputs: Dict,
                   correct_token_id: int, incorrect_token_id: int) -> float:
    """Return logit(correct) - logit(incorrect)."""
    with torch.no_grad():
        out = model.model(**inputs)
    logits = out.logits[0, -1, :]
    return float(logits[correct_token_id]) - float(logits[incorrect_token_id])


def validate_path(
    path_nodes: List[str],
    prompts: List[Dict],
    model: ModelWrapper,
    transcoder_set: TranscoderSet,
    epsilon: float = 0.01,
    max_prompts: int = 50,
) -> Dict:
    """
    Validate a single path by ablating node A and measuring Δact_B and Δlogit.

    Args:
        path_nodes: List of node IDs e.g. ["input", "L21_F27974", "L22_F41906", "output_correct"]
        prompts: Full prompt list (used to get token IDs)
        model: Loaded ModelWrapper
        transcoder_set: Loaded TranscoderSet
        epsilon: Threshold for propagation_consistency (default 0.01)
        max_prompts: Max prompts to evaluate (cap for speed)

    Returns:
        dict with mean_delta_act_B, mean_delta_logit, propagation_consistency,
        n_prompts_evaluated, path, A_id, B_id
    """
    # Find first intermediate pair: A → B (both must be feature nodes)
    feature_nodes = [n for n in path_nodes if parse_node_id(n) is not None]
    if len(feature_nodes) < 2:
        return {
            "path": path_nodes,
            "skip_reason": "fewer than 2 feature nodes",
            "mean_delta_act_B": None,
            "mean_delta_logit": None,
            "propagation_consistency": None,
            "n_prompts_evaluated": 0,
        }

    a_id = feature_nodes[0]
    b_id = feature_nodes[1]
    layer_a, feat_a = parse_node_id(a_id)
    layer_b, feat_b = parse_node_id(b_id)

    # Determine output tokens from prompts
    # Prompts have 'correct_answer' and 'incorrect_answer' keys (or token IDs)
    tokenizer = model.tokenizer

    delta_acts = []
    delta_logits = []

    eval_prompts = prompts[:max_prompts]

    for prompt in eval_prompts:
        text = prompt.get("prompt", prompt.get("text", ""))
        correct = prompt.get("correct_answer", " True")
        incorrect = prompt.get("incorrect_answer", " False")

        correct_ids = tokenizer.encode(correct, add_special_tokens=False)
        incorrect_ids = tokenizer.encode(incorrect, add_special_tokens=False)
        if not correct_ids or not incorrect_ids:
            continue
        correct_tok = correct_ids[0]
        incorrect_tok = incorrect_ids[0]

        inputs = tokenizer(text, return_tensors="pt").to(model.model.device)

        tc_a = transcoder_set[layer_a]
        tc_b = transcoder_set[layer_b]

        # ── Baseline: no ablation ─────────────────────────────────────────
        with capture_activation_hook(model.model, layer_b) as cap_b_base:
            logit_base = get_logit_diff(model, inputs, correct_tok, incorrect_tok)
        act_b_base = get_feature_activation(cap_b_base["activation"].to(tc_b.W_enc.device), tc_b, feat_b)

        # ── Ablated: zero feature A at layer A ────────────────────────────
        with ablate_feature_hook(model.model, layer_a, feat_a, tc_a):
            with capture_activation_hook(model.model, layer_b) as cap_b_abl:
                logit_abl = get_logit_diff(model, inputs, correct_tok, incorrect_tok)
        act_b_abl = get_feature_activation(cap_b_abl["activation"].to(tc_b.W_enc.device), tc_b, feat_b)

        delta_act = abs(act_b_abl - act_b_base)
        delta_logit = abs(logit_abl - logit_base)

        delta_acts.append(delta_act)
        delta_logits.append(delta_logit)

    if not delta_acts:
        return {
            "path": path_nodes,
            "skip_reason": "no valid prompts",
            "A_id": a_id,
            "B_id": b_id,
            "mean_delta_act_B": None,
            "mean_delta_logit": None,
            "propagation_consistency": None,
            "n_prompts_evaluated": 0,
        }

    arr = np.array(delta_acts)
    return {
        "path": path_nodes,
        "A_id": a_id,
        "B_id": b_id,
        "layer_A": layer_a,
        "feat_A": feat_a,
        "layer_B": layer_b,
        "feat_B": feat_b,
        "mean_delta_act_B": float(arr.mean()),
        "mean_delta_logit": float(np.mean(delta_logits)),
        "propagation_consistency": float((arr > epsilon).mean()),
        "n_prompts_evaluated": len(delta_acts),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Path-centric causal validation (FIX 3).",
    )
    parser.add_argument("--behaviour", type=str, default="multilingual_circuits_b1")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument(
        "--circuits_json", type=str, default=None,
        help="Path to circuits JSON (overrides default config path).",
    )
    parser.add_argument(
        "--n_paths", type=int, default=10,
        help="Number of top paths to validate (default: 10).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Threshold for propagation_consistency: |Δact_B| > epsilon (default: 0.01).",
    )
    parser.add_argument(
        "--max_prompts", type=int, default=50,
        help="Max prompts per path (default: 50; cap for speed).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for results JSON. "
             "Default: data/results/path_validation/{behaviour}/",
    )
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument(
        "--transcoder_config", type=str, default="configs/transcoder_config.yaml",
    )
    parser.add_argument("--model_size", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    tc_config = load_transcoder_config(args.transcoder_config)

    # Load circuit
    if args.circuits_json:
        cpath = Path(args.circuits_json)
        if not cpath.exists():
            raise FileNotFoundError(f"circuits_json not found: {cpath}")
        circuit = json.loads(cpath.read_text())
    else:
        circuit = load_circuit(args.behaviour, args.split, config)

    paths = circuit.get("paths", [])
    if not paths:
        logger.error("No paths found in circuit JSON. Run script 08 first.")
        sys.exit(1)

    top_paths = paths[: args.n_paths]
    logger.info(f"Validating {len(top_paths)} paths (of {len(paths)} total)")

    # Load model and transcoders
    model_size = args.model_size or tc_config.get("model_size", "4b")
    model_name = tc_config["transcoders"][model_size]["model_name"]
    layers = tc_config.get("analysis_layers", {}).get("default", list(range(10, 26)))

    logger.info(f"Loading model: {model_name}")
    model = ModelWrapper(
        model_name=model_name,
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )

    device = next(model.model.parameters()).device
    logger.info("Loading transcoder set...")
    transcoder_set = load_transcoder_set(
        model_size=model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=layers,
    )

    # Load prompts
    prompts = load_prompts(config, args.behaviour, args.split)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Run path validation
    print("=" * 70)
    print("PATH-CENTRIC CAUSAL VALIDATION (FIX 3)")
    print("=" * 70)
    print(f"  Behaviour:   {args.behaviour}")
    print(f"  Split:       {args.split}")
    print(f"  Paths:       {len(top_paths)}")
    print(f"  Max prompts: {args.max_prompts}")
    print(f"  Epsilon:     {args.epsilon}")
    print("=" * 70)

    results = []
    for i, path_info in enumerate(top_paths):
        path_nodes = path_info["path"]
        rank = path_info.get("rank", i + 1)
        score = path_info.get("score", None)

        print(f"\nPath {rank} (score={score:.4f}): {' → '.join(path_nodes)}")

        res = validate_path(
            path_nodes=path_nodes,
            prompts=prompts,
            model=model,
            transcoder_set=transcoder_set,
            epsilon=args.epsilon,
            max_prompts=args.max_prompts,
        )
        res["rank"] = rank
        res["circuit_score"] = score
        results.append(res)

        if res.get("skip_reason"):
            print(f"  SKIPPED: {res['skip_reason']}")
        else:
            print(f"  A={res['A_id']} → B={res['B_id']}")
            print(f"  mean_delta_act_B:          {res['mean_delta_act_B']:.4f}")
            print(f"  mean_delta_logit:           {res['mean_delta_logit']:.4f}")
            print(f"  propagation_consistency:    {res['propagation_consistency']:.3f}  "
                  f"(n={res['n_prompts_evaluated']})")

    # Summary
    valid = [r for r in results if r.get("mean_delta_act_B") is not None]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if valid:
        print(f"  Paths validated:             {len(valid)} / {len(results)}")
        print(f"  Mean Δact_B:                 {np.mean([r['mean_delta_act_B'] for r in valid]):.4f}")
        print(f"  Mean Δlogit:                 {np.mean([r['mean_delta_logit'] for r in valid]):.4f}")
        print(f"  Mean propagation_consistency:{np.mean([r['propagation_consistency'] for r in valid]):.3f}")
    else:
        print("  No valid paths to summarize.")

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(config["paths"]["results"]) / "path_validation" / args.behaviour
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"path_validation_{args.behaviour}_{args.split}.json"

    output = {
        "behaviour": args.behaviour,
        "split": args.split,
        "n_paths_requested": args.n_paths,
        "n_paths_validated": len(valid),
        "epsilon": args.epsilon,
        "max_prompts": args.max_prompts,
        "summary": {
            "mean_delta_act_B": float(np.mean([r["mean_delta_act_B"] for r in valid])) if valid else None,
            "mean_delta_logit": float(np.mean([r["mean_delta_logit"] for r in valid])) if valid else None,
            "mean_propagation_consistency": float(np.mean([r["propagation_consistency"] for r in valid])) if valid else None,
        },
        "paths": results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved path validation results: {out_file}")
    print(f"\n  Output: {out_file}")


if __name__ == "__main__":
    main()
