"""
Script 27: Cluster-level joint ablation for physics_decay_type_probe.

For each co-importance Louvain cluster C = {f1, f2, ..., fn} spanning
layers {l1, l2, ..., lk}:

  1. Extract MLP input activations at each unique layer li in one baseline pass.
  2. For each layer: encode → zero cluster features → decode → modified MLP input.
  3. Patch ALL layers simultaneously in one hooked forward pass.
  4. Measure joint_logit_diff.

Key measured quantities per (cluster, prompt):
  - baseline_logit_diff    (from single unpatched pass)
  - joint_logit_diff       (from multi-layer simultaneous patch)
  - individual_sum         (sum of single-feature effects from contributions CSV)
  - joint_effect           = joint_logit_diff - baseline_logit_diff
  - interaction_term       = joint_effect - individual_sum
  - interaction_ratio      = joint_effect / individual_sum   (1 = additive, <1 = redundant, >1 = synergistic)
  - sign_flipped_joint     (did joint ablation flip the decision?)
  - predicted_sign_flip    (does individual_sum predict a flip?)

Usage (CSD3):
    python scripts/27_cluster_joint_ablation.py \\
        --behaviour physics_decay_type_probe \\
        --split train \\
        --clusters all          # or "0,6,10" for specific clusters
        --n_prompts 470         # all prompts; reduce for debugging
"""

import json, yaml, argparse, sys, logging, contextlib, time, gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers (copied from script 07, self-contained) ───────────────────────────

def compute_logit_diff(model: ModelWrapper, prompt: str,
                       correct_token: str, incorrect_token: str) -> float:
    """Correct - incorrect log-prob at final token."""
    device   = next(model.model.parameters()).device
    inputs   = model.tokenize([prompt])
    inputs   = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out  = model.model(**inputs, use_cache=False)
    logits   = out.logits[0, -1, :]
    lp       = torch.log_softmax(logits, dim=0)
    cid      = model.tokenizer.encode(correct_token,   add_special_tokens=False)[0]
    iid      = model.tokenizer.encode(incorrect_token, add_special_tokens=False)[0]
    return float(lp[cid] - lp[iid])


def get_mlp_input(model: ModelWrapper, inputs: dict,
                  layer_idx: int, token_pos: int = -1) -> torch.Tensor:
    """Extract post_attention_layernorm output at token_pos (matches script 04 point)."""
    try:
        blocks = model.model.model.layers
    except AttributeError:
        blocks = model.model.transformer.h
    hook_mod = blocks[layer_idx].post_attention_layernorm
    captured = {}

    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["x"] = t.detach()

    h = hook_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
    finally:
        h.remove()
    return captured["x"][:, token_pos, :]  # (1, hidden_dim)


@contextlib.contextmanager
def patch_mlp_layer(model_hf, layer_idx: int, token_pos: int,
                    new_mlp_input: torch.Tensor):
    """Context manager: inject modified MLP input at one layer (Qwen3/Llama style)."""
    try:
        block = model_hf.model.layers[layer_idx]
    except AttributeError:
        block = model_hf.transformer.h[layer_idx]
    hook_mod = block.post_attention_layernorm
    called   = {"n": 0}

    def hook(module, inp, out):
        called["n"] += 1
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        t[:, token_pos, :] = new_mlp_input.to(t.dtype).to(t.device)
        return (t,) + out[1:] if isinstance(out, tuple) else t

    h = hook_mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
    assert called["n"] > 0, f"Hook at layer {layer_idx} never fired!"


def run_joint_ablation(
    model: ModelWrapper,
    transcoder_set,
    prompt: str,
    cluster_by_layer: dict[int, list[int]],  # {layer_idx: [feat_idx, ...]}
    token_pos: int = -1,
) -> tuple[float, float]:
    """
    Returns (baseline_logit_diff, joint_logit_diff).
    Patches all layers simultaneously in a single forward pass.
    """
    device = next(model.model.parameters()).device
    inputs = model.tokenize([prompt])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── Baseline ──────────────────────────────────────────────────────────
    with torch.no_grad():
        base_out = model.model(**inputs, use_cache=False)
    # logit diff computed separately so we reuse the same inputs
    base_logits = base_out.logits[0, -1, :]

    # ── Extract MLP inputs at all relevant layers ─────────────────────────
    modified_per_layer: dict[int, torch.Tensor] = {}
    for layer_idx, feat_indices in cluster_by_layer.items():
        act = get_mlp_input(model, inputs, layer_idx, token_pos)  # (1, H)
        tc  = transcoder_set[layer_idx]
        with torch.no_grad():
            feats = tc.encode(act.to(tc.dtype))        # (1, d_tc)
            feats[:, feat_indices] = 0.0               # zero cluster features
            mod   = tc.decode(feats).to(act.dtype)     # (1, H)
        modified_per_layer[layer_idx] = mod.squeeze(0)  # (H,)

    # ── Joint patched forward pass ─────────────────────────────────────────
    with contextlib.ExitStack() as stack:
        for layer_idx, mod_input in modified_per_layer.items():
            stack.enter_context(
                patch_mlp_layer(model.model, layer_idx, token_pos, mod_input)
            )
        with torch.no_grad():
            joint_out = model.model(**inputs, use_cache=False)
    joint_logits = joint_out.logits[0, -1, :]

    return base_logits, joint_logits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour",    default="physics_decay_type_probe")
    parser.add_argument("--split",        default="train")
    parser.add_argument("--clusters",     default="all",
                        help="Comma-separated cluster IDs or 'all'")
    parser.add_argument("--n_prompts",    type=int, default=None,
                        help="Max prompts per cluster (None = all)")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--out_dir",      default=None)
    args = parser.parse_args()

    run_dir = ROOT / "data/results/cluster_joint_ablation"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) if args.out_dir else run_dir / f"joint_ablation_{args.behaviour}_{args.split}.csv"

    # ── Load cluster definitions ──────────────────────────────────────────
    import csv as csvlib
    clu_dir = ROOT / "data/results/clustering"
    with open(clu_dir / "cluster_labels.csv") as f:
        rows = list(csvlib.DictReader(f))
    coimp = {r["feature_id"]: int(r["coimp_louvain"]) for r in rows}
    clusters: dict[int, list[str]] = defaultdict(list)
    for fid, cid in coimp.items():
        clusters[cid].append(fid)

    # Filter clusters
    if args.clusters.lower() == "all":
        cluster_ids = sorted(clusters.keys())
    else:
        cluster_ids = [int(x.strip()) for x in args.clusters.split(",")]

    logger.info(f"Running joint ablation for clusters: {cluster_ids}")

    # Cluster features by layer: {cluster_id: {layer: [feat_idx, ...]}}
    def parse_layer_feat(fid: str):
        layer = int(fid.split("_")[0][1:])       # e.g. "L10_F128064" → 10
        feat  = int(fid.split("_F")[1])           # → 128064
        return layer, feat

    cluster_by_layer: dict[int, dict[int, list[int]]] = {}
    for cid in cluster_ids:
        by_layer: dict[int, list[int]] = defaultdict(list)
        for fid in clusters[cid]:
            l, f = parse_layer_feat(fid)
            by_layer[l].append(f)
        cluster_by_layer[cid] = dict(by_layer)
        logger.info(f"  C{cid} (n={len(clusters[cid])}): {dict(by_layer)}")

    # ── Load prompts ──────────────────────────────────────────────────────
    prompts_path = ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts_all = []
    with open(prompts_path) as f:
        for line in f:
            prompts_all.append(json.loads(line.strip()))

    if args.n_prompts:
        prompts_all = prompts_all[:args.n_prompts]
    logger.info(f"Loaded {len(prompts_all)} prompts from {prompts_path.name}")

    # ── Load individual ablation effects (for comparison) ─────────────────
    grouping_dir = ROOT / "data/results/grouping"
    contrib = pd.read_csv(grouping_dir / "feature_prompt_contributions.csv",
                          usecols=["prompt_idx","feature_id","effect_size","abs_effect_size",
                                   "baseline_logit_diff"])
    # Pre-compute individual_sum per (cluster_id, prompt_idx)
    contrib["cluster_id"] = contrib["feature_id"].map(coimp)
    indiv_sum = (
        contrib.dropna(subset=["cluster_id"])
        .groupby(["cluster_id","prompt_idx"])
        .agg(individual_sum=("effect_size","sum"),
             individual_abs_sum=("abs_effect_size","sum"),
             n_features=("feature_id","count"))
        .reset_index()
    )
    indiv_idx = {(int(r.cluster_id), int(r.prompt_idx)): r
                 for _, r in indiv_sum.iterrows()}

    # ── Load model and transcoders ────────────────────────────────────────
    logger.info("Loading model and transcoders…")
    tc_cfg_path = ROOT / "configs/transcoder_config.yaml"
    with open(tc_cfg_path) as f:
        tc_cfgd = yaml.safe_load(f)

    model_size   = tc_cfgd.get("model_size", "4b")
    model_name   = tc_cfgd["transcoders"][model_size]["model_name"]
    needed_layers = sorted(set(l for by_l in cluster_by_layer.values() for l in by_l))
    logger.info(f"Model: {model_name}  |  Transcoder layers needed: {needed_layers}")

    # Load model first (matches script 07 pattern)
    model = ModelWrapper(
        model_name=model_name,
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )
    try:
        device = next(model.model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model loaded on device: {device}")

    # Then load transcoders on the same device
    transcoder_set = load_transcoder_set(
        model_size=model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=needed_layers,
    )
    logger.info("Transcoders loaded.")

    # ── Run joint ablation ────────────────────────────────────────────────
    results = []
    t0 = time.time()

    # Build prompt_id → prompt_idx mapping from contributions CSV
    pid_to_idx = dict(zip(
        contrib["prompt_id"] if "prompt_id" in contrib.columns else pd.Series([], dtype=str),
        contrib["prompt_idx"]
    )) if "prompt_id" in contrib.columns else {}

    for p_i, p in enumerate(tqdm(prompts_all, desc="Prompts")):
        prompt_text   = p["prompt"]
        prompt_id     = p.get("prompt_id", str(p_i))
        # Use prompt_idx from contributions table; fall back to ordinal index
        prompt_idx    = int(pid_to_idx.get(prompt_id, p_i))
        # Answers already have leading space in JSONL (e.g. " alpha")
        correct_tok   = p.get("correct_answer",   " alpha")
        incorrect_tok = p.get("incorrect_answer",  " beta")

        # Validate tokens (skip multi-token answers)
        try:
            cid_tok = model.tokenizer.encode(correct_tok,   add_special_tokens=False)
            iid_tok = model.tokenizer.encode(incorrect_tok, add_special_tokens=False)
            assert len(cid_tok)==1 and len(iid_tok)==1
        except AssertionError:
            logger.warning(f"Skipping prompt {prompt_idx}: multi-token answer")
            continue

        correct_id   = cid_tok[0]
        incorrect_id = iid_tok[0]

        for cid in cluster_ids:
            by_layer = cluster_by_layer[cid]

            base_logits, joint_logits = run_joint_ablation(
                model, transcoder_set, prompt_text, by_layer, token_pos=-1
            )

            lp_base  = torch.log_softmax(base_logits.float(),  dim=0)
            lp_joint = torch.log_softmax(joint_logits.float(), dim=0)

            base_margin  = float(lp_base[correct_id]  - lp_base[incorrect_id])
            joint_margin = float(lp_joint[correct_id] - lp_joint[incorrect_id])
            joint_effect = joint_margin - base_margin  # = intervened - baseline

            # Individual sum from pre-computed table
            key = (cid, prompt_idx)
            indiv = indiv_idx.get(key)
            indiv_sum_val = float(indiv.individual_sum) if indiv is not None else float("nan")
            indiv_abs_val = float(indiv.individual_abs_sum) if indiv is not None else float("nan")

            # Interaction metrics
            eps = 1e-6
            if abs(indiv_sum_val) > eps:
                interaction_ratio = joint_effect / indiv_sum_val
            else:
                interaction_ratio = float("nan")
            interaction_term = joint_effect - indiv_sum_val

            base_sign  = 1 if base_margin > eps else -1
            joint_sign = 1 if joint_margin > eps else -1
            sign_flip  = base_sign != joint_sign

            pred_joint_margin = base_margin + indiv_sum_val
            pred_flip = (base_margin * pred_joint_margin) < 0

            results.append({
                "cluster_id":           cid,
                "prompt_idx":           prompt_idx,
                "n_cluster_features":   len(clusters[cid]),
                "baseline_logit_diff":  round(base_margin, 4),
                "joint_logit_diff":     round(joint_margin, 4),
                "joint_effect":         round(joint_effect, 4),
                "individual_sum":       round(indiv_sum_val, 4),
                "individual_abs_sum":   round(indiv_abs_val, 4),
                "interaction_term":     round(interaction_term, 4),
                "interaction_ratio":    round(interaction_ratio, 4) if not (interaction_ratio != interaction_ratio) else float("nan"),
                "sign_flipped_joint":   sign_flip,
                "predicted_sign_flip":  pred_flip,
            })

        # Progress log
        if (p_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (p_i + 1) / elapsed
            eta  = (len(prompts_all) - p_i - 1) / rate if rate > 0 else 0
            logger.info(f"  {p_i+1}/{len(prompts_all)} prompts | "
                        f"{rate:.2f} prompts/s | ETA {eta/60:.1f} min")

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    logger.info(f"Done. {len(df)} rows saved to {out_path} in {elapsed/60:.1f} min.")
    print(f"\n=== Quick summary ===")
    print(df.groupby("cluster_id").agg(
        n_prompts=("prompt_idx","count"),
        mean_joint_effect=("joint_effect","mean"),
        mean_indiv_sum=("individual_sum","mean"),
        mean_interaction_ratio=("interaction_ratio","mean"),
        sign_flip_rate=("sign_flipped_joint","mean"),
        pred_flip_rate=("predicted_sign_flip","mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
