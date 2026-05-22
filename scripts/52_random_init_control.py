"""
52_random_init_control.py — Random-init control for transcoder IIA

Addresses the non-linear representation dilemma (Wu et al. 2024):
  High IIA might reflect transcoder capacity, not meaningful model computation.

Protocol:
  1. Load trained transcoder (same as main experiment).
  2. Instantiate Qwen3-4B with RANDOM weights (from_config, not from_pretrained).
  3. For each prompt pair (p_alpha, p_beta), collect MLP-input activations
     from the RANDOM-INIT model at the cluster's layer.
  4. Encode those random-init activations with the TRAINED transcoder.
  5. Do the same feature-level interchange on cluster G.
  6. Decode and patch into the TRAINED model, measure IIA_rand.

Also reports:
  - Fraction of features in G that are active on random-init activations
    (sanity: JumpReLU threshold should kill most, since random-init is OOD)
  - Reconstruction error ||x_rand − dec(enc(x_rand))|| vs
    ||x_trained − dec(enc(x_trained))|| (OOD gap)
  - IIA_trained  (main experiment number, for direct comparison)
  - IIA_rand     (this control)

Usage:
    python scripts/52_random_init_control.py \\
        --behaviour physics_decay_type_probe \\
        --cluster_id 7 \\
        --layer 24 \\
        --max_pairs 200 \\
        --seed 42

    # all clusters, multiple layers:
    python scripts/52_random_init_control.py \\
        --behaviour physics_decay_type_probe \\
        --all_clusters --seed 42

On CSD3:
    sbatch jobs/run_random_init_control.sbatch
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def ensure_single_token(model, tok: str) -> int:
    ids = model.tokenizer.encode(tok, add_special_tokens=False)
    return ids[0]


def get_mlp_input(hf_model, tokenizer, prompt: str, layer_idx: int,
                  device: str) -> torch.Tensor:
    """Capture post_attention_layernorm output at last token — (1, d)."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}

    block = hf_model.model.layers[layer_idx]
    hook_mod = block.post_attention_layernorm

    def _hook(module, inp, out):
        captured["x"] = out[:, -1:, :].detach()   # (1, 1, d)

    handle = hook_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            hf_model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert "x" in captured
    return captured["x"][0]    # (1, d)


def compute_logit_diff(hf_model, tokenizer, prompt: str,
                       correct: str, incorrect: str,
                       device: str) -> float:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = hf_model(**inputs, use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct, add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


def compute_logit_diff_with_patch(hf_model, tokenizer, prompt: str,
                                  correct: str, incorrect: str,
                                  layer_idx: int, x_patch: torch.Tensor,
                                  device: str) -> float:
    """Forward pass on hf_model with x_patch at MLP input, layer_idx, last token."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    block = hf_model.model.layers[layer_idx]
    called = {"n": 0}
    _p = x_patch.to(device)   # (1, d)

    def _hook(module, inp, out):
        called["n"] += 1
        mod = out.clone()
        mod[0, -1, :] = _p[0, :]
        return mod

    handle = block.post_attention_layernorm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = hf_model(**inputs, use_cache=False)
    finally:
        handle.remove()

    assert called["n"] > 0
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct, add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


# ── random-init model factory ─────────────────────────────────────────────────

def build_random_model(model_name: str, seed: int, device: str):
    """Instantiate model architecture with random weights (no pretrained)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    logger.info(f"Building random-init model: {model_name} (seed={seed})")
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = AutoConfig.from_pretrained(model_name)
    rng_state = torch.get_rng_state()

    # from_config gives random HF-default init
    rand_model = AutoModelForCausalLM.from_config(config)
    rand_model = rand_model.to(device).eval()

    logger.info(f"  Parameters: {sum(p.numel() for p in rand_model.parameters()):,}")
    return rand_model


# ── core experiment ───────────────────────────────────────────────────────────

def run_control_for_cluster(
    cluster_id: int,
    feature_ids: List[str],          # e.g. ["L24_F52031", "L24_F18943", ...]
    trained_hf_model,                # trained model (for logit-diff eval)
    rand_hf_model,                   # random-init model (for activations)
    tokenizer,
    transcoder_set,
    prompts: List[Dict],
    device: str,
    max_pairs: Optional[int],
    batch_log: int = 10,
) -> Dict:
    """
    Run trained-IIA and rand-IIA for a single cluster.

    The cluster may span multiple layers. For multi-layer clusters we run
    the interchange independently at each layer and average.
    """
    # Parse feature indices per layer
    layer_feats: Dict[int, List[int]] = defaultdict(list)
    for fid in feature_ids:
        # format: "L{layer}_F{index}"
        parts = fid.lstrip("L").split("_F")
        layer, fidx = int(parts[0]), int(parts[1])
        layer_feats[layer].append(fidx)

    layers = list(layer_feats.keys())

    # Build alpha/beta pairs
    alpha_prompts = [p for p in prompts if p["correct_answer"].strip() == "alpha"]
    beta_prompts  = [p for p in prompts if p["correct_answer"].strip() == "beta"]

    rng = np.random.default_rng(seed=0)
    n_pairs = min(len(alpha_prompts), len(beta_prompts))
    if max_pairs:
        n_pairs = min(n_pairs, max_pairs)

    alpha_idx = rng.choice(len(alpha_prompts), size=n_pairs, replace=False)
    beta_idx  = rng.choice(len(beta_prompts),  size=n_pairs, replace=False)
    pairs = [(alpha_prompts[i], beta_prompts[j])
             for i, j in zip(alpha_idx, beta_idx)]

    # Pre-collect activations (trained + random) for all prompts × all layers
    logger.info(f"  Collecting activations for {len(pairs)*2} unique prompts "
                f"across {len(layers)} layers…")

    all_prompts_needed = list({
        (p["prompt_id"], p["prompt"], p["correct_answer"], p["incorrect_answer"])
        for pa, pb in pairs for p in (pa, pb)
    })

    # key: (prompt_id, layer) → {"trained": tensor(1,d), "rand": tensor(1,d)}
    act_cache: Dict[Tuple, Dict] = {}

    for pi, (pid, prompt_text, correct, incorrect) in enumerate(all_prompts_needed):
        if pi % batch_log == 0:
            logger.info(f"    activation {pi+1}/{len(all_prompts_needed)}")
        for layer in layers:
            if layer not in transcoder_set._transcoders:
                continue
            key = (pid, layer)
            x_tr   = get_mlp_input(trained_hf_model, tokenizer, prompt_text, layer, device)
            x_rand = get_mlp_input(rand_hf_model,    tokenizer, prompt_text, layer, device)
            act_cache[key] = {"trained": x_tr, "rand": x_rand,
                              "correct": correct, "incorrect": incorrect,
                              "prompt": prompt_text}

    # Compute per-pair metrics
    rows = []
    recon_errs_trained, recon_errs_rand = [], []
    active_frac_trained_list, active_frac_rand_list = [], []

    for pair_idx, (p_alpha, p_beta) in enumerate(pairs):
        # For each pair: target=alpha, source=beta
        # Interchange: put beta's G-features into alpha's context
        for layer, feat_indices in layer_feats.items():
            if layer not in transcoder_set._transcoders:
                continue
            tc = transcoder_set[layer]

            key_alpha = (p_alpha["prompt_id"], layer)
            key_beta  = (p_beta["prompt_id"],  layer)
            if key_alpha not in act_cache or key_beta not in act_cache:
                continue

            x_tr_alpha  = act_cache[key_alpha]["trained"].to(tc.dtype)
            x_tr_beta   = act_cache[key_beta]["trained"].to(tc.dtype)
            x_rand_alpha = act_cache[key_alpha]["rand"].to(tc.dtype)
            x_rand_beta  = act_cache[key_beta]["rand"].to(tc.dtype)

            with torch.no_grad():
                # ── trained interchange ────────────────────────────────────
                a_tr_alpha = tc.encode(x_tr_alpha)     # (1, d_tc)
                a_tr_beta  = tc.encode(x_tr_beta)
                a_tr_patched = a_tr_alpha.clone()
                a_tr_patched[:, feat_indices] = a_tr_beta[:, feat_indices]
                x_tr_patched = tc.decode(a_tr_patched).to(x_tr_alpha.dtype)

                # ── random-init interchange ────────────────────────────────
                a_rand_alpha = tc.encode(x_rand_alpha)
                a_rand_beta  = tc.encode(x_rand_beta)
                a_rand_patched = a_rand_alpha.clone()
                a_rand_patched[:, feat_indices] = a_rand_beta[:, feat_indices]
                x_rand_patched = tc.decode(a_rand_patched).to(x_rand_alpha.dtype)

                # ── reconstruction errors ──────────────────────────────────
                x_tr_recon   = tc.decode(a_tr_alpha).to(x_tr_alpha.dtype)
                x_rand_recon = tc.decode(a_rand_alpha).to(x_rand_alpha.dtype)
                recon_err_tr   = (x_tr_recon   - x_tr_alpha).norm().item()
                recon_err_rand = (x_rand_recon - x_rand_alpha).norm().item()
                recon_errs_trained.append(recon_err_tr)
                recon_errs_rand.append(recon_err_rand)

                # ── active feature fraction in G ───────────────────────────
                active_tr   = (a_tr_alpha[:,   feat_indices] > 0).float().mean().item()
                active_rand = (a_rand_alpha[:, feat_indices] > 0).float().mean().item()
                active_frac_trained_list.append(active_tr)
                active_frac_rand_list.append(active_rand)

            # ── logit diffs ────────────────────────────────────────────────
            try:
                delta_orig = compute_logit_diff(
                    trained_hf_model, tokenizer,
                    p_alpha["prompt"], p_alpha["correct_answer"], p_alpha["incorrect_answer"],
                    device)
                # Trained: patch trained-derived features into trained model
                delta_tr_patched = compute_logit_diff_with_patch(
                    trained_hf_model, tokenizer,
                    p_alpha["prompt"], p_alpha["correct_answer"], p_alpha["incorrect_answer"],
                    layer, x_tr_patched, device)
                # Rand:    patch rand-derived features into trained model
                delta_rand_patched = compute_logit_diff_with_patch(
                    trained_hf_model, tokenizer,
                    p_alpha["prompt"], p_alpha["correct_answer"], p_alpha["incorrect_answer"],
                    layer, x_rand_patched, device)
            except Exception as e:
                logger.debug(f"pair {pair_idx} layer {layer}: {e}")
                continue

            # IIA convention: did the answer flip toward beta (source)?
            # We expect alpha→ answer changes toward beta after patching beta's features
            iia_tr   = int(np.sign(delta_tr_patched)   != np.sign(delta_orig))
            iia_rand = int(np.sign(delta_rand_patched) != np.sign(delta_orig))

            rows.append({
                "cluster_id":      cluster_id,
                "layer":           layer,
                "pair_idx":        pair_idx,
                "prompt_alpha":    p_alpha["prompt_id"],
                "prompt_beta":     p_beta["prompt_id"],
                "delta_orig":      delta_orig,
                "delta_tr_patched":    delta_tr_patched,
                "delta_rand_patched":  delta_rand_patched,
                "iia_trained":     iia_tr,
                "iia_rand":        iia_rand,
                "active_frac_G_trained": active_tr,
                "active_frac_G_rand":    active_rand,
                "recon_err_trained":     recon_err_tr,
                "recon_err_rand":        recon_err_rand,
            })

        if pair_idx % 20 == 0:
            logger.info(f"  pair {pair_idx+1}/{len(pairs)}")

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    summary = {
        "cluster_id":      cluster_id,
        "n_pairs":         len(pairs),
        "n_rows":          len(df),
        "layers":          layers,
        # Main metric
        "iia_trained":     float(df["iia_trained"].mean()),
        "iia_rand":        float(df["iia_rand"].mean()),
        "delta_ctrl":      float(df["iia_trained"].mean() - df["iia_rand"].mean()),
        # Active features
        "mean_active_frac_G_trained": float(df["active_frac_G_trained"].mean()),
        "mean_active_frac_G_rand":    float(df["active_frac_G_rand"].mean()),
        # Reconstruction error ratio
        "mean_recon_err_trained": float(df["recon_err_trained"].mean()),
        "mean_recon_err_rand":    float(df["recon_err_rand"].mean()),
        "recon_err_ratio":        float(
            df["recon_err_rand"].mean() / max(df["recon_err_trained"].mean(), 1e-8)),
    }
    return summary, df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour",    default="physics_decay_type_probe")
    p.add_argument("--split",        default="train")
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--cluster_id",   type=int, default=None,
                   help="Single cluster to run. Omit to run --all_clusters.")
    p.add_argument("--all_clusters", action="store_true")
    p.add_argument("--cluster_json", default=None,
                   help="Path to cluster_semantics.json. Defaults to dashboard.")
    p.add_argument("--max_pairs",    type=int, default=None)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   default="data/analysis/random_init_control")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_name",   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    # ── resolve paths ────────────────────────────────────────────────────────
    out_dir = root / args.output_dir / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = (Path(args.prompts_file) if args.prompts_file
                   else root / "data" / "prompts" /
                        f"{args.behaviour}_{args.split}.jsonl")
    assert prompt_path.exists(), f"Prompts not found: {prompt_path}"

    cluster_json = (Path(args.cluster_json) if args.cluster_json
                    else root / "dashboard_probe" / "public" / "data" /
                         "cluster_semantics.json")
    assert cluster_json.exists(), f"Cluster JSON not found: {cluster_json}"

    # ── load prompts ─────────────────────────────────────────────────────────
    prompts = [json.loads(l) for l in open(prompt_path)]
    logger.info(f"Loaded {len(prompts)} prompts")

    # ── load cluster metadata ─────────────────────────────────────────────────
    with open(cluster_json) as f:
        csem = json.load(f)
    clusters = {c["id"]: c for c in csem["clusters"]}

    if args.cluster_id is not None:
        target_clusters = [args.cluster_id]
    elif args.all_clusters:
        target_clusters = sorted(clusters.keys())
    else:
        raise ValueError("Specify --cluster_id N or --all_clusters")

    # ── load model config ─────────────────────────────────────────────────────
    import yaml
    tc_cfg = yaml.safe_load(open(root / "configs" / "transcoder_config.yaml"))
    model_size = tc_cfg["model_size"]
    tc_info    = tc_cfg["transcoders"][model_size]
    model_name = args.model_name or tc_info["model_name"]

    # ── build models ─────────────────────────────────────────────────────────
    logger.info("Loading TRAINED model…")
    trained_wrapper = ModelWrapper(model_name, device=args.device)
    trained_hf  = trained_wrapper.model
    tokenizer   = trained_wrapper.tokenizer

    rand_hf = build_random_model(model_name, seed=args.seed, device=args.device)

    # ── load transcoders (all layers needed across clusters) ─────────────────
    all_layers = sorted({
        int(fid.split("_F")[0].lstrip("L"))
        for cid in target_clusters
        for fid in clusters[cid]["feature_ids"]
    })
    logger.info(f"Loading transcoders for layers: {all_layers}")
    tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                                 layers=all_layers, device=args.device)

    # ── run ──────────────────────────────────────────────────────────────────
    all_summaries = []
    all_dfs       = []

    for cid in target_clusters:
        c = clusters[cid]
        logger.info(f"\n=== Cluster {cid}: {c['name']} "
                    f"({len(c['feature_ids'])} features) ===")

        result = run_control_for_cluster(
            cluster_id   = cid,
            feature_ids  = c["feature_ids"],
            trained_hf_model = trained_hf,
            rand_hf_model    = rand_hf,
            tokenizer        = tokenizer,
            transcoder_set   = tc_set,
            prompts          = prompts,
            device           = args.device,
            max_pairs        = args.max_pairs,
        )

        if not result:
            logger.warning(f"  No results for cluster {cid}")
            continue

        summary, df = result
        all_summaries.append(summary)
        all_dfs.append(df)

        logger.info(f"  IIA_trained = {summary['iia_trained']:.3f}  "
                    f"IIA_rand = {summary['iia_rand']:.3f}  "
                    f"Δ_ctrl = {summary['delta_ctrl']:+.3f}")
        logger.info(f"  Active G (trained): {summary['mean_active_frac_G_trained']:.2%}  "
                    f"Active G (rand): {summary['mean_active_frac_G_rand']:.2%}")
        logger.info(f"  Recon err ratio (rand/trained): {summary['recon_err_ratio']:.1f}×")

    if not all_summaries:
        logger.error("No results produced.")
        sys.exit(1)

    # ── save ─────────────────────────────────────────────────────────────────
    summary_path = out_dir / f"random_init_control_{args.behaviour}_{args.split}_summary.json"
    csv_path     = out_dir / f"random_init_control_{args.behaviour}_{args.split}.csv"

    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(csv_path, index=False)

    logger.info(f"\nSaved summary → {summary_path}")
    logger.info(f"Saved raw CSV → {csv_path}")

    # ── print table ───────────────────────────────────────────────────────────
    print("\n=== RANDOM-INIT CONTROL RESULTS ===")
    print(f"{'C':>3}  {'IIA_tr':>8}  {'IIA_rand':>9}  {'Δ_ctrl':>8}  "
          f"{'ActG_tr':>8}  {'ActG_rand':>10}  {'ReconRatio':>11}  Name")
    print("-" * 85)
    for s in all_summaries:
        name = clusters[s["cluster_id"]]["name"][:28]
        print(f"  {s['cluster_id']:>2}  "
              f"{s['iia_trained']:>8.3f}  "
              f"{s['iia_rand']:>9.3f}  "
              f"{s['delta_ctrl']:>+8.3f}  "
              f"{s['mean_active_frac_G_trained']:>7.1%}  "
              f"{s['mean_active_frac_G_rand']:>10.1%}  "
              f"{s['recon_err_ratio']:>10.1f}×  "
              f"{name}")
    print()
    print("Δ_ctrl > 0.1 → IIA signal is model-specific, not transcoder artifact.")
    print("ActG_rand ≈ 0 → random-init activations are OOD for trained transcoder.")
    print("ReconRatio >> 1 → reconstruction error confirms OOD (expected).")


if __name__ == "__main__":
    main()
