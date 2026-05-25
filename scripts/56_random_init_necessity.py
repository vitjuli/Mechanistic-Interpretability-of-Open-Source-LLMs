"""
56_random_init_necessity.py — Random-init control for ablation-based necessity

Measures necessity(G) = Sign Flip Rate under joint zero-ablation of cluster G,
comparing trained-model activations vs random-init activations.

This provides the Δ_ctrl_nec value required for T1.2 (operational form):

    necessity_trained(G) ≥ τ_nec
    necessity_rand(G)    ≤ τ_nec − Δ_ctrl_nec        [random-init control]
    noop_sfr             ≤ τ_nec − Δ_recon_nec        [reconstruction noise floor]

Protocol for each cluster G and each prompt p:

  1. TRAINED necessity:
       x = MLP input from trained model at layer ℓ
       a = enc(x)
       a_abl = a with all G features zeroed
       x_hat = dec(a_abl)
       patch x_hat → trained model → Δ_abl
       sign_flip = int(sign(Δ_abl) ≠ sign(Δ_orig))

  2. RANDOM-INIT necessity:
       x_rand = MLP input from random-init model at layer ℓ
       a_rand = enc(x_rand)          [most entries = 0: JumpReLU kills OOD input]
       a_rand_abl = a_rand with G features zeroed  [≈ no change since already 0]
       x_hat_rand = dec(a_rand_abl)
       patch x_hat_rand → trained model → Δ_rand
       sign_flip_rand = int(sign(Δ_rand) ≠ sign(Δ_orig))

  3. NO-OP baseline (reconstruction noise floor):
       x_hat_noop = dec(enc(x))      [no feature modification]
       patch x_hat_noop → trained model → Δ_noop
       sign_flip_noop = int(sign(Δ_noop) ≠ sign(Δ_orig))

Key comparison:
  necessity_trained >> necessity_rand → ablation effect is model-specific
  necessity_trained ≈ necessity_rand  → effect is reconstruction noise artifact
  necessity_rand ≈ noop_sfr           → random-init control is behaving as expected
                                        (zeroing already-zero features changes nothing)

Usage:
    python scripts/56_random_init_necessity.py \\
        --behaviour physics_decay_type_probe_v2 \\
        --split train \\
        --cluster_id 10 \\
        --seed 42

    python scripts/56_random_init_necessity.py \\
        --behaviour physics_decay_type_probe_v2 \\
        --split train \\
        --all_clusters \\
        --seed 42

    sbatch jobs/run_random_init_necessity.sbatch
"""

import argparse
import json
import logging
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


# ── helpers (shared with scripts 51, 52, 55) ─────────────────────────────────

def get_mlp_input(hf_model, tokenizer, prompt: str,
                  layer_idx: int, device: str) -> torch.Tensor:
    """Capture post_attention_layernorm output at last token — (1, d)."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}

    block = hf_model.model.layers[layer_idx]

    def _hook(module, inp, out):
        captured["x"] = out[:, -1:, :].detach()

    h = block.post_attention_layernorm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            hf_model(**inputs, use_cache=False)
    finally:
        h.remove()

    return captured["x"][0]   # (1, d)


def compute_logit_diff(hf_model, tokenizer, prompt: str,
                       correct: str, incorrect: str, device: str) -> float:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = hf_model(**inputs, use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct,   add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


def compute_logit_diff_with_patch(hf_model, tokenizer, prompt: str,
                                  correct: str, incorrect: str,
                                  layer_idx: int, x_patch: torch.Tensor,
                                  device: str) -> float:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    block = hf_model.model.layers[layer_idx]
    _p = x_patch.to(device)

    def _hook(module, inp, out):
        mod = out.clone()
        mod[0, -1, :] = _p[0, :]
        return mod

    h = block.post_attention_layernorm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = hf_model(**inputs, use_cache=False)
    finally:
        h.remove()

    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct,   add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


def build_random_model(model_name: str, seed: int, device: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    logger.info(f"Building random-init model: {model_name} (seed={seed})")
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    rand_model = AutoModelForCausalLM.from_config(config)
    return rand_model.to(device).eval()


# ── core: necessity for one cluster ──────────────────────────────────────────

def run_necessity_for_cluster(
    cluster_id: int,
    feature_ids: List[str],
    trained_hf_model,
    rand_hf_model,
    tokenizer,
    transcoder_set,
    prompts: List[Dict],
    device: str,
    max_prompts: Optional[int] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    For each prompt, run three ablation variants and record SFR:
      - trained_abl:  zero G features in trained activations
      - rand_abl:     zero G features in random-init activations
      - noop:         enc→dec no modification (reconstruction noise floor)

    Returns (summary_dict, raw_df).
    """
    # Parse feature indices per layer
    layer_feats: Dict[int, List[int]] = defaultdict(list)
    for fid in feature_ids:
        parts = fid.lstrip("L").split("_F")
        layer, fidx = int(parts[0]), int(parts[1])
        layer_feats[layer].append(fidx)

    layers = sorted(layer_feats.keys())

    if max_prompts:
        prompts = prompts[:max_prompts]

    rows = []

    for pi, p in enumerate(prompts):
        if pi % 50 == 0:
            logger.info(f"  prompt {pi+1}/{len(prompts)}")

        prompt_text = p["prompt"]
        correct     = p["correct_answer"]
        incorrect   = p["incorrect_answer"]
        correct_ans = correct.strip()

        try:
            delta_orig = compute_logit_diff(
                trained_hf_model, tokenizer, prompt_text,
                correct, incorrect, device)
        except Exception as e:
            logger.debug(f"prompt {pi}: baseline failed — {e}")
            continue

        # For multi-layer clusters: run each layer independently
        # (cluster may span multiple layers — average across layers)
        layer_results = {}

        for layer in layers:
            if layer not in transcoder_set._transcoders:
                continue
            tc = transcoder_set[layer]
            feat_indices = layer_feats[layer]

            try:
                x_tr   = get_mlp_input(trained_hf_model, tokenizer,
                                       prompt_text, layer, device).to(tc.dtype)
                x_rand = get_mlp_input(rand_hf_model, tokenizer,
                                       prompt_text, layer, device).to(tc.dtype)

                with torch.no_grad():
                    a_tr   = tc.encode(x_tr)     # (1, d_tc)
                    a_rand = tc.encode(x_rand)

                    # ── 1. trained ablation ─────────────────────────────────
                    a_tr_abl = a_tr.clone()
                    a_tr_abl[:, feat_indices] = 0.0
                    x_hat_tr = tc.decode(a_tr_abl).to(x_tr.dtype)

                    # ── 2. random-init ablation ─────────────────────────────
                    a_rand_abl = a_rand.clone()
                    a_rand_abl[:, feat_indices] = 0.0
                    x_hat_rand = tc.decode(a_rand_abl).to(x_tr.dtype)

                    # ── 3. no-op (reconstruction noise floor) ──────────────
                    x_hat_noop = tc.decode(a_tr).to(x_tr.dtype)

                    # ── diagnostics ─────────────────────────────────────────
                    recon_err_tr   = (x_hat_noop - x_tr).norm().item()
                    recon_err_rand = (tc.decode(a_rand).to(x_tr.dtype)
                                     - x_rand).norm().item()
                    active_tr_frac   = (a_tr[0, feat_indices]   > 0).float().mean().item()
                    active_rand_frac = (a_rand[0, feat_indices] > 0).float().mean().item()
                    # How much do G features actually carry?
                    g_activation_tr   = a_tr[0, feat_indices].abs().mean().item()
                    g_activation_rand = a_rand[0, feat_indices].abs().mean().item()

                delta_tr_abl = compute_logit_diff_with_patch(
                    trained_hf_model, tokenizer, prompt_text,
                    correct, incorrect, layer, x_hat_tr, device)

                delta_rand_abl = compute_logit_diff_with_patch(
                    trained_hf_model, tokenizer, prompt_text,
                    correct, incorrect, layer, x_hat_rand, device)

                delta_noop = compute_logit_diff_with_patch(
                    trained_hf_model, tokenizer, prompt_text,
                    correct, incorrect, layer, x_hat_noop, device)

            except Exception as e:
                logger.debug(f"prompt {pi} layer {layer}: {e}")
                continue

            sf_tr   = int(np.sign(delta_tr_abl)   != np.sign(delta_orig))
            sf_rand = int(np.sign(delta_rand_abl) != np.sign(delta_orig))
            sf_noop = int(np.sign(delta_noop)     != np.sign(delta_orig))

            rows.append({
                "cluster_id":       cluster_id,
                "layer":            layer,
                "prompt_idx":       pi,
                "prompt_id":        p.get("prompt_id", str(pi)),
                "correct_answer":   correct_ans,
                "delta_orig":       delta_orig,
                "delta_tr_abl":     delta_tr_abl,
                "delta_rand_abl":   delta_rand_abl,
                "delta_noop":       delta_noop,
                "sf_trained":       sf_tr,
                "sf_rand":          sf_rand,
                "sf_noop":          sf_noop,
                "effect_tr":        delta_tr_abl   - delta_orig,
                "effect_rand":      delta_rand_abl - delta_orig,
                "effect_noop":      delta_noop     - delta_orig,
                "active_G_tr":      active_tr_frac,
                "active_G_rand":    active_rand_frac,
                "g_act_mean_tr":    g_activation_tr,
                "g_act_mean_rand":  g_activation_rand,
                "recon_err_tr":     recon_err_tr,
                "recon_err_rand":   recon_err_rand,
            })

    if not rows:
        return {}, pd.DataFrame()

    df = pd.DataFrame(rows)

    # Average across layers if multi-layer cluster (per-prompt mean)
    per_prompt = df.groupby("prompt_idx").agg(
        sf_trained  = ("sf_trained",  "max"),   # any layer flips = flip
        sf_rand     = ("sf_rand",     "max"),
        sf_noop     = ("sf_noop",     "max"),
        effect_tr   = ("effect_tr",   "mean"),
        effect_rand = ("effect_rand", "mean"),
        correct_answer = ("correct_answer", "first"),
    ).reset_index()

    necessity_trained = float(per_prompt["sf_trained"].mean())
    necessity_rand    = float(per_prompt["sf_rand"].mean())
    noop_sfr          = float(per_prompt["sf_noop"].mean())

    summary = {
        "cluster_id":         cluster_id,
        "n_prompts":          len(per_prompt),
        "n_rows":             len(df),
        "layers":             layers,
        "n_features":         len(feature_ids),
        # ── main metrics ───────────────────────────────────────────────────
        "necessity_trained":  necessity_trained,
        "necessity_rand":     necessity_rand,
        "noop_sfr":           noop_sfr,
        "delta_ctrl_nec":     necessity_trained - necessity_rand,
        "delta_recon_nec":    necessity_trained - noop_sfr,
        # ── by answer class ────────────────────────────────────────────────
        "necessity_alpha": float(
            per_prompt[per_prompt["correct_answer"] == "alpha"]["sf_trained"].mean()),
        "necessity_beta": float(
            per_prompt[per_prompt["correct_answer"] == "beta"]["sf_trained"].mean()),
        "necessity_rand_alpha": float(
            per_prompt[per_prompt["correct_answer"] == "alpha"]["sf_rand"].mean()),
        "necessity_rand_beta": float(
            per_prompt[per_prompt["correct_answer"] == "beta"]["sf_rand"].mean()),
        # ── diagnostics ────────────────────────────────────────────────────
        "mean_active_G_tr":   float(df["active_G_tr"].mean()),
        "mean_active_G_rand": float(df["active_G_rand"].mean()),
        "mean_g_act_tr":      float(df["g_act_mean_tr"].mean()),
        "mean_g_act_rand":    float(df["g_act_mean_rand"].mean()),
        "mean_recon_err_tr":  float(df["recon_err_tr"].mean()),
        "mean_recon_err_rand":float(df["recon_err_rand"].mean()),
        "recon_err_ratio":    float(
            df["recon_err_rand"].mean() / max(df["recon_err_tr"].mean(), 1e-8)),
    }

    return summary, df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour",    default="physics_decay_type_probe_v2")
    p.add_argument("--split",        default="train")
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--cluster_id",   type=int, default=None)
    p.add_argument("--all_clusters", action="store_true")
    p.add_argument("--cluster_json", default=None,
                   help="Path to cluster_semantics.json. Defaults to dashboard_probe.")
    p.add_argument("--max_prompts",  type=int, default=None)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   default="data/analysis/random_init_necessity")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_name",   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

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

    prompts = [json.loads(l) for l in open(prompt_path)]
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")

    with open(cluster_json) as f:
        csem = json.load(f)
    clusters = {c["id"]: c for c in csem["clusters"]}

    if args.cluster_id is not None:
        target_clusters = [args.cluster_id]
    elif args.all_clusters:
        target_clusters = sorted(clusters.keys())
    else:
        raise ValueError("Specify --cluster_id N or --all_clusters")

    import yaml
    tc_cfg     = yaml.safe_load(open(root / "configs" / "transcoder_config.yaml"))
    tc_info    = tc_cfg["transcoders"][tc_cfg["model_size"]]
    model_name = args.model_name or tc_info["model_name"]

    logger.info("Loading TRAINED model…")
    trained_wrapper = ModelWrapper(model_name, device=args.device)
    trained_hf  = trained_wrapper.model
    tokenizer   = trained_wrapper.tokenizer

    rand_hf = build_random_model(model_name, seed=args.seed, device=args.device)

    all_layers = sorted({
        int(fid.split("_F")[0].lstrip("L"))
        for cid in target_clusters
        for fid in clusters[cid]["feature_ids"]
    })
    logger.info(f"Loading transcoders for layers: {all_layers}")
    tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                                 layers=all_layers, device=args.device)

    all_summaries, all_dfs = [], []

    for cid in target_clusters:
        c = clusters[cid]
        logger.info(f"\n=== Cluster {cid}: {c['name']} "
                    f"({len(c['feature_ids'])} features) ===")

        summary, df = run_necessity_for_cluster(
            cluster_id       = cid,
            feature_ids      = c["feature_ids"],
            trained_hf_model = trained_hf,
            rand_hf_model    = rand_hf,
            tokenizer        = tokenizer,
            transcoder_set   = tc_set,
            prompts          = prompts,
            device           = args.device,
            max_prompts      = args.max_prompts,
        )

        if not summary:
            logger.warning(f"  No results for cluster {cid}")
            continue

        all_summaries.append(summary)
        all_dfs.append(df)

        logger.info(
            f"  necessity_trained = {summary['necessity_trained']:.3f}  "
            f"necessity_rand = {summary['necessity_rand']:.3f}  "
            f"noop_sfr = {summary['noop_sfr']:.3f}")
        logger.info(
            f"  Δ_ctrl_nec = {summary['delta_ctrl_nec']:+.3f}  "
            f"Δ_recon_nec = {summary['delta_recon_nec']:+.3f}")
        logger.info(
            f"  Active G (trained): {summary['mean_active_G_tr']:.1%}  "
            f"Active G (rand): {summary['mean_active_G_rand']:.1%}  "
            f"ReconRatio: {summary['recon_err_ratio']:.1f}×")

    if not all_summaries:
        logger.error("No results produced.")
        sys.exit(1)

    summary_path = (out_dir /
        f"random_init_necessity_{args.behaviour}_{args.split}_summary.json")
    csv_path = (out_dir /
        f"random_init_necessity_{args.behaviour}_{args.split}.csv")

    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    pd.concat(all_dfs, ignore_index=True).to_csv(csv_path, index=False)

    logger.info(f"\nSaved summary → {summary_path}")
    logger.info(f"Saved raw CSV → {csv_path}")

    # ── print comparison table ────────────────────────────────────────────────
    print("\n=== RANDOM-INIT NECESSITY CONTROL ===")
    print(f"{'C':>3}  {'nec_tr':>8}  {'nec_rand':>9}  {'noop_sfr':>9}  "
          f"{'Δ_ctrl':>8}  {'Δ_recon':>8}  {'ActG_tr':>8}  "
          f"{'ActG_rnd':>9}  {'RcnRatio':>9}  Name")
    print("-" * 110)
    for s in all_summaries:
        name = clusters[s["cluster_id"]]["name"][:28]
        print(f"  {s['cluster_id']:>2}  "
              f"{s['necessity_trained']:>8.3f}  "
              f"{s['necessity_rand']:>9.3f}  "
              f"{s['noop_sfr']:>9.3f}  "
              f"{s['delta_ctrl_nec']:>+8.3f}  "
              f"{s['delta_recon_nec']:>+8.3f}  "
              f"{s['mean_active_G_tr']:>7.1%}  "
              f"{s['mean_active_G_rand']:>9.1%}  "
              f"{s['recon_err_ratio']:>8.1f}×  "
              f"{name}")

    print()
    print("Interpretation:")
    print("  necessity_trained >> noop_sfr → cluster G is causally necessary beyond recon noise")
    print("  necessity_rand ≈ noop_sfr     → random-init ablation = pure recon noise (expected)")
    print("  Δ_ctrl_nec > 0.1              → necessity is model-specific, not transcoder capacity")
    print()
    print("T1.2 operational form: use necessity_trained as τ_nec,")
    print("  and necessity_rand as the upper bound for necessity_rand(G) ≤ τ_nec − Δ_ctrl_nec.")


if __name__ == "__main__":
    main()
