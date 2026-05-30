"""
57_graded_patching.py — Graded patching experiment.

Tests whether carrier sub-structure is CONTINUOUS or DISCRETE:
  For each fraction α ∈ {0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0}:
    Sample N_subsets random subsets of size round(α × |pool|) from feature pool
    For each subset: run IIA (interchange patching) and record SFR.

  Hypotheses:
    - Population coding: IIA(α) grows LINEARLY with α (each feature contributes
      equally to state carrying)
    - Redundant / distributed coding: IIA(α) SATURATES at small α (few features
      carry most of the signal)
    - Threshold coding: IIA(α) jumps at some critical α*

Predicted patterns for L24 vs L18 (based on previous findings):
  - L24 (β-carrier with high Jaccard overlap 0.90-0.99 between sub-detectors):
    expected SATURATION at α ≈ 0.1 (2/20 features = C16 pair already gives 94%
    of full IIA)
  - L18 (monolithic α-carrier, population-coded by ICC diagnostic):
    expected LINEAR or near-linear growth
  - Random control (matched n features from non-target layer):
    expected near-zero IIA at all α

Inputs:
  data/analysis/runD_v2/clustering_full/cluster_labels.csv
    (column `agglo_coimp_subgroup_k30` for sub-cluster mapping)
  data/prompts/{behaviour}_{split}.jsonl

Outputs:
  {out_dir}/graded_patching_raw.csv
    Per (pool, alpha, subset_seed, pair, direction) flip flag.
  {out_dir}/graded_patching_summary.csv
    Per (pool, alpha) mean/std IIA across subsets.
  {out_dir}/graded_patching_curve.png
    Visualisation of IIA(α) curves.

Usage:
  python3 scripts/57_graded_patching.py \\
    --target_layers L24 L18 random_l25 \\
    --alphas 0.05 0.10 0.15 0.25 0.50 0.75 1.0 \\
    --n_subsets 5 \\
    --max_pairs 200 \\
    --out_dir data/analysis/runD_v2/graded_patching
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def parse_feature_id(fid: str) -> Tuple[int, int]:
    layer, idx = fid.lstrip("L").split("_F")
    return int(layer), int(idx)


# ─── IIA primitives (shared with script 37b) ─────────────────────────────────

def get_features_and_margin(
    hf_model, tokenizer, prompt: str, correct: str, incorrect: str,
    transcoder_set, layers: List[int], device: str,
) -> Tuple[float, Dict[int, torch.Tensor]]:
    """Get baseline margin and feature activations at specified layers."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False)["input_ids"]
    if not correct_ids or not incorrect_ids:
        return 0.0, {}
    correct_id = correct_ids[0]
    incorrect_id = incorrect_ids[0]

    captured = {}
    handles = []
    for layer in layers:
        block = hf_model.model.layers[layer]
        def _make(layer=layer):
            def _hook(module, inp, out):
                captured[layer] = out[:, -1:, :].detach()
            return _hook
        handles.append(block.post_attention_layernorm.register_forward_hook(_make()))

    try:
        with torch.no_grad():
            outputs = hf_model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]
            margin = float((logits[correct_id] - logits[incorrect_id]).item())
    finally:
        for h in handles:
            h.remove()

    feats = {}
    for layer in layers:
        if layer in captured:
            x_l = captured[layer][0]
            tc = transcoder_set[layer]
            with torch.no_grad():
                a_l = tc.encode(x_l.to(tc.dtype))
            feats[layer] = a_l.float()
    return margin, feats


def get_patched_margin(
    hf_model, tokenizer, prompt: str, correct: str, incorrect: str,
    feats_self: Dict[int, torch.Tensor],
    feats_other: Dict[int, torch.Tensor],
    patch_by_layer: Dict[int, List[int]],
    transcoder_set, device: str,
) -> float:
    """Patch features at specified layers, measure margin."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False)["input_ids"]
    if not correct_ids or not incorrect_ids:
        return 0.0
    correct_id = correct_ids[0]
    incorrect_id = incorrect_ids[0]

    handles = []
    for layer, patch_feats in patch_by_layer.items():
        if not patch_feats:
            continue
        block = hf_model.model.layers[layer]
        tc = transcoder_set[layer]
        a_self = feats_self[layer].clone()
        a_other = feats_other[layer]
        a_self[:, patch_feats] = a_other[:, patch_feats]
        with torch.no_grad():
            x_patched = tc.decode(a_self.to(tc.dtype)).to(torch.bfloat16)

        def _make(layer=layer, _x=x_patched.to(device)):
            def _hook(module, inp, out):
                mod = out.clone()
                mod[0, -1, :] = _x[0, :]
                return mod
            return _hook
        handles.append(block.post_attention_layernorm.register_forward_hook(_make()))

    try:
        with torch.no_grad():
            outputs = hf_model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]
            margin = float((logits[correct_id] - logits[incorrect_id]).item())
    finally:
        for h in handles:
            h.remove()
    return margin


def compute_iia_for_subset(
    hf_model, tokenizer, transcoder_set,
    pool_features: List[str],
    subset_features: List[str],
    pairs: List[Tuple[Dict, Dict]],
    baseline_cache: Dict,
    device: str,
) -> Tuple[int, int]:
    """Compute IIA for a subset of features. Returns (n_flips, n_tested)."""
    # Group subset by layer
    layer_feats = defaultdict(list)
    for fid in subset_features:
        layer, idx = parse_feature_id(fid)
        layer_feats[layer].append(idx)
    if not layer_feats:
        return 0, 0
    layers = sorted(layer_feats.keys())

    flip_count = 0
    n_tested = 0
    for pair_id, (p_a, p_b) in enumerate(pairs):
        pa_idx = p_a["prompt_idx"] if "prompt_idx" in p_a else pair_id * 2
        pb_idx = p_b["prompt_idx"] if "prompt_idx" in p_b else pair_id * 2 + 1
        # Use cached baseline if available
        if (pa_idx, "margin") not in baseline_cache:
            continue
        if (pb_idx, "margin") not in baseline_cache:
            continue
        m_a = baseline_cache[(pa_idx, "margin")]
        m_b = baseline_cache[(pb_idx, "margin")]
        feats_a = {l: baseline_cache[(pa_idx, l)] for l in layers if (pa_idx, l) in baseline_cache}
        feats_b = {l: baseline_cache[(pb_idx, l)] for l in layers if (pb_idx, l) in baseline_cache}
        if not feats_a or not feats_b:
            continue

        correct_a = " " + p_a["correct_answer"].strip()
        incorrect_a = " " + p_a["incorrect_answer"].strip()
        correct_b = " " + p_b["correct_answer"].strip()
        incorrect_b = " " + p_b["incorrect_answer"].strip()

        # Direction 1: α-prompt with β features
        try:
            m_a_p = get_patched_margin(
                hf_model, tokenizer, p_a["prompt"], correct_a, incorrect_a,
                feats_a, feats_b, layer_feats, transcoder_set, device)
            if m_a != 0 and m_a_p != 0:
                flip_count += int(np.sign(m_a_p) != np.sign(m_a))
                n_tested += 1
        except Exception as e:
            log.debug(f"  pair {pair_id} dir1 failed: {e}")

        # Direction 2: β-prompt with α features
        try:
            m_b_p = get_patched_margin(
                hf_model, tokenizer, p_b["prompt"], correct_b, incorrect_b,
                feats_b, feats_a, layer_feats, transcoder_set, device)
            if m_b != 0 and m_b_p != 0:
                flip_count += int(np.sign(m_b_p) != np.sign(m_b))
                n_tested += 1
        except Exception as e:
            log.debug(f"  pair {pair_id} dir2 failed: {e}")

    return flip_count, n_tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_layers", nargs="+", default=["L24", "L18", "random_l25"],
                    help="Target pools: 'L<n>' (all features of layer), or 'random_l<n>' "
                         "(random pool from layer n same size as L24)")
    ap.add_argument("--cluster_col", default="agglo_coimp_subgroup_k30")
    ap.add_argument("--clustering_dir", type=Path,
                    default=Path("data/analysis/runD_v2/clustering_full"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("data/analysis/runD_v2/graded_patching"))
    ap.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0])
    ap.add_argument("--n_subsets", type=int, default=5,
                    help="Number of random subsets per α")
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Build feature pools ────────────────────────────────────────────────
    cl = pd.read_csv(args.clustering_dir / "cluster_labels.csv")
    cl["layer"] = cl["feature_id"].str.extract(r"L(\d+)_").astype(int)

    pools: Dict[str, List[str]] = {}
    # Reference for sizing random pools = L24 size
    ref_layer = 24
    ref_pool = cl[cl["layer"] == ref_layer]["feature_id"].tolist()
    ref_size = len(ref_pool)

    for spec in args.target_layers:
        if spec.startswith("random_l"):
            layer_n = int(spec.replace("random_l", ""))
            layer_features = cl[cl["layer"] == layer_n]["feature_id"].tolist()
            n = min(ref_size, len(layer_features))
            if n < ref_size:
                log.warning(f"  {spec}: only {len(layer_features)} features at L{layer_n}, "
                            f"using {n} (less than L{ref_layer}'s {ref_size})")
            pool = list(rng.choice(layer_features, size=n, replace=False))
            pools[spec] = pool
        elif spec.startswith("L"):
            layer_n = int(spec.lstrip("L"))
            pool = cl[cl["layer"] == layer_n]["feature_id"].tolist()
            pools[spec] = pool
        else:
            log.warning(f"Unknown target spec: {spec} — skip")

    for pname, pool in pools.items():
        log.info(f"Pool {pname}: {len(pool)} features (layer {parse_feature_id(pool[0])[0] if pool else 'n/a'})")

    # ── Load prompts ────────────────────────────────────────────────────────
    prompt_path = Path(args.prompts_file) if args.prompts_file else \
                  ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts_all = [json.loads(l) for l in open(prompt_path)]
    log.info(f"Total prompts: {len(prompts_all)}")

    alpha_prompts = [p for p in prompts_all if p["correct_answer"].strip() == "alpha"]
    beta_prompts = [p for p in prompts_all if p["correct_answer"].strip() == "beta"]
    rng2 = np.random.default_rng(args.seed)
    n_pairs = min(len(alpha_prompts), len(beta_prompts), args.max_pairs)
    a_idx = rng2.choice(len(alpha_prompts), size=n_pairs, replace=False)
    b_idx = rng2.choice(len(beta_prompts), size=n_pairs, replace=False)
    # Add unique indices for caching
    for i, p in enumerate(alpha_prompts):
        p["prompt_idx"] = i
    for i, p in enumerate(beta_prompts):
        p["prompt_idx"] = i + 10**6
    pairs = [(alpha_prompts[i], beta_prompts[j]) for i, j in zip(a_idx, b_idx)]
    log.info(f"Using {n_pairs} α/β pairs")

    # ── Load model + transcoders ────────────────────────────────────────────
    all_layers = sorted({parse_feature_id(f)[0] for pool in pools.values() for f in pool})
    log.info(f"Loading model + transcoders for layers: {all_layers}")
    from src.model_utils import load_model
    from src.transcoder.transcoder_loader import load_transcoder_set

    model, model_size = load_model(args.device)
    model.model.eval()
    try:
        args.device = str(next(model.model.parameters()).device)
    except StopIteration:
        pass
    tc_set = load_transcoder_set(model_size=model_size, device=args.device,
                                  dtype=torch.bfloat16, lazy_load=True,
                                  layers=all_layers)
    log.info("Model + transcoders loaded.")

    # ── Cache baseline activations for all pairs ───────────────────────────
    log.info("\nCaching baseline activations for pairs...")
    baseline_cache = {}
    unique_prompts = set()
    for p_a, p_b in pairs:
        unique_prompts.add((p_a["prompt_idx"], "alpha"))
        unique_prompts.add((p_b["prompt_idx"], "beta"))

    # Map from prompt_idx to prompt
    prompt_lookup = {}
    for p in alpha_prompts:
        prompt_lookup[p["prompt_idx"]] = p
    for p in beta_prompts:
        prompt_lookup[p["prompt_idx"]] = p

    for pi_idx, _ in unique_prompts:
        p = prompt_lookup[pi_idx]
        correct = " " + p["correct_answer"].strip()
        incorrect = " " + p["incorrect_answer"].strip()
        m, feats = get_features_and_margin(
            model.model, model.tokenizer, p["prompt"], correct, incorrect,
            tc_set, all_layers, args.device)
        baseline_cache[(pi_idx, "margin")] = m
        for layer, a in feats.items():
            baseline_cache[(pi_idx, layer)] = a
    log.info(f"Cached {len(unique_prompts)} prompts × baseline + activations")

    # ── Run graded experiment ────────────────────────────────────────────
    rng3 = np.random.default_rng(args.seed)
    rows = []
    for pname, pool in pools.items():
        log.info(f"\n=== Pool: {pname} ({len(pool)} features) ===")
        for alpha in sorted(args.alphas):
            k = max(1, int(round(alpha * len(pool))))
            log.info(f"  α={alpha:.2f} → k={k} features per subset")
            for subset_seed in range(args.n_subsets):
                seed = args.seed + 1000 * int(alpha * 100) + subset_seed
                rng_local = np.random.default_rng(seed)
                subset = list(rng_local.choice(pool, size=k, replace=False))
                n_flips, n_tested = compute_iia_for_subset(
                    model.model, model.tokenizer, tc_set,
                    pool, subset, pairs, baseline_cache, args.device)
                iia = n_flips / n_tested if n_tested > 0 else 0.0
                rows.append({
                    "pool": pname,
                    "alpha": alpha,
                    "k_features": k,
                    "subset_seed": subset_seed,
                    "n_pairs": n_pairs,
                    "n_tested": n_tested,
                    "n_flips": n_flips,
                    "iia": iia,
                    "features": ",".join(subset),
                })
                log.info(f"    seed={subset_seed} k={k}: IIA={iia:.4f} ({n_flips}/{n_tested})")

    df = pd.DataFrame(rows)
    raw_path = args.out_dir / "graded_patching_raw.csv"
    df.to_csv(raw_path, index=False)
    log.info(f"\nSaved raw → {raw_path}")

    # Summary
    summary = df.groupby(["pool", "alpha"]).agg(
        k_features=("k_features", "first"),
        n_subsets=("subset_seed", "count"),
        mean_iia=("iia", "mean"),
        std_iia=("iia", "std"),
        min_iia=("iia", "min"),
        max_iia=("iia", "max"),
    ).reset_index()
    sum_path = args.out_dir / "graded_patching_summary.csv"
    summary.to_csv(sum_path, index=False)
    log.info(f"Saved summary → {sum_path}")

    print("\n=== GRADED PATCHING — IIA(α) per pool ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
