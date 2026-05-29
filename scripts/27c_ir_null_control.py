"""
Script 27c: IR Null Control Experiment.

Tests whether the low Interaction Ratios (IR ≤ 0.35) observed in real
co-importance clusters are meaningful, by comparing against random feature
bundles of the same size.

Motivation
----------
The IR threshold of 0.35 is post-hoc (observed maximum from runB). The claim
that IR_null ≈ 1 for random bundles is analytically plausible but untested.
This script provides the empirical null distribution.

Method
------
For each unique cluster size k among the real clusters:
  1. Sample N_null random bundles of k features from the full feature pool.
  2. For each bundle, run joint ablation on N_prompts_null prompts
     (random subsample of the 538-prompt corpus).
  3. Compute IR = joint_effect / individual_sum for each (bundle, prompt).
  4. Report empirical IR_null distribution and compare to real cluster IR values.

Outputs (in --run_dir):
  ir_null_distribution.csv    — per (bundle, prompt) IR values
  ir_null_summary.csv         — per unique-size summary (mean, std, percentiles)
  ir_null_vs_real.csv         — per real cluster: real IR vs null p-value

Usage (CSD3):
    python scripts/27c_ir_null_control.py \\
        --grouping_dir   data/analysis/runD_v2/grouping \\
        --clustering_dir data/analysis/runD_v2/clustering_full \\
        --joint_dir      data/analysis/runD_v2/cluster_joint_ablation \\
        --run_dir        data/analysis/runD_v2/ir_null_control \\
        --n_null_per_size 20 \\
        --n_prompts_null  100 \\
        --device cuda
"""

import json, yaml, argparse, sys, logging, contextlib, time, gc
from pathlib import Path
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


# ── Reuse core helpers from script 27 (self-contained copy) ──────────────────

def get_mlp_input(model, inputs, layer_idx, token_pos=-1):
    try:
        block = model.model.model.layers[layer_idx]
    except AttributeError:
        block = model.model.transformer.h[layer_idx]
    hook_mod = block.post_attention_layernorm
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
    return captured["x"][:, token_pos, :]


@contextlib.contextmanager
def patch_mlp_layer(model_hf, layer_idx, token_pos, new_mlp_input):
    try:
        block = model_hf.model.layers[layer_idx]
    except AttributeError:
        block = model_hf.transformer.h[layer_idx]
    hook_mod = block.post_attention_layernorm

    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        t[:, token_pos, :] = new_mlp_input.to(t.dtype).to(t.device)
        return (t,) + out[1:] if isinstance(out, tuple) else t

    h = hook_mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def run_joint_ablation(model, transcoder_set, prompt, cluster_by_layer, token_pos=-1):
    """Returns (base_logits, joint_logits). Patches all layers in one pass."""
    device = next(model.model.parameters()).device
    inputs = model.tokenize([prompt])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        base_out = model.model(**inputs, use_cache=False)
    base_logits = base_out.logits[0, -1, :]

    modified_per_layer = {}
    for layer_idx, feat_indices in cluster_by_layer.items():
        act = get_mlp_input(model, inputs, layer_idx, token_pos)
        tc  = transcoder_set[layer_idx]
        with torch.no_grad():
            feats = tc.encode(act.to(tc.dtype))
            feats[:, feat_indices] = 0.0
            mod = tc.decode(feats).to(act.dtype)
        modified_per_layer[layer_idx] = mod.squeeze(0)

    with contextlib.ExitStack() as stack:
        for layer_idx, mod_input in modified_per_layer.items():
            stack.enter_context(
                patch_mlp_layer(model.model, layer_idx, token_pos, mod_input)
            )
        with torch.no_grad():
            joint_out = model.model(**inputs, use_cache=False)
    joint_logits = joint_out.logits[0, -1, :]

    return base_logits, joint_logits


def parse_layer_feat(fid):
    layer = int(fid.split("_")[0][1:])
    feat  = int(fid.split("_F")[1])
    return layer, feat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour",        default="physics_decay_type_probe_v2")
    parser.add_argument("--split",            default="train")
    parser.add_argument("--grouping_dir",     type=Path, required=True)
    parser.add_argument("--clustering_dir",   type=Path, required=True)
    parser.add_argument("--joint_dir",        type=Path, default=None,
                        help="Dir with real joint ablation results (joint_ablation_*.csv). "
                             "If omitted, real IR comparison is skipped.")
    parser.add_argument("--run_dir",          type=Path, required=True,
                        help="Output directory for null results")
    parser.add_argument("--n_null_per_size",  type=int, default=20,
                        help="Random bundles to sample per unique cluster size")
    parser.add_argument("--n_prompts_null",   type=int, default=100,
                        help="Prompts per bundle (subsample of full corpus)")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--device",           default="cuda")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load cluster definitions ──────────────────────────────────────────
    import csv as csvlib
    with open(args.clustering_dir / "cluster_labels.csv") as f:
        rows = list(csvlib.DictReader(f))

    all_feature_ids = [r["feature_id"] for r in rows]
    coimp = {r["feature_id"]: int(r["coimp_louvain"]) for r in rows}
    real_clusters: dict[int, list[str]] = defaultdict(list)
    for fid, cid in coimp.items():
        real_clusters[cid].append(fid)

    logger.info(f"Feature pool: {len(all_feature_ids)} features")
    logger.info(f"Real clusters: {len(real_clusters)} clusters")
    for cid in sorted(real_clusters):
        logger.info(f"  C{cid}: {len(real_clusters[cid])} features")

    # Unique cluster sizes (group runs to save redundant model loads)
    size_to_clusters = defaultdict(list)
    for cid, feats in real_clusters.items():
        size_to_clusters[len(feats)].append(cid)

    unique_sizes = sorted(size_to_clusters.keys())
    logger.info(f"Unique cluster sizes: {unique_sizes}")

    # ── Load individual effects (Δk per feature per prompt) ───────────────
    logger.info("Loading individual effects from feature_prompt_contributions.csv…")
    contrib = pd.read_csv(
        args.grouping_dir / "feature_prompt_contributions.csv",
        usecols=["prompt_idx", "feature_id", "effect_size"],
    )
    # Build lookup: {feature_id: {prompt_idx: effect_size}}
    indiv_effects: dict[str, dict[int, float]] = defaultdict(dict)
    for _, row in contrib.iterrows():
        indiv_effects[row["feature_id"]][int(row["prompt_idx"])] = float(row["effect_size"])
    logger.info(f"  Loaded effects for {len(indiv_effects)} features")

    # ── Load prompts ──────────────────────────────────────────────────────
    prompts_path = ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    all_prompts = []
    with open(prompts_path) as f:
        for line in f:
            all_prompts.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(all_prompts)} prompts")

    # Subsample prompt indices (deterministic; same subset across all bundles)
    n_sub = min(args.n_prompts_null, len(all_prompts))
    sub_indices = sorted(rng.choice(len(all_prompts), size=n_sub, replace=False).tolist())
    sub_prompts = [all_prompts[i] for i in sub_indices]
    logger.info(f"Using {n_sub} prompts for null ablations (seed={args.seed})")

    # ── Load model and transcoders ────────────────────────────────────────
    logger.info("Loading model…")
    tc_cfg = yaml.safe_load(open(ROOT / "configs/transcoder_config.yaml"))
    model_size = tc_cfg.get("model_size", "4b")
    model_name = tc_cfg["transcoders"][model_size]["model_name"]

    all_layers = sorted(set(parse_layer_feat(fid)[0] for fid in all_feature_ids))
    logger.info(f"Transcoder layers needed: {all_layers}")

    model = ModelWrapper(
        model_name=model_name, dtype="bfloat16", device="auto", trust_remote_code=True
    )
    try:
        device = next(model.model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model on: {device}")

    transcoder_set = load_transcoder_set(
        model_size=model_size, device=device, dtype=torch.bfloat16,
        lazy_load=True, layers=all_layers,
    )
    logger.info("Transcoders loaded.")

    # ── Precompute baseline logit diffs for sub_prompts ───────────────────
    logger.info("Precomputing baselines for subsample prompts…")
    baselines: dict[int, tuple[int, int, float]] = {}  # prompt_i → (cid_tok, iid_tok, margin)
    skip_count = 0
    for p_i, p in enumerate(tqdm(sub_prompts, desc="Baselines")):
        correct_tok   = p.get("correct_answer",   " alpha")
        incorrect_tok = p.get("incorrect_answer",  " beta")
        try:
            cid_tok = model.tokenizer.encode(correct_tok,   add_special_tokens=False)
            iid_tok = model.tokenizer.encode(incorrect_tok, add_special_tokens=False)
            assert len(cid_tok) == 1 and len(iid_tok) == 1
        except AssertionError:
            skip_count += 1
            continue
        correct_id   = cid_tok[0]
        incorrect_id = iid_tok[0]

        device_ = next(model.model.parameters()).device
        inputs  = model.tokenize([p["prompt"]])
        inputs  = {k: v.to(device_) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.model(**inputs, use_cache=False)
        lp     = torch.log_softmax(out.logits[0, -1, :].float(), dim=0)
        margin = float(lp[correct_id] - lp[incorrect_id])
        baselines[p_i] = (correct_id, incorrect_id, margin)

    logger.info(f"Baselines computed: {len(baselines)} prompts ({skip_count} skipped)")

    # ── Run null bundles ──────────────────────────────────────────────────
    all_rows = []
    t0 = time.time()
    bundle_id = 0

    for k in unique_sizes:
        real_cids = size_to_clusters[k]
        logger.info(f"\n=== Null bundles for size k={k} (real clusters: {real_cids}) ===")
        logger.info(f"    Sampling {args.n_null_per_size} random bundles × {n_sub} prompts")

        for b_i in range(args.n_null_per_size):
            # Sample k features uniformly from the full pool (without replacement)
            bundle_fids = rng.choice(all_feature_ids, size=k, replace=False).tolist()
            bundle_id  += 1

            # Build layer→feat_idx mapping for this bundle
            bundle_by_layer: dict[int, list[int]] = defaultdict(list)
            for fid in bundle_fids:
                l, f = parse_layer_feat(fid)
                bundle_by_layer[int(l)].append(f)
            bundle_by_layer = dict(bundle_by_layer)

            # Run joint ablation for each subsample prompt
            n_done = 0
            for p_i, p in enumerate(sub_prompts):
                if p_i not in baselines:
                    continue

                correct_id, incorrect_id, base_margin = baselines[p_i]
                prompt_idx = p.get("prompt_idx", sub_indices[p_i])

                base_logits, joint_logits = run_joint_ablation(
                    model, transcoder_set, p["prompt"], bundle_by_layer, token_pos=-1
                )

                lp_joint   = torch.log_softmax(joint_logits.float(), dim=0)
                joint_margin = float(lp_joint[correct_id] - lp_joint[incorrect_id])
                joint_effect = joint_margin - base_margin

                # Individual sum from pre-computed contributions
                indiv_sum_val = sum(
                    indiv_effects.get(fid, {}).get(int(prompt_idx), 0.0)
                    for fid in bundle_fids
                )

                eps = 1e-6
                ir  = (joint_effect / indiv_sum_val
                       if abs(indiv_sum_val) > eps else float("nan"))

                all_rows.append({
                    "bundle_id":     bundle_id,
                    "size_k":        k,
                    "bundle_index":  b_i,
                    "prompt_i":      p_i,
                    "prompt_idx":    prompt_idx,
                    "joint_effect":  round(joint_effect, 5),
                    "indiv_sum":     round(indiv_sum_val, 5),
                    "ir":            round(ir, 5) if ir == ir else float("nan"),
                    "base_margin":   round(base_margin, 5),
                    "joint_margin":  round(joint_margin, 5),
                })
                n_done += 1

            elapsed = time.time() - t0
            logger.info(
                f"  k={k} bundle {b_i+1}/{args.n_null_per_size} "
                f"({n_done} prompts) | {elapsed/60:.1f} min elapsed"
            )

    # ── Save raw distribution ─────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(args.run_dir / "ir_null_distribution.csv", index=False)
    logger.info(f"Saved ir_null_distribution.csv ({len(df)} rows)")

    # ── Summary per unique size ───────────────────────────────────────────
    summary_rows = []
    for k in unique_sizes:
        sub = df[df["size_k"] == k].dropna(subset=["ir"])
        if len(sub) == 0:
            continue
        ir_vals = sub["ir"].values
        summary_rows.append({
            "size_k":         k,
            "n_bundles":      args.n_null_per_size,
            "n_prompts_each": n_sub,
            "n_ir_values":    len(ir_vals),
            "mean_ir":        round(float(np.mean(ir_vals)),   4),
            "median_ir":      round(float(np.median(ir_vals)), 4),
            "std_ir":         round(float(np.std(ir_vals)),    4),
            "p5_ir":          round(float(np.percentile(ir_vals, 5)),  4),
            "p25_ir":         round(float(np.percentile(ir_vals, 25)), 4),
            "p75_ir":         round(float(np.percentile(ir_vals, 75)), 4),
            "p95_ir":         round(float(np.percentile(ir_vals, 95)), 4),
            "frac_near_1":    round(float(((ir_vals > 0.8) & (ir_vals < 1.2)).mean()), 3),
            "frac_below_05":  round(float((ir_vals < 0.5).mean()), 3),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.run_dir / "ir_null_summary.csv", index=False)

    print("\n=== IR Null Distribution Summary ===")
    print(summary_df[["size_k", "mean_ir", "median_ir", "std_ir",
                       "p5_ir", "p95_ir", "frac_near_1"]].to_string(index=False))
    print("\nKEY: If null IR ≈ 1, random bundles are approximately additive.")
    print("     Real cluster IR << 1 would then be meaningful (p-values below).")

    # ── Compare real cluster IR to null distribution ───────────────────────
    joint_csv = None
    if args.joint_dir is not None:
        cands = list(args.joint_dir.glob(f"joint_ablation_{args.behaviour}_{args.split}.csv"))
        if cands:
            joint_csv = cands[0]

    if joint_csv and joint_csv.exists():
        real_df = pd.read_csv(joint_csv)
        # Compute mean IR per real cluster
        real_ir = (
            real_df.dropna(subset=["interaction_ratio"])
            .groupby("cluster_id")["interaction_ratio"]
            .mean()
            .reset_index()
            .rename(columns={"interaction_ratio": "real_mean_ir", "cluster_id": "cluster_id"})
        )

        comparison_rows = []
        for _, rrow in real_ir.iterrows():
            cid    = int(rrow["cluster_id"])
            real_k = len(real_clusters.get(cid, []))
            r_ir   = float(rrow["real_mean_ir"])

            # Null IR values for this cluster size
            null_sub = df[df["size_k"] == real_k].dropna(subset=["ir"])
            if len(null_sub) == 0:
                p_val = float("nan")
                null_mean = float("nan")
            else:
                null_ir_vals = null_sub["ir"].values
                # p-value: fraction of null IR values ≤ real IR
                p_val      = float((null_ir_vals <= r_ir).mean())
                null_mean  = float(np.mean(null_ir_vals))

            comparison_rows.append({
                "cluster_id":   cid,
                "size_k":       real_k,
                "real_mean_ir": round(r_ir, 4),
                "null_mean_ir": round(null_mean, 4) if null_mean == null_mean else float("nan"),
                "p_value":      round(p_val, 4) if p_val == p_val else float("nan"),
                "significant":  (p_val < 0.05) if p_val == p_val else False,
            })

        comp_df = pd.DataFrame(comparison_rows).sort_values("cluster_id")
        comp_df.to_csv(args.run_dir / "ir_null_vs_real.csv", index=False)

        print("\n=== Real IR vs Null Distribution ===")
        print(comp_df.to_string(index=False))
        n_sig = int(comp_df["significant"].sum())
        print(f"\n{n_sig}/{len(comp_df)} clusters have IR significantly below null (p < 0.05)")
        print("(p_value = fraction of null bundles with IR ≤ real cluster IR)")
    else:
        logger.info("Joint ablation CSV not found — skipping real vs null comparison.")
        logger.info(f"  Looked in: {args.joint_dir}")
        logger.info("  Run script 27 first, then re-run this script.")

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary_json = {
        "n_null_per_size":   args.n_null_per_size,
        "n_prompts_null":    n_sub,
        "seed":              args.seed,
        "unique_sizes":      unique_sizes,
        "real_clusters":     {str(cid): len(feats)
                              for cid, feats in sorted(real_clusters.items())},
        "null_summary":      summary_df.to_dict(orient="records"),
    }
    with open(args.run_dir / "ir_null_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    elapsed_total = time.time() - t0
    logger.info(f"\nDone in {elapsed_total/60:.1f} min. Results in: {args.run_dir}")


if __name__ == "__main__":
    main()
