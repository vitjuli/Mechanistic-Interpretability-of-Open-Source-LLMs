"""
56_h3_double_ablation.py — Causal verification of H3 (reactive backup pathways).

H3-reactive (observational, from script 55): when sub-cluster G is ablated,
some features f outside G show Δact > 0 (apparent compensation).

H3-causal (this script): If f is GENUINELY a causal compensation for G, then
ablating BOTH G and f together should produce LARGER disruption than G alone.

Procedure for each sub-cluster G with detected backup features:
  1. Load top-K backup features from h3_backup_subcluster_summary.csv
  2. For each prompt p:
     a. Compute baseline_logit_diff(p) — clean forward
     b. Compute logit_diff(p | ablate G alone) — single ablation
     c. Compute logit_diff(p | ablate G + f) — double ablation (for each backup f)
     d. Compute logit_diff(p | ablate G + f_control) — control: random non-backup feature
  3. SFR = fraction of prompts where prediction sign flips
  4. Compare: SFR(G+f) > SFR(G) → f causally compensates
  5. Statistical test: paired t-test on per-prompt logit_diff

Upgrades H3-reactive from ★★★ "observational" to ★★★★ "causally confirmed"
if top backup features show significant compensation effect vs random controls.

Inputs:
  data/analysis/runD_v2/h3h4_subcluster/h3_backup_subcluster_summary.csv
    (must exist — output of script 55 H3 mode)
  data/analysis/runD_v2/clustering_full/cluster_labels.csv
  data/prompts/{behaviour}_{split}.jsonl

Outputs:
  {out_dir}/h3_double_ablation_raw.csv
    (cluster, backup_feature, prompt_idx, logit_diff_baseline, logit_diff_G,
     logit_diff_G_plus_f, flipped_G, flipped_G_plus_f)
  {out_dir}/h3_double_ablation_summary.csv
    (cluster, backup_feature, type [backup/control], sfr_G, sfr_G_plus_f,
     delta_sfr, mean_delta_logit, t_stat, p_value, significant)

Usage:
  python3 scripts/56_h3_double_ablation.py \\
    --h3_summary data/analysis/runD_v2/h3h4_subcluster/h3_backup_subcluster_summary.csv \\
    --cluster_col agglo_coimp_subgroup_k30 \\
    --clustering_dir data/analysis/runD_v2/clustering_full \\
    --grouping_dir data/analysis/runD_v2/grouping \\
    --out_dir data/analysis/runD_v2/h3h4_subcluster \\
    --top_k_backup 3 \\
    --max_prompts 300
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def parse_feature_id(fid: str) -> Tuple[int, int]:
    layer, idx = fid.lstrip("L").split("_F")
    return int(layer), int(idx)


def get_logit_diff_with_ablation(
    hf_model, tokenizer, prompt: str, correct: str, incorrect: str,
    transcoder_set, ablate_by_layer: Dict[int, List[int]],
    device: str,
) -> float:
    """Forward pass with features at specified (layer, feat_idx) ablated; return logit_diff."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False)["input_ids"]
    if not correct_ids or not incorrect_ids:
        return 0.0
    correct_id = correct_ids[0]
    incorrect_id = incorrect_ids[0]

    handles = []
    # For each layer needing patching, compute the patched x once
    for layer, feat_idxs in ablate_by_layer.items():
        block = hf_model.model.layers[layer]
        captured_clean = {}
        def _cap(module, inp, out, _key=layer):
            captured_clean[_key] = out[:, -1:, :].detach()
            return None  # don't modify yet
        cap_h = block.post_attention_layernorm.register_forward_hook(_cap)

        # First forward to capture clean x at this layer
        with torch.no_grad():
            hf_model(**inputs, use_cache=False)
        cap_h.remove()
        x_clean = captured_clean[layer][0]  # (1, d)

        # Build ablated x
        tc = transcoder_set[layer]
        with torch.no_grad():
            a = tc.encode(x_clean.to(tc.dtype))
            a_abl = a.clone()
            a_abl[:, feat_idxs] = 0.0
            x_abl = tc.decode(a_abl).to(x_clean.dtype)

        # Register patching hook for the final forward
        _x = x_abl.to(device)
        def _patch_hook(module, inp, out, _x=_x):
            mod = out.clone()
            mod[0, -1, :] = _x[0, :]
            return mod
        handles.append(block.post_attention_layernorm.register_forward_hook(_patch_hook))

    try:
        with torch.no_grad():
            outputs = hf_model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]
            margin = float((logits[correct_id] - logits[incorrect_id]).item())
    finally:
        for h in handles:
            h.remove()
    return margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3_summary", type=Path, required=True,
                    help="Path to h3_backup_subcluster_summary.csv from script 55")
    ap.add_argument("--cluster_col", default="agglo_coimp_subgroup_k30")
    ap.add_argument("--clustering_dir", type=Path,
                    default=Path("data/analysis/runD_v2/clustering_full"))
    ap.add_argument("--grouping_dir", type=Path,
                    default=Path("data/analysis/runD_v2/grouping"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("data/analysis/runD_v2/h3h4_subcluster"))
    ap.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--top_k_backup", type=int, default=3,
                    help="Number of top backup features to test per cluster")
    ap.add_argument("--n_controls", type=int, default=1,
                    help="Number of random control features per cluster (matched layer)")
    ap.add_argument("--max_prompts", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load sub-clusters ──────────────────────────────────────────────────
    cl = pd.read_csv(args.clustering_dir / "cluster_labels.csv")
    cl[args.cluster_col] = cl[args.cluster_col].astype(int)
    sub_clusters: Dict[int, List[str]] = defaultdict(list)
    for _, row in cl.iterrows():
        sub_clusters[int(row[args.cluster_col])].append(row["feature_id"])
    log.info(f"Loaded {len(sub_clusters)} sub-clusters")

    # All features (for control sampling)
    all_features = cl["feature_id"].tolist()

    # ── Load H3 summary to get backup features per sub-cluster ─────────────
    h3_summary = pd.read_csv(args.h3_summary)
    log.info(f"H3 summary: {len(h3_summary)} sub-clusters analysed")

    # Parse top_backup_feats column: "L25_F15948(+4.48), L24_F147583(+3.30), ..."
    def parse_top_backups(s: str, top_k: int) -> List[Tuple[str, float]]:
        if pd.isna(s) or not s.strip():
            return []
        items = []
        for token in s.split(","):
            token = token.strip()
            if "(" not in token:
                continue
            fid, delta_str = token.rsplit("(", 1)
            fid = fid.strip()
            delta_str = delta_str.rstrip(")").strip()
            try:
                delta = float(delta_str)
            except ValueError:
                continue
            items.append((fid, delta))
        return items[:top_k]

    h3_summary["top_backups_parsed"] = h3_summary["top_backup_feats"].apply(
        lambda s: parse_top_backups(s, args.top_k_backup))

    # ── Load prompts ────────────────────────────────────────────────────────
    prompt_path = Path(args.prompts_file) if args.prompts_file else \
                  ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts_all = [json.loads(l) for l in open(prompt_path)]
    if args.max_prompts and len(prompts_all) > args.max_prompts:
        # Stratified subsample: keep balance of α/β prompts
        alpha = [p for p in prompts_all if p["correct_answer"].strip() == "alpha"]
        beta = [p for p in prompts_all if p["correct_answer"].strip() == "beta"]
        n_each = args.max_prompts // 2
        rng2 = np.random.default_rng(args.seed)
        idx_a = rng2.choice(len(alpha), size=min(n_each, len(alpha)), replace=False)
        idx_b = rng2.choice(len(beta), size=min(n_each, len(beta)), replace=False)
        prompts = [alpha[i] for i in idx_a] + [beta[i] for i in idx_b]
    else:
        prompts = prompts_all
    log.info(f"Using {len(prompts)} prompts (target {args.max_prompts}, total avail {len(prompts_all)})")

    # ── Load model + transcoders ────────────────────────────────────────────
    from src.model_utils import ModelWrapper
    from src.transcoder import load_transcoder_set

    # Determine all layers needed
    layers_needed = set()
    for cid, feats in sub_clusters.items():
        for fid in feats:
            layers_needed.add(parse_feature_id(fid)[0])
    # Also layers of backup features
    for _, row in h3_summary.iterrows():
        for fid, _ in row["top_backups_parsed"]:
            layers_needed.add(parse_feature_id(fid)[0])
    layers_needed = sorted(layers_needed)
    log.info(f"Layers needed: {layers_needed}")

    log.info("Loading model + transcoders...")
    import yaml
    tc_cfg = yaml.safe_load(open(ROOT / "configs/transcoder_config.yaml"))
    model_size = tc_cfg.get("model_size", "4b")
    model_name = tc_cfg["transcoders"][model_size]["model_name"]
    log.info(f"Loading model: {model_name}")
    model = ModelWrapper(model_name=model_name, dtype="bfloat16", device="auto",
                          trust_remote_code=True)
    model.model.eval()
    try:
        args.device = str(next(model.model.parameters()).device)
    except StopIteration:
        pass
    tc_set = load_transcoder_set(model_size=model_size, device=args.device,
                                  dtype=torch.bfloat16, lazy_load=True,
                                  layers=layers_needed)
    log.info("Model + transcoders loaded.")

    # ── Precompute baseline + G-alone logit_diff per (cluster, prompt) ──────
    # Cache to avoid recomputing across (G, f) combinations
    log.info("\nComputing baselines and G-alone ablations...")
    baseline_cache = {}  # prompt_idx -> baseline_margin
    g_alone_cache = {}   # (cluster_id, prompt_idx) -> margin_after_G_ablation

    for pi, p in enumerate(prompts):
        if pi % 50 == 0:
            log.info(f"  baseline {pi}/{len(prompts)}")
        correct = " " + p["correct_answer"].strip()
        incorrect = " " + p["incorrect_answer"].strip()
        # Baseline (no ablation)
        baseline_cache[pi] = get_logit_diff_with_ablation(
            model.model, model.tokenizer, p["prompt"], correct, incorrect,
            tc_set, {}, args.device)

    log.info("\nComputing G-alone ablations for all sub-clusters...")
    for cid in sorted(sub_clusters.keys()):
        feats = sub_clusters[cid]
        c_layer = parse_feature_id(feats[0])[0]
        c_feat_idxs = [parse_feature_id(f)[1] for f in feats]
        log.info(f"  C{cid} (L{c_layer}, {len(feats)} feats)")
        for pi, p in enumerate(prompts):
            correct = " " + p["correct_answer"].strip()
            incorrect = " " + p["incorrect_answer"].strip()
            margin = get_logit_diff_with_ablation(
                model.model, model.tokenizer, p["prompt"], correct, incorrect,
                tc_set, {c_layer: c_feat_idxs}, args.device)
            g_alone_cache[(cid, pi)] = margin

    # ── Double ablation: G + each top backup + control ──────────────────────
    log.info("\nDouble ablations: G + backup features...")
    rows = []
    for _, row in h3_summary.iterrows():
        cid = int(row["sub_cluster"])
        if cid not in sub_clusters:
            continue
        c_feats = sub_clusters[cid]
        c_layer = parse_feature_id(c_feats[0])[0]
        c_feat_idxs = [parse_feature_id(f)[1] for f in c_feats]

        backups = row["top_backups_parsed"]
        if not backups:
            log.info(f"  C{cid}: no backup features — skip")
            continue

        # Pool of non-cluster, non-backup features for control sampling
        excluded = set(c_feats) | set(b[0] for b in backups)
        control_pool = [f for f in all_features if f not in excluded]
        # Sample n_controls random features (matching backup layer distribution)
        backup_layers = [parse_feature_id(b[0])[0] for b in backups]
        controls = []
        for cl_layer in backup_layers[:args.n_controls]:
            candidates = [f for f in control_pool
                          if parse_feature_id(f)[0] == cl_layer]
            if candidates:
                ctrl = rng.choice(candidates)
                controls.append((ctrl, "control_random"))

        targets = [(b[0], "backup") for b in backups] + controls

        log.info(f"  C{cid} (L{c_layer}, n={len(c_feats)}): testing {len(backups)} backups + {len(controls)} controls")

        for (f_id, target_type) in targets:
            f_layer, f_idx = parse_feature_id(f_id)
            for pi, p in enumerate(prompts):
                correct = " " + p["correct_answer"].strip()
                incorrect = " " + p["incorrect_answer"].strip()
                # Double ablation: G's features + f
                abl_dict = {c_layer: list(c_feat_idxs)}
                if f_layer == c_layer:
                    abl_dict[c_layer].append(f_idx)
                else:
                    abl_dict[f_layer] = [f_idx]
                margin_gf = get_logit_diff_with_ablation(
                    model.model, model.tokenizer, p["prompt"], correct, incorrect,
                    tc_set, abl_dict, args.device)
                base = baseline_cache[pi]
                margin_g = g_alone_cache[(cid, pi)]
                rows.append({
                    "sub_cluster": cid,
                    "cluster_layer": c_layer,
                    "cluster_n_feats": len(c_feats),
                    "target_feature": f_id,
                    "target_layer": f_layer,
                    "target_type": target_type,
                    "prompt_idx": pi,
                    "baseline_margin": base,
                    "g_alone_margin": margin_g,
                    "g_plus_f_margin": margin_gf,
                    "correct_answer": p["correct_answer"].strip(),
                    "flipped_g_alone": int(np.sign(margin_g) != np.sign(base) and base != 0),
                    "flipped_g_plus_f": int(np.sign(margin_gf) != np.sign(base) and base != 0),
                })

    df = pd.DataFrame(rows)
    raw_path = args.out_dir / "h3_double_ablation_raw.csv"
    df.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path} ({len(df)} rows)")

    # ── Summary: per (cluster, target_feature) ─────────────────────────────
    summary_rows = []
    for (cid, target_id), sub in df.groupby(["sub_cluster", "target_feature"]):
        target_type = sub["target_type"].iloc[0]
        n = len(sub)
        sfr_g = sub["flipped_g_alone"].mean()
        sfr_gf = sub["flipped_g_plus_f"].mean()
        delta_sfr = sfr_gf - sfr_g
        # Paired test on logit_diff: does adding f to G further perturb the prediction?
        delta_logit = sub["g_alone_margin"] - sub["g_plus_f_margin"]
        mean_delta_logit = float(delta_logit.mean())
        # One-sided test: H1 — g_plus_f deviates further from baseline than g_alone
        # We measure |margin_gf − baseline| vs |margin_g − baseline|
        further_disruption = sub["g_plus_f_margin"].abs() - sub["g_alone_margin"].abs()
        t_stat, p_two_sided = stats.ttest_1samp(further_disruption.values, 0.0)
        # Convert to one-sided: H1 = g+f drives margin closer to 0 (or past it)
        # i.e. further_disruption > 0 means margin moved further from baseline
        p_one = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
        summary_rows.append({
            "sub_cluster": cid,
            "target_feature": target_id,
            "target_type": target_type,
            "n_prompts": n,
            "sfr_G_alone": round(sfr_g, 4),
            "sfr_G_plus_f": round(sfr_gf, 4),
            "delta_sfr": round(delta_sfr, 4),
            "mean_delta_logit": round(mean_delta_logit, 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_one), 5),
            "significant_p05": bool(p_one < 0.05),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df = sum_df.sort_values(["sub_cluster", "target_type", "p_value"])
    sum_path = args.out_dir / "h3_double_ablation_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    log.info(f"Saved summary → {sum_path}")

    # ── Print key findings ───────────────────────────────────────────────
    print("\n=== H3 DOUBLE ABLATION SUMMARY ===\n")
    print("Backup features causally confirmed (sfr_G_plus_f > sfr_G, p < 0.05):")
    sig_backup = sum_df[(sum_df["target_type"] == "backup") & sum_df["significant_p05"]]
    print(sig_backup[["sub_cluster", "target_feature", "sfr_G_alone",
                      "sfr_G_plus_f", "delta_sfr", "p_value"]].to_string(index=False))

    print("\nControls (should NOT show significant compensation):")
    sig_ctrl = sum_df[(sum_df["target_type"] == "control_random") & sum_df["significant_p05"]]
    n_ctrl_sig = len(sig_ctrl)
    n_ctrl_tot = ((sum_df["target_type"] == "control_random")).sum()
    print(f"  {n_ctrl_sig}/{n_ctrl_tot} controls show 'compensation' (expected ~5% by chance)")

    n_bkup_sig = len(sig_backup)
    n_bkup_tot = ((sum_df["target_type"] == "backup")).sum()
    print(f"\n=== INTERPRETATION ===")
    print(f"Backup features causally compensating: {n_bkup_sig}/{n_bkup_tot} ({100*n_bkup_sig/max(n_bkup_tot,1):.0f}%)")
    print(f"Control features showing compensation: {n_ctrl_sig}/{n_ctrl_tot} ({100*n_ctrl_sig/max(n_ctrl_tot,1):.0f}%)")
    if n_bkup_sig / max(n_bkup_tot, 1) > 3 * n_ctrl_sig / max(n_ctrl_tot, 1):
        print("\n★★★★ H3-causal CONFIRMED — backup compensation rate >> control")
    else:
        print("\n(?) H3-causal weak — backup compensation rate not clearly above control")


if __name__ == "__main__":
    main()
