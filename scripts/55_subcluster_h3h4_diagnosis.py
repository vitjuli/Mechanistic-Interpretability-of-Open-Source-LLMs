"""
55_subcluster_h3h4_diagnosis.py — H3 (backup pathways) + H4 (early/late IIA)
diagnostics for the k=30 sub-cluster partition (agglo_coimp_subgroup_k30).

H3 — Reactive backup pathways (Wang et al. 2023):
  For each sub-cluster G:
    1. Forward pass (clean) on each prompt → collect activations of ALL 227 features
    2. Ablate G's features (zero in transcoder space) → re-forward, collect again
    3. Compute Δact[f] = post_act[f] − baseline_act[f] for all f ∉ G
    4. Identify backup features: Δact[f] > 2σ AND mean Δact > 0.5
  Output: per-sub-cluster table of top backup features.
  Testable prediction (cf. thesis_1_2 H3): sub-clusters in L25 (last analysed
  layer) should have ZERO backup features (no downstream to compensate).

H4 — Early vs late layer IIA:
  Combine all features from early sub-clusters (layer < L_split) and late
  sub-clusters (layer >= L_split), patch each set in IIA setup, measure IIA.
  Tests: does patching early-only features produce HIGHER IIA than late-only?

Inputs:
  data/analysis/runD_v2/clustering_full/cluster_labels.csv
    (column 'agglo_coimp_subgroup_k30' must be present)
  data/prompts/{behaviour}_{split}.jsonl

Outputs:
  {out_dir}/h3_backup_subcluster_raw.csv
  {out_dir}/h3_backup_subcluster_summary.csv
  {out_dir}/h4_early_late_results.csv

Usage:
  python3 scripts/55_subcluster_h3h4_diagnosis.py \\
    --mode both \\
    --cluster_col agglo_coimp_subgroup_k30 \\
    --clustering_dir data/analysis/runD_v2/clustering_full \\
    --grouping_dir data/analysis/runD_v2/grouping \\
    --out_dir data/analysis/runD_v2/h3h4_subcluster \\
    --max_prompts_h3 80 \\
    --max_pairs_h4 200
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


# ── Helpers (shared) ─────────────────────────────────────────────────────────

def get_mlp_input(hf_model, tokenizer, prompt: str, layer_idx: int, device: str):
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}
    block = hf_model.model.layers[layer_idx]

    def _hook(module, inp, out):
        captured["x"] = out[:, -1:, :].detach()

    handle = block.post_attention_layernorm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            hf_model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return captured["x"][0]


def parse_feature_id(fid: str) -> Tuple[int, int]:
    layer, idx = fid.lstrip("L").split("_F")
    return int(layer), int(idx)


# ─────────────────────────────────────────────────────────────────────────────
# MODE H3: Backup pathways analysis
# ─────────────────────────────────────────────────────────────────────────────

def collect_acts_under_ablation(hf_model, tokenizer, prompt: str,
                                 transcoder_set,
                                 ablate_layer: int, ablate_features: List[int],
                                 monitor_features: Dict[int, List[int]],
                                 device: str) -> Dict[str, float]:
    """Forward pass with G ablated at ablate_layer; capture monitored features."""
    # Build patched x_abl at ablate_layer
    x_clean = get_mlp_input(hf_model, tokenizer, prompt, ablate_layer, device)
    tc = transcoder_set[ablate_layer]
    x_t = x_clean.to(tc.dtype)
    with torch.no_grad():
        a = tc.encode(x_t)
        a_abl = a.clone()
        a_abl[:, ablate_features] = 0.0
        x_abl = tc.decode(a_abl).to(x_clean.dtype)

    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handles = []
    captured = {}

    # Patch hook at ablate_layer
    block_abl = hf_model.model.layers[ablate_layer]
    _x = x_abl.to(device)
    def _patch_hook(module, inp, out):
        mod = out.clone()
        mod[0, -1, :] = _x[0, :]
        return mod
    handles.append(block_abl.post_attention_layernorm.register_forward_hook(_patch_hook))

    # Capture hooks at all monitor layers
    for layer in monitor_features:
        block_mon = hf_model.model.layers[layer]
        def _make(layer=layer):
            def _hook(module, inp, out):
                captured[layer] = out[:, -1:, :].detach()
            return _hook
        handles.append(block_mon.post_attention_layernorm.register_forward_hook(_make()))

    try:
        with torch.no_grad():
            hf_model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out_acts = {}
    for layer, fids in monitor_features.items():
        if layer not in captured:
            continue
        x_l = captured[layer][0]
        tc_l = transcoder_set[layer]
        with torch.no_grad():
            a_l = tc_l.encode(x_l.to(tc_l.dtype))
        for fid in fids:
            out_acts[f"L{layer}_F{fid}"] = float(a_l[0, fid].item())
    return out_acts


def run_h3(args, sub_clusters: Dict[int, List[str]], prompts: List[Dict],
           hf_model, tokenizer, transcoder_set, all_layers: List[int],
           device: str) -> pd.DataFrame:
    """For each sub-cluster, ablate and measure Δact on other features."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Monitor all 227 features
    monitor_features = defaultdict(list)
    all_fids = []
    for cid, feats in sub_clusters.items():
        for fid in feats:
            layer, idx = parse_feature_id(fid)
            monitor_features[layer].append(idx)
            all_fids.append(fid)
    monitor_features = dict(monitor_features)
    log.info(f"H3 monitoring {len(all_fids)} features across {len(monitor_features)} layers")

    # ── Baseline (clean) activations ────────────────────────────────────────
    log.info("Computing baseline activations…")
    baseline = {}
    for pi, p in enumerate(prompts):
        if pi % 10 == 0:
            log.info(f"  baseline {pi}/{len(prompts)}")
        for layer, fids in monitor_features.items():
            x = get_mlp_input(hf_model, tokenizer, p["prompt"], layer, device)
            tc = transcoder_set[layer]
            with torch.no_grad():
                a = tc.encode(x.to(tc.dtype))
            for fid in fids:
                baseline[(pi, f"L{layer}_F{fid}")] = float(a[0, fid].item())

    # ── Per sub-cluster ablation ────────────────────────────────────────────
    rows = []
    for cid in sorted(sub_clusters.keys()):
        cluster_feats = sub_clusters[cid]
        # Group by layer (should be one layer per sub-cluster — single-layer)
        c_by_layer = defaultdict(list)
        for fid in cluster_feats:
            layer, idx = parse_feature_id(fid)
            c_by_layer[layer].append(idx)

        for abl_layer, abl_feats in c_by_layer.items():
            log.info(f"=== Ablating sub-cluster C{cid} at L{abl_layer} ({len(abl_feats)} features) ===")
            for pi, p in enumerate(prompts):
                if pi % 20 == 0:
                    log.info(f"  prompt {pi}/{len(prompts)}")
                try:
                    post_acts = collect_acts_under_ablation(
                        hf_model, tokenizer, p["prompt"],
                        transcoder_set, abl_layer, abl_feats,
                        monitor_features, device,
                    )
                except Exception as e:
                    log.debug(f"  failed pi={pi}: {e}")
                    continue
                for fid, post_act in post_acts.items():
                    if fid in cluster_feats:
                        continue  # Skip features in the ablated cluster itself
                    base = baseline.get((pi, fid), 0.0)
                    rows.append({
                        "sub_cluster": cid,
                        "ablate_layer": abl_layer,
                        "prompt_idx": pi,
                        "feature_id": fid,
                        "feature_layer": parse_feature_id(fid)[0],
                        "baseline_act": base,
                        "post_act": post_act,
                        "delta_act": post_act - base,
                    })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "h3_backup_subcluster_raw.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved raw → {csv_path} ({len(df)} rows)")

    # Per-sub-cluster summary: top backup features
    summary_rows = []
    for cid in sorted(sub_clusters.keys()):
        sub = df[df["sub_cluster"] == cid]
        if sub.empty:
            continue
        per_feat = sub.groupby("feature_id").agg(
            mean_delta=("delta_act", "mean"),
            std_delta=("delta_act", "std"),
            mean_baseline=("baseline_act", "mean"),
            n_prompts=("delta_act", "size"),
        ).reset_index()
        per_feat["feature_layer"] = per_feat["feature_id"].map(lambda f: parse_feature_id(f)[0])
        per_feat = per_feat.sort_values("mean_delta", ascending=False)

        # Significant backup: Δ > 2σ AND mean > 0.5
        sig = per_feat[(per_feat["mean_delta"] > 0.5) &
                       (per_feat["mean_delta"] > 2 * per_feat["std_delta"])]
        # Significant suppression: Δ < -0.5 AND |Δ| > 2σ
        sup = per_feat[(per_feat["mean_delta"] < -0.5) &
                       (per_feat["mean_delta"].abs() > 2 * per_feat["std_delta"])]

        ablated_layer = parse_feature_id(sub_clusters[cid][0])[0]
        top_backups = per_feat.head(5).to_dict(orient="records")
        top_suppress = per_feat.tail(5).to_dict(orient="records")

        summary_rows.append({
            "sub_cluster": cid,
            "ablated_layer": ablated_layer,
            "ablated_n_features": len(sub_clusters[cid]),
            "n_sig_backup": len(sig),
            "n_sig_suppress": len(sup),
            "max_mean_delta": float(per_feat["mean_delta"].max()),
            "min_mean_delta": float(per_feat["mean_delta"].min()),
            "top_backup_feats": ", ".join(
                f"{r['feature_id']}({r['mean_delta']:+.2f})"
                for r in top_backups[:3]),
            "top_suppress_feats": ", ".join(
                f"{r['feature_id']}({r['mean_delta']:+.2f})"
                for r in top_suppress[:3]),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_path = out_dir / "h3_backup_subcluster_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    log.info(f"Saved summary → {sum_path}")

    print("\n=== H3 BACKUP PATHWAYS — per sub-cluster ===")
    print(sum_df[["sub_cluster", "ablated_layer", "ablated_n_features",
                  "n_sig_backup", "n_sig_suppress", "max_mean_delta",
                  "top_backup_feats"]].to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODE H4: Early vs Late layer IIA
# ─────────────────────────────────────────────────────────────────────────────

def get_features_and_margin(hf_model, tokenizer, prompt: str,
                             correct: str, incorrect: str,
                             transcoder_set, layers: List[int],
                             device: str) -> Tuple[float, Dict[int, torch.Tensor]]:
    """Get baseline margin and feature activations at specified layers."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False)["input_ids"]
    correct_id = correct_ids[0] if correct_ids else 0
    incorrect_id = incorrect_ids[0] if incorrect_ids else 0

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


def get_patched_margin(hf_model, tokenizer, prompt: str,
                        correct: str, incorrect: str,
                        feats_self: Dict[int, torch.Tensor],
                        feats_other: Dict[int, torch.Tensor],
                        patch_by_layer: Dict[int, List[int]],
                        transcoder_set, device: str) -> float:
    """Patch features at specified layers (replace with feats_other values), measure margin."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False)["input_ids"]
    correct_id = correct_ids[0] if correct_ids else 0
    incorrect_id = incorrect_ids[0] if incorrect_ids else 0

    handles = []
    for layer, patch_feats in patch_by_layer.items():
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


def run_h4(args, sub_clusters: Dict[int, List[str]], prompts: List[Dict],
           hf_model, tokenizer, transcoder_set, device: str) -> pd.DataFrame:
    """Test H4-inverted: early-layer patches give higher IIA than late-layer."""
    out_dir = Path(args.out_dir)

    # Define feature sets for each L_split
    feature_sets = []
    for split_layer in [18, 20, 22]:
        early_feats, late_feats = [], []
        for cid, feats in sub_clusters.items():
            for fid in feats:
                layer, _ = parse_feature_id(fid)
                if layer < split_layer:
                    early_feats.append(fid)
                else:
                    late_feats.append(fid)
        feature_sets.append({
            "name": f"early_only_lt{split_layer}",
            "feature_ids": early_feats,
            "n_layers": len(set(parse_feature_id(f)[0] for f in early_feats)),
        })
        feature_sets.append({
            "name": f"late_only_ge{split_layer}",
            "feature_ids": late_feats,
            "n_layers": len(set(parse_feature_id(f)[0] for f in late_feats)),
        })

    # Also: each layer separately (per-layer IIA scan)
    by_layer = defaultdict(list)
    for cid, feats in sub_clusters.items():
        for fid in feats:
            layer, _ = parse_feature_id(fid)
            by_layer[layer].append(fid)
    for layer in sorted(by_layer.keys()):
        feature_sets.append({
            "name": f"single_layer_L{layer}",
            "feature_ids": by_layer[layer],
            "n_layers": 1,
        })

    # Set up pairs
    alpha_prompts = [p for p in prompts if p["correct_answer"].strip() == "alpha"]
    beta_prompts = [p for p in prompts if p["correct_answer"].strip() == "beta"]
    rng = np.random.default_rng(42)
    n_pairs = min(len(alpha_prompts), len(beta_prompts), args.max_pairs_h4)
    a_idx = rng.choice(len(alpha_prompts), size=n_pairs, replace=False)
    b_idx = rng.choice(len(beta_prompts), size=n_pairs, replace=False)
    pairs = [(alpha_prompts[i], beta_prompts[j]) for i, j in zip(a_idx, b_idx)]
    log.info(f"H4: {n_pairs} α/β pairs")

    rows = []
    for fset in feature_sets:
        name = fset["name"]
        fids = fset["feature_ids"]
        layer_feats = defaultdict(list)
        for fid in fids:
            layer, idx = parse_feature_id(fid)
            layer_feats[int(layer)].append(idx)
        layers = sorted(layer_feats.keys())
        if not layers:
            log.warning(f"  {name}: no features")
            continue

        log.info(f"=== H4 set: {name}  ({len(fids)} features, {len(layers)} layers) ===")
        flip_count = 0
        n_tested = 0
        for pi, (p_a, p_b) in enumerate(pairs):
            if pi % 50 == 0:
                log.info(f"  pair {pi}/{n_pairs}")
            try:
                correct_a = " " + p_a["correct_answer"]
                incorrect_a = " " + p_a["incorrect_answer"]
                m_a, feats_a = get_features_and_margin(
                    hf_model, tokenizer, p_a["prompt"],
                    correct_a, incorrect_a, transcoder_set, layers, device)
                correct_b = " " + p_b["correct_answer"]
                incorrect_b = " " + p_b["incorrect_answer"]
                m_b, feats_b = get_features_and_margin(
                    hf_model, tokenizer, p_b["prompt"],
                    correct_b, incorrect_b, transcoder_set, layers, device)

                # Direction 1: α-prompt with β features
                m_a_patched = get_patched_margin(
                    hf_model, tokenizer, p_a["prompt"],
                    correct_a, incorrect_a, feats_a, feats_b, layer_feats,
                    transcoder_set, device)
                if m_a != 0 and m_a_patched != 0:
                    flip_count += int(np.sign(m_a_patched) != np.sign(m_a))
                    n_tested += 1

                # Direction 2: β-prompt with α features
                m_b_patched = get_patched_margin(
                    hf_model, tokenizer, p_b["prompt"],
                    correct_b, incorrect_b, feats_b, feats_a, layer_feats,
                    transcoder_set, device)
                if m_b != 0 and m_b_patched != 0:
                    flip_count += int(np.sign(m_b_patched) != np.sign(m_b))
                    n_tested += 1
            except Exception as e:
                log.debug(f"  pair {pi} failed: {e}")
                continue

        iia = flip_count / n_tested if n_tested > 0 else 0
        rows.append({
            "feature_set": name,
            "n_features": len(fids),
            "n_layers": fset["n_layers"],
            "n_pairs": n_pairs,
            "n_tested": n_tested,
            "n_flips": flip_count,
            "iia": iia,
        })
        log.info(f"  → IIA = {iia:.4f} ({flip_count}/{n_tested})")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "h4_early_late_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")

    print("\n=== H4 EARLY vs LATE — IIA per layer set ===")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["h3_backup", "h4_early_late", "both"], default="both")
    ap.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--cluster_col", default="agglo_coimp_subgroup_k30")
    ap.add_argument("--clustering_dir", type=Path,
                    default=Path("data/analysis/runD_v2/clustering_full"))
    ap.add_argument("--grouping_dir", type=Path,
                    default=Path("data/analysis/runD_v2/grouping"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("data/analysis/runD_v2/h3h4_subcluster"))
    ap.add_argument("--max_prompts_h3", type=int, default=80)
    ap.add_argument("--max_pairs_h4", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sub-cluster assignments ────────────────────────────────────────
    cl = pd.read_csv(args.clustering_dir / "cluster_labels.csv")
    assert args.cluster_col in cl.columns, f"Missing column: {args.cluster_col}"
    cl[args.cluster_col] = cl[args.cluster_col].astype(int)
    sub_clusters: Dict[int, List[str]] = defaultdict(list)
    for _, row in cl.iterrows():
        sub_clusters[int(row[args.cluster_col])].append(row["feature_id"])
    log.info(f"Loaded {len(sub_clusters)} sub-clusters (column={args.cluster_col})")

    # ── Load prompts ────────────────────────────────────────────────────────
    prompt_path = Path(args.prompts_file) if args.prompts_file else \
                  ROOT / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts_all = [json.loads(l) for l in open(prompt_path)]
    log.info(f"Total prompts: {len(prompts_all)}")

    prompts_h3 = prompts_all[:args.max_prompts_h3]
    log.info(f"Using {len(prompts_h3)} prompts for H3")

    # ── Layers needed ────────────────────────────────────────────────────────
    all_layers = sorted({parse_feature_id(f)[0] for fids in sub_clusters.values() for f in fids})
    log.info(f"Layers needed: {all_layers}")

    # ── Load model + transcoders ─────────────────────────────────────────────
    log.info("Loading model + transcoders...")
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

    # ── Run modes ────────────────────────────────────────────────────────────
    if args.mode in ("h3_backup", "both"):
        log.info("\n────────── H3: Backup pathway analysis ──────────")
        run_h3(args, sub_clusters, prompts_h3,
               model.model, model.tokenizer, tc_set, all_layers, args.device)

    if args.mode in ("h4_early_late", "both"):
        log.info("\n────────── H4: Early vs Late IIA ──────────")
        run_h4(args, sub_clusters, prompts_all,
               model.model, model.tokenizer, tc_set, args.device)

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
