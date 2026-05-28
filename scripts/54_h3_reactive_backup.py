"""
54_h3_reactive_backup.py — H3-reactive (Wang 2022): does ablation of one cluster
cause systematic re-activation of OTHER features (backup pathway)?

Procedure (per prompt):
  1. Forward pass (clean) — collect MLP-input activations at every layer
  2. For each target cluster G_i:
     a. Ablate G_i (zero out a_k for k in G_i at G_i's layer via transcoder patching)
     b. Re-forward with G_i ablated, collect MLP-input activations again
     c. For every OTHER feature (l, k), compute Δ_act = a_k^ℓ(post-ablation) − a_k^ℓ(clean)
  3. Aggregate: which features systematically increase in activation after
     ablation of G_i? → those are reactive backup candidates for G_i.

Reads:
  - prompts JSONL
  - dashboard_probe/public/data/cluster_semantics.json
  - feature_ids.json from clustering (top 40 features used in analysis)

Outputs:
  data/analysis/iia_failure_diagnosis/h3_reactive_backup.csv
    (cluster_id, target_feature, prompt_id, layer, delta_act, baseline_act)
  data/analysis/iia_failure_diagnosis/h3_reactive_summary.json
    (per cluster: top backup features ranked by mean Δ_act)

Literature:
  Wang et al. 2023 (IOI, ICLR 2023) — backup heads emerge after ablation
  McGrath et al. 2023 — self-repair quantification
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_mlp_input(hf_model, tokenizer, prompt: str, layer_idx: int, device: str) -> torch.Tensor:
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


def collect_acts_under_ablation(
    hf_model, tokenizer, prompt: str,
    transcoder_set,
    ablate_layer: int, ablate_features: List[int],
    monitor_features: Dict[int, List[int]],
    device: str,
) -> Dict[str, float]:
    """
    Forward pass with G ablated at ablate_layer; collect activations of
    monitor_features (other layers) for backup detection.

    Returns {feature_id: activation_value}.
    """
    # Build patched x at ablate_layer: encode→zero G→decode
    x_clean = get_mlp_input(hf_model, tokenizer, prompt, ablate_layer, device)
    tc = transcoder_set[ablate_layer]
    x_t = x_clean.to(tc.dtype)
    with torch.no_grad():
        a = tc.encode(x_t)
        a_abl = a.clone()
        a_abl[:, ablate_features] = 0.0
        x_abl = tc.decode(a_abl).to(x_clean.dtype)

    # Patch ablate_layer + capture monitor layers in same forward
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handles = []
    captured = {}

    # patch hook at ablate_layer
    block_abl = hf_model.model.layers[ablate_layer]
    _x = x_abl.to(device)
    def _patch_hook(module, inp, out):
        mod = out.clone()
        mod[0, -1, :] = _x[0, :]
        return mod
    handles.append(block_abl.post_attention_layernorm.register_forward_hook(_patch_hook))

    # capture hooks at monitor layers
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

    # Encode captured activations through transcoder, read monitor features
    out_acts = {}
    for layer, fids in monitor_features.items():
        if layer not in captured:
            continue
        x_l = captured[layer][0]   # (1, d)
        tc_l = transcoder_set[layer]
        with torch.no_grad():
            a_l = tc_l.encode(x_l.to(tc_l.dtype))
        for fid in fids:
            out_acts[f"L{layer}_F{fid}"] = float(a_l[0, fid].item())
    return out_acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",   default="physics_decay_type_probe_v2")
    ap.add_argument("--split",       default="train")
    ap.add_argument("--prompts_file",default=None)
    ap.add_argument("--cluster_json",
                    default="dashboard_probe/public/data/cluster_semantics.json")
    ap.add_argument("--monitor_features_json",
                    default="data/results/clustering/feat_ids.json",
                    help="List of feature IDs to monitor for backup activation")
    ap.add_argument("--max_prompts", type=int, default=80)
    ap.add_argument("--target_clusters", nargs="+", type=int, default=None,
                    help="Cluster IDs to ablate. Default = all multi-feature clusters")
    ap.add_argument("--output_dir",  default="data/analysis/iia_failure_diagnosis")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ────────────────────────────────────────────────────────────────
    prompt_path = Path(args.prompts_file) if args.prompts_file else \
                  root / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts = [json.loads(l) for l in open(prompt_path)][:args.max_prompts]
    logger.info(f"Loaded {len(prompts)} prompts")

    cs = json.load(open(root / args.cluster_json))
    clusters = {c["id"]: c for c in cs["clusters"]}

    monitor_fids = json.load(open(root / args.monitor_features_json))
    monitor_features = defaultdict(list)
    for fid in monitor_fids:
        layer, idx = fid.lstrip("L").split("_F")
        monitor_features[int(layer)].append(int(idx))
    monitor_features = dict(monitor_features)
    logger.info(f"Monitoring {len(monitor_fids)} features across "
                f"{len(monitor_features)} layers")

    targets = args.target_clusters or sorted(clusters.keys())

    # ── load model + transcoders ───────────────────────────────────────────
    import yaml
    tc_cfg = yaml.safe_load(open(root / "configs/transcoder_config.yaml"))
    tc_info = tc_cfg["transcoders"][tc_cfg["model_size"]]

    all_layers = sorted(set(list(monitor_features.keys()) +
                            [f["layer"] for c in clusters.values() for f in c["features"]]))

    logger.info(f"Loading model + transcoders for {len(all_layers)} layers")
    wrapper = ModelWrapper(tc_info["model_name"], device=args.device)
    tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                                 layers=all_layers, device=args.device)

    # ── baseline (clean) activations ────────────────────────────────────────
    logger.info("Computing baseline activations…")
    baseline = {}   # (prompt_idx, feature_id) → activation
    for pi, p in enumerate(prompts):
        if pi % 10 == 0:
            logger.info(f"  baseline prompt {pi+1}/{len(prompts)}")
        for layer, fids in monitor_features.items():
            x = get_mlp_input(wrapper.model, wrapper.tokenizer, p["prompt"], layer, args.device)
            tc = tc_set[layer]
            with torch.no_grad():
                a = tc.encode(x.to(tc.dtype))
            for fid in fids:
                baseline[(pi, f"L{layer}_F{fid}")] = float(a[0, fid].item())

    # ── per-cluster ablation → measure Δ on every monitored feature ─────────
    rows = []
    for cid in targets:
        cluster = clusters[cid]
        cluster_features = cluster["features"]
        if len(cluster_features) == 0:
            continue
        # Group cluster features by layer
        c_layers = defaultdict(list)
        for f in cluster_features:
            c_layers[f["layer"]].append(int(f["id"].split("_F")[1]))

        # For each cluster layer, ablate cluster features there
        for abl_layer, abl_feats in c_layers.items():
            logger.info(f"\n=== Ablating C{cid} at L{abl_layer} ({len(abl_feats)} feats) ===")
            for pi, p in enumerate(prompts):
                if pi % 20 == 0:
                    logger.info(f"    prompt {pi+1}/{len(prompts)}")
                try:
                    post_acts = collect_acts_under_ablation(
                        wrapper.model, wrapper.tokenizer, p["prompt"],
                        tc_set, abl_layer, abl_feats,
                        monitor_features, args.device,
                    )
                except Exception as e:
                    logger.debug(f"  pi={pi}: {e}")
                    continue
                for fid, post_act in post_acts.items():
                    # Skip if the feature itself is in the ablated cluster (Δ = -baseline)
                    if fid in [f["id"] for f in cluster_features]:
                        continue
                    base = baseline.get((pi, fid), 0.0)
                    rows.append({
                        "cluster_id":   cid,
                        "ablate_layer": abl_layer,
                        "prompt_idx":   pi,
                        "prompt_id":    p.get("prompt_id", str(pi)),
                        "feature_id":   fid,
                        "baseline_act": base,
                        "post_act":     post_act,
                        "delta_act":    post_act - base,
                    })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "h3_reactive_backup.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved raw → {csv_path}  ({len(df)} rows)")

    # ── summarise: top backup features per cluster ──────────────────────────
    summary = []
    for cid in targets:
        sub = df[df["cluster_id"] == cid]
        if sub.empty:
            continue
        # Mean Δ_act per monitored feature
        per_feat = sub.groupby("feature_id").agg(
            n=("delta_act", "size"),
            mean_delta=("delta_act", "mean"),
            std_delta=("delta_act", "std"),
            mean_baseline=("baseline_act", "mean"),
        ).reset_index()
        per_feat["rel_delta"] = per_feat["mean_delta"] / per_feat["mean_baseline"].replace(0, 1e-6)
        per_feat = per_feat.sort_values("mean_delta", ascending=False)

        top_up = per_feat.head(10).to_dict(orient="records")
        top_down = per_feat.tail(5).to_dict(orient="records")

        summary.append({
            "cluster_id": cid,
            "name":       clusters[cid]["name"],
            "n_features_monitored": int(per_feat.shape[0]),
            "max_mean_delta":  float(per_feat["mean_delta"].max()),
            "n_features_with_significant_increase": int(
                ((per_feat["mean_delta"] > 0.5) &
                 (per_feat["mean_delta"] > 2 * per_feat["std_delta"])).sum()),
            "top_up":   top_up,
            "top_down": top_down,
        })

    with open(out_dir / "h3_reactive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary → {out_dir / 'h3_reactive_summary.json'}")

    # ── print table ─────────────────────────────────────────────────────────
    print("\n=== H3-REACTIVE BACKUP RESULTS ===")
    print(f"{'cid':>4}  {'max Δact':>9}  {'n_signif':>9}  {'top backup features':>30}")
    for s in summary:
        top_str = ", ".join(f"{f['feature_id']}({f['mean_delta']:+.2f})"
                            for f in s['top_up'][:3])
        print(f"  C{s['cluster_id']:2d}  {s['max_mean_delta']:>+9.3f}  "
              f"{s['n_features_with_significant_increase']:>9}  {top_str}")


if __name__ == "__main__":
    main()
