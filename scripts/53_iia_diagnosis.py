"""
53_iia_diagnosis.py — Unified IIA diagnostic for hypotheses H1, H2, H4.

Reuses the random-init control infrastructure (script 52) but patches
arbitrary feature sets instead of pre-defined clusters:

  --mode h1_circuit       : top-K features by causal edge weight (script 08 / graph)
  --mode h2_pairs         : joint patching of cluster PAIRS (detector+executor)
  --mode h4_layer_split   : patch only EARLY-half or LATE-half layers of cluster

The same IIA / Δ_shift / active_frac diagnostics are computed.

Outputs:
  data/analysis/iia_failure_diagnosis/{mode}_results.csv
  data/analysis/iia_failure_diagnosis/{mode}_summary.json

Literature:
  H1: Zhang & Nanda 2024 (patching best practices) — patching choice matters
  H2: Chen et al. 2026 (OASR, parallel circuits) — IoU = 4% between parallel paths
  H4: Anthropic 2025 — reading vs writing components
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
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── hooks (copied from script 52) ────────────────────────────────────────────

def get_mlp_input(hf_model, tokenizer, prompt: str, layer_idx: int,
                  device: str) -> torch.Tensor:
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


def compute_logit_diff(hf_model, tokenizer, prompt: str, correct: str,
                       incorrect: str, device: str) -> float:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = hf_model(**inputs, use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct, add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


def compute_logit_diff_with_patches(hf_model, tokenizer, prompt: str,
                                    correct: str, incorrect: str,
                                    patches: Dict[int, torch.Tensor],
                                    device: str) -> float:
    """Patch multiple layers simultaneously. patches: {layer: x_patched (1,d)}."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handles = []
    for layer, x in patches.items():
        block = hf_model.model.layers[layer]
        _x = x.to(device)
        def _make(_x=_x):
            def _hook(module, inp, out):
                mod = out.clone()
                mod[0, -1, :] = _x[0, :]
                return mod
            return _hook
        handles.append(block.post_attention_layernorm.register_forward_hook(_make()))
    try:
        with torch.no_grad():
            out = hf_model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    lp = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct, add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


# ── feature set definitions per mode ─────────────────────────────────────────

def get_feature_sets(args, root: Path) -> List[Dict]:
    """Returns list of {name, feature_ids, layer_feats} dicts."""
    sets = []

    cf = json.load(open(root / "data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json"))

    if args.mode == "h1_circuit":
        # v2: rank by mean_abs_effect (causal effect on logit margin),
        # not graph edge weights (v1→v2 lesson: attribution sign ≠ causal direction)
        feats_ranked = cf["top_by_causal_effect"]
        for K in [5, 10, 15, 20, 25, 30]:
            feats = [f["feature_id"] for f in feats_ranked[:K]]
            sets.append({
                "name": f"top_by_causal_effect_top{K}",
                "feature_ids": feats,
            })
        # Also: cumulative cluster-by-cluster (top-3 per cluster)
        cumulative = []
        for cid in [13, 8, 11, 12, 7, 9, 6]:  # strongest first by |orient_delta|
            top3 = [f["feature_id"] for f in cf["top_per_cluster"][str(cid)]["features"]]
            cumulative.extend(top3)
            sets.append({
                "name": f"cumul_through_C{cid}",
                "feature_ids": list(cumulative),
            })

    elif args.mode == "h2_pairs":
        # v2 cluster_semantics at k=16 (after recut)
        # Cluster IDs (k=16): C7=L18 (strongest α, orient_Δ=−0.896),
        #                     C4=L24 (strongest β, orient_Δ=+1.334),
        #                     C1=L23 (β, +0.962), C2=L21 (β, +0.709),
        #                     C0=L22 (β, +0.408), C6=L25 (β, +0.445),
        #                     C11=L13 (α, −0.609), C13=L10 (α, −0.483),
        #                     C15=L19 (α, −0.410), C3=L15 (α, −0.122)
        cs = json.load(open(root / "data/analysis/iia_failure_diagnosis/cluster_semantics_v2.json"))
        clusters = {c["id"]: c for c in cs["clusters"]}

        # KEY pairs: strongest opposite-polarity (α + β)
        priority_pairs = [
            (7, 4,  "α+β STRONGEST (L18+L24)"),     # SFR=0.96 + SFR=1.00
            (7, 1,  "α+β (L18+L23)"),               # both strong
            (7, 2,  "α+β (L18+L21)"),
            (7, 0,  "α+β (L18+L22)"),               # k=16 separates L22
            (11, 4, "α-early+β-late (L13+L24)"),
            (13, 4, "α-veryEarly+β-late (L10+L24)"),
            (15, 4, "α+β (L19+L24)"),
            (7, 6,  "α+β (L18+L25)"),
            # Triplets — both strong α + both strong β
            # Same-polarity controls (should NOT give IIA)
            (7, 15, "α+α control (L18+L19)"),
            (4, 1,  "β+β control (L24+L23)"),
            (4, 2,  "β+β control (L24+L21)"),
        ]
        for (ci, cj, tag) in priority_pairs:
            feats = [f["id"] for f in clusters[ci]["features"]] + \
                    [f["id"] for f in clusters[cj]["features"]]
            sets.append({
                "name": f"pair_C{ci}_C{cj}_{tag.replace(' ', '_')}",
                "feature_ids": feats,
                "meta": {"cluster_a": ci, "cluster_b": cj, "tag": tag},
            })
        # Triplet (top-3 opposite-polarity) — closes the question of how many clusters needed
        triplet = [f["id"] for f in clusters[7]["features"]] + \
                  [f["id"] for f in clusters[4]["features"]] + \
                  [f["id"] for f in clusters[1]["features"]]
        sets.append({
            "name": "triplet_C7_C4_C1_(L18+L24+L23)",
            "feature_ids": triplet,
            "meta": {"clusters": [7, 4, 1], "tag": "strongest α + top 2 β"},
        })
        # Cumulative ensemble: add clusters by |orient_delta| descending
        ranked_cids = sorted(clusters.keys(),
                             key=lambda c: -abs(clusters[c]["orient_delta"]))
        cumul = []
        for cid in ranked_cids[:8]:
            cumul.extend([f["id"] for f in clusters[cid]["features"]])
            sets.append({
                "name": f"ensemble_top{len(cumul)//5*5}_thru_C{cid}",
                "feature_ids": list(cumul),
                "meta": {"cumulative_clusters": [
                    int(c) for c in ranked_cids[:ranked_cids.index(cid)+1]]},
            })

    elif args.mode == "h4_layer_split":
        # Split top-30 by causal effect into early vs late
        feats = cf["top_by_causal_effect"][:30]
        for split_layer in [18, 20, 22]:
            early = [f["feature_id"] for f in feats if f["layer"] < split_layer]
            late  = [f["feature_id"] for f in feats if f["layer"] >= split_layer]
            sets.append({"name": f"early_only_lt{split_layer}", "feature_ids": early})
            sets.append({"name": f"late_only_ge{split_layer}",  "feature_ids": late})

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return sets


def parse_feature_ids(feature_ids: List[str]) -> Dict[int, List[int]]:
    """Parse 'L{layer}_F{idx}' strings into {layer: [idx, ...]}."""
    layer_feats = defaultdict(list)
    for fid in feature_ids:
        layer, idx = fid.lstrip("L").split("_F")
        layer_feats[int(layer)].append(int(idx))
    return dict(layer_feats)


# ── core experiment ──────────────────────────────────────────────────────────

def run_feature_set(
    name: str,
    feature_ids: List[str],
    trained_hf, tokenizer, transcoder_set,
    prompts: List[Dict],
    device: str,
    max_pairs: int,
    meta: dict = None,
) -> Tuple[Dict, pd.DataFrame]:
    layer_feats = parse_feature_ids(feature_ids)
    layers = sorted(layer_feats.keys())
    layers = [l for l in layers if l in transcoder_set._transcoders]
    if not layers:
        logger.warning(f"  {name}: no transcoder layers")
        return {}, pd.DataFrame()

    alpha_prompts = [p for p in prompts if p["correct_answer"].strip() == "alpha"]
    beta_prompts  = [p for p in prompts if p["correct_answer"].strip() == "beta"]
    rng = np.random.default_rng(seed=42)
    n_pairs = min(len(alpha_prompts), len(beta_prompts), max_pairs)
    a_idx = rng.choice(len(alpha_prompts), size=n_pairs, replace=False)
    b_idx = rng.choice(len(beta_prompts),  size=n_pairs, replace=False)
    pairs = [(alpha_prompts[i], beta_prompts[j]) for i, j in zip(a_idx, b_idx)]

    rows = []
    for pi, (p_a, p_b) in enumerate(pairs):
        # Collect MLP inputs at all needed layers
        x_a, x_b = {}, {}
        for layer in layers:
            x_a[layer] = get_mlp_input(trained_hf, tokenizer, p_a["prompt"], layer, device)
            x_b[layer] = get_mlp_input(trained_hf, tokenizer, p_b["prompt"], layer, device)

        # Build patches per layer: alpha's activation with G-features replaced from beta
        patches = {}
        active_fracs = []
        for layer in layers:
            tc = transcoder_set[layer]
            fi = layer_feats[layer]
            x_a_t = x_a[layer].to(tc.dtype)
            x_b_t = x_b[layer].to(tc.dtype)
            with torch.no_grad():
                a_alpha = tc.encode(x_a_t)
                a_beta  = tc.encode(x_b_t)
                a_patched = a_alpha.clone()
                a_patched[:, fi] = a_beta[:, fi]
                x_patched = tc.decode(a_patched).to(x_a[layer].dtype)
                active_fracs.append(((a_alpha[:, fi] > 0).float().mean()).item())
            patches[layer] = x_patched

        try:
            delta_orig = compute_logit_diff(
                trained_hf, tokenizer, p_a["prompt"],
                p_a["correct_answer"], p_a["incorrect_answer"], device)
            delta_patched = compute_logit_diff_with_patches(
                trained_hf, tokenizer, p_a["prompt"],
                p_a["correct_answer"], p_a["incorrect_answer"],
                patches, device)
        except Exception as e:
            logger.debug(f"  pair {pi}: {e}")
            continue

        iia = int(np.sign(delta_patched) != np.sign(delta_orig))
        rows.append({
            "set_name":         name,
            "pair_idx":         pi,
            "n_features":       len(feature_ids),
            "n_layers":         len(layers),
            "delta_orig":       delta_orig,
            "delta_patched":    delta_patched,
            "delta_shift":      abs(delta_patched - delta_orig),
            "iia":              iia,
            "mean_active_frac": float(np.mean(active_fracs)),
        })

        if pi % 20 == 0:
            logger.info(f"    {name}: pair {pi+1}/{len(pairs)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return {}, df

    summary = {
        "name":              name,
        "n_features":        len(feature_ids),
        "n_layers":          len(layers),
        "layers":            layers,
        "n_pairs":           len(df),
        "iia":               float(df["iia"].mean()),
        "mean_delta_shift":  float(df["delta_shift"].mean()),
        "median_delta_shift":float(df["delta_shift"].median()),
        "mean_active_frac":  float(df["mean_active_frac"].mean()),
    }
    if meta:
        summary["meta"] = meta
    return summary, df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        required=True,
                    choices=["h1_circuit", "h2_pairs", "h4_layer_split"])
    ap.add_argument("--behaviour",   default="physics_decay_type_probe_v2")
    ap.add_argument("--split",       default="train")
    ap.add_argument("--prompts_file",default=None)
    ap.add_argument("--max_pairs",   type=int, default=200)
    ap.add_argument("--output_dir",  default="data/analysis/iia_failure_diagnosis")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompts
    prompt_path = Path(args.prompts_file) if args.prompts_file else \
                  root / "data/prompts" / f"{args.behaviour}_{args.split}.jsonl"
    prompts = [json.loads(l) for l in open(prompt_path)]
    logger.info(f"Loaded {len(prompts)} prompts")

    # Feature sets per mode
    sets = get_feature_sets(args, root)
    logger.info(f"Mode {args.mode}: {len(sets)} feature sets to test")

    # All layers needed across all sets
    all_layers = sorted({
        int(fid.split("_F")[0].lstrip("L"))
        for s in sets for fid in s["feature_ids"]
    })

    # Model + transcoders
    import yaml
    tc_cfg = yaml.safe_load(open(root / "configs/transcoder_config.yaml"))
    tc_info = tc_cfg["transcoders"][tc_cfg["model_size"]]
    logger.info("Loading trained model + transcoders…")
    wrapper = ModelWrapper(tc_info["model_name"], device=args.device)
    tc_set  = load_transcoder_set(repo_id=tc_info["repo_id"],
                                  layers=all_layers, device=args.device)

    # Run each set
    summaries, dfs = [], []
    for s in sets:
        logger.info(f"\n=== {s['name']} ({len(s['feature_ids'])} feats) ===")
        summary, df = run_feature_set(
            name=s["name"],
            feature_ids=s["feature_ids"],
            trained_hf=wrapper.model,
            tokenizer=wrapper.tokenizer,
            transcoder_set=tc_set,
            prompts=prompts,
            device=args.device,
            max_pairs=args.max_pairs,
            meta=s.get("meta"),
        )
        if summary:
            summaries.append(summary)
            dfs.append(df)
            logger.info(f"    → IIA={summary['iia']:.3f}  mean|Δ_shift|={summary['mean_delta_shift']:.3f}")

    # Save
    csv_path = out_dir / f"{args.mode}_results.csv"
    json_path = out_dir / f"{args.mode}_summary.json"
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"\nSaved → {csv_path}\n        {json_path}")

    # Print summary table
    print(f"\n=== {args.mode.upper()} RESULTS ===")
    print(f"{'name':>45}  {'n_feat':>7}  {'IIA':>6}  {'|Δshift|':>9}  active%")
    for s in summaries:
        print(f"  {s['name'][:43]:>45}  {s['n_features']:>7}  "
              f"{s['iia']:.3f}  {s['mean_delta_shift']:>9.3f}  {s['mean_active_frac']*100:>5.1f}%")


if __name__ == "__main__":
    main()
