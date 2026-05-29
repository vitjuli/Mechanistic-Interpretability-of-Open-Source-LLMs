#!/usr/bin/env python3
"""
Script 37: IIA (Interchange Intervention Accuracy) on matched alpha/beta pairs.

Condition II test: for each cluster G_i and each pair (p_alpha, p_beta)
with V_h(p_alpha) != V_h(p_beta), patch the feature activations of G_i in
p_alpha with activations from p_beta (and vice versa). If the sign of the
logit margin changes -> +1 flip.

IIA(G_i) = n_flips / (2 * n_pairs)

Pairs are formed by random matching (like script 52): any alpha-prompt paired
with any beta-prompt. Default: min(n_alpha, n_beta) pairs, seeded for
reproducibility. Run with the full corpus (train + aux, 470 or 538 prompts
depending on the run).

Requires GPU. Run on CSD3: sbatch jobs/run_iia_probe_clusters.sbatch

Outputs
-------
data/analysis/carrier_stability/
    iia_probe_clusters.csv    per-cluster IIA results
    iia_probe_clusters.json   summary
"""
import json
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROUPING    = ROOT / "data/results/grouping"
CLU         = ROOT / "data/results/clustering"
OUT         = ROOT / "data/analysis/carrier_stability"

CLUSTER_COL = "agglo_coimp_k12"
SEED        = 42


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(device: str) -> ModelWrapper:
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/experiment_config.yaml"))
    model_name = cfg.get("model_name", "Qwen/Qwen3-4B")
    log.info(f"Loading model: {model_name}")
    return ModelWrapper(model_name=model_name, device=device)


# ── transcoder hooks ──────────────────────────────────────────────────────────

@contextmanager
def collect_hooks(model_hf, layers: List[int], token_pos: int = -1):
    """Collect MLP inputs at given layers during a forward pass."""
    mlp_inputs: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            mlp_inputs[layer_idx] = x[:, token_pos, :].detach().float()
        return hook

    for layer_idx in layers:
        block = model_hf.model.layers[layer_idx]
        h = block.post_attention_layernorm.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    try:
        yield mlp_inputs
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def patch_hooks(model_hf, patched_mlp_inputs: Dict[int, torch.Tensor], token_pos: int = -1):
    """Replace MLP input at given layers with precomputed tensors."""
    hooks = []

    def make_hook(new_input):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, token_pos, :] = new_input.to(h.dtype).to(h.device)
                return (h,) + out[1:]
            else:
                h = out.clone()
                h[:, token_pos, :] = new_input.to(h.dtype).to(h.device)
                return h
        return hook

    for layer_idx, new_input in patched_mlp_inputs.items():
        block = model_hf.model.layers[layer_idx]
        h = block.post_attention_layernorm.register_forward_hook(make_hook(new_input))
        hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ── forward passes ────────────────────────────────────────────────────────────

def get_features_and_margin(
    model: ModelWrapper,
    transcoders: dict,
    text: str,
    correct_tok: str,
    incorrect_tok: str,
    layers: List[int],
    device: str,
) -> Tuple[float, Dict[int, torch.Tensor]]:
    """Run forward pass, return (logit_margin, {layer: feature_activations})."""
    inputs = {k: v.to(device) for k, v in model.tokenize([text]).items()}

    with collect_hooks(model.model, layers) as mlp_cache:
        with torch.no_grad():
            out = model.model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :]

    feat_cache: Dict[int, torch.Tensor] = {}
    for layer_idx, mlp_in in mlp_cache.items():
        with torch.no_grad():
            feat_cache[layer_idx] = transcoders[layer_idx].encode(mlp_in).float()

    log_probs = torch.log_softmax(logits.float(), dim=0)
    cid = model.tokenizer.encode(correct_tok, add_special_tokens=False)[0]
    iid = model.tokenizer.encode(incorrect_tok, add_special_tokens=False)[0]
    margin = (log_probs[cid] - log_probs[iid]).item()
    return margin, feat_cache


def get_patched_margin(
    model: ModelWrapper,
    transcoders: dict,
    text: str,
    correct_tok: str,
    incorrect_tok: str,
    target_feats: Dict[int, torch.Tensor],
    source_feats: Dict[int, torch.Tensor],
    cluster_feat_by_layer: Dict[int, List[int]],
    device: str,
) -> float:
    """Patch cluster features of target with values from source, return new margin."""
    patched_mlp_inputs: Dict[int, torch.Tensor] = {}
    for layer_idx, feat_indices in cluster_feat_by_layer.items():
        t_feat = target_feats[layer_idx].clone()
        s_feat = source_feats[layer_idx]
        idx_t = torch.tensor(feat_indices, device=t_feat.device, dtype=torch.long)
        t_feat[:, idx_t] = s_feat[:, idx_t]
        with torch.no_grad():
            patched_mlp_inputs[layer_idx] = transcoders[layer_idx].decode(t_feat).float()

    inputs = {k: v.to(device) for k, v in model.tokenize([text]).items()}
    with patch_hooks(model.model, patched_mlp_inputs):
        with torch.no_grad():
            out = model.model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :]

    log_probs = torch.log_softmax(logits.float(), dim=0)
    cid = model.tokenizer.encode(correct_tok, add_special_tokens=False)[0]
    iid = model.tokenizer.encode(incorrect_tok, add_special_tokens=False)[0]
    return (log_probs[cid] - log_probs[iid]).item()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs to use (default: min(n_alpha, n_beta))")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--k", type=int, default=12, help="agglomerative cluster count (used only if --cluster_col not set)")
    parser.add_argument("--cluster_col", type=str, default=None,
                        help="Column name in cluster_labels.csv to use (e.g. 'coimp_louvain'). "
                             "Defaults to agglo_coimp_k{k}.")
    parser.add_argument("--prompt_metadata", type=str,
                        default=str(GROUPING / "prompt_metadata.csv"),
                        help="Path to prompt_metadata.csv (use RunD path for 538-prompt corpus)")
    parser.add_argument("--clustering_dir", type=str, default=str(CLU),
                        help="Directory with cluster_labels.csv (use RunD path for probe_v2)")
    parser.add_argument("--out_dir", type=str, default=str(OUT),
                        help="Output directory for IIA results")
    args = parser.parse_args()

    clustering_dir = Path(args.clustering_dir)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_col = args.cluster_col if args.cluster_col else f"agglo_coimp_k{args.k}"

    device = args.device
    rng = np.random.default_rng(args.seed)
    log.info(f"Device: {device}  seed={args.seed}  k={args.k}")

    # Load cluster assignments
    cl = pd.read_csv(clustering_dir / "cluster_labels.csv")[["feature_id", cluster_col]]
    cl = cl.rename(columns={cluster_col: "cluster"})

    def parse_feat(fid):
        layer = int(fid.split("_F")[0][1:])
        feat  = int(fid.split("_F")[1])
        return layer, feat

    cl[["layer", "feat_idx"]] = pd.DataFrame(
        cl["feature_id"].map(parse_feat).tolist(), index=cl.index
    )
    clusters     = sorted(cl["cluster"].unique())
    all_layers   = sorted(cl["layer"].unique())

    cluster_feat_by_layer: Dict[int, Dict[int, List[int]]] = {}
    for cid in clusters:
        sub = cl[cl["cluster"] == cid]
        by_layer: Dict[int, List[int]] = {}
        for layer, grp in sub.groupby("layer"):
            by_layer[int(layer)] = grp["feat_idx"].tolist()
        cluster_feat_by_layer[cid] = by_layer

    # Load prompts and form matched pairs (random matching, like script 52)
    pm = pd.read_csv(args.prompt_metadata)
    alpha_rows = pm[pm["correct_answer"] == "alpha"].reset_index(drop=True)
    beta_rows  = pm[pm["correct_answer"] == "beta"].reset_index(drop=True)
    n_pairs = min(len(alpha_rows), len(beta_rows))
    if args.max_pairs:
        n_pairs = min(n_pairs, args.max_pairs)

    alpha_idx = rng.choice(len(alpha_rows), size=n_pairs, replace=False)
    beta_idx  = rng.choice(len(beta_rows),  size=n_pairs, replace=False)
    pairs = [(alpha_rows.iloc[i], beta_rows.iloc[j])
             for i, j in zip(alpha_idx, beta_idx)]

    log.info(f"Corpus: {len(pm)} prompts  |  alpha={len(alpha_rows)}  beta={len(beta_rows)}")
    log.info(f"Pairs: {n_pairs}  |  Clusters: {len(clusters)}")

    # Load model + transcoders
    model = load_model(device)
    model.model.eval()
    tc_set = load_transcoder_set(device=device)
    transcoders = {layer: tc_set.get_transcoder(layer) for layer in all_layers}

    # Precompute baseline margins + feature activations for all unique prompts
    log.info("Collecting baseline activations...")
    prompt_cache: Dict[int, dict] = {}
    unique_idxs = set()
    for pa, pb in pairs:
        unique_idxs.add(int(pa["prompt_idx"]))
        unique_idxs.add(int(pb["prompt_idx"]))

    pm_by_idx = pm.set_index("prompt_idx")
    for pidx in sorted(unique_idxs):
        row = pm_by_idx.loc[pidx]
        correct   = " " + row["correct_answer"]
        incorrect = " " + row["incorrect_answer"]
        margin, feats = get_features_and_margin(
            model, transcoders, row["prompt_text"], correct, incorrect, all_layers, device
        )
        prompt_cache[pidx] = {
            "text": row["prompt_text"], "correct": correct, "incorrect": incorrect,
            "margin": margin, "feats": feats,
        }
    log.info(f"Cached {len(prompt_cache)} prompts")

    # IIA per cluster
    rows = []
    for cid in clusters:
        by_layer = cluster_feat_by_layer[cid]
        n_feats  = sum(len(v) for v in by_layer.values())
        layers_str = ",".join(f"L{l}" for l in sorted(by_layer.keys()))

        flip_count = 0
        n_tested   = 0

        for pa_row, pb_row in pairs:
            pa_idx = int(pa_row["prompt_idx"])
            pb_idx = int(pb_row["prompt_idx"])
            if pa_idx not in prompt_cache or pb_idx not in prompt_cache:
                continue
            pa = prompt_cache[pa_idx]
            pb = prompt_cache[pb_idx]

            # Direction 1: patch alpha with beta features
            m_a = get_patched_margin(
                model, transcoders, pa["text"], pa["correct"], pa["incorrect"],
                pa["feats"], pb["feats"], by_layer, device,
            )
            if pa["margin"] != 0 and m_a != 0:
                flip_count += int(np.sign(m_a) != np.sign(pa["margin"]))
                n_tested += 1

            # Direction 2: patch beta with alpha features
            m_b = get_patched_margin(
                model, transcoders, pb["text"], pb["correct"], pb["incorrect"],
                pb["feats"], pa["feats"], by_layer, device,
            )
            if pb["margin"] != 0 and m_b != 0:
                flip_count += int(np.sign(m_b) != np.sign(pb["margin"]))
                n_tested += 1

        iia = round(flip_count / n_tested, 3) if n_tested > 0 else None
        log.info(f"C{cid:2d} ({layers_str:20s}) n_feat={n_feats:2d}: "
                 f"IIA={iia:.3f} ({flip_count}/{n_tested})")
        rows.append(dict(
            cluster=cid, layers=layers_str, n_features=n_feats,
            n_pairs=n_pairs, n_tested=n_tested, n_flips=flip_count, iia=iia,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "iia_probe_clusters.csv", index=False)

    print("\n── IIA per cluster ──────────────────────────────────────────────────")
    print(df[["cluster", "layers", "n_features", "n_pairs", "n_flips", "iia"]].to_string(index=False))
    print(f"\nMean IIA: {df['iia'].mean():.3f}")
    print(f"Max  IIA: {df['iia'].max():.3f}  (C{df.loc[df['iia'].idxmax(), 'cluster']})")

    # RFR for reference (L24 is highest = 17.1%)
    rfr_l24 = 0.171
    n_above_rfr = int((df["iia"] > rfr_l24).sum())

    summary = dict(
        n_prompts=len(pm),
        n_pairs=n_pairs,
        n_clusters=len(df),
        mean_iia=round(float(df["iia"].mean()), 3),
        max_iia=round(float(df["iia"].max()), 3),
        n_above_rfr_l24=n_above_rfr,
        rfr_l24=rfr_l24,
        seed=args.seed,
    )
    json.dump(summary, open(out_dir / "iia_probe_clusters.json", "w"), indent=2)
    log.info("Saved: iia_probe_clusters.csv, iia_probe_clusters.json")


if __name__ == "__main__":
    main()
