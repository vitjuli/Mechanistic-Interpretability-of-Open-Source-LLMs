"""
Compute intervention-based causal feature-to-feature edges.

Pipeline:
  Phase 0   Load attribution graph → VW edges as candidate set
  Phase 1   Compute AGW score (|vw_weight| × freq_i × freq_j) as prefilter
  Phase 2   Load model + transcoders
  Phase 3   Baseline forward passes → cache mlp_input + feature acts per prompt
  Phase 4   Ablation loop: for each source feature, ablate at L_i,
            collect downstream feature activations, compute delta
  Phase 5   Aggregate edges → causal_edges_{behaviour}_{split}.json
  Phase 6   Path tracing → circuits_{behaviour}_{split}.json
  Phase 7   Necessity validation: group-ablate all circuit features, measure disruption
  Phase 8   S1 Sufficiency: complement ablation (keep-only-circuit test)
  Phase 9   S2 Sufficiency: cross-prompt injection (EN→FR, optional)
  Phase 10  Presentation graph (Graph B): typed edge classes, display-only

AGW is a PREFILTER only. Actual edge weights come from
  delta_ij(p) = feat_j_baseline(p) - feat_j_ablated(p)
aggregated over all prompts.

Ablation method (exact): hook blocks[L].mlp OUTPUT (after-MLP).
  Subtract a_i * W_dec[i] from MLP output at token_pos.
  This removes exactly feature i's decoded contribution from the residual,
  with no Jacobian distortion.

Usage:
  python scripts/08_causal_edges.py --behaviour multilingual_circuits_b1 --split train
  python scripts/08_causal_edges.py --behaviour multilingual_circuits_b1 --split train \\
      --agw_top_frac 0.6 --tau_causal 0.05 --n_paths 20 --graph_json PATH
  python scripts/08_causal_edges.py ... --skip_sufficiency   # skip Phases 8+9
  python scripts/08_causal_edges.py ... --skip_s2            # skip Phase 9 only
  python scripts/08_causal_edges.py ... --top_n_io_edges 15  # more IO edges in Graph B
"""

import json
import yaml
import torch
import numpy as np
import math
import argparse
import sys
import logging
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import heapq

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HPC / reproducibility helpers
# ---------------------------------------------------------------------------

def _get_git_commit() -> Optional[str]:
    """Return current HEAD commit hash, or None if unavailable."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _check_writable(path: Path) -> None:
    """Raise RuntimeError if path is not writable (HPC NFS check)."""
    try:
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise RuntimeError(f"Output directory not writable: {path}  ({e})")


class _SourceTimeout(Exception):
    """Raised by SIGALRM handler when a single-source ablation exceeds the per-source wall."""


def _sigalrm_handler(signum, frame):
    raise _SourceTimeout("per-source timeout exceeded")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/experiment_config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_transcoder_config(path: str = "configs/transcoder_config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(prompt_dir: Path, behaviour: str, split: str) -> List[Dict]:
    fp = prompt_dir / f"{behaviour}_{split}.jsonl"
    if not fp.exists():
        raise FileNotFoundError(f"Prompts not found: {fp}")
    out = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Phase 0 – load graph
# ---------------------------------------------------------------------------

def load_graph(graph_json: Path) -> Dict:
    if not graph_json.exists():
        raise FileNotFoundError(f"Graph not found: {graph_json}")
    with open(graph_json) as f:
        return json.load(f)


def extract_graph_features(graph: Dict) -> List[Dict]:
    """Return list of feature node dicts."""
    return [n for n in graph["nodes"] if n.get("type") == "feature"]


def extract_vw_edges(graph: Dict) -> List[Dict]:
    """Return list of VW edge dicts (source, target, weight)."""
    return [e for e in graph["edges"] if e.get("edge_type") == "virtual_weight"]


def extract_star_edges(graph: Dict) -> List[Dict]:
    """Return attribution (star) edges: feature→output and input→feature."""
    return [e for e in graph["edges"] if e.get("edge_type") != "virtual_weight"]


# ---------------------------------------------------------------------------
# Phase 1 – AGW prefilter
# ---------------------------------------------------------------------------

def compute_agw_scores(
    vw_edges: List[Dict],
    node_freq: Dict[str, float],
) -> List[Tuple[float, Dict]]:
    """
    For each VW edge, compute AGW = |vw_weight| × freq_src × freq_tgt.
    Returns list of (agw_score, edge_dict) sorted descending.

    node_freq: {node_id: frequency (fraction of prompts active)}
    """
    scored = []
    for edge in vw_edges:
        src = edge["source"]
        tgt = edge["target"]
        w = abs(edge.get("weight", 0.0))
        f_src = node_freq.get(src, 0.0)
        f_tgt = node_freq.get(tgt, 0.0)
        agw = w * f_src * f_tgt
        scored.append((agw, edge))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def filter_candidate_pairs(
    scored_edges: List[Tuple[float, Dict]],
    top_frac: float,
) -> List[Tuple[float, Dict]]:
    """Keep top_frac fraction by AGW score."""
    n_keep = max(1, int(math.ceil(len(scored_edges) * top_frac)))
    kept = scored_edges[:n_keep]
    logger.info(
        f"AGW filter: {len(scored_edges)} VW edges → {n_keep} candidates "
        f"(top {top_frac:.0%})"
    )
    return kept


# ---------------------------------------------------------------------------
# Phase 2 – Model & transcoder loading  (see script 07 for pattern)
# ---------------------------------------------------------------------------

def load_model_and_transcoders(
    tc_config: Dict,
    model_size: str,
    layers: List[int],
    allow_sharded: bool = False,
) -> Tuple[ModelWrapper, TranscoderSet, torch.device]:
    model_name = tc_config["transcoders"][model_size]["model_name"]
    logger.info(f"Loading model: {model_name}")
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

    if hasattr(model.model, "hf_device_map"):
        unique_devices = set(model.model.hf_device_map.values())
        if len(unique_devices) > 1 and not allow_sharded:
            raise RuntimeError(
                f"Model sharded across {unique_devices}. Pass --allow_sharded to proceed."
            )

    logger.info(f"Loading transcoders for layers {layers}")
    tc_set = load_transcoder_set(
        model_size=model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=layers,
    )
    return model, tc_set, device


# ---------------------------------------------------------------------------
# Phase 3 – Baseline activation cache
# ---------------------------------------------------------------------------

def _get_blocks(model_hf: torch.nn.Module) -> List[torch.nn.Module]:
    try:
        return model_hf.model.layers
    except AttributeError:
        return model_hf.transformer.h


def _get_layernorm(blocks, layer_idx: int) -> torch.nn.Module:
    block = blocks[layer_idx]
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    if hasattr(block, "ln_2"):
        return block.ln_2
    raise RuntimeError(f"Cannot find post_attention_layernorm at layer {layer_idx}")


def collect_mlp_inputs_multi_layer(
    model_hf: torch.nn.Module,
    inputs: Dict,
    layers: List[int],
    token_pos: int = -1,
) -> Dict[int, torch.Tensor]:
    """
    Single forward pass that captures post_attention_layernorm outputs
    (= MLP inputs) at all requested layers.

    Returns {layer_idx: tensor(hidden_dim), ...}  (detached, on CPU)
    """
    captured: Dict[int, torch.Tensor] = {}
    blocks = _get_blocks(model_hf)
    handles = []

    def make_hook(lidx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[lidx] = h[:, token_pos, :].detach().float().cpu()
        return hook

    try:
        for lyr in layers:
            ln = _get_layernorm(blocks, lyr)
            handles.append(ln.register_forward_hook(make_hook(lyr)))
        with torch.no_grad():
            _ = model_hf(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return captured


def build_baseline_cache(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    graph_layers: List[int],
    token_pos: int = -1,
) -> Tuple[Dict, Dict]:
    """
    For each prompt, run one clean forward pass and store:
      mlp_inputs[p][layer]        = tensor(hidden_dim) on CPU
      feat_acts[p][layer][feat_i] = float  (from transcoder.encode)

    Returns (mlp_inputs, feat_acts)
    """
    mlp_inputs: Dict[int, Dict[int, torch.Tensor]] = {}
    feat_acts:  Dict[int, Dict[int, Dict[int, float]]] = {}

    for p_idx, prompt_data in enumerate(tqdm(prompts, desc="Baseline cache")):
        text = prompt_data["prompt"]
        inp = model.tokenize([text])
        inp = {k: v.to(device) for k, v in inp.items()}

        # One clean forward pass capturing all graph layers
        caps = collect_mlp_inputs_multi_layer(
            model.model, inp, graph_layers, token_pos=token_pos
        )

        mlp_inputs[p_idx] = caps
        feat_acts[p_idx] = {}

        for lyr, mlp_in in caps.items():
            tc = tc_set[lyr]
            with torch.no_grad():
                feats = tc.encode(mlp_in.to(device).to(tc.dtype))  # (1, d_tc)
            feat_acts[p_idx][lyr] = feats[0].float().cpu()  # tensor(d_tc)

    return mlp_inputs, feat_acts


# ---------------------------------------------------------------------------
# Phase 4 – Ablation loop
# ---------------------------------------------------------------------------

def compute_ablation_params(
    tc_set: TranscoderSet,
    device: torch.device,
    src_layer: int,
    src_feat_idx: int,
    baseline_feat_acts_at_src: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """
    Return (a_i, W_dec_i) needed to ablate feature src_feat_idx at src_layer.

    The exact ablation removes feature i's contribution from the MLP OUTPUT:
        mlp_output_ablated = mlp_output_baseline - a_i * W_dec[i, :]

    This is implemented by hooking blocks[src_layer].mlp (after-MLP hook),
    NOT post_attention_layernorm (before-MLP hook).  Hooking after the MLP
    removes exactly a_i * W_dec[i] from the residual stream, with no Jacobian
    distortion.  Hooking before-MLP would produce -a_i * J_mlp @ W_dec[i],
    which differs from the true decoder contribution.

    Args:
        baseline_feat_acts_at_src: shape (d_tc,), from tc.encode at decision token.
    Returns:
        a_i:     scalar float — baseline feature activation
        W_dec_i: tensor (d_model,) on device — decoder direction for feature i
    """
    tc = tc_set[src_layer]
    a_i     = float(baseline_feat_acts_at_src[src_feat_idx])
    W_dec_i = tc.W_dec[src_feat_idx, :].to(device).to(tc.dtype)  # (d_model,)
    return a_i, W_dec_i


def run_ablated_pass_with_collection(
    model_hf: torch.nn.Module,
    inputs: Dict,
    src_layer: int,
    a_i: float,
    W_dec_i: torch.Tensor,
    collect_layers: List[int],
    token_pos: int = -1,
) -> Dict[int, torch.Tensor]:
    """
    Single forward pass: ablate feature i at src_layer via MLP OUTPUT hook,
    then collect post_attention_layernorm outputs at all collect_layers > src_layer.

    Ablation method (exact):
        Hook blocks[src_layer].mlp (after-MLP).
        Subtract a_i * W_dec_i from the MLP output at token_pos.
        This removes exactly feature i's decoded contribution from the residual,
        with no Jacobian distortion from the model MLP.

    Collection:
        Hook post_attention_layernorm at each downstream layer.
        Capture only token_pos — no pooling across positions.

    Returns {layer_idx: tensor(1, d_model) on CPU}
    """
    collected: Dict[int, torch.Tensor] = {}
    blocks = _get_blocks(model_hf)
    handles = []

    # --- Exact ablation: hook MLP OUTPUT at src_layer ---
    # Subtract a_i * W_dec_i from the MLP output at the decision token position.
    # Effect on residual: exactly -a_i * W_dec_i (no MLP Jacobian term).
    def make_mlp_out_hook(ai: float, w_dec: torch.Tensor, tpos: int):
        def hook(module, inp, out):
            h = out[0].clone() if isinstance(out, tuple) else out.clone()
            contrib = (ai * w_dec).to(h.dtype).to(h.device)  # (d_model,)
            h[:, tpos, :] = h[:, tpos, :] - contrib
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    # --- Collection: hook post_attention_layernorm at downstream layers ---
    # Capture strictly token_pos — never pool across positions.
    def make_col_hook(lidx: int, tpos: int):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Shape: h is (batch, seq_len, d_model); extract single token → (1, d_model)
            collected[lidx] = h[:, tpos, :].detach().float().cpu()
        return hook

    try:
        if hasattr(blocks[src_layer], "mlp"):
            handles.append(blocks[src_layer].mlp.register_forward_hook(
                make_mlp_out_hook(a_i, W_dec_i, token_pos)
            ))
        else:
            raise RuntimeError(
                f"Cannot find .mlp attribute on block {src_layer}. "
                f"Available: {[n for n, _ in blocks[src_layer].named_children()]}"
            )

        for lj in collect_layers:
            ln_j = _get_layernorm(blocks, lj)
            handles.append(ln_j.register_forward_hook(make_col_hook(lj, token_pos)))

        with torch.no_grad():
            _ = model_hf(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return collected


def run_ablation_for_source(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    src_layer: int,
    src_feat_idx: int,
    target_pairs: List[Tuple[int, int]],   # [(tgt_layer, tgt_feat_idx), ...]
    baseline_mlp_inputs: Dict[int, Dict[int, torch.Tensor]],
    baseline_feat_acts: Dict[int, Dict[int, torch.Tensor]],
    token_pos: int = -1,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    For one source feature, run ablated forward passes over all prompts.
    Returns {(tgt_layer, tgt_feat_idx): delta_array (n_prompts,)}
    where delta = baseline_act - ablated_act  (positive = ablation reduced activation).
    """
    tgt_layers = sorted(set(lj for lj, _ in target_pairs))
    tgt_layers_downstream = [lj for lj in tgt_layers if lj > src_layer]

    if not tgt_layers_downstream:
        return {}

    # Group targets by layer
    tgts_by_layer: Dict[int, List[int]] = defaultdict(list)
    for lj, fj in target_pairs:
        if lj > src_layer:
            tgts_by_layer[lj].append(fj)

    n_prompts = len(prompts)
    # delta_records[pair] = list of per-prompt deltas
    delta_records: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for p_idx, prompt_data in enumerate(prompts):
        text = prompt_data["prompt"]
        inp = model.tokenize([text])
        inp = {k: v.to(device) for k, v in inp.items()}

        # Get cached feature activations at src_layer for this prompt.
        # Computed at token_pos=-1 (decision token) in build_baseline_cache.
        # These are the only values needed for the ablation: a_i and W_dec[i].
        # The baseline mlp_input tensor is no longer required (MLP output hook).
        bl_feats_src = baseline_feat_acts[p_idx].get(src_layer)
        if bl_feats_src is None:
            logger.warning(f"No baseline feat acts for prompt {p_idx} layer {src_layer}; skipping")
            for lj, fj in target_pairs:
                if lj > src_layer:
                    delta_records[(lj, fj)].append(0.0)
            continue

        # Compute ablation parameters: a_i (scalar) and W_dec[i] (d_model,)
        a_i, W_dec_i = compute_ablation_params(
            tc_set, device, src_layer, src_feat_idx, bl_feats_src
        )

        # Guard: skip if a_i is zero (nothing to ablate) or non-finite (NaN/Inf
        # would corrupt the residual and can cause flash-attention to hang on GPU).
        if not math.isfinite(a_i) or a_i == 0.0:
            for lj, fj in target_pairs:
                if lj > src_layer:
                    delta_records[(lj, fj)].append(0.0)
            continue

        # Run ablated forward pass:
        #   - hook blocks[src_layer].mlp output → subtract a_i * W_dec_i at token_pos
        #   - collect post_attention_layernorm outputs at all downstream layers
        collected = run_ablated_pass_with_collection(
            model.model, inp, src_layer, a_i, W_dec_i, tgt_layers_downstream, token_pos
        )

        # Compute feature activations at each downstream layer and delta
        for lj in tgt_layers_downstream:
            if lj not in collected:
                for fj in tgts_by_layer[lj]:
                    delta_records[(lj, fj)].append(0.0)
                continue

            tc_j = tc_set[lj]
            with torch.no_grad():
                abl_feats_j = tc_j.encode(
                    collected[lj].to(device).to(tc_j.dtype)
                )  # (1, d_tc)
            abl_feats_j_cpu = abl_feats_j[0].float().cpu()

            # Baseline feat acts at lj for this prompt
            bl_feats_j = baseline_feat_acts[p_idx].get(lj)

            for fj in tgts_by_layer[lj]:
                bl_act = float(bl_feats_j[fj]) if bl_feats_j is not None else 0.0
                abl_act = float(abl_feats_j_cpu[fj])
                delta_records[(lj, fj)].append(bl_act - abl_act)

    return {pair: np.array(deltas) for pair, deltas in delta_records.items()}


# ---------------------------------------------------------------------------
# Phase 5 – Aggregate edges
# ---------------------------------------------------------------------------

def aggregate_edges(
    candidate_pairs: List[Tuple[float, Dict]],
    delta_results: Dict[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray],
    tau_causal: float,
    n_prompts: int,
) -> List[Dict]:
    """
    For each candidate pair, compute mean_delta and diagnostics.
    Keep pairs with mean_delta_abs >= tau_causal.

    Primary edge weight: mean_delta_abs (causal magnitude averaged over prompts).
    Diagnostics only:    std_delta, effect_size (mean/std = SNR).

    delta_results key: ((src_layer, src_feat_idx), (tgt_layer, tgt_feat_idx))
    """
    edges = []
    for agw_score, edge in candidate_pairs:
        src_id = edge["source"]   # e.g. "L10_F12345"
        tgt_id = edge["target"]
        src_parts = src_id.split("_")
        tgt_parts = tgt_id.split("_")
        src_layer = int(src_parts[0][1:])
        src_feat  = int(src_parts[1][1:])
        tgt_layer = int(tgt_parts[0][1:])
        tgt_feat  = int(tgt_parts[1][1:])

        if tgt_layer <= src_layer:
            continue  # strictly downstream; same-layer edges skipped

        key = ((src_layer, src_feat), (tgt_layer, tgt_feat))
        deltas = delta_results.get(key)
        if deltas is None or len(deltas) == 0:
            continue

        mean_d     = float(np.mean(deltas))
        mean_d_abs = abs(mean_d)
        std_d      = float(np.std(deltas) + 1e-8)
        effect     = mean_d / std_d   # diagnostic SNR only; NOT used for filtering

        # Primary gate: mean causal magnitude must exceed tau_causal
        if mean_d_abs < tau_causal:
            continue

        edges.append({
            "source":          src_id,
            "target":          tgt_id,
            "src_layer":       src_layer,
            "src_feat_idx":    src_feat,
            "tgt_layer":       tgt_layer,
            "tgt_feat_idx":    tgt_feat,
            "mean_delta":      round(mean_d, 6),
            "mean_delta_abs":  round(mean_d_abs, 6),   # primary causal weight
            "std_delta":       round(std_d, 6),         # diagnostic
            "effect_size":     round(effect, 4),        # diagnostic SNR (mean/std)
            "n_prompts":       int(len(deltas)),
            "agw_score":       round(agw_score, 6),
            "vw_weight":       round(edge.get("weight", 0.0), 6),
        })

    # Sort by primary causal weight, largest first
    edges.sort(key=lambda e: e["mean_delta_abs"], reverse=True)
    return edges


# ---------------------------------------------------------------------------
# Phase 6 – Path tracing
# ---------------------------------------------------------------------------

def build_causal_dag(
    causal_edges: List[Dict],
    graph: Dict,
    include_star_edges: bool = True,
) -> "nx.DiGraph":  # type: ignore
    import networkx as nx

    G = nx.DiGraph()

    # Add all feature nodes with attributes
    for node in graph["nodes"]:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

    # Add IO nodes if present
    for node in graph["nodes"]:
        if node.get("type") in ("input", "output"):
            G.add_node(node["id"], **node)

    # Add causal edges
    for edge in causal_edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            edge_type="causal",
            **{k: v for k, v in edge.items() if k not in ("source", "target")},
        )

    if include_star_edges:
        for edge in graph["edges"]:
            if edge.get("edge_type") == "virtual_weight":
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src and tgt:
                # Use the actual attribution weight magnitude for path scoring.
                # placeholder=1.0 caused all features to tie at neg_log≈-1e-10,
                # breaking ties alphabetically and filling n_paths with the first
                # K alphabetical features before any causal-edge paths were explored.
                # Real attribution weights make path scoring meaningful: high-attribution
                # features are explored first, and 3-hop causal paths compete fairly
                # against 2-hop attribution-only paths.
                star_weight = max(abs(edge.get("weight", 1.0)), 1e-6)
                G.add_edge(
                    src, tgt,
                    edge_type="star",
                    weight=edge.get("weight", 0.0),
                    mean_delta_abs=star_weight,
                )

    return G


def find_top_k_paths(
    G: "nx.DiGraph",  # type: ignore
    source_nodes: List[str],
    sink_nodes: List[str],
    K: int,
    weight_key: str = "mean_delta_abs",
    min_weight: float = 1e-8,
    max_path_len: int = 25,
    require_causal_edge: bool = True,
) -> List[Dict]:
    """
    Priority-queue search for top-K paths from any source to any sink
    by product of edge weights.

    require_causal_edge (default True):
        Only record paths that contain at least one edge with edge_type="causal".
        2-hop star-only paths (input → feature → output) are explored for heap
        ordering but NOT counted toward K, so the circuit is guaranteed to contain
        at least one intervention-confirmed feature-to-feature connection.
        Set False to allow star-only paths (old behaviour).

    Works correctly because the graph is layer-ordered (no cycles).
    """
    found = []
    seen_paths = set()

    # heap: (neg_log_score, path_tuple, has_causal_edge)
    # has_causal_edge tracks whether any causal-typed edge has been traversed so far.
    heap = []
    for s in source_nodes:
        if s in G:
            heapq.heappush(heap, (0.0, (s,), False))

    while heap:
        neg_log, path, has_causal = heapq.heappop(heap)
        node = path[-1]

        if node in sink_nodes:
            if require_causal_edge and not has_causal:
                # Pure-star path: skip recording but do NOT add to seen_paths so a
                # different path through the same sink can still be recorded.
                continue
            path_key = path
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                score = math.exp(-neg_log) if neg_log < 700 else 0.0
                found.append({
                    "rank":    len(found) + 1,
                    "score":   round(score, 8),
                    "path":    list(path),
                    "n_edges": len(path) - 1,
                })
                if len(found) >= K:
                    break
            continue

        if len(path) >= max_path_len:
            continue

        for _, nbr, data in G.out_edges(node, data=True):
            w = data.get(weight_key, abs(data.get("weight", 0.0)))
            if w < min_weight:
                continue
            is_causal = data.get("edge_type") == "causal"
            new_score = neg_log - math.log(w + 1e-10)
            new_has_causal = has_causal or is_causal
            heapq.heappush(heap, (new_score, path + (nbr,), new_has_causal))

    return found


def extract_circuit(
    paths: List[Dict],
    G: "nx.DiGraph",  # type: ignore
) -> Dict:
    """
    Circuit = union of all nodes and causal edges from top-K paths.
    Returns circuit summary dict.
    """
    circuit_nodes = set()
    circuit_edges = []
    seen_edge_pairs = set()

    for path_info in paths:
        path = path_info["path"]
        for n in path:
            circuit_nodes.add(n)
        for i in range(len(path) - 1):
            pair = (path[i], path[i + 1])
            if pair not in seen_edge_pairs:
                seen_edge_pairs.add(pair)
                edata = G.edges.get(pair, {})
                circuit_edges.append({
                    "source":    pair[0],
                    "target":    pair[1],
                    "edge_type": edata.get("edge_type", "unknown"),
                    "mean_delta_abs": edata.get("mean_delta_abs",
                                                abs(edata.get("weight", 0.0))),
                })

    return {
        "feature_nodes": sorted(circuit_nodes),
        "edges": circuit_edges,
        "n_features": len(circuit_nodes),
        "n_paths": len(paths),
        "n_edges": len(circuit_edges),
    }


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _parse_circuit_by_layer(circuit_feature_nodes: List[str]) -> Dict[int, List[int]]:
    """
    Parse circuit feature node IDs → {layer: [feat_idx, ...]}.
    Skips IO sentinel nodes (those not starting with 'L').

    INVARIANT: IO hubs (input/output nodes) never enter the returned dict,
    because their IDs don't start with 'L'.  Callers must not assume otherwise.
    """
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for node_id in circuit_feature_nodes:
        if not node_id.startswith("L"):
            continue
        parts = node_id.split("_")
        try:
            lyr  = int(parts[0][1:])
            fidx = int(parts[1][1:])
            by_layer[lyr].append(fidx)
        except (IndexError, ValueError):
            continue
    return dict(by_layer)


# ---------------------------------------------------------------------------
# Phase 7 – Circuit-level validation
# ---------------------------------------------------------------------------

def _get_answer_tokens(prompt_data: Dict) -> Tuple[str, str]:
    for c, i in [("correct_answer", "incorrect_answer"),
                 ("answer_matching", "answer_not_matching"),
                 ("correct", "incorrect")]:
        if c in prompt_data and i in prompt_data:
            return prompt_data[c], prompt_data[i]
    raise KeyError(f"No answer fields in prompt keys: {list(prompt_data.keys())}")


def validate_circuit(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    circuit_feature_nodes: List[str],
    token_pos: int = -1,
) -> Dict:
    """
    Circuit-level validation: group-ablate ALL circuit features simultaneously
    and measure the causal effect on the output logit difference.

    For each circuit feature F_i at layer L_i:
        output_ablated += hooks that subtract a_i * W_dec[i] from blocks[L_i].mlp output

    All circuit features across all their layers are ablated in a SINGLE forward pass.
    This measures the joint causal necessity of the circuit, not individual feature effects.

    Returns:
        disruption_rate:  fraction of prompts where ablation reduces the correct margin
        mean_effect:      mean(ablated_margin - baseline_margin) over prompts
        std_effect:       standard deviation
        n:                number of prompts evaluated
    """
    circuit_by_layer = _parse_circuit_by_layer(circuit_feature_nodes)

    if not circuit_by_layer:
        logger.warning("validate_circuit: no feature nodes found in circuit")
        return {"disruption_rate": None, "mean_effect": None, "std_effect": None, "n": 0}

    circuit_layers = sorted(circuit_by_layer.keys())
    blocks = _get_blocks(model.model)
    effects: List[float] = []

    for p_idx, prompt_data in enumerate(tqdm(prompts, desc="Circuit validation")):
        try:
            correct, incorrect = _get_answer_tokens(prompt_data)
        except KeyError as e:
            logger.debug(f"Skipping prompt {p_idx}: {e}")
            continue

        text = prompt_data["prompt"]
        inp  = model.tokenize([text])
        inp  = {k: v.to(device) for k, v in inp.items()}

        # --- Baseline logit diff ---
        with torch.no_grad():
            out = model.model(**inp, use_cache=False)
        log_probs  = torch.log_softmax(out.logits[0, -1, :], dim=0)
        cid = model.tokenizer.encode(correct,   add_special_tokens=False)
        iid = model.tokenizer.encode(incorrect, add_special_tokens=False)
        if len(cid) != 1 or len(iid) != 1:
            logger.debug(f"Skipping prompt {p_idx}: multi-token answer")
            continue
        baseline_margin = (log_probs[cid[0]] - log_probs[iid[0]]).item()

        # --- Pre-compute per-layer contributions from baseline activations ---
        # contribution[lyr] = Σ_i a_i * W_dec[i]  for all circuit features at lyr
        # Requires one baseline collection pass (no modification).
        caps = collect_mlp_inputs_multi_layer(model.model, inp, circuit_layers, token_pos)
        contributions: Dict[int, torch.Tensor] = {}
        for lyr in circuit_layers:
            tc = tc_set[lyr]
            mlp_in_lyr = caps.get(lyr)
            if mlp_in_lyr is None:
                continue
            with torch.no_grad():
                feats = tc.encode(mlp_in_lyr.to(device).to(tc.dtype))  # (1, d_tc)
            contrib = torch.zeros(tc.W_dec.shape[1], device=device, dtype=tc.dtype)
            for fi in circuit_by_layer[lyr]:
                a_i = feats[0, fi].item()
                if abs(a_i) > 1e-12:
                    contrib = contrib + a_i * tc.W_dec[fi, :].to(device).to(tc.dtype)
            contributions[lyr] = contrib  # (d_model,)

        if not contributions:
            continue

        # --- Group-ablation pass: hook all circuit layers simultaneously ---
        # Each hook subtracts the precomputed contribution at token_pos from the MLP output.
        handles = []

        def _make_hook(contrib: torch.Tensor, tpos: int):
            def hook(module, inp_m, out):
                h = out[0].clone() if isinstance(out, tuple) else out.clone()
                c = contrib.to(h.dtype).to(h.device)
                h[:, tpos, :] = h[:, tpos, :] - c
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook

        try:
            for lyr, contrib in contributions.items():
                if hasattr(blocks[lyr], "mlp"):
                    handles.append(
                        blocks[lyr].mlp.register_forward_hook(
                            _make_hook(contrib, token_pos)
                        )
                    )
            with torch.no_grad():
                out_abl = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        log_probs_abl  = torch.log_softmax(out_abl.logits[0, -1, :], dim=0)
        ablated_margin = (log_probs_abl[cid[0]] - log_probs_abl[iid[0]]).item()
        effects.append(ablated_margin - baseline_margin)

    if not effects:
        return {"disruption_rate": None, "mean_effect": None, "std_effect": None, "n": 0}

    arr = np.array(effects)
    return {
        "disruption_rate": round(float(np.mean(arr < 0)), 4),   # fraction margins decreased
        "mean_effect":     round(float(np.mean(arr)), 4),
        "std_effect":      round(float(np.std(arr)), 4),
        "n":               len(effects),
    }


# ---------------------------------------------------------------------------
# Phase 8 – Circuit sufficiency S1 (complement ablation)
# ---------------------------------------------------------------------------

def validate_circuit_sufficiency_s1(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    circuit_feature_nodes: List[str],
    token_pos: int = -1,
) -> Dict:
    """
    S1 Sufficiency: complement ablation.

    Remove all non-circuit feature contributions from every MLP output.
    If the circuit alone explains the behaviour, the logit direction should
    be preserved after this removal.

    Method per prompt per layer L:
      1. Encode MLP input  → feats  (1, d_tc)
      2. Build feats_circ  → copy of feats with only circuit feature indices non-zero
      3. nc_contrib = tc.decode(feats) - tc.decode(feats_circ)
         bias b_dec cancels exactly: decode is linear in feats, so
         W_dec @ feats + b_dec  −  W_dec @ feats_circ − b_dec
                                =  W_dec @ (feats − feats_circ)   (no bias term)
         Approximation: transcoder reconstruction error leaks through.
      4. Hook blocks[L].mlp output: subtract nc_contrib at token_pos.
      5. Measure logit diff of the resulting (circuit-only) forward pass.

    Metrics:
      sign_preserved_rate:  fraction of prompts where sign(correct − incorrect) preserved
      mean_retention_ratio: mean(s1_margin / baseline_margin) for |baseline_margin| > 1e-6
      verdict: STRONG  (sign_rate ≥ 0.80 AND retention ≥ 0.70)
               PARTIAL (exactly one criterion met)
               WEAK    (neither met)
    """
    circuit_by_layer = _parse_circuit_by_layer(circuit_feature_nodes)
    if not circuit_by_layer:
        logger.warning("validate_circuit_sufficiency_s1: no feature nodes found")
        return {
            "sign_preserved_rate": None, "mean_retention_ratio": None,
            "verdict": "N/A", "n": 0,
        }

    circuit_layers = sorted(circuit_by_layer.keys())
    blocks = _get_blocks(model.model)
    sign_preserved_count = 0
    retention_ratios: List[float] = []
    n_evaluated = 0

    for p_idx, prompt_data in enumerate(tqdm(prompts, desc="S1 Sufficiency")):
        try:
            correct, incorrect = _get_answer_tokens(prompt_data)
        except KeyError as e:
            logger.debug(f"S1: skipping prompt {p_idx}: {e}")
            continue

        text = prompt_data["prompt"]
        inp  = model.tokenize([text])
        inp  = {k: v.to(device) for k, v in inp.items()}

        # Baseline logit diff
        with torch.no_grad():
            out = model.model(**inp, use_cache=False)
        log_probs = torch.log_softmax(out.logits[0, -1, :], dim=0)
        cid = model.tokenizer.encode(correct,   add_special_tokens=False)
        iid = model.tokenizer.encode(incorrect, add_special_tokens=False)
        if len(cid) != 1 or len(iid) != 1:
            logger.debug(f"S1: skipping prompt {p_idx}: multi-token answer")
            continue
        baseline_margin = (log_probs[cid[0]] - log_probs[iid[0]]).item()

        # Collect MLP inputs at circuit layers
        caps = collect_mlp_inputs_multi_layer(model.model, inp, circuit_layers, token_pos)

        # Per-layer: compute non-circuit contribution (bias cancels exactly)
        nc_contribs: Dict[int, torch.Tensor] = {}
        for lyr in circuit_layers:
            tc = tc_set[lyr]
            mlp_in = caps.get(lyr)
            if mlp_in is None:
                continue
            with torch.no_grad():
                feats = tc.encode(mlp_in.to(device).to(tc.dtype))   # (1, d_tc)
                feats_circ = torch.zeros_like(feats)
                for fi in circuit_by_layer[lyr]:
                    feats_circ[0, fi] = feats[0, fi]
                # nc = W_dec @ (feats - feats_circ)  (bias cancels)
                nc = tc.decode(feats) - tc.decode(feats_circ)        # (1, d_model)
                nc = nc.squeeze(0)                                    # (d_model,)

            nc_norm = nc.norm().item()
            if nc_norm < 1e-6:
                logger.debug(
                    f"S1: nc_contrib near-zero at L{lyr} prompt {p_idx} "
                    f"(norm={nc_norm:.2e}); circuit may dominate this layer"
                )
            nc_contribs[lyr] = nc

        if not nc_contribs:
            continue

        # Hook MLP output: subtract non-circuit contribution
        handles = []

        def _make_nc_hook(nc: torch.Tensor, tpos: int):
            def hook(module, inp_m, out):
                h = out[0].clone() if isinstance(out, tuple) else out.clone()
                c = nc.to(h.dtype).to(h.device)
                h[:, tpos, :] = h[:, tpos, :] - c
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook

        try:
            for lyr, nc in nc_contribs.items():
                if hasattr(blocks[lyr], "mlp"):
                    handles.append(
                        blocks[lyr].mlp.register_forward_hook(
                            _make_nc_hook(nc, token_pos)
                        )
                    )
            with torch.no_grad():
                out_s1 = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        log_probs_s1 = torch.log_softmax(out_s1.logits[0, -1, :], dim=0)
        s1_margin    = (log_probs_s1[cid[0]] - log_probs_s1[iid[0]]).item()

        if (baseline_margin > 0) == (s1_margin > 0):
            sign_preserved_count += 1
        if abs(baseline_margin) > 1e-6:
            retention_ratios.append(s1_margin / baseline_margin)

        n_evaluated += 1

    if n_evaluated == 0:
        return {
            "sign_preserved_rate": None, "mean_retention_ratio": None,
            "verdict": "N/A", "n": 0,
        }

    sign_rate      = sign_preserved_count / n_evaluated
    mean_retention = float(np.mean(retention_ratios)) if retention_ratios else None

    strong_sign   = sign_rate >= 0.80
    strong_retain = mean_retention is not None and mean_retention >= 0.70
    if strong_sign and strong_retain:
        verdict = "STRONG"
    elif strong_sign or strong_retain:
        verdict = "PARTIAL"
    else:
        verdict = "WEAK"

    return {
        "sign_preserved_rate":  round(sign_rate, 4),
        "mean_retention_ratio": round(mean_retention, 4) if mean_retention is not None else None,
        "verdict":              verdict,
        "n":                    n_evaluated,
    }


# ---------------------------------------------------------------------------
# Phase 8b – Circuit sufficiency S1.5 (layerwise constrained forward pass)
# ---------------------------------------------------------------------------

def validate_circuit_sufficiency_s1_5(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    circuit_feature_nodes: List[str],
    token_pos: int = -1,
    debug_sanity: bool = False,
) -> Dict:
    """
    S1.5 Sufficiency: layerwise constrained forward propagation.

    Runs all three modes per prompt for direct side-by-side comparison:
      baseline:    full unconstrained forward pass
      s1_linear:   subtract nc_contrib computed from baseline activations (S1, linear)
      s1_5:        replace MLP output at each circuit layer with tc.decode(feats_masked)
                   where feats_masked keeps ONLY circuit feature activations

    KEY DIFFERENCE from S1 (linear):
      S1 computes nc_contrib from the BASELINE residual stream.  If the constraint
      at layer L changes the residual, layer L+1's feature activations also change —
      but S1 never models this propagation.  It answers:
        "if non-circuit features had been absent at their BASELINE magnitudes,
         what would the logit be?"

      S1.5 answers:
        "can the model compute the correct answer using ONLY circuit features?"
      At each circuit layer L, the hook fires DURING the forward pass.  inp_m[0]
      at layer L is the post-attention-layernorm of the residual that has already
      been modified by hooks at layers 0..L-1.  Therefore tc.encode(inp_m[0])
      gives feature activations under the CONSTRAINED residual, not the baseline.
      The constraint propagates causally, layer by layer.

    Intervention point: blocks[L].mlp OUTPUT — consistent with S1 and Phase 7.
      - inp_m[0] in hook = post_attention_layernorm output = TC encode input
      - Replacing out at token_pos → new_residual = prev_residual + tc.decode(feats_masked)
      - Layer L+1 attention and MLP see new_residual → genuine forward propagation

    No-leakage guarantee (by construction):
      Each hook modifies ONLY token_pos of the MLP output.  The modified residual
      stream flows into all subsequent layers normally.  No prompt-to-prompt or
      layer-to-layer state persists across forward passes.

    Vectorised feature masking:
      Boolean masks are pre-built per layer before the prompt loop (d_tc = 163,840).
      The hook uses `feats_masked[0, ~mask] = 0` (single GPU operation).
      No Python loop over feature indices inside the hook.

    Compute cost: 3 forward passes per prompt (same as standalone S1):
      Pass 1 — baseline logits + collect mlp_inputs (single hooked pass)
      Pass 2 — S1 linear (hook MLP outputs, subtract nc_contribs)
      Pass 3 — S1.5 layerwise (hook MLP outputs, encode-mask-decode)

    Returns:
      sign_preserved_rate_s1:   fraction of prompts with correct sign (S1)
      sign_preserved_rate_s15:  fraction of prompts with correct sign (S1.5)
      mean_retention_s1:        mean(s1_margin / baseline_margin)   (S1)
      mean_retention_s15:       mean(s15_margin / baseline_margin)  (S1.5)
      verdict_s1:               STRONG / PARTIAL / WEAK  (S1)
      verdict_s15:              STRONG / PARTIAL / WEAK  (S1.5)
      retention_improvement:    mean_retention_s15 − mean_retention_s1
                                positive = S1.5 retains more of the baseline logit gap
      n:                        number of prompts evaluated

    Remaining approximation:
      Transcoder reconstruction error: true_mlp_output ≠ tc.decode(tc.encode(x)) exactly.
      In S1.5, tc.decode(feats_masked) replaces the MLP output — so reconstruction
      error from non-circuit features is already excluded, but reconstruction error
      from the circuit features themselves still leaks through.
    """
    circuit_by_layer = _parse_circuit_by_layer(circuit_feature_nodes)
    if not circuit_by_layer:
        logger.warning("validate_circuit_sufficiency_s1_5: no feature nodes found")
        return {
            "sign_preserved_rate_s1": None, "sign_preserved_rate_s15": None,
            "mean_retention_s1": None,      "mean_retention_s15": None,
            "verdict_s1": "N/A",            "verdict_s15": "N/A",
            "retention_improvement": None,  "n": 0,
        }

    circuit_layers = sorted(circuit_by_layer.keys())
    blocks = _get_blocks(model.model)

    # Pre-build Boolean circuit masks — one per layer, on device.
    # Shape (d_tc,): True at circuit feature indices, False elsewhere.
    # Used for vectorised masking inside hooks (avoids Python loop at d_tc=163,840).
    circuit_masks: Dict[int, torch.Tensor] = {}
    for lyr in circuit_layers:
        tc  = tc_set[lyr]
        d_tc = tc.W_dec.shape[0]
        mask = torch.zeros(d_tc, dtype=torch.bool, device=device)
        oob = []
        for fi in circuit_by_layer[lyr]:
            if fi < d_tc:
                mask[fi] = True
            else:
                oob.append(fi)
        if oob:
            logger.warning(
                f"S1.5 L{lyr}: {len(oob)} circuit feature indices out of bounds "
                f"(d_tc={d_tc}): {oob[:5]}{'...' if len(oob)>5 else ''}  — skipped in mask"
            )
        circuit_masks[lyr] = mask
        logger.debug(f"S1.5 mask L{lyr}: {mask.sum().item()} circuit features / {d_tc}")

    # ── Hook factories ──────────────────────────────────────────────────────

    def _make_collect_hook(lidx: int, caps: Dict[int, torch.Tensor], tpos: int):
        """Capture post_attention_layernorm output (= MLP input) at tpos."""
        def hook(module, inp_m, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[lidx] = h[:, tpos, :].detach().float().cpu()
        return hook

    def _make_s1_hook(nc: torch.Tensor, tpos: int):
        """S1 linear: subtract precomputed nc_contrib from MLP output at tpos."""
        def hook(module, inp_m, out):
            h = out[0].clone() if isinstance(out, tuple) else out.clone()
            h[:, tpos, :] -= nc.to(h.dtype).to(h.device)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    def _make_s15_hook(
        tc, mask: torch.Tensor, tpos: int, lyr: int, do_sanity: bool,
    ):
        """
        S1.5 layerwise: encode constrained inp, mask, decode, replace out at tpos.

        inp_m[0] is the MLP input (post_attention_layernorm output).
        At circuit layer L, this tensor already reflects the constrained residual
        from all previous circuit layers (their hooks replaced MLP outputs earlier
        in this same forward pass).  Encoding it gives feature activations under
        the CONSTRAINED computation — not the baseline.

        Masking is vectorised: feats_masked[0, ~mask] = 0 (single GPU op).
        """
        def hook(module, inp_m, out):
            # inp_m[0]: (batch, seq, d_model) — MLP input, constrained residual
            tc_dev  = tc.W_dec.device
            h_in    = inp_m[0][:, tpos, :].to(tc_dev).to(tc.dtype)  # (1, d_model)
            mask_d  = mask.to(tc_dev)

            with torch.no_grad():
                feats        = tc.encode(h_in)          # (1, d_tc)
                feats_masked = feats.clone()
                feats_masked[0, ~mask_d] = 0            # zero all non-circuit features

                if do_sanity:
                    # S1.5 sanity check 1: non-circuit features must be zero after masking
                    nc_nonzero = int((feats_masked[0, ~mask_d] != 0).sum())
                    if nc_nonzero > 0:
                        logger.warning(
                            f"S1.5 sanity FAIL L{lyr}: {nc_nonzero} non-circuit features "
                            f"remain nonzero after masking"
                        )
                    # S1.5 sanity check 2: circuit features must be preserved exactly
                    circ_err = (feats_masked[0, mask_d] - feats[0, mask_d]).abs().max().item()
                    if circ_err > 1e-6:
                        logger.warning(
                            f"S1.5 sanity FAIL L{lyr}: circuit feature drift={circ_err:.2e}"
                        )
                    # S1.5 sanity check 3: log how many circuit features are actually active
                    n_active = int((feats[0, mask_d] != 0).sum())
                    logger.debug(
                        f"S1.5 L{lyr}: {n_active}/{mask.sum().item()} circuit features active "
                        f"(in constrained residual, not baseline)"
                    )

                h_circuit = tc.decode(feats_masked)     # (1, d_model)

            # Replace MLP output at tpos only; all other positions unchanged.
            # (Sanity check 4: only tpos is modified — guaranteed by construction.)
            h_out = out[0].clone() if isinstance(out, tuple) else out.clone()
            h_out[:, tpos, :] = h_circuit.to(h_out.dtype).to(h_out.device)
            return (h_out,) + out[1:] if isinstance(out, tuple) else h_out
        return hook

    # ── Per-prompt loop ─────────────────────────────────────────────────────
    sign_s1: List[bool]  = []
    sign_s15: List[bool] = []
    ret_s1: List[float]  = []
    ret_s15: List[float] = []
    n_eval = 0
    _sanity_done = False  # run detailed sanity logging only on first valid prompt

    for p_idx, prompt_data in enumerate(tqdm(prompts, desc="S1.5 Sufficiency")):
        try:
            correct, incorrect = _get_answer_tokens(prompt_data)
        except KeyError as e:
            logger.debug(f"S1.5: skipping prompt {p_idx}: {e}")
            continue

        text = prompt_data["prompt"]
        inp  = model.tokenize([text])
        inp  = {k: v.to(device) for k, v in inp.items()}

        cid = model.tokenizer.encode(correct,   add_special_tokens=False)
        iid = model.tokenizer.encode(incorrect, add_special_tokens=False)
        if len(cid) != 1 or len(iid) != 1:
            logger.debug(f"S1.5: skipping prompt {p_idx}: multi-token answer")
            continue

        do_sanity = debug_sanity and not _sanity_done

        # ── Pass 1: baseline logits + collect mlp_inputs (single forward pass) ──
        baseline_caps: Dict[int, torch.Tensor] = {}
        handles = []
        try:
            for lyr in circuit_layers:
                ln = _get_layernorm(blocks, lyr)
                handles.append(ln.register_forward_hook(
                    _make_collect_hook(lyr, baseline_caps, token_pos)
                ))
            with torch.no_grad():
                out_bl = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        lp_bl          = torch.log_softmax(out_bl.logits[0, -1, :], dim=0)
        baseline_margin = (lp_bl[cid[0]] - lp_bl[iid[0]]).item()

        # Compute S1 nc_contribs from baseline mlp_inputs (vectorised masking)
        nc_contribs: Dict[int, torch.Tensor] = {}
        for lyr in circuit_layers:
            tc     = tc_set[lyr]
            mlp_in = baseline_caps.get(lyr)
            if mlp_in is None:
                continue
            mask_d = circuit_masks[lyr].to(device)
            with torch.no_grad():
                feats      = tc.encode(mlp_in.to(device).to(tc.dtype))  # (1, d_tc)
                feats_circ = feats.clone()
                feats_circ[0, ~mask_d] = 0                               # vectorised
                nc = tc.decode(feats) - tc.decode(feats_circ)            # (1, d_model)
                nc_contribs[lyr] = nc.squeeze(0)                         # (d_model,)

        # ── Pass 2: S1 linear ──────────────────────────────────────────────
        handles = []
        try:
            for lyr, nc in nc_contribs.items():
                if hasattr(blocks[lyr], "mlp"):
                    handles.append(blocks[lyr].mlp.register_forward_hook(
                        _make_s1_hook(nc, token_pos)
                    ))
            with torch.no_grad():
                out_s1 = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        lp_s1     = torch.log_softmax(out_s1.logits[0, -1, :], dim=0)
        s1_margin = (lp_s1[cid[0]] - lp_s1[iid[0]]).item()

        # ── Pass 3: S1.5 layerwise constrained ────────────────────────────
        # Each hook encodes the MLP input under the CONSTRAINED residual stream
        # (which has already been modified by all previous circuit layers' hooks).
        handles = []
        try:
            for lyr in circuit_layers:
                if hasattr(blocks[lyr], "mlp"):
                    handles.append(blocks[lyr].mlp.register_forward_hook(
                        _make_s15_hook(
                            tc_set[lyr], circuit_masks[lyr], token_pos, lyr, do_sanity,
                        )
                    ))
            with torch.no_grad():
                out_s15 = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        lp_s15     = torch.log_softmax(out_s15.logits[0, -1, :], dim=0)
        s15_margin = (lp_s15[cid[0]] - lp_s15[iid[0]]).item()

        if do_sanity:
            _sanity_done = True

        # Accumulate metrics
        sign_s1.append( (baseline_margin > 0) == (s1_margin  > 0))
        sign_s15.append((baseline_margin > 0) == (s15_margin > 0))
        if abs(baseline_margin) > 1e-6:
            ret_s1.append( s1_margin  / baseline_margin)
            ret_s15.append(s15_margin / baseline_margin)

        n_eval += 1

    if n_eval == 0:
        return {
            "sign_preserved_rate_s1": None, "sign_preserved_rate_s15": None,
            "mean_retention_s1": None,      "mean_retention_s15": None,
            "verdict_s1": "N/A",            "verdict_s15": "N/A",
            "retention_improvement": None,  "n": 0,
        }

    def _verdict(sign_rate: float, ret: Optional[float]) -> str:
        ss = sign_rate >= 0.80
        sr = ret is not None and ret >= 0.70
        if ss and sr: return "STRONG"
        if ss or sr:  return "PARTIAL"
        return "WEAK"

    spsr_s1  = float(np.mean(sign_s1))
    spsr_s15 = float(np.mean(sign_s15))
    mr_s1    = float(np.mean(ret_s1))  if ret_s1  else None
    mr_s15   = float(np.mean(ret_s15)) if ret_s15 else None
    ri       = (mr_s15 - mr_s1) if (mr_s15 is not None and mr_s1 is not None) else None

    return {
        "sign_preserved_rate_s1":  round(spsr_s1,  4),
        "sign_preserved_rate_s15": round(spsr_s15, 4),
        "mean_retention_s1":       round(mr_s1,  4) if mr_s1  is not None else None,
        "mean_retention_s15":      round(mr_s15, 4) if mr_s15 is not None else None,
        "verdict_s1":              _verdict(spsr_s1,  mr_s1),
        "verdict_s15":             _verdict(spsr_s15, mr_s15),
        "retention_improvement":   round(ri, 4) if ri is not None else None,
        "n":                       n_eval,
    }


# ---------------------------------------------------------------------------
# Phase 9 – Circuit sufficiency S2 (cross-prompt injection, optional)
# ---------------------------------------------------------------------------

def _get_prompt_language(prompt_data: Dict) -> Optional[str]:
    """Try multiple field names; return 'en', 'fr', or None."""
    for field in ("language", "lang", "prompt_language", "language_code"):
        val = prompt_data.get(field)
        if val is not None:
            v = str(val).lower().strip()
            if v in ("en", "english"):
                return "en"
            if v in ("fr", "french", "français"):
                return "fr"
    return None


def validate_circuit_sufficiency_s2(
    model: ModelWrapper,
    tc_set: TranscoderSet,
    device: torch.device,
    prompts: List[Dict],
    circuit_feature_nodes: List[str],
    baseline_feat_acts: Dict[int, Dict[int, torch.Tensor]],
    token_pos: int = -1,
) -> Dict:
    """
    S2 Sufficiency (optional): cross-prompt circuit injection.

    For each concept, pair an EN prompt with a FR prompt (matched by concept_index).
    Inject the circuit-feature activations from the source (EN) into the target (FR):
        delta_L = Σ (src_a_i - tgt_a_i) * W_dec[i]  for circuit features i at L
    Hook blocks[L].mlp output of the target pass: add delta_L at token_pos.
    Measure whether the target's logit diff shifts toward the source's expected answer.

    Approximation: assumes linear superposition; non-linear interactions not captured.
    This is a directional test, not a magnitude test.

    Returns:
      n_pairs:       number of EN-FR concept pairs tested
      transfer_rate: fraction of pairs where target margin shifts in source's direction
      mean_shift:    mean(injected_margin − baseline_target_margin)
    """
    circuit_by_layer = _parse_circuit_by_layer(circuit_feature_nodes)
    if not circuit_by_layer:
        logger.warning("validate_circuit_sufficiency_s2: no feature nodes found")
        return {"n_pairs": 0, "transfer_rate": None, "mean_shift": None}

    circuit_layers = sorted(circuit_by_layer.keys())
    blocks = _get_blocks(model.model)

    # Group prompts by concept_index and language
    concept_groups: Dict[int, Dict[str, List[Tuple[int, Dict]]]] = defaultdict(
        lambda: {"en": [], "fr": []}
    )
    n_no_lang = 0
    for p_idx, pd in enumerate(prompts):
        ci = pd.get("concept_index")
        if ci is None:
            continue
        lang = _get_prompt_language(pd)
        if lang is None:
            n_no_lang += 1
            continue
        concept_groups[ci][lang].append((p_idx, pd))

    if n_no_lang > 0:
        logger.warning(
            f"S2: {n_no_lang}/{len(prompts)} prompts have no language field; skipped"
        )

    # Build EN-FR pairs (one per concept, first available)
    pairs: List[Tuple[Tuple[int, Dict], Tuple[int, Dict]]] = []  # (src, tgt)
    for ci, groups in sorted(concept_groups.items()):
        en_list = groups["en"]
        fr_list = groups["fr"]
        if en_list and fr_list:
            pairs.append((en_list[0], fr_list[0]))   # EN → FR injection

    if not pairs:
        logger.warning(
            "S2: no EN-FR concept pairs found. "
            "Check that prompts have 'concept_index' and a language field."
        )
        return {"n_pairs": 0, "transfer_rate": None, "mean_shift": None}

    logger.info(f"S2: testing {len(pairs)} EN-FR concept pairs")
    shifts: List[float] = []

    for (src_idx, src_pd), (tgt_idx, tgt_pd) in tqdm(pairs, desc="S2 Injection"):
        try:
            tgt_correct, tgt_incorrect = _get_answer_tokens(tgt_pd)
        except KeyError:
            continue

        tgt_text = tgt_pd["prompt"]
        tgt_inp  = model.tokenize([tgt_text])
        tgt_inp  = {k: v.to(device) for k, v in tgt_inp.items()}

        # Baseline target logit diff
        with torch.no_grad():
            out_base = model.model(**tgt_inp, use_cache=False)
        lp_base = torch.log_softmax(out_base.logits[0, -1, :], dim=0)
        cid = model.tokenizer.encode(tgt_correct,   add_special_tokens=False)
        iid = model.tokenizer.encode(tgt_incorrect, add_special_tokens=False)
        if len(cid) != 1 or len(iid) != 1:
            continue
        baseline_tgt_margin = (lp_base[cid[0]] - lp_base[iid[0]]).item()

        # Per-layer injection delta:  delta_L = Σ (src_a_i - tgt_a_i) * W_dec[i]
        # Uses cached baseline feature activations (no extra forward passes needed)
        src_fa = baseline_feat_acts.get(src_idx, {})
        tgt_fa = baseline_feat_acts.get(tgt_idx, {})

        deltas: Dict[int, torch.Tensor] = {}
        for lyr in circuit_layers:
            tc    = tc_set[lyr]
            sa    = src_fa.get(lyr)
            ta    = tgt_fa.get(lyr)
            if sa is None or ta is None:
                continue
            delta = torch.zeros(tc.W_dec.shape[1], device=device, dtype=tc.dtype)
            for fi in circuit_by_layer[lyr]:
                diff = float(sa[fi]) - float(ta[fi])
                if abs(diff) > 1e-12:
                    delta = delta + diff * tc.W_dec[fi, :].to(device).to(tc.dtype)
            if delta.norm().item() > 1e-8:
                deltas[lyr] = delta

        if not deltas:
            continue

        # Hook: add delta to target MLP output
        handles = []

        def _make_inject_hook(d: torch.Tensor, tpos: int):
            def hook(module, inp_m, out):
                h = out[0].clone() if isinstance(out, tuple) else out.clone()
                dv = d.to(h.dtype).to(h.device)
                h[:, tpos, :] = h[:, tpos, :] + dv
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook

        try:
            for lyr, d in deltas.items():
                if hasattr(blocks[lyr], "mlp"):
                    handles.append(
                        blocks[lyr].mlp.register_forward_hook(
                            _make_inject_hook(d, token_pos)
                        )
                    )
            with torch.no_grad():
                out_inj = model.model(**tgt_inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        lp_inj     = torch.log_softmax(out_inj.logits[0, -1, :], dim=0)
        inj_margin = (lp_inj[cid[0]] - lp_inj[iid[0]]).item()
        shifts.append(inj_margin - baseline_tgt_margin)

    if not shifts:
        return {"n_pairs": len(pairs), "transfer_rate": None, "mean_shift": None}

    arr           = np.array(shifts)
    transfer_rate = float(np.mean(arr > 0))   # shifted toward correct answer
    mean_shift    = float(np.mean(arr))
    return {
        "n_pairs":       len(shifts),
        "transfer_rate": round(transfer_rate, 4),
        "mean_shift":    round(mean_shift, 4),
    }


# ---------------------------------------------------------------------------
# Phase 10 – Presentation graph (Graph B)
# ---------------------------------------------------------------------------

_VALID_GRAPH_B_EDGE_TYPES = frozenset({
    "input_to_feature",
    "causal_feature_to_feature",
    "feature_to_output_correct",
    "feature_to_output_incorrect",
})


def build_presentation_graph(
    circuit: Dict,
    causal_edges: List[Dict],
    graph: Dict,
    top_n_io_edges: int = 10,
) -> Dict:
    """
    Build Graph B: Anthropic-style star graph for visualisation.

    STRICTLY separate from Graph A (extraction graph):
      - Circuit membership derives from extract_circuit() path union.
      - This function is called AFTER extraction completes.
      - It does NOT modify or re-derive circuit membership.

    Nodes: circuit feature nodes + IO nodes referenced by displayed edges.

    Edge types (four only):
      input_to_feature:            input node  → circuit feature
                                   top_n_io_edges strongest by |weight| from graph
      causal_feature_to_feature:   circuit feature → circuit feature
                                   from causal_edges, both endpoints in circuit
      feature_to_output_correct:   circuit feature → correct output node
                                   top_n_io_edges strongest by |weight| from graph
      feature_to_output_incorrect: circuit feature → incorrect output node
                                   top_n_io_edges strongest by |weight| from graph

    Output node classification: by 'label' field containing
      correct/matching  → correct;  incorrect/not_matching → incorrect.
    Nodes with no matching label default to the smaller output set.
    """
    circuit_nodes_set = set(circuit["feature_nodes"])

    # Classify IO nodes from graph
    input_node_ids  = {n["id"] for n in graph["nodes"] if n.get("type") == "input"}
    output_nodes    = [n for n in graph["nodes"] if n.get("type") == "output"]

    def _is_correct_output(n: Dict) -> bool:
        lbl = (n.get("label") or n.get("id") or "").lower()
        return ("correct" in lbl and "incorrect" not in lbl) or "matching" in lbl

    correct_output_ids   = {n["id"] for n in output_nodes if _is_correct_output(n)}
    incorrect_output_ids = {n["id"] for n in output_nodes if not _is_correct_output(n)}

    # Sanity: warn if IO nodes appear in circuit_nodes_set
    io_in_circuit = circuit_nodes_set & (input_node_ids | {n["id"] for n in output_nodes})
    if io_in_circuit:
        logger.warning(
            f"build_presentation_graph: IO nodes in circuit_nodes_set: {io_in_circuit}. "
            "Included in Graph B display nodes but excluded from layer-parsed circuit."
        )

    star_edges = [e for e in graph["edges"] if e.get("edge_type") != "virtual_weight"]
    graph_b_edges: List[Dict] = []

    # 1. input_to_feature
    in_to_feat = [
        e for e in star_edges
        if e.get("source", "") in input_node_ids
        and e.get("target", "") in circuit_nodes_set
    ]
    in_to_feat.sort(key=lambda e: abs(e.get("weight", 0.0)), reverse=True)
    for e in in_to_feat[:top_n_io_edges]:
        graph_b_edges.append({
            "source":    e["source"],
            "target":    e["target"],
            "edge_type": "input_to_feature",
            "weight":    round(e.get("weight", 0.0), 6),
        })

    # 2. causal_feature_to_feature (both endpoints must be circuit members)
    for e in causal_edges:
        if e["source"] in circuit_nodes_set and e["target"] in circuit_nodes_set:
            graph_b_edges.append({
                "source":         e["source"],
                "target":         e["target"],
                "edge_type":      "causal_feature_to_feature",
                "mean_delta_abs": e["mean_delta_abs"],
                "effect_size":    e.get("effect_size"),
            })

    # 3. feature_to_output_correct
    feat_to_correct = [
        e for e in star_edges
        if e.get("source", "") in circuit_nodes_set
        and e.get("target", "") in correct_output_ids
    ]
    feat_to_correct.sort(key=lambda e: abs(e.get("weight", 0.0)), reverse=True)
    for e in feat_to_correct[:top_n_io_edges]:
        graph_b_edges.append({
            "source":    e["source"],
            "target":    e["target"],
            "edge_type": "feature_to_output_correct",
            "weight":    round(e.get("weight", 0.0), 6),
        })

    # 4. feature_to_output_incorrect
    feat_to_incorrect = [
        e for e in star_edges
        if e.get("source", "") in circuit_nodes_set
        and e.get("target", "") in incorrect_output_ids
    ]
    feat_to_incorrect.sort(key=lambda e: abs(e.get("weight", 0.0)), reverse=True)
    for e in feat_to_incorrect[:top_n_io_edges]:
        graph_b_edges.append({
            "source":    e["source"],
            "target":    e["target"],
            "edge_type": "feature_to_output_incorrect",
            "weight":    round(e.get("weight", 0.0), 6),
        })

    # Assert all edge types are valid (catches future bugs early)
    for e in graph_b_edges:
        assert e["edge_type"] in _VALID_GRAPH_B_EDGE_TYPES, (
            f"Invalid edge type in Graph B: {e['edge_type']}"
        )

    # Build node list for Graph B
    nodes_in_graph_b: set = set(circuit_nodes_set)
    for e in graph_b_edges:
        nodes_in_graph_b.add(e["source"])
        nodes_in_graph_b.add(e["target"])

    node_by_id = {n["id"]: n for n in graph["nodes"]}
    graph_b_nodes = [
        node_by_id.get(nid, {"id": nid})
        for nid in sorted(nodes_in_graph_b)
    ]

    return {
        "graph_type": "presentation",
        "note": (
            "Graph B (presentation only). "
            "Circuit membership derives from extract_circuit() path union in Graph A; "
            "this graph does NOT affect circuit membership."
        ),
        "n_nodes":        len(graph_b_nodes),
        "n_edges":        len(graph_b_edges),
        "top_n_io_edges": top_n_io_edges,
        "nodes":          graph_b_nodes,
        "edges":          graph_b_edges,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute intervention-based causal feature-to-feature edges"
    )
    parser.add_argument("--behaviour", type=str, default="multilingual_circuits_b1")
    parser.add_argument("--split",     type=str, default="train")
    parser.add_argument("--graph_json", type=str, default=None,
                        help="Path to attribution graph JSON (default: auto-detect)")
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--agw_top_frac", type=float, default=0.6,
                        help="Fraction of VW edges to keep after AGW prefilter (default 0.6)")
    parser.add_argument("--tau_causal", type=float, default=0.10,
                        help="mean_delta_abs threshold for keeping a causal edge (default 0.10)")
    parser.add_argument("--n_paths", type=int, default=50,
                        help="Number of top causal paths to trace (default 50)")
    parser.add_argument("--token_pos", type=int, default=-1,
                        help="Token position for feature collection (-1 = last/decision token)")
    parser.add_argument("--allow_sharded", action="store_true",
                        help="Allow model sharded across multiple devices")
    parser.add_argument("--skip_path_tracing", action="store_true",
                        help="Skip path tracing (only compute causal_edges)")
    parser.add_argument("--skip_sufficiency", action="store_true",
                        help="Skip Phase 8+9 sufficiency tests (S1 + S1.5 + S2)")
    parser.add_argument("--skip_s1_5", action="store_true",
                        help="Skip Phase 8b (S1.5 layerwise constrained); run S1 only")
    parser.add_argument("--skip_s2", action="store_true",
                        help="Skip Phase 9 (S2 cross-prompt injection) but run S1/S1.5")
    parser.add_argument("--debug_sanity", action="store_true",
                        help="Enable sanity logging inside S1.5 hooks (first prompt only)")
    parser.add_argument("--top_n_io_edges", type=int, default=10,
                        help="Max IO edges to include in presentation graph (default 10)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Override graph layers (default: inferred from graph)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Truncate prompt list to N prompts (debug/sanity mode only)")
    parser.add_argument("--allow_star_only_paths", action="store_true",
                        help="Allow 2-hop input→feature→output paths in circuit (no causal edge "
                             "required). Default: off — only paths with ≥1 causal edge are kept.")
    parser.add_argument("--from_causal_edges", type=str, default=None,
                        help="Skip Phases 1-5 (no ablation). Load causal edges from this "
                             "JSON path and jump directly to Phase 6 (path tracing) + "
                             "Phase 10 (presentation graph). "
                             "No model needed; does NOT rerun necessity/sufficiency. "
                             "Use after fixing path tracing without redoing the full ablation.")
    parser.add_argument("--from_circuit", type=str, default=None,
                        help="Skip Phases 1-6 (no ablation or path tracing). Load an existing "
                             "circuits JSON from this path and rerun only Phases 7-9 "
                             "(necessity + S1/S1.5 sufficiency + S2 injection) for the "
                             "circuit feature nodes already in that file. "
                             "Needs model + GPU. Updates the circuits JSON in-place, "
                             "preserving paths/circuit; also rewrites presentation graph.")
    args = parser.parse_args()

    config    = load_config()
    tc_config = load_transcoder_config()

    # ── Paths ──────────────────────────────────────────────────────────────
    base = Path("data")
    prompt_dir = base / "prompts"

    n_prompts_map = {"multilingual_circuits_b1": 96, "physics_conservation": 150}
    n_prompts_default = n_prompts_map.get(args.behaviour, 48)

    if args.graph_json:
        graph_path = Path(args.graph_json)
    else:
        # Auto-detect: prefer roleaware graph for b1
        agg_dir = base / "results" / "attribution_graphs" / args.behaviour
        if args.behaviour.endswith("_b1"):
            graph_path = agg_dir / f"attribution_graph_{args.split}_n{n_prompts_default}_roleaware.json"
        else:
            graph_path = agg_dir / f"attribution_graph_{args.split}_n{n_prompts_default}.json"

    # Debug mode: use separate output subdir to avoid overwriting full-run outputs
    if args.max_prompts:
        out_dir = base / "results" / "causal_edges" / args.behaviour / f"debug_n{args.max_prompts}"
    else:
        out_dir = base / "results" / "causal_edges" / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    causal_edges_path = out_dir / f"causal_edges_{args.behaviour}_{args.split}.json"
    circuits_path     = out_dir / f"circuits_{args.behaviour}_{args.split}.json"
    manifest_path     = out_dir / f"manifest_{args.behaviour}_{args.split}.json"

    # ── Failure-fast: output directory writable ────────────────────────────
    _check_writable(out_dir)

    # ── Failure-fast: graph file must exist before loading model ───────────
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Attribution graph not found: {graph_path}\n"
            f"  Run script 06 (roleaware mode) first.\n"
            f"  Expected: {graph_path}"
        )

    # ── Startup banner ─────────────────────────────────────────────────────
    mode_tag = f"DEBUG (max_prompts={args.max_prompts})" if args.max_prompts else "FULL"
    print("=" * 70)
    print(f"CAUSAL EDGES (INTERVENTION-BASED)  [{mode_tag}]")
    print("=" * 70)
    print(f"  Behaviour:        {args.behaviour}")
    print(f"  Split:            {args.split}")
    print(f"  Graph:            {graph_path}")
    print(f"  AGW top frac:     {args.agw_top_frac}")
    print(f"  tau_causal:       {args.tau_causal}")
    print(f"  n_paths:          {args.n_paths}")
    print(f"  token_pos:        {args.token_pos}")
    print(f"  Output dir:       {out_dir}")
    print(f"  skip_path_tracing:{args.skip_path_tracing}")
    print(f"  skip_sufficiency: {args.skip_sufficiency}")
    print(f"  skip_s1_5:        {args.skip_s1_5}")
    print(f"  skip_s2:          {args.skip_s2}")
    print(f"  debug_sanity:     {args.debug_sanity}")
    print(f"  max_prompts:      {args.max_prompts or 'all'}")

    # ------------------------------------------------------------------
    # Phase 0 – Load graph
    # ------------------------------------------------------------------
    logger.info("Phase 0: Loading graph")
    graph = load_graph(graph_path)
    feature_nodes = extract_graph_features(graph)
    vw_edges      = extract_vw_edges(graph)

    # ── Failure-fast: graph must have feature nodes ────────────────────
    if len(feature_nodes) == 0:
        raise RuntimeError(
            f"Graph has ZERO feature nodes: {graph_path}\n"
            "  Check that the graph was built with --graph_node_mode role_aware "
            "or that node type='feature' exists."
        )
    if len(vw_edges) == 0:
        raise RuntimeError(
            f"Graph has ZERO virtual-weight edges: {graph_path}\n"
            "  VW edges are required for AGW prefilter (Phase 1).  "
            "Re-run script 06 with --vw_threshold set."
        )

    node_by_id = {n["id"]: n for n in graph["nodes"]}
    node_freq  = {n["id"]: n.get("frequency", 0.0) for n in feature_nodes}

    logger.info(
        f"Graph: {len(feature_nodes)} feature nodes, {len(vw_edges)} VW edges"
    )

    # Infer graph layers
    graph_layers = sorted(set(n["layer"] for n in feature_nodes))
    if args.layers:
        graph_layers = sorted(args.layers)
    logger.info(f"Graph layers: {graph_layers}")

    # ------------------------------------------------------------------
    # --from_causal_edges fast path: skip Phases 1-9, redo 6+10 only
    # ------------------------------------------------------------------
    if args.from_causal_edges:
        ce_path = Path(args.from_causal_edges)
        if not ce_path.exists():
            raise FileNotFoundError(f"--from_causal_edges: file not found: {ce_path}")
        logger.info(f"--from_causal_edges mode: loading causal edges from {ce_path}")
        with open(ce_path) as f:
            ce_data = json.load(f)
        causal_edges = ce_data["edges"]
        logger.info(f"Loaded {len(causal_edges)} causal edges; jumping to Phase 6")

        if not causal_edges:
            logger.warning("No causal edges in provided file; nothing to trace")
            return

        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx not available; cannot retrace paths")
            return

        G = build_causal_dag(causal_edges, graph, include_star_edges=True)
        input_nodes  = [n["id"] for n in graph["nodes"] if n.get("type") == "input"]
        output_nodes = [n["id"] for n in graph["nodes"] if n.get("type") == "output"]
        logger.info(
            f"DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
            f"sources={input_nodes}, sinks={output_nodes}"
        )

        paths = find_top_k_paths(
            G, input_nodes, output_nodes, K=args.n_paths,
            weight_key="mean_delta_abs",
            require_causal_edge=not args.allow_star_only_paths,
        )
        if not paths:
            logger.warning("No paths found")
            return

        circuit = extract_circuit(paths, G)
        logger.info(
            f"Circuit: {circuit['n_features']} features, {circuit['n_edges']} edges, "
            f"from {circuit['n_paths']} paths"
        )

        # Phase 10: presentation graph
        graph_b_path = out_dir / f"presentation_graph_{args.behaviour}_{args.split}.json"
        graph_b = build_presentation_graph(
            circuit=circuit,
            causal_edges=causal_edges,
            graph=graph,
            top_n_io_edges=args.top_n_io_edges,
        )
        with open(graph_b_path, "w") as f:
            json.dump(graph_b, f, indent=2)
        logger.info(f"Saved presentation graph to {graph_b_path}")

        # Save updated circuits JSON (preserve existing validation if present)
        existing_circuits = {}
        if circuits_path.exists():
            with open(circuits_path) as f:
                existing_circuits = json.load(f)

        circuit_data = {
            "metadata": {
                "behaviour":         args.behaviour,
                "split":             args.split,
                "n_paths_requested": args.n_paths,
                "n_paths_found":     len(paths),
                "presentation_graph": str(graph_b_path),
                "timestamp":          datetime.now().isoformat(),
                "retraced_from":      str(ce_path),
            },
            "paths":            paths,
            "circuit":          circuit,
            # Preserve validation / sufficiency from the previous full run if available
            "validation":       existing_circuits.get("validation"),
            "sufficiency_s1":   existing_circuits.get("sufficiency_s1"),
            "sufficiency_s1_5": existing_circuits.get("sufficiency_s1_5"),
            "sufficiency_s2":   existing_circuits.get("sufficiency_s2"),
        }
        with open(circuits_path, "w") as f:
            json.dump(circuit_data, f, indent=2)
        logger.info(f"Saved updated circuits to {circuits_path}")

        print(f"\nCircuit (retraced): {circuit['n_features']} features, "
              f"{circuit['n_edges']} edges ({circuit['n_paths']} paths)")
        print(f"Top path (score={paths[0]['score']:.6f}):")
        print("  " + " → ".join(paths[0]["path"]))
        print(f"\nNote: validation/sufficiency metrics preserved from prior run.")
        print(f"      Use --from_circuit <circuits_json> to recompute them for the updated circuit.")
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # --from_circuit fast path: load existing circuit, rerun Phases 7-9
    # ------------------------------------------------------------------
    if args.from_circuit:
        fc_path = Path(args.from_circuit)
        if not fc_path.exists():
            raise FileNotFoundError(f"--from_circuit: file not found: {fc_path}")
        logger.info(f"--from_circuit mode: loading circuit from {fc_path}")
        with open(fc_path) as f:
            existing_circuits = json.load(f)

        # Extract circuit feature nodes (exclude I/O nodes like "input", "output_*")
        raw_feature_nodes = existing_circuits["circuit"].get("feature_nodes", [])
        circuit_feature_nodes = [
            n for n in raw_feature_nodes
            if not n.startswith("input") and not n.startswith("output")
        ]
        if not circuit_feature_nodes:
            raise ValueError(
                f"--from_circuit: no feature nodes found in circuit "
                f"(raw_feature_nodes={raw_feature_nodes})"
            )
        logger.info(
            f"--from_circuit: {len(circuit_feature_nodes)} feature nodes "
            f"(from {len(raw_feature_nodes)} total circuit nodes)"
        )

        # ── Phase 2: Load model + transcoders ────────────────────────
        model_size = args.model_size or tc_config.get("model_size", "4b")
        logger.info(f"Phase 2: Loading model ({model_size}) and transcoders")
        model, tc_set, device = load_model_and_transcoders(
            tc_config, model_size, graph_layers, args.allow_sharded
        )
        available_tc_layers = set(tc_set.layers) if hasattr(tc_set, "layers") else set(graph_layers)
        missing_tc_layers   = [l for l in graph_layers if l not in available_tc_layers]
        if missing_tc_layers:
            raise RuntimeError(
                f"Transcoder missing layers required by graph: {missing_tc_layers}\n"
                "  Check transcoder_config.yaml layer range vs graph layers."
            )

        # ── Load prompts ──────────────────────────────────────────────
        prompts = load_prompts(prompt_dir, args.behaviour, args.split)
        if len(prompts) == 0:
            raise RuntimeError(
                f"Zero prompts loaded from "
                f"{prompt_dir / (args.behaviour + '_' + args.split + '.jsonl')}"
            )
        logger.info(f"Loaded {len(prompts)} prompts from disk")
        if args.max_prompts and args.max_prompts < len(prompts):
            prompts = prompts[:args.max_prompts]
            logger.info(f"DEBUG: truncated to {len(prompts)} prompts (--max_prompts {args.max_prompts})")

        # ── Phase 3: Build baseline cache (needed for S2) ────────────
        logger.info("Phase 3: Building baseline activation cache (needed for S2 injection)")
        baseline_mlp_inputs, baseline_feat_acts = build_baseline_cache(
            model, tc_set, device, prompts, graph_layers, token_pos=args.token_pos
        )

        # ── Phase 7: Necessity (group ablation) ──────────────────────
        logger.info("Phase 7: Circuit-level validation (group ablation / necessity)")
        val = validate_circuit(
            model=model,
            tc_set=tc_set,
            device=device,
            prompts=prompts,
            circuit_feature_nodes=circuit_feature_nodes,
            token_pos=args.token_pos,
        )
        logger.info(
            f"Circuit validation: disruption_rate={val['disruption_rate']}, "
            f"mean_effect={val['mean_effect']}, n={val['n']}"
        )

        # ── Phase 8: Sufficiency S1 / S1.5 ───────────────────────────
        sufficiency_s1   = None
        sufficiency_s1_5 = None
        sufficiency_s2   = None

        if not args.skip_sufficiency:
            if not args.skip_s1_5:
                logger.info(
                    "Phase 8: Circuit sufficiency S1 + S1.5 "
                    "(linear complement ablation + layerwise constrained)"
                )
                sufficiency_s1_5 = validate_circuit_sufficiency_s1_5(
                    model=model,
                    tc_set=tc_set,
                    device=device,
                    prompts=prompts,
                    circuit_feature_nodes=circuit_feature_nodes,
                    token_pos=args.token_pos,
                    debug_sanity=args.debug_sanity,
                )
                logger.info(
                    f"S1 linear:    sign={sufficiency_s1_5['sign_preserved_rate_s1']}, "
                    f"retention={sufficiency_s1_5['mean_retention_s1']}, "
                    f"verdict={sufficiency_s1_5['verdict_s1']}"
                )
                logger.info(
                    f"S1.5 layerwise: sign={sufficiency_s1_5['sign_preserved_rate_s15']}, "
                    f"retention={sufficiency_s1_5['mean_retention_s15']}, "
                    f"verdict={sufficiency_s1_5['verdict_s15']}, "
                    f"improvement={sufficiency_s1_5['retention_improvement']}"
                )
                sufficiency_s1 = {
                    "sign_preserved_rate":  sufficiency_s1_5["sign_preserved_rate_s1"],
                    "mean_retention_ratio": sufficiency_s1_5["mean_retention_s1"],
                    "verdict":              sufficiency_s1_5["verdict_s1"],
                    "n":                    sufficiency_s1_5["n"],
                }
            else:
                logger.info("Phase 8: Circuit sufficiency S1 (linear only, --skip_s1_5)")
                sufficiency_s1 = validate_circuit_sufficiency_s1(
                    model=model,
                    tc_set=tc_set,
                    device=device,
                    prompts=prompts,
                    circuit_feature_nodes=circuit_feature_nodes,
                    token_pos=args.token_pos,
                )
                logger.info(
                    f"S1 Sufficiency: sign_preserved={sufficiency_s1['sign_preserved_rate']}, "
                    f"retention={sufficiency_s1['mean_retention_ratio']}, "
                    f"verdict={sufficiency_s1['verdict']}, n={sufficiency_s1['n']}"
                )

            # ── Phase 9: S2 cross-prompt injection ────────────────────
            if not args.skip_s2:
                logger.info("Phase 9: Circuit sufficiency S2 (cross-prompt injection)")
                sufficiency_s2 = validate_circuit_sufficiency_s2(
                    model=model,
                    tc_set=tc_set,
                    device=device,
                    prompts=prompts,
                    circuit_feature_nodes=circuit_feature_nodes,
                    baseline_feat_acts=baseline_feat_acts,
                    token_pos=args.token_pos,
                )
                logger.info(
                    f"S2 Sufficiency: n_pairs={sufficiency_s2['n_pairs']}, "
                    f"transfer_rate={sufficiency_s2['transfer_rate']}, "
                    f"mean_shift={sufficiency_s2['mean_shift']}"
                )

        # ── Phase 10: Rebuild presentation graph ─────────────────────
        logger.info("Phase 10: Rebuilding presentation graph (Graph B) for updated circuit")
        # Load causal edges (needed by build_presentation_graph)
        causal_edges_path_fc = out_dir / f"causal_edges_{args.behaviour}_{args.split}.json"
        if causal_edges_path_fc.exists():
            with open(causal_edges_path_fc) as f:
                ce_data = json.load(f)
            causal_edges_fc = ce_data.get("edges", [])
        else:
            logger.warning(
                f"Causal edges file not found at {causal_edges_path_fc}; "
                "presentation graph will have no causal edges"
            )
            causal_edges_fc = []

        circuit_obj = existing_circuits["circuit"]
        graph_b = build_presentation_graph(
            circuit=circuit_obj,
            causal_edges=causal_edges_fc,
            graph=graph,
            top_n_io_edges=args.top_n_io_edges,
        )
        logger.info(
            f"Graph B: {graph_b['n_nodes']} nodes, {graph_b['n_edges']} edges "
            f"(top_n_io_edges={args.top_n_io_edges})"
        )
        graph_b_path = out_dir / f"presentation_graph_{args.behaviour}_{args.split}.json"
        with open(graph_b_path, "w") as f:
            json.dump(graph_b, f, indent=2)
        logger.info(f"Saved updated presentation graph to {graph_b_path}")

        # ── Save updated circuits JSON (overwrite, preserve paths/circuit) ──
        updated_circuits = dict(existing_circuits)  # shallow copy preserves all existing keys
        updated_circuits["validation"]       = val
        updated_circuits["sufficiency_s1"]   = sufficiency_s1
        updated_circuits["sufficiency_s1_5"] = sufficiency_s1_5
        updated_circuits["sufficiency_s2"]   = sufficiency_s2
        updated_circuits.setdefault("metadata", {})["revalidated_from"] = str(fc_path)
        updated_circuits["metadata"]["revalidated_timestamp"] = datetime.now().isoformat()

        with open(fc_path, "w") as f:
            json.dump(updated_circuits, f, indent=2)
        logger.info(f"Saved updated circuits to {fc_path}")

        # ── Print summary ─────────────────────────────────────────────
        print(f"\nCircuit (revalidated): {circuit_obj['n_features']} features, "
              f"{circuit_obj['n_edges']} edges ({circuit_obj['n_paths']} paths)")
        print(f"Phase 7 necessity:    disruption_rate={val['disruption_rate']}, "
              f"mean_effect={val['mean_effect']:.4f}, n={val['n']}")
        if sufficiency_s1_5 is not None:
            print(f"Phase 8 S1  linear:   sign={sufficiency_s1_5['sign_preserved_rate_s1']}, "
                  f"retention={sufficiency_s1_5['mean_retention_s1']}, "
                  f"verdict={sufficiency_s1_5['verdict_s1']}")
            print(f"Phase 8 S1.5 layerwise: sign={sufficiency_s1_5['sign_preserved_rate_s15']}, "
                  f"retention={sufficiency_s1_5['mean_retention_s15']}, "
                  f"verdict={sufficiency_s1_5['verdict_s15']}, "
                  f"improvement={sufficiency_s1_5['retention_improvement']}")
        elif sufficiency_s1 is not None:
            print(f"Phase 8 S1 sufficiency: sign_preserved={sufficiency_s1['sign_preserved_rate']}, "
                  f"retention={sufficiency_s1['mean_retention_ratio']}, "
                  f"verdict={sufficiency_s1['verdict']}")
        if sufficiency_s2 is not None:
            print(f"Phase 9 S2 injection:   n_pairs={sufficiency_s2['n_pairs']}, "
                  f"transfer_rate={sufficiency_s2['transfer_rate']}, "
                  f"mean_shift={sufficiency_s2['mean_shift']}")
        print(f"Phase 10 Graph B:     {graph_b['n_nodes']} nodes, {graph_b['n_edges']} edges "
              f"→ {graph_b_path}")
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # Phase 1 – AGW prefilter
    # ------------------------------------------------------------------
    logger.info("Phase 1: Computing AGW scores")
    scored_edges = compute_agw_scores(vw_edges, node_freq)
    candidate_pairs = filter_candidate_pairs(scored_edges, args.agw_top_frac)

    # Collect unique source features from candidate pairs
    src_features = sorted(set(
        (int(e["source"].split("_")[0][1:]), int(e["source"].split("_")[1][1:]))
        for _, e in candidate_pairs
    ))
    logger.info(f"Unique source features to ablate: {len(src_features)}")

    # Map source → list of (tgt_layer, tgt_feat_idx)
    src_to_targets: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for agw_score, edge in candidate_pairs:
        sl = int(edge["source"].split("_")[0][1:])
        sf = int(edge["source"].split("_")[1][1:])
        tl = int(edge["target"].split("_")[0][1:])
        tf = int(edge["target"].split("_")[1][1:])
        if tl > sl:
            src_to_targets[(sl, sf)].append((tl, tf))

    # ------------------------------------------------------------------
    # Phase 2 – Load model + transcoders
    # ------------------------------------------------------------------
    model_size = args.model_size or tc_config.get("model_size", "4b")
    logger.info(f"Phase 2: Loading model ({model_size}) and transcoders")
    model, tc_set, device = load_model_and_transcoders(
        tc_config, model_size, graph_layers, args.allow_sharded
    )

    # ── Failure-fast: TC layers must cover all graph layers ────────────
    available_tc_layers = set(tc_set.layers) if hasattr(tc_set, "layers") else set(graph_layers)
    missing_tc_layers   = [l for l in graph_layers if l not in available_tc_layers]
    if missing_tc_layers:
        raise RuntimeError(
            f"Transcoder missing layers required by graph: {missing_tc_layers}\n"
            "  Check transcoder_config.yaml layer range vs graph layers."
        )

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    prompts = load_prompts(prompt_dir, args.behaviour, args.split)
    if len(prompts) == 0:
        raise RuntimeError(
            f"Zero prompts loaded from {prompt_dir / (args.behaviour + '_' + args.split + '.jsonl')}"
        )
    logger.info(f"Loaded {len(prompts)} prompts from disk")

    # ── Debug mode: truncate prompt list ──────────────────────────────
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
        logger.info(
            f"DEBUG: truncated to {len(prompts)} prompts "
            f"(--max_prompts {args.max_prompts})"
        )

    # ── Log estimated forward passes ──────────────────────────────────
    n_abl_passes  = len(src_features) * len(prompts)
    n_val_passes  = len(prompts) if not args.skip_path_tracing else 0
    n_suf_passes  = 3 * len(prompts) if not args.skip_sufficiency else 0
    n_total_est   = len(prompts) + n_abl_passes + n_val_passes + n_suf_passes
    logger.info(
        f"Estimated forward passes:"
        f"  baseline={len(prompts)}"
        f"  ablation={n_abl_passes} ({len(src_features)} src × {len(prompts)} prompts)"
        f"  necessity={n_val_passes}"
        f"  sufficiency={n_suf_passes} (3 × {len(prompts)} prompts)"
        f"  TOTAL≈{n_total_est}"
    )

    # ── Save run manifest (before heavy computation) ──────────────────
    import socket
    manifest = {
        "timestamp":         datetime.now().isoformat(),
        "git_commit":        _get_git_commit(),
        "host":              socket.gethostname(),
        "device":            str(device),
        "behaviour":         args.behaviour,
        "split":             args.split,
        "graph_path":        str(graph_path),
        "model_size":        model_size,
        "model_name":        tc_config["transcoders"][model_size]["model_name"],
        "n_prompts":         len(prompts),
        "max_prompts_arg":   args.max_prompts,
        "graph_layers":      graph_layers,
        "n_feature_nodes":   len(feature_nodes),
        "n_vw_edges":        len(vw_edges),
        "agw_top_frac":      args.agw_top_frac,
        "tau_causal":        args.tau_causal,
        "n_paths":           args.n_paths,
        "token_pos":         args.token_pos,
        "skip_path_tracing": args.skip_path_tracing,
        "skip_sufficiency":  args.skip_sufficiency,
        "skip_s1_5":         args.skip_s1_5,
        "skip_s2":           args.skip_s2,
        "debug_sanity":      args.debug_sanity,
        "debug_mode":        bool(args.max_prompts),
        "output_dir":        str(out_dir),
        "causal_edges_path": str(causal_edges_path),
        "circuits_path":     str(circuits_path),
        "sufficiency_modes": (
            ["s1_linear", "s1_5_layerwise"] if not args.skip_sufficiency and not args.skip_s1_5
            else (["s1_linear"] if not args.skip_sufficiency else [])
        ),
        "estimated_passes":  n_total_est,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")

    # ------------------------------------------------------------------
    # Phase 3 – Baseline cache
    # ------------------------------------------------------------------
    logger.info("Phase 3: Building baseline activation cache")
    baseline_mlp_inputs, baseline_feat_acts = build_baseline_cache(
        model, tc_set, device, prompts, graph_layers, token_pos=args.token_pos
    )

    # ------------------------------------------------------------------
    # Phase 4 – Ablation loop
    # ------------------------------------------------------------------
    logger.info(
        f"Phase 4: Ablation loop — {len(src_features)} sources × {len(prompts)} prompts "
        f"= {len(src_features) * len(prompts)} forward passes"
    )

    # Store: {((sl,sf),(tl,tf)): delta_array}
    all_deltas: Dict[Tuple[Tuple[int,int], Tuple[int,int]], np.ndarray] = {}

    # Per-source timeout: 10 min per source (96 prompts × ~80s/source = ~1.5 min normal;
    # 10 min gives 6× headroom before SIGALRM fires and we skip the hung source).
    # signal.alarm is only available on POSIX (Linux/CSD3); no-op on Windows.
    _has_sigalrm = hasattr(signal, "SIGALRM")
    _PER_SOURCE_TIMEOUT_S = 600  # 10 minutes
    if _has_sigalrm:
        signal.signal(signal.SIGALRM, _sigalrm_handler)

    skipped_sources: List[Tuple[int, int]] = []

    for src_idx, (src_layer, src_feat) in enumerate(
        tqdm(src_features, desc="Ablating source features")
    ):
        tgt_list = src_to_targets.get((src_layer, src_feat), [])
        if not tgt_list:
            continue

        logger.info(
            f"  Phase 4 source {src_idx+1}/{len(src_features)}: "
            f"L{src_layer}_F{src_feat}  ({len(tgt_list)} targets)"
        )

        if _has_sigalrm:
            signal.alarm(_PER_SOURCE_TIMEOUT_S)
        try:
            delta_by_tgt = run_ablation_for_source(
                model=model,
                tc_set=tc_set,
                device=device,
                prompts=prompts,
                src_layer=src_layer,
                src_feat_idx=src_feat,
                target_pairs=tgt_list,
                baseline_mlp_inputs=baseline_mlp_inputs,
                baseline_feat_acts=baseline_feat_acts,
                token_pos=args.token_pos,
            )
        except _SourceTimeout:
            logger.warning(
                f"  TIMEOUT: L{src_layer}_F{src_feat} exceeded {_PER_SOURCE_TIMEOUT_S}s — skipping"
            )
            skipped_sources.append((src_layer, src_feat))
            continue
        except Exception as exc:
            logger.warning(
                f"  ERROR: L{src_layer}_F{src_feat} raised {type(exc).__name__}: {exc} — skipping"
            )
            skipped_sources.append((src_layer, src_feat))
            continue
        finally:
            if _has_sigalrm:
                signal.alarm(0)  # cancel pending alarm

        for (tl, tf), deltas in delta_by_tgt.items():
            all_deltas[((src_layer, src_feat), (tl, tf))] = deltas

    if skipped_sources:
        logger.warning(
            f"Phase 4: {len(skipped_sources)} sources skipped (timeout/error): "
            + ", ".join(f"L{l}_F{f}" for l, f in skipped_sources)
        )

    # ------------------------------------------------------------------
    # Phase 5 – Aggregate edges
    # ------------------------------------------------------------------
    logger.info("Phase 5: Aggregating causal edges")
    causal_edges = aggregate_edges(
        candidate_pairs, all_deltas, args.tau_causal, len(prompts)
    )
    logger.info(
        f"Causal edges: {len(candidate_pairs)} candidates → "
        f"{len(causal_edges)} passed tau_causal={args.tau_causal}"
    )

    # Save causal edges
    out_data = {
        "metadata": {
            "behaviour":      args.behaviour,
            "split":          args.split,
            "graph_json":     str(graph_path),
            "n_prompts":      len(prompts),
            "n_vw_edges":     len(vw_edges),
            "agw_top_frac":   args.agw_top_frac,
            "n_candidates":   len(candidate_pairs),
            "tau_causal":     args.tau_causal,
            "n_causal_edges": len(causal_edges),
            "token_pos":      args.token_pos,
            "timestamp":      datetime.now().isoformat(),
        },
        "edges": causal_edges,
    }

    with open(causal_edges_path, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info(f"Saved causal edges to {causal_edges_path}")

    print(f"\nCausal edges: {len(causal_edges)} (from {len(candidate_pairs)} candidates)")
    if causal_edges:
        top5 = causal_edges[:5]
        print("Top 5 by mean_delta_abs:")
        for e in top5:
            print(
                f"  {e['source']} → {e['target']}: "
                f"mean_delta={e['mean_delta']:.4f}, mean_delta_abs={e['mean_delta_abs']:.4f}, "
                f"effect_size(diag)={e['effect_size']:.3f}"
            )

    # ------------------------------------------------------------------
    # Phase 6 – Path tracing (optional)
    # ------------------------------------------------------------------
    if not args.skip_path_tracing and causal_edges:
        logger.info("Phase 6: Building causal DAG and tracing paths")
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx not available; skipping path tracing")
            return

        G = build_causal_dag(causal_edges, graph, include_star_edges=True)

        # Identify sources (input node) and sinks (output nodes)
        input_nodes  = [n["id"] for n in graph["nodes"] if n.get("type") == "input"]
        output_nodes = [n["id"] for n in graph["nodes"] if n.get("type") == "output"]

        logger.info(
            f"DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
            f"sources={input_nodes}, sinks={output_nodes}"
        )

        paths = find_top_k_paths(
            G, input_nodes, output_nodes, K=args.n_paths,
            weight_key="mean_delta_abs",
            require_causal_edge=not args.allow_star_only_paths,
        )

        if not paths:
            logger.warning("No complete paths found from input to output nodes")
        else:
            circuit = extract_circuit(paths, G)
            logger.info(
                f"Circuit: {circuit['n_features']} features, {circuit['n_edges']} edges, "
                f"from {circuit['n_paths']} paths"
            )

            # ------------------------------------------------------------------
            # Phase 7 – Circuit-level validation (necessity)
            # Group-ablate all circuit feature nodes simultaneously; measure
            # output disruption.  This is the minimal test for a strict circuit:
            # ablating the union of top-path nodes should disrupt behaviour.
            # ------------------------------------------------------------------
            logger.info("Phase 7: Circuit-level validation (group ablation / necessity)")
            val = validate_circuit(
                model=model,
                tc_set=tc_set,
                device=device,
                prompts=prompts,
                circuit_feature_nodes=circuit["feature_nodes"],
                token_pos=args.token_pos,
            )
            logger.info(
                f"Circuit validation: disruption_rate={val['disruption_rate']}, "
                f"mean_effect={val['mean_effect']}, n={val['n']}"
            )

            # ------------------------------------------------------------------
            # Phase 8 – Sufficiency S1 (complement ablation)
            # Remove all non-circuit MLP contributions; check whether circuit
            # alone preserves the logit direction and magnitude.
            # ------------------------------------------------------------------
            sufficiency_s1   = None
            sufficiency_s1_5 = None
            sufficiency_s2   = None

            if not args.skip_sufficiency:
                if not args.skip_s1_5:
                    # S1.5 runs all three modes in 3 forward passes per prompt.
                    # It subsumes the standalone S1 call; S1 linear metrics are
                    # also returned for backward compatibility.
                    logger.info(
                        "Phase 8: Circuit sufficiency S1 + S1.5 "
                        "(linear complement ablation + layerwise constrained)"
                    )
                    sufficiency_s1_5 = validate_circuit_sufficiency_s1_5(
                        model=model,
                        tc_set=tc_set,
                        device=device,
                        prompts=prompts,
                        circuit_feature_nodes=circuit["feature_nodes"],
                        token_pos=args.token_pos,
                        debug_sanity=args.debug_sanity,
                    )
                    logger.info(
                        f"S1 linear:    sign={sufficiency_s1_5['sign_preserved_rate_s1']}, "
                        f"retention={sufficiency_s1_5['mean_retention_s1']}, "
                        f"verdict={sufficiency_s1_5['verdict_s1']}"
                    )
                    logger.info(
                        f"S1.5 layerwise: sign={sufficiency_s1_5['sign_preserved_rate_s15']}, "
                        f"retention={sufficiency_s1_5['mean_retention_s15']}, "
                        f"verdict={sufficiency_s1_5['verdict_s15']}, "
                        f"improvement={sufficiency_s1_5['retention_improvement']}"
                    )
                    # Expose S1 metrics in legacy format for downstream consumers
                    sufficiency_s1 = {
                        "sign_preserved_rate":  sufficiency_s1_5["sign_preserved_rate_s1"],
                        "mean_retention_ratio": sufficiency_s1_5["mean_retention_s1"],
                        "verdict":              sufficiency_s1_5["verdict_s1"],
                        "n":                    sufficiency_s1_5["n"],
                    }
                else:
                    # Fallback: standalone S1 only (--skip_s1_5 mode)
                    logger.info("Phase 8: Circuit sufficiency S1 (linear only, --skip_s1_5)")
                    sufficiency_s1 = validate_circuit_sufficiency_s1(
                        model=model,
                        tc_set=tc_set,
                        device=device,
                        prompts=prompts,
                        circuit_feature_nodes=circuit["feature_nodes"],
                        token_pos=args.token_pos,
                    )
                    logger.info(
                        f"S1 Sufficiency: sign_preserved={sufficiency_s1['sign_preserved_rate']}, "
                        f"retention={sufficiency_s1['mean_retention_ratio']}, "
                        f"verdict={sufficiency_s1['verdict']}, n={sufficiency_s1['n']}"
                    )

                # --------------------------------------------------------------
                # Phase 9 – Sufficiency S2 (cross-prompt injection, optional)
                # Inject EN circuit activations into paired FR prompt; check
                # whether the injected circuit steers FR logits toward EN answer.
                # --------------------------------------------------------------
                if not args.skip_s2:
                    logger.info("Phase 9: Circuit sufficiency S2 (cross-prompt injection)")
                    sufficiency_s2 = validate_circuit_sufficiency_s2(
                        model=model,
                        tc_set=tc_set,
                        device=device,
                        prompts=prompts,
                        circuit_feature_nodes=circuit["feature_nodes"],
                        baseline_feat_acts=baseline_feat_acts,
                        token_pos=args.token_pos,
                    )
                    logger.info(
                        f"S2 Sufficiency: n_pairs={sufficiency_s2['n_pairs']}, "
                        f"transfer_rate={sufficiency_s2['transfer_rate']}, "
                        f"mean_shift={sufficiency_s2['mean_shift']}"
                    )

            # ------------------------------------------------------------------
            # Phase 10 – Presentation graph (Graph B)
            # Anthropic-style star graph: typed edge classes, decoupled from
            # Graph A.  Circuit membership is NOT re-derived here.
            # ------------------------------------------------------------------
            logger.info("Phase 10: Building presentation graph (Graph B)")
            graph_b = build_presentation_graph(
                circuit=circuit,
                causal_edges=causal_edges,
                graph=graph,
                top_n_io_edges=args.top_n_io_edges,
            )
            logger.info(
                f"Graph B: {graph_b['n_nodes']} nodes, {graph_b['n_edges']} edges "
                f"(top_n_io_edges={args.top_n_io_edges})"
            )
            graph_b_path = out_dir / f"presentation_graph_{args.behaviour}_{args.split}.json"
            with open(graph_b_path, "w") as f:
                json.dump(graph_b, f, indent=2)
            logger.info(f"Saved presentation graph to {graph_b_path}")

            # ------------------------------------------------------------------
            # Save circuits JSON (includes necessity + sufficiency results)
            # ------------------------------------------------------------------
            circuit_data = {
                "metadata": {
                    "behaviour":              args.behaviour,
                    "split":                  args.split,
                    "n_paths_requested":      args.n_paths,
                    "n_paths_found":          len(paths),
                    "presentation_graph":     str(graph_b_path),
                    "timestamp":              datetime.now().isoformat(),
                },
                "paths":            paths,
                "circuit":          circuit,
                "validation":       val,             # Phase 7: necessity (group ablation)
                "sufficiency_s1":   sufficiency_s1,  # Phase 8: S1 linear (legacy format)
                "sufficiency_s1_5": sufficiency_s1_5,# Phase 8b: S1.5 layerwise + S1 comparison
                "sufficiency_s2":   sufficiency_s2,  # Phase 9: cross-prompt injection
            }

            with open(circuits_path, "w") as f:
                json.dump(circuit_data, f, indent=2)
            logger.info(f"Saved circuits to {circuits_path}")

            print(f"\nCircuit: {circuit['n_features']} features, "
                  f"{circuit['n_edges']} edges ({circuit['n_paths']} paths)")
            print(f"Phase 7 necessity:    disruption_rate={val['disruption_rate']}, "
                  f"mean_effect={val['mean_effect']:.4f}, n={val['n']}")
            if sufficiency_s1_5 is not None:
                print(f"Phase 8 S1  linear:   sign={sufficiency_s1_5['sign_preserved_rate_s1']}, "
                      f"retention={sufficiency_s1_5['mean_retention_s1']}, "
                      f"verdict={sufficiency_s1_5['verdict_s1']}")
                print(f"Phase 8 S1.5 layerwise: sign={sufficiency_s1_5['sign_preserved_rate_s15']}, "
                      f"retention={sufficiency_s1_5['mean_retention_s15']}, "
                      f"verdict={sufficiency_s1_5['verdict_s15']}, "
                      f"improvement={sufficiency_s1_5['retention_improvement']}")
            elif sufficiency_s1 is not None:
                print(f"Phase 8 S1 sufficiency: sign_preserved={sufficiency_s1['sign_preserved_rate']}, "
                      f"retention={sufficiency_s1['mean_retention_ratio']}, "
                      f"verdict={sufficiency_s1['verdict']}")
            if sufficiency_s2 is not None:
                print(f"Phase 9 S2 injection:   n_pairs={sufficiency_s2['n_pairs']}, "
                      f"transfer_rate={sufficiency_s2['transfer_rate']}, "
                      f"mean_shift={sufficiency_s2['mean_shift']}")
            print(f"Phase 10 Graph B:     {graph_b['n_nodes']} nodes, {graph_b['n_edges']} edges "
                  f"→ {graph_b_path}")
            if paths:
                print(f"Top path (score={paths[0]['score']:.6f}):")
                print("  " + " → ".join(paths[0]["path"]))

    print("\nDone.")


if __name__ == "__main__":
    main()
