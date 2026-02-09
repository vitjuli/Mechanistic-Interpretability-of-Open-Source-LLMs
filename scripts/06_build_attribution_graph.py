"""
Build attribution graphs from transcoder features to model outputs.

Implements a simplified version of the methodology from:
"On the Biology of a Large Language Model" (Lindsey, Gurnee, et al., 2025)

This version uses pre-trained transcoders instead of custom-trained SAEs.
The attribution graph shows causal relationships:
    Input tokens -> Transcoder features -> ... -> Output logits

Pre-trained transcoders from: https://github.com/safety-research/circuit-tracer

Usage:
    python scripts/06_build_attribution_graph.py
    python scripts/06_build_attribution_graph.py --model_size 4b --n_prompts 40
"""

import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_transcoder_config(config_path: str = "configs/transcoder_config.yaml") -> Dict:
    """Load transcoder configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompt_path: Path, behaviour: str, split: str = "train") -> List[Dict]:
    """Load prompts from JSONL file."""
    file_path = prompt_path / f"{behaviour}_{split}.jsonl"
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_extracted_features(
    features_path: Path,
    behaviour: str,
    split: str,
    layers: List[int],
) -> Dict[int, Dict]:
    """
    Load pre-extracted transcoder features from script 04 outputs.
    
    This is CRITICAL: we use features already computed from MLP inputs,
    not residual stream, ensuring correct distribution for transcoders.
    
    Args:
        features_path: Path to data/results/transcoder_features/
        behaviour: Behaviour name (e.g., "grammar_agreement")
        split: "train" or "test"
        layers: List of layer indices to load
    
    Returns:
        Dictionary mapping layer_idx -> {
            'top_k_indices': torch.Tensor (n_samples, top_k),
            'top_k_values': torch.Tensor (n_samples, top_k),
            'feature_frequencies': np.ndarray (d_transcoder,),
            'metadata': dict,
        }
    """
    extracted_features = {}
    
    logger.info(f"Loading extracted features for {behaviour}_{split}...")
    
    for layer_idx in layers:
        layer_dir = features_path / f"layer_{layer_idx}"
        
        if not layer_dir.exists():
            raise FileNotFoundError(
                f"Layer {layer_idx} directory not found: {layer_dir}\n"
                f"Run script 04 first to extract features."
            )
        
        # Load arrays
        top_k_indices_path = layer_dir / f"{behaviour}_{split}_top_k_indices.npy"
        top_k_values_path = layer_dir / f"{behaviour}_{split}_top_k_values.npy"
        freq_path = layer_dir / f"{behaviour}_{split}_feature_frequencies.npy"
        meta_path = layer_dir / f"{behaviour}_{split}_layer_meta.json"
        
        # Check all files exist
        for path in [top_k_indices_path, top_k_values_path, freq_path, meta_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
        
        # Load numpy arrays
        top_k_indices = np.load(top_k_indices_path)
        top_k_values = np.load(top_k_values_path)
        feature_frequencies = np.load(freq_path)
        
        # Load metadata
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        # Convert to torch tensors
        extracted_features[layer_idx] = {
            'top_k_indices': torch.from_numpy(top_k_indices).long(),
            'top_k_values': torch.from_numpy(top_k_values).float(),
            'feature_frequencies': feature_frequencies,
            'metadata': metadata,
        }
        
        logger.info(
            f"  Layer {layer_idx}: {top_k_indices.shape[0]} samples, "
            f"top-{top_k_indices.shape[1]} features, "
            f"{metadata['n_active_features']} active features, "
            f"token_positions={metadata.get('token_positions')}, "
            f"context_tokens={metadata.get('context_tokens')}"
        )
    
    return extracted_features


def load_position_map(
    features_path: Path,
    behaviour: str,
    split: str,
) -> List[Dict]:
    """
    Load position map that links sample indices to (prompt_idx, token_pos, token_id).
    
    CRITICAL for attribution: each sample in extracted features corresponds to
    one token in one prompt. We need this mapping to attribute features back
    to specific prompts.
    
    Args:
        features_path: Path to data/results/transcoder_features/
        behaviour: Behaviour name
        split: "train" or "test"
    
    Returns:
        List of dicts: [
            {'prompt_idx': 0, 'token_pos': 3, 'token_id': 1234},
            ...
        ]
    """
    position_map_path = features_path / f"{behaviour}_{split}_position_map.json"
    
    if not position_map_path.exists():
        raise FileNotFoundError(
            f"Position map not found: {position_map_path}\n"
            f"Run script 04 first to extract features."
        )
    
    with open(position_map_path, "r") as f:
        position_map = json.load(f)
    
    logger.info(f"Loaded position map: {len(position_map)} samples")
    
    return position_map


def build_prompt_to_samples_map(position_map: List[Dict]) -> Dict[int, List[int]]:
    """
    Build reverse mapping from prompt_idx to sample indices.
    
    Args:
        position_map: Output from load_position_map()
    
    Returns:
        Dict mapping prompt_idx -> [sample_idx_1, sample_idx_2, ...]
    
    Example:
        {
            0: [0, 1, 2, 3, 4],  # Prompt 0 has 5 token samples
            1: [5, 6, 7, 8, 9],  # Prompt 1 has 5 token samples
            ...
        }
    """
    from collections import defaultdict
    
    prompt_to_samples = defaultdict(list)
    
    for sample_idx, entry in enumerate(position_map):
        prompt_idx = entry['prompt_idx']
        prompt_to_samples[prompt_idx].append(sample_idx)
    
    # Convert to regular dict and sort sample indices
    prompt_to_samples = {
        prompt_idx: sorted(sample_indices)
        for prompt_idx, sample_indices in prompt_to_samples.items()
    }
    
    logger.info(
        f"Built prompt-to-samples map: {len(prompt_to_samples)} prompts, "
        f"avg {np.mean([len(s) for s in prompt_to_samples.values()]):.1f} samples/prompt"
    )
    
    return prompt_to_samples


def compute_margin_batch(
    model: ModelWrapper,
    prompts: List[Dict],
) -> np.ndarray:
    """
    Δ_p = log p(y⁺|c) - log p(y⁻|c) for each prompt (single-token answers only).
    """
    margins = []

    model_device = next(model.model.parameters()).device
    logger.info(f"Computing margins for {len(prompts)} prompts on device={model_device}...")

    for prompt_data in tqdm(prompts, desc="Computing margins"):
        prompt_text = prompt_data["prompt"]

        # Use the SAME keys everywhere (choose one schema and stick to it)
        correct_tok = prompt_data["answer_matching"]
        incorrect_tok = prompt_data["answer_not_matching"]

        inputs = model.tokenizer(prompt_text, return_tensors="pt").to(model_device)

        with torch.no_grad():
            outputs = model.model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]  # next-token logits at final position
            log_probs = torch.log_softmax(logits, dim=0)

        # Enforce single-token answers (critical for correctness)
        ids_pos = model.tokenizer.encode(correct_tok, add_special_tokens=False)
        ids_neg = model.tokenizer.encode(incorrect_tok, add_special_tokens=False)
        
        # CRITICAL: Use raise instead of assert (assert can be disabled with -O flag)
        if len(ids_pos) != 1 or len(ids_neg) != 1:
            raise ValueError(
                f"Answers must be single token.\n"
                f"correct_tok={correct_tok!r} -> ids={ids_pos}\n"
                f"incorrect_tok={incorrect_tok!r} -> ids={ids_neg}\n"
                f"Fix dataset/tokenization or implement multi-token continuation scoring."
            )

        correct_id = ids_pos[0]
        incorrect_id = ids_neg[0]

        margin = (log_probs[correct_id] - log_probs[incorrect_id]).item()
        margins.append(margin)

    margins = np.array(margins, dtype=np.float64)

    logger.info(
        f"Margins: mean={margins.mean():.4f}, std={margins.std():.4f}, "
        f"min={margins.min():.4f}, max={margins.max():.4f}"
    )
    return margins




class TranscoderAttributionBuilder:
    """
    Builds attribution graphs using pre-trained transcoders.

    Based on the methodology from Anthropic's attribution graph paper:
    - Compute gradients of output logits w.r.t. feature activations
    - Edge weight = activation x gradient (attribution)
    - Prune to keep top-k edges per node
    """

    def __init__(
        self,
        model: ModelWrapper,
        transcoder_set: TranscoderSet,
        extracted_features: Dict[int, Dict],
        position_map: List[Dict],
        prompts: List[Dict],              # required for margins
        top_k_edges: int = 10,
        attribution_threshold: float = 0.01,
        layers: Optional[List[int]] = None,
    ):
        """
        Initialize correlation-based attribution builder.
        
        Device is auto-detected from model parameters (no need to specify).
        """
        self.model = model
        self.transcoder_set = transcoder_set
        self.position_map = position_map
        self.extracted_features = extracted_features
        self.prompts = prompts
        self.top_k_edges = top_k_edges
        self.attribution_threshold = attribution_threshold

        # Use only layers that exist in extracted_features
        if layers is None:
            self.layers = sorted(list(extracted_features.keys()))
        else:
            self.layers = [l for l in layers if l in extracted_features]

        # Reverse map prompt_idx -> sample_indices
        self.prompt_to_samples = build_prompt_to_samples_map(position_map)

        # Compute margins for ALL prompts (robust indexing by prompt_idx)
        logger.info("Computing margins for correlation attribution (all prompts)...")
        self.margins = compute_margin_batch(model, prompts)

        # quick sanity: position_map prompt_idx must be valid
        max_prompt = max(e["prompt_idx"] for e in position_map) if len(position_map) else -1
        if max_prompt >= len(prompts):
            raise ValueError(
                f"position_map refers to prompt_idx={max_prompt}, but prompts has len={len(prompts)}."
            )

        logger.info(f"Builder initialized. layers={self.layers}")
        logger.info(f"position_map samples={len(position_map)}, prompts_in_map={len(self.prompt_to_samples)}")


    def collect_feature_activations(self, prompt_idx: int) -> Dict[int, Dict[int, float]]:
        """
        Returns sparse activations for the DECISION token of this prompt:
          {layer: {feature_idx: activation}}
        """
        if prompt_idx not in self.prompt_to_samples:
            return {}

        sample_indices_all = self.prompt_to_samples[prompt_idx]
        decision_samples = [
            s for s in sample_indices_all
            if self.position_map[s].get("is_decision_position", False)
        ]
        sample_indices = decision_samples if decision_samples else sample_indices_all

        # CRITICAL: For correlation attribution, we need exactly 1 decision sample per prompt
        # Otherwise averaging semantics become ambiguous (average over samples vs over top-k appearances)
        if len(sample_indices) != 1:
            raise ValueError(
                f"Expected exactly 1 decision sample for prompt {prompt_idx}, got {len(sample_indices)}. "
                f"decision_samples={len(decision_samples)}, all_samples={len(sample_indices_all)}. "
                f"Check position_map is_decision_position / token_positions."
            )

        activations = {}

        for layer in self.layers:
            layer_data = self.extracted_features[layer]
            topk_idx = layer_data["top_k_indices"][sample_indices]   # (1, K)
            topk_val = layer_data["top_k_values"][sample_indices]    # (1, K)

            layer_act = {}
            # Since we have exactly 1 sample, no averaging needed
            for feat_idx, feat_val in zip(topk_idx[0].tolist(), topk_val[0].tolist()):
                layer_act[int(feat_idx)] = float(feat_val)

            activations[layer] = layer_act

        return activations

    def aggregate_graphs(
        self,
        prompts: List[Dict],
        n_prompts: int = 20,
        min_frequency: float = 0.2,
        use_abs_corr: bool = True,
    ) -> nx.DiGraph:
        """
        Correlation-based attribution.

        For each feature f=(layer, idx), define x_p as activation on prompt p (0 if absent).
        Compute Pearson r(x, margin) over the selected prompts.

        Efficient sparse Pearson:
          sum_x, sum_x2, sum_xy computed only over nonzeros (zeros contribute nothing),
          sum_y, sum_y2 computed once globally.
          
        IMPORTANT: Features not in top-k for a prompt are treated as activation=0.
        This is correct since transcoders produce sparse activations.
        
        We treat features absent from top-k as zero; this approximates sparse 
        transcoder activations under top-k truncation.
        """
        N = min(n_prompts, len(prompts))
        
        # CRITICAL: Need at least 2 samples to compute correlation
        if N < 2:
            raise ValueError(
                f"Need at least 2 prompts to compute correlation (got N={N}). "
                f"Set --n_prompts >= 2."
            )
        
        prompt_indices = list(range(N))

        y = np.array([self.margins[p] for p in prompt_indices], dtype=np.float64)
        
        # CRITICAL: Log margins subset to debug var_y issues
        logger.info(
            f"Margins subset (N={N}): mean={y.mean():.4f}, std={y.std():.4f}, "
            f"min={y.min():.4f}, max={y.max():.4f}"
        )
        
        sum_y = float(y.sum())
        sum_y2 = float((y * y).sum())

        # stats[(layer, feat)] = {sum_x, sum_x2, sum_xy, count_nonzero}
        stats = defaultdict(lambda: {"sum_x": 0.0, "sum_x2": 0.0, "sum_xy": 0.0, "count": 0})

        logger.info(f"Correlation aggregation over N={N} prompts. min_frequency={min_frequency:.2f}")

        for p in tqdm(prompt_indices, desc="Collecting activations"):
            acts_by_layer = self.collect_feature_activations(p)
            yp = float(self.margins[p])

            for layer, layer_acts in acts_by_layer.items():
                for feat_idx, x in layer_acts.items():
                    key = (layer, int(feat_idx))
                    st = stats[key]
                    st["sum_x"] += x
                    st["sum_x2"] += x * x
                    st["sum_xy"] += x * yp
                    st["count"] += 1

        # Build aggregated graph
        G = nx.DiGraph()
        G.add_node("input", type="input")
        G.add_node("output_correct", type="output")
        G.add_node("output_incorrect", type="output")

        min_count = max(1, int(np.ceil(min_frequency * N)))
        logger.info(f"Feature filter: min_count_nonzero >= {min_count} (of N={N})")

        added = 0

        # Optional: keep per-layer top_k_edges only (more readable graphs)
        per_layer_candidates = defaultdict(list)

        for (layer, feat_idx), st in stats.items():
            c = st["count"]
            if c < min_count:
                continue

            sum_x = st["sum_x"]
            sum_x2 = st["sum_x2"]
            sum_xy = st["sum_xy"]

            # Pearson components (population-style; scaling cancels in r)
            var_x = sum_x2 - (sum_x * sum_x) / N
            var_y = sum_y2 - (sum_y * sum_y) / N
            if var_x <= 1e-12 or var_y <= 1e-12:
                continue

            cov_xy = sum_xy - (sum_x * sum_y) / N
            r = cov_xy / np.sqrt(var_x * var_y)

            score = abs(r) if use_abs_corr else r
            
            # CRITICAL: Distinguish two different means for semantic clarity
            # mean_given_present: average activation when feature is in top-k (c samples)
            # mean_all: average over all N prompts (missing = 0), consistent with correlation
            mean_act_given_present = sum_x / c
            mean_act_all = sum_x / N

            per_layer_candidates[layer].append(
                (score, r, feat_idx, c, mean_act_given_present, mean_act_all)
            )

        # Add top features per layer
        for layer, items in per_layer_candidates.items():
            items.sort(key=lambda t: t[0], reverse=True)
            items = items[: self.top_k_edges]

            for score, r, feat_idx, c, mean_act_present, mean_act_all in items:
                if abs(r) < self.attribution_threshold:
                    continue

                feat_id = f"L{layer}_F{feat_idx}"
                freq = c / N

                G.add_node(
                    feat_id,
                    type="feature",
                    layer=int(layer),
                    feature_idx=int(feat_idx),
                    count=int(c),
                    frequency=float(freq),
                    corr=float(r),
                    abs_corr=float(abs(r)),
                    mean_activation_given_present=float(mean_act_present),
                    mean_activation_all=float(mean_act_all),
                )

                # Edge weights: use corr (signed)
                # Positive r -> pushes toward output_correct, away from output_incorrect
                # Negative r -> pushes toward output_incorrect, away from output_correct
                w = float(r)
                G.add_edge("input", feat_id, weight=w)
                G.add_edge(feat_id, "output_correct", weight=w)
                G.add_edge(feat_id, "output_incorrect", weight=-w)  # Symmetric margin interpretation

                added += 1

        logger.info(f"Aggregated correlation graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, features_added={added}")
        if added == 0:
            logger.warning("No features survived thresholds. Consider lowering min_frequency or attribution_threshold.")
        return G

    def compute_virtual_weights(
        self,
        source_layer: int,
        target_layer: int,
    ) -> torch.Tensor:
        """
        Compute virtual weight matrix between features at adjacent layers.

        Virtual weight = W_enc_target @ W_dec_source.T

        This approximates the linear pathway from source features to target features.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index

        Returns:
            Virtual weight matrix (d_transcoder_target, d_transcoder_source)
        """
        source_tc = self.transcoder_set[source_layer]
        target_tc = self.transcoder_set[target_layer]

        with torch.no_grad():
            # Virtual weight: how source decoder directions project onto target encoder
            virtual_weight = target_tc.W_enc @ source_tc.W_dec.T

        return virtual_weight


def save_graph(G: nx.DiGraph, output_path: Path, name: str, metadata: Dict = None):
    """Save graph to multiple formats."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as GraphML
    graphml_path = output_path / f"{name}.graphml"
    nx.write_graphml(G, graphml_path)

    # Save as JSON
    json_path = output_path / f"{name}.json"
    graph_data = {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if k != "id"}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ],
        "metadata": metadata or {},
    }
    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(f"Saved graph: {graphml_path.name}, {json_path.name}")
    return graphml_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Build attribution graphs with transcoders")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--transcoder_config",
        type=str,
        default="configs/transcoder_config.yaml",
        help="Path to transcoder config",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        choices=["grammar_agreement"],
        default="grammar_agreement",
        help="Which behaviour to analyze (currently only grammar_agreement)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to use",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=20,
        help="Number of prompts for aggregation",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Model size (0.6b, 1.7b, 4b, 8b, 14b)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    tc_config = load_transcoder_config(args.transcoder_config)

    torch.manual_seed(config["seeds"]["torch_seed"])

    # Model size
    model_size = args.model_size or tc_config.get("model_size", "4b")

    # Layers
    if args.layers:
        layers = args.layers
    else:
        layers = tc_config.get("analysis_layers", {}).get("default", list(range(10, 26)))

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION (TRANSCODER-BASED)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model size: {model_size}")
    print(f"  Layers: {layers}")
    print(f"  Device: {device}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"  Prompts per behaviour: {args.n_prompts}")

    # Load transcoders
    print(f"\nLoading pre-trained transcoders...")
    transcoder_set = load_transcoder_set(
        model_size=model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=layers,
    )

    # Load language model
    print(f"\nLoading language model...")
    model_name = tc_config["transcoders"][model_size]["model_name"]
    
    # CRITICAL: Use BASE model to match transcoders and script 04
    # Transcoders are trained on base models, NOT instruct variants
    # Script 04 extracts features from base model
    # Attribution MUST use same model for consistency
    # DO NOT use Instruct-2507 or any fine-tuned variant!

    model = ModelWrapper(
        model_name=model_name,  # Use base model from transcoder config
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )

    # CRITICAL: Load pre-extracted features from script 04
    print(f"\nLoading pre-extracted features from script 04...")
    features_path = Path(config["paths"]["results"]) / "transcoder_features"
    
    # For grammar_agreement only (single behaviour)
    behaviour = behaviours[0]
    
    extracted_features = load_extracted_features(
        features_path,
        behaviour=behaviour,
        split=args.split,  # Use specified split (train/test)
        layers=layers,
    )
    
    position_map = load_position_map(
        features_path,
        behaviour=behaviour,
        split=args.split,  # Use specified split (train/test)
    )
    
    print(f"Loaded {len(position_map)} samples from {len(extracted_features)} layers")
    
    # CRITICAL SANITY CHECKS: Ensure data consistency
    n_pos = len(position_map)
    logger.info(f"Position map has {n_pos} samples")
    
    # Check each layer has same number of samples as position_map
    for layer_idx, layer_data in extracted_features.items():
        n_layer = layer_data["top_k_indices"].shape[0]
        if n_layer != n_pos:
            raise ValueError(
                f"Mismatch: position_map has {n_pos} samples, "
                f"but layer {layer_idx} has {n_layer} samples in top_k_indices. "
                f"This indicates mixed files from different runs."
            )
    
    # Check token_positions is consistent across layers
    tokpos = None
    for layer_idx, layer_data in extracted_features.items():
        meta = layer_data["metadata"]
        layer_tokpos = meta.get("token_positions", None)
        if tokpos is None:
            tokpos = layer_tokpos
        elif layer_tokpos != tokpos:
            raise ValueError(
                f"token_positions differs across layers: "
                f"expected {tokpos}, but layer {layer_idx} has {layer_tokpos}"
            )
    logger.info(f"✓ All layers use token_positions={tokpos}")
    
    # PATCH 4: Validate decision mode consistency
    if tokpos in ("decision", "last"):
        # CRITICAL: For correlation attribution, validate is_decision_position exists
        dec_samples = [i for i, e in enumerate(position_map) if e.get("is_decision_position", False)]
        n_prompts_in_map = len(set(e["prompt_idx"] for e in position_map))
        
        # Must have decision positions marked
        if len(dec_samples) == 0:
            raise ValueError(
                f"token_positions={tokpos!r} but no entries have is_decision_position=True. "
                f"This means position_map is inconsistent with extraction mode. "
                f"Regenerate features with script 04 or check extraction metadata."
            )
        
        dec_prompts = len(set(position_map[i]["prompt_idx"] for i in dec_samples))
        logger.info(f"Decision positions: {len(dec_samples)} samples covering {dec_prompts} prompts")
        
        # Must have exactly 1 decision sample per prompt (strict for correlation)
        if dec_prompts != n_prompts_in_map:
            raise ValueError(
                f"Decision samples cover {dec_prompts} prompts but position_map has {n_prompts_in_map} prompts. "
                f"Each prompt must have exactly 1 decision sample for correlation attribution. "
                f"Check script 04 extraction logic for token_positions='{tokpos}'."
            )
    
    # PATCH 6: Load prompts BEFORE builder (required for margins)
    prompt_path = Path(config["paths"]["prompts"])
    prompts = load_prompts(prompt_path, behaviour, args.split)
    
    # PATCH 5: Validate prompt indices
    max_prompt_in_map = max(entry["prompt_idx"] for entry in position_map)
    if max_prompt_in_map >= len(prompts):
        raise ValueError(
            f"position_map references prompt_idx {max_prompt_in_map}, "
            f"but only {len(prompts)} prompts loaded. "
            f"Mismatch between position_map and prompts file."
        )

    # Build attribution builder with pre-extracted features
    # Device is auto-detected from model (no need to pass it)
    builder = TranscoderAttributionBuilder(
        model=model,
        transcoder_set=transcoder_set,
        extracted_features=extracted_features,
        position_map=position_map,
        prompts=prompts,  # <-- ADDED for margins
        top_k_edges=config["attribution"]["top_k_edges"],
        attribution_threshold=config["attribution"]["attribution_threshold"],
        layers=layers,
    )

    # Process behaviours
    output_base = Path(config["paths"]["results"]) / "attribution_graphs"

    # PATCH 6: Prompts already loaded before builder, remove from loop
    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)
        
        print(f"Using {len(prompts)} prompts (already loaded)")

        # Build aggregated graph
        print(f"\nBuilding aggregated attribution graph...")
        G = builder.aggregate_graphs(
            prompts,
            n_prompts=args.n_prompts,
            min_frequency=tc_config["attribution"]["min_feature_frequency"],
        )

        # Save graph
        output_path = output_base / behaviour
        
        # Get topk size from extracted_features for metadata
        first_layer = list(extracted_features.keys())[0]
        topk_size = extracted_features[first_layer]["top_k_indices"].shape[1]
        
        metadata = {
            "behaviour": behaviour,
            "split": args.split,
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "layers": layers,
            "n_prompts": args.n_prompts,
            "timestamp": datetime.now().isoformat(),
            # Patch E: Attribution method metadata
            "attribution_method": "correlation",
            "feature_score": "abs_corr",
            "edge_weight": "corr",
            "missing_feature_activation": 0.0,
            # Patch 5: Top-k truncation metadata
            "activation_observation": "topk_truncated",
            "topk_per_token": int(topk_size),
        }
        save_graph(G, output_path, f"attribution_graph_{args.split}", metadata)

        # Print summary
        n_features = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "feature")
        n_edges = G.number_of_edges()

        print(f"\nGraph summary:")
        print(f"  Feature nodes: {n_features}")
        print(f" Edges: {n_edges}")

        # PATCH 7: Print top features by correlation (not avg_attribution)
        feature_rows = [
            (n, d.get("abs_corr", 0.0), d.get("corr", 0.0), d.get("frequency", 0.0))
            for n, d in G.nodes(data=True)
            if d.get("type") == "feature"
        ]
        feature_rows.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 features by |corr|:")
        for feat_id, abs_r, r, freq in feature_rows[:10]:
            print(f"  {feat_id}: corr={r:+.4f}, |corr|={abs_r:.4f}, freq={freq:.1%}")

        # Patch C: Only print what we actually save
        print(f"\nSaved graph to {output_path / f'attribution_graph_{args.split}.graphml'}")
    print("\n" + "=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_base.absolute()}")
    print("\nNext step: python scripts/07_run_interventions.py")


if __name__ == "__main__":
    main()
