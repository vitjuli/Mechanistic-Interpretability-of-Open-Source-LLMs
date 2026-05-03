"""
Build attribution graphs from transcoder features to model outputs.

This version uses pre-trained transcoders.
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
from typing import List, Dict, Optional, Tuple
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


def continuation_logprob(model: ModelWrapper, prompt_text: str, answer_text: str) -> float:
    """
    Compute log p(answer_text | prompt_text) using teacher forcing.
    """
    device = next(model.model.parameters()).device

    # Tokenize prompt and answer separately (no special tokens)
    prompt_ids = model.tokenizer.encode(prompt_text, add_special_tokens=False)
    ans_ids = model.tokenizer.encode(answer_text, add_special_tokens=False)

    if len(ans_ids) == 0:
        raise ValueError(f"Empty answer after tokenization: answer_text={answer_text!r}")

    # Build full input: [prompt_ids, ans_ids]
    input_ids = torch.tensor([prompt_ids + ans_ids], device=device)
    attn_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        out = model.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = out.logits  # (1, T, V)

    # We need probabilities of ans tokens at their positions.
    # Token at position t is predicted by logits at position t-1.
    start = len(prompt_ids)
    total_lp = 0.0
    for i, tok_id in enumerate(ans_ids):
        pos = start + i
        prev_pos = pos - 1
        log_probs = torch.log_softmax(logits[0, prev_pos, :], dim=-1)
        total_lp += float(log_probs[tok_id].item())

    return total_lp


def compute_margin_batch(model: ModelWrapper, prompts: List[Dict]) -> np.ndarray:
    """
    Δ_p = log p(y⁺|c) - log p(y⁻|c) for each prompt (multi-token answers supported).
    """
    margins = []
    logger.info(f"Computing margins for {len(prompts)} prompts on device={next(model.model.parameters()).device}...")

    for prompt_data in tqdm(prompts, desc="Computing margins"):
        prompt_text = prompt_data["prompt"]

        y_pos = prompt_data.get("correct_answer") or prompt_data.get("answer_matching")
        y_neg = prompt_data.get("incorrect_answer") or prompt_data.get("answer_not_matching")

        if not y_pos or not y_neg:
            raise ValueError(
                f"Prompt missing answer fields. Found keys: {list(prompt_data.keys())}"
            )

        lp_pos = continuation_logprob(model, prompt_text, y_pos)
        lp_neg = continuation_logprob(model, prompt_text, y_neg)

        margins.append(lp_pos - lp_neg)

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
        Initialize union/beta attribution builder.
        
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
        logger.info("Computing margins...")
        self.margins = compute_margin_batch(model, prompts)

        # quick sanity: position_map prompt_idx must be valid
        max_prompt = max(e["prompt_idx"] for e in position_map) if len(position_map) else -1
        if max_prompt >= len(prompts):
            raise ValueError(
                f"position_map refers to prompt_idx={max_prompt}, but prompts has len={len(prompts)}."
            )

        # CRITICAL: Validate exactly 1 decision sample per prompt
        for p_idx, s_indices in self.prompt_to_samples.items():
            dec_samples = [s for s in s_indices if self.position_map[s].get("is_decision_position", False)]
            if len(dec_samples) != 1:
                n_tok = len(self.position_map[s_indices[0]].get("token_positions", [])) if s_indices else "?"
                logger.warning(
                    f"Prompt {p_idx} has {len(dec_samples)} decision samples (expected 1). "
                    f"Total samples={len(s_indices)}. "
                    f"Check extraction logic or 'is_decision_position' flag."
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
            # PERFORMANCE: Use numpy instead of .tolist() to avoid GPU→CPU sync overhead
            # This is critical in loops over prompts × layers × features
            idx = topk_idx[0].cpu().numpy() if topk_idx.is_cuda else topk_idx[0].numpy()
            val = topk_val[0].cpu().numpy() if topk_val.is_cuda else topk_val[0].numpy()
            
            for feat_idx, feat_val in zip(idx, val):
                layer_act[int(feat_idx)] = float(feat_val)

            activations[layer] = layer_act

        return activations

    def top_features_for_prompt(
        self, 
        prompt_idx: int, 
        beta: Dict[Tuple[int, int], float],
        k: int = 20,
        use_abs: bool = True,
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get top-k features for specific prompt using activation * beta scoring.
        
        This is a correlation-based LINEAR PROXY for attribution, not causal.
        The score combines:
        - Feature activation on this prompt (how active is feature f?)
        - Beta coefficient (how predictive is feature f globally?)
        
        This gives a prompt-specific importance score that can differ across prompts.
        
        Args:
            prompt_idx: Index of prompt
            beta: Dict mapping (layer, feat) -> regression coefficient
            k: Number of top features to return
            use_abs: If True, sort by |score|; if False, sort by signed score (largest positive first)
        
        Returns:
            List of ((layer, feat_idx), score) sorted by |score|
            
        Note:
            This is NOT causal attribution. Use interventions (ablation/patching)
            to validate which features are actually causal for this prompt.
        """
        acts = self.collect_feature_activations(prompt_idx)
        scores = []
        
        for layer, feats in acts.items():
            for feat, x in feats.items():
                b = beta.get((layer, feat), 0.0)
                score = x * b
                scores.append(((layer, feat), score))
        
        # If use_abs: sort by |score| descending (most important by magnitude).
        # Else: sort by signed score descending (largest positive first, then negative).
        if use_abs:
            return sorted(scores, key=lambda t: abs(t[1]), reverse=True)[:k]
        else:
            return sorted(scores, key=lambda t: t[1], reverse=True)[:k]

    def compute_beta(self, prompt_indices: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Compute beta coefficients for prompt-specific scoring.
        
        Beta = Cov(x, y) / Var(x) using missing=0 model (consistent with correlation).
        
        IMPORTANT: This assumes top-k truncation: features not in top-k have x=0.
        This is a LINEAR PROXY for attribution, not causal. The formulas are:
        - Var(x) = E[x²] - E[x]² where E is over all N prompts (missing → 0)
        - Cov(x,y) = E[xy] - E[x]E[y] where missing x → 0 (so xy term is only from nonzero)
        
        Performance: sum_x, sum_x2, sum_xy accumulate only over nonzero (zeros don't contribute),
        but division by N treats all prompts equally (missing=0 assumption).
        
        Args:
            prompt_indices: List of prompt indices to compute beta over
        
        Returns:
            Dict mapping (layer, feat_idx) -> beta coefficient
        """
        N = len(prompt_indices)
        if N < 2:
            raise ValueError(f"Need at least 2 prompts to compute beta, got {N}")
        
        y = np.array([self.margins[p] for p in prompt_indices], dtype=np.float64)
        sum_y = float(y.sum())
        # Note: sum_y2 not needed for beta = Cov(x,y) / Var(x)
        
        stats = defaultdict(lambda: {"sum_x": 0.0, "sum_x2": 0.0, "sum_xy": 0.0, "count": 0})
        
        for p in prompt_indices:
            acts_by_layer = self.collect_feature_activations(p)
            yp = float(self.margins[p])
            for layer, layer_acts in acts_by_layer.items():
                for feat_idx, x in layer_acts.items():
                    st = stats[(layer, int(feat_idx))]
                    st["sum_x"] += x
                    st["sum_x2"] += x * x
                    st["sum_xy"] += x * yp
                    st["count"] += 1
        
        beta = {}
        for (layer, feat_idx), st in stats.items():
            if st["count"] < 2:
                continue
            sum_x, sum_x2, sum_xy = st["sum_x"], st["sum_x2"], st["sum_xy"]
            # CRITICAL: Use global N (missing=0 model) - consistent with correlation
            var_x = sum_x2 - (sum_x * sum_x) / N
            cov_xy = sum_xy - (sum_x * sum_y) / N
            beta[(layer, feat_idx)] = (cov_xy / var_x) if var_x > 1e-12 else 0.0
        
        logger.info(f"Computed beta for {len(beta)} features over {N} prompts (missing=0 model)")
        return beta

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
        # Correlation aggregation: compute Pearson r between feature activations and margins
        # Current implementation: GLOBAL graph (top features by overall correlation)
        # Future: Phase C will use union of per-prompt top-k for more diversity
        
        # CRITICAL: Get actual prompt indices first, then set N
        # This ensures N matches len(prompt_indices) for variance/covariance formulas
        available_prompts = sorted(list(self.prompt_to_samples.keys()))
        prompt_indices = available_prompts[: min(n_prompts, len(available_prompts))]
        
        N = len(prompt_indices)
        if N < 2:
            raise ValueError(
                f"Need at least 2 prompts to compute correlation (got N={N}). "
                f"Check that position_map has sufficient prompts."
            )
        
        logger.info(f"Using {N} prompts: indices {prompt_indices[:5]}... (first 5)")

        y = np.array([self.margins[p] for p in prompt_indices], dtype=np.float64)
        
        # CRITICAL: Log margins subset to debug var_y issues
        logger.info(
            f"Margins subset (N={N}): mean={y.mean():.4f}, std={y.std():.4f}, "
            f"min={y.min():.4f}, max={y.max():.4f}"
        )
        
        sum_y = float(y.sum())
        sum_y2 = float((y * y).sum())

        # stats[(layer, feat)] = {sum_x, sum_x2, sum_xy, count_nonzero}
        # CRITICAL: Missing features treated as x=0 (consistent with correlation)
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

        # Compute beta coefficients for prompt-specific scoring
        # CRITICAL: Use global N (treating missing features as x=0)
        # This is consistent with correlation computation below
        beta = {}
        for (layer, feat_idx), st in stats.items():
            if st["count"] < 2:  # Need at least 2 prompts where feature is active
                continue
            
            sum_x = st["sum_x"]
            sum_x2 = st["sum_x2"]
            sum_xy = st["sum_xy"]
            
            # Variance and covariance using global N (missing = 0 model)
            var_x = sum_x2 - (sum_x * sum_x) / N
            cov_xy = sum_xy - (sum_x * sum_y) / N
            
            if var_x > 1e-12:
                beta[(layer, feat_idx)] = cov_xy / var_x
            else:
                beta[(layer, feat_idx)] = 0.0
        
        logger.info(f"Computed beta for {len(beta)} features (using missing=0 model)")
        
        # Store beta in builder for prompt-specific scoring
        self._last_beta = beta  # Temporary storage for diagnostic access
        
        # Build aggregated graph
        G = nx.DiGraph()
        G.add_node("input", type="input")
        G.add_node("output_correct", type="output")
        G.add_node("output_incorrect", type="output")

        # CRITICAL: Respect min_frequency=0.0 from config (allows rare features)
        if min_frequency <= 0.0:
            min_count = 1  # Include all features that appear at least once
        else:
            min_count = max(1, int(np.ceil(min_frequency * N)))
        logger.info(f"Feature filter: min_count_nonzero >= {min_count} (of N={N}, min_frequency={min_frequency})")

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

    def aggregate_graphs_per_prompt_union(
        self,
        n_prompts: int = 20,
        k_per_prompt: int = 20,
        min_prompts: Optional[int] = None,
        max_frequency: float = 0.90,
        seed: int = 0,
        vw_threshold: Optional[float] = None,
    ) -> nx.DiGraph:
        """
        Build attribution graph from UNION of per-prompt top-k features.
        
        This is the TRUE prompt-specific graph:
        1. For each prompt, get top-k features (ranked by activation * beta)
        2. Take union of all these features
        3. Filter: keep only features appearing in >= min_prompts prompts
        4. Build graph with these features
        
        This captures prompt-specific diversity unlike global correlation graph.
        
        Args:
            n_prompts: Number of prompts to analyze
            k_per_prompt: Top-k features to extract per prompt
            min_prompts: Feature must appear in >= this many prompts to be included
            max_frequency: Drop features appearing in >= this fraction of prompts (always-on noise)
            seed: RNG seed for prompt shuffling (reproducible but unbiased selection)
        
        Returns:
            NetworkX graph with union of per-prompt features

        .. deprecated::
            Use ``aggregate_graphs_role_aware()`` instead.
            This method uses β-proxy scoring (activation × β_k) which is an
            importance approximation. The role-aware method uses
            gradient × activation attribution (activation_approx_v1 VW edges)
            and is the canonical downstream graph for scripts 07+.
        """
        import warnings
        warnings.warn(
            "aggregate_graphs_per_prompt_union() is deprecated. "
            "Use aggregate_graphs_role_aware() with --graph_node_mode role_aware "
            "for gradient×activation attribution (attribution_approx_v1 VW edges). "
            "This β-proxy method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Seed-shuffled prompt selection: avoids dataset ordering bias
        rng = np.random.default_rng(seed)
        available_prompts = sorted(list(self.prompt_to_samples.keys()))
        rng.shuffle(available_prompts)
        prompt_indices = available_prompts[: min(n_prompts, len(available_prompts))]
        
        N = len(prompt_indices)
        if N < 2:
            raise ValueError(f"Need at least 2 prompts, got {N}")
        
        # Adaptive min_prompts: avoid empty graphs for small N
        if min_prompts is None:
            min_prompts = 1 if N <= 5 else max(1, int(np.ceil(0.1 * N)))
        
        logger.info(f"Building per-prompt union graph: N={N}, k_per_prompt={k_per_prompt}, min_prompts={min_prompts}")
        
        # Compute beta coefficients
        beta = self.compute_beta(prompt_indices)
        
        # Collect per-prompt top-k features
        feature_scores = defaultdict(list)  # (layer, feat) -> [score per prompt where it's top-k]
        
        for p in tqdm(prompt_indices, desc="Collecting per-prompt top-k"):
            topk = self.top_features_for_prompt(p, beta, k=k_per_prompt)
            for (layer, feat), score in topk:
                feature_scores[(layer, feat)].append(float(score))
        
        # Build graph from union
        G = nx.DiGraph()
        G.add_node("input", type="input")
        G.add_node("output_correct", type="output")
        G.add_node("output_incorrect", type="output")
        
        added = 0
        skipped_rare = 0
        skipped_always_on = 0
        included_features: set = set()
        for (layer, feat), scores in feature_scores.items():
            n_seen = len(scores)
            freq = n_seen / N
            
            # Filter 1: must appear in enough prompts
            if n_seen < min_prompts:
                skipped_rare += 1
                continue
            
            # Filter 2: drop always-on features (background/positional noise)
            if freq >= max_frequency:
                skipped_always_on += 1
                continue
            
            # Conditional mean (over prompts where feature in top-k)
            mean_score_given = float(np.mean(scores))
            mean_abs_given = float(np.mean([abs(s) for s in scores]))
            std_score = float(np.std(scores))
            
            # Missing=0 mean (over all N prompts, treating absent as 0)
            mean_score_missing0 = float(np.sum(scores) / N)
            mean_abs_missing0 = float(np.sum([abs(s) for s in scores]) / N)
            
            # Specificity score: penalises high-frequency features
            # Features active on many prompts are less discriminative
            specific_score = mean_abs_given * (1.0 - freq)
            
            feat_id = f"L{layer}_F{feat}"
            
            G.add_node(
                feat_id,
                type="feature",
                layer=int(layer),
                feature_idx=int(feat),
                n_prompts=int(n_seen),
                frequency=float(freq),
                # Conditional means (over prompts where in top-k)
                mean_score_conditional=mean_score_given,
                mean_abs_score_conditional=mean_abs_given,
                std_score=std_score,
                # Missing=0 means (over all N prompts - frequency-penalized)
                mean_score_missing0=mean_score_missing0,
                mean_abs_missing0=mean_abs_missing0,
                # Specificity: high score + low frequency = most prompt-specific
                specific_score=float(specific_score),
                # Store beta for stable sign-based ablations downstream (script 07)
                beta=float(beta.get((layer, feat), 0.0)),
                beta_sign=int(1 if beta.get((layer, feat), 0.0) > 0 else (-1 if beta.get((layer, feat), 0.0) < 0 else 0)),
            )
            
            # Edge weight: use conditional mean_score (signed)
            # Positive -> promotes output_correct
            # Negative -> promotes output_incorrect
            w = mean_score_given
            G.add_edge("input", feat_id, weight=w)
            G.add_edge(feat_id, "output_correct", weight=w)
            G.add_edge(feat_id, "output_incorrect", weight=-w)
            
            added += 1
            included_features.add((layer, feat))

        logger.info(
            f"Per-prompt union graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
            f"features={added}, total_candidates={len(feature_scores)}, "
            f"skipped_rare={skipped_rare}, skipped_always_on={skipped_always_on}"
        )
        
        if added == 0:
            logger.warning(f"No features appeared in >= {min_prompts} prompts. Try lowering min_prompts.")
        
        # Store union params in graph for accurate metadata and reproducibility
        G.graph["union_params"] = {
            "prompt_indices": [int(p) for p in prompt_indices],
            "N": int(N),
            "k_per_prompt": int(k_per_prompt),
            "min_prompts": int(min_prompts),
            "max_frequency": float(max_frequency),
            "seed": int(seed),
        }

        # Phase 2: Add virtual-weight inter-feature edges
        # Virtual weight W_vw[tgt, src] = W_enc_tgt[tgt, :] · W_dec_src[src, :]
        # approximates the linear influence of source feature src (at src_layer)
        # on target feature tgt (at tgt_layer) through the residual stream.
        # Only adjacent layer pairs are connected (nearest-layer DAG).
        if vw_threshold is not None:
            graph_layers = sorted(set(layer for (layer, _) in included_features))
            n_vw = 0
            for i in range(len(graph_layers) - 1):
                src_layer = graph_layers[i]
                tgt_layer = graph_layers[i + 1]
                src_feats = sorted(f for (l, f) in included_features if l == src_layer)
                tgt_feats = sorted(f for (l, f) in included_features if l == tgt_layer)
                if not src_feats or not tgt_feats:
                    continue
                source_tc = self.transcoder_set[src_layer]
                target_tc = self.transcoder_set[tgt_layer]
                with torch.no_grad():
                    # Submatrix only — never materialise the full (d_tc × d_tc) product
                    W_dec_src = source_tc.W_dec[src_feats, :].float()   # (n_src, d_model)
                    W_enc_tgt = target_tc.W_enc[tgt_feats, :].float()   # (n_tgt, d_model)
                    vw_sub = (W_enc_tgt @ W_dec_src.T).cpu()            # (n_tgt, n_src)
                for ti, tgt_feat in enumerate(tgt_feats):
                    for si, src_feat in enumerate(src_feats):
                        w = float(vw_sub[ti, si])
                        if abs(w) >= vw_threshold:
                            G.add_edge(
                                f"L{src_layer}_F{src_feat}",
                                f"L{tgt_layer}_F{tgt_feat}",
                                weight=w,
                                edge_type="virtual_weight",
                            )
                            n_vw += 1
            G.graph["union_params"]["vw_threshold"] = float(vw_threshold)
            G.graph["union_params"]["n_vw_edges"] = int(n_vw)
            logger.info(
                f"Virtual-weight edges: {n_vw} added "
                f"(threshold={vw_threshold}, layers={graph_layers})"
            )

        return G

    # ─── Phase 3: Role-Aware Methods ────────────────────────────────────────

    def compute_per_prompt_gradient_attribution(
        self, prompt_idx: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Gradient × activation attribution: α_k^ℓ = a_k^ℓ × (∂Δlogit / ∂a_k^ℓ)

        Uses the chain rule through transcoder decoder weights:
            ∂Δlogit/∂a_k^ℓ ≈ (∂Δlogit/∂MLP_output^ℓ) · W_dec[k, :]

        Captures ∂Δlogit/∂MLP_output via register_full_backward_hook on each
        block.mlp during a gradient-enabled forward pass. Saved top-k activations
        (from extracted_features) provide a_k values without re-running the model.

        Falls back to raw activations (no gradient info) for multi-token answers.

        Returns:
            Dict mapping (layer, feature_idx) -> attribution_score (a × grad)
        """
        prompt_data = self.prompts[prompt_idx]
        prompt_text = prompt_data["prompt"]
        y_pos = prompt_data.get("correct_answer") or prompt_data.get("answer_matching")
        y_neg = prompt_data.get("incorrect_answer") or prompt_data.get("answer_not_matching")

        model_hf = self.model.model
        device = next(model_hf.parameters()).device

        # Tokenize
        prompt_ids = self.model.tokenizer.encode(prompt_text, add_special_tokens=True)
        pos_ids = self.model.tokenizer.encode(y_pos, add_special_tokens=False)
        neg_ids = self.model.tokenizer.encode(y_neg, add_special_tokens=False)

        # Only single-token answers support gradient flow through a single logit difference
        if len(pos_ids) != 1 or len(neg_ids) != 1:
            acts = self.collect_feature_activations(prompt_idx)
            return {(layer, feat): val for layer, d in acts.items() for feat, val in d.items()}

        pos_tok = int(pos_ids[0])
        neg_tok = int(neg_ids[0])

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        # Register backward hooks to capture ∂Δlogit/∂MLP_output at last token position
        mlp_output_grads: Dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in self.layers:
            try:
                block = model_hf.model.layers[layer_idx]
            except (AttributeError, IndexError):
                continue

            def make_hook(l):
                def hook_fn(module, grad_in, grad_out):
                    # grad_out[0]: (1, T, d_model) — gradient w.r.t. MLP output
                    if grad_out[0] is not None:
                        mlp_output_grads[l] = grad_out[0][0, -1, :].detach().float()
                return hook_fn

            h = block.mlp.register_full_backward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            # Forward pass with gradients (no torch.no_grad context)
            out = model_hf(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[0, -1, :]  # (V,)
            delta_logit = logits[pos_tok] - logits[neg_tok]
            delta_logit.backward()
        except Exception as e:
            logger.warning(f"Gradient attribution failed for prompt {prompt_idx}: {e}. Using activations only.")
            for h in hooks:
                h.remove()
            acts = self.collect_feature_activations(prompt_idx)
            return {(layer, feat): val for layer, d in acts.items() for feat, val in d.items()}
        finally:
            for h in hooks:
                h.remove()
            # Clear gradients to avoid accumulation across calls
            model_hf.zero_grad(set_to_none=True)

        # Get saved top-k activations for this prompt
        acts = self.collect_feature_activations(prompt_idx)  # {layer: {feat: activation}}

        attributions: Dict[Tuple[int, int], float] = {}

        for layer in self.layers:
            layer_acts = acts.get(layer, {})
            if not layer_acts:
                continue

            if layer not in mlp_output_grads:
                # No gradient captured; fall back to activation magnitude
                for feat, val in layer_acts.items():
                    attributions[(layer, feat)] = val
                continue

            grad = mlp_output_grads[layer]  # (d_model,) float32 on CPU after detach
            tc = self.transcoder_set[layer]

            feat_indices = list(layer_acts.keys())
            feat_vals = torch.tensor(
                [layer_acts[f] for f in feat_indices], dtype=torch.float32
            )

            # W_dec: (d_tc, d_model); slice rows for our top-k features
            with torch.no_grad():
                W_dec_sub = tc.W_dec[feat_indices, :].float().cpu()  # (n_feats, d_model)

            grad_cpu = grad.cpu()
            # ∂Δlogit/∂a_k ≈ grad_mlp_out · W_dec[k, :]
            dot_products = (W_dec_sub @ grad_cpu.unsqueeze(-1)).squeeze(-1)  # (n_feats,)
            attr_scores = feat_vals * dot_products  # a_k × grad_k

            for feat_idx, score in zip(feat_indices, attr_scores.tolist()):
                attributions[(layer, feat_idx)] = float(score)

        return attributions

    def _find_content_word_samples(self, prompt_idx: int) -> Tuple[List[int], str]:
        """
        Find position_map sample indices for content-word tokens in this prompt.

        Primary: regex r'"([^"]+)"' to find quoted span in prompt text.
        Fallback: text.find using prompt["word"].

        Returns:
            (content_sample_indices, method)
            method: "regex" | "find" | "none"
        """
        import re as _re

        if prompt_idx not in self.prompt_to_samples:
            return [], "none"

        prompt = self.prompts[prompt_idx]
        # Support both 'prompt' key (multilingual_circuits) and 'text' key (other behaviours)
        text = prompt.get("prompt", prompt.get("text", ""))

        # Primary: regex quoted span
        m = _re.search(r'"([^"]+)"', text)
        if m:
            span_start = m.start(1)
            span_end = m.end(1)
            method = "regex"
        else:
            # Fallback: text.find using 'word' key
            cw = prompt.get("word", "")
            if cw:
                idx = text.find(cw)
                if idx >= 0:
                    span_start = idx
                    span_end = idx + len(cw)
                    method = "find"
                else:
                    return [], "none"
            else:
                return [], "none"

        # Tokenize with offset_mapping to find content token IDs
        content_token_ids: set = set()
        cw = prompt.get("word", "")
        try:
            enc = self.model.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            for i, (cs, ce) in enumerate(enc["offset_mapping"]):
                if ce > span_start and cs < span_end:
                    content_token_ids.add(enc["input_ids"][i])
        except Exception:
            # Fallback: encode the content word directly (less precise but robust)
            if cw:
                try:
                    for variant in [cw, " " + cw]:
                        ids = self.model.tokenizer.encode(variant, add_special_tokens=False)
                        content_token_ids.update(ids)
                except Exception as e2:
                    logger.warning(f"Tokenizer fallback failed for prompt {prompt_idx}: {e2}")
                    return [], "none"

        if not content_token_ids:
            return [], "none"

        # Match against non-decision samples in position_map
        content_samples = [
            s for s in self.prompt_to_samples[prompt_idx]
            if (
                not self.position_map[s].get("is_decision_position", False)
                and self.position_map[s].get("token_id") in content_token_ids
            )
        ]
        return content_samples, method

    def collect_content_activations(
        self,
        prompt_idx: int,
        content_sample_indices: List[int],
    ) -> Dict[int, Dict[int, float]]:
        """
        Returns max activation over content-word subtokens for each feature.
        {layer: {feature_idx: max_activation}}
        Multiple subtokens for same feature: take max.
        """
        if not content_sample_indices:
            return {}

        activations: Dict[int, Dict[int, float]] = {}
        for layer in self.layers:
            layer_data = self.extracted_features[layer]
            layer_act: Dict[int, float] = {}
            for s in content_sample_indices:
                topk_idx = layer_data["top_k_indices"][s]
                topk_val = layer_data["top_k_values"][s]
                idx_arr = topk_idx.cpu().numpy() if topk_idx.is_cuda else topk_idx.numpy()
                val_arr = topk_val.cpu().numpy() if topk_val.is_cuda else topk_val.numpy()
                for fi, fv in zip(idx_arr, val_arr):
                    fi, fv = int(fi), float(fv)
                    if fi not in layer_act or fv > layer_act[fi]:
                        layer_act[fi] = fv
            activations[layer] = layer_act
        return activations

    def aggregate_graphs_role_aware(
        self,
        n_prompts: int = 20,
        k_per_prompt: int = 20,
        min_prompts: Optional[int] = None,
        max_frequency: float = 0.90,
        seed: int = 0,
        vw_threshold: Optional[float] = None,
        k_content: int = 10,
        min_lang_asym: float = 0.0,
        top_k_per_layer: Optional[int] = None,
        activation_weighted: bool = True,
    ) -> nx.DiGraph:
        """
        Build role-aware attribution graph: decision nodes + content-word nodes.

        Decision nodes: same selection as aggregate_graphs_per_prompt_union.
        Content nodes: features active at content-word token positions, selected by:
          Stage 1: activation existence (>=2 prompts; fallback >=1 if too few candidates)
          Stage 2: rank by |en_freq - fr_freq|; diversity constraint max 2 per layer
          VW connectivity check: must have >=1 VW edge above threshold to adjacent-layer
            decision node (connectivity_threshold = vw_threshold or 0.01).

        "Both" nodes (same (layer, feat) in decision and content sets) get flat-prefixed
        dual attributes: decision_*, content_*. Backward-compatible flat attrs use
        decision values.

        Content-only nodes have no output edges (upstream_candidate role only).
        """
        import re as _re

        # ── 0. Prompt selection ──────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        available_prompts = sorted(list(self.prompt_to_samples.keys()))
        rng.shuffle(available_prompts)
        prompt_indices = available_prompts[:min(n_prompts, len(available_prompts))]
        N = len(prompt_indices)
        if N < 2:
            raise ValueError(f"Need at least 2 prompts, got {N}")

        if min_prompts is None:
            min_prompts = 1 if N <= 5 else max(1, int(np.ceil(0.1 * N)))

        logger.info(
            f"Role-aware graph: N={N}, k_per_prompt={k_per_prompt}, "
            f"min_prompts={min_prompts}, k_content={k_content}, "
            f"vw_threshold={vw_threshold}"
        )

        # ── 1. Language assignment ───────────────────────────────────────────
        en_prompt_set = [p for p in prompt_indices
                         if self.prompts[p].get("language", "en") == "en"]
        fr_prompt_set = [p for p in prompt_indices
                         if self.prompts[p].get("language", "fr") == "fr"]
        n_en = len(en_prompt_set)
        n_fr = len(fr_prompt_set)
        logger.info(f"Language split: EN={n_en}, FR={n_fr}")

        # ── 2. Decision graph (gradient × activation attribution) ───────────
        # Per-prompt top-k; track per-language counts for "both" node attrs
        feature_scores: Dict[Tuple[int, int], list] = defaultdict(list)
        feature_en_count: Dict[Tuple[int, int], int] = defaultdict(int)
        feature_fr_count: Dict[Tuple[int, int], int] = defaultdict(int)
        # Raw activation values per feature — used for activation-weighted VW edges.
        # Collected alongside attribution scores so we only call the model once per prompt.
        feature_raw_acts: Dict[Tuple[int, int], list] = defaultdict(list)

        for p in tqdm(prompt_indices, desc="Decision: per-prompt gradient attribution"):
            lang = self.prompts[p].get("language", "en")
            # gradient × activation attribution (α_k = a_k × ∂Δlogit/∂a_k)
            attr_scores = self.compute_per_prompt_gradient_attribution(p)
            # Raw activations from stored top_k data (cheap numpy read, no model inference)
            raw_acts_p = self.collect_feature_activations(p)
            # Take top-k by |attribution score|
            topk = sorted(attr_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k_per_prompt]
            for (layer, feat), score in topk:
                feature_scores[(layer, feat)].append(float(score))
                if lang == "en":
                    feature_en_count[(layer, feat)] += 1
                else:
                    feature_fr_count[(layer, feat)] += 1
                # Collect raw activation value if available at this token position
                if layer in raw_acts_p and feat in raw_acts_p[layer]:
                    feature_raw_acts[(layer, feat)].append(abs(raw_acts_p[layer][feat]))

        G = nx.DiGraph()
        G.add_node("input", type="input")
        G.add_node("output_correct", type="output")
        G.add_node("output_incorrect", type="output")

        decision_added = 0
        skipped_rare = 0
        skipped_always_on = 0
        included_features: set = set()
        decision_attrs: Dict[Tuple[int, int], dict] = {}

        for (layer, feat), scores in feature_scores.items():
            n_seen = len(scores)
            freq = n_seen / N
            if n_seen < min_prompts:
                skipped_rare += 1
                continue
            if freq >= max_frequency:
                skipped_always_on += 1
                continue

            mean_score_given = float(np.mean(scores))
            mean_abs_given = float(np.mean([abs(s) for s in scores]))
            std_score = float(np.std(scores))
            mean_score_missing0 = float(np.sum(scores) / N)
            mean_abs_missing0 = float(np.sum([abs(s) for s in scores]) / N)
            specific_score = mean_abs_given * (1.0 - freq)
            grad_attr_sign = int(1 if mean_score_given > 0 else (-1 if mean_score_given < 0 else 0))
            # Mean raw activation (conditional on feature being active)
            raw_act_vals = feature_raw_acts.get((layer, feat), [])
            mean_act_cond = float(np.mean(raw_act_vals)) if raw_act_vals else 0.0

            attrs = dict(
                type="feature",
                layer=int(layer),
                feature_idx=int(feat),
                n_prompts=int(n_seen),
                frequency=float(freq),
                # Gradient attribution fields (primary)
                mean_grad_attr_conditional=mean_score_given,
                mean_abs_grad_attr_conditional=mean_abs_given,
                grad_attr_sign=grad_attr_sign,
                # Raw activation (used for activation-weighted VW edges)
                mean_activation_conditional=mean_act_cond,
                # Legacy-compatible aliases (for backward compat with script 07/08)
                mean_score_conditional=mean_score_given,
                mean_abs_score_conditional=mean_abs_given,
                specific_score=float(specific_score),
                std_score=std_score,
                mean_score_missing0=mean_score_missing0,
                mean_abs_missing0=mean_abs_missing0,
                position_role="decision",
                causal_status="output_attributed",
            )
            decision_attrs[(layer, feat)] = attrs
            feat_id = f"L{layer}_F{feat}"
            G.add_node(feat_id, **attrs)
            w = mean_score_given
            # Output edges: typed as "output_attribution" (gradient-based, not β-proxy)
            G.add_edge("input", feat_id, weight=w, edge_type="input_attribution")
            G.add_edge(feat_id, "output_correct", weight=w, edge_type="output_attribution")
            G.add_edge(feat_id, "output_incorrect", weight=-w, edge_type="output_attribution")
            decision_added += 1
            included_features.add((layer, feat))

        logger.info(
            f"Decision nodes: {decision_added} added "
            f"(skipped_rare={skipped_rare}, skipped_always_on={skipped_always_on})"
        )

        # ── 2b. Optional per-layer top-k pruning ─────────────────────────────
        # Remove nodes that don't make the top-k cut per layer.
        # Applied after the frequency/min_prompts filter so thresholds still apply;
        # top-k is an additional constraint that limits graph size.
        if top_k_per_layer is not None and top_k_per_layer > 0:
            from collections import defaultdict as _dd
            by_layer: dict = _dd(list)
            for (layer, feat) in list(included_features):
                score = abs(decision_attrs[(layer, feat)].get("mean_abs_grad_attr_conditional", 0.0))
                by_layer[layer].append(((layer, feat), score))
            removed_count = 0
            for layer, candidates in by_layer.items():
                candidates.sort(key=lambda x: x[1], reverse=True)
                for (lyr, ft), _ in candidates[top_k_per_layer:]:
                    feat_id = f"L{lyr}_F{ft}"
                    G.remove_node(feat_id)
                    included_features.discard((lyr, ft))
                    del decision_attrs[(lyr, ft)]
                    removed_count += 1
            logger.info(
                f"top_k_per_layer={top_k_per_layer}: removed {removed_count} nodes, "
                f"{len(included_features)} decision nodes remain"
            )

        # ── 3. Feature→feature attribution edges for decision nodes ─────────
        #
        # Edge weight formulation (activation_weighted=True, DEFAULT):
        #   edge(i→j) = mean_act_i × VW_static(i,j)
        #   where VW_static(i,j) = W_enc_{L+1}[j,:] · W_dec_L[:,i]
        #         mean_act_i = mean conditional activation of feature i across prompts
        #
        # This is labeled edge_type="attribution_approx_v1":
        #   APPROXIMATION — assumes transcoder encoder is locally linear (constant
        #   Jacobian). The true causal edge requires a per-prompt backward pass
        #   through the encoder nonlinearity, which is not computed here.
        #   Version label allows future upgrade to a prompt-local Jacobian ("_v2").
        #
        # activation_weighted=False uses the raw static VW weight (legacy behaviour,
        # edge_type="virtual_weight"). Kept for backward compatibility only.
        #
        # Both modes threshold on |w| >= vw_threshold. With activation weighting the
        # effective threshold is stricter for rarely-active features (their
        # mean_act ≈ 0 suppresses the edge weight even if VW_static is large).
        n_vw = 0
        if vw_threshold is not None:
            # Build mean activation lookup for source features
            mean_act_lookup: Dict[Tuple[int, int], float] = {}
            for (layer, feat) in included_features:
                raw_vals = feature_raw_acts.get((layer, feat), [])
                mean_act_lookup[(layer, feat)] = float(np.mean(raw_vals)) if raw_vals else 0.0

            graph_layers = sorted(set(layer for (layer, _) in included_features))
            for i in range(len(graph_layers) - 1):
                src_layer = graph_layers[i]
                tgt_layer = graph_layers[i + 1]
                src_feats = sorted(f for (l, f) in included_features if l == src_layer)
                tgt_feats = sorted(f for (l, f) in included_features if l == tgt_layer)
                if not src_feats or not tgt_feats:
                    continue
                source_tc = self.transcoder_set[src_layer]
                target_tc = self.transcoder_set[tgt_layer]
                with torch.no_grad():
                    W_dec_src = source_tc.W_dec[src_feats, :].float()   # (n_src, d_model)
                    W_enc_tgt = target_tc.W_enc[tgt_feats, :].float()   # (n_tgt, d_model)
                    vw_sub = (W_enc_tgt @ W_dec_src.T).cpu()            # (n_tgt, n_src)
                for ti, tgt_feat in enumerate(tgt_feats):
                    for si, src_feat in enumerate(src_feats):
                        vw_static = float(vw_sub[ti, si])
                        if activation_weighted:
                            mean_act = mean_act_lookup.get((src_layer, src_feat), 0.0)
                            w = mean_act * vw_static
                            etype = "attribution_approx_v1"
                        else:
                            w = vw_static
                            etype = "virtual_weight"
                        if abs(w) >= vw_threshold:
                            G.add_edge(
                                f"L{src_layer}_F{src_feat}",
                                f"L{tgt_layer}_F{tgt_feat}",
                                weight=w,
                                edge_type=etype,
                                vw_static=vw_static,  # always stored for reference
                            )
                            n_vw += 1
            logger.info(
                f"Decision attribution edges: {n_vw} added "
                f"(threshold={vw_threshold}, mode="
                f"{'activation_weighted' if activation_weighted else 'static_vw'})"
            )

        # ── 4. Content-word node analysis ────────────────────────────────────
        connectivity_threshold = vw_threshold if vw_threshold is not None else 0.01

        # 4a. Collect content activations per prompt
        content_feature_en: Dict[Tuple[int, int], int] = defaultdict(int)
        content_feature_fr: Dict[Tuple[int, int], int] = defaultdict(int)
        content_feature_acts: Dict[Tuple[int, int], list] = defaultdict(list)
        n_content_found = 0
        n_fallback_find = 0

        for p in tqdm(prompt_indices, desc="Content: per-prompt word detection"):
            content_samples, method = self._find_content_word_samples(p)
            if not content_samples:
                continue
            if method == "find":
                n_fallback_find += 1
            n_content_found += 1
            lang = self.prompts[p].get("language", "en")
            content_acts = self.collect_content_activations(p, content_samples)
            for layer, layer_act in content_acts.items():
                for feat, act in layer_act.items():
                    key = (layer, feat)
                    if lang == "en":
                        content_feature_en[key] += 1
                    else:
                        content_feature_fr[key] += 1
                    content_feature_acts[key].append(abs(act))

        logger.info(
            f"Content-word detection: {n_content_found}/{N} prompts found content samples "
            f"({n_fallback_find} used text.find fallback)"
        )

        # 4b. Stage 1 filter: activation existence (>=2 prompts; fallback >=1)
        min_content_prompts = 2
        stage1 = {
            k: v for k, v in content_feature_acts.items()
            if content_feature_en[k] + content_feature_fr[k] >= min_content_prompts
        }
        if len(stage1) < max(1, k_content // 2):
            min_content_prompts = 1
            stage1 = {
                k: v for k, v in content_feature_acts.items()
                if content_feature_en[k] + content_feature_fr[k] >= 1
            }
            logger.info(
                f"Stage 1 fallback: relaxed to min_content_prompts=1 "
                f"(candidates after fallback: {len(stage1)})"
            )
        else:
            logger.info(f"Stage 1: {len(stage1)} candidates (min_content_prompts={min_content_prompts})")

        # 4c. Stage 2: rank by lang_asym, diversity (max 2 per layer), VW check
        content_candidates = []
        for key, acts in stage1.items():
            layer, feat = key
            n_en_c = content_feature_en[key]
            n_fr_c = content_feature_fr[key]
            en_freq = n_en_c / n_en if n_en > 0 else 0.0
            fr_freq = n_fr_c / n_fr if n_fr > 0 else 0.0
            lang_asym = abs(en_freq - fr_freq)
            content_candidates.append({
                "key": key,
                "layer": layer,
                "feat": feat,
                "n_en": n_en_c,
                "n_fr": n_fr_c,
                "en_freq": en_freq,
                "fr_freq": fr_freq,
                "lang_asym": lang_asym,
                "mean_abs_act": float(np.mean(acts)),
            })

        content_candidates = [c for c in content_candidates if c["lang_asym"] >= min_lang_asym]
        content_candidates.sort(key=lambda x: x["lang_asym"], reverse=True)

        layer_content_count: Dict[int, int] = defaultdict(int)
        selected_content = []

        for c in content_candidates:
            if len(selected_content) >= k_content:
                break
            if layer_content_count[c["layer"]] >= 2:
                continue

            # VW connectivity check: must connect to adjacent-layer decision node
            layer, feat = c["layer"], c["feat"]
            connected = False
            for adj_layer in [layer - 1, layer + 1]:
                adj_dec_feats = [f for (l, f) in included_features if l == adj_layer]
                if not adj_dec_feats:
                    continue
                adj_tc = self.transcoder_set[adj_layer]
                curr_tc = self.transcoder_set[layer]
                with torch.no_grad():
                    if adj_layer < layer:
                        # adj is src, current is tgt
                        W_dec_adj = adj_tc.W_dec[adj_dec_feats, :].float()
                        W_enc_curr = curr_tc.W_enc[[feat], :].float()
                        vw_row = (W_enc_curr @ W_dec_adj.T).cpu()   # (1, n_adj)
                    else:
                        # current is src, adj is tgt
                        W_dec_curr = curr_tc.W_dec[[feat], :].float()
                        W_enc_adj = adj_tc.W_enc[adj_dec_feats, :].float()
                        vw_row = (W_enc_adj @ W_dec_curr.T).cpu()   # (n_adj, 1)
                if float(vw_row.abs().max()) >= connectivity_threshold:
                    connected = True
                    break

            if not connected:
                continue

            selected_content.append(c)
            layer_content_count[c["layer"]] += 1

        logger.info(
            f"Content nodes selected: {len(selected_content)} "
            f"(after Stage 2 + VW connectivity check)"
        )

        # 4d. Add content nodes to graph
        content_added = 0
        both_upgraded = 0

        for c in selected_content:
            layer, feat = c["layer"], c["feat"]
            feat_id = f"L{layer}_F{feat}"
            key = (layer, feat)

            if key in decision_attrs:
                # "Both" node: upgrade with flat-prefixed dual attrs
                d_en_freq = feature_en_count[key] / n_en if n_en > 0 else 0.0
                d_fr_freq = feature_fr_count[key] / n_fr if n_fr > 0 else 0.0
                upgrades = {
                    "position_role": "both",
                    "causal_status": "both",
                    "decision_n_prompts": decision_attrs[key]["n_prompts"],
                    "decision_en_freq": d_en_freq,
                    "decision_fr_freq": d_fr_freq,
                    "content_n_en_prompts": c["n_en"],
                    "content_n_fr_prompts": c["n_fr"],
                    "content_en_freq": c["en_freq"],
                    "content_fr_freq": c["fr_freq"],
                    "content_lang_asym": c["lang_asym"],
                    "content_mean_abs_act": c["mean_abs_act"],
                }
                for k_attr, v_attr in upgrades.items():
                    G.nodes[feat_id][k_attr] = v_attr
                both_upgraded += 1
            else:
                # Pure content node — no output edges
                G.add_node(
                    feat_id,
                    type="feature",
                    layer=int(layer),
                    feature_idx=int(feat),
                    position_role="content",
                    causal_status="upstream_candidate",
                    content_n_en_prompts=c["n_en"],
                    content_n_fr_prompts=c["n_fr"],
                    content_en_freq=c["en_freq"],
                    content_fr_freq=c["fr_freq"],
                    content_lang_asym=c["lang_asym"],
                    content_mean_abs_act=c["mean_abs_act"],
                )
                G.add_edge("input", feat_id, weight=c["lang_asym"])
                content_added += 1

        logger.info(
            f"Content nodes added: {content_added} new + {both_upgraded} upgraded to 'both'. "
            f"Total feature nodes: {sum(1 for _, d in G.nodes(data=True) if d.get('type')=='feature')}"
        )

        # ── 5. Store params in graph ─────────────────────────────────────────
        G.graph["union_params"] = {
            "prompt_indices": [int(p) for p in prompt_indices],
            "N": int(N),
            "k_per_prompt": int(k_per_prompt),
            "min_prompts": int(min_prompts),
            "max_frequency": float(max_frequency),
            "seed": int(seed),
            "graph_node_mode": "role_aware",
            "k_content": int(k_content),
            "min_lang_asym": float(min_lang_asym),
            "n_en_prompts": int(n_en),
            "n_fr_prompts": int(n_fr),
            "n_content_found": int(n_content_found),
            "n_fallback_find": int(n_fallback_find),
            "min_content_prompts_used": int(min_content_prompts),
            "vw_threshold": float(vw_threshold) if vw_threshold is not None else None,
            "n_vw_edges": int(n_vw),
            "connectivity_threshold": float(connectivity_threshold),
            "edge_method": (
                "activation_weighted_vw_linear_approx"
                if activation_weighted and vw_threshold is not None
                else "static_vw" if vw_threshold is not None
                else "star_only"
            ),
        }

        logger.info(
            f"Role-aware graph complete: nodes={G.number_of_nodes()}, "
            f"edges={G.number_of_edges()}, "
            f"decision={decision_added}, content={content_added}, both={both_upgraded}"
        )
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
    """Save graph as JSON only (GraphML disabled: doesn't support dict values)."""
    output_path.mkdir(parents=True, exist_ok=True)

    # GraphML disabled: nx.write_graphml fails on dict/list node attrs (e.g. union_params)
    # and requires lxml. JSON is simpler and fully supports all value types.

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
        "graph_attrs": dict(G.graph),  # includes union_params for reproducibility
    }
    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(f"Saved graph JSON: {json_path.name}")

    # Save metadata separately for easier access
    if metadata is not None:
        meta_path = output_path / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {meta_path.name}")

    return None, json_path


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
        default="grammar_agreement",
        help="Which behaviour to analyze",
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
    parser.add_argument(
        "--vw_threshold",
        type=float,
        default=None,
        help=(
            "If set, add virtual-weight edges between feature nodes at adjacent layers. "
            "Only edges with |weight| >= vw_threshold are kept. "
            "Use 0.0 to add all edges; None (default) disables virtual weights (star topology)."
        ),
    )
    parser.add_argument(
        "--graph_node_mode",
        type=str,
        default="decision_only",
        choices=["decision_only", "role_aware"],
        help=(
            "Node selection mode. 'decision_only' (default): same as before — "
            "features from decision token position only. "
            "'role_aware': adds content-word position features (Phase 3)."
        ),
    )
    parser.add_argument(
        "--k_content",
        type=int,
        default=10,
        help="(role_aware only) Max content-word nodes to add globally (default: 10).",
    )
    parser.add_argument(
        "--min_lang_asym",
        type=float,
        default=0.0,
        help=(
            "(role_aware only) Minimum |en_freq - fr_freq| for content node inclusion "
            "(default: 0.0 = no filter beyond Stage 1/2)."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix appended to output filename (e.g. '_roleaware').",
    )
    parser.add_argument(
        "--top_k_per_layer",
        type=int,
        default=None,
        help="(role_aware only) Keep top-k features per layer by mean_abs_grad_attr_conditional. "
             "Applied after frequency/min_prompts filters. None = keep all (default).",
    )
    parser.add_argument(
        "--activation_weighted",
        action="store_true",
        default=True,
        help="(role_aware only) Weight VW edges by mean source activation "
             "(attribution_approx_v1). Default: True. Pass --no_activation_weighted to use "
             "raw static VW edges (legacy virtual_weight mode).",
    )
    parser.add_argument(
        "--no_activation_weighted",
        dest="activation_weighted",
        action="store_false",
        help="Use raw static VW edges instead of activation-weighted edges (legacy mode).",
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
    
    # PATCH 4:    # Validate decision positions if in decision mode
    if tokpos == "decision":
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
    # NEW (leave windowed extraction; require decision-flag)
    if tokpos != "decision":
        logger.warning(
            f"token_positions={tokpos!r} (windowed extraction). "
            "OK: Script 06 will use is_decision_position=True per prompt."
        )
    dec_samples = [i for i, e in enumerate(position_map) if e.get("is_decision_position", False)]
    n_prompts_in_map = len(set(e["prompt_idx"] for e in position_map))
    dec_prompts = len(set(position_map[i]["prompt_idx"] for i in dec_samples))

    if len(dec_samples) == 0:
        raise ValueError(
            "No entries have is_decision_position=True in position_map. "
            "Script 06 requires decision positions to be marked."
        )

    # Check exactly ONE decision sample per prompt
    from collections import Counter
    cnt = Counter(position_map[i]["prompt_idx"] for i in dec_samples)
    bad = [p for p, c in cnt.items() if c != 1]
    if bad:
        raise ValueError(
            f"Some prompts have !=1 decision sample: e.g. {bad[:10]} (showing first 10). "
            f"Counts: {[cnt[p] for p in bad[:10]]}"
        )

    if dec_prompts != n_prompts_in_map:
        raise ValueError(
            f"Decision samples cover {dec_prompts} prompts but position_map has {n_prompts_in_map} prompts. "
            "Each prompt must have exactly 1 decision sample."
        )

    logger.info(
        f"✓ Validation passed: windowed extraction OK (samples={len(position_map)}, "
        f"prompts={n_prompts_in_map}, decision_samples={len(dec_samples)})"
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

    # DIAGNOSTIC: Test prompt-specific scoring on test data
    # This validates that different prompts get different top-k features
    if args.split == "test":
        logger.info("=" * 70)
        logger.info("DIAGNOSTIC: Testing prompt-specific scoring")
        logger.info("=" * 70)
        
        # We'll run aggregate_graphs once to compute beta, then test per-prompt scoring
        # Note: This is a temporary diagnostic approach for Phase A
        # In Phase C, we'll refactor to make beta computation a separate method
        pass  # Diagnostic code will be added after graph building loop

    # Process behaviours
    output_base = Path(config["paths"]["results"]) / "attribution_graphs"

    # PATCH 6: Prompts already loaded before builder, remove from loop
    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)
        
        # CRITICAL: Log actual prompts used (not loaded prompts)
        available_prompts = sorted(list(builder.prompt_to_samples.keys()))
        print(f"Prompts loaded: {len(prompts)} | prompts with extracted samples: {len(available_prompts)}")

        # Build attribution graph
        k_per_prompt = tc_config.get("features", {}).get("top_k_per_prompt", 20)
        max_frequency = tc_config.get("features", {}).get("max_frequency", 0.90)

        if args.graph_node_mode == "role_aware":
            print(f"\nBuilding role-aware attribution graph (Phase 3)...")
            G = builder.aggregate_graphs_role_aware(
                n_prompts=args.n_prompts,
                k_per_prompt=k_per_prompt,
                min_prompts=None,
                max_frequency=max_frequency,
                vw_threshold=args.vw_threshold,
                k_content=args.k_content,
                min_lang_asym=args.min_lang_asym,
                top_k_per_layer=args.top_k_per_layer,
                activation_weighted=args.activation_weighted,
            )
        else:
            print(f"\nBuilding per-prompt union attribution graph...")
            G = builder.aggregate_graphs_per_prompt_union(
                n_prompts=args.n_prompts,
                k_per_prompt=k_per_prompt,
                min_prompts=None,  # Auto-adaptive: 1 if N<=5 else 10% of N
                max_frequency=max_frequency,
                vw_threshold=args.vw_threshold,
            )

        # Save graph
        output_path = output_base / behaviour
        
        # STABILITY FIX: Get actual N from graph (not from len(prompts))
        # builder.prompt_to_samples may have fewer prompts than loaded
        union_params = G.graph.get("union_params", {})
        # Fallback should use actual available prompts, not len(prompts)
        available_prompts = sorted(list(builder.prompt_to_samples.keys()))
        n_fallback = min(args.n_prompts, len(available_prompts)) if args.n_prompts else len(available_prompts)
        n_used = int(union_params.get("N", n_fallback))
        suffix = args.output_suffix if args.output_suffix else ""
        name = f"attribution_graph_{args.split}_n{n_used}{suffix}"
        
        # Get topk size from extracted_features for metadata
        first_layer = list(extracted_features.keys())[0]
        topk_size = extracted_features[first_layer]["top_k_indices"].shape[1]
        
        # Get actual params from graph
        union_params = G.graph.get("union_params", {})
        actual_min_prompts = int(union_params.get("min_prompts", 1))
        actual_k_per_prompt = int(union_params.get("k_per_prompt", k_per_prompt))
        
        metadata = {
            "behaviour": behaviour,
            "split": args.split,
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "layers": layers,
            "n_prompts": args.n_prompts,
            "n_prompts_used": int(n_used),
            "timestamp": datetime.now().isoformat(),
            "graph_type": "per_prompt_union",
            "attribution_method": "linear_proxy",  # Not causal, correlation-based
            "scoring": {
                "per_prompt_score": "activation_times_beta",
                "beta_model": "missing_is_zero",
                "edge_weight": "mean_score_conditional",  # Uses mean_score_conditional from nodes
                "feature_rank": "mean_abs_score_conditional",
                "k_per_prompt": int(actual_k_per_prompt),
                "min_prompts": int(actual_min_prompts),
                "note": "Nodes have mean_score_conditional (over top-k prompts) and mean_score_missing0 (frequency-penalized)",
            },
            "activation_observation": "topk_truncated",
            "topk_per_token": int(topk_size),
            "vw_threshold": args.vw_threshold,
            "n_vw_edges": int(G.graph.get("union_params", {}).get("n_vw_edges", 0)),
            "graph_node_mode": args.graph_node_mode,
            "output_suffix": suffix,
        }
        save_graph(G, output_path, name, metadata)  # Use variable name with _nN suffix

        # Print summary
        n_features = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "feature")
        n_edges = G.number_of_edges()

        print(f"\nGraph summary:")
        print(f"  Edges: {n_edges}")
        print(f"  Feature nodes: {n_features}")
        print(f"  Layers: {sorted(set(d['layer'] for _, d in G.nodes(data=True) if d.get('type') == 'feature'))}")
        
        # DIAGNOSTIC: Test prompt-specific scoring (validation)
        if args.split == "test":
            print("\n" + "=" * 70)
            print("DIAGNOSTIC: Prompt-Specific Scoring Test")
            print("=" * 70)
            
            # Use EXACT same prompt_indices as graph (from union_params)
            # This guarantees diagnostic beta matches the graph's beta
            union_params = G.graph.get("union_params", {})
            prompt_indices = union_params.get("prompt_indices", [])
            if not prompt_indices:
                # Fallback: recompute with same seed
                rng = np.random.default_rng(union_params.get("seed", 0))
                avail = sorted(list(builder.prompt_to_samples.keys()))
                rng.shuffle(avail)
                prompt_indices = avail[:min(args.n_prompts, len(avail))]
            
            # Compute beta on SAME set as graph
            beta = builder.compute_beta(prompt_indices)
            
            # But show diversity on subset for readability
            n_test = min(3, len(prompt_indices))
            test_prompts = prompt_indices[:n_test]
            
            print(f"\nBeta computed on {len(prompt_indices)} prompts, showing diversity on first {n_test}...")
            print(f"Beta coefficients for {len(beta)} features")
            
            # Show top-k by |score| AND by signed score (to see positive vs negative drivers)
            for p in test_prompts:
                top_k_abs = builder.top_features_for_prompt(p, beta, k=10, use_abs=True)
                top_k_signed = builder.top_features_for_prompt(p, beta, k=10, use_abs=False)
                
                prompt_text = prompts[p]['prompt'][:50] if p < len(prompts) else "???"
                print(f"\nPrompt {p} ('{prompt_text}...') top-10 by |score|:")
                for (layer, feat), score in top_k_abs:
                    print(f"  L{layer}_F{feat}: score={score:.4f}")
                
                print(f"\n  Top-10 by signed score (positive drivers):")
                for (layer, feat), score in top_k_signed[:5]:
                    print(f"    L{layer}_F{feat}: score={score:+.4f}")
            
            if n_test > 1:
                print(f"\nDiversity Check (Jaccard Similarity):")
                
                # Compute pairwise Jaccard for BOTH abs and signed rankings
                # abs: overall importance; signed: direction-specific (correct vs incorrect)
                tops_abs = [

                    set(lf for lf, _ in builder.top_features_for_prompt(p, beta, k=10, use_abs=True))
                    for p in test_prompts
                ]
                tops_signed = [
                    set(lf for lf, _ in builder.top_features_for_prompt(p, beta, k=10, use_abs=False))
                    for p in test_prompts
                ]
                
                j_abs_scores, j_signed_scores = [], []
                for i in range(n_test):
                    for j in range(i+1, n_test):
                        inter_abs = len(tops_abs[i] & tops_abs[j])
                        union_abs = len(tops_abs[i] | tops_abs[j])
                        j_abs = inter_abs / max(1, union_abs)
                        j_abs_scores.append(j_abs)
                        
                        inter_signed = len(tops_signed[i] & tops_signed[j])
                        union_signed = len(tops_signed[i] | tops_signed[j])
                        j_signed = inter_signed / max(1, union_signed)
                        j_signed_scores.append(j_signed)
                        
                        print(f"  J(p{test_prompts[i]}, p{test_prompts[j]}): "
                              f"|score|={j_abs:.2%}, signed={j_signed:.2%}")
                
                avg_j_abs = float(np.mean(j_abs_scores)) if j_abs_scores else 0.0
                avg_j_signed = float(np.mean(j_signed_scores)) if j_signed_scores else 0.0
                print(f"\n  Average Jaccard: |score|={avg_j_abs:.2%}, signed={avg_j_signed:.2%}")
                
                if avg_j_abs < 0.3:
                    print("  ✓ EXCELLENT: Prompts have highly diverse top features (J < 30%)")
                elif avg_j_abs < 0.5:
                    print("  ✓ GOOD: Prompts show different top features (J < 50%)")
                elif avg_j_abs < 0.7:
                    print("  ~ MODERATE: Some overlap but still prompt-specific (J < 70%)")
                else:
                    print("  ⚠ WARNING: High overlap, prompt-specific scoring may need tuning (J >= 70%)")
            else:
                print(f"\n(Skipping diversity check for single prompt)")
            print("=" * 70)

        # Print top features by mean |score| (for union graph)
        feature_rows = [
            (n,
             d.get("specific_score", 0.0),
             d.get("mean_abs_score_conditional", 0.0),
             d.get("mean_score_conditional", 0.0),
             d.get("frequency", 0.0),
             d.get("n_prompts", 0))
            for n, d in G.nodes(data=True)
            if d.get("type") == "feature"
        ]
        feature_rows.sort(key=lambda x: x[1], reverse=True)  # sort by specific_score

        print("\nTop 10 features by specific_score (mean|score| × (1-freq)):")
        for feat_id, spec, mean_abs, mean_signed, freq, npr in feature_rows[:10]:
            print(
                f"  {feat_id}: specific={spec:.4f}, mean|score|={mean_abs:.4f}, "
                f"mean_score={mean_signed:+.4f}, freq={freq:.1%}, n_prompts={npr}"
            )

        # Patch C: Only print what we actually save
        print(f"\nSaved graph to {output_path / f'{name}.json'}")
    print("\n" + "=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_base.absolute()}")
    print("\nNext step: python scripts/07_run_interventions.py")


if __name__ == "__main__":
    main()
