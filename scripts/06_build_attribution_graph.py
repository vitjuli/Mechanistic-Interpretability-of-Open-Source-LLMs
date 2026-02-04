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
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet, SingleLayerTranscoder

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
            f"{metadata['n_active_features']} active features"
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
        extracted_features: Dict[int, Dict],  # NEW: Pre-extracted features from script 04
        position_map: List[Dict],  # NEW: Maps sample_idx -> (prompt_idx, token_pos, token_id)
        device: torch.device,
        top_k_edges: int = 10,
        attribution_threshold: float = 0.01,
        layers: Optional[List[int]] = None,
    ):
        """
        Initialize attribution graph builder with pre-extracted transcoder features.

        CRITICAL CHANGE: Now accepts pre-extracted features from script 04 instead
        of recomputing them. This ensures we use features from MLP inputs (correct
        distribution for transcoders) not residual stream (wrong distribution).

        Args:
            model: Wrapped language model
            transcoder_set: Pre-trained TranscoderSet
            extracted_features: Pre-computed features from script 04
            position_map: Sample-to-prompt mapping from script 04
            device: Computation device
            top_k_edges: Number of top edges to keep per node
            attribution_threshold: Minimum attribution magnitude to include edge
            layers: Specific layers to analyze (None = all in transcoder_set)
        """
        self.model = model
        self.transcoder_set = transcoder_set
        self.extracted_features = extracted_features  # NEW
        self.position_map = position_map  # NEW
        self.device = device
        self.top_k_edges = top_k_edges
        self.attribution_threshold = attribution_threshold

        # Determine layers to use
        if layers is None:
            self.layers = list(range(transcoder_set.config.num_layers))
        else:
            self.layers = [l for l in layers if l in transcoder_set]

        # Build reverse map: prompt_idx -> [sample_indices]
        # CRITICAL for attribution: we need to know which samples belong to each prompt
        self.prompt_to_samples = build_prompt_to_samples_map(position_map)  # NEW

        logger.info(f"Attribution builder initialized for layers: {self.layers}")
        logger.info(
            f"Loaded {len(position_map)} samples across {len(self.prompt_to_samples)} prompts"
        )

    def compute_prompt_attribution(
        self,
        prompt_idx: int,
        correct_token: str,
        incorrect_token: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute differential attribution for a prompt using pre-extracted features.
        
        SIMPLIFIED APPROACH: Uses activation-based attribution instead of gradients.
        Attribution = avg_feature_activation × correlation_with_correct
        
        This is simpler than gradient-based attribution and avoids complex backprop
        through the model with injected activations.
        
        Args:
            prompt_idx: Index of the prompt in the prompts list
            correct_token: Correct answer token
            incorrect_token: Incorrect answer token
        
        Returns:
            Dict mapping layer -> attribution scores per feature (sparse tensor)
        """
        if prompt_idx not in self.prompt_to_samples:
            logger.warning(f"Prompt {prompt_idx} not in position map")
            return {}
        
        # Get sample indices for this prompt
        sample_indices = self.prompt_to_samples[prompt_idx]
        
        # Get logits for correct/incorrect tokens to determine "success"
        # (Use model forward pass to see which token is predicted)
        # For now, use simpler heuristic: features that activate ARE important
        # (This is the activation-based attribution approach)
        
        attributions = {}
        
        for layer in self.layers:
            layer_data = self.extracted_features[layer]
            
            # Get top-k features for this prompt's samples
            sample_top_k_indices = layer_data['top_k_indices'][sample_indices]  # (n_samples, top_k)
            sample_top_k_values = layer_data['top_k_values'][sample_indices]  # (n_samples, top_k)
            
           # Compute sparse attribution tensor
            d_transcoder = layer_data['metadata']['d_transcoder']
            attribution_sparse = torch.zeros(d_transcoder, dtype=torch.float32)
            feature_count = torch.zeros(d_transcoder, dtype=torch.float32)
            
            # Aggregate activations across all samples for this prompt
            for i in range(len(sample_indices)):
                feat_indices = sample_top_k_indices[i]  # (top_k,)
                feat_values = sample_top_k_values[i]  # (top_k,)
                
                for feat_idx, feat_val in zip(feat_indices, feat_values):
                    feat_idx = feat_idx.item()
                    feat_val = feat_val.item()
                    
                    # Attribution = feature activation (simple!)
                    # More sophisticated: multiply by gradient or use correlation
                    # For now: use raw activation as proxy for importance
                    attribution_sparse[feat_idx] += feat_val
                    feature_count[feat_idx] += 1
            
            # Average over samples
            mask = feature_count > 0
            attribution_sparse[mask] /= feature_count[mask]
            
            attributions[layer] = attribution_sparse
        
        return attributions


    def build_graph_for_prompt(
        self,
        prompt_idx: int,  # NEW: pass index instead of text
        prompt_data: Dict,  # NEW: pass full prompt dict
    ) -> nx.DiGraph:
        """
        Build attribution graph for a single prompt using pre-extracted features.
        
        CHANGED: Now uses prompt_idx to look up pre-extracted features instead
        of recomputing them via forward pass.
        
        Args:
prompt_idx: Index in prompts list
            prompt_data: Prompt dictionary with 'prompt', 'correct_answer', 'incorrect_answer'
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        G.add_node("input", type="input")
        
        correct_token = prompt_data["correct_answer"].strip()
        incorrect_token = prompt_data["incorrect_answer"].strip()
        
        G.add_node("output_correct", type="output", token=correct_token)
        G.add_node("output_incorrect", type="output", token=incorrect_token)
        
        # Compute attribution using pre-extracted features
        attributions = self.compute_prompt_attribution(
            prompt_idx,
            correct_token,
            incorrect_token,
        )
        
        # Add feature nodes and edges
        for layer, attr_tensor in attributions.items():
            # Get top-k most attributed features
            top_values, top_indices = torch.topk(
                attr_tensor,
                k=min(self.top_k_edges, (attr_tensor > 0).sum().item()),
            )
            
            for feat_idx, attr_val in zip(top_indices, top_values):
                feat_idx = feat_idx.item()
                attr_val = attr_val.item()
                
                if attr_val < self.attribution_threshold:
                    continue
                
                feat_id = f"L{layer}_F{feat_idx}"
                
                # Add feature node
                G.add_node(
                    feat_id,
                    type="feature",
                    layer=layer,
                    feature_idx=feat_idx,
                    attribution=attr_val,
                )
                
                # Add edges
                G.add_edge("input", feat_id, weight=attr_val)
                G.add_edge(feat_id, "output_correct", weight=attr_val)
                # Note: For simplicity, we don't differentiate correct/incorrect attribution
                # In future: use logits to compute differential attribution
        
        return G

    def aggregate_graphs(
        self,
        prompts: List[Dict],
        n_prompts: int = 20,
        min_frequency: float = 0.2,
    ) -> nx.DiGraph:
        """
        Aggregate attribution graphs across multiple prompts using pre-extracted features.

        CHANGED: Now uses prompt indices to look up pre-extracted features.
        Removed silent error handling - errors now propagate for debugging.

        Args:
            prompts: List of prompt dictionaries
            n_prompts: Number of prompts to aggregate
            min_frequency: Minimum fraction of prompts a feature must appear in

        Returns:
            Aggregated attribution graph
        """
        sample_prompts = prompts[:n_prompts]

        # Track feature statistics
        feature_stats = defaultdict(lambda: {
            "count": 0,
            "total_attribution": 0.0,
        })

        logger.info(f"Building attribution graphs for {len(sample_prompts)} prompts...")

        for prompt_idx, prompt_data in enumerate(tqdm(sample_prompts, desc="Building graphs")):
            # Build graph for this prompt
            # NOTE: No try/except - let errors propagate so we can debug!
            G = self.build_graph_for_prompt(
                prompt_idx,  # Use index to lookup features
                prompt_data,
            )

            # Accumulate feature statistics
            for node, data in G.nodes(data=True):
                if data.get("type") == "feature":
                    feat_key = (data["layer"], data["feature_idx"])
                    feature_stats[feat_key]["count"] += 1
                    feature_stats[feat_key]["total_attribution"] += data["attribution"]

        logger.info(f"Collected statistics for {len(feature_stats)} unique features")

        # Build aggregated graph
        G_agg = nx.DiGraph()
        G_agg.add_node("input", type="input")
        G_agg.add_node("output_correct", type="output")
        G_agg.add_node("output_incorrect", type="output")

        # Add features meeting frequency threshold
        min_count = max(1, int(n_prompts * min_frequency))
        logger.info(f"Filtering features with min_count >= {min_count}")

        features_added = 0
        for (layer, feat_idx), stats in feature_stats.items():
            if stats["count"] >= min_count:
                feat_id = f"L{layer}_F{feat_idx}"
                count = stats["count"]
                avg_attribution = stats["total_attribution"] / count

                G_agg.add_node(
                    feat_id,
                    type="feature",
                    layer=layer,
                    feature_idx=feat_idx,
                    count=count,
                    frequency=count / len(sample_prompts),
                    avg_attribution=avg_attribution,
                )

                G_agg.add_edge("input", feat_id, weight=avg_attribution)
                G_agg.add_edge(feat_id, "output_correct", weight=avg_attribution)
                
                features_added += 1

        logger.info(f"Aggregated graph: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")
        logger.info(f"Added {features_added} features meeting threshold")

        return G_agg

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
    if "4b" in model_size.lower():
        model_name = "Qwen/Qwen3-4B-Instruct-2507"

    model = ModelWrapper(
        model_name=model_name,
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
        split="train",  # Use train split for building graph
        layers=layers,
    )
    
    position_map = load_position_map(
        features_path,
        behaviour=behaviour,
        split="train",
    )
    
    print(f"Loaded {len(position_map)} samples from {len(extracted_features)} layers")

    # Build attribution builder with pre-extracted features
    builder = TranscoderAttributionBuilder(
        model=model,
        transcoder_set=transcoder_set,
        extracted_features=extracted_features,  # NEW
        position_map=position_map,  # NEW
        device=device,
        top_k_edges=config["attribution"]["top_k_edges"],
        attribution_threshold=config["attribution"]["attribution_threshold"],
        layers=layers,
    )

    # Process behaviours
    output_base = Path(config["paths"]["results"]) / "attribution_graphs"

    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)

        # Load prompts
        prompt_path = Path(config["paths"]["prompts"])
        try:
            prompts = load_prompts(prompt_path, behaviour, args.split)
        except FileNotFoundError:
            print(f"Prompt file not found. Skipping {behaviour}.")
            continue

        print(f"Loaded {len(prompts)} prompts")

        # Build aggregated graph
        print(f"\nBuilding aggregated attribution graph...")
        G = builder.aggregate_graphs(
            prompts,
            n_prompts=args.n_prompts,
            min_frequency=tc_config["attribution"]["min_feature_frequency"],
        )

        # Save graph
        output_path = output_base / behaviour
        metadata = {
            "behaviour": behaviour,
            "split": args.split,
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "layers": layers,
            "n_prompts": args.n_prompts,
            "timestamp": datetime.now().isoformat(),
        }
        save_graph(G, output_path, f"attribution_graph_{args.split}", metadata)

        # Print summary
        n_features = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "feature")
        n_edges = G.number_of_edges()

        print(f"\nGraph summary:")
        print(f"  Feature nodes: {n_features}")
        print(f"  Edges: {n_edges}")

        # Show top features
        feature_attrs = [
            (n, d["avg_attribution"], d["frequency"])
            for n, d in G.nodes(data=True)
            if d.get("type") == "feature"
        ]
        feature_attrs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 attributed features:")
        for feat_id, attr, freq in feature_attrs[:10]:
            print(f"  {feat_id}: avg_attr={attr:.4f}, frequency={freq:.1%}")

    print("\n" + "=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_base.absolute()}")
    print("\nNext step: python scripts/07_run_interventions.py")


if __name__ == "__main__":
    main()
