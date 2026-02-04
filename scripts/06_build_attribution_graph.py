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
        device: torch.device,
        top_k_edges: int = 10,
        attribution_threshold: float = 0.01,
        layers: Optional[List[int]] = None,
    ):
        """
        Initialize attribution graph builder with transcoders.

        Args:
            model: Wrapped language model
            transcoder_set: Pre-trained TranscoderSet
            device: Computation device
            top_k_edges: Number of top edges to keep per node
            attribution_threshold: Minimum attribution magnitude to include edge
            layers: Specific layers to analyze (None = all in transcoder_set)
        """
        self.model = model
        self.transcoder_set = transcoder_set
        self.device = device
        self.top_k_edges = top_k_edges
        self.attribution_threshold = attribution_threshold

        # Determine layers to use
        if layers is None:
            self.layers = list(range(transcoder_set.config.num_layers))
        else:
            self.layers = [l for l in layers if l in transcoder_set]

        logger.info(f"Attribution builder initialized for layers: {self.layers}")

    def compute_feature_activations(
        self,
        prompt: str,
        position: str = "last",
    ) -> Dict[int, torch.Tensor]:
        """
        Compute transcoder feature activations for each layer.

        Args:
            prompt: Input text
            position: Token position to analyze ("last" or "all")

        Returns:
            Dictionary mapping layer -> feature activations
        """
        # Get model hidden states
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Extract transcoder features for each layer
        feature_acts = {}
        for layer in self.layers:
            # Get residual stream at this layer (MLP input)
            # For transcoders, we need the pre-MLP activation
            if position == "last":
                layer_act = hidden_states[layer][:, -1, :]  # (1, hidden_dim)
            else:
                layer_act = hidden_states[layer]  # (1, seq_len, hidden_dim)

            # Apply transcoder encoder
            transcoder = self.transcoder_set[layer]
            features = transcoder.encode(layer_act.to(transcoder.dtype))
            feature_acts[layer] = features

        return feature_acts

    def compute_output_attribution(
        self,
        prompt: str,
        target_token: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute attribution of each transcoder feature to target token logit.

        Uses gradient-based attribution: attribution = activation x gradient

        Args:
            prompt: Input text
            target_token: Token to compute attribution for

        Returns:
            Dictionary mapping layer -> attribution scores per feature
        """
        # Get target token ID
        target_ids = self.model.tokenizer.encode(target_token, add_special_tokens=False)
        if not target_ids:
            logger.warning(f"Could not encode token: {target_token}")
            return {}
        target_id = target_ids[0]

        # Get model activations with gradient tracking
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass with hidden states and gradient tracking
        self.model.model.eval()
        outputs = self.model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        logits = outputs.logits

        # Get target logit
        target_logit = logits[0, -1, target_id]

        attributions = {}

        for layer in self.layers:
            # Get activation at this layer and enable gradients
            layer_act = hidden_states[layer][:, -1:, :].clone().detach()
            layer_act.requires_grad_(True)

            # Get transcoder for this layer
            transcoder = self.transcoder_set[layer]

            # Encode to get features
            with torch.no_grad():
                features = transcoder.encode(layer_act.squeeze(1).to(transcoder.dtype))

            # Compute gradient of target logit w.r.t. layer activation
            try:
                grad = torch.autograd.grad(
                    target_logit,
                    layer_act,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                if grad is not None:
                    with torch.no_grad():
                        # Project gradient into feature space via encoder
                        # This approximates the sensitivity of each feature to the output
                        grad_features = transcoder.encode(
                            grad.squeeze(1).to(transcoder.dtype),
                            apply_activation_function=False,  # Use pre-activation
                        )

                        # Attribution = feature activation x gradient magnitude
                        attribution = features * grad_features.abs()
                        attributions[layer] = attribution.squeeze(0).cpu()
                else:
                    attributions[layer] = torch.zeros(transcoder.d_transcoder)

            except Exception as e:
                logger.warning(f"Gradient computation failed for layer {layer}: {e}")
                attributions[layer] = torch.zeros(transcoder.d_transcoder)

        return attributions

    def compute_differential_attribution(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute differential attribution between correct and incorrect tokens.

        This highlights features that distinguish the correct answer.

        Args:
            prompt: Input text
            correct_token: Correct answer token
            incorrect_token: Incorrect alternative

        Returns:
            Dictionary mapping layer -> differential attribution per feature
        """
        correct_attr = self.compute_output_attribution(prompt, correct_token)
        incorrect_attr = self.compute_output_attribution(prompt, incorrect_token)

        diff_attr = {}
        for layer in self.layers:
            if layer in correct_attr and layer in incorrect_attr:
                diff_attr[layer] = correct_attr[layer] - incorrect_attr[layer]
            elif layer in correct_attr:
                diff_attr[layer] = correct_attr[layer]

        return diff_attr

    def build_graph_for_prompt(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
    ) -> nx.DiGraph:
        """
        Build attribution graph for a single prompt.

        Args:
            prompt: Input text
            correct_token: Correct next token
            incorrect_token: Incorrect alternative

        Returns:
            NetworkX directed graph with attributed features
        """
        # Compute attributions
        correct_attr = self.compute_output_attribution(prompt, correct_token)
        incorrect_attr = self.compute_output_attribution(prompt, incorrect_token)

        # Build graph
        G = nx.DiGraph()

        # Add input node
        G.add_node("input", type="input", prompt=prompt[:100])

        # Add output nodes
        G.add_node(f"output_{correct_token}", type="output", token=correct_token)
        G.add_node(f"output_{incorrect_token}", type="output", token=incorrect_token)

        # Add feature nodes and edges for each layer
        for layer in self.layers:
            if layer not in correct_attr:
                continue

            attr_correct = correct_attr[layer]
            attr_incorrect = incorrect_attr.get(layer, torch.zeros_like(attr_correct))

            # Differential attribution
            diff_attr = attr_correct - attr_incorrect

            # Get top-k features by absolute differential attribution
            abs_attr = diff_attr.abs()
            top_k_values, top_k_indices = torch.topk(
                abs_attr, k=min(self.top_k_edges, len(abs_attr))
            )

            for feat_idx, attr_val in zip(top_k_indices, top_k_values):
                if attr_val.item() < self.attribution_threshold:
                    continue

                feat_id = f"L{layer}_F{feat_idx.item()}"
                G.add_node(
                    feat_id,
                    type="feature",
                    layer=layer,
                    feature_idx=feat_idx.item(),
                    attribution_correct=attr_correct[feat_idx].item(),
                    attribution_incorrect=attr_incorrect[feat_idx].item(),
                    differential_attribution=diff_attr[feat_idx].item(),
                )

                # Add edges
                G.add_edge("input", feat_id, weight=abs_attr[feat_idx].item())
                G.add_edge(
                    feat_id,
                    f"output_{correct_token}",
                    weight=attr_correct[feat_idx].item(),
                )
                G.add_edge(
                    feat_id,
                    f"output_{incorrect_token}",
                    weight=attr_incorrect[feat_idx].item(),
                )

        return G

    def aggregate_graphs(
        self,
        prompts: List[Dict],
        n_prompts: int = 20,
        min_frequency: float = 0.2,
    ) -> nx.DiGraph:
        """
        Aggregate attribution graphs across multiple prompts.

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
            "total_attr_correct": 0.0,
            "total_attr_incorrect": 0.0,
            "total_diff_attr": 0.0,
        })

        logger.info(f"Building attribution graphs for {len(sample_prompts)} prompts...")

        for prompt_data in tqdm(sample_prompts, desc="Building graphs"):
            try:
                G = self.build_graph_for_prompt(
                    prompt_data["prompt"],
                    prompt_data["correct_answer"].strip(),
                    prompt_data["incorrect_answer"].strip(),
                )

                # Accumulate feature statistics
                for node, data in G.nodes(data=True):
                    if data.get("type") == "feature":
                        feat_key = (data["layer"], data["feature_idx"])
                        feature_stats[feat_key]["count"] += 1
                        feature_stats[feat_key]["total_attr_correct"] += data["attribution_correct"]
                        feature_stats[feat_key]["total_attr_incorrect"] += data["attribution_incorrect"]
                        feature_stats[feat_key]["total_diff_attr"] += abs(data["differential_attribution"])

            except Exception as e:
                logger.warning(f"Error processing prompt: {e}")
                continue

        # Build aggregated graph
        G_agg = nx.DiGraph()
        G_agg.add_node("input", type="input")
        G_agg.add_node("output_correct", type="output")
        G_agg.add_node("output_incorrect", type="output")

        # Add features meeting frequency threshold
        min_count = max(1, int(n_prompts * min_frequency))

        for (layer, feat_idx), stats in feature_stats.items():
            if stats["count"] >= min_count:
                feat_id = f"L{layer}_F{feat_idx}"
                count = stats["count"]
                avg_diff_attr = stats["total_diff_attr"] / count
                avg_correct = stats["total_attr_correct"] / count
                avg_incorrect = stats["total_attr_incorrect"] / count

                G_agg.add_node(
                    feat_id,
                    type="feature",
                    layer=layer,
                    feature_idx=feat_idx,
                    count=count,
                    frequency=count / len(sample_prompts),
                    avg_attribution_correct=avg_correct,
                    avg_attribution_incorrect=avg_incorrect,
                    avg_differential_attribution=avg_diff_attr,
                )

                G_agg.add_edge("input", feat_id, weight=avg_diff_attr)
                G_agg.add_edge(feat_id, "output_correct", weight=avg_correct)
                G_agg.add_edge(feat_id, "output_incorrect", weight=avg_incorrect)

        logger.info(f"Aggregated graph: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")

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

    # Build attribution builder
    builder = TranscoderAttributionBuilder(
        model=model,
        transcoder_set=transcoder_set,
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
            (n, d["avg_differential_attribution"], d["frequency"])
            for n, d in G.nodes(data=True)
            if d.get("type") == "feature"
        ]
        feature_attrs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 attributed features:")
        for feat_id, attr, freq in feature_attrs[:10]:
            print(f"  {feat_id}: avg_diff_attr={attr:.4f}, frequency={freq:.1%}")

    print("\n" + "=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_base.absolute()}")
    print("\nNext step: python scripts/07_run_interventions.py")


if __name__ == "__main__":
    main()
