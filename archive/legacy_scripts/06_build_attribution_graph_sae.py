"""
Build attribution graphs from SAE features to model outputs.

Implements a simplified version of the methodology from:
"On the Biology of a Large Language Model" (Lindsey, Gurnee, et al., 2025)

The attribution graph shows causal relationships:
    Input tokens -> SAE features -> ... -> Output logits

Usage:
    python scripts/05_build_attribution_graph.py --behaviour grammar_agreement
    python scripts/05_build_attribution_graph.py --all
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.sae import SparseAutoencoder


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
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


def load_sae(sae_path: Path, layer: int, device: torch.device) -> SparseAutoencoder:
    """Load trained SAE for a specific layer."""
    model_file = sae_path / f"layer_{layer}_best.pt"
    if not model_file.exists():
        model_file = sae_path / f"layer_{layer}_final.pt"

    if not model_file.exists():
        raise FileNotFoundError(f"No SAE model found for layer {layer} at {sae_path}")

    checkpoint = torch.load(model_file, map_location=device)

    # Reconstruct SAE from config
    config = checkpoint["config"]
    # Infer input_dim from model state
    encoder_weight = checkpoint["model_state"]["encoder.weight"]
    input_dim = encoder_weight.shape[1]

    sae = SparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=config["expansion_factor"],
        l1_lambda=config["l1_lambda"],
    )
    sae.load_state_dict(checkpoint["model_state"])
    sae = sae.to(device)
    sae.eval()

    return sae


class AttributionGraphBuilder:
    """
    Builds attribution graphs showing feature dependencies.

    Based on the methodology from Anthropic's attribution graph paper:
    - Compute gradients of output logits w.r.t. feature activations
    - Edge weight = activation × gradient (attribution)
    - Prune to keep top-k edges per node
    """

    def __init__(
        self,
        model: ModelWrapper,
        saes: Dict[int, SparseAutoencoder],
        device: torch.device,
        top_k_edges: int = 10,
        attribution_threshold: float = 0.01,
    ):
        """
        Initialize attribution graph builder.

        Args:
            model: Wrapped language model
            saes: Dictionary mapping layer index to trained SAE
            device: Computation device
            top_k_edges: Number of top edges to keep per node
            attribution_threshold: Minimum attribution to include edge
        """
        self.model = model
        self.saes = saes
        self.device = device
        self.top_k_edges = top_k_edges
        self.attribution_threshold = attribution_threshold
        self.layers = sorted(saes.keys())

    def compute_feature_activations(
        self,
        prompt: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute SAE feature activations for each layer.

        Args:
            prompt: Input text

        Returns:
            Dictionary mapping layer -> feature activations
        """
        # Get model activations with gradient tracking
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Extract SAE features for each layer
        feature_acts = {}
        for layer in self.layers:
            # Get residual stream at this layer (last token position)
            layer_act = hidden_states[layer][:, -1, :]  # (1, hidden_dim)

            # Apply SAE encoder
            sae = self.saes[layer]
            features = sae.encode(layer_act)  # (1, latent_dim)
            feature_acts[layer] = features

        return feature_acts

    def compute_output_attribution(
        self,
        prompt: str,
        target_token: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute attribution of each SAE feature to target token logit.

        Uses gradient-based attribution: attribution = activation × gradient

        Args:
            prompt: Input text
            target_token: Token to compute attribution for

        Returns:
            Dictionary mapping layer -> attribution scores per feature
        """
        # Get target token ID
        target_id = self.model.tokenizer.encode(
            target_token, add_special_tokens=False
        )[0]

        # Get model activations with gradient tracking
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Enable gradient computation
        self.model.model.eval()

        # Forward pass capturing hidden states
        outputs = self.model.model(**inputs, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)  # Make mutable

        # For each layer, apply SAE and track gradients
        attributions = {}

        for layer in self.layers:
            # Get activation at this layer
            layer_act = hidden_states[layer][:, -1:, :].clone().detach()
            layer_act.requires_grad_(True)

            # Apply SAE
            sae = self.saes[layer]
            features = sae.encode(layer_act.squeeze(1))

            # Reconstruct (for gradient flow)
            reconstructed = sae.decode(features)

            # Simple approximation: use reconstruction to approximate effect on output
            # Compute target logit using reconstructed activation
            # (This is a simplification - full method would use replacement model)

            # Get logit for target token
            target_logit = outputs.logits[0, -1, target_id]

            # Compute gradient of target logit w.r.t. features
            # Using a simplified approximation
            feature_grad = torch.autograd.grad(
                target_logit,
                layer_act,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if feature_grad is not None:
                # Approximate feature attribution via encoder weight projection
                with torch.no_grad():
                    # Project gradient into feature space
                    feature_grad_proj = sae.encoder(feature_grad.squeeze(1) - sae.pre_bias)
                    # Attribution = activation × gradient
                    attribution = features * feature_grad_proj
                    attributions[layer] = attribution.squeeze(0).cpu()
            else:
                attributions[layer] = torch.zeros(sae.latent_dim)

        return attributions

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
        # Compute attributions for correct token
        correct_attr = self.compute_output_attribution(prompt, correct_token)

        # Compute attributions for incorrect token
        incorrect_attr = self.compute_output_attribution(prompt, incorrect_token)

        # Build graph
        G = nx.DiGraph()

        # Add input node
        G.add_node("input", type="input", prompt=prompt)

        # Add output nodes
        G.add_node(f"output_{correct_token}", type="output", token=correct_token)
        G.add_node(f"output_{incorrect_token}", type="output", token=incorrect_token)

        # Add feature nodes and edges for each layer
        for layer in self.layers:
            attr_correct = correct_attr[layer]
            attr_incorrect = incorrect_attr[layer]

            # Differential attribution (correct - incorrect)
            diff_attr = attr_correct - attr_incorrect

            # Get top-k features by absolute differential attribution
            abs_attr = diff_attr.abs()
            top_k_values, top_k_indices = torch.topk(
                abs_attr, k=min(self.top_k_edges, len(abs_attr))
            )

            for i, (feat_idx, attr_val) in enumerate(zip(top_k_indices, top_k_values)):
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

                # Add edge from input to feature
                G.add_edge("input", feat_id, weight=abs_attr[feat_idx].item())

                # Add edge from feature to outputs
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
    ) -> nx.DiGraph:
        """
        Aggregate attribution graphs across multiple prompts.

        Args:
            prompts: List of prompt dictionaries
            n_prompts: Number of prompts to aggregate

        Returns:
            Aggregated attribution graph
        """
        # Sample prompts
        sample_prompts = prompts[:n_prompts]

        # Feature attribution counts
        feature_counts = defaultdict(lambda: {"count": 0, "total_attr": 0.0})

        print(f"Building attribution graphs for {len(sample_prompts)} prompts...")

        for prompt_data in tqdm(sample_prompts):
            try:
                G = self.build_graph_for_prompt(
                    prompt_data["prompt"],
                    prompt_data["correct_answer"].strip(),
                    prompt_data["incorrect_answer"].strip(),
                )

                # Accumulate feature attributions
                for node, data in G.nodes(data=True):
                    if data.get("type") == "feature":
                        feat_key = (data["layer"], data["feature_idx"])
                        feature_counts[feat_key]["count"] += 1
                        feature_counts[feat_key]["total_attr"] += abs(
                            data["differential_attribution"]
                        )

            except Exception as e:
                print(f"  Error processing prompt: {e}")
                continue

        # Build aggregated graph with features appearing in multiple prompts
        G_agg = nx.DiGraph()
        G_agg.add_node("input", type="input")
        G_agg.add_node("output_correct", type="output")
        G_agg.add_node("output_incorrect", type="output")

        # Add features that appear frequently
        min_count = max(1, n_prompts // 5)  # At least 20% of prompts

        for (layer, feat_idx), stats in feature_counts.items():
            if stats["count"] >= min_count:
                feat_id = f"L{layer}_F{feat_idx}"
                avg_attr = stats["total_attr"] / stats["count"]

                G_agg.add_node(
                    feat_id,
                    type="feature",
                    layer=layer,
                    feature_idx=feat_idx,
                    count=stats["count"],
                    avg_attribution=avg_attr,
                )

                G_agg.add_edge("input", feat_id, weight=avg_attr)
                G_agg.add_edge(feat_id, "output_correct", weight=avg_attr)

        return G_agg


def save_graph(G: nx.DiGraph, output_path: Path, name: str):
    """Save graph to multiple formats."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as GraphML (for visualization tools)
    graphml_path = output_path / f"{name}.graphml"
    nx.write_graphml(G, graphml_path)

    # Save as JSON (for custom processing)
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
    }
    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"  Saved graph: {graphml_path.name}, {json_path.name}")
    return graphml_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Build attribution graphs")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        choices=["grammar_agreement", "factual_recall", "sentiment_continuation", "arithmetic"],
        help="Which behaviour to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all behaviours",
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
        "--layer",
        type=int,
        default=None,
        help="Specific layer to analyze (default: all available)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    torch.manual_seed(config["seeds"]["torch_seed"])

    # Determine behaviours
    if args.all:
        behaviours = ["grammar_agreement", "factual_recall", "sentiment_continuation", "arithmetic"]
    elif args.behaviour:
        behaviours = [args.behaviour]
    else:
        print("Error: Must specify --behaviour or --all")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Device: {device}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"  Prompts per behaviour: {args.n_prompts}")

    # Load model
    print(f"\nLoading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    # Determine layers to analyze
    layer_range = config["activations"]["layer_range"]
    if args.layer:
        layers = [args.layer]
    else:
        layers = list(range(layer_range[0], layer_range[1]))

    # Load SAEs
    print(f"\nLoading SAEs for layers {layers}...")
    sae_path = Path(config["paths"]["saes"])
    saes = {}

    for layer in layers:
        try:
            saes[layer] = load_sae(sae_path, layer, device)
            print(f"  Loaded SAE for layer {layer}")
        except FileNotFoundError:
            print(f"  Warning: No SAE for layer {layer}, skipping")

    if not saes:
        print("\nError: No trained SAEs found. Run 04_train_sae.py first.")
        return

    # Build attribution graph builder
    builder = AttributionGraphBuilder(
        model=model,
        saes=saes,
        device=device,
        top_k_edges=config["attribution"]["top_k_edges"],
        attribution_threshold=config["attribution"]["attribution_threshold"],
    )

    # Process each behaviour
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
            print(f"  Prompt file not found. Skipping {behaviour}.")
            continue

        print(f"Loaded {len(prompts)} prompts")

        # Build aggregated attribution graph
        print(f"\nBuilding aggregated attribution graph...")
        G = builder.aggregate_graphs(prompts, n_prompts=args.n_prompts)

        # Save graph
        output_path = output_base / behaviour
        save_graph(G, output_path, f"attribution_graph_{args.split}")

        # Print summary
        n_features = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "feature")
        n_edges = G.number_of_edges()

        print(f"\nGraph summary:")
        print(f"  Feature nodes: {n_features}")
        print(f"  Edges: {n_edges}")

        # Show top features by attribution
        feature_attrs = [
            (n, d["avg_attribution"], d["count"])
            for n, d in G.nodes(data=True)
            if d.get("type") == "feature"
        ]
        feature_attrs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 attributed features:")
        for feat_id, attr, count in feature_attrs[:10]:
            print(f"  {feat_id}: avg_attr={attr:.4f}, count={count}")

    print("\n" + "=" * 70)
    print("ATTRIBUTION GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_base.absolute()}")
    print("\nNext step: python scripts/06_run_interventions.py")


if __name__ == "__main__":
    main()
