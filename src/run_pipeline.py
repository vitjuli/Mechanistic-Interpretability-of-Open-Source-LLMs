#!/usr/bin/env python3
"""Main pipeline for mechanistic interpretability analysis of Qwen3-4B.

This script orchestrates the full pipeline:
1. Run Qwen3-4B on behavior prompts
2. Capture activations for target layers
3. Train sparse autoencoders on activations
4. Build and prune dependency graphs
5. Validate with interventions

Usage:
    python run_pipeline.py [--behavior BEHAVIOR] [--device DEVICE]

Reference:
    Lindsey, J., Gurnee, W., et al. (2025): On the Biology of a Large Language
    Model. Anthropic Transformer.
"""

import argparse
import json
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mi_pipeline import (
    ActivationCapture,
    Config,
    DependencyGraph,
    InterventionValidator,
)
from src.mi_pipeline.sparse_autoencoder import SAETrainer, SparseAutoencoder
from prompts import get_prompts, get_all_prompts


def setup_output_dirs(config: Config) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.activations_dir, exist_ok=True)
    os.makedirs(config.sae_dir, exist_ok=True)
    os.makedirs(config.graphs_dir, exist_ok=True)


def run_activation_capture(
    config: Config,
    behavior: str,
    prompts: list[str],
) -> dict[int, torch.Tensor]:
    """Capture activations for the given behavior and prompts.

    Args:
        config: Pipeline configuration.
        behavior: Behavior category name.
        prompts: List of prompts to process.

    Returns:
        Dictionary mapping layer indices to activation tensors.
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Capturing activations for '{behavior}'")
    print(f"{'='*60}")
    print(f"Processing {len(prompts)} prompts...")
    print(f"Target layers: {config.target_layers}")

    capturer = ActivationCapture(config)
    capturer.load_model()

    activations = capturer.capture(prompts, show_progress=True)

    # Save activations
    capturer.save_activations(activations, behavior)
    print(f"Activations saved to {config.activations_dir}")

    for layer_idx, acts in activations.items():
        print(f"  Layer {layer_idx}: {acts.shape}")

    return activations


def train_sparse_autoencoders(
    config: Config,
    behavior: str,
    activations: dict[int, torch.Tensor],
) -> dict[int, SAETrainer]:
    """Train SAEs for each layer's activations.

    Args:
        config: Pipeline configuration.
        behavior: Behavior category name.
        activations: Dictionary of layer activations.

    Returns:
        Dictionary mapping layer indices to trained SAE trainers.
    """
    print(f"\n{'='*60}")
    print(f"Step 2: Training Sparse Autoencoders for '{behavior}'")
    print(f"{'='*60}")

    trainers = {}

    for layer_idx, acts in activations.items():
        print(f"\nTraining SAE for layer {layer_idx}...")
        print(f"  Input dim: {acts.shape[-1]}")
        print(f"  Hidden dim: {config.sae_hidden_dim}")
        print(f"  Samples: {acts.shape[0]}")

        sae = SparseAutoencoder(
            input_dim=acts.shape[-1],
            hidden_dim=config.sae_hidden_dim,
        )

        trainer = SAETrainer(sae, config, device=config.device)
        history = trainer.train(acts.float(), show_progress=True)

        # Save the trained SAE
        save_path = os.path.join(
            config.sae_dir,
            f"{behavior}_layer_{layer_idx}_sae.pt",
        )
        trainer.save(save_path)
        print(f"  Saved to {save_path}")

        trainers[layer_idx] = trainer

    return trainers


def build_dependency_graph(
    config: Config,
    behavior: str,
    activations: dict[int, torch.Tensor],
    trainers: dict[int, SAETrainer],
    tokenizer,
    prompts: list[str],
) -> DependencyGraph:
    """Build and prune the dependency graph.

    Args:
        config: Pipeline configuration.
        behavior: Behavior category name.
        activations: Dictionary of layer activations.
        trainers: Dictionary of SAE trainers.
        tokenizer: Model tokenizer.
        prompts: Original prompts (for token information).

    Returns:
        Pruned dependency graph.
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Building Dependency Graph for '{behavior}'")
    print(f"{'='*60}")

    # Get SAE features for each layer
    layer_features = {}
    for layer_idx, trainer in trainers.items():
        features = trainer.get_feature_activations(activations[layer_idx])
        layer_features[layer_idx] = features
        print(f"  Layer {layer_idx} features: {features.shape}")

    # Get tokens from first prompt as example
    tokens = tokenizer.tokenize(prompts[0])
    print(f"  Example tokens: {tokens[:10]}...")

    # Get decisive logits (top-k most likely next tokens)
    # For demo, use placeholder logit indices
    decisive_logits = list(range(10))

    # Build graph
    graph = DependencyGraph(config)
    graph.build_from_activations(
        tokens=tokens,
        layer_activations=activations,
        layer_features=layer_features,
        decisive_logits=decisive_logits,
    )

    print(f"  Initial graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Prune the graph
    pruned_graph = graph.prune(threshold=config.graph_prune_threshold)
    print(
        f"  Pruned graph: {len(pruned_graph.nodes)} nodes, "
        f"{len(pruned_graph.edges)} edges"
    )

    # Save graph
    graph_path = os.path.join(config.graphs_dir, f"{behavior}_graph.json")
    pruned_graph.save(graph_path)
    print(f"  Saved to {graph_path}")

    # Get top features
    top_features = pruned_graph.get_top_features(k=10)
    print(f"\n  Top 10 features by importance:")
    for feat_id, importance in top_features:
        print(f"    {feat_id}: {importance:.4f}")

    return pruned_graph


def run_interventions(
    config: Config,
    behavior: str,
    model,
    saes: dict[int, SparseAutoencoder],
    tokenizer,
    prompts: list[str],
    graph: DependencyGraph,
) -> dict:
    """Run intervention validation experiments.

    Args:
        config: Pipeline configuration.
        behavior: Behavior category name.
        model: The transformer model.
        saes: Dictionary of trained SAEs.
        tokenizer: Model tokenizer.
        prompts: Test prompts.
        graph: Dependency graph with important features.

    Returns:
        Dictionary of intervention results.
    """
    print(f"\n{'='*60}")
    print(f"Step 4: Running Intervention Validation for '{behavior}'")
    print(f"{'='*60}")

    validator = InterventionValidator(model, saes, config)

    # Get top features to validate
    top_features = graph.get_top_features(k=5)

    results_by_layer = {}

    for prompt in prompts[:3]:  # Test on first 3 prompts
        print(f"\nPrompt: {prompt[:50]}...")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        input_ids = inputs["input_ids"].to(config.device)

        # Group features by layer
        layer_features: dict[int, list[int]] = {}
        for feat_id, _ in top_features:
            # Parse feature ID (format: L{layer}_F{idx})
            parts = feat_id.split("_")
            layer = int(parts[0][1:])
            feat_idx = int(parts[1][1:])
            if layer not in layer_features:
                layer_features[layer] = []
            layer_features[layer].append(feat_idx)

        # Run inhibition for each layer
        for layer, feat_indices in layer_features.items():
            if layer not in saes:
                continue

            print(f"  Layer {layer}: inhibiting features {feat_indices[:3]}...")

            try:
                result = validator.validate_inhibition(
                    input_ids=input_ids,
                    layer=layer,
                    feature_indices=feat_indices[:3],
                    target_logit=0,  # First token as target
                )

                print(f"    Logit diff: {result.logit_diff:.4f}")
                print(f"    KL divergence: {result.kl_divergence:.4f}")
                print(f"    Success: {result.success}")

                if layer not in results_by_layer:
                    results_by_layer[layer] = []
                results_by_layer[layer].append(result.to_dict())

            except Exception as e:
                print(f"    Error: {e}")

    # Summarize results
    print(f"\n  Intervention Summary:")
    for layer, results in results_by_layer.items():
        print(f"    Layer {layer}: {len(results)} interventions")

    return results_by_layer


def run_full_pipeline(
    behavior: str,
    config: Config,
) -> dict:
    """Run the complete mechanistic interpretability pipeline.

    Args:
        behavior: Behavior category to analyze.
        config: Pipeline configuration.

    Returns:
        Dictionary with all pipeline results.
    """
    print(f"\n{'#'*60}")
    print(f"# Running Mechanistic Interpretability Pipeline")
    print(f"# Behavior: {behavior}")
    print(f"# Model: {config.model_name}")
    print(f"{'#'*60}")

    setup_output_dirs(config)

    # Get prompts for this behavior
    prompts = get_prompts(behavior)
    print(f"\nLoaded {len(prompts)} prompts for '{behavior}'")

    results = {
        "behavior": behavior,
        "config": {
            "model_name": config.model_name,
            "target_layers": config.target_layers,
            "sae_hidden_dim": config.sae_hidden_dim,
            "sae_sparsity_penalty": config.sae_sparsity_penalty,
        },
    }

    # Step 1: Capture activations
    activations = run_activation_capture(config, behavior, prompts)
    results["activation_shapes"] = {
        str(k): list(v.shape) for k, v in activations.items()
    }

    # Step 2: Train SAEs
    trainers = train_sparse_autoencoders(config, behavior, activations)

    # Get SAE models for later use
    saes = {layer: trainer.sae for layer, trainer in trainers.items()}

    # Step 3: Build dependency graph
    # Need to reload model/tokenizer for graph building
    capturer = ActivationCapture(config)
    capturer.load_model()

    graph = build_dependency_graph(
        config, behavior, activations, trainers,
        capturer.tokenizer, prompts
    )
    results["graph"] = {
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "top_features": graph.get_top_features(k=10),
    }

    # Step 4: Run interventions
    intervention_results = run_interventions(
        config, behavior, capturer.model, saes,
        capturer.tokenizer, prompts, graph
    )
    results["interventions"] = intervention_results

    # Save final results
    results_path = os.path.join(config.output_dir, f"{behavior}_results.json")
    with open(results_path, "w") as f:
        # Convert non-serializable items
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'#'*60}")
    print(f"# Pipeline Complete for '{behavior}'")
    print(f"{'#'*60}")

    return results


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Mechanistic Interpretability Pipeline for Qwen3-4B"
    )
    parser.add_argument(
        "--behavior",
        type=str,
        default="factual_recall",
        choices=["factual_recall", "reasoning", "code_generation", "multilingual", "all"],
        help="Behavior category to analyze (default: factual_recall)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="8,16,24,31",
        help="Comma-separated layer indices (default: 8,16,24,31)",
    )
    parser.add_argument(
        "--sae-hidden-dim",
        type=int,
        default=4096,
        help="SAE hidden dimension (default: 4096)",
    )
    parser.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        help="SAE training epochs (default: 10)",
    )

    args = parser.parse_args()

    # Create config
    config = Config(
        device=args.device,
        output_dir=args.output_dir,
        activations_dir=os.path.join(args.output_dir, "activations"),
        sae_dir=os.path.join(args.output_dir, "sae_models"),
        graphs_dir=os.path.join(args.output_dir, "graphs"),
        target_layers=tuple(int(x) for x in args.layers.split(",")),
        sae_hidden_dim=args.sae_hidden_dim,
        sae_epochs=args.sae_epochs,
    )

    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.model_name}")
    print(f"  Layers: {config.target_layers}")
    print(f"  SAE hidden dim: {config.sae_hidden_dim}")
    print(f"  SAE epochs: {config.sae_epochs}")

    if args.behavior == "all":
        # Run for all behaviors
        all_results = {}
        for behavior in config.behaviors:
            try:
                all_results[behavior] = run_full_pipeline(behavior, config)
            except Exception as e:
                print(f"Error processing {behavior}: {e}")
                all_results[behavior] = {"error": str(e)}

        # Save combined results
        combined_path = os.path.join(config.output_dir, "all_behaviors_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined results saved to {combined_path}")
    else:
        run_full_pipeline(args.behavior, config)


if __name__ == "__main__":
    main()
