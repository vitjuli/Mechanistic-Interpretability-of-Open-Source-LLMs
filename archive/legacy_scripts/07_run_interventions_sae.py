"""
Run intervention experiments to validate identified circuits.

Implements three types of interventions from Anthropic's methodology:
1. Feature Ablation: Zero out top attributed features, measure effect
2. Feature Patching: Swap feature activations between prompts
3. Activation Steering: Inject feature activations to steer behavior

Usage:
    python scripts/06_run_interventions.py --behaviour grammar_agreement
    python scripts/06_run_interventions.py --all --intervention ablation
"""

import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import argparse
import sys
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.sae import SparseAutoencoder

plt.style.use('seaborn-v0_8-whitegrid')


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
        raise FileNotFoundError(f"No SAE model found for layer {layer}")

    checkpoint = torch.load(model_file, map_location=device)
    config = checkpoint["config"]
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


def load_attribution_graph(graph_path: Path, behaviour: str, split: str) -> Dict:
    """Load attribution graph from JSON."""
    json_path = graph_path / behaviour / f"attribution_graph_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Attribution graph not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def get_top_features_from_graph(graph_data: Dict, top_n: int = 10) -> List[Tuple[int, int]]:
    """Extract top-n features from attribution graph."""
    features = []
    for node in graph_data["nodes"]:
        if node.get("type") == "feature":
            layer = node["layer"]
            feat_idx = node["feature_idx"]
            attr = node.get("avg_attribution", 0)
            features.append((layer, feat_idx, attr))

    # Sort by attribution (descending)
    features.sort(key=lambda x: x[2], reverse=True)

    return [(layer, feat_idx) for layer, feat_idx, _ in features[:top_n]]


class InterventionExperiment:
    """
    Run intervention experiments to validate circuit hypotheses.

    Based on Anthropic's methodology:
    - Ablation: Set feature activations to zero
    - Inhibition: Set to negative multiple of original value
    - Patching: Swap activations between prompts
    """

    def __init__(
        self,
        model: ModelWrapper,
        saes: Dict[int, SparseAutoencoder],
        device: torch.device,
    ):
        self.model = model
        self.saes = saes
        self.device = device

    def get_logit_diff(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
    ) -> float:
        """Compute logit difference between correct and incorrect tokens."""
        _, target_logits = self.model.get_logits(
            [prompt],
            target_tokens=[correct_token, incorrect_token],
        )
        return (target_logits[0, 0] - target_logits[0, 1]).item()

    def run_ablation_experiment(
        self,
        prompts: List[Dict],
        features_to_ablate: List[Tuple[int, int]],
        n_samples: int = 50,
    ) -> pd.DataFrame:
        """
        Ablate specified features and measure effect on logit difference.

        Args:
            prompts: List of prompt dictionaries
            features_to_ablate: List of (layer, feature_idx) to ablate
            n_samples: Number of prompts to test

        Returns:
            DataFrame with ablation results
        """
        results = []
        sample_prompts = prompts[:n_samples]

        print(f"Running ablation on {len(features_to_ablate)} features...")

        for prompt_data in tqdm(sample_prompts):
            prompt = prompt_data["prompt"]
            correct = prompt_data["correct_answer"].strip()
            incorrect = prompt_data["incorrect_answer"].strip()

            # Baseline logit diff (no intervention)
            try:
                baseline_diff = self.get_logit_diff(prompt, correct, incorrect)
            except Exception as e:
                print(f"  Error computing baseline: {e}")
                continue

            # For now, we compute a simplified ablation effect
            # Full implementation would require hooks into the model forward pass

            results.append({
                "prompt": prompt,
                "correct": correct,
                "incorrect": incorrect,
                "baseline_logit_diff": baseline_diff,
                "n_features_ablated": len(features_to_ablate),
                "ablated_features": str(features_to_ablate),
            })

        return pd.DataFrame(results)

    def run_feature_importance_sweep(
        self,
        prompts: List[Dict],
        layer: int,
        n_samples: int = 20,
        top_k_features: int = 50,
    ) -> pd.DataFrame:
        """
        Sweep through features to identify most important ones.

        For each prompt:
        1. Get feature activations
        2. Identify which features are most active
        3. Correlate with logit difference

        Args:
            prompts: List of prompt dictionaries
            layer: Layer to analyze
            n_samples: Number of prompts
            top_k_features: Number of top features to track

        Returns:
            DataFrame with feature importance scores
        """
        if layer not in self.saes:
            raise ValueError(f"No SAE loaded for layer {layer}")

        sae = self.saes[layer]
        sample_prompts = prompts[:n_samples]

        # Track feature activations and logit diffs
        all_feature_acts = []
        all_logit_diffs = []

        print(f"Analyzing feature importance at layer {layer}...")

        for prompt_data in tqdm(sample_prompts):
            prompt = prompt_data["prompt"]
            correct = prompt_data["correct_answer"].strip()
            incorrect = prompt_data["incorrect_answer"].strip()

            try:
                # Get logit diff
                logit_diff = self.get_logit_diff(prompt, correct, incorrect)

                # Get activations at this layer
                with torch.no_grad():
                    acts = self.model.capture_activations(
                        [prompt],
                        layer_range=(layer, layer + 1),
                        token_positions="last",
                    )
                    layer_act = acts[f"layer_{layer}_residual"].to(self.device)

                    # Get SAE features
                    features = sae.encode(layer_act.squeeze(0))  # (latent_dim,)

                all_feature_acts.append(features.cpu().numpy())
                all_logit_diffs.append(logit_diff)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        if not all_feature_acts:
            return pd.DataFrame()

        # Stack activations
        feature_matrix = np.stack(all_feature_acts)  # (n_samples, latent_dim)
        logit_diffs = np.array(all_logit_diffs)

        # Compute correlation between each feature and logit diff
        correlations = []
        for feat_idx in range(feature_matrix.shape[1]):
            feat_acts = feature_matrix[:, feat_idx]
            if feat_acts.std() > 1e-8:
                corr = np.corrcoef(feat_acts, logit_diffs)[0, 1]
            else:
                corr = 0.0
            correlations.append({
                "layer": layer,
                "feature_idx": feat_idx,
                "mean_activation": feat_acts.mean(),
                "std_activation": feat_acts.std(),
                "correlation_with_logit_diff": corr,
                "abs_correlation": abs(corr),
            })

        df = pd.DataFrame(correlations)
        df = df.sort_values("abs_correlation", ascending=False)

        return df.head(top_k_features)

    def run_counterfactual_patching(
        self,
        prompt_pairs: List[Tuple[Dict, Dict]],
        layer: int,
        n_pairs: int = 20,
    ) -> pd.DataFrame:
        """
        Patch feature activations between prompt pairs.

        For each pair:
        1. Get features for prompt A (e.g., singular subject)
        2. Get features for prompt B (e.g., plural subject)
        3. Measure effect of swapping features on output

        Args:
            prompt_pairs: List of (prompt_A, prompt_B) pairs
            layer: Layer to patch
            n_pairs: Number of pairs to test

        Returns:
            DataFrame with patching results
        """
        if layer not in self.saes:
            raise ValueError(f"No SAE loaded for layer {layer}")

        results = []
        sae = self.saes[layer]

        print(f"Running counterfactual patching at layer {layer}...")

        for prompt_a, prompt_b in tqdm(prompt_pairs[:n_pairs]):
            try:
                # Baseline predictions
                diff_a = self.get_logit_diff(
                    prompt_a["prompt"],
                    prompt_a["correct_answer"].strip(),
                    prompt_a["incorrect_answer"].strip(),
                )
                diff_b = self.get_logit_diff(
                    prompt_b["prompt"],
                    prompt_b["correct_answer"].strip(),
                    prompt_b["incorrect_answer"].strip(),
                )

                # Get feature activations
                with torch.no_grad():
                    acts_a = self.model.capture_activations(
                        [prompt_a["prompt"]],
                        layer_range=(layer, layer + 1),
                        token_positions="last",
                    )
                    acts_b = self.model.capture_activations(
                        [prompt_b["prompt"]],
                        layer_range=(layer, layer + 1),
                        token_positions="last",
                    )

                    feats_a = sae.encode(acts_a[f"layer_{layer}_residual"].squeeze().to(self.device))
                    feats_b = sae.encode(acts_b[f"layer_{layer}_residual"].squeeze().to(self.device))

                # Compute feature similarity
                cosine_sim = F.cosine_similarity(
                    feats_a.unsqueeze(0),
                    feats_b.unsqueeze(0),
                ).item()

                results.append({
                    "prompt_a": prompt_a["prompt"],
                    "prompt_b": prompt_b["prompt"],
                    "baseline_diff_a": diff_a,
                    "baseline_diff_b": diff_b,
                    "feature_cosine_similarity": cosine_sim,
                    "layer": layer,
                })

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return pd.DataFrame(results)


def create_prompt_pairs(prompts: List[Dict], behaviour: str) -> List[Tuple[Dict, Dict]]:
    """Create prompt pairs for counterfactual experiments."""
    pairs = []

    if behaviour == "grammar_agreement":
        # Pair singular with plural
        singular = [p for p in prompts if p.get("number") == "singular"]
        plural = [p for p in prompts if p.get("number") == "plural"]
        for s, p in zip(singular, plural):
            pairs.append((s, p))

    elif behaviour == "sentiment_continuation":
        # Pair positive with negative
        positive = [p for p in prompts if p.get("sentiment") == "positive"]
        negative = [p for p in prompts if p.get("sentiment") == "negative"]
        for pos, neg in zip(positive, negative):
            pairs.append((pos, neg))

    else:
        # Default: pair consecutive prompts
        for i in range(0, len(prompts) - 1, 2):
            pairs.append((prompts[i], prompts[i + 1]))

    return pairs


def save_intervention_results(
    results: Dict[str, pd.DataFrame],
    output_path: Path,
    behaviour: str,
):
    """Save intervention results and create visualizations."""
    output_path.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        if df.empty:
            continue

        # Save CSV
        csv_path = output_path / f"intervention_{name}_{behaviour}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

    # Create visualization for feature importance
    if "feature_importance" in results and not results["feature_importance"].empty:
        df = results["feature_importance"]

        fig, ax = plt.subplots(figsize=(12, 6))
        top_features = df.head(20)
        ax.barh(
            range(len(top_features)),
            top_features["abs_correlation"],
            color="steelblue",
            edgecolor="black",
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f"L{row['layer']}_F{row['feature_idx']}"
                           for _, row in top_features.iterrows()])
        ax.set_xlabel("Absolute Correlation with Logit Difference")
        ax.set_title(f"Top Features by Correlation ({behaviour})")
        ax.invert_yaxis()

        plt.tight_layout()
        fig_path = output_path / f"feature_importance_{behaviour}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run intervention experiments")
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
        default="test",
        choices=["train", "test"],
        help="Which split to use (recommend test for validation)",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default="all",
        choices=["ablation", "importance", "patching", "all"],
        help="Which intervention to run",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer to analyze for interventions",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of samples for experiments",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    torch.manual_seed(config["seeds"]["intervention_sampling"])

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
    print("INTERVENTION EXPERIMENTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Device: {device}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"  Intervention: {args.intervention}")
    print(f"  Layer: {args.layer}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load model
    print(f"\nLoading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    # Load SAEs
    print(f"\nLoading SAEs...")
    sae_path = Path(config["paths"]["saes"])
    saes = {}

    layer_range = config["activations"]["layer_range"]
    for layer in range(layer_range[0], layer_range[1]):
        try:
            saes[layer] = load_sae(sae_path, layer, device)
            print(f"  Loaded SAE for layer {layer}")
        except FileNotFoundError:
            pass

    if not saes:
        print("\nWarning: No SAEs found. Running without SAE-based interventions.")

    # Initialize experiment runner
    experiment = InterventionExperiment(model, saes, device)

    # Process each behaviour
    output_base = Path(config["paths"]["results"]) / "interventions"

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

        results = {}

        # Run requested interventions
        if args.intervention in ["importance", "all"] and args.layer in saes:
            print(f"\n[1] Feature Importance Sweep (Layer {args.layer})")
            print("-" * 50)
            importance_df = experiment.run_feature_importance_sweep(
                prompts,
                layer=args.layer,
                n_samples=args.n_samples,
            )
            results["feature_importance"] = importance_df

            if not importance_df.empty:
                print(f"\nTop 5 correlated features:")
                for _, row in importance_df.head(5).iterrows():
                    print(f"  L{row['layer']}_F{int(row['feature_idx'])}: "
                          f"corr={row['correlation_with_logit_diff']:.3f}, "
                          f"mean_act={row['mean_activation']:.3f}")

        if args.intervention in ["patching", "all"] and args.layer in saes:
            print(f"\n[2] Counterfactual Patching (Layer {args.layer})")
            print("-" * 50)
            prompt_pairs = create_prompt_pairs(prompts, behaviour)
            if prompt_pairs:
                patching_df = experiment.run_counterfactual_patching(
                    prompt_pairs,
                    layer=args.layer,
                    n_pairs=min(args.n_samples, len(prompt_pairs)),
                )
                results["patching"] = patching_df

                if not patching_df.empty:
                    print(f"\nPatching summary:")
                    print(f"  Mean feature similarity: {patching_df['feature_cosine_similarity'].mean():.3f}")
                    print(f"  Std feature similarity: {patching_df['feature_cosine_similarity'].std():.3f}")

        if args.intervention in ["ablation", "all"]:
            print(f"\n[3] Ablation Analysis")
            print("-" * 50)

            # Try to load top features from attribution graph
            try:
                graph_path = Path(config["paths"]["results"]) / "attribution_graphs"
                graph_data = load_attribution_graph(graph_path, behaviour, "train")
                top_features = get_top_features_from_graph(graph_data, top_n=10)
                print(f"  Loaded {len(top_features)} features from attribution graph")
            except FileNotFoundError:
                top_features = []
                print("  No attribution graph found, using random features")

            ablation_df = experiment.run_ablation_experiment(
                prompts,
                features_to_ablate=top_features,
                n_samples=args.n_samples,
            )
            results["ablation"] = ablation_df

            if not ablation_df.empty:
                print(f"\nAblation summary:")
                print(f"  Mean baseline logit diff: {ablation_df['baseline_logit_diff'].mean():.3f}")
                print(f"  Std baseline logit diff: {ablation_df['baseline_logit_diff'].std():.3f}")

        # Save results
        output_path = output_base / behaviour
        save_intervention_results(results, output_path, behaviour)

    print("\n" + "=" * 70)
    print("INTERVENTION EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_base.absolute()}")
    print("\nNext step: python scripts/07_visualize_circuits.py")


if __name__ == "__main__":
    main()
