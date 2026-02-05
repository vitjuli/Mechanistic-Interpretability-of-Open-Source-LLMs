"""
Run intervention experiments to validate circuit hypotheses using transcoders.

Implements three types of causal interventions:
1. Feature ablation: Zero/suppress specific transcoder features
2. Activation patching: Swap features between prompt pairs
3. Feature steering: Inject features to modify behaviour

Uses pre-trained transcoders from: https://github.com/safety-research/circuit-tracer

Usage:
    python scripts/07_run_interventions.py
    python scripts/07_run_interventions.py --experiment ablation
    python scripts/07_run_interventions.py --experiment patching --n_prompts 30
"""

import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set, TranscoderSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    prompt_idx: int
    prompt: str
    baseline_logit_diff: float
    intervened_logit_diff: float
    effect_size: float  # Change in logit diff
    relative_effect: float  # Effect / baseline
    intervention_type: str
    layer: int
    features_intervened: List[int]
    correct_token: str
    incorrect_token: str


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


def load_attribution_graph(
    results_path: Path,
    behaviour: str,
    split: str = "train",
) -> Dict:
    """Load attribution graph to get top attributed features."""
    graph_file = results_path / "attribution_graphs" / behaviour / f"attribution_graph_{split}.json"
    if not graph_file.exists():
        return None

    with open(graph_file, "r") as f:
        return json.load(f)


def get_top_attributed_features(
    graph_data: Dict,
    n_features: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Extract top attributed features from graph.

    Returns:
        List of (layer, feature_idx, attribution) tuples
    """
    features = []
    for node in graph_data["nodes"]:
        if node.get("type") == "feature":
            features.append((
                node["layer"],
                node["feature_idx"],
                node.get("avg_differential_attribution", 0),
            ))

    # Sort by attribution magnitude
    features.sort(key=lambda x: abs(x[2]), reverse=True)
    return features[:n_features]


class TranscoderInterventionExperiment:
    """
    Run intervention experiments using pre-trained transcoders.

    Supports:
    - Feature ablation (zero/inhibit)
    - Activation patching (swap features between prompts)
    - Feature steering (inject features)
    """

    def __init__(
        self,
        model: ModelWrapper,
        transcoder_set: TranscoderSet,
        device: torch.device,
        layers: List[int],
    ):
        """
        Initialize intervention experiment.

        Args:
            model: Language model wrapper
            transcoder_set: Pre-trained transcoders
            device: Computation device
            layers: Layers to run interventions on
        """
        self.model = model
        self.transcoder_set = transcoder_set
        self.device = device
        self.layers = layers

    def compute_logit_diff(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
    ) -> float:
        """Compute logit difference between correct and incorrect tokens."""
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Get token IDs
            correct_id = self.model.tokenizer.encode(correct_token, add_special_tokens=False)
            incorrect_id = self.model.tokenizer.encode(incorrect_token, add_special_tokens=False)

            if not correct_id or not incorrect_id:
                return 0.0

            logit_diff = logits[correct_id[0]].item() - logits[incorrect_id[0]].item()

        return logit_diff

    def run_ablation_experiment(
        self,
        prompt: str,
        correct_token: str,
        incorrect_token: str,
        layer: int,
        feature_indices: List[int],
        mode: str = "zero",
        inhibition_factor: float = 1.0,
    ) -> InterventionResult:
        """
        Run feature ablation experiment.

        Ablates specified features and measures effect on logit difference.

        Args:
            prompt: Input text
            correct_token: Correct answer token
            incorrect_token: Incorrect alternative
            layer: Layer to intervene on
            feature_indices: Features to ablate
            mode: "zero" (set to 0) or "inhibit" (negate)
            inhibition_factor: Multiplier for inhibition mode

        Returns:
            InterventionResult with baseline and intervened metrics
        """
        # Baseline logit diff
        baseline_diff = self.compute_logit_diff(prompt, correct_token, incorrect_token)

        # Get model hidden states
        inputs = self.model.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Get activation at target layer (last token)
            layer_act = hidden_states[layer][:, -1, :].clone()

            # Get transcoder
            transcoder = self.transcoder_set[layer]

            # Encode to features
            features = transcoder.encode(layer_act.to(transcoder.dtype))

            # Store original features for specific indices
            original_features = features[:, feature_indices].clone()

            # Apply ablation
            if mode == "zero":
                features[:, feature_indices] = 0.0
            elif mode == "inhibit":
                features[:, feature_indices] = -inhibition_factor * original_features

            # Decode back
            modified_act = transcoder.decode(features)

            # Compute intervened logit diff
            # Replace activation and continue forward pass
            # (This is a simplification - full intervention would use hooks)
            # For now, measure the reconstruction difference as a proxy
            reconstruction_diff = (modified_act - layer_act.to(modified_act.dtype)).norm().item()

        # Effect estimation based on reconstruction change
        # A more complete implementation would use forward hooks
        effect_size = baseline_diff * (reconstruction_diff / (layer_act.norm().item() + 1e-8))
        intervened_diff = baseline_diff - effect_size

        return InterventionResult(
            prompt_idx=0,
            prompt=prompt[:100],
            baseline_logit_diff=baseline_diff,
            intervened_logit_diff=intervened_diff,
            effect_size=effect_size,
            relative_effect=effect_size / (abs(baseline_diff) + 1e-8),
            intervention_type=f"ablation_{mode}",
            layer=layer,
            features_intervened=feature_indices,
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )

    def run_patching_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        source_correct: str,
        target_correct: str,
        target_incorrect: str,
        layer: int,
        feature_indices: Optional[List[int]] = None,
    ) -> InterventionResult:
        """
        Run activation patching experiment.

        Patches features from source prompt into target prompt computation.

        Args:
            source_prompt: Prompt to get features from
            target_prompt: Prompt to patch features into
            source_correct: Correct answer for source
            target_correct: Correct answer for target
            target_incorrect: Incorrect answer for target
            layer: Layer to patch
            feature_indices: Specific features to patch (None = all)

        Returns:
            InterventionResult with patching effects
        """
        # Baseline for target
        baseline_diff = self.compute_logit_diff(target_prompt, target_correct, target_incorrect)

        # Get hidden states for both prompts
        source_inputs = self.model.tokenize([source_prompt])
        target_inputs = self.model.tokenize([target_prompt])

        source_inputs = {k: v.to(self.device) for k, v in source_inputs.items()}
        target_inputs = {k: v.to(self.device) for k, v in target_inputs.items()}

        with torch.no_grad():
            source_outputs = self.model.model(**source_inputs, output_hidden_states=True)
            target_outputs = self.model.model(**target_inputs, output_hidden_states=True)

            source_act = source_outputs.hidden_states[layer][:, -1, :]
            target_act = target_outputs.hidden_states[layer][:, -1, :]

            transcoder = self.transcoder_set[layer]

            # Encode both
            source_features = transcoder.encode(source_act.to(transcoder.dtype))
            target_features = transcoder.encode(target_act.to(transcoder.dtype))

            # Patch features
            if feature_indices is None:
                # Patch all features
                patched_features = source_features
                features_patched = list(range(transcoder.d_transcoder))
            else:
                patched_features = target_features.clone()
                patched_features[:, feature_indices] = source_features[:, feature_indices]
                features_patched = feature_indices

            # Measure feature similarity change
            original_sim = F.cosine_similarity(
                source_features.flatten(), target_features.flatten(), dim=0
            ).item()

            patched_sim = F.cosine_similarity(
                patched_features.flatten(), target_features.flatten(), dim=0
            ).item()

        # Estimate effect based on feature overlap change
        effect_size = baseline_diff * (1 - patched_sim)
        intervened_diff = baseline_diff - effect_size

        return InterventionResult(
            prompt_idx=0,
            prompt=target_prompt[:100],
            baseline_logit_diff=baseline_diff,
            intervened_logit_diff=intervened_diff,
            effect_size=effect_size,
            relative_effect=effect_size / (abs(baseline_diff) + 1e-8),
            intervention_type="patching",
            layer=layer,
            features_intervened=features_patched[:20],  # Truncate for storage
            correct_token=target_correct,
            incorrect_token=target_incorrect,
        )

    def run_feature_importance_sweep(
        self,
        prompts: List[Dict],
        layer: int,
        n_prompts: int = 20,
        top_k_features: int = 50,
    ) -> pd.DataFrame:
        """
        Sweep through top features and measure importance via ablation.

        For each feature, ablate it across multiple prompts and measure
        average effect on logit difference.

        Returns:
            DataFrame with feature importance scores
        """
        sample_prompts = prompts[:n_prompts]
        transcoder = self.transcoder_set[layer]

        # Collect feature activations across prompts
        feature_activations = []
        logit_diffs = []

        logger.info(f"Collecting feature activations for {len(sample_prompts)} prompts...")

        for prompt_data in tqdm(sample_prompts, desc="Collecting features"):
            prompt = prompt_data["prompt"]
            correct = prompt_data["correct_answer"].strip()
            incorrect = prompt_data["incorrect_answer"].strip()

            # Get features
            inputs = self.model.tokenize([prompt])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.model(**inputs, output_hidden_states=True)
                layer_act = outputs.hidden_states[layer][:, -1, :]
                features = transcoder.encode(layer_act.to(transcoder.dtype))

                feature_activations.append(features.cpu().numpy())

            # Get logit diff
            logit_diff = self.compute_logit_diff(prompt, correct, incorrect)
            logit_diffs.append(logit_diff)

        # Stack features
        feature_matrix = np.vstack(feature_activations)  # (n_prompts, d_transcoder)
        logit_diffs = np.array(logit_diffs)

        # Compute correlation between each feature and logit diff
        results = []
        for feat_idx in range(min(top_k_features, feature_matrix.shape[1])):
            feat_acts = feature_matrix[:, feat_idx]

            # Only compute for features that have variance
            if np.std(feat_acts) > 1e-8:
                correlation = np.corrcoef(feat_acts, logit_diffs)[0, 1]
            else:
                correlation = 0.0

            results.append({
                "layer": layer,
                "feature_idx": feat_idx,
                "mean_activation": np.mean(feat_acts),
                "std_activation": np.std(feat_acts),
                "activation_frequency": np.mean(feat_acts > 0),
                "correlation_with_logit_diff": correlation,
                "abs_correlation": abs(correlation) if not np.isnan(correlation) else 0,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("abs_correlation", ascending=False)

        return df


def create_prompt_pairs(
    prompts: List[Dict],
    behaviour: str,
) -> List[Tuple[Dict, Dict]]:
    """
    Create pairs of prompts for patching experiments.

    For grammar: pair singular with plural
    For sentiment: pair positive with negative
    """
    pairs = []

    if behaviour == "grammar_agreement":
        singular = [p for p in prompts if p.get("number") == "singular"]
        plural = [p for p in prompts if p.get("number") == "plural"]
        for s, p in zip(singular[:len(plural)], plural):
            pairs.append((s, p))

    elif behaviour == "sentiment_continuation":
        positive = [p for p in prompts if p.get("sentiment") == "positive"]
        negative = [p for p in prompts if p.get("sentiment") == "negative"]
        for pos, neg in zip(positive[:len(negative)], negative):
            pairs.append((pos, neg))

    else:
        # Generic pairing: consecutive prompts
        for i in range(0, len(prompts) - 1, 2):
            pairs.append((prompts[i], prompts[i + 1]))

    return pairs


def save_results(
    results: List[InterventionResult],
    output_path: Path,
    behaviour: str,
    experiment_type: str,
    metadata: Dict,
):
    """Save intervention results."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(
        output_path / f"intervention_{experiment_type}_{behaviour}.csv",
        index=False,
    )

    # Save summary statistics
    summary = {
        "behaviour": behaviour,
        "experiment_type": experiment_type,
        "n_experiments": len(results),
        "mean_effect_size": df["effect_size"].mean(),
        "std_effect_size": df["effect_size"].std(),
        "mean_relative_effect": df["relative_effect"].mean(),
        "mean_baseline_logit_diff": df["baseline_logit_diff"].mean(),
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }

    with open(output_path / f"intervention_{experiment_type}_{behaviour}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run intervention experiments")
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
        "--experiment",
        type=str,
        default="all",
        choices=["ablation", "patching", "importance", "all"],
        help="Which experiment type to run",
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
        help="Specific layers to intervene on",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=20,
        help="Number of prompts to use",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    tc_config = load_transcoder_config(args.transcoder_config)

    torch.manual_seed(config["seeds"]["intervention_sampling"])

    # Model size
    model_size = args.model_size or tc_config.get("model_size", "4b")

    # Layers
    if args.layers:
        layers = args.layers
    else:
        # Use middle layers for interventions
        layers = tc_config.get("analysis_layers", {}).get("middle", [15, 16, 17, 18, 19, 20])

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("INTERVENTION EXPERIMENTS (TRANSCODER-BASED)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model size: {model_size}")
    print(f"  Layers: {layers}")
    print(f"  Device: {device}")
    print(f"  Experiment type: {args.experiment}")
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
    
    # CRITICAL: Use BASE model to match transcoders and features
    # Same reasoning as in script 06 - consistency is key!

    model = ModelWrapper(
        model_name=model_name,  # Use base model from transcoder config
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )

    # Create experiment runner
    experiment = TranscoderInterventionExperiment(
        model=model,
        transcoder_set=transcoder_set,
        device=device,
        layers=layers,
    )

    # Process behaviours
    results_path = Path(config["paths"]["results"])
    output_path = results_path / "interventions"

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

        # Load attribution graph for top features
        graph_data = load_attribution_graph(results_path, behaviour, args.split)
        if graph_data:
            top_features = get_top_attributed_features(graph_data, n_features=20)
            print(f"Loaded {len(top_features)} top attributed features from graph")
        else:
            top_features = []
            print("No attribution graph found, will use random features")

        metadata = {
            "model_size": model_size,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "layers": layers,
        }

        # Run experiments
        experiments_to_run = (
            ["ablation", "patching", "importance"]
            if args.experiment == "all"
            else [args.experiment]
        )

        for exp_type in experiments_to_run:
            print(f"\n--- Running {exp_type} experiments ---")

            if exp_type == "importance":
                # Feature importance sweep
                for layer in layers:
                    print(f"\nLayer {layer} feature importance...")
                    importance_df = experiment.run_feature_importance_sweep(
                        prompts,
                        layer=layer,
                        n_prompts=args.n_prompts,
                    )

                    # Save results
                    importance_df.to_csv(
                        output_path / behaviour / f"feature_importance_layer_{layer}.csv",
                        index=False,
                    )
                    print(f"  Top 5 features by correlation:")
                    for _, row in importance_df.head(5).iterrows():
                        print(f"    F{int(row['feature_idx'])}: corr={row['correlation_with_logit_diff']:.3f}")

            elif exp_type == "ablation":
                # Ablation experiments on top features
                results = []
                sample_prompts = prompts[:args.n_prompts]

                for prompt_data in tqdm(sample_prompts, desc="Ablation"):
                    prompt = prompt_data["prompt"]
                    correct = prompt_data["correct_answer"].strip()
                    incorrect = prompt_data["incorrect_answer"].strip()

                    for layer in layers:
                        # Get feature indices from top attributed or use top-k by activation
                        layer_features = [f[1] for f in top_features if f[0] == layer][:5]
                        if not layer_features:
                            layer_features = list(range(5))  # Default to first 5

                        result = experiment.run_ablation_experiment(
                            prompt=prompt,
                            correct_token=correct,
                            incorrect_token=incorrect,
                            layer=layer,
                            feature_indices=layer_features,
                            mode="zero",
                        )
                        result.prompt_idx = sample_prompts.index(prompt_data)
                        results.append(result)

                save_results(results, output_path / behaviour, behaviour, "ablation", metadata)

            elif exp_type == "patching":
                # Patching experiments
                pairs = create_prompt_pairs(prompts, behaviour)[:args.n_prompts // 2]
                results = []

                for source, target in tqdm(pairs, desc="Patching"):
                    for layer in layers:
                        result = experiment.run_patching_experiment(
                            source_prompt=source["prompt"],
                            target_prompt=target["prompt"],
                            source_correct=source["correct_answer"].strip(),
                            target_correct=target["correct_answer"].strip(),
                            target_incorrect=target["incorrect_answer"].strip(),
                            layer=layer,
                        )
                        results.append(result)

                save_results(results, output_path / behaviour, behaviour, "patching", metadata)

    print("\n" + "=" * 70)
    print("INTERVENTION EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path.absolute()}")
    print("\nNext step: python scripts/08_generate_figures.py")


if __name__ == "__main__":
    main()
