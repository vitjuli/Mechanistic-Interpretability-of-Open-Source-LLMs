"""
⚠️ LEGACY SCRIPT - NOT COMPATIBLE WITH CURRENT PIPELINE ⚠️

This script uses legacy SAE (Sparse Autoencoder) code and is NOT compatible
with the current transcoder-based pipeline. It is kept for reference only.

For transcoder feature interpretation, see:
- Feature results: data/results/transcoder_features/
- Script 04 already extracts and saves top-k features per layer
- Script 06 provides attribution-based feature analysis

If you need feature interpretation for transcoders, features are automatically
extracted and saved by scripts/04_extract_transcoder_features.py with metadata
including top activations and frequencies.

---

ORIGINAL DOCSTRING (for SAE-based analysis):

Interpret SAE features by analyzing their activation patterns.

For each trained SAE layer, this script:
1. Identifies the top-activating prompts for each feature
2. Computes feature-token associations
3. Labels features by dominant behaviour/token pattern
4. Generates feature dashboards

Based on Anthropic's feature interpretation methodology from
"On the Biology of a Large Language Model" (2025).

Usage:
    python scripts/05_interpret_features.py --behaviour grammar_agreement --layer 15
    python scripts/05_interpret_features.py --all --layer 15
"""

import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.sae import SparseAutoencoder


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompt_path: Path, behaviour: str, split: str) -> List[Dict]:
    file_path = prompt_path / f"{behaviour}_{split}.jsonl"
    if not file_path.exists():
        return []
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_sae(sae_path: Path, layer: int, device: torch.device) -> SparseAutoencoder:
    model_file = sae_path / f"layer_{layer}_best.pt"
    if not model_file.exists():
        model_file = sae_path / f"layer_{layer}_final.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"No SAE for layer {layer}")

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
    sae = sae.to(device).eval()
    return sae


def get_feature_activations_for_prompts(
    model: ModelWrapper,
    sae: SparseAutoencoder,
    prompts: List[Dict],
    layer: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Get SAE feature activations for all prompts.

    Returns:
        feature_matrix: (n_prompts, latent_dim)
        valid_prompts: prompts that were successfully processed
    """
    all_features = []
    valid_prompts = []

    for prompt_data in tqdm(prompts, desc="Computing features"):
        try:
            with torch.no_grad():
                acts = model.capture_activations(
                    [prompt_data["prompt"]],
                    layer_range=(layer, layer + 1),
                    token_positions="last",
                )
                layer_act = acts[f"layer_{layer}_residual"].to(device)
                features = sae.encode(layer_act.squeeze(0))
                all_features.append(features.cpu().numpy())
                valid_prompts.append(prompt_data)
        except Exception as e:
            continue

    if not all_features:
        return np.array([]), []

    return np.stack(all_features), valid_prompts


def interpret_features(
    feature_matrix: np.ndarray,
    prompts: List[Dict],
    top_k_prompts: int = 5,
    top_k_features: int = 50,
) -> pd.DataFrame:
    """
    Interpret features based on their activation patterns.

    For each feature, find:
    - Top-activating prompts
    - Mean/max activation
    - Frequency of activation
    - Associated metadata (behaviour, category, etc.)
    """
    n_prompts, n_features = feature_matrix.shape

    results = []

    # Find most active features overall
    mean_acts = feature_matrix.mean(axis=0)
    max_acts = feature_matrix.max(axis=0)
    activation_freq = (feature_matrix > 0).mean(axis=0)

    # Sort by mean activation
    top_feature_indices = np.argsort(-mean_acts)[:top_k_features]

    for feat_idx in top_feature_indices:
        feat_acts = feature_matrix[:, feat_idx]

        # Skip dead features
        if feat_acts.max() == 0:
            continue

        # Find top-activating prompts
        top_prompt_indices = np.argsort(-feat_acts)[:top_k_prompts]
        top_prompts = [prompts[i] for i in top_prompt_indices]
        top_activations = feat_acts[top_prompt_indices]

        # Extract common metadata from top prompts
        metadata_keys = set()
        for p in top_prompts:
            metadata_keys.update(p.keys())
        metadata_keys -= {"prompt", "correct_answer", "incorrect_answer"}

        # Check for common categories
        common_metadata = {}
        for key in metadata_keys:
            values = [p.get(key) for p in top_prompts if key in p]
            if values:
                # Most common value
                from collections import Counter
                counter = Counter(values)
                most_common = counter.most_common(1)[0]
                if most_common[1] > len(values) * 0.5:  # >50% agreement
                    common_metadata[key] = most_common[0]

        results.append({
            "feature_idx": int(feat_idx),
            "mean_activation": float(mean_acts[feat_idx]),
            "max_activation": float(max_acts[feat_idx]),
            "activation_frequency": float(activation_freq[feat_idx]),
            "top_prompts": [p["prompt"] for p in top_prompts],
            "top_activations": top_activations.tolist(),
            "common_metadata": common_metadata,
            "label": _auto_label(top_prompts, common_metadata),
        })

    return pd.DataFrame(results)


def _auto_label(prompts: List[Dict], metadata: Dict) -> str:
    """Generate an automatic label for a feature based on its top prompts."""
    # Check for number agreement
    if "number" in metadata:
        return f"number_{metadata['number']}"
    if "sentiment" in metadata:
        return f"sentiment_{metadata['sentiment']}"
    if "country" in metadata:
        return f"country_{metadata['country']}"
    if "operand_a" in metadata:
        return "arithmetic_operand"

    # Default: use first prompt words
    first_prompt = prompts[0]["prompt"] if prompts else ""
    return f"unlabeled_{first_prompt[:20]}"


def save_interpretation(
    df: pd.DataFrame,
    output_path: Path,
    behaviour: str,
    layer: int,
):
    """Save feature interpretation results."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV (without nested structures)
    csv_df = df[["feature_idx", "mean_activation", "max_activation",
                  "activation_frequency", "label"]].copy()
    csv_path = output_path / f"features_{behaviour}_layer_{layer}.csv"
    csv_df.to_csv(csv_path, index=False)

    # Save full JSON
    json_path = output_path / f"features_{behaviour}_layer_{layer}.json"
    df.to_json(json_path, orient="records", indent=2)

    print(f"  Saved: {csv_path.name}, {json_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Interpret SAE features")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--behaviour", type=str,
                        choices=["grammar_agreement", "factual_recall",
                                 "sentiment_continuation", "arithmetic"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--top_k_features", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["seeds"]["torch_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        behaviours = ["grammar_agreement", "factual_recall",
                       "sentiment_continuation", "arithmetic"]
    elif args.behaviour:
        behaviours = [args.behaviour]
    else:
        print("Error: Must specify --behaviour or --all")
        return

    print("=" * 70)
    print(f"FEATURE INTERPRETATION - Layer {args.layer}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae_path = Path(config["paths"]["saes"])
    try:
        sae = load_sae(sae_path, args.layer, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Process each behaviour
    output_base = Path(config["paths"]["results"]) / "feature_interpretations"

    for behaviour in behaviours:
        print(f"\n{'='*70}")
        print(f"BEHAVIOUR: {behaviour}")
        print(f"{'='*70}")

        prompts = load_prompts(Path(config["paths"]["prompts"]), behaviour, args.split)
        if not prompts:
            print(f"  No prompts found for {behaviour}")
            continue

        print(f"  Loaded {len(prompts)} prompts")

        # Get feature activations
        feature_matrix, valid_prompts = get_feature_activations_for_prompts(
            model, sae, prompts, args.layer, device
        )

        if feature_matrix.size == 0:
            print(f"  No valid activations obtained")
            continue

        print(f"  Feature matrix: {feature_matrix.shape}")

        # Interpret features
        interpretation_df = interpret_features(
            feature_matrix, valid_prompts,
            top_k_features=args.top_k_features,
        )

        # Print top features
        print(f"\n  Top 10 features:")
        for _, row in interpretation_df.head(10).iterrows():
            print(f"    F{row['feature_idx']:5d} | "
                  f"mean={row['mean_activation']:.4f} | "
                  f"freq={row['activation_frequency']:.2%} | "
                  f"label={row['label']}")

        # Save
        save_interpretation(interpretation_df, output_base, behaviour, args.layer)

    print(f"\n{'='*70}")
    print("FEATURE INTERPRETATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {output_base.absolute()}")
    print("Next step: python scripts/06_build_attribution_graph.py")


if __name__ == "__main__":
    main()
