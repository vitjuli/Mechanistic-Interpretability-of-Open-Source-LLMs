"""
Extract transcoder features from Qwen3 models using pre-trained transcoders.

This script replaces SAE training by using pre-trained per-layer transcoders
(PLTs) from the circuit-tracer project. Features are extracted for each
behaviour and saved for downstream attribution and intervention analysis.

Pre-trained transcoders available:
- mwhanna/qwen3-0.6b-transcoders-lowl0
- mwhanna/qwen3-1.7b-transcoders-lowl0
- mwhanna/qwen3-4b-transcoders
- mwhanna/qwen3-8b-transcoders
- mwhanna/qwen3-14b-transcoders-lowl0

Usage:
    python scripts/04_extract_transcoder_features.py
    python scripts/04_extract_transcoder_features.py --layers 15 16 17 18 19 20
    python scripts/04_extract_transcoder_features.py --model_size 4b --split test

Reference:
    https://github.com/safety-research/circuit-tracer
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
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import TranscoderSet, load_transcoder_set


logger = logging.getLogger(__name__)


def contiguous_groups(indices: List[int]) -> List[List[int]]:
    """
    Group layer indices into contiguous ranges.
    
    Example:
        [10, 12, 13, 14, 20] -> [[10], [12, 13, 14], [20]]
    
    This avoids hooking unnecessary layers when layer_indices are not contiguous.
    """
    if not indices:
        return []
    
    indices = sorted(indices)
    groups = []
    current = [indices[0]]
    
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    
    return groups

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_transcoder_config(config_path: str = "configs/transcoder_config.yaml") -> Dict:
    """Load transcoder-specific configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(
    prompt_path: Path,
    behaviour: str,
    split: str = "train",
) -> List[Dict]:
    """Load prompts from JSONL file."""
    file_path = prompt_path / f"{behaviour}_{split}.jsonl"
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def capture_mlp_inputs(
    model: ModelWrapper,
    prompts: List[Dict],
    layer_indices: List[int],
    batch_size: int = 8,
    token_positions: str = "decision",
) -> Tuple[Dict[int, torch.Tensor], List[Dict]]:
    """
    Capture MLP input activations (pre-MLP residual stream) for transcoder analysis.

    Unlike SAE training which uses post-MLP activations, transcoders need
    pre-MLP activations (the MLP input) to decompose what the MLP computes.

    Args:
        model: ModelWrapper instance
        prompts: List of prompt dictionaries
        layer_indices: Which layers to capture
        batch_size: Prompts per batch
        token_positions: Which tokens to capture ("all", "last", "last_N", "decision")

    Returns:
        Tuple of:
            - Dict mapping layer_idx -> activation tensor (n_samples, d_model)
            - position_map: List of dicts mapping sample_idx to (prompt_idx, token_pos, token_id)
    """
    prompt_texts = [p["prompt"] for p in prompts]
    n_batches = (len(prompts) + batch_size - 1) // batch_size

    # Storage for activations per layer
    layer_activations = {layer_idx: [] for layer_idx in layer_indices}
    
    # CRITICAL: Accumulate position_map across batches
    position_map_all = []

    logger.info(f"Capturing MLP inputs for {len(prompts)} prompts across {len(layer_indices)} layers")
    
    # Optimize: Group layer_indices into contiguous ranges to avoid hooking unnecessary layers
    # E.g., [10, 20] would hook all 10-20 without grouping, but only 10 and 20 are used
    layer_groups = contiguous_groups(layer_indices)
    logger.info(f"Layer indices grouped into {len(layer_groups)} contiguous ranges: {layer_groups}")

    for batch_idx in tqdm(range(n_batches), desc="Capturing MLP inputs"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_texts = prompt_texts[start_idx:end_idx]

        # Track first batch_result for position_map (same across all groups)
        batch_result_for_posmap = None

        # Process each contiguous group separately
        for group in layer_groups:
            group_start = group[0]
            group_end = group[-1] + 1
            
            # Capture MLP inputs for this batch
            # For grammar_agreement with token_positions="decision":
            # Captures activations at last token of prompt (e.g., " dogs")
            # which are used to compute logits for next token (" sleep" vs " sleeps")
            batch_result = model.capture_mlp_inputs(
                batch_texts,
                layer_range=(group_start, group_end),
                token_positions=token_positions,  # Pass through argument
            )
            
            # Save first result for position_map extraction
            if batch_result_for_posmap is None:
                batch_result_for_posmap = batch_result

            # Extract per-layer activations with CORRECT key
            for layer_idx in group:
                if layer_idx not in layer_indices:
                    continue  # Skip if not in requested layers
                    
                layer_key = f"layer_{layer_idx}_mlp_input"  # Correct key for MLP inputs
                if layer_key in batch_result["activations"]:
                    acts = batch_result["activations"][layer_key]
                    # acts is already flattened numpy array from capture_mlp_inputs
                    layer_activations[layer_idx].append(torch.from_numpy(acts))
                else:
                    logger.warning(f"Layer {layer_idx} not found in batch result. Available keys: {list(batch_result['activations'].keys())}")
        
        # Accumulate position_map (adjust prompt_idx for batch offset)
        # Only need to do this once per batch (same for all groups)
        if layer_groups and batch_result_for_posmap is not None:
            # Use result from FIRST group (position_map is same for all)
            batch_position_map = batch_result_for_posmap["metadata"]["position_map"]
            for entry in batch_position_map:
                entry_adjusted = entry.copy()
                entry_adjusted["prompt_idx"] += start_idx  # Offset by batch start
                position_map_all.append(entry_adjusted)

    # Concatenate all batches
    for layer_idx in layer_indices:
        if len(layer_activations[layer_idx]) > 0:
            layer_activations[layer_idx] = torch.cat(layer_activations[layer_idx], dim=0)
            logger.info(f"Layer {layer_idx}: {layer_activations[layer_idx].shape}")
        else:
            logger.error(f"No activations captured for layer {layer_idx}!")
            layer_activations[layer_idx] = None

    logger.info(f"Captured {len(position_map_all)} samples with token_positions='{token_positions}'")
    return layer_activations, position_map_all



def extract_features(
    transcoder_set: TranscoderSet,
    mlp_inputs: Dict[int, torch.Tensor],
    top_k: int = 50,
    device: torch.device = None,
) -> Dict[int, Dict]:
    """
    Extract transcoder features from MLP inputs.

    Args:
        transcoder_set: Loaded TranscoderSet
        mlp_inputs: Dict mapping layer_idx -> activation tensor
        top_k: Number of top features to track per sample
        device: Device for computation

    Returns:
        Dict mapping layer_idx -> {
            "feature_activations": sparse tensor of all activations,
            "top_k_indices": top-k feature indices per sample,
            "top_k_values": top-k activation values per sample,
            "active_features": set of all features active in this layer,
            "feature_frequencies": how often each feature is active,
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for layer_idx, activations in tqdm(mlp_inputs.items(), desc="Extracting features"):
        # Handle None activations (if capture failed)
        if activations is None:
            logger.warning(f"Skipping layer {layer_idx}: no activations captured")
            continue
        
        logger.info(f"Processing layer {layer_idx} ({activations.shape[0]} samples)")

        # Get transcoder for this layer
        transcoder = transcoder_set[layer_idx]

        # Move activations to device and convert dtype in single step (optimization)
        activations = activations.to(device=device, dtype=transcoder.dtype, non_blocking=True)
        
        # Encode to transcoder features
        with torch.no_grad():
            features = transcoder.encode(activations)  # [N, d_transcoder])

        # Get top-k features per sample
        top_k_values, top_k_indices = torch.topk(features, k=min(top_k, features.shape[1]), dim=1)

        # Compute active features (store as numpy array for efficiency)
        active_mask = (features > 0).cpu()  # [N, d_transcoder]
        active_feature_indices = torch.where(active_mask.any(dim=0))[0].cpu().numpy().astype(np.int32)
        feature_frequencies = active_mask.float().mean(dim=0).cpu().numpy()

        # Store results
        # NOTE: Don't save full feature_activations by default - too large!
        # For 80 prompts × 5 tokens × 65536 features × 4 bytes ≈ 100+ MB per layer
        results[layer_idx] = {
            # "feature_activations": features.cpu(),  # Removed - use save_full_acts flag if needed
            "top_k_indices": top_k_indices.cpu(),
            "top_k_values": top_k_values.cpu(),
            "active_feature_indices": active_feature_indices,  # numpy int32 array (more efficient than set)
            "feature_frequencies": feature_frequencies,
            "n_active_features": len(active_feature_indices),
            "mean_active_per_sample": active_mask.sum(dim=1).float().mean().item(),
            "d_transcoder": transcoder.d_transcoder,
        }

        logger.info(f"  Layer {layer_idx}: {len(active_feature_indices)} active features, "
                   f"{results[layer_idx]['mean_active_per_sample']:.1f} mean active per sample")

        # Clear GPU memory (only if using CUDA)
        del features, activations
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def save_features(
    results: Dict[int, Dict],
    output_path: Path,
    behaviour: str,
    split: str,
    metadata: Dict,
):
    """
    Save extracted features to disk.

    Saves:
    - Per-layer feature activations (numpy arrays)
    - Top-k indices and values (for quick lookup)
    - Feature frequencies (for filtering)
    - Summary metadata
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for layer_idx, layer_data in results.items():
        layer_dir = output_path / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        # Save top-k features (most commonly needed)
        # Convert to float32 if bfloat16 (numpy doesn't support bfloat16)
        np.save(
            layer_dir / f"{behaviour}_{split}_top_k_indices.npy",
            layer_data["top_k_indices"].numpy()
        )
        top_k_values = layer_data["top_k_values"]
        if top_k_values.dtype == torch.bfloat16:
            top_k_values = top_k_values.float()
        np.save(
            layer_dir / f"{behaviour}_{split}_top_k_values.npy",
            top_k_values.numpy()
        )

        # Save feature frequencies
        np.save(
            layer_dir / f"{behaviour}_{split}_feature_frequencies.npy",
            layer_data["feature_frequencies"]
        )

        # Save full activations (OPTIONAL - only if present and not too large)
        # NOTE: feature_activations removed from extract_features by default to save memory
        # Only present if --save_full_acts flag used (future feature)
        if "feature_activations" in layer_data:
            full_acts_tensor = layer_data["feature_activations"]
            if full_acts_tensor.dtype == torch.bfloat16:
                full_acts_tensor = full_acts_tensor.float()
            full_acts = full_acts_tensor.numpy()
            if full_acts.nbytes < 1e9:  # < 1GB
                np.save(
                    layer_dir / f"{behaviour}_{split}_full_activations.npy",
                    full_acts.astype(np.float16)  # Compress to float16
                )
                logger.info(f"Saved full activations for layer {layer_idx} ({full_acts.nbytes / 1e6:.1f} MB)")
            else:
                logger.warning(f"Skipping full activations for layer {layer_idx} (too large: {full_acts.nbytes / 1e9:.2f} GB)")
        else:
            # This is the default now - feature_activations not saved
            logger.debug(f"Full activations not requested for layer {layer_idx}")

        # Save layer metadata
        layer_meta = {
            "layer_idx": layer_idx,
            "behaviour": behaviour,
            "split": split,
            "n_active_features": layer_data["n_active_features"],
            "mean_active_per_sample": layer_data["mean_active_per_sample"],
            "d_transcoder": layer_data["d_transcoder"],
            "active_features": layer_data["active_feature_indices"].tolist(),  # Convert numpy to list for JSON
        }
        with open(layer_dir / f"{behaviour}_{split}_layer_meta.json", "w") as f:
            json.dump(layer_meta, f, indent=2)

        logger.info(f"Saved layer {layer_idx} features to {layer_dir}")

    # Save overall metadata
    summary = {
        "behaviour": behaviour,
        "split": split,
        "layers_processed": list(results.keys()),
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    with open(output_path / f"{behaviour}_{split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract transcoder features")
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
        help="Which behaviour to process (currently only grammar_agreement)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to process",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Model size (0.6b, 1.7b, 4b, 8b, 14b). Overrides config.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to process. Overrides config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for activation capture",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top features to track per sample",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    tc_config = load_transcoder_config(args.transcoder_config)

    # Determinism
    torch.manual_seed(config["seeds"]["torch_seed"])
    torch.backends.cudnn.deterministic = True

    # Model size
    model_size = args.model_size or tc_config.get("model_size", "4b")

    # Layers to process
    if args.layers:
        layer_indices = args.layers
    else:
        layer_indices = tc_config.get("analysis_layers", {}).get(
            "default", list(range(10, 26))
        )

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    print("=" * 70)
    print("TRANSCODER FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model size: {model_size}")
    print(f"  Layers: {layer_indices}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"  Split: {args.split}")
    print(f"  Top-k features: {args.top_k}")
    print(f"  Batch size: {args.batch_size}")

    # Load transcoder set
    print(f"\nLoading pre-trained transcoders for Qwen3-{model_size}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transcoder_set = load_transcoder_set(
        model_size=model_size,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=layer_indices,
    )
    print(f"Loaded transcoders for {len(layer_indices)} layers")

    # Load the language model for activation capture
    print(f"\nLoading language model...")
    
    # CRITICAL: Use EXACT model that transcoders were trained on!
    # Transcoders are trained on specific model (usually base, not instruct)
    # Using different model causes distribution shift and breaks feature alignment
    model_name = tc_config["transcoders"][model_size]["model_name"]
    
    # DO NOT OVERRIDE MODEL NAME!
    # If transcoders trained on Qwen3-4B (base), MUST use base.
    # For instruct-format prompts, apply chat template to prompts themselves,
    # but model must match transcoder training data to avoid distribution shift.
    logger.info(f"Using model: {model_name} (matches transcoder training)")


    model = ModelWrapper(
        model_name=model_name,
        dtype="bfloat16",
        device="auto",
        trust_remote_code=True,
    )

    # Process each behaviour
    for behaviour in behaviours:
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR: {behaviour}")
        print("=" * 70)

        # Load prompts
        prompt_path = Path(config["paths"]["prompts"])
        prompts = load_prompts(prompt_path, behaviour, args.split)
        print(f"\nLoaded {len(prompts)} prompts")

        # Capture MLP inputs
        print(f"\nCapturing MLP inputs for layers {layer_indices}...")
        mlp_inputs, position_map = capture_mlp_inputs(
            model,
            prompts,
            layer_indices,
            batch_size=args.batch_size,
            token_positions="decision",  # Next-token prediction position
        )
        
        # Save position_map (CRITICAL for attribution analysis)
        # Maps sample_idx -> (prompt_idx, token_pos, token_id)
        output_path = Path(config["paths"]["results"]) / "transcoder_features"
        output_path.mkdir(parents=True, exist_ok=True)
        position_map_path = output_path / f"{behaviour}_{args.split}_position_map.json"
        with open(position_map_path, "w") as f:
            json.dump(position_map, f, indent=2)
        logger.info(f"Saved position_map: {position_map_path}")

        # Extract features through transcoders
        print(f"\nExtracting transcoder features...")
        feature_results = extract_features(
            transcoder_set,
            mlp_inputs,
            top_k=args.top_k,
            device=device,
        )

        # Save results
        output_path = Path(config["paths"]["results"]) / "transcoder_features"
        metadata = {
            "model_size": model_size,
            "model_name": model_name,
            "transcoder_repo": tc_config["transcoders"][model_size]["repo_id"],
            "layers": layer_indices,
            "n_prompts": len(prompts),
            "top_k": args.top_k,
        }

        print(f"\nSaving features to {output_path}...")
        save_features(
            feature_results,
            output_path,
            behaviour,
            args.split,
            metadata,
        )

        # Clear memory
        del mlp_inputs, feature_results
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {Path(config['paths']['results']) / 'transcoder_features'}")
    print("\nNext step: python scripts/06_build_attribution_graph.py")


if __name__ == "__main__":
    main()
