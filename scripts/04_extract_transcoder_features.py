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
                    acts_t = torch.from_numpy(acts)
                    assert acts_t.ndim == 2, f"{layer_key}: expected 2D (n_samples, d_model), got {acts_t.shape}"
                    layer_activations[layer_idx].append(acts_t)
                else:
                    logger.warning(f"Layer {layer_idx} not found in batch result. Available keys: {list(batch_result['activations'].keys())}")
        
        # Accumulate position_map (adjust prompt_idx for batch offset)
        # Only need to do this once per batch (same for all groups)
        if layer_groups and batch_result_for_posmap is not None:
            # Use result from FIRST group (position_map is same for all)
            batch_position_map = batch_result_for_posmap["metadata"]["position_map"]
            
            # Batch-level sanity check: ensure activations rows match position_map entries
            # This catches any synchronization issues early
            ref_layer = layer_groups[0][0]
            ref_key = f"layer_{ref_layer}_mlp_input"
            if ref_key in batch_result_for_posmap["activations"]:
                n_act = batch_result_for_posmap["activations"][ref_key].shape[0]
                n_pos = len(batch_position_map)
                assert n_act == n_pos, (
                    f"Batch {batch_idx} mismatch: activations rows={n_act} but position_map entries={n_pos} "
                    f"(token_positions={token_positions})"
                )
            
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

    # Add prompt hashes (AFTER batch assembly, using global indices)
    import hashlib
    prompt_hashes = {
        i: hashlib.sha256(prompt_texts[i].encode()).hexdigest()[:16]
        for i in range(len(prompt_texts))
    }
    for e in position_map_all:
        # e["prompt_idx"] is now global index
        e["prompt_hash"] = prompt_hashes[e["prompt_idx"]]

    # Post-process: set is_decision_position flag correctly
    # Decision token = token with max token_pos within each prompt
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for i, e in enumerate(position_map_all):
        by_prompt[e["prompt_idx"]].append((i, e["token_pos"]))
    
    for prompt_idx, items in by_prompt.items():
        max_pos = max(tp for _, tp in items)
        for i, tp in items:
            position_map_all[i]["is_decision_position"] = (tp == max_pos)
    
    # Add within-window ordering metadata (useful for analysis)
    # This tracks: position within the K-token window, and total window size
    for prompt_idx, items in by_prompt.items():
        # Sort by token_pos to get within-window ordering
        items_sorted = sorted(items, key=lambda x: x[1])
        window_len = len(items_sorted)
        for j, (global_i, _) in enumerate(items_sorted):
            position_map_all[global_i]["within_window_index"] = j
            position_map_all[global_i]["prompt_window_len"] = window_len
    
    # Robust sanity checks: verify sample structure aligns with token_positions mode
    n_samples = len(position_map_all)
    
    # Per-prompt statistics
    cnt = defaultdict(int)
    dec_cnt = defaultdict(int)
    max_pos_seen = defaultdict(lambda: -10**9)
    
    for e in position_map_all:
        p = e["prompt_idx"]
        cnt[p] += 1
        dec_cnt[p] += int(bool(e.get("is_decision_position", False)))
        max_pos_seen[p] = max(max_pos_seen[p], e["token_pos"])
    
    # Verify decision token corresponds to max token_pos in each prompt
    for idx, e in enumerate(position_map_all):
        p = e["prompt_idx"]
        if e.get("is_decision_position", False):
            assert e["token_pos"] == max_pos_seen[p], (
                f"Decision token_pos mismatch for prompt {p}: "
                f"decision token_pos={e['token_pos']} but max token_pos={max_pos_seen[p]}"
            )
    
    # Ensure every prompt appears (for decision/last modes)
    if token_positions != "all":
        assert len(cnt) == len(prompts), (
            f"Position map covers {len(cnt)} prompts, but prompts list has {len(prompts)}."
        )
    
    # Check sample counts per prompt (robust to variable-length prompts)
    if token_positions.startswith("last_"):
        K = int(token_positions.split("_")[1])
        # Each prompt should have between 1 and K samples (depends on prompt length)
        bad = [p for p, c in cnt.items() if not (1 <= c <= K)]
        assert not bad, (
            f"Some prompts have sample counts outside [1, {K}]: {bad[:10]} "
            f"(this indicates a bug in capture_mlp_inputs)"
        )
        logger.info(f"Sample count range per prompt: [{min(cnt.values())}, {max(cnt.values())}] (expected: [1, {K}])")
    
    elif token_positions in ("decision", "last"):
        # Each prompt should have exactly 1 sample
        bad = [p for p, c in cnt.items() if c != 1]
        assert not bad, (
            f"Some prompts do not have exactly 1 sample: {bad[:10]}"
        )
    
    # Decision flag: exactly one decision token per prompt (for non-all modes)
    if token_positions != "all":
        bad = [p for p, c in dec_cnt.items() if c != 1]
        assert not bad, (
            f"Some prompts do not have exactly one decision token: {bad[:10]}"
        )
    
    # Verify activations match position_map
    for layer_idx in layer_indices:
        acts = layer_activations[layer_idx]
        if acts is None:
            continue
        assert acts.shape[0] == n_samples, (
            f"Layer {layer_idx}: activations N={acts.shape[0]} != position_map N={n_samples}"
        )
    
    # Log window length statistics (useful for multi-token analysis)
    prompt_window_lens = {p: len(items) for p, items in by_prompt.items()}
    logger.info(f"Captured {n_samples} samples with token_positions='{token_positions}'")
    logger.info(f"Sanity checks passed: {len(cnt)} prompts, {n_samples} total samples")
    if len(prompt_window_lens) > 0:
        logger.info(
            f"Window lengths: min={min(prompt_window_lens.values())}, "
            f"max={max(prompt_window_lens.values())}, "
            f"mean={np.mean(list(prompt_window_lens.values())):.2f}"
        )
    
    return layer_activations, position_map_all



def extract_features(
    transcoder_set: TranscoderSet,
    mlp_inputs: Dict[int, torch.Tensor],
    top_k: int = 50,
    activation_threshold: float = 0.0,
    save_full_acts: bool = False,
    device: torch.device = None,
) -> Dict[int, Dict]:
    """
    Extract transcoder features from MLP inputs.

    Args:
        transcoder_set: Loaded TranscoderSet
        mlp_inputs: Dict mapping layer_idx -> activation tensor
        top_k: Number of top features to track per sample
        activation_threshold: Minimum activation value to consider feature "active"
        save_full_acts: Whether to save full feature activations (large files)
        device: Device for computation

    Returns:
        Dict mapping layer_idx -> {
            "feature_activations": (optional) full tensor of all activations,
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
        # Cast threshold to features dtype to avoid BF16 comparison issues
        thr = torch.as_tensor(activation_threshold, device=features.device, dtype=features.dtype)
        active_mask = (features > thr)  # [N, d_transcoder] bool on GPU
        active_feature_indices = torch.where(active_mask.any(dim=0))[0].detach().cpu().numpy().astype(np.int32)
        feature_frequencies = active_mask.float().mean(dim=0).detach().cpu().numpy()
        mean_active_per_sample = active_mask.sum(dim=1).float().mean().item()

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
            "mean_active_per_sample": mean_active_per_sample,  # Reuse computed value
            "d_transcoder": transcoder.d_transcoder,
        }
        
        # Free GPU memory early (active_mask no longer needed)
        del active_mask

        # Optionally store full activations (float16 on CPU) for downstream analysis
        if save_full_acts:
            feats_cpu = features.detach().to("cpu")
            if feats_cpu.dtype == torch.bfloat16:
                feats_cpu = feats_cpu.float()
            results[layer_idx]["feature_activations"] = feats_cpu.half()

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
        # NOTE: feature_activations already saved as float16 in extract_features
        if "feature_activations" in layer_data:
            full_acts_tensor = layer_data["feature_activations"]
            
            # Early check: verify size before conversion
            nbytes = full_acts_tensor.numel() * 2  # float16 = 2 bytes
            if nbytes < 1e9:  # < 1GB
                # Already float16 from extract_features - no need to cast
                full_acts = full_acts_tensor.numpy()
                np.save(
                    layer_dir / f"{behaviour}_{split}_full_activations.npy",
                    full_acts
                )
                logger.info(f"Saved full activations for layer {layer_idx} ({nbytes / 1e6:.1f} MB)")
            else:
                logger.warning(f"Skipping full activations for layer {layer_idx} (too large: {nbytes / 1e9:.2f} GB)")
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
            "token_positions": metadata.get("token_positions"),
            "context_tokens": metadata.get("context_tokens"),
            "activation_threshold": metadata.get("activation_threshold", 0.0),
            "saved_full_activations": bool("feature_activations" in layer_data),
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
    parser.add_argument(
        "--context_tokens",
        type=int,
        default=1,
        help="How many final prompt tokens to capture (1 = decision token only, 2+ = multi-token window)",
    )
    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=0.0,
        help="Minimum activation value to consider feature 'active' (default: 0.0)",
    )
    parser.add_argument(
        "--save_full_acts",
        action="store_true",
        help="Save full feature activations (large files, for PCA/distribution analysis)",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    tc_config = load_transcoder_config(args.transcoder_config)
    
    # Research hygiene: compute config hashes for reproducibility
    import hashlib
    import json as _json
    cfg_hash = hashlib.sha256(_json.dumps(config, sort_keys=True).encode()).hexdigest()
    tc_hash = hashlib.sha256(_json.dumps(tc_config, sort_keys=True).encode()).hexdigest()

    # Determinism
    torch.manual_seed(config["seeds"]["torch_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures full reproducibility

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
    
    # Ensure model is in eval mode for consistent activations across layer groups
    # Critical: same batch must give identical position_map when captured multiple times
    model.model.eval()

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
        
        # Determine token_positions mode based on context_tokens
        if args.context_tokens <= 1:
            token_positions = "decision"  # Single decision token
        else:
            token_positions = f"last_{args.context_tokens}"  # Multi-token window
        
        print(f"Token positions mode: {token_positions} (capturing {args.context_tokens} token(s) per prompt)")
        
        mlp_inputs, position_map = capture_mlp_inputs(
            model,
            prompts,
            layer_indices,
            batch_size=args.batch_size,
            token_positions=token_positions,
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
            activation_threshold=args.activation_threshold,
            save_full_acts=args.save_full_acts,
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
            "context_tokens": args.context_tokens,
            "token_positions": token_positions,
            "activation_threshold": args.activation_threshold,
            "top_k": args.top_k,
            # Feature computation metadata
            "frequencies_over": "all_features",
            "saved_activations": "top_k_only" if not args.save_full_acts else "full_float16",
            # Semantics note (honest, no assumptions)
            "token_pos_semantics": "token_pos as returned by ModelWrapper.capture_mlp_inputs metadata.position_map",
            # Config versioning for reproducibility
            "config_path": args.config,
            "transcoder_config_path": args.transcoder_config,
            "experiment_config_sha256": cfg_hash,
            "transcoder_config_sha256": tc_hash,
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