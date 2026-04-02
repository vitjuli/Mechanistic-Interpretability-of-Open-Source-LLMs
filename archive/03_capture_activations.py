"""
Capture residual stream activations from Qwen3-4B-Instruct for transcoder-based analysis.

This script captures activations at the residual stream (pre-MLP layer input)
for use with pre-trained transcoders. Saves as float16 numpy arrays.

Usage:
    python scripts/03_capture_activations.py
    python scripts/03_capture_activations.py --split test --batch_size 16
"""

import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import sys
from tqdm import tqdm
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
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


def capture_activations_batch(
    model: ModelWrapper,
    prompts: List[Dict],
    layer_range: tuple,
    batch_size: int = 8,
    token_positions: str = "all",
) -> Dict:
    """
    Capture activations with efficient batching.

    Args:
        model: ModelWrapper instance
        prompts: List of prompt dictionaries
        layer_range: (start, end) layer indices
        batch_size: Number of prompts per batch
        token_positions: Token selection strategy ("all" for SAE, "answer" for attribution)

    Returns:
        Dictionary with:
            - activations: Dict[layer_name, np.ndarray]
            - metadata: Comprehensive metadata dict
    """
    # Extract prompt texts
    prompt_texts = [p["prompt"] for p in prompts]
    
    # Process in batches to manage memory
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"Processing {len(prompts)} prompts in {n_batches} batches...")
    
    # Storage for accumulated activations
    activations_all = {}
    all_metadata = []
    
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompt_texts = prompt_texts[start_idx:end_idx]
        
        # Capture activations for this batch
        batch_result = model.capture_activations(
            batch_prompt_texts,
            layer_range=layer_range,
            token_positions=token_positions,
            include_logits=False,  # Not needed for SAE training
        )
        
        batch_acts = batch_result["activations"]
        batch_meta = batch_result["metadata"]
        
        # Accumulate activations
        for layer_name, acts in batch_acts.items():
            if layer_name not in activations_all:
                activations_all[layer_name] = []
            activations_all[layer_name].append(acts)
        
        # Accumulate metadata
        all_metadata.append(batch_meta)
    
    # Concatenate all batches
    print("\nConcatenating batches...")
    activations_final = {}
    for layer_name, acts_list in activations_all.items():
        activations_final[layer_name] = np.concatenate(acts_list, axis=0)
    
    # Merge metadata
    merged_metadata = {
        "n_prompts": len(prompts),
        "n_samples": sum(m["n_samples"] for m in all_metadata),
        "token_selection": token_positions,
        "layer_range": list(layer_range),
        "shapes": {k: list(v.shape) for k, v in activations_final.items()},
    }
    
    return {
        "activations": activations_final,
        "metadata": merged_metadata,
    }


def save_activations(
    result: Dict,
    output_path: Path,
    behaviour: str,
    split: str,
    extra_metadata: Dict = None,
):
    """
    Save activations to disk with metadata.

    Args:
        result: Dictionary with "activations" and "metadata" keys
        output_path: Output directory
        behaviour: Behaviour name
        split: Data split name
        extra_metadata: Additional metadata to save
    """
    import re
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    activations = result["activations"]
    metadata = result["metadata"]

    # Save each layer separately (for lazy loading during SAE training)
    for layer_name, acts in activations.items():
        # Robust layer index extraction
        match = re.search(r'layer_(\d+)', layer_name)
        if not match:
            raise ValueError(f"Cannot parse layer index from '{layer_name}'")
        layer_idx = match.group(1)

        output_file = output_path / f"{behaviour}_{split}_layer_{layer_idx}.npy"

        # Save as float16 (already converted in capture_activations)
        np.save(output_file, acts)

        print(f"  Saved {layer_name}: shape {acts.shape}, "
              f"dtype {acts.dtype}, size {acts.nbytes / 1e6:.1f} MB → {output_file.name}")

    # Save metadata
    metadata_file = output_path / f"{behaviour}_{split}_metadata.json"
    full_metadata = {
        "behaviour": behaviour,
        "split": split,
        **metadata,  # n_prompts, n_samples, token_selection, shapes, layer_range
        **(extra_metadata or {}),
    }

    with open(metadata_file, "w") as f:
        # Pretty print for human readability, exclude large arrays
        metadata_to_save = {k: v for k, v in full_metadata.items() 
                           if k not in ["input_ids", "attention_mask", "position_map"]}
        json.dump(metadata_to_save, f, indent=2)

    print(f"  Saved metadata → {metadata_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Capture model activations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
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
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # CRITICAL: Determinism controls
    torch.manual_seed(config["seeds"]["torch_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Behaviours (single behaviour for pipeline testing)
    behaviours = [args.behaviour]

    # Token selection default
    token_positions = config["activations"].get("token_positions", "all")
    
    # Batch size
    batch_size = args.batch_size or config["activations"]["batch_size"]

    # Layer range
    layer_range = tuple(config["activations"]["layer_range"])

    print("=" * 70)
    print("ACTIVATION CAPTURE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Layers: {layer_range[0]}-{layer_range[1]}")
    print(f"  Token positions: {token_positions}")
    print(f"  Batch size: {batch_size}")
    print(f"  Split: {args.split}")
    print(f"  Behaviours: {', '.join(behaviours)}")
    print(f"\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Determinism: cudnn.deterministic={torch.backends.cudnn.deterministic}")

    # Load model
    print(f"\nLoading model...")
    model = ModelWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
        trust_remote_code=config["model"]["trust_remote_code"],
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

        # Capture activations
        print(f"\nCapturing activations from layers {layer_range[0]}-{layer_range[1]}...")
        result = capture_activations_batch(
            model,
            prompts,
            layer_range,
            batch_size=batch_size,
            token_positions=token_positions,
        )
        
        activations = result["activations"]
        metadata = result["metadata"]
        
        # Calculate total size
        total_size = sum(acts.nbytes for acts in activations.values())
        print(f"\nTotal activation size: {total_size / 1e9:.2f} GB")
        print(f"Total samples: {metadata['n_samples']} ({metadata['n_prompts']} prompts)")

        # Save
        output_path = Path(config["paths"]["activations"])
        extra_metadata = {
            "model": config["model"]["name"],
            "dtype": config["model"]["dtype"],
            "device": str(model.device),
            "torch_version": torch.__version__,
            "deterministic": torch.backends.cudnn.deterministic,
        }
        
        print(f"\nSaving activations to {output_path}...")
        save_activations(
            result,
            output_path,
            behaviour,
            args.split,
            extra_metadata,
        )

    print("\n" + "=" * 70)
    print("ACTIVATION CAPTURE COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {Path(config['paths']['activations']).absolute()}")
    print("\nNext step: python scripts/04_extract_transcoder_features.py")


if __name__ == "__main__":
    main()
