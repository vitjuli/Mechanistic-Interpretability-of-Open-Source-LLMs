"""
Train Sparse Autoencoders (SAEs) on captured activations.

Supports checkpointing and basic progress logging.
Designed for HPC environments with reproducible splits and seed management.

Usage:
    # Train single layer
    python scripts/04_train_sae.py --behaviour grammar_agreement --layer 15

    # Resume from checkpoint (preserves train/val split)
    python scripts/04_train_sae.py --layer 15 --resume_from models/saes/checkpoint.pt

    # Train with custom hyperparameters
    python scripts/04_train_sae.py --layer 15 --expansion_factor 8 --l1_lambda 0.01
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
import sys
from tqdm import tqdm
import time
import os
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sae import SparseAutoencoder, SAETrainer


def get_git_hash() -> str:
    """Get current git commit hash for reproducibility tracking."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent
        ).decode("ascii").strip()
    except:
        return "unknown"


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Deterministic operations (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_states():
    """Get current RNG states for checkpointing."""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
    }


def set_rng_states(states: Dict):
    """Restore RNG states from checkpoint."""
    torch.set_rng_state(states["torch_rng_state"])
    if states["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(states["cuda_rng_state"])
    np.random.set_state(states["numpy_rng_state"])


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_activations(
    activation_path: Path,
    behaviour: str,
    split: str,
    layer: int,
) -> torch.Tensor:
    """
    Load activations for a specific layer.

    Returns:
        Tensor of shape (n_prompts, n_tokens, hidden_dim)
    """
    act_file = activation_path / f"{behaviour}_{split}_layer_{layer}.npy"

    if not act_file.exists():
        raise FileNotFoundError(
            f"Activation file not found: {act_file}\n"
            f"Run 'python scripts/03_capture_activations.py' first."
        )

    print(f"Loading activations from {act_file.name}...")
    acts = np.load(act_file)

    # Convert to tensor
    acts_tensor = torch.from_numpy(acts).float()

    # Reshape: (n_prompts, n_tokens, hidden_dim) → (n_prompts * n_tokens, hidden_dim)
    n_prompts, n_tokens, hidden_dim = acts_tensor.shape
    acts_flat = acts_tensor.reshape(-1, hidden_dim)

    print(f"  Shape: {acts_tensor.shape} → {acts_flat.shape}")
    print(f"  Total samples: {acts_flat.shape[0]:,}")
    print(f"  Hidden dim: {acts_flat.shape[1]}")

    return acts_flat


def evaluate_batched(
    trainer: SAETrainer,
    val_data: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> Dict:
    """
    Evaluate on validation set in batches to avoid OOM.
    
    Note: dead_fraction is averaged across batches, which is an approximation.
    For exact dead_fraction, would need to accumulate feature activations
    across all batches. Averaging is sufficient for monitoring purposes.
    
    Args:
        trainer: SAE trainer instance
        val_data: Validation data (n_samples, hidden_dim) on CPU
        device: Device to evaluate on
        batch_size: Batch size for evaluation
    
    Returns:
        Aggregated metrics dictionary
    """
    n_samples = val_data.shape[0]
    
    # If validation set is small, move to GPU once
    if n_samples * val_data.shape[1] * 4 < 1e9:  # <1GB
        val_data_gpu = val_data.to(device)
        return trainer.evaluate(val_data_gpu)
    
    # Otherwise, evaluate in batches
    metrics_list = []
    for i in range(0, n_samples, batch_size):
        batch = val_data[i:i+batch_size].to(device)
        batch_metrics = trainer.evaluate(batch)
        metrics_list.append(batch_metrics)
    
    # Aggregate metrics (mean for most, last for dead_fraction)
    return {
        "r2": float(np.mean([m["r2"] for m in metrics_list])),
        "l0": float(np.mean([m["l0"] for m in metrics_list])),
        "dead_fraction": float(np.mean([m["dead_fraction"] for m in metrics_list])),  # Average across batches
    }


def train_sae(
    activations: torch.Tensor,
    config: Dict,
    layer: int,
    device: torch.device,
    git_hash: str,  # Pre-computed git hash for metadata
    resume_from: Optional[str] = None,
    checkpoint_every: int = 5000,
) -> SparseAutoencoder:
    """
    Train SAE on activations with checkpointing.

    Args:
        activations: Training data (n_samples, hidden_dim)
        config: Configuration dictionary
        layer: Layer index
        device: Training device
        git_hash: Git commit hash for reproducibility tracking
        resume_from: Path to checkpoint to resume from
        checkpoint_every: Save checkpoint every N steps

    Returns:
        Trained SAE model
    
    Note: Validation R² and dead_fraction are computed via evaluate_batched(),
    which uses batch-wise approximations for large validation sets.
    """
    # Extract hyperparameters
    sae_config = config["sae"]
    expansion_factor = sae_config["expansion_factor"]
    l1_lambda = sae_config["l1_lambda"]
    lr = sae_config["learning_rate"]
    batch_size = sae_config["batch_size"]
    max_steps = sae_config["max_steps"]
    val_split = sae_config["validation_split"]

    # Split train/validation
    n_samples = activations.shape[0]
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    start_step = 0
    best_val_r2 = -float("inf")
    model_state_dict = None
    optimizer_state_dict = None

    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")

        # Load checkpoint first
        checkpoint = torch.load(resume_from, map_location="cpu")

        # Restore RNG states ASAP (before any new random ops)
        if "rng_states" in checkpoint:
            set_rng_states(checkpoint["rng_states"])
            print("  ✓ Restored RNG states")

        # Restore step + best tracking + states
        start_step = int(checkpoint.get("step", 0))
        best_val_r2 = float(checkpoint.get("best_val_r2", -float("inf")))
        model_state_dict = checkpoint.get("model_state")
        optimizer_state_dict = checkpoint.get("optimizer_state")  # may be None (e.g. best.pt)

        # Restore split if available; else fallback
        if "train_indices" in checkpoint and "val_indices" in checkpoint:
            train_indices = checkpoint["train_indices"].cpu().long()
            val_indices = checkpoint["val_indices"].cpu().long()
            train_data = activations[train_indices]
            val_data = activations[val_indices]
        else:
            print("  ⚠ Warning: Checkpoint missing split indices; creating new split")
            indices = torch.randperm(n_samples)
            train_indices = indices[:n_train].cpu()
            val_indices = indices[n_train:].cpu()
            train_data = activations[train_indices]
            val_data = activations[val_indices]

        # Recompute sizes from actual tensors
        n_train = train_data.shape[0]
        n_val = val_data.shape[0]

        print(f"  ✓ Resumed from step {start_step}")
        print(f"    Train: {n_train:,} samples, Val: {n_val:,} samples")
        print(f"    Best val R² so far: {best_val_r2:.4f}")
    else:
        # Fresh training: create random split
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train].cpu()  # Ensure CPU
        val_indices = indices[n_train:].cpu()    # Ensure CPU

        train_data = activations[train_indices]
        val_data = activations[val_indices]
        
        # Recalculate from actual sizes
        n_train = train_data.shape[0]
        n_val = val_data.shape[0]

    
    # Print dataset split info
    print(f"\nDataset split:")
    print(f"  Training: {n_train:,} samples")
    print(f"  Validation: {n_val:,} samples")
    
    # Initialize SAE (after split is determined)
    hidden_dim = activations.shape[1]
    sae = SparseAutoencoder(
        input_dim=hidden_dim,
        expansion_factor=expansion_factor,
        l1_lambda=l1_lambda,
    ).to(device)

    # Initialize trainer
    trainer = SAETrainer(sae, learning_rate=lr)
    
    # Load model/optimizer states if resuming
    if model_state_dict is not None:
        sae.load_state_dict(model_state_dict)
        if optimizer_state_dict is not None:
            trainer.optimizer.load_state_dict(optimizer_state_dict)
        print(f"  ✓ Loaded model and optimizer states")

    # Training loop
    print(f"\nTraining SAE:")
    print(f"  Expansion factor: {expansion_factor}×")
    print(f"  Latent dim: {sae.latent_dim}")
    print(f"  L1 lambda: {l1_lambda}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max steps: {max_steps}")

    checkpoint_dir = Path(config["paths"]["saes"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_start = time.time()
    
    # Initialize step before loop to prevent UnboundLocalError if loop never runs
    step = start_step - 1

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps)

    for step in pbar:
        # Sample batch
        batch_indices = torch.randint(0, n_train, (batch_size,))
        batch = train_data[batch_indices].to(device)

        # Training step
        losses = trainer.train_step(batch)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{losses['total']:.4f}",
            "mse": f"{losses['mse']:.4f}",
            "l1": f"{losses['l1']:.4f}",
        })

        # Validation and checkpointing
        if (step + 1) % checkpoint_every == 0:
            # Evaluate on validation set (batched to avoid OOM)
            metrics = evaluate_batched(trainer, val_data, device)

            elapsed = time.time() - training_start
            samples_per_sec = (step + 1 - start_step) * batch_size / elapsed

            print(f"\n[Step {step + 1}/{max_steps}]")
            print(f"  Train loss: {losses['total']:.4f}")
            print(f"  Val R²: {metrics['r2']:.4f}")
            print(f"  Val L0: {metrics['l0']:.4f}")
            print(f"  Dead features: {metrics['dead_fraction']:.1%}")
            print(f"  Speed: {samples_per_sec:.0f} samples/sec")

            # CRITICAL: Check if new best (epsilon for float safety)
            is_new_best = metrics["r2"] > best_val_r2 + 1e-12
            
            # Update best_val_r2 BEFORE saving checkpoint
            if is_new_best:
                best_val_r2 = metrics["r2"]
                print(f"  ✓ New best model (R² = {best_val_r2:.4f})")

            # Save checkpoint (now contains correct best_val_r2)
            checkpoint_path = checkpoint_dir / f"layer_{layer}_checkpoint_{step+1}.pt"
            checkpoint_data = {
                "step": step + 1,
                "model_state": sae.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "train_indices": train_indices.cpu(),  # Ensure CPU
                "val_indices": val_indices.cpu(),      # Ensure CPU
                "rng_states": get_rng_states(),
                "best_val_r2": best_val_r2,  # Updated value
                "config": sae_config,
                "full_config": config,  # Full config for reproducibility
                "metrics": metrics,
                # Reproducibility metadata
                "git_hash": git_hash,  # Pre-computed in main()
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # ISO format
            }
            
            # Atomic save: write to temp file, then rename
            temp_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, checkpoint_path)  # Atomic on POSIX

            # Save separate best model file if this IS best (with atomic save)
            if is_new_best:
                best_path = checkpoint_dir / f"layer_{layer}_best.pt"
                best_data = {
                    "step": step + 1,
                    "model_state": sae.state_dict(),
                    "config": sae_config,
                    "metrics": metrics,
                }
                best_temp = best_path.with_suffix(".pt.tmp")
                torch.save(best_data, best_temp)
                os.replace(best_temp, best_path)

            # Check stopping criteria
            if metrics["r2"] >= sae_config["reconstruction_target"]:
                print(f"\n✓ Reached target R² ({sae_config['reconstruction_target']})")
                break

            if metrics["dead_fraction"] > sae_config["dead_neuron_threshold"]:
                print(f"\n⚠ Warning: {metrics['dead_fraction']:.1%} dead features (threshold: {sae_config['dead_neuron_threshold']:.1%})")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_batched(trainer, val_data, device)

    print("\nTraining complete:")
    print(f"  Final R²: {final_metrics['r2']:.4f}")
    print(f"  Final L0: {final_metrics['l0']:.4f}")
    print(f"  Dead features: {final_metrics['dead_fraction']:.1%}")

    # Save final model (atomic)
    final_path = checkpoint_dir / f"layer_{layer}_final.pt"
    final_data = {
        "step": step + 1,
        "model_state": sae.state_dict(),
        "config": sae_config,
        "metrics": final_metrics,
    }
    final_temp = final_path.with_suffix(".pt.tmp")
    torch.save(final_data, final_temp)
    os.replace(final_temp, final_path)
    print(f"\nSaved final model to: {final_path}")

    return sae


def main():
    parser = argparse.ArgumentParser(description="Train sparse autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        default="grammar_agreement",
        help="Which behaviour to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Which layer to train SAE for",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=None,
        help="Latent expansion factor (overrides config)",
    )
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=None,
        help="L1 penalty (overrides config)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (overrides config)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    set_seed(config["seeds"]["sae_training"])
    
    # Compute git hash once (expensive on HPC compute nodes)
    git_hash = get_git_hash()

    # Override config with CLI args (use 'is not None' to allow zero values)
    if args.expansion_factor is not None:
        config["sae"]["expansion_factor"] = args.expansion_factor
    if args.l1_lambda is not None:
        config["sae"]["l1_lambda"] = args.l1_lambda
    if args.max_steps is not None:
        config["sae"]["max_steps"] = args.max_steps

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"SPARSE AUTOENCODER TRAINING - Layer {args.layer}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Behaviour: {args.behaviour}")
    print(f"  Layer: {args.layer}")
    print(f"  Device: {device}")
    print(f"  Expansion factor: {config['sae']['expansion_factor']}×")
    print(f"  L1 lambda: {config['sae']['l1_lambda']}")

    # Load activations
    activation_path = Path(config["paths"]["activations"])
    activations = load_activations(
        activation_path,
        args.behaviour,
        args.split,
        args.layer,
    )

    # Train SAE
    sae = train_sae(
        activations,
        config,
        args.layer,
        device,
        git_hash,  # Pass pre-computed git hash
        resume_from=args.resume_from,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n" + "=" * 70)
    print("SAE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {Path(config['paths']['saes']).absolute()}")
    print("\nNext step: Interpret features or build attribution graph")


if __name__ == "__main__":
    main()
