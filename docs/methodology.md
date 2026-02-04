# Methodology

## Overview

This document describes the technical methodology for reproducing circuit-level analysis
from Anthropic's "On the Biology of a Large Language Model" (2025) on Qwen3-4B-Instruct-2507.

## Pipeline

### Phase 1: Prompt Generation and Baseline

**Goal:** Validate that the model can perform each behaviour with sufficient clarity.

- Generate 80 train + 20 test prompts per behaviour (4 behaviours)
- Measure logit difference between correct and incorrect answers
- Success criterion: accuracy >= threshold with sufficient logit separation
- Fixed seeds ensure exact reproducibility

### Phase 2: Activation Capture

**Goal:** Extract MLP activations for downstream SAE training.

- Forward pass through model on all prompts
- Capture residual stream activations at layers 10-24
- Extract last 5 token positions per prompt
- Store as numpy arrays per layer (~500MB per behaviour)

### Phase 3: Feature Extraction with Pre-trained Transcoders

**Goal:** Decompose MLP computations into interpretable features using pre-trained transcoders.

**Change from Original Design:** Instead of training SAEs from scratch, we use pre-trained
per-layer transcoders (PLTs) from the circuit-tracer project. This provides:
- Reproducibility (same features across researchers)
- Computational efficiency (no training required)
- Larger feature dictionaries from extensive training

Available Transcoders:
- mwhanna/qwen3-0.6b-transcoders-lowl0 (28 layers)
- mwhanna/qwen3-1.7b-transcoders-lowl0 (28 layers)
- mwhanna/qwen3-4b-transcoders (36 layers)
- mwhanna/qwen3-8b-transcoders (36 layers)
- mwhanna/qwen3-14b-transcoders-lowl0 (40 layers)

Architecture (PLT):
- Encoder: MLP input -> feature space via JumpReLU activation
- Decoder: feature space -> MLP output approximation
- JumpReLU: x if x > threshold, else 0 (learnable threshold)
- Skip connection: optional linear bypass for unmodeled components

Loading:
- Weights stored in safetensors format on HuggingFace
- Lazy loading supported (load layers on-demand)
- ~1.68 GB per layer for 4B model

### Phase 4: Feature Interpretation

**Goal:** Understand what each SAE feature encodes.

- For each feature, find top-activating prompts
- Compute feature-token and feature-category associations
- Auto-label features based on dominant patterns
- Manual review of top features per behaviour

### Phase 5: Attribution Graph Construction

**Goal:** Build causal dependency graphs from inputs to outputs.

Method (simplified from Anthropic's approach):
1. For each prompt, compute gradient of target logit w.r.t. hidden states
2. Project gradients into SAE feature space
3. Attribution = feature_activation * projected_gradient
4. Build graph with edges above attribution threshold
5. Prune to top-k edges per node

### Phase 6: Intervention Experiments

**Goal:** Validate circuit hypotheses through causal perturbations.

Three intervention types:
1. **Ablation**: Zero out feature activations, measure effect on logit difference
2. **Patching**: Swap features between prompt pairs (e.g., singular<->plural)
3. **Feature importance**: Correlate feature activations with logit differences

### Phase 7: Visualization

**Goal:** Generate publication-quality figures for thesis.

- Attribution graph diagrams (NetworkX)
- Feature activation heatmaps
- Intervention effect plots
- Comparison figures with Anthropic's results

## Technical Notes

### Model: Qwen3-4B-Instruct-2507

- Parameters: ~4B
- Hidden size: to be confirmed at runtime
- Layers: to be confirmed at runtime
- Context length: 262,144 tokens
- Non-thinking mode (no `<think>` blocks)

### Computational Requirements

- Single A100 GPU (40GB or 80GB)
- Model in BF16: ~8GB VRAM
- Peak during activation capture: ~15GB
- SAE training: ~6GB per layer
- Total pipeline: ~220 GPU-hours

### Reproducibility

All random operations use fixed seeds:
- Prompt generation: 42
- SAE training: 123
- Intervention sampling: 456
- PyTorch global: 789

## References

1. Lindsey et al. (2025). On the Biology of a Large Language Model.
2. Anthropic (2025). Attribution Graphs: Methods.
3. Bricken et al. (2023). Towards Monosemanticity.
4. Cunningham et al. (2023). Sparse Autoencoders Find Highly Interpretable Features.
