# Archive: Legacy SAE Training Code

This folder contains the original sparse autoencoder (SAE) training implementation that was part of the initial project design.

## Why Archived?

The project methodology was updated based on supervisor guidance to use **pre-trained transcoders** from the [circuit-tracer repository](https://github.com/safety-research/circuit-tracer) instead of training SAEs from scratch.

### Original Approach (Archived)
- Train lightweight SAEs per layer on collected MLP activations
- Map discovered features to tokens or behaviours
- Required significant compute for training

### New Approach (Active)
- Use pre-trained transcoders from HuggingFace:
  - `mwhanna/qwen3-0.6b-transcoders-lowl0`
  - `mwhanna/qwen3-1.7b-transcoders-lowl0`
  - `mwhanna/qwen3-4b-transcoders`
  - `mwhanna/qwen3-8b-transcoders`
  - `mwhanna/qwen3-14b-transcoders-lowl0`
- Implement attribution graphs and interventions using the transcoder features
- Focus on interpretability analysis rather than feature learning

## Archived Contents

### `legacy_sae/`
- `sae.py` — SparseAutoencoder and SAETrainer classes

### `legacy_scripts/`
- `04_train_sae.py` — SAE training pipeline script
- `05_interpret_features.py` — Feature interpretation for trained SAEs
- `06_build_attribution_graph_sae.py` — Attribution graphs using trained SAEs
- `07_run_interventions_sae.py` — Interventions using trained SAEs

## Notes

These files remain functional and could be used if:
1. Custom SAE training is needed for specific layers/behaviours
2. Comparison between trained SAEs and pre-trained transcoders is desired
3. The transcoder approach proves insufficient for certain analyses

The active pipeline now uses `src/transcoder/` for all feature decomposition.

---

*Archived: 2025-02*
