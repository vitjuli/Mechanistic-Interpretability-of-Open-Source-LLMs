# Legacy SAE Code (Archived)

This directory contains **legacy code** from the original SAE (Sparse Autoencoder) implementation, before the project migrated to using pre-trained transcoders.

## ⚠️ These files are NOT USED in the current pipeline

The current project uses **pre-trained transcoders** from the [circuit-tracer](https://github.com/safety-research/circuit-tracer) project instead of training custom SAEs.

---

## Archived Files

### From `src/`:

1. **`attribution.py`** (7,671 bytes)
   - Original gradient-based attribution implementation for SAEs
   - **Replaced by:** `scripts/06_build_attribution_graph.py` (uses transcoders)
   - **Reason for archival:** Imports and depends on `SparseAutoencoder` class

2. **`interventions.py`** (7,707 bytes)
   - Original intervention methods (ablation, patching, steering) for SAEs
   - **Replaced by:** `scripts/07_run_interventions.py` (uses transcoders)
   - **Reason for archival:** Imports and depends on `SparseAutoencoder` class

### From project root:

3. **`sae.py`** (8,061 bytes)
   - SparseAutoencoder class implementation
   - **Replaced by:** `src/transcoder/` module (loads pre-trained transcoders)

4. **`04_train_sae.py`** (17,702 bytes)
   - SAE training script
   - **Replaced by:** `scripts/04_extract_transcoder_features.py`

5. **`sae_config.yaml`** (1,902 bytes)
   - Configuration for SAE training
   - **Replaced by:** `configs/transcoder_config.yaml`

---

## Why Were These Archived?

### Original approach (SAE-based):
1. Generate prompts
2. Run baseline
3. Capture activations
4. **Train SAEs from scratch** ← Expensive, requires compute
5. Build attribution graphs with trained SAEs
6. Run interventions with trained SAEs

### Current approach (Transcoder-based):
1. Generate prompts
2. Run baseline
3. Capture activations
4. **Load pre-trained transcoders from HuggingFace** ← Fast, no training needed
5. Build attribution graphs with transcoders
6. Run interventions with transcoders

**Benefits of transcoders:**
- ✅ No training required (saves GPU hours)
- ✅ Pre-trained on large datasets
- ✅ Available for multiple model sizes (0.6B-14B)
- ✅ Professionally trained with high quality
- ✅ Allows multi-scale comparison

---

## Can This Code Be Used?

**No, not without modification.**

These files depend on:
- `src.sae.SparseAutoencoder` class (archived)
- SAE weight files (`.pt` format) which are not generated in current pipeline
- SAE-specific configuration

To use this code, you would need to:
1. Restore `sae.py` to `src/`
2. Run `archive/04_train_sae.py` to train SAEs
3. Update imports in `attribution.py` and `interventions.py`

However, **this is not recommended**. The current transcoder-based approach is superior.

---

## Files in This Directory

```
archive/legacy_sae/
├── README.md                 # This file
├── sae.py                    # SAE class implementation
├── sae_config.yaml           # SAE training configuration
├── attribution.py            # SAE-based attribution (was src/attribution.py)
├── interventions.py          # SAE-based interventions (was src/interventions.py)
└── 04_train_sae.py          # SAE training script (was scripts/04_train_sae.py)
```

---

## Migration Details

**Date:** February 2026

**Reason:** Switched to pre-trained transcoders to:
- Align with circuit-tracer project methodology
- Enable multi-scale analysis across Qwen3 model sizes
- Reduce computational requirements
- Improve reproducibility

**Key changes:**
- Replaced `scripts/04_train_sae.py` → `scripts/04_extract_transcoder_features.py`
- Replaced `src/sae.py` → `src/transcoder/` module
- Reimplemented attribution (script 06) and interventions (script 07) for transcoders
- Updated all configs and documentation

---

## Reference

For the current implementation, see:
- `src/transcoder/` - Transcoder module
- `configs/transcoder_config.yaml` - Transcoder configuration
- `scripts/04_extract_transcoder_features.py` - Feature extraction
- `scripts/06_build_attribution_graph.py` - Attribution graphs
- `scripts/07_run_interventions.py` - Interventions
- `PAPER_COMPARISON.md` - Comparison with original SAE approach
