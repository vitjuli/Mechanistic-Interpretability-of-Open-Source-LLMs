# Computational Specifications for Mechanistic Interpretability Study

## Executive Summary

This document specifies the computational requirements, optimizations, and expected performance for the mechanistic interpretability pipeline on Cambridge CD3.

---

## 1. Model Specifications

### 1.1 Qwen2.5-3B-Instruct

- **Parameters:** 3.09B total (using 2.5-3B as proxy for unavailable 4B-2507)
- **Architecture:** Transformer decoder
- **Layers:** 28 hidden layers
- **Hidden dimension:** 3584 (MLP: 3584 → 18944 → 3584)
- **Attention heads:** 28
- **Vocabulary size:** 152,064 tokens
- **Context length:** 32,768 tokens (we use ≤512)

### 1.2 Memory Footprint

| Precision | Model Weights | Peak Memory (inference) | Peak Memory (training SAE) |
|-----------|---------------|-------------------------|---------------------------|
| FP32 | 12GB | ~20GB | ~35GB |
| FP16 | 6GB | ~12GB | ~20GB |
| BF16 | 6GB | ~12GB | ~20GB |

**Recommendation:** Use BF16 on A100 GPUs for optimal speed/memory trade-off.

---

## 2. Phase-by-Phase Resource Requirements

### 2.1 Phase 0: Prompt Generation

- **Computation:** CPU-only
- **Time:** <5 minutes
- **Memory:** <1GB
- **Storage:** ~1MB (100 prompts × 2 behaviours × 2 splits)

### 2.2 Phase 1: Baseline Measurements

**Per behaviour (80 prompts):**
- **GPU:** 1× A100 (40GB)
- **Walltime:** 30-45 minutes
- **GPU utilization:** ~60% (inference-bound)
- **Memory:** ~15GB GPU, 8GB RAM
- **Storage:** ~100KB CSV output

**Total for 2 behaviours (train + test):** ~2 hours

### 2.3 Phase 2: Activation Capture

**Configuration:**
- Layers: 10-24 (15 layers)
- Tokens per prompt: 5 (last 5 tokens)
- Components: MLP post-activations
- Prompts: 80 train

**Memory calculation:**
```
Activation size per prompt:
  15 layers × 5 tokens × 3584 hidden_dim × 2 bytes (BF16)
  = 15 × 5 × 3584 × 2 = 537,600 bytes ≈ 0.5MB per prompt

Total for 80 prompts: 80 × 0.5MB = 40MB per behaviour
Total for 2 behaviours: ~80MB
```

**Actual requirements (with overhead):**
- **GPU:** 1× A100 (40GB)
- **Walltime:** 2-3 hours (includes model loading)
- **GPU utilization:** 70-80%
- **Memory:** ~20GB GPU, 32GB RAM (for batching)
- **Storage:** ~500MB (with metadata)

### 2.4 Phase 3: SAE Training

**Per layer configuration:**
- Input dim: 3584
- Expansion factor: 4× → 14,336 latent features
- Training samples: 1M-5M activation vectors
- Batch size: 256
- Steps: 50,000

**Memory:**
```
Model parameters:
  Encoder: 3584 × 14,336 × 4 bytes (FP32) = 206MB
  Decoder: 14,336 × 3584 × 4 bytes = 206MB
  Total: ~412MB per SAE

Gradients + optimizer states (Adam): 3× parameters = 1.2GB
Batch activations: 256 × 3584 × 4 bytes = 3.7MB
Peak memory: ~2GB per layer
```

**Resources per layer:**
- **GPU:** 1× A100 (40GB, heavily underutilized)
- **Walltime:** 8-12 hours
- **GPU utilization:** 40-60% (optimization-bound)
- **Memory:** ~10GB GPU, 16GB RAM
- **Storage:** ~500MB checkpoint per layer

**Array job (15 layers in parallel):**
- **Total GPUs:** 15
- **Walltime:** 12 hours (wall time, not GPU-hours)
- **Total GPU-hours:** 15 × 12 = 180 GPU-hours
- **Storage:** 15 × 500MB = 7.5GB

**Alternative (sequential):**
- **Total GPUs:** 1
- **Walltime:** 15 × 12 = 180 hours (~7.5 days)
- **Total GPU-hours:** 180 GPU-hours (same compute cost)

**Recommendation:** Use array job for fast turnaround.

### 2.5 Phase 4: Attribution Graph Construction

**Per behaviour:**
- Layers analyzed: 3-5 key layers (e.g., L15, L18, L21)
- Features per layer: ~14,336
- Gradient computations: ~100 forward/backward passes

**Resources:**
- **GPU:** 1× A100 (40GB)
- **Walltime:** 4-6 hours
- **GPU utilization:** 85-95% (gradient-heavy)
- **Memory:** ~30GB GPU, 32GB RAM
- **Storage:** ~100MB (graph structure + attributions)

### 2.6 Phase 5: Causal Interventions

**Types:**
- Ablation: 100 samples × 10 feature groups = 1,000 forward passes
- Patching: 50 prompt pairs × 20 positions = 1,000 forward passes
- Resampling: 100 samples × 10 feature groups = 1,000 forward passes

**Total:** ~3,000 forward passes

**Resources:**
- **GPU:** 1× A100 (40GB)
- **Walltime:** 6-8 hours
- **GPU utilization:** 75-85%
- **Memory:** ~25GB GPU, 32GB RAM
- **Storage:** ~500MB (results + statistics)

---

## 3. Total Resource Summary

### 3.1 Compute Requirements

| Metric | Sequential Pipeline | Parallel SAE (Recommended) |
|--------|---------------------|---------------------------|
| **Total walltime** | ~200 hours (~8 days) | ~30 hours (~1.3 days) |
| **Total GPU-hours** | ~200 GPU-hours | ~220 GPU-hours |
| **Peak GPUs** | 1 | 15 (SAE phase only) |
| **Total storage** | ~100GB | ~100GB |

### 3.2 Cost Estimation (CD3 Service Units)

Approximate costs (1 GPU-hour ≈ 1 service unit on CD3):

| Phase | GPU-hours | Service Units |
|-------|-----------|---------------|
| Baseline | 2 | 2 |
| Activations | 3 | 3 |
| SAE training | 180 | 180 |
| Attribution | 6 | 6 |
| Interventions | 8 | 8 |
| Overhead/retries | 20 | 20 |
| **Total** | **~220** | **~220** |

**Note:** Actual costs depend on CD3 pricing model. Check with `mybalance`.

### 3.3 Storage Breakdown

| Component | Size | Keep? |
|-----------|------|-------|
| Prompts | 1MB | ✓ (commit to git) |
| Baseline results | 1MB | ✓ |
| Activations | 20GB | ✗ (regenerate if needed) |
| SAE checkpoints | 75GB | ✓ (archive) |
| Attribution graphs | 100MB | ✓ |
| Intervention results | 500MB | ✓ |
| Figures | 100MB | ✓ |
| **Total (archived)** | **~1GB** | |

**Recommendation:** Archive activations and intermediate checkpoints after publication.

---

## 4. Optimization Strategies

### 4.1 Memory Optimization

**For inference (activations, interventions):**
```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(inputs)

# Batch efficiently
batch_size = 16  # Adjust based on available memory
```

**For SAE training:**
```python
# Use smaller batch sizes with gradient accumulation
batch_size = 128
accumulation_steps = 2  # Effective batch size: 256

# Use FP32 for SAE weights (stability), BF16 for activations
sae = SparseAutoencoder(...).float()
activations = activations.to(torch.bfloat16)
```

### 4.2 Speed Optimization

**Activation capture:**
- Cache model on node before batch processing
- Use `torch.compile()` (PyTorch 2.0+) for 20-30% speedup
- Process multiple behaviours in single job (shared model loading)

**SAE training:**
- Use larger batch sizes if memory allows (256 → 512)
- Enable TF32 on A100: `torch.backends.cuda.matmul.allow_tf32 = True`
- Use DataLoader with `num_workers=4` for CPU-side preprocessing

**Attribution graphs:**
- Compute gradients in mixed precision
- Parallelize across prompts (data parallelism)
- Prune graph online (discard low-attribution edges immediately)

### 4.3 I/O Optimization

**Problem:** Activations are large (20GB) and slow to load.

**Solutions:**
1. **Memory-mapped files:** Use `np.memmap()` or `torch.load(map_location='cpu', mmap=True)`
2. **Compressed storage:** Use `torch.save(..., _use_new_zipfile_serialization=True)`
3. **Lazy loading:** Load activations per layer on-demand during SAE training
4. **Parallel I/O:** Use Lustre striping on CD3:
   ```bash
   lfs setstripe -c 4 data/activations/  # Stripe across 4 OSTs
   ```

---

## 5. Parallelization Strategies

### 5.1 Data Parallelism (Within Phase)

**Activation capture:**
```python
# Distribute prompts across multiple GPUs
python scripts/03_capture_activations.py --distributed --world_size 4
```

Each GPU processes 80/4 = 20 prompts.

### 5.2 Job Parallelism (Across Layers)

**SAE training:**
```bash
# SLURM array job: 15 simultaneous jobs
sbatch --array=10-24 slurm_scripts/03_train_sae_array.slurm
```

Each job is independent (embarrassingly parallel).

### 5.3 Pipeline Parallelism (Across Phases)

**Dependencies:**
```
Baseline → Activations → SAE Training → Attribution → Interventions
```

Use SLURM job dependencies:
```bash
JOB1=$(sbatch --parsable slurm_scripts/01_baseline.slurm)
JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable slurm_scripts/02_capture_activations.slurm)
JOB3=$(sbatch --dependency=afterok:$JOB2 --array=10-24 slurm_scripts/03_train_sae_array.slurm)
# etc.
```

---

## 6. Failure Recovery

### 6.1 Checkpointing Strategy

**SAE training (critical):**
- Checkpoint every 5,000 steps
- Save optimizer state, step count, and RNG state
- Resume with: `--resume_from models/saes/checkpoint.pt`

**Intervention experiments:**
- Save results incrementally (every 100 samples)
- Use checkpoint files: `results_partial_N.csv`
- Skip completed samples on resume

### 6.2 Timeout Handling

If job hits walltime limit:
```bash
# Check last checkpoint
ls -lht models/saes/layer_15_checkpoint_*.pt | head -1

# Resubmit with resume flag
sbatch --time=12:00:00 slurm_scripts/03_train_sae_single.slurm \
    --layer 15 \
    --resume_from models/saes/layer_15_checkpoint_45000.pt
```

---

## 7. Monitoring and Profiling

### 7.1 GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi dmon -s u -o T > gpu_utilization.log &
```

**Expected utilization:**
- Inference: 60-80%
- SAE training: 40-60%
- Attribution: 85-95%

Low utilization → CPU bottleneck (increase `num_workers` in DataLoader).

### 7.2 Memory Profiling

```python
# In Python script
import torch

torch.cuda.memory._record_memory_history(enabled=True)

# ... run code ...

torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
```

Analyze with `memory_viz` tool.

### 7.3 Performance Profiling

```bash
# Profile with PyTorch profiler
python scripts/04_train_sae.py --profile

# Generates TensorBoard logs
tensorboard --logdir=runs/
```

---

## 8. Validation Tests

Before full-scale runs, validate:

### 8.1 Memory Test
```bash
# Single prompt, single layer
python scripts/03_capture_activations.py --n_samples 1 --layers 15
```

### 8.2 Speed Test
```bash
# Time 10 prompts
time python scripts/03_capture_activations.py --n_samples 10
```

### 8.3 SAE Convergence Test
```bash
# Train for 1,000 steps
python scripts/04_train_sae.py --layer 15 --max_steps 1000
```

Check that reconstruction R² > 0.7 after 1k steps.

---

## 9. Reproducibility Requirements

For exact reproduction on CD3:

1. **Record environment:**
   ```bash
   module list > logs/modules_$(date +%Y%m%d).txt
   pip freeze > logs/packages_$(date +%Y%m%d).txt
   nvidia-smi > logs/gpu_info.txt
   ```

2. **Log resource usage:**
   ```bash
   sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,GPUUtil > logs/job_${SLURM_JOB_ID}_resources.txt
   ```

3. **Save random seeds:** All seeds in `configs/experiment_config.yaml`

4. **Archive outputs:** Save all outputs, not just final results

---

## 10. Troubleshooting Performance Issues

### Issue: Low GPU utilization (<50%)

**Diagnosis:**
- CPU bottleneck (data loading)
- Small batch size
- Synchronization overhead

**Solutions:**
- Increase DataLoader `num_workers`
- Increase batch size
- Use pinned memory: `DataLoader(..., pin_memory=True)`

### Issue: Out of memory errors

**Diagnosis:**
- Batch size too large
- Activation accumulation
- Memory leak

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision
- Clear cache: `torch.cuda.empty_cache()`

### Issue: Slow I/O

**Diagnosis:**
- Large activation files
- Single-threaded loading
- Network filesystem latency

**Solutions:**
- Use memory-mapped files
- Enable Lustre striping
- Load data to local `/tmp` on compute node first

---

## Contact

For computational optimization questions, consult CD3 documentation or contact support.
