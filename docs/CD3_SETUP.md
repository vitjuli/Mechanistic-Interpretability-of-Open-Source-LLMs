# Cambridge CD3 Supercomputer Setup Guide

## Overview

This document provides detailed instructions for running the mechanistic interpretability pipeline on **Cambridge CD3 (CSD3)**, the University's High Performance Computing service.

---

## 1. Initial Setup on CD3

### 1.1 Connect to CD3

```bash
# SSH to login node
ssh <your_crsid>@login.hpc.cam.ac.uk

# Or directly to CD3
ssh <your_crsid>@login-gpu.hpc.cam.ac.uk
```

### 1.2 Navigate to Project Directory

```bash
# Use your Raven scratch space for large data
cd /rds/user/<your_crsid>/hpc-work/

# Clone repository
git clone <repo-url> qwen-circuits
cd qwen-circuits
```

### 1.3 Load Modules

CD3 uses module system for software management:

```bash
# Check available modules
module avail

# Load required modules (adjust versions as needed)
module load python/3.10
module load cuda/12.1
module load gcc/11.2.0
module load anaconda/python3

# Add to your ~/.bashrc for persistence
echo "module load python/3.10 cuda/12.1" >> ~/.bashrc
```

### 1.4 Create Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 1.5 Configure HuggingFace Cache

Avoid filling home directory quota:

```bash
# Create cache directory in scratch space
mkdir -p /rds/user/<your_crsid>/hpc-work/hf_cache

# Set environment variables (add to ~/.bashrc)
export TRANSFORMERS_CACHE=/rds/user/<your_crsid>/hpc-work/hf_cache
export HF_HOME=/rds/user/<your_crsid>/hpc-work/hf_cache
export TORCH_HOME=/rds/user/<your_crsid>/hpc-work/torch_cache
```

---

## 2. Resource Requirements

### 2.1 Computational Estimates

| Phase | Walltime | GPUs | Memory | Storage |
|-------|----------|------|--------|---------|
| **Baseline** | 1-2 hours | 1 | 32GB | 1GB |
| **Activations** | 2-4 hours | 1 | 64GB | 20GB |
| **SAE Training (per layer)** | 6-12 hours | 1 | 48GB | 5GB |
| **SAE Training (all layers, parallel)** | 12 hours | 15 | 48GB each | 75GB |
| **Attribution Graphs** | 4-6 hours | 1 | 64GB | 5GB |
| **Interventions** | 6-8 hours | 1 | 48GB | 10GB |
| **Full Pipeline** | 24-48 hours | 2 | 128GB | 100GB |

### 2.2 CD3 Partition Information

CD3 has several GPU partitions:

```bash
# Check available partitions
sinfo -o "%20P %10a %10l %5D %6t %15C %8z %15m"

# Common GPU partitions:
# - ampere: NVIDIA A100 GPUs (40GB/80GB)
# - pascal: NVIDIA P100 GPUs (older, 16GB)
# - volta: NVIDIA V100 GPUs (32GB)
```

**Recommendation:** Use `ampere` partition for this project (A100 GPUs optimal for BF16 inference).

### 2.3 Storage Quotas

```bash
# Check your quota
quota -s

# Typical limits:
# - Home directory: 40GB (DO NOT store models/activations here)
# - Scratch (hpc-work): 1TB+ (use for all project data)
```

---

## 3. Running the Pipeline

### 3.1 Quick Start (Individual Phases)

```bash
# Make SLURM scripts executable
chmod +x slurm_scripts/*.slurm

# Phase 0+1: Generate prompts and run baseline
sbatch slurm_scripts/01_baseline.slurm

# Phase 2: Capture activations
sbatch slurm_scripts/02_capture_activations.slurm

# Phase 3: Train SAEs in parallel (15 jobs, one per layer)
sbatch slurm_scripts/03_train_sae_array.slurm

# Phase 4: Build attribution graphs
sbatch slurm_scripts/04_attribution_graph.slurm

# Phase 5: Run interventions
sbatch slurm_scripts/05_interventions.slurm
```

### 3.2 Full Pipeline (Single Job)

```bash
# Submit full pipeline (runs sequentially, 48 hours)
sbatch slurm_scripts/pipeline_full.slurm
```

### 3.3 Monitoring Jobs

```bash
# Check job status
squeue -u <your_crsid>

# Check specific job
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View output logs
tail -f logs/baseline_<job_id>.out

# Check GPU utilization (if job is running)
ssh <node_name>
nvidia-smi
```

---

## 4. Optimizations for CD3

### 4.1 Parallel SAE Training

The most compute-intensive phase is SAE training. Use job arrays:

```bash
# Train layers 10-24 in parallel (15 jobs)
sbatch slurm_scripts/03_train_sae_array.slurm

# This creates 15 simultaneous jobs, reducing wall time from ~180 hours to ~12 hours
```

### 4.2 Multi-GPU Inference (Advanced)

For very large prompt sets, enable data parallelism:

```python
# In scripts, set:
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs
python scripts/03_capture_activations.py --distributed
```

### 4.3 Checkpointing Long Jobs

SAE training includes automatic checkpointing:

```bash
python scripts/04_train_sae.py \
    --layer 15 \
    --checkpoint_every 5000 \
    --resume_from models/saes/layer_15_checkpoint_25000.pt
```

If job times out, resubmit with `--resume_from` flag.

---

## 5. Data Management

### 5.1 Directory Structure on CD3

```
/rds/user/<crsid>/hpc-work/qwen-circuits/
├── data/
│   ├── prompts/              (1MB, version controlled)
│   ├── activations/          (20GB, intermediate data)
│   └── results/              (5GB, final outputs)
├── models/
│   └── saes/                 (75GB, trained SAEs)
├── figures/                  (100MB, publication plots)
└── logs/                     (1GB, SLURM outputs)
```

### 5.2 Transferring Results to Local Machine

```bash
# From your local machine:
scp -r <crsid>@login.hpc.cam.ac.uk:/rds/user/<crsid>/hpc-work/qwen-circuits/figures ./
scp -r <crsid>@login.hpc.cam.ac.uk:/rds/user/<crsid>/hpc-work/qwen-circuits/data/results ./
```

### 5.3 Cleanup After Completion

```bash
# Remove large intermediate files to save quota
rm -rf data/activations/*   # 20GB saved
tar -czf saes_backup.tar.gz models/saes/  # Archive SAEs
rm -rf models/saes/*        # 75GB saved (keep archive)
```

---

## 6. Troubleshooting

### 6.1 Common Issues

**Issue:** Job fails with "Out of memory"
```bash
# Solution: Increase --mem in SLURM script
#SBATCH --mem=64G  →  --mem=128G
```

**Issue:** Module not found
```bash
# Solution: Check available modules
module spider python
module load python/3.10.4  # Use specific version
```

**Issue:** CUDA out of memory during inference
```python
# Solution: Reduce batch size in config
batch_size: 16  →  batch_size: 4
```

**Issue:** Job stuck in queue
```bash
# Check job reason
squeue -u <crsid> --start

# Consider requesting fewer resources
--nodes=1 --gres=gpu:1  # Instead of multi-GPU
```

### 6.2 CD3 Support

- **Documentation:** https://docs.hpc.cam.ac.uk/
- **Help desk:** support@hpc.cam.ac.uk
- **Status page:** https://status.hpc.cam.ac.uk/

---

## 7. Best Practices

1. **Test locally first:** Run `01_generate_prompts.py` locally to verify prompt quality before submitting large jobs

2. **Start small:** Test with single layer SAE before launching array job for all 15 layers

3. **Monitor costs:** CD3 uses service units. Check consumption:
   ```bash
   mybalance
   ```

4. **Save checkpoints:** Long jobs should checkpoint every 1-2 hours

5. **Version control:** Commit code changes before submitting jobs (track which code produced which results)

6. **Document experiments:** Keep a lab notebook:
   ```bash
   echo "$(date): Submitted job ${SLURM_JOB_ID} with config X" >> experiments.log
   ```

---

## 8. Estimated Timeline

For a complete reproducibility study (2 behaviours):

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Setup | 1 day | Initial environment setup |
| Baseline | 2 hours | None |
| Activation capture | 4 hours | Baseline must pass |
| SAE training | 12 hours (parallel) | Activations ready |
| Feature interpretation | 2 days (manual) | Trained SAEs |
| Attribution graphs | 6 hours | Interpreted features |
| Interventions | 8 hours | Attribution graphs |
| Analysis & figures | 3 days | All data collected |

**Total wall time:** ~1 week compute + 1 week analysis

**Total compute time:** ~200 GPU-hours

---

## 9. Reproducibility Checklist

Before publishing results:

- [ ] All random seeds documented in configs
- [ ] SLURM scripts committed to repository
- [ ] Module versions recorded (`module list > environment.txt`)
- [ ] Package versions saved (`pip freeze > requirements_frozen.txt`)
- [ ] GPU type logged (A100 40GB vs 80GB matters)
- [ ] Walltime and memory usage recorded
- [ ] Failed jobs and reasons documented
- [ ] Raw outputs archived (not just processed results)

---

## Contact

For project-specific questions, contact [project lead].

For CD3 technical issues, contact CSD3 support.
