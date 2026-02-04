# CSD3 Pipeline Guide: Grammar Agreement Behaviour

Complete step-by-step instructions for running the mechanistic interpretability pipeline on CSD3.

## Table of Contents
1. [Initial Setup](#1-initial-setup)
2. [Environment Configuration](#2-environment-configuration)
3. [Pipeline Steps](#3-pipeline-steps)
4. [SLURM Job Scripts](#4-slurm-job-scripts)
5. [Monitoring & Troubleshooting](#5-monitoring--troubleshooting)

---

## 1. Initial Setup

### 1.1 Connect to CSD3

```bash
# Login to CSD3 login node
ssh <username>@login.hpc.cam.ac.uk

# Or use specific login node
ssh <username>@login-cpu.hpc.cam.ac.uk
```

### 1.2 Navigate to Project Directory

```bash
# Create project directory if not exists
mkdir -p ~/rds/hpc-work/thesis
cd ~/rds/hpc-work/thesis

# Clone or copy your project
# Option 1: Git clone
git clone <your-repo-url> project
cd project

# Option 2: Copy from local (run from your local machine)
# rsync -avz --progress /Users/julia/Desktop/courses/thesis/project/ <username>@login.hpc.cam.ac.uk:~/rds/hpc-work/thesis/project/
```

### 1.3 Project Structure Verification

```bash
# Verify project structure
ls -la
# Should see:
# configs/
# scripts/
# src/
# data/
# docs/
# requirements.txt
```

---

## 2. Environment Configuration

### 2.1 Load Required Modules

```bash
# Load Python and CUDA modules
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

# Verify
python --version  # Should be 3.11.x
which python
```

### 2.2 Create Virtual Environment

```bash
# Create venv in project directory
python -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2.3 Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install additional dependencies if needed
pip install transformers accelerate safetensors
pip install huggingface_hub
pip install pyyaml tqdm pandas matplotlib seaborn networkx
pip install scipy scikit-learn
```

### 2.4 Hugging Face Authentication

```bash
# Login to Hugging Face (for model access)
huggingface-cli login
# Enter your token when prompted

# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

### 2.5 Create Data Directories

```bash
mkdir -p data/prompts
mkdir -p data/activations
mkdir -p data/results
mkdir -p figures
mkdir -p logs
```

---

## 3. Pipeline Steps

### Overview

| Step | Script | GPU Required | Approx. Time | Output |
|------|--------|--------------|--------------|--------|
| 1 | 01_generate_prompts.py | No | <1 min | data/prompts/*.jsonl |
| 2 | 02_run_baseline.py | Yes | 5-10 min | data/results/baseline_* |
| 3 | 03_capture_activations.py | Yes | 10-20 min | data/activations/*.npy |
| 4 | 04_extract_transcoder_features.py | Yes | 15-30 min | data/results/transcoder_features/ |
| 5 | 06_build_attribution_graph.py | Yes | 20-40 min | data/results/attribution_graphs/ |
| 6 | 07_run_interventions.py | Yes | 30-60 min | data/results/interventions/ |
| 7 | 08_generate_figures.py | No | <5 min | figures/*.png |

---

## 4. SLURM Job Scripts

### 4.1 Create Job Scripts Directory

```bash
mkdir -p jobs
```

### 4.2 Step 1: Generate Prompts (CPU only)

Create `jobs/01_generate_prompts.sh`:

```bash
#!/bin/bash
#SBATCH -J gen_prompts
#SBATCH -A <YOUR_ACCOUNT>          # e.g., COMPUTERLAB-SL2-CPU
#SBATCH -p cclake                   # CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -o logs/01_prompts_%j.out
#SBATCH -e logs/01_prompts_%j.err

# Load modules
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# Activate environment
source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

# Run
echo "=== Starting prompt generation ==="
date
python scripts/01_generate_prompts.py
echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/01_generate_prompts.sh
```

### 4.3 Step 2: Run Baseline (GPU)

Create `jobs/02_run_baseline.sh`:

```bash
#!/bin/bash
#SBATCH -J baseline
#SBATCH -A <YOUR_ACCOUNT>          # e.g., COMPUTERLAB-SL2-GPU
#SBATCH -p ampere                   # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -o logs/02_baseline_%j.out
#SBATCH -e logs/02_baseline_%j.err

# Load modules
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

# Activate environment
source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

# Set environment
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# Run
echo "=== Starting baseline evaluation ==="
echo "GPU Info:"
nvidia-smi
date

python scripts/02_run_baseline.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/02_run_baseline.sh
```

### 4.4 Step 3: Capture Activations (GPU)

Create `jobs/03_capture_activations.sh`:

```bash
#!/bin/bash
#SBATCH -J capture_acts
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH -o logs/03_activations_%j.out
#SBATCH -e logs/03_activations_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

echo "=== Starting activation capture ==="
nvidia-smi
date

python scripts/03_capture_activations.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/03_capture_activations.sh
```

### 4.5 Step 4: Extract Transcoder Features (GPU)

Create `jobs/04_extract_features.sh`:

```bash
#!/bin/bash
#SBATCH -J transcoder
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH -o logs/04_transcoder_%j.out
#SBATCH -e logs/04_transcoder_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

echo "=== Starting transcoder feature extraction ==="
nvidia-smi
date

python scripts/04_extract_transcoder_features.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/04_extract_features.sh
```

### 4.6 Step 5: Build Attribution Graph (GPU)

Create `jobs/06_attribution_graph.sh`:

```bash
#!/bin/bash
#SBATCH -J attribution
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -o logs/06_attribution_%j.out
#SBATCH -e logs/06_attribution_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

echo "=== Starting attribution graph construction ==="
nvidia-smi
date

python scripts/06_build_attribution_graph.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/06_attribution_graph.sh
```

### 4.7 Step 6: Run Interventions (GPU)

Create `jobs/07_interventions.sh`:

```bash
#!/bin/bash
#SBATCH -J interventions
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH -o logs/07_interventions_%j.out
#SBATCH -e logs/07_interventions_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

echo "=== Starting intervention experiments ==="
nvidia-smi
date

python scripts/07_run_interventions.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/07_interventions.sh
```

### 4.8 Step 7: Generate Figures (CPU)

Create `jobs/08_figures.sh`:

```bash
#!/bin/bash
#SBATCH -J figures
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -o logs/08_figures_%j.out
#SBATCH -e logs/08_figures_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

echo "=== Generating figures ==="
date

python scripts/08_generate_figures.py

echo "=== Completed ==="
date
```

Submit:
```bash
sbatch jobs/08_figures.sh
```

---

## 4.9 Run All Steps Sequentially (Dependency Chain)

Create `jobs/run_full_pipeline.sh`:

```bash
#!/bin/bash
# Submit all jobs with dependencies

cd ~/rds/hpc-work/thesis/project

# Step 1: Generate prompts (no dependency)
JOB1=$(sbatch --parsable jobs/01_generate_prompts.sh)
echo "Submitted Step 1: Job $JOB1"

# Step 2: Baseline (depends on Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/02_run_baseline.sh)
echo "Submitted Step 2: Job $JOB2 (depends on $JOB1)"

# Step 3: Capture activations (depends on Step 1)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/03_capture_activations.sh)
echo "Submitted Step 3: Job $JOB3 (depends on $JOB1)"

# Step 4: Extract features (depends on Step 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 jobs/04_extract_features.sh)
echo "Submitted Step 4: Job $JOB4 (depends on $JOB3)"

# Step 5: Attribution graph (depends on Step 4)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/06_attribution_graph.sh)
echo "Submitted Step 5: Job $JOB5 (depends on $JOB4)"

# Step 6: Interventions (depends on Step 5)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 jobs/07_interventions.sh)
echo "Submitted Step 6: Job $JOB6 (depends on $JOB5)"

# Step 7: Figures (depends on Step 6)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB6 jobs/08_figures.sh)
echo "Submitted Step 7: Job $JOB7 (depends on $JOB6)"

echo ""
echo "=== Full pipeline submitted ==="
echo "Monitor with: squeue -u \$USER"
```

Run:
```bash
chmod +x jobs/run_full_pipeline.sh
./jobs/run_full_pipeline.sh
```

---

## 5. Monitoring & Troubleshooting

### 5.1 Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# View all pending/running jobs
squeue -u $USER -t PENDING,RUNNING
```

### 5.2 View Logs

```bash
# View output log
cat logs/02_baseline_<JOB_ID>.out

# Tail log in real-time
tail -f logs/02_baseline_<JOB_ID>.out

# View error log
cat logs/02_baseline_<JOB_ID>.err
```

### 5.3 Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

### 5.4 Check GPU Availability

```bash
# Check ampere partition status
sinfo -p ampere

# Check your allocation usage
mybalance
```

### 5.5 Interactive Session (for debugging)

```bash
# Request interactive GPU session
sintr -A <YOUR_ACCOUNT> -p ampere --gres=gpu:1 --time=01:00:00 --mem=32G

# Once allocated, load modules and test
module load python/3.11.0-icl cuda/12.1
source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

# Test a script interactively
python scripts/01_generate_prompts.py
```

### 5.6 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check venv is activated, reinstall package |
| `CUDA out of memory` | Reduce batch_size in config, request more GPU memory |
| `FileNotFoundError` | Check paths, ensure previous step completed |
| `Permission denied` | Check file permissions with `ls -la` |
| Job pending long time | Check allocation balance with `mybalance` |

### 5.7 Check Results

```bash
# Verify outputs exist
ls -la data/prompts/
ls -la data/results/
ls -la data/activations/
ls -la figures/

# Check prompt files
head data/prompts/grammar_agreement_train.jsonl

# Check baseline results
cat data/results/baseline_metrics_train.json
```

---

## Quick Reference

### Account Setup
Replace `<YOUR_ACCOUNT>` in all job scripts with your actual SLURM account:
- CPU jobs: Usually ends with `-CPU` (e.g., `COMPUTERLAB-SL2-CPU`)
- GPU jobs: Usually ends with `-GPU` (e.g., `COMPUTERLAB-SL2-GPU`)

Check your accounts:
```bash
sacctmgr show assoc user=$USER format=account%30
```

### Complete Workflow

```bash
# 1. SSH to CSD3
ssh <username>@login.hpc.cam.ac.uk

# 2. Navigate to project
cd ~/rds/hpc-work/thesis/project

# 3. Activate environment
source venv/bin/activate

# 4. Submit full pipeline
./jobs/run_full_pipeline.sh

# 5. Monitor progress
watch -n 30 'squeue -u $USER'

# 6. Check final results
ls -la figures/
ls -la data/results/
```

### Download Results to Local Machine

From your local machine:
```bash
# Download figures
scp -r <username>@login.hpc.cam.ac.uk:~/rds/hpc-work/thesis/project/figures/ ./

# Download all results
rsync -avz <username>@login.hpc.cam.ac.uk:~/rds/hpc-work/thesis/project/data/results/ ./results/
```
