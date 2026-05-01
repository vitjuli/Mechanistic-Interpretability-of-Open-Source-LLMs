#!/bin/bash
#SBATCH --job-name=cja_probe
#SBATCH --output=logs/cja_probe-%j.out
#SBATCH --error=logs/cja_probe-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# ── Cluster joint ablation: physics_decay_type_probe ──────────────────────
# Runs all 11 clusters × 470 prompts in one job.
# Expected time: ~2.5–3.5 hours for all clusters.
# Output: data/results/cluster_joint_ablation/joint_ablation_physics_decay_type_probe_train.csv

set -euo pipefail

echo "=== Cluster Joint Ablation: physics_decay_type_probe ==="
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | $(date)"
echo ""

# ── Environment setup ───────────────────────────────────────────────────
module purge
module load rhel8/default-amp

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate transcoder

cd /home/$USER/projects/thesis/project

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── Verify inputs ───────────────────────────────────────────────────────
echo ""
echo "=== Input verification ==="
ls data/results/clustering/cluster_labels.csv && echo "cluster_labels.csv OK"
ls data/results/grouping/feature_prompt_contributions.csv && echo "contributions.csv OK"
ls data/prompts/physics_decay_type_probe_train.jsonl && echo "prompts OK"

mkdir -p data/results/cluster_joint_ablation logs

# ── Run joint ablation ──────────────────────────────────────────────────
echo ""
echo "=== Running joint ablation: all clusters, all prompts ==="
echo "Start: $(date)"

python scripts/27_cluster_joint_ablation.py \
    --behaviour physics_decay_type_probe \
    --split     train \
    --clusters  all \
    --device    cuda

echo ""
echo "=== Ablation complete: $(date) ==="

# ── Run analysis ────────────────────────────────────────────────────────
echo "=== Running analysis script (27b) ==="
python scripts/27b_analyse_joint_ablation.py

echo ""
echo "=== DONE: $(date) ==="
ls -lh data/results/cluster_joint_ablation/
