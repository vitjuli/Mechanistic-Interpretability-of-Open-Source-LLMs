#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=h3-reactive
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/h3-reactive-%j.out
#SBATCH --error=logs/h3-reactive-%j.out
#
# H3-reactive: ablate each cluster, measure backup activation in remaining
# monitored features (40 cluster features from grouping/feat_ids.json).
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/iia_failure_diagnosis
#   sbatch jobs/run_h3_reactive_backup.sbatch

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  H3-Reactive Backup Detection (Wang 2022 style)"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
nvidia-smi || true
echo ""

date
python -u scripts/54_h3_reactive_backup.py \
    --behaviour    physics_decay_type_probe_v2 \
    --split        train \
    --max_prompts  80

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/ data/analysis/iia_failure_diagnosis/"
