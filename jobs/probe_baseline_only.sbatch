#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe_base
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=logs/probe_base-%j.out
#SBATCH --error=logs/probe_base-%j.out
#
# Runs baseline evaluation only for physics_decay_type_probe.
# Section A of the main pipeline failed because physics_decay_type_probe
# was missing from experiment_config.yaml — now fixed.
#
# BEFORE SUBMITTING:
#   git pull   (picks up the config fix)
#   sbatch jobs/probe_baseline_only.sbatch

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID  Node $(hostname)  Started $(date)"

python -u scripts/02_run_baseline.py \
    --behaviour physics_decay_type_probe \
    --split train \
    --no-viz

echo "Done $(date)"
echo "Output: data/results/baseline_physics_decay_type_probe_train.csv"
