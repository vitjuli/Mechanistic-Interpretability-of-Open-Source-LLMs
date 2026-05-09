#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=abs-baseline
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/abs-baseline-%j.out
#SBATCH --error=logs/abs-baseline-%j.out

# sbatch jobs/run_abstraction_baseline.sbatch
# For 8B: #SBATCH --mem=64G --time=05:00:00 and add --model_size 8b below

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-4b}

cd /rds/user/iv294/hpc-work/thesis/project
module purge
module load rhel8/default-amp 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Abstraction probe baseline — ${MODEL_SIZE}"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -4 || true

# Generate datasets (idempotent — skips if already present)
python -u scripts/60_generate_abstraction_probe_datasets.py

echo ""
echo "── Evaluating all families ──"
python -u scripts/61_run_abstraction_baselines.py \
    --model_size "$MODEL_SIZE" \
    --device cuda \
    --dtype bfloat16 \
    --split train \
    --families A B C D E \
    --rep_analysis \
    --rep_family A

echo ""
echo "── Exit: $?  $(date) ──"
echo ""
echo "Results in: data/results/abstraction_probe/${MODEL_SIZE/./}"
echo ""
echo "To sync locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/abstraction_probe/ data/results/abstraction_probe/"
echo "========================================================"
