#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=val-tests
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/val-tests-%j.out
#SBATCH --error=logs/val-tests-%j.out

set -uo pipefail
mkdir -p logs
echo "==== Validation Tests  Job: ${SLURM_JOB_ID}  $(date) ===="
module purge
module load rhel8/default-amp 2>/dev/null || true
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python -u scripts/54_validation_tests.py
echo "==== Done: $(date) ===="
