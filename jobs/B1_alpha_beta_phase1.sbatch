#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=B1ab-p1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/B1ab-p1-%j.out
#SBATCH --error=logs/B1ab-p1-%j.out
#
# B1 alpha_beta — Phase 1: precheck + field dump (~20 min real, 1h walltime).
# High priority due to short time request.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "==== Phase 1: 121 precheck + 119 field dump ===="
date

# Stage 1: precheck gate (writes precheck_alpha_beta.json next to corpus)
python -u scripts/121_b1_precheck.py \
    --corpus data/prompts/B1_alpha_beta.jsonl
PRECHECK=$?
if [ $PRECHECK -ne 0 ]; then
    echo "==== PRECHECK FAILED (exit=$PRECHECK) ===="
    exit $PRECHECK
fi

# Stage 2: field dump
python -u scripts/119_capture_field_dump.py \
    --prompts data/prompts/B1_alpha_beta.jsonl \
    --out_dir data/analysis/runD_v2/B1_alpha_beta/field_dump

echo "==== Phase 1 done ===="
date
