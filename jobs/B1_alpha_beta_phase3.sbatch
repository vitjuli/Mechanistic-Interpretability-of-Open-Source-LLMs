#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=B1ab-p3
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/B1ab-p3-%j.out
#SBATCH --error=logs/B1ab-p3-%j.out
#
# B1 alpha_beta — Phase 3: tier 2 dense grid on auto-selected layers (~1h
# real, 1.5h walltime). Requires field_dump + tier1 csv.

set -uo pipefail
mkdir -p logs

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

echo "==== Phase 3: 122 tier 2 dense grid ===="
date

python -u scripts/122_b1_steering_sweep.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --tier 2 \
    --tier1_csv data/analysis/runD_v2/B1_alpha_beta/steering_sweep_tier1.csv \
    --out_dir data/analysis/runD_v2/B1_alpha_beta

echo "==== Phase 3 done ===="
date
