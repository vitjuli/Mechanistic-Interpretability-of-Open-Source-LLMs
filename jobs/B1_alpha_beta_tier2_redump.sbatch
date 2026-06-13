#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=B1ab-t2dump
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/B1ab-t2dump-%j.out
#SBATCH --error=logs/B1ab-t2dump-%j.out
#
# Re-run tier 2 on B1_alpha_beta with --dump_cells to get per-target records.
# Required for flip_norm_intact and intact_given_flip metrics. ~30 min.
# Tier2 layers as auto-selected in original sweep: [16, 22, 23, 24, 35].

set -uo pipefail
mkdir -p logs

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "==== B1 alpha_beta tier 2 RE-DUMP with --dump_cells ===="
date

python -u scripts/122_b1_steering_sweep.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --tier 2 \
    --tier2_layers 16 22 23 24 35 \
    --dump_cells \
    --out_dir data/analysis/runD_v2/B1_alpha_beta

echo "==== Done: $(date) ===="
