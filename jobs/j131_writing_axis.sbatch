#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j131-write
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j131-%j.out
#SBATCH --error=logs/j131-%j.out
#
# 131 — Writing-axis test. Introduces the GENUINE per-layer writing direction
# delta_write[L] = unit(mean_beta(attn_out+mlp_out) - mean_alpha(...)) and
# compares it against the three state-side axes (delta_state, w_res, u) to settle
# whether the old "delta" deserves the name "writing axis" and to directly
# measure the four-way dissociation decoded != written != used.
#
# One forward pass per prompt with hooks on self_attn + mlp outputs at the last
# position. No backward (gradients for u come from the existing field dump).
# Expected runtime: model load ~6 min, captures ~minutes on 538 prompts.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta/writing_axis

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

echo "==== 131: writing-axis test (delta_write vs state-side axes) ===="
date

python -u scripts/131_writing_axis_test.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --out_dir data/analysis/runD_v2/B1_alpha_beta/writing_axis \
    --band_lo 19 --band_hi 24 \
    --n_null 500

echo ""
echo "==== Done: $(date) ===="
