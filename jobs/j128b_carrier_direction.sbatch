#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j128b-direction
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/j128b-%j.out
#SBATCH --error=logs/j128b-%j.out
#
# 128b — Carrier direction-resolved decomposition. Mirrors 128 (same Q1
# feature sets, same frequency-matched nulls, same transcoder API), but
# measures the DIRECTION of the ablation delta vector:
#   - along_u   = |⟨delta, û⟩|
#   - total_norm = ‖delta‖
#   - u_fraction = along_u / total_norm
# Compared against null. u_fraction > null → Q1 features are
# direction-selective for the causal axis (partial localization).
# u_fraction ≈ null → above-null |Δm| from 128 was a magnitude artifact,
# not direction alignment.
#
# Speed-tuned: --n_null 10 (was 20), --max_targets 80 (was 188).
# Expected runtime: ~5-6 hours on A100 (~45-60 min per layer × 6 layers).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta/carrier_direction

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

echo "==== 128b: carrier direction-resolved decomposition ===="
date

python -u scripts/128b_carrier_direction_resolved.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --feature_metrics data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv \
    --repo_root . \
    --layers 19 20 21 22 23 24 \
    --n_null 10 \
    --max_targets 80 \
    --out_dir data/analysis/runD_v2/B1_alpha_beta/carrier_direction

echo ""
echo "==== Done: $(date) ===="
