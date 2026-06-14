#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j128-carrier
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/j128-%j.out
#SBATCH --error=logs/j128-%j.out
#
# 128 — Attribution-carrier ablation. Ablates Q1 features (is_attr=1, 97
# features peaking L19-24) and compares effect vs frequency-matched null
# sets (n_null=20). Tests if used-axis is localizable in dictionary.
#
# Verdict: Q1 moves output > null → partial localization of u via attribution
#          Q1 ≈ null → no dictionary localization, used-axis is distributed
#
# Expected: ~30-45 min real (transcoder load + ablated forward passes).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta/carrier_ablation

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

echo "==== 128: attribution-carrier ablation (Q1 vs freq-matched null) ===="
date

python -u scripts/128_attribution_carrier_ablation.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --feature_metrics data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv \
    --repo_root . \
    --layers 19 20 21 22 23 24 \
    --n_null 20 \
    --out_dir data/analysis/runD_v2/B1_alpha_beta/carrier_ablation

echo ""
echo "==== Done: $(date) ===="
