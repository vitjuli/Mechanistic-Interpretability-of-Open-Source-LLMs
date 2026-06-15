#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j130-span
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j130-%j.out
#SBATCH --error=logs/j130-%j.out
#
# 130 — Attribution-span geometric test. Reuses 110's span machinery on the
# Q1 attribution-selected feature set. Tests whether Q1 features actually
# span the causal direction u in the dictionary subspace, vs frequency-
# matched / random feature sets.
#
# Geometric (no forward passes, no ablation) — purely transcoder W_dec
# span analysis. Expected runtime: ~20-30 min (transcoder load ~6 min,
# computation per layer ~3-5 min × 6 layers, null sets n=500).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta/attribution_span

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

echo "==== 130: attribution-span geometric test ===="
date

python -u scripts/130_attribution_span_test.py \
    --corpus data/prompts/B1_alpha_beta.jsonl \
    --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --feature_metrics data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv \
    --repo_root . \
    --layers 19 20 21 22 23 24 \
    --n_null 500 \
    --out_dir data/analysis/runD_v2/B1_alpha_beta/attribution_span

echo ""
echo "==== Done: $(date) ===="
