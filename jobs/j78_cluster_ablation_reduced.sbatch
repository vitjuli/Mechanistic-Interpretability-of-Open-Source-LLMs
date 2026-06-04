#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j78-r
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/j78r-%j.out
#SBATCH --error=logs/j78r-%j.out
#
# j78 reduced — cluster C4 ablation null for intensive/extensive.
#
# Reductions vs original (which hit 2h limit on real cluster alone):
#   --n_prompts 96  (was 384) — 4x speedup per ablate_metrics call
#   --n_null 15     (was 40)  — fewer null draws, still gives p95 estimate
#
# Expected: per ablate_metrics ≈ 22 min @ 96 prompts × 26 layers × encode/decode
#           1 real + 15 null = 16 calls × 22 min ≈ 6 hours

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  78 reduced — cluster ablation null"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
nvidia-smi || true
echo ""
date

CSV=data/results/abstraction_ie/physics_intensive_extensive_v1/ie_cluster_csv_k6.csv
if [ ! -f "$CSV" ]; then
    echo "Cluster CSV not found at $CSV"; exit 1
fi

python -u scripts/78_cluster_ablation_null.py \
    --prompts     data/prompts/physics_intensive_extensive_v1_train.jsonl \
    --cluster_csv "$CSV" \
    --cluster_id  4 \
    --correct     " extensive" \
    --incorrect   " intensive" \
    --target_class extensive \
    --n_prompts 96 \
    --n_null 15 \
    --out_dir data/analysis/iia_failure_diagnosis

echo ""
echo "==== Done: $(date) ===="
