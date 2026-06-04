#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j78
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j78-%j.out
#SBATCH --error=logs/j78-%j.out
#
# j78 — cluster-C4 ablation null for intensive/extensive.
# Runs script 78 with direct python invocation to avoid the run_final.sbatch
# word-splitting bug (single-quoted args with spaces get mangled).
#
# Defaults in script 78 already match what we need:
#   --correct " extensive"  --incorrect " intensive"  --target_class extensive
# So we only need to override the prompts and cluster_csv paths.

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  78 — cluster ablation null"
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

# Verify the cluster CSV exists first
CSV=data/results/abstraction_ie/physics_intensive_extensive_v1/ie_cluster_csv_k6.csv
if [ ! -f "$CSV" ]; then
    echo "Cluster CSV not found at $CSV"
    echo "Run scripts/78a_build_ie_cluster_csv.py first."
    exit 1
fi

python -u scripts/78_cluster_ablation_null.py \
    --prompts     data/prompts/physics_intensive_extensive_v1_train.jsonl \
    --cluster_csv "$CSV" \
    --cluster_id  4 \
    --correct     " extensive" \
    --incorrect   " intensive" \
    --target_class extensive \
    --n_null 40 \
    --out_dir data/analysis/iia_failure_diagnosis

echo ""
echo "==== Done: $(date) ===="
