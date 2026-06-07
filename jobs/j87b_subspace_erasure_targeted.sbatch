#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j87b-tgt
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/j87b-%j.out
#SBATCH --error=logs/j87b-%j.out
#
# 87b — Targeted re-run of subspace erasure on the candidate usage layers found in j87:
#   L18, L23, L24, L25 (partial late-mid) and L34, L35 (largest drop).  k=13 only.
#   Tighten the random-frame and shuffled-INLP nulls: n_random=30, n_shuffle=10 (vs 4/1
#   in the first sweep) -> proper percentile statistics on the 6 below-null candidates.
#   ~2 h. Mass control still skipped (auto coverage 0.01 from text); provide --mass_labels
#   when explicit labels exist.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/subspace_erasure_targeted

echo "========================================================"
echo "  87b — targeted subspace erasure (n_random=30, n_shuffle=10)"
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
date

echo ""
echo "==== TARGETED RUN (L18 L23 L24 L25 L34 L35, k=13, n_random=30) ===="
date
python -u scripts/87_subspace_erasure.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 18 23 24 25 34 35 \
    --k_grid 13 \
    --n_random 30 \
    --n_shuffle 10 \
    --no_mass \
    --out_dir data/analysis/runD_v2/subspace_erasure_targeted

echo ""
echo "==== Done: $(date) ===="
