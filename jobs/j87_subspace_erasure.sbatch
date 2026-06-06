#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j87-eras
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/j87-%j.out
#SBATCH --error=logs/j87-%j.out
#
# 87 — Subspace erasure (INLP) + mass-axis control. The one job with truly open outcome:
#   * if margin-accuracy survives erasing 13-D concept subspace → bypass is iron-clad
#   * if it collapses below random-k and shuffled-INLP nulls → we found a usage locus

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/subspace_erasure

echo "========================================================"
echo "  87 — subspace erasure (k-D concept axis + null + mass)"
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

# ── smoke (1 min) ──────────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (layers 21 24, 5 targets) ===="
date
python -u scripts/87_subspace_erasure.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 21 24 \
    --k_grid 5 \
    --n_random 2 \
    --n_shuffle 1 \
    --max_targets 5 \
    --no_mass \
    --out_dir data/analysis/runD_v2/subspace_erasure/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run (with mass control) ────────────────────────────────────────────
echo ""
echo "==== FULL RUN (all layers, k=1,5,13, with mass control) ===="
date
python -u scripts/87_subspace_erasure.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/subspace_erasure

echo ""
echo "==== Done: $(date) ===="
