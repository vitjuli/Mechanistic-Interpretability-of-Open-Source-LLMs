#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j86-use
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/j86-%j.out
#SBATCH --error=logs/j86-%j.out
#
# 86 — Usage-direction (gradient of logit_β − logit_α), DLA writers, steering efficiency.
# Backward pass on float32 model: ~35-40 GB peak on A100-80GB.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/usage_direction

echo "========================================================"
echo "  86 — usage direction (gradient + DLA + steering eff)"
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
echo "==== SMOKE (layers 21 35, 5 targets) ===="
date
python -u scripts/86_usage_direction.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 21 35 \
    --steer_layers 21 \
    --max_targets 5 \
    --n_random 1 \
    --c_grid 4 \
    --out_dir data/analysis/runD_v2/usage_direction/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run ────────────────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (all 36 layers) ===="
date
python -u scripts/86_usage_direction.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/usage_direction

echo ""
echo "==== Done: $(date) ===="
