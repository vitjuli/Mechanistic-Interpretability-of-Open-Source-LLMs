#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j100-u
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j100-%j.out
#SBATCH --error=logs/j100-%j.out
#
# 100 — Usage-direction deep-dive across all 36 layers in forced regime:
#   (A) intact-flip profile of u vs w_res per layer (lever depth)
#   (B) AUC along u per layer (where does the used direction encode the concept?)
#   (C) logit-lens β-rank along u per layer (anatomy of what u carries)

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/u_deepdive

echo "========================================================"
echo "  100 — usage-direction deep-dive (all 36 layers)"
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

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
nvidia-smi || true
date

# ── smoke (~5 min) ────────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L17 L21 L24, c=2, n=8) ===="
date
python -u scripts/100_usage_direction_deepdive.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 17 21 24 --c_list 2 --n_steer 8 \
    --out_dir data/analysis/runD_v2/u_deepdive_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~45-60 min — all 36 layers × 2 amplitudes × 24 prompts) ─────────
echo ""
echo "==== FULL RUN (all 36 layers, c=2,4, n=24) ===="
date
python -u scripts/100_usage_direction_deepdive.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers -1 --c_list 2 4 --n_steer 24 \
    --out_dir data/analysis/runD_v2/u_deepdive

echo ""
echo "==== Done: $(date) ===="
