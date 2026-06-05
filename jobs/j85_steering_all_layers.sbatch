#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j85-str
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j85-%j.out
#SBATCH --error=logs/j85-%j.out
#
# 85 — steering at ALL layers (extends j75 from 4 layers to full residual).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/steering_all_layers

echo "========================================================"
echo "  85 — steering at all layers"
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

# ── smoke (2 layers, 5 prompts) ──────────────────────────────────────────────
echo ""
echo "==== SMOKE (layers 20 24, 5 prompts) ===="
date
python -u scripts/85_steering_all_layers.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 20 24 \
    --max_targets 5 \
    --out_dir data/analysis/runD_v2/steering_all_layers/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run ────────────────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (all 36 layers) ===="
date
python -u scripts/85_steering_all_layers.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/steering_all_layers

echo ""
echo "==== Done: $(date) ===="
