#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j84-car
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j84-%j.out
#SBATCH --error=logs/j84-%j.out
#
# 84 — full carrier capture across ALL features ALL transcoder layers.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/full_carrier

echo "========================================================"
echo "  84 — full carrier capture (all features, all layers)"
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

# ── smoke test (2 transcoder layers, lightweight) ───────────────────────────
echo ""
echo "==== SMOKE (layers 18 24) ===="
date
python -u scripts/84_full_carrier_capture.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 18 24 \
    --out_dir data/analysis/runD_v2/full_carrier/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run ────────────────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (all transcoder layers L10-L25) ===="
date
python -u scripts/84_full_carrier_capture.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/full_carrier

echo ""
echo "==== Done: $(date) ===="
