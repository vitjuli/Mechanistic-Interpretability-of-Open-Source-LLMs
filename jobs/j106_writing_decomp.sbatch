#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j106-wdec
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j106-%j.out
#SBATCH --error=logs/j106-%j.out
#
# 106 — Writing decomposition. For each layer, compute δ = μ_β − μ_α (the
# direction the model WRITES between classes) and decompose its energy:
#   * used: projection on u (gradient of margin)
#   * quiet: projection on the orthogonal complement of u
# Forced regime, ~3-4 min real.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/writing_decomp

echo "========================================================"
echo "  106 — writing decomposition (used vs quiet)"
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

python -u scripts/106_writing_decomposition.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/writing_decomp

echo ""
echo "==== Done: $(date) ===="
