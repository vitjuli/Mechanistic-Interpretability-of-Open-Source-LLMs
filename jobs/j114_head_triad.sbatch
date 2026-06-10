#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j114-head-triad
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j114-%j.out
#SBATCH --error=logs/j114-%j.out
#
# 114 — Head→triad bridge. For every (layer, head), decompose the head's
# o_proj-mediated contribution to the residual stream into shares along
# the 4 reference axes {γ̄, u, w_res, δ}. Identifies "answer-writer"
# heads (large γ̄/u share) vs "axis-writer" heads (large w_res share).
# All 36 layers (-1 sentinel). Expected runtime: ~8-12 min real.
#
# No transcoders needed.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/head_triad

echo "========================================================"
echo "  114 — head→triad bridge — ALL 36 LAYERS"
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

python -u scripts/114_head_triad_bridge.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers -1 \
    --topk_heads 5 \
    --out_dir data/analysis/runD_v2/head_triad

echo ""
echo "==== Done: $(date) ===="
