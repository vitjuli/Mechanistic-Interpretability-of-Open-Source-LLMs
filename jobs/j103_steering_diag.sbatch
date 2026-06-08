#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j103-diag
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j103-%j.out
#SBATCH --error=logs/j103-%j.out
#
# 103 — Steering diagnostic. For layers {17, 21, 24, 30}, print all the quantities
# needed to decide whether "w_res-steering does not flip" is:
#   (A) BUG — addition c·σ is tiny vs ||h||, hook didn't apply, norms mismatched
#   (B) PHYSICS — proj_shift = c·σ exactly, but cos(g, w_res) ≈ 0.03 and cos(g, u) ≈ 1.0,
#       so the same code along u flips and the output is genuinely blind to w_res
#
# Sanity rows printed at the top:
#   * ZERO-ablation (set h=0 at layer L) intact-rate must drop below 1.0 → hook works
#   * Same code along u at same c·σ → intact ≈ 1.0
#   * Same code along w_res at same c·σ → intact ≈ 0
#   → if both sanity pass, the code is correct and the difference is direction-only

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/steer_diag

echo "========================================================"
echo "  103 — steering diagnostic (decide bug vs physics)"
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

echo ""
echo "==== Steering diagnostic run (L17 L21 L24 L30, c=2,4,8, n=16) ===="
date
python -u scripts/103_steering_diagnostic.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 17 21 24 30 --c_list 2 4 8 --n 16 \
    --out_dir data/analysis/runD_v2/steer_diag

echo ""
echo "==== Done: $(date) ===="
