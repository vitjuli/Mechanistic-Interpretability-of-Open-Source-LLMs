#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j119-field-dump
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j119-%j.out
#SBATCH --error=logs/j119-%j.out
#
# 119 — Field capture dump. ONE forward+backward per prompt, full corpus,
# all 36 layers. Dumps per-prompt: residuals (post-block), gradients of
# logit-margin wrt residual, margins, top-50 logits.
#
# Two outputs (recommendation a from README): preserve apples-to-apples
# split with 86/89 retro-compatibility.
#   pass 1: field_dump      — v2_train only         (for retro-join in 116)
#   pass 2: field_dump_full — v2_train + supplement (for future B1 battery)
#
# Built-in checks: raw-margin = log-softmax-margin identity (from 89);
# cos(u_final, γ̄) sanity for machinery validation.
# Expected runtime: ~30-60 min per pass on A100 → ~75-120 min total.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/field_dump data/analysis/runD_v2/field_dump_full

echo "========================================================"
echo "  119 — field dump (residuals + grad + margins + topk)"
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
echo "==== PASS 1 — v2_train only (retro-compatible with 86/89) ===="
python -u scripts/119_capture_field_dump.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/field_dump

echo ""
echo "==== PASS 2 — v2_train + supplement (full for B1 battery) ===="
python -u scripts/119_capture_field_dump.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --extra_prompts data/prompts/physics_decay_type_probe_supplement.jsonl \
    --out_dir data/analysis/runD_v2/field_dump_full

echo ""
echo "==== Done: $(date) ===="
