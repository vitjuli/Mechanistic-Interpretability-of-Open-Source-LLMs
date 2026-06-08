#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j95-ord
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/j95-%j.out
#SBATCH --error=logs/j95-%j.out
#
# 95 — Answer-order control. Tests primacy hypothesis for the α-default:
#   original prompt: "...Is the decay type alpha or beta?"   (α-recall 0.996, β-recall 0.446)
#   swapped prompt:  "...Is the decay type beta or alpha?"   (does β-recall jump?)
#
# Three possible outcomes:
#   (1) β-recall jumps and α-recall drops symmetrically → α-default is positional (primacy)
#   (2) Almost no change → α-default is content-bound (semantic prior favours α-decay)
#   (3) Partial shift → average the two orders as canonical
#
# Closes the LAST methodological confound before designing the healing intervention.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/answer_order

echo "========================================================"
echo "  95 — answer-order control (alpha-first vs beta-first)"
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
echo "==== Answer-order control run ===="
date
python -u scripts/95_answer_order_control.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/answer_order

echo ""
echo "==== Done: $(date) ===="
