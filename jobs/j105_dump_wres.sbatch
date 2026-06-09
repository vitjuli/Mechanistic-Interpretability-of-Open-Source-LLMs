#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j105-wres
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j105-%j.out
#SBATCH --error=logs/j105-%j.out
#
# 105 — Dump Fisher LDA w_res at ALL 36 residual-stream layers + final.
# Enables the dashboard ROTATION mode (Page 1, Mode C) to visualise w_res
# rotation across the full depth. ~5 min real, output ~750 KB.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/wres_all_layers

echo "========================================================"
echo "  105 — dump w_res at all 36 layers"
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
echo "==== Dumping w_res at all 36 layers + final ===="
date
python -u scripts/105_dump_wres_all_layers.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/wres_all_layers

echo ""
echo "==== Done: $(date) ===="
