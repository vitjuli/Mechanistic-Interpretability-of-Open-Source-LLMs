#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j88e-car
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/j88e-%j.out
#SBATCH --error=logs/j88e-%j.out
#
# 88e — Carrier-capture (E) dozapusk for j88. The main j88 run (j88-30199249) completed
# A+B+C+D but skipped (E) because transcoder_loader was not importable.
# Here we:
#   - set PYTHONPATH=src/transcoder so the import resolves
#   - run only carrier layers 10-25, k=5, small max_targets to skip the long erasure phase
#   - the (E) section uses gradient-mean u^(L) and the 163,840 transcoder decoder columns

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/usage_subspace

echo "========================================================"
echo "  88e — carrier-only dozapusk (E section)"
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

# CRITICAL: transcoder_loader lives at src/transcoder/transcoder_loader.py
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

echo "Python: $(which python)"
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

# Run with --carrier; --max_targets 5 keeps the C-erasure phase tiny so we get to E fast.
# Carrier loop iterates lazily over layers 10-25, loads transcoders one at a time,
# computes max|cos(d_f, u^L)| plain + causal vs random-max null.
python -u scripts/88_usage_subspace.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --k_grid 5 \
    --max_targets 5 \
    --cum_windows \
    --carrier \
    --carrier_layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --out_dir data/analysis/runD_v2/usage_subspace_e

echo ""
echo "==== Done: $(date) ===="
