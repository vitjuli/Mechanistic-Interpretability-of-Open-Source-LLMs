#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j110-span
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j110-%j.out
#SBATCH --error=logs/j110-%j.out
#
# 110 — Active span triad. For each layer ℓ: from the active features
# (act > 0 on ≥ 5% of prompts), how much of u^(ℓ) and w_res^(ℓ) can be
# spanned by greedy weighted combinations of their decoders W_dec[f]?
# Tests whether the model's used and readable axes are actually within the
# dictionary's reach. Forced regime. Needs transcoders. ~7-10 min real.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/active_span

echo "========================================================"
echo "  110 — active-span triad (u, w_res reachable by dict?)"
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
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

python -u scripts/110_active_span_triad.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/active_span

echo ""
echo "==== Done: $(date) ===="
