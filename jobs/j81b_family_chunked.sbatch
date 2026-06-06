#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j81b-chunk
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j81b-chunk-%j.out
#SBATCH --error=logs/j81b-chunk-%j.out
#
# JOB B — family mode chunked. 6 families, submit one per family:
#
#   sbatch --export=FAMILY=charge_Z   jobs/j81b_family_chunked.sbatch
#   sbatch --export=FAMILY=mass_A     jobs/j81b_family_chunked.sbatch
#   sbatch --export=FAMILY=quark_weak jobs/j81b_family_chunked.sbatch
#   sbatch --export=FAMILY=lepton     jobs/j81b_family_chunked.sbatch
#   sbatch --export=FAMILY=energy     jobs/j81b_family_chunked.sbatch
#   sbatch --export=FAMILY=emission   jobs/j81b_family_chunked.sbatch
#
# Reduced workload: n_random_union=10 (was 20); k_max=4 (was 5).
# From prior run: charge_Z k=2 took ~7h, k=3 took ~7h more. With k_max=4 and
# 10 null draws, one family ≈ 8-12 hours. Fits in 24h with margin.

set -uo pipefail
mkdir -p logs

: ${FAMILY:?must set --export=FAMILY=name}

OUT_DIR="data/analysis/runD_v2/union_cue/family_${FAMILY}"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  81b chunked — family $FAMILY"
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

python -u scripts/81_cluster_union_cue_specificity.py \
    --mode family \
    --families "$FAMILY" \
    --k_max 4 \
    --n_random_union 10 \
    --behaviour physics_decay_type_probe_v2 --split train \
    --out_dir "$OUT_DIR"

echo ""
echo "==== Done: $(date) ===="
