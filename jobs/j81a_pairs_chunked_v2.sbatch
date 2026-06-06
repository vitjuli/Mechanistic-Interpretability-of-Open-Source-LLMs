#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j81a-v2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j81a-v2-%j.out
#SBATCH --error=logs/j81a-v2-%j.out
#
# JOB A v2 — pairs mode, 10 chunks of ~44 pairs each (~13h per chunk).
# Incremental CSV save: every pair is checkpointed, so timeouts lose only the
# current pair-in-progress.
#
# Submit 10 chunks (parallel; CSD3 will queue them based on quota):
#
#   sbatch --export=PAIR_START=0,PAIR_END=44      jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=44,PAIR_END=88     jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=88,PAIR_END=132    jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=132,PAIR_END=176   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=176,PAIR_END=220   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=220,PAIR_END=264   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=264,PAIR_END=308   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=308,PAIR_END=352   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=352,PAIR_END=396   jobs/j81a_pairs_chunked_v2.sbatch
#   sbatch --export=PAIR_START=396,PAIR_END=435   jobs/j81a_pairs_chunked_v2.sbatch
#
# Each chunk produces its own CSV:
#   data/analysis/runD_v2/union_cue/chunk_v2_<start>_<end>/union_cue_pairs.csv

set -uo pipefail
mkdir -p logs

: ${PAIR_START:?must set --export=PAIR_START=N}
: ${PAIR_END:?must set --export=PAIR_END=N}

OUT_DIR="data/analysis/runD_v2/union_cue/chunk_v2_${PAIR_START}_${PAIR_END}"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  81a v2 — pairs $PAIR_START..$PAIR_END (incremental save)"
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
    --mode pairs \
    --pair_start "$PAIR_START" --pair_end "$PAIR_END" \
    --behaviour physics_decay_type_probe_v2 --split train \
    --out_dir "$OUT_DIR"

echo ""
echo "==== Done: $(date) ===="
