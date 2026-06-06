#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j81a-chunk
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j81a-chunk-%j.out
#SBATCH --error=logs/j81a-chunk-%j.out
#
# JOB A — pairs mode chunked. Submit 5 copies, each does 87 pairs:
#
#   sbatch --export=PAIR_START=0,PAIR_END=87    jobs/j81a_pairs_chunked.sbatch
#   sbatch --export=PAIR_START=87,PAIR_END=174  jobs/j81a_pairs_chunked.sbatch
#   sbatch --export=PAIR_START=174,PAIR_END=261 jobs/j81a_pairs_chunked.sbatch
#   sbatch --export=PAIR_START=261,PAIR_END=348 jobs/j81a_pairs_chunked.sbatch
#   sbatch --export=PAIR_START=348,PAIR_END=435 jobs/j81a_pairs_chunked.sbatch
#
# Each chunk produces union_cue_pairs.csv in its own subdirectory:
#   data/analysis/runD_v2/union_cue/chunk_<start>_<end>/union_cue_pairs.csv
#
# After all 5 finish, merge with:
#   cat data/analysis/runD_v2/union_cue/chunk_*/union_cue_pairs.csv | \
#     awk 'NR==1 || !/^combo,/' > data/analysis/runD_v2/union_cue/union_cue_pairs.csv

set -uo pipefail
mkdir -p logs

: ${PAIR_START:?must set --export=PAIR_START=N}
: ${PAIR_END:?must set --export=PAIR_END=N}

OUT_DIR="data/analysis/runD_v2/union_cue/chunk_${PAIR_START}_${PAIR_END}"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  81a chunked — pairs $PAIR_START..$PAIR_END"
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
