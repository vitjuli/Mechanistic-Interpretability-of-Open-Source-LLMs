#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j81b-fam
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/j81b-%j.out
#SBATCH --error=logs/j81b-%j.out
#
# JOB B — family-driven agglomerative growth (top-2..top-5 per cue family)
#         + random-union null on the specificity metric.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/union_cue

echo "========================================================"
echo "  81b — family mode (Job B)"
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

# Full family-driven run with random-union null (reduced n_random_union=20
# to keep runtime <10h; family k_max=5)
python -u scripts/81_cluster_union_cue_specificity.py \
    --mode family \
    --k_max 5 \
    --n_random_union 20 \
    --behaviour physics_decay_type_probe_v2 --split train \
    --out_dir data/analysis/runD_v2/union_cue

echo ""
echo "==== Done: $(date) ===="
