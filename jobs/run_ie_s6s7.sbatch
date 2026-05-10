#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-s6s7
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ie-s6s7-%j.out
#SBATCH --error=logs/ie-s6s7-%j.out

# sbatch jobs/run_ie_s6s7.sbatch
# Runs S6 (cluster analysis + cross-domain transfer) and S7 (cluster interventions)
# for physics_intensive_extensive_v1, after S0-S5 already completed.

set -euo pipefail
BEHAVIOUR=physics_intensive_extensive_v1

cd /rds/user/iv294/hpc-work/thesis/project
module purge
module load rhel8/default-amp 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TSTART=$(date +%s)
stamp() { echo "  [$(( ($(date +%s) - TSTART) / 60 ))m elapsed]  $(date)"; }

echo "========================================================"
echo "  IE S6+S7 — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -6 || true
echo ""

echo "── S6: Cluster analysis + cross-domain transfer ──"
python -u scripts/71_ie_cluster_analysis.py \
    --behaviour  "$BEHAVIOUR" \
    --device     cuda \
    --k          6
stamp

echo ""
echo "── S7: Cluster interventions ─────────────────"
python -u scripts/72_ie_cluster_interventions.py \
    --behaviour  "$BEHAVIOUR" \
    --device     cuda \
    --k          6 \
    --n_prompts  100
stamp

ELAPSED=$(( ($(date +%s) - TSTART) / 60 ))
echo ""
echo "========================================================"
echo "  S6+S7 COMPLETE — ${ELAPSED}m total — $(date)"
echo "  Results: data/results/abstraction_ie/${BEHAVIOUR}/"
echo "========================================================"
