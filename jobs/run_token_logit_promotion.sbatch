#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=tok-logit-promo
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tok-logit-promo-%j.out
#SBATCH --error=logs/tok-logit-promo-%j.out

# sbatch jobs/run_token_logit_promotion.sbatch

set -euo pipefail
BEHAVIOUR=physics_internal_candidate_selection_v2

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

echo "========================================================"
echo "  Token-logit competitor promotion — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -4 || true

python -u scripts/50_token_logit_competitor_promotion.py \
    --behaviour "$BEHAVIOUR" \
    --n_prompts 100 \
    --k 6 \
    --device cuda

echo "Exit: $?  $(date)"

ADIR="data/results/internal_candidate_analysis/${BEHAVIOUR}"
echo ""
echo "  git add -f ${ADIR}/token_logit_promotion_*"
echo "  git commit -m 'add token logit promotion results'"
echo "  git push"
echo "========================================================"
