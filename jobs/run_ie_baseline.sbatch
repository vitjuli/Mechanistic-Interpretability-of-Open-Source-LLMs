#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-baseline
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ie-baseline-%j.out
#SBATCH --error=logs/ie-baseline-%j.out

# sbatch jobs/run_ie_baseline.sbatch
# PURPOSE: Baseline gate for physics_intensive_extensive_v1.
# GATE: sign_accuracy >= 0.85, mean_norm_diff >= 0.3
# If sign_accuracy < 0.65 on BASE model, the prompt format needs revision —
# regenerate with Yes/No format: python scripts/63_generate_ie_prompts.py --yes_no

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

echo "========================================================"
echo "  IE Baseline — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -4 || true

# Generate prompts if needed
python -u scripts/63_generate_ie_prompts.py

# Run baseline (uses Instruct model for behavioural evaluation)
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split train

echo ""
echo "GATE: check sign_accuracy >= 0.85 in the output above"
echo "If sign_accuracy < 0.65, prompt format needs ' Yes'/' No' fallback."
echo "Exit: $?  $(date)"
echo "========================================================"
