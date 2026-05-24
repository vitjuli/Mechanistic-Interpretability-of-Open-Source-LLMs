#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=grad-sens
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/grad-sens-%j.out
#SBATCH --error=logs/grad-sens-%j.out
#
# Script 53: Gradient sensitivity analysis
# Projects reconstruction error onto ∇_x logit_diff.
# One backward pass per prompt covers all 16 layers simultaneously.
#
# Independent of Run C ablation — can run in parallel with runC_p2.
# Runtime estimate: ~3-4h (538 prompts × 1 backward + 16 patch passes)

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Script 53: Gradient Sensitivity Analysis"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
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
python --version
nvidia-smi || true
echo ""

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
OUTPUT_DIR="data/analysis/gradient_sensitivity"

PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1; }
echo "Prompts: $PROMPT_FILE"
echo ""

python -u scripts/53_gradient_sensitivity_analysis.py \
    --behaviour   "$BEHAVIOUR" \
    --split       "$SPLIT" \
    --layers      $LAYERS \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
