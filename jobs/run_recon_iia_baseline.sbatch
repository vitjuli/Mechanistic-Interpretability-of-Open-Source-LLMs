#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=recon-baseline
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/recon-baseline-%j.out
#SBATCH --error=logs/recon-baseline-%j.out
#
# Script 51: Transcoder reconstruction noise floor
# Measures Δ_recon = enc→dec no-op intervention vs clean baseline.
# Expected: sign_flip_rate ≈ 0, mean|Δ_recon| ≪ mean|Δ_ablation|
#
# Independent of Run C ablation — can run in parallel with runC_p2.
# Runtime estimate: ~2-3h (538 prompts × 16 layers × 2 passes each)

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Script 51: Reconstruction IIA Baseline"
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
OUTPUT_DIR="data/analysis/recon_baseline"

PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1; }
echo "Prompts: $PROMPT_FILE"
echo ""

python -u scripts/51_recon_iia_baseline.py \
    --behaviour   "$BEHAVIOUR" \
    --split       "$SPLIT" \
    --layers      $LAYERS \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
