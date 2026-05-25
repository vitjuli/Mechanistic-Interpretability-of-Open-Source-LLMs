#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runD-s04
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/probe-runD-s04-%j.out
#SBATCH --error=logs/probe-runD-s04-%j.out
#
# runD Step 1: Extract transcoder features for physics_decay_type_probe_v2
#
# Runs script 04 on all 538 probe_v2 train prompts.
# Saves top-50 features per layer per prompt to:
#   data/results/transcoder_features/layer_*/physics_decay_type_probe_v2_train_top_k_*.npy
#
# This is the prerequisite for script 06 (graph building from probe_v2 prompts).
# Previously the graph used physics_decay_type (108 prompts) — runD fixes this.
#
# Runtime estimate: ~3-4h (538 prompts × forward+gradient pass per layer)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/run_probe_runD_s04.sbatch

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD Step 04"
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

# Verify prompts exist
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: Prompts not found: $PROMPT_FILE"; exit 1; }
N_PROMPTS=$(wc -l < "$PROMPT_FILE")
echo "Prompts: $PROMPT_FILE  ($N_PROMPTS prompts)"
echo ""

echo "==== [Step 04] Extract transcoder features ===="
date

python -u scripts/04_extract_transcoder_features.py \
    --behaviour  "$BEHAVIOUR" \
    --split      "$SPLIT" \
    --model_size 4b \
    --layers     $LAYERS \
    --top_k      50 \
    --batch_size 4

echo ""
echo "==== Step 04 done: $(date) ===="
echo ""

# Verify output
for layer in 10 18 24; do
    f="data/results/transcoder_features/layer_${layer}/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "========================================================"
echo "  Done: $(date)  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Next: sbatch jobs/run_probe_runD_s06.sbatch"
