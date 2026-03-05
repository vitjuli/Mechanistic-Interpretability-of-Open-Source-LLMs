#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ml_C1C2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ml_C1C2-%j.out
#SBATCH --error=logs/ml_C1C2-%j.err
#
# Re-runs patching C1 and C2 ONLY (C3 results already exist from job 24194284).
# Needed because all three patch_mode runs previously wrote to the same output file;
# C1 (25 pairs) and C2 (3 pairs) data was overwritten by C3 (27 pairs).
#
# Fix applied: script 07 now writes to intervention_patching_C1_{beh}.csv etc.
# prepare.py now discovers all patching variant CSVs automatically via glob.
#
# BEFORE SUBMITTING:
#   git pull    # get the script 07 + prepare.py fix
#   sbatch jobs/multilingual_rerun_C1C2.sbatch

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Antonym: Re-run C1 + C2 patching"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

BEHAVIOUR=multilingual_antonym
SPLIT=train
N_PROMPTS=80
TOP_K=20
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

echo ""
echo "==== Patching C1 — operation swap (antonym→synonym) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C1 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

echo ""
echo "==== Patching C2 — operand swap (hot→small) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C2 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

echo ""
echo "==== Re-run step 09 to incorporate all three patching CSVs ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --community_method louvain

RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | tail -1)
if [ -n "$RUN_DIR" ]; then
    python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
fi

echo ""
echo "========================================"
echo "C1+C2 re-run complete. Job: $SLURM_JOB_ID"
date
