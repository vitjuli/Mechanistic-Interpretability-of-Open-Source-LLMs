#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_patch
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/phys_dt_patch-%j.out
#SBATCH --error=logs/phys_dt_patch-%j.out
#
# physics_decay_type — patching experiment (step 07)
#
# Patching: take the TOP-K circuit features active on alpha prompts and
# patch them into beta prompts (and vice versa). Tests whether the circuit
# is causally sufficient to transfer the decision across concept.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   # Ensure attribution graph exists:
#   ls data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json
#   sbatch jobs/physics_decay_type_patching.sbatch

set -uo pipefail

BEHAVIOUR=physics_decay_type
SPLIT=train
N_PROMPTS=108
GRAPH_SUFFIX=_roleaware_static
TOP_K=20
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type — Patching Experiment"
echo "  GRAPH_SUFFIX=${GRAPH_SUFFIX}  TOP_K=${TOP_K}"
echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo ""
echo "Python: $(which python)"
python --version
nvidia-smi || true
echo ""

# ── Patching: alpha→beta and beta→alpha ──────────────────────────────────────
# Source = prompts of concept X, target = prompts of concept Y.
# We run patching with the full train file: the script internally splits
# by correct_answer to form source/target pairs.

echo "==== Step 07: Patching ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K" \
    --per_feature

echo ""
echo "==== Patching complete: $(date) ===="
echo ""
echo "Output:"
echo "  data/results/interventions/${BEHAVIOUR}/intervention_patching_${BEHAVIOUR}_${SPLIT}.csv"
echo "  data/results/interventions/${BEHAVIOUR}/intervention_patching_${BEHAVIOUR}_${SPLIT}_summary.json"
echo ""
echo "After rsync, re-run UI prep to include patching data:"
echo "  python scripts/09_prepare_offline_ui.py \\"
echo "    --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "    --graph_n_prompts ${N_PROMPTS} --graph_suffix ${GRAPH_SUFFIX}"
