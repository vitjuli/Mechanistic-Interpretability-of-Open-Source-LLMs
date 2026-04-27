#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_steer
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/phys_dt_steer-%j.out
#SBATCH --error=logs/phys_dt_steer-%j.out
#
# physics_decay_type — steering experiment (step 07)
#
# Steering: artificially activate (add) the top circuit features on prompts
# where they are normally absent or weak, then measure whether this drives
# the model toward the corresponding answer.
#
# Key question: can we steer the model from beta→alpha (or vice versa) by
# amplifying circuit features?  Coefficient 10.0 is the default; adjust if
# effects are too small (try 20.0) or cause gibberish output (try 5.0).
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   ls data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json
#   sbatch jobs/physics_decay_type_steering.sbatch

set -uo pipefail

BEHAVIOUR=physics_decay_type
SPLIT=train
N_PROMPTS=108
GRAPH_SUFFIX=_roleaware_static
TOP_K=20
STEERING_COEFF=10.0
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type — Steering Experiment"
echo "  coeff=${STEERING_COEFF}  TOP_K=${TOP_K}"
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
python --version
nvidia-smi || true
echo ""

echo "==== Step 07: Steering (coeff=${STEERING_COEFF}) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment steering \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K" \
    --steering_coeff "$STEERING_COEFF" \
    --per_feature

echo ""

# If coefficient was too weak or too strong, re-run with adjusted value.
# Check log for: mean_sign_flip_rate (want > 0.10) and n_incoherent (want < 10%)
echo "==== Steering complete: $(date) ===="
echo ""
echo "Output:"
echo "  data/results/interventions/${BEHAVIOUR}/intervention_steering_${BEHAVIOUR}_${SPLIT}.csv"
echo ""
echo "Key diagnostic — sign flip rate from steering summary:"
python3 -c "
import json, sys
from pathlib import Path
p = Path('data/results/interventions/${BEHAVIOUR}/intervention_steering_${BEHAVIOUR}_${SPLIT}_summary.json')
if p.exists():
    d = json.load(open(p))
    print(f'  sign_flip_rate: {d.get(\"sign_flip_rate\",\"?\"):.3f}')
    print(f'  mean_abs_effect: {d.get(\"mean_abs_effect_size\",\"?\"):.3f}')
else:
    print('  Summary not found')
" || true
echo ""
echo "If sign_flip_rate < 0.10: increase STEERING_COEFF to 20.0 and resubmit."
echo "If sign_flip_rate > 0.50: decrease STEERING_COEFF to 5.0 (too aggressive)."
