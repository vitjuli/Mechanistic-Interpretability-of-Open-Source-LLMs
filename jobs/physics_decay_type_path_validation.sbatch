#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_pathval
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=logs/phys_dt_pathval-%j.out
#SBATCH --error=logs/phys_dt_pathval-%j.out
#
# physics_decay_type — path-centric causal validation (script 14)
#
# Validates the top causal paths from the extracted circuit:
#   input → L22_F110496 → L23_F83556 → L24_F60777 → L25_F71226 → output_correct
#
# For each path (A→B hop):
#   - Ablates feature A at layer_A
#   - Measures change in feature B activation at layer_B
#   - Reports propagation_consistency: fraction of prompts where |Δact_B| > epsilon
#
# Expected: top path should show strong propagation consistency (>0.70),
# consistent with the b1 result (0.820).
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   # Circuit JSON must exist:
#   ls data/results/causal_edges/physics_decay_type/circuits_physics_decay_type_train.json
#   sbatch jobs/physics_decay_type_path_validation.sbatch

set -uo pipefail

BEHAVIOUR=physics_decay_type
SPLIT=train
N_PATHS=10
MAX_PROMPTS=108
EPSILON=0.01
CIRCUITS_JSON=data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type — Path Validation (script 14)"
echo "  n_paths=${N_PATHS}  max_prompts=${MAX_PROMPTS}  epsilon=${EPSILON}"
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

if [ ! -f "$CIRCUITS_JSON" ]; then
    echo "FATAL: circuits JSON not found: $CIRCUITS_JSON"
    echo "       Run step 08 first (circuit extraction)."
    exit 1
fi
echo "Circuits JSON: $CIRCUITS_JSON"
echo ""

echo "==== Script 14: Path Validation ===="
date
python -u scripts/14_path_validation.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --circuits_json "$CIRCUITS_JSON" \
    --n_paths "$N_PATHS" \
    --max_prompts "$MAX_PROMPTS" \
    --epsilon "$EPSILON" \
    --model_size 4b

echo ""
echo "==== Path Validation complete: $(date) ===="
echo ""
echo "Output: data/results/path_validation/${BEHAVIOUR}/path_validation_${BEHAVIOUR}_${SPLIT}.json"
echo ""

# Print summary
python3 -c "
import json
from pathlib import Path
p = Path('data/results/path_validation/${BEHAVIOUR}/path_validation_${BEHAVIOUR}_${SPLIT}.json')
if p.exists():
    d = json.load(open(p))
    summary = d.get('summary', {})
    print('=== Path Validation Summary ===')
    print(f'  mean_delta_act_B         : {summary.get(\"mean_delta_act_B\", \"?\"):.4f}')
    print(f'  mean_delta_logit         : {summary.get(\"mean_delta_logit\", \"?\"):.4f}')
    print(f'  propagation_consistency  : {summary.get(\"propagation_consistency\", \"?\"):.4f}')
    paths = d.get('paths', [])
    print(f'  n_paths_validated        : {len(paths)}')
    for i, path_res in enumerate(paths[:3]):
        print(f'  Path {i+1}: {path_res.get(\"path\", [])} | consistency={path_res.get(\"propagation_consistency\",\"?\"):.3f}')
" || true
