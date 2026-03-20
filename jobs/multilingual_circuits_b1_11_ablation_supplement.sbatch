#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_supp
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mc_b1_supp-%j.out
#SBATCH --error=logs/mc_b1_supp-%j.err
#
# multilingual_circuits_b1 — script 11: ablation supplement
#
# Runs ablation_zero for the 17 circuit features that script 07 did not
# cover (they were not in the star-graph top-k used by that run).
# This closes the late-layer hub blindspot in the reasoning-trace analysis.
#
# Specifically measures:
#   L10_F141643, L11_F56459             (early, not in original ablation)
#   L22_F108295, L22_F32734             (mid gateway to L23 hub)
#   L23_F56267,  L23_F64429             (L23 hub — KEY)
#   L24_F131457, L24_F134204, L24_F136810,
#   L24_F29680,  L24_F48363,  L24_F76363, L24_F81698  (L24 outputs)
#   L25_F111603, L25_F138698, L25_F34754, L25_F59053  (L25 terminals)
#
# Method: identical ablation_zero to script 07 (encode → zero feature → decode → patch → forward)
# Output: data/results/reasoning_traces/multilingual_circuits_b1/ablation_supplement_train.csv
#
# Estimated forward passes: 17 × 96 + 96 = 1,728
# Estimated wall time: ~15 min (45-min slot = safe margin)
#
# AFTER COMPLETION, rerun script 10 with --supplement_csv to get updated traces:
#   sbatch jobs/multilingual_circuits_b1_10_reasoning_trace_v2.sbatch
#
# BEFORE SUBMITTING:
#   Verify circuit file:
#     ls -lh data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
#   Then:
#     cd /rds/user/iv294/hpc-work/thesis/project
#     git pull
#     mkdir -p logs
#     sbatch jobs/multilingual_circuits_b1_11_ablation_supplement.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_b1_supp-<JOBID>.out for:
#     "Ablation-zero combined:      27/27"
#     "L23_F64429: measured [SUPPLEMENT (NEW)]"
#     "L24_F136810: measured [SUPPLEMENT (NEW)]"
#     "Saved: data/results/reasoning_traces/multilingual_circuits_b1/ablation_supplement_train.csv"
#   Then submit the v2 reasoning trace job (script 10 + --supplement_csv).

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Ablation Supplement (Script 11)"
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
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train

# Verify circuit file
CIRCUITS_PATH="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"
if [ ! -f "$CIRCUITS_PATH" ]; then
    echo "ERROR: Circuit file not found: $CIRCUITS_PATH"
    exit 1
fi
echo "Circuit: $CIRCUITS_PATH (OK)"

# ══════════════════════════════════════════════════════════════
# Script 11: ablation supplement
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Script 11: Ablation Supplement ===="
echo "  --behaviour ${BEHAVIOUR}"
echo "  --split ${SPLIT}"
echo "  --model_size 4b"
date

python -u scripts/11_ablation_supplement.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b

echo ""
echo "========================================"
echo "Script 11 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Output:"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/ablation_supplement_${SPLIT}.csv"
echo ""
echo "Next step — rerun reasoning trace with supplement:"
echo "  sbatch jobs/multilingual_circuits_b1_10_reasoning_trace_v2.sbatch"
