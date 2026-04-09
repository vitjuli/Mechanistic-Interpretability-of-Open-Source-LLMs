#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=mc_b1_trace_v2
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mc_b1_trace_v2-%j.out
#SBATCH --error=logs/mc_b1_trace_v2-%j.err
#
# multilingual_circuits_b1 — script 10 v2: reasoning trace with late-hub supplement
#
# Runs the full reasoning-trace engine with the ablation supplement from script 11.
# Coverage upgrade: 10/27 → 27/27 circuit features with ablation_zero measurements.
#
# MUST run AFTER script 11 (mc_b1_supp) completes.
#
# Key diagnostics to check in output:
#   "Ablation-zero covered:  27/27 [FULL COVERAGE]"
#   "L23_F64429: ✓ ablation_zero"
#   "L24_F136810: ✓ ablation_zero"
#   "Type A (trajectory=correct, outcome wrong): N  ← should be REDUCED vs v1"
#   "Trajectory prediction accuracy: X  ← should be IMPROVED vs 0.6979"
#
# BEFORE SUBMITTING:
#   Verify supplement file exists:
#     ls -lh data/results/reasoning_traces/multilingual_circuits_b1/ablation_supplement_train.csv
#   Then:
#     cd /rds/user/iv294/hpc-work/thesis/project
#     git pull
#     mkdir -p logs
#     sbatch jobs/multilingual_circuits_b1_10_reasoning_trace_v2.sbatch
#
# AFTER COMPLETION:
#   Sync outputs:
#     git add data/results/reasoning_traces/multilingual_circuits_b1/
#     git commit -m "script 10 v2: reasoning traces with late-hub supplement (27/27 coverage)"

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Reasoning Trace v2 (Script 10 + supplement)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

module purge
module load rhel8/default-cpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
python --version
echo

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train
SUPPLEMENT="data/results/reasoning_traces/${BEHAVIOUR}/ablation_supplement_${SPLIT}.csv"

# Verify supplement exists
if [ ! -f "$SUPPLEMENT" ]; then
    echo "ERROR: Supplement CSV not found: $SUPPLEMENT"
    echo "  Run multilingual_circuits_b1_11_ablation_supplement.sbatch first."
    exit 1
fi
echo "Supplement: $SUPPLEMENT (OK)"
echo "  Rows: $(wc -l < "$SUPPLEMENT")"

# ══════════════════════════════════════════════════════════════
# Script 10 v2: reasoning trace with late-hub supplement
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Script 10 v2: Reasoning Trace with Supplement ===="
echo "  --behaviour ${BEHAVIOUR}"
echo "  --split ${SPLIT}"
echo "  --supplement_csv ${SUPPLEMENT}"
echo "  --top_k_paths 5"
date

python -u scripts/10_reasoning_trace.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --run_id "20260318-013309_${BEHAVIOUR}_${SPLIT}_n96" \
    --supplement_csv "$SUPPLEMENT" \
    --top_k_paths 5

echo ""
echo "========================================"
echo "Script 10 v2 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Outputs (updated in-place):"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/reasoning_traces_${SPLIT}.jsonl"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/prompt_features_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/prompt_paths_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/layerwise_decision_trace_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/error_cases_${SPLIT}.json"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/figures/"
