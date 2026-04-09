#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=mc_b1_trace
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mc_b1_trace-%j.out
#SBATCH --error=logs/mc_b1_trace-%j.err
#
# multilingual_circuits_b1 — script 10: reasoning trace engine
#
# CPU-only. No model needed. Reads from existing circuit artifacts:
#   data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
#   data/results/causal_edges/multilingual_circuits_b1/causal_edges_multilingual_circuits_b1_train.json
#   data/ui_offline/20260318-013309_multilingual_circuits_b1_train_n96/interventions.csv
#   data/prompts/multilingual_circuits_b1_train.jsonl
#
# Estimated runtime: ~5 min (pandas + matplotlib only)
#
# BEFORE SUBMITTING:
#   Verify prerequisite files exist:
#     ls -lh data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
#     ls -lh data/ui_offline/20260318-013309_multilingual_circuits_b1_train_n96/interventions.csv
#   Then:
#     cd /rds/user/iv294/hpc-work/thesis/project
#     git pull
#     mkdir -p logs
#     sbatch jobs/multilingual_circuits_b1_10_reasoning_trace.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_b1_trace-<JOBID>.out for:
#     "Correct predictions: 78/96 (81.2%)"
#     "Ablation-measured circuit features: 10/27"
#     "Zone early: mean ablation contrib = +X ± Y"
#     "Trajectory prediction accuracy: Z"
#     "Done."
#   Inspect outputs:
#     ls -lh data/results/reasoning_traces/multilingual_circuits_b1/
#     python -c "
#       import json
#       traces = [json.loads(l) for l in open('data/results/reasoning_traces/multilingual_circuits_b1/reasoning_traces_train.jsonl')]
#       print('traces:', len(traces))
#       err = json.load(open('data/results/reasoning_traces/multilingual_circuits_b1/error_cases_train.json'))
#       print('error patterns:', err['error_patterns'])
#     "
#   Sync outputs:
#     git add data/results/reasoning_traces/multilingual_circuits_b1/
#     git commit -m "script 10: reasoning traces for b1 train (96 prompts)"
#

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Reasoning Trace (Script 10)"
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

# Verify prerequisite files
CIRCUITS_PATH="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"
CAUSAL_PATH="data/results/causal_edges/${BEHAVIOUR}/causal_edges_${BEHAVIOUR}_${SPLIT}.json"
UI_DIR="data/ui_offline/20260318-013309_${BEHAVIOUR}_${SPLIT}_n96"

for f in "$CIRCUITS_PATH" "$CAUSAL_PATH" "${UI_DIR}/interventions.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
    echo "OK: $f"
done

# ══════════════════════════════════════════════════════════════
# Script 10: reasoning trace engine
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Script 10: Reasoning Trace Engine ===="
echo "  --behaviour ${BEHAVIOUR}"
echo "  --split ${SPLIT}"
echo "  --run_id 20260318-013309_${BEHAVIOUR}_${SPLIT}_n96"
echo "  --top_k_paths 5"
date

python -u scripts/10_reasoning_trace.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --run_id "20260318-013309_${BEHAVIOUR}_${SPLIT}_n96" \
    --top_k_paths 5

echo ""
echo "========================================"
echo "Script 10 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Outputs:"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/reasoning_traces_${SPLIT}.jsonl"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/prompt_features_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/prompt_paths_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/layerwise_decision_trace_${SPLIT}.parquet"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/error_cases_${SPLIT}.json"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/figures/"
