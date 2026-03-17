#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=mc_v5_analysis
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mc_v5_analysis-%j.out
#SBATCH --error=logs/mc_v5_analysis-%j.err
#
# multilingual_circuits v5 — concept-paired IoU + decision-only node labels (Phase 4)
#
# Runs ONLY the analysis script (no model / no GPU needed).
# Uses existing features from SLURM 25058380 (v2, last_5, 240 samples).
#
# New analyses:
#   --concept_paired          : per-concept EN vs FR IoU averaged over 8 concepts
#                               Removes cross-concept coincidental overlaps that inflate
#                               early-layer all-vs-all IoU and flatten the Claim 3 gradient.
#                               Outputs: iou_per_layer_concept_paired_decision.csv
#                                        iou_per_layer_concept_paired_pooled.csv  (multi-token)
#   --decision_only_labels    : re-runs Phase 3.1 with only is_decision_position=True rows
#                               Confirms that late-layer fr_leaning features are genuinely
#                               active at the output-prediction step, not just at content positions.
#                               Outputs: node_language_labels_decision.csv
#   --graph_json ...roleaware : uses the v4 role-aware graph for node label computation
#
# Success criterion:
#   concept-paired decision middle/early ratio > 1.60× → Claim 3 "Moderately supported"
#   concept-paired decision middle/early ratio > 2.00× → Claim 3 "Strongly supported"
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_v5_analysis.sbatch
#
# AFTER COMPLETION:
#   Download logs/mc_v5_analysis-<JOBID>.out
#   Download data/analysis/multilingual_circuits/iou_per_layer_concept_paired_decision.csv
#   Download data/analysis/multilingual_circuits/iou_per_layer_concept_paired_pooled.csv
#   Download data/analysis/multilingual_circuits/node_language_labels_decision.csv
#   Check middle/early ratio in "Claim 3 comparison" section of log
#   Update data/runs/RUN_INDEX.md with v5 entry

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits v5 — Concept-Paired IoU + Decision-Only Labels"
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

echo "Python: $(which python)"
python --version
echo ""

BEHAVIOUR=multilingual_circuits
SPLIT=train
GRAPH_JSON="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n48_roleaware.json"

# Verify features are present (v2 run must be complete)
FEAT_PROBE="data/results/transcoder_features/layer_10/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
if [ ! -f "$FEAT_PROBE" ]; then
    echo "ERROR: Feature file not found: $FEAT_PROBE"
    echo "  Run multilingual_circuits_v2_last5_02_09.sbatch first (SLURM 25058380)."
    exit 1
fi
echo "Feature probe: $FEAT_PROBE (OK)"

# Verify role-aware graph is present (v4 must be complete)
if [ ! -f "$GRAPH_JSON" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_JSON"
    echo "  Run multilingual_circuits_v4_roleaware_06.sbatch first (SLURM 25266717)."
    exit 1
fi
echo "Graph JSON: $GRAPH_JSON (OK)"
echo ""

# ══════════════════════════════════════════════════════════════
# Analysis: concept-paired IoU + decision-only labels
# ══════════════════════════════════════════════════════════════
echo "==== Analysis: concept-paired IoU + decision-only labels ===="
date

python -u scripts/a_analyze_multilingual_circuits.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --concept_paired \
    --decision_only_labels \
    --graph_json "$GRAPH_JSON"

echo ""
echo "========================================"
echo "Analysis complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Key output files:"
echo "  data/analysis/${BEHAVIOUR}/iou_per_layer_concept_paired_decision.csv"
echo "  data/analysis/${BEHAVIOUR}/iou_per_layer_concept_paired_pooled.csv   (if multi-token)"
echo "  data/analysis/${BEHAVIOUR}/node_language_labels_decision.csv"
echo ""
echo "Check 'Claim 3 comparison' in the log above for the middle/early ratio."
