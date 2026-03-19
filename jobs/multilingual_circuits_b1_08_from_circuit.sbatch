#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_revalidate
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mc_b1_revalidate-%j.out
#SBATCH --error=logs/mc_b1_revalidate-%j.err
#
# multilingual_circuits_b1 — script 08: --from_circuit revalidation
#
# Loads the existing circuits JSON (27-feature causal-path circuit) and reruns
# ONLY Phases 7-9 (necessity + S1/S1.5 + S2) for those features.
# Skips the 2h ablation (Phases 1-5) and path tracing (Phase 6).
# Also rewrites the presentation graph (Phase 10) in-place.
#
# Phases run:
#   Phase 0   Load roleaware attribution graph (graph_layers, I/O nodes)
#   Phase 2   Load model + transcoders (layers 10-25)
#   Phase 3   Baseline cache: 96 forward passes
#   Phase 7   Necessity (group ablation): ~96 passes
#   Phase 8   Sufficiency S1 (linear) + S1.5 (layerwise): 3 × 96 = 288 passes
#   Phase 9   Sufficiency S2 (cross-prompt injection): ~8 passes
#   Phase 10  Rewrite presentation graph (Graph B)
#
# Estimated forward passes: ~480  (vs ~7,300 for full run)
# Timing (from SLURM 25475632): Phase3=10min, Phase7=28min, Phase8=52min, Phase9=<5min
# Estimated wall time: ~100 min (3h slot = safe margin)
#
# BEFORE SUBMITTING:
#   Verify the circuit file exists:
#     ls -lh data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
#   Verify it has 27 features:
#     python -c "
#       import json
#       d = json.load(open('data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json'))
#       print('features:', d['circuit']['n_features'])
#       print('edges:', d['circuit']['n_edges'])
#       print('paths:', d['circuit']['n_paths'])
#     "
#   Then:
#      cd /rds/user/iv294/hpc-work/thesis/project
#      git pull
#      mkdir -p logs
#      sbatch jobs/multilingual_circuits_b1_08_from_circuit.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_b1_revalidate-<JOBID>.out for:
#     "Circuit (revalidated): 27 features, 69 edges (50 paths)"
#     "Phase 7 necessity:    disruption_rate=X"
#     "Phase 8 S1  linear:   sign=X, retention=Y, verdict=Z"
#     "Phase 8 S1.5 layerwise: sign=X, retention=Y, verdict=Z, improvement=D"
#     "Phase 9 S2 injection:   n_pairs=X, transfer_rate=Y, mean_shift=Z"
#   Circuits JSON is updated IN-PLACE (paths/circuit preserved, validation overwritten).
#   Quick inspect:
#     python -c "
#       import json
#       d = json.load(open('data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json'))
#       print('necessity:', d['validation'])
#       print('S1.5:', d.get('sufficiency_s1_5'))
#       print('S2:', d.get('sufficiency_s2'))
#     "
#   Sync outputs:
#     git add data/results/causal_edges/multilingual_circuits_b1/
#     git commit -m "script 08: revalidate 27-feature circuit (necessity + S1/S1.5/S2)"

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Revalidate Circuit (Script 08 --from_circuit)"
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
N_PROMPTS=96
GRAPH_SUFFIX=_roleaware
CIRCUITS_PATH="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"
GRAPH_PATH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"

# Verify prerequisite files exist
if [ ! -f "$CIRCUITS_PATH" ]; then
    echo "ERROR: Circuits JSON not found: $CIRCUITS_PATH"
    echo "  Run multilingual_circuits_b1_08_causal_edges.sbatch first (or --from_causal_edges rerun)."
    exit 1
fi
echo "Circuits: $CIRCUITS_PATH (OK)"
python -c "
import json
d = json.load(open('${CIRCUITS_PATH}'))
print(f'  {d[\"circuit\"][\"n_features\"]} features, {d[\"circuit\"][\"n_edges\"]} edges, {d[\"circuit\"][\"n_paths\"]} paths')
"

if [ ! -f "$GRAPH_PATH" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_PATH"
    echo "  Run multilingual_circuits_b1_02_09.sbatch first."
    exit 1
fi
echo "Graph: $GRAPH_PATH (OK)"

# ══════════════════════════════════════════════════════════════
# Step 08 --from_circuit: rerun Phases 7-9 only
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 08 --from_circuit: necessity + S1/S1.5/S2 sufficiency ===="
echo "  --behaviour ${BEHAVIOUR}"
echo "  --split ${SPLIT}"
echo "  --graph_json ${GRAPH_PATH}"
echo "  --from_circuit ${CIRCUITS_PATH}"
echo "  S1.5 layerwise: ENABLED (default)"
echo "  S2 injection:   ENABLED (default)"
date

python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --from_circuit "$CIRCUITS_PATH" \
    --top_n_io_edges 10
# Sufficiency flags (all enabled by default):
#   --skip_sufficiency  → skip Phases 8+9 entirely
#   --skip_s1_5         → run S1 linear only (skip S1.5 layerwise)
#   --skip_s2           → skip cross-prompt injection (S2)
#   --debug_sanity      → log S1.5 hook sanity checks on first prompt
#   --max_prompts N     → truncate to N prompts (debug only)

echo ""
echo "========================================"
echo "Step 08 --from_circuit complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Outputs (updated in-place):"
echo "  ${CIRCUITS_PATH}"
echo "  data/results/causal_edges/${BEHAVIOUR}/presentation_graph_${BEHAVIOUR}_${SPLIT}.json"
