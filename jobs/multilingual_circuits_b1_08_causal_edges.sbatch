#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_causal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/mc_b1_causal-%j.out
#SBATCH --error=logs/mc_b1_causal-%j.err
#
# multilingual_circuits_b1 — script 08: intervention-based causal edges + S1.5 sufficiency
#
# Phases:
#   Phase 0   Load roleaware attribution graph (96 prompts)
#   Phase 1   AGW prefilter: top 60% of ~575 VW edges → ~345 candidate pairs
#   Phase 2   Load model + transcoders (layers 10-25)
#   Phase 3   Baseline cache: 96 forward passes
#   Phase 4   Ablation loop: ~71 sources × 96 prompts ≈ 6,816 passes
#   Phase 5   Aggregate: mean_delta_abs → causal edges; filter tau_causal=0.10
#   Phase 6   Path tracing: top-20 paths → circuit JSON
#   Phase 7   Necessity (group ablation): ~96 passes
#   Phase 8   Sufficiency S1 (linear) + S1.5 (layerwise): 3 × 96 = 288 passes
#   Phase 9   Sufficiency S2 (cross-prompt injection): ~8 passes
#   Phase 10  Presentation graph (Graph B)
#   Manifest  Saved at startup to out_dir/manifest_*.json
#
# Estimated forward passes: ~7,300  (Phase 4 dominates)
# Estimated wall time: ~2.5h (5h slot = safe margin including S1.5)
#
# BEFORE SUBMITTING:
#   1. Run debug job first: sbatch jobs/multilingual_circuits_b1_08_s15_debug.sbatch
#   2. Verify no "S1.5 sanity FAIL" in debug log
#   3. Then:
#      cd /rds/user/iv294/hpc-work/thesis/project
#      git pull
#      mkdir -p logs
#      sbatch jobs/multilingual_circuits_b1_08_causal_edges.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_b1_causal-<JOBID>.out for:
#     "Causal edges: N (from M candidates)"
#     "Circuit: F features, E edges (P paths)"
#     "Phase 8 S1  linear:   sign=X, retention=Y, verdict=Z"
#     "Phase 8 S1.5 layerwise: sign=X, retention=Y, verdict=Z, improvement=D"
#   Sync outputs:
#     git add data/results/causal_edges/multilingual_circuits_b1/
#     git commit -m "script 08: causal edges + circuits + S1.5 sufficiency for b1 train"
#   Quick inspect:
#     python -c "
#       import json
#       d = json.load(open('data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json'))
#       print('necessity:', d['validation'])
#       print('S1.5:', d.get('sufficiency_s1_5'))
#     "

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Causal Edges + S1.5 (Script 08)"
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
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=20

# Verify roleaware graph is present
GRAPH_PATH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"
if [ ! -f "$GRAPH_PATH" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_PATH"
    echo "  Run multilingual_circuits_b1_02_09.sbatch first (SLURM 25307123)."
    exit 1
fi
echo "Graph: $GRAPH_PATH (OK)"
python -c "import json; g=json.load(open('${GRAPH_PATH}')); print(f'  {len([n for n in g[\"nodes\"] if n[\"type\"]==\"feature\"])} feature nodes, {len([e for e in g[\"edges\"] if e.get(\"edge_type\")==\"virtual_weight\"])} VW edges')"

# ══════════════════════════════════════════════════════════════
# Step 08: Causal edges + sufficiency (S1 linear + S1.5 layerwise)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 08: Causal edges + S1 + S1.5 sufficiency ===="
echo "  --behaviour ${BEHAVIOUR}"
echo "  --split ${SPLIT}"
echo "  --graph_json ${GRAPH_PATH}"
echo "  --agw_top_frac ${AGW_TOP_FRAC}"
echo "  --tau_causal ${TAU_CAUSAL}"
echo "  --n_paths ${N_PATHS}"
echo "  S1.5 layerwise: ENABLED (default)"
echo "  S2 injection:   ENABLED (default)"
date

python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --agw_top_frac "$AGW_TOP_FRAC" \
    --tau_causal "$TAU_CAUSAL" \
    --n_paths "$N_PATHS" \
    --top_n_io_edges 10
# Sufficiency flags (all enabled by default — see debug job to test first):
#   --skip_sufficiency  → skip Phases 8+9 entirely
#   --skip_s1_5         → run S1 linear only (skip S1.5 layerwise)
#   --skip_s2           → skip cross-prompt injection (S2)
#   --debug_sanity      → log S1.5 hook sanity checks on first prompt

echo ""
echo "========================================"
echo "Step 08 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Outputs:"
echo "  data/results/causal_edges/${BEHAVIOUR}/manifest_${BEHAVIOUR}_${SPLIT}.json"
echo "  data/results/causal_edges/${BEHAVIOUR}/causal_edges_${BEHAVIOUR}_${SPLIT}.json"
echo "  data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"
echo "  data/results/causal_edges/${BEHAVIOUR}/presentation_graph_${BEHAVIOUR}_${SPLIT}.json"
