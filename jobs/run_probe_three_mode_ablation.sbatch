#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-3mode
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probe-3mode-%j.out
#SBATCH --error=logs/probe-3mode-%j.out
#
# Task 1: Three-mode ablation control for Run B gating clusters
#
# Validates whether L18/L24 fixed-axis causal direction survives mean and
# resample ablation baselines (Run B used zero-ablation only).
#
# Clusters ablated:
#   L18 gating group   (C8 in Run B CSD3 labeling):  4 features, pro-α / anti-β
#   L24 gating group   (C6 in Run B CSD3 labeling):  5 features, anti-α / pro-β
#   L16 single-layer   (C11 in Run B CSD3 labeling): 5 features, α-composition
#   L10 control        (C0 in Run B CSD3 labeling):  5 features, early β routing
#
# Three modes run:
#   zero     — already done in Run B; re-run here for exact same feature subset
#   mean     — set to empirical mean activation across training prompts
#   resample — replace with activation sampled from a random different prompt
#
# Outputs (all under data/analysis/runB_validation/three_mode_ablation_control/):
#   intervention_ablation_{behaviour}_{mode}.csv
#   After job: sync, run scripts/runB_three_mode_analysis.py locally
#
# Prereqs:
#   - Run B ablation done (SLURM 29432300)
#   - Top-k activation files at:
#       data/results/transcoder_features/layer_*/physics_decay_type_probe_train_top_k_{indices,values}.npy
#
# Runtime estimate: 19 features × 470 prompts × 3 modes × ~2 passes ≈ 30-45 min

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe — Three-mode ablation control"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

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
echo ""

# ── Paths ──────────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type_probe
SPLIT=train
N_PROMPTS=470
LAYERS="10 16 18 24"          # only the four target cluster layers

# Run B graph (same as used for Run B ablation)
GRAPH_BEHAVIOUR=physics_decay_type
GRAPH_N=108
GRAPH_SUFFIX=_roleaware_static
GRAPH_PATH="data/results/attribution_graphs/${GRAPH_BEHAVIOUR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"

OUT_BASE="data/analysis/runB_validation/three_mode_ablation_control"
mkdir -p "$OUT_BASE"

# Verify
for f in "$GRAPH_PATH" "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

# Symlink graph for probe behaviour namespace (idempotent)
PROBE_GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
mkdir -p "$PROBE_GRAPH_DIR"
SYMLINK="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK" 2>/dev/null || true

echo "Graph: $GRAPH_PATH"
echo "Layers: $LAYERS"
echo "Modes:  zero mean resample"
echo ""

# ── Run all three modes ────────────────────────────────────────────────────────
for MODE in zero mean resample; do
    echo "========== Mode: ${MODE} ========== $(date)"

    python -u scripts/07_run_interventions.py \
        --behaviour           "$BEHAVIOUR" \
        --split               "$SPLIT" \
        --experiment          ablation \
        --n_prompts           "$N_PROMPTS" \
        --graph_n_prompts     "$GRAPH_N" \
        --graph_suffix        "$GRAPH_SUFFIX" \
        --layers              $LAYERS \
        --top_k               20 \
        --ablation_sign       all \
        --ablation_mode       "$MODE" \
        --output_subdir       "three_mode_${MODE}" \
        --per_feature

    # Copy output to three_mode dir (script 07 saves to interventions/{behaviour}/{subdir}/)
    SRC="data/results/interventions/${BEHAVIOUR}/three_mode_${MODE}/intervention_ablation_${BEHAVIOUR}.csv"
    DST="${OUT_BASE}/intervention_ablation_${BEHAVIOUR}_${MODE}.csv"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST"
        echo "  Saved: $DST"
    else
        echo "  WARNING: output not found at $SRC"
    fi

    echo "  Mode ${MODE} done: $(date)"
    echo ""
done

echo "========================================================"
echo "  Three-mode ablation done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Sync outputs:"
echo "     rsync -avz iv294@login.hpc.cam.ac.uk:$(pwd)/data/analysis/runB_validation/three_mode_ablation_control/ \\"
echo "         data/analysis/runB_validation/three_mode_ablation_control/"
echo ""
echo "  2. Run local analysis:"
echo "     python3 scripts/runB_three_mode_analysis.py"
