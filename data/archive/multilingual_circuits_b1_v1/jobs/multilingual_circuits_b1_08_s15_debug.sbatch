#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_s15_dbg
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mc_b1_s15_debug-%j.out
#SBATCH --error=logs/mc_b1_s15_debug-%j.err
#
# multilingual_circuits_b1 — script 08: S1.5 SANITY / DEBUG RUN
#
# Purpose: cheap sanity check before the full 4h job.
# Uses 3 prompts only.  Runs full pipeline (Phases 0–10) with:
#   --max_prompts 3       → 3 prompts from train set
#   --debug_sanity        → S1.5 hook checks on first eligible prompt
#   --skip_s2             → skip cross-prompt injection (needs pairing, not useful at 3)
#
# Estimated passes:
#   Phase 3  baseline:    3
#   Phase 4  ablation:  ~71 src × 3 = ~213
#   Phase 7  necessity:   3
#   Phase 8  S1+S1.5:    3×3 = 9
#   TOTAL ≈ 228 passes → ~5–10 min compute + ~5 min model load ≈ 20 min
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_b1_08_s15_debug.sbatch
#
# WHAT TO CHECK IN LOG (logs/mc_b1_s15_debug-<JOBID>.out):
#   1. "Manifest saved: ..."                       → startup OK
#   2. "DEBUG: truncated to 3 prompts"             → max_prompts active
#   3. "S1.5 mask L<N>: K circuit features / ..."  → masks built correctly
#   4. "S1.5 L<N>: N/K circuit features active"    → sanity hook fired
#   5. No "S1.5 sanity FAIL" lines                 → masking correct
#   6. "S1.5 layerwise: sign=..., retention=..., verdict=..."
#   7. Outputs: data/results/causal_edges/.../debug_n3/...
#
# HARD STOP conditions (do NOT submit full job if any appear):
#   - "S1.5 sanity FAIL": non-circuit features nonzero after masking
#   - "ERROR" or Python exception in log
#   - NaN in any logit diff (check "retention=" line for None/nan)
#   - "out of bounds" features in mask build (S1.5 L<N>: X circuit feature indices OOB)
#

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — S1.5 DEBUG RUN"
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
    echo "  Run multilingual_circuits_b1_02_09.sbatch first."
    exit 1
fi
echo "Graph: $GRAPH_PATH (OK)"
python -c "import json; g=json.load(open('${GRAPH_PATH}')); print(f'  {len([n for n in g[\"nodes\"] if n[\"type\"]==\"feature\"])} feature nodes, {len([e for e in g[\"edges\"] if e.get(\"edge_type\")==\"virtual_weight\"])} VW edges')"

echo ""
echo "==== DEBUG RUN: 3 prompts, debug_sanity=True, skip_s2 ===="
date

# Enable DEBUG-level logging for full S1.5 hook diagnostics
export PYTHONUNBUFFERED=1

python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --agw_top_frac "$AGW_TOP_FRAC" \
    --tau_causal "$TAU_CAUSAL" \
    --n_paths "$N_PATHS" \
    --top_n_io_edges 5 \
    --max_prompts 3 \
    --debug_sanity \
    --skip_s2

echo ""
echo "========================================"
echo "DEBUG RUN complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Debug outputs (separate dir, safe to delete):"
echo "  data/results/causal_edges/${BEHAVIOUR}/debug_n3/manifest_${BEHAVIOUR}_${SPLIT}.json"
echo "  data/results/causal_edges/${BEHAVIOUR}/debug_n3/circuits_${BEHAVIOUR}_${SPLIT}.json"
echo ""
echo "Quick log check:"
echo "  grep -E 'S1.5 sanity|S1.5 mask|retention_improvement|verdict|ERROR|NaN' \\"
echo "    logs/mc_b1_s15_debug-${SLURM_JOB_ID}.out"
