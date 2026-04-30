#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe_cd
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/probe_cd-%j.out
#SBATCH --error=logs/probe_cd-%j.out
#
# physics_decay_type_probe — Sections C + D only
#
# Prerequisites (already done):
#   Section A (baseline):           data/results/baseline_physics_decay_type_probe_train.csv
#   Section B (feature extraction): data/results/transcoder_features/layer_{10..25}/physics_decay_type_probe_train_*.npy
#
# This job runs:
#   Section C — ablation (step 07) using existing physics_decay_type train graph
#   Section D — UI preparation (step 09)
#
# After this job completes, rsync to local and run:
#   python scripts/18_latent_state_analysis.py --behaviour physics_decay_type_probe --split train --baseline_only
#   python scripts/18_latent_state_analysis.py --behaviour physics_decay_type_probe --split train
#
# BEFORE SUBMITTING:
#   git pull
#   sbatch jobs/probe_cd_only.sbatch

BEHAVIOUR=physics_decay_type_probe
SPLIT=train
N_PROMPTS=470
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_ABLATE=20
GRAPH_BEHAVIOUR=physics_decay_type
GRAPH_SPLIT=train
GRAPH_N=108
GRAPH_SUFFIX=_roleaware_static

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  probe_cd_only: Sections C + D"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  Started $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

# ── Pre-flight ────────────────────────────────────────────────────────────────
GRAPH_PATH="data/results/attribution_graphs/${GRAPH_BEHAVIOUR}/attribution_graph_${GRAPH_SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
PROBE_PROMPTS="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"

echo ""
echo "==== Pre-flight ===="
[ -f "$PROBE_PROMPTS" ] && echo "  Prompts: $(wc -l < $PROBE_PROMPTS) lines — OK" || { echo "  [FATAL] Prompts not found: $PROBE_PROMPTS"; exit 1; }
[ -f "$GRAPH_PATH" ]   && echo "  Graph:   $GRAPH_PATH — OK"                     || { echo "  [FATAL] Graph not found: $GRAPH_PATH"; exit 1; }

# Confirm step 04 outputs present
python3 -c "
from pathlib import Path
missing = [l for l in range(10, 26)
           if not (Path('data/results/transcoder_features') / f'layer_{l}' / 'physics_decay_type_probe_train_top_k_indices.npy').exists()]
if missing:
    print(f'  [FATAL] Step 04 outputs missing for layers: {missing}')
    exit(1)
print(f'  Step 04 outputs: all 16 layers present — OK')
"
[ $? -ne 0 ] && exit 1

echo "  Pre-flight done at $(date)"

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION C: Ablation (reuses existing physics_decay_type train graph)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION C: Ablation  $(date)"

PROBE_GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
mkdir -p "$PROBE_GRAPH_DIR"
SYMLINK_TARGET="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK_TARGET"
echo "  Symlinked graph → $SYMLINK_TARGET"

python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature

RC_C=$?
echo "Section C exit code: $RC_C  ($(date))"

if [ $RC_C -eq 0 ]; then
    python3 -c "
import pandas as pd
from pathlib import Path
candidates = [
    Path('data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv'),
    Path('data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}.csv'),
]
p = next((c for c in candidates if c.exists()), None)
if p is None:
    print('  [WARN] Ablation CSV not found after Section C'); exit()
df = pd.read_csv(p)
print(f'  Ablation CSV: {p.name}')
print(f'  Rows: {len(df)}  Features: {df[\"feature_id\"].nunique()}  Prompts: {df[\"prompt_idx\"].nunique()}')
print(f'  Overall sign_flip_rate: {df[\"sign_flipped\"].mean():.4f}')
" || true
fi

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION D: UI preparation
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION D: UI prep  $(date)"

python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --community_method louvain

RC_D=$?
echo "Section D exit code: $RC_D  ($(date))"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  FINAL SUMMARY — Job $SLURM_JOB_ID — $(date)"
[ $RC_C -eq 0 ] && echo "  ✓ Section C (ablation) PASSED" || echo "  ✗ Section C (ablation) FAILED (exit $RC_C)"
[ $RC_D -eq 0 ] && echo "  ✓ Section D (UI prep)  PASSED" || echo "  ✗ Section D (UI prep)  FAILED (exit $RC_D)"
echo ""
echo "Key outputs:"
echo "  Ablation: data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv"
echo "  UI run:   data/ui_offline/*_${BEHAVIOUR}_*/"
echo ""
echo "After rsync to local, run:"
echo "  python scripts/18_latent_state_analysis.py --behaviour ${BEHAVIOUR} --split ${SPLIT} --baseline_only"
echo "  python scripts/18_latent_state_analysis.py --behaviour ${BEHAVIOUR} --split ${SPLIT}"
echo "========================================================"

[ $RC_C -ne 0 ] && exit 1
[ $RC_D -ne 0 ] && exit 2
exit 0
