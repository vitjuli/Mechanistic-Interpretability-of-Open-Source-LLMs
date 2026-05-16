#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runB
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/probe-runB-%j.out
#SBATCH --error=logs/probe-runB-%j.out
#
# physics_decay_type_probe — Run B ablation (sign-complete, 69 features)
#
# Run A used --ablation_sign pos (default), which silently dropped 29 negative-
# attribution (suppressor) features and ran only the 40 positive-attribution ones.
#
# Run B uses --ablation_sign all so all 69 graph features are ablated.
# Results are saved to interventions/physics_decay_type_probe/runB/ (via
# --output_subdir runB) and do NOT overwrite the Run A output.
#
# The --require_n_graph_features 69 guard aborts if the wrong graph is loaded.
#
# Prereqs: physics_decay_type_probe_pipeline.sbatch sections A+B already done.
#          Graph at data/results/attribution_graphs/physics_decay_type/
#                     attribution_graph_train_n108_roleaware_static.json
#
# Runtime estimate: 69 features × 470 prompts × 16 layers ≈ 4-5 h.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe — Run B ablation (sign-complete)"
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

# ── Parameters ────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type_probe
SPLIT=train
N_PROMPTS=470
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_ABLATE=20
GRAPH_BEHAVIOUR=physics_decay_type
GRAPH_N=108
GRAPH_SUFFIX=_roleaware_static
GRAPH_PATH="data/results/attribution_graphs/${GRAPH_BEHAVIOUR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"

echo "Parameters:"
echo "  BEHAVIOUR:    $BEHAVIOUR"
echo "  N_PROMPTS:    $N_PROMPTS"
echo "  LAYERS:       $LAYERS"
echo "  GRAPH:        $GRAPH_PATH"
echo "  ablation_sign: all  (Run B — sign-complete, 69 features)"
echo "  output_subdir: runB"
echo ""

# ── Symlink graph into probe behaviour directory ───────────────────────────────
PROBE_GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
mkdir -p "$PROBE_GRAPH_DIR"
SYMLINK_TARGET="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK_TARGET"
echo "Symlinked graph: $SYMLINK_TARGET -> $(pwd)/${GRAPH_PATH}"
echo ""

# ── Run ablation ──────────────────────────────────────────────────────────────
echo "==== [Run B] Ablation — all 69 graph features ===="
date

python -u scripts/07_run_interventions.py \
    --behaviour           "$BEHAVIOUR" \
    --split               "$SPLIT" \
    --experiment          ablation \
    --n_prompts           "$N_PROMPTS" \
    --graph_n_prompts     "$GRAPH_N" \
    --graph_suffix        "$GRAPH_SUFFIX" \
    --layers              $LAYERS \
    --top_k               "$TOP_K_ABLATE" \
    --ablation_sign       all \
    --output_subdir       runB \
    --require_n_graph_features 69 \
    --per_feature

echo ""
echo "==== Ablation done: $(date) ===="
echo ""

# ── Quick summary ─────────────────────────────────────────────────────────────
python3 -c "
import pandas as pd
from pathlib import Path

p = Path('data/results/interventions/physics_decay_type_probe/runB/intervention_ablation_physics_decay_type_probe.csv')
if not p.exists():
    print('Output CSV not found:', p)
    exit()
df = pd.read_csv(p)
print(f'Run B ablation summary:')
print(f'  Rows:            {len(df)}')
print(f'  Unique features: {df[\"feature_id\"].nunique()}')
print(f'  Unique prompts:  {df[\"prompt_idx\"].nunique()}')
sfr = df['sign_flipped'].mean()
print(f'  Overall sfr:     {sfr:.4f}')
by_layer = df.groupby('layer')['sign_flipped'].mean()
print(f'  SFR by layer:')
for l, v in by_layer.items():
    print(f'    L{l}: {v:.4f}')

# Sign breakdown
pos_feats = df[df['effect_size'] > 0]['feature_id'].nunique()
neg_feats = df[df['effect_size'] < 0]['feature_id'].nunique()
print(f'  Features with any pos effect: {pos_feats}')
print(f'  Features with any neg effect: {neg_feats}')
" || true

echo ""
echo "========================================================"
echo "  Run B done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Sync to local:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_decay_type_probe/runB/ data/results/interventions/physics_decay_type_probe/runB/"
