#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runC
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/probe-runC-%j.out
#SBATCH --error=logs/probe-runC-%j.out
#
# Run C: Top-10-per-layer sign-complete ablation
#
# Run C tests whether Run B's sign-complete structure persists under a larger
# attribution budget (top_k_per_layer=10 vs Run B's 5).
#
# Pipeline:
#   Step 06: Rebuild roleaware_static graph with top_k_per_layer=10
#            → new suffix _roleaware_static_k10
#            → expected ~160 feature nodes (VERIFY, do not assume)
#   Step 07: Sign-complete ablation on all Run C graph features
#            → saved to interventions/physics_decay_type_probe/runC/
#            → --ablation_sign all, --ablation_mode zero
#
# Output namespace: data/analysis/runC_top10_sign_complete/
# DOES NOT overwrite Run A or Run B outputs.
#
# Strict requirements:
#   - Record exact graph feature count; fail if 0
#   - Record positive:negative attribution sign ratio; compare to Run B 40:29
#   - Fail hard on control/fallback features (--require_n_graph_features is informational only)
#
# Runtime estimate:
#   Script 06: ~20 min (gradient computation over 108 prompts × 16 layers)
#   Script 07: ~6-8 h (N_features × 470 prompts × 16 layers)
#
# Prereqs:
#   - Script 04 features: data/results/transcoder_features/layer_*/physics_decay_type_train_top_k*.npy
#   - Prompts: data/prompts/physics_decay_type_probe_train.jsonl (470 prompts)
#   - Run B probe graph: data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe — Run C (top_k_per_layer=10)"
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

# ── Parameters ─────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type              # graph is built from the base behaviour
PROBE_BEHAVIOUR=physics_decay_type_probe  # ablation runs on probe prompts
SPLIT=train
N_GRAPH=108          # prompts used to build the attribution graph (same as Run B)
N_PROBE=470          # prompts ablated
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

TOP_K_PER_LAYER=10   # Run C: double the Run B budget of 5
VW_THRESH=0.01       # same static threshold as Run B roleaware_static graph
TOP_K_ABLATE=20      # top-k selection at ablation time (same as Run B)

GRAPH_SUFFIX="_roleaware_static_k10"
GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
GRAPH_PATH="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json"
OUT_DIR="data/analysis/runC_top10_sign_complete"

mkdir -p "$GRAPH_DIR" "$OUT_DIR"

echo "Parameters:"
echo "  BEHAVIOUR:       $BEHAVIOUR (graph) / $PROBE_BEHAVIOUR (ablation)"
echo "  TOP_K_PER_LAYER: $TOP_K_PER_LAYER"
echo "  GRAPH_SUFFIX:    $GRAPH_SUFFIX"
echo "  N_GRAPH:         $N_GRAPH"
echo "  N_PROBE:         $N_PROBE"
echo "  OUTPUT:          $OUT_DIR"
echo ""

# Verify script 04 features exist for graph-building behaviour
for layer in 10 18 24; do
    f="data/results/transcoder_features/layer_${layer}/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
    [ -f "$f" ] || { echo "ERROR: Missing script 04 feature file: $f"; exit 1; }
done
echo "Script 04 feature files verified (spot-check: L10, L18, L24)."
echo ""

# ── Step 06: Build Run C attribution graph ─────────────────────────────────────
echo "==== [Step 06] Build Run C graph (top_k_per_layer=${TOP_K_PER_LAYER}) ===="
date

python -u scripts/06_build_attribution_graph.py \
    --behaviour          "$BEHAVIOUR" \
    --split              "$SPLIT" \
    --n_prompts          "$N_GRAPH" \
    --model_size         4b \
    --layers             $LAYERS \
    --graph_node_mode    role_aware \
    --vw_threshold       "$VW_THRESH" \
    --top_k_per_layer    "$TOP_K_PER_LAYER" \
    --no_activation_weighted \
    --output_suffix      "$GRAPH_SUFFIX"

echo "  Step 06 done: $(date)"
echo ""

# Verify graph was created
[ -f "$GRAPH_PATH" ] || { echo "ERROR: Graph not created at $GRAPH_PATH"; exit 1; }

# ── Record exact Run C graph statistics ────────────────────────────────────────
echo "==== Run C graph statistics ===="
python3 -c "
import json
g = json.load(open('${GRAPH_PATH}'))
feat_nodes = [n for n in g['nodes'] if n.get('type') == 'feature']
n_feat = len(feat_nodes)
n_pos  = sum(1 for n in feat_nodes if n.get('grad_attr_sign', 0) > 0)
n_neg  = sum(1 for n in feat_nodes if n.get('grad_attr_sign', 0) < 0)
n_zero = n_feat - n_pos - n_neg
by_layer = {}
for n in feat_nodes:
    by_layer.setdefault(n['layer'], 0)
    by_layer[n['layer']] += 1
print(f'Total feature nodes: {n_feat}')
print(f'Positive attribution: {n_pos}  Negative: {n_neg}  Zero: {n_zero}')
print(f'Positive:Negative ratio = {n_pos}:{n_neg}  (Run B was 40:29)')
print(f'VW edges: {sum(1 for e in g[\"edges\"] if e.get(\"edge_type\") in {\"attribution_approx_v1\",\"virtual_weight\"})}')
print('Feature count by layer:')
for l in sorted(by_layer):
    print(f'  L{l}: {by_layer[l]}')

# Save stats for manifest
import json as j2
stats = {'n_features': n_feat, 'n_pos': n_pos, 'n_neg': n_neg, 'by_layer': by_layer,
         'graph_path': '${GRAPH_PATH}', 'runB_comparison': {'n_features': 69, 'n_pos': 40, 'n_neg': 29}}
j2.dump(stats, open('${OUT_DIR}/runC_graph_stats.json', 'w'), indent=2)
print(f'Stats saved to ${OUT_DIR}/runC_graph_stats.json')
" || { echo "ERROR: Graph stats failed"; exit 1; }

echo ""

# ── Step 07: Run C ablation ────────────────────────────────────────────────────
# Symlink Run C graph into probe behaviour namespace
PROBE_GRAPH_DIR="data/results/attribution_graphs/${PROBE_BEHAVIOUR}"
mkdir -p "$PROBE_GRAPH_DIR"
SYMLINK="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json"
ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK"
echo "Symlinked: $SYMLINK"
echo ""

echo "==== [Step 07] Run C ablation (sign-complete, all features) ===="
date

python -u scripts/07_run_interventions.py \
    --behaviour           "$PROBE_BEHAVIOUR" \
    --split               "$SPLIT" \
    --experiment          ablation \
    --n_prompts           "$N_PROBE" \
    --graph_n_prompts     "$N_GRAPH" \
    --graph_suffix        "$GRAPH_SUFFIX" \
    --layers              $LAYERS \
    --top_k               "$TOP_K_ABLATE" \
    --ablation_sign       all \
    --ablation_mode       zero \
    --output_subdir       runC \
    --per_feature

echo ""
echo "==== Step 07 done: $(date) ===="
echo ""

# ── Record Run C ablation statistics ──────────────────────────────────────────
ABL_CSV="data/results/interventions/${PROBE_BEHAVIOUR}/runC/intervention_ablation_${PROBE_BEHAVIOUR}.csv"
[ -f "$ABL_CSV" ] || { echo "ERROR: Ablation output not found: $ABL_CSV"; exit 1; }

python3 -c "
import pandas as pd, json
df = pd.read_csv('${ABL_CSV}')
n_feat = df.feature_id.nunique()
n_rows = len(df)
sfr    = df.sign_flipped.mean()
print(f'Run C ablation: {n_rows} rows | {n_feat} features | {df.prompt_idx.nunique()} prompts | SFR={sfr:.4f}')
print(f'Expected rows: {n_feat} × 470 = {n_feat*470}')
print(f'feature_source: {dict(df.feature_source.value_counts())}')
print(f'SFR by layer:')
print(df.groupby(\"layer\")[\"sign_flipped\"].mean().round(4).to_string())

stats = json.load(open('${OUT_DIR}/runC_graph_stats.json'))
stats['ablation_n_rows'] = n_rows
stats['ablation_n_features'] = n_feat
stats['ablation_n_prompts'] = int(df.prompt_idx.nunique())
stats['ablation_sfr'] = float(sfr)
stats['ablation_csv'] = '${ABL_CSV}'
import json as j2
j2.dump(stats, open('${OUT_DIR}/runC_graph_stats.json', 'w'), indent=2)
print(f'Updated stats saved.')
" || true

echo ""
echo "========================================================"
echo "  Run C pipeline done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""

# ── Manifest ──────────────────────────────────────────────────────────────────
cat > "${OUT_DIR}/runC_manifest.json" <<EOF
{
  "behaviour": "${PROBE_BEHAVIOUR}",
  "split": "${SPLIT}",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "top_k_per_layer": ${TOP_K_PER_LAYER},
  "graph_suffix": "${GRAPH_SUFFIX}",
  "graph_path": "${GRAPH_PATH}",
  "n_graph_prompts": ${N_GRAPH},
  "n_ablation_prompts": ${N_PROBE},
  "ablation_sign": "all",
  "ablation_mode": "zero",
  "ablation_csv": "${ABL_CSV}",
  "runB_reference": {
    "n_features": 69,
    "n_pos": 40,
    "n_neg": 29,
    "top_k_per_layer": 5,
    "slurm_job_id": "29432300"
  },
  "layers": "${LAYERS}"
}
EOF
echo "Manifest saved: ${OUT_DIR}/runC_manifest.json"
echo ""
echo "Next steps:"
echo "  1. Sync to local:"
echo "     rsync -avz iv294@login.hpc.cam.ac.uk:$(pwd)/data/results/interventions/${PROBE_BEHAVIOUR}/runC/ data/results/interventions/${PROBE_BEHAVIOUR}/runC/"
echo "     rsync -avz iv294@login.hpc.cam.ac.uk:$(pwd)/data/analysis/runC_top10_sign_complete/ data/analysis/runC_top10_sign_complete/"
echo "     rsync -avz iv294@login.hpc.cam.ac.uk:$(pwd)/data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json data/results/attribution_graphs/${BEHAVIOUR}/"
echo ""
echo "  2. Run local analysis pipeline:"
echo "     bash scripts/run_runC_pipeline.sh"
