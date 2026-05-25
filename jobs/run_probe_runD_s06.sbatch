#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=probe-runD-s06
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-runD-s06-%j.out
#SBATCH --error=logs/probe-runD-s06-%j.out
#
# runD Step 2: Build attribution graph from 538 probe_v2 prompts, top_k_per_layer=20
#
# KEY DIFFERENCE from runC:
#   runC graph: built from physics_decay_type 108 prompts → biased towards v2 circuit features
#   runD graph: built from physics_decay_type_probe_v2 538 prompts → covers all probe types
#              (L1 atomic / L2 relational / L3 abstract / AUX / keyword)
#
# With 538 prompts × top_k_per_layer=20, the union covers many more unique features
# per layer, especially for early layers (L10-L18) which were thin in runC.
#
# Output: data/results/attribution_graphs/physics_decay_type_probe_v2/
#           attribution_graph_train_n538_roleaware_static_k20.json
#
# PREREQ: run_probe_runD_s04.sbatch must complete first
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs
#   sbatch jobs/run_probe_runD_s06.sbatch

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD Step 06"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python: $(which python)"
python --version
echo ""

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
N_PROMPTS=538
TOP_K_PER_LAYER=20
VW_THRESH=0.01
GRAPH_SUFFIX="_roleaware_static_k20"
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

GRAPH_PATH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"

# Verify script 04 features exist
for layer in 10 18 24; do
    f="data/results/transcoder_features/layer_${layer}/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
    [ -f "$f" ] || { echo "ERROR: Missing script 04 output: $f"; echo "Run run_probe_runD_s04.sbatch first"; exit 1; }
done
echo "Script 04 features verified."
echo ""

mkdir -p "data/results/attribution_graphs/${BEHAVIOUR}"

echo "==== [Step 06] Build attribution graph ===="
echo "  Behaviour:        $BEHAVIOUR"
echo "  N_prompts:        $N_PROMPTS (all 538 probe_v2 train prompts)"
echo "  top_k_per_layer:  $TOP_K_PER_LAYER"
echo "  Graph suffix:     $GRAPH_SUFFIX"
date

python -u scripts/06_build_attribution_graph.py \
    --behaviour          "$BEHAVIOUR" \
    --split              "$SPLIT" \
    --n_prompts          "$N_PROMPTS" \
    --model_size         4b \
    --layers             $LAYERS \
    --graph_node_mode    role_aware \
    --vw_threshold       "$VW_THRESH" \
    --top_k_per_layer    "$TOP_K_PER_LAYER" \
    --no_activation_weighted \
    --output_suffix      "$GRAPH_SUFFIX"

echo ""
echo "==== Step 06 done: $(date) ===="
echo ""

[ -f "$GRAPH_PATH" ] || { echo "ERROR: Graph not created: $GRAPH_PATH"; exit 1; }

# Print graph statistics
python3 -c "
import json
from collections import Counter
g = json.load(open('$GRAPH_PATH'))
feat = [n for n in g['nodes'] if n.get('type') == 'feature']
n = len(feat)
by_layer = Counter(n_['layer'] for n_ in feat)
n_pos  = sum(1 for n_ in feat if n_.get('grad_attr_sign', 0) > 0)
n_neg  = sum(1 for n_ in feat if n_.get('grad_attr_sign', 0) < 0)
n_zero = n - n_pos - n_neg
print(f'Total feature nodes: {n}  (runC had 102)')
print(f'Attribution sign: pos={n_pos}  neg={n_neg}  unknown={n_zero}')
print(f'VW edges: {sum(1 for e in g[\"edges\"])}')
print()
print('Features per layer:')
for l in sorted(by_layer):
    bar = \"#\" * by_layer[l]
    print(f'  L{l:>2}: {by_layer[l]:>3}  {bar}')
"

echo ""
echo "========================================================"
echo "  Done: $(date)  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Next: submit ablation jobs in parallel:"
echo "  sbatch jobs/run_probe_runD_abl_p1.sbatch"
echo "  sbatch jobs/run_probe_runD_abl_p2.sbatch"
echo "  sbatch jobs/run_probe_runD_abl_p3.sbatch"
