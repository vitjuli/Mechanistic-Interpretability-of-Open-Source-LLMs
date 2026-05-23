#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runC-p1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=13:00:00
#SBATCH --output=logs/probe-runC-p1-%j.out
#SBATCH --error=logs/probe-runC-p1-%j.out
#
# Run C ablation — Part 1: prompts 0-269
#
# Step 06 already done (Job 29502667): graph at
#   data/results/attribution_graphs/physics_decay_type/
#     attribution_graph_train_n108_roleaware_static_k10.json
#   102 features (51 pos / 51 neg)
#
# Submit both p1 and p2 simultaneously; merge CSVs locally after both sync.
# Part 2: sbatch jobs/run_probe_runC_ablation_p2.sbatch

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Run C ablation — Part 1 (prompts 0-269)"
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

PROBE_BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
N_GRAPH=108
GRAPH_SUFFIX="_roleaware_static_k10"
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

GRAPH_PATH="data/results/attribution_graphs/physics_decay_type/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json"
[ -f "$GRAPH_PATH" ] || { echo "ERROR: Graph not found: $GRAPH_PATH"; exit 1; }
echo "Graph verified: $GRAPH_PATH"

PROBE_GRAPH_DIR="data/results/attribution_graphs/${PROBE_BEHAVIOUR}"
mkdir -p "$PROBE_GRAPH_DIR"
SYMLINK="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json"
ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK" 2>/dev/null || true

echo "==== [Step 07] Run C ablation — Part 1 (prompts 0-269) ===="
date

python -u scripts/07_run_interventions.py \
    --behaviour           "$PROBE_BEHAVIOUR" \
    --split               "$SPLIT" \
    --experiment          ablation \
    --n_prompts           270 \
    --prompt_offset       0 \
    --graph_n_prompts     "$N_GRAPH" \
    --graph_suffix        "$GRAPH_SUFFIX" \
    --layers              $LAYERS \
    --top_k               20 \
    --ablation_sign       all \
    --ablation_mode       zero \
    --output_subdir       runC_p1 \
    --per_feature

echo ""
echo "==== Part 1 done: $(date) ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${PROBE_BEHAVIOUR}/runC_p1/ data/results/interventions/${PROBE_BEHAVIOUR}/runC_p1/"
