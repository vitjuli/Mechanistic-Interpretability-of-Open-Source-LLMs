#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runC-p2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --output=logs/probe-runC-p2-%j.out
#SBATCH --error=logs/probe-runC-p2-%j.out
#
# Run C ablation — Part 2: prompts 270-537
#
# Submit alongside Part 1; merge CSVs locally after both sync.
# Part 1: sbatch jobs/run_probe_runC_ablation_p1.sbatch

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Run C ablation — Part 2 (prompts 270-537)"
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

echo "==== [Step 07] Run C ablation — Part 2 (prompts 270-537) ===="
date

python -u scripts/07_run_interventions.py \
    --behaviour           "$PROBE_BEHAVIOUR" \
    --split               "$SPLIT" \
    --experiment          ablation \
    --n_prompts           268 \
    --prompt_offset       270 \
    --graph_n_prompts     "$N_GRAPH" \
    --graph_suffix        "$GRAPH_SUFFIX" \
    --layers              $LAYERS \
    --top_k               20 \
    --ablation_sign       all \
    --ablation_mode       zero \
    --output_subdir       runC_p2 \
    --per_feature

echo ""
echo "==== Part 2 done: $(date) ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${PROBE_BEHAVIOUR}/runC_p2/ data/results/interventions/${PROBE_BEHAVIOUR}/runC_p2/"
