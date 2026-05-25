#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runD-abl-p3
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/probe-runD-abl-p3-%j.out
#SBATCH --error=logs/probe-runD-abl-p3-%j.out
#
# runD Ablation Part 3: prompts 360-537
# Run in parallel with p1 and p2.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD Ablation Part 3 (360-537)"
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

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
N_GRAPH=538
GRAPH_SUFFIX="_roleaware_static_k20"
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

GRAPH_PATH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_GRAPH}${GRAPH_SUFFIX}.json"
[ -f "$GRAPH_PATH" ] || { echo "ERROR: Graph not found: $GRAPH_PATH"; exit 1; }
echo "Graph verified: $GRAPH_PATH"

echo "==== [Step 07] Ablation: prompts 360-537 ===="
date

python -u scripts/07_run_interventions.py \
    --behaviour        "$BEHAVIOUR" \
    --split            "$SPLIT" \
    --experiment       ablation \
    --n_prompts        178 \
    --prompt_start     360 \
    --graph_n_prompts  "$N_GRAPH" \
    --graph_suffix     "$GRAPH_SUFFIX" \
    --layers           $LAYERS \
    --top_k            20 \
    --output_subdir    runD_p3 \
    --per_feature

echo ""
echo "==== Done: $(date) ===="
