#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=rand-init-ctrl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rand-init-ctrl-%j.out
#SBATCH --error=logs/rand-init-ctrl-%j.out
#
# Script 52: Random-init control (Wu et al. 2024)
# Tests whether IIA signal reflects model computation or transcoder capacity.
# Loads BOTH trained Qwen3-4B AND a random-init copy (~24GB total VRAM).
#
# Independent of Run C ablation — can run in parallel with runC_p2.
# Runtime estimate: ~4-6h (all clusters, ~269 pairs each, 2 models)

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Script 52: Random-Init IIA Control"
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
CLUSTER_JSON="dashboard_probe/public/data/cluster_semantics.json"
OUTPUT_DIR="data/analysis/random_init_control"

PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
[ -f "$PROMPT_FILE" ]   || { echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1; }
[ -f "$CLUSTER_JSON" ]  || { echo "ERROR: Cluster JSON not found: $CLUSTER_JSON"; exit 1; }
echo "Prompts:     $PROMPT_FILE"
echo "Clusters:    $CLUSTER_JSON"
echo ""

python -u scripts/52_random_init_control.py \
    --behaviour   "$BEHAVIOUR" \
    --split       "$SPLIT" \
    --all_clusters \
    --cluster_json "$CLUSTER_JSON" \
    --seed        42 \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
