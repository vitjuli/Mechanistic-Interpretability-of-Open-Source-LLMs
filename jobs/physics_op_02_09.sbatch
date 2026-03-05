#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_sv_pipe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phys_sv_pipe-%j.out
#SBATCH --error=logs/phys_sv_pipe-%j.err
#
# BEFORE SUBMITTING: run from the project root and ensure logs/ exists:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   mkdir -p logs
#   sbatch jobs/physics_op_02_09.sbatch

set -euo pipefail

# Belt-and-suspenders: ensure logs dir exists for any mid-script output
mkdir -p logs

echo "========================================"
echo "Physics Scalar/Vector Operator Pipeline"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# --- Purge Intel/Cascade Lake modules (AVX-512 -> SIGILL on AMD EPYC) ---
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# CPU instruction safety for AMD EPYC nodes (no AVX-512)
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=physics_scalar_vector_operator
SPLIT=train
N_PROMPTS=80
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

# ══════════════════════════════════════════════════════════════
# Step 01: Generate prompts
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 01: Generate prompts ===="
date
python -u scripts/01_generate_prompts.py --behaviour "$BEHAVIOUR"

# ══════════════════════════════════════════════════════════════
# Step 02: Baseline evaluation
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ══════════════════════════════════════════════════════════════
# Step 04: Extract transcoder features
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 04: Extract transcoder features ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50

# ══════════════════════════════════════════════════════════════
# Step 06: Build attribution graph
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 06: Build attribution graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

# ══════════════════════════════════════════════════════════════
# Step 07: Run interventions (ablation, patching, steering)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 07: Interventions ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment all \
    --n_prompts 20 \
    --top_k 5 \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

# ══════════════════════════════════════════════════════════════
# Step 09: Prepare offline UI data
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 09: Prepare offline UI ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --community_method louvain

# ══════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Sanity check ===="
date

# Find the most recent run directory for this behaviour
RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | tail -1)
if [ -n "$RUN_DIR" ]; then
    python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
else
    echo "WARNING: Could not find run directory for sanity check"
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
date
