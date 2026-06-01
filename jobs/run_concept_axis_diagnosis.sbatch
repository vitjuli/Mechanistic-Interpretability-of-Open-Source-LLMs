#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=concept-axis
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/concept-axis-%j.out
#SBATCH --error=logs/concept-axis-%j.out
#
# 64: Concept axis diagnosis (D1 + D2). Tests the premise of all prior scripts:
# is γ̄ (= γ(beta) − γ(alpha) from unembedding) actually the concept axis
# in RESIDUAL STREAM, or did we measure the wrong axis all along?
#
# D1 (core, ~15 min): residual Fisher axis w_res at depths L14..L25 + final,
#                     compare cos_C(w_res, γ̄) against null band.
# D2 (optional, slow): carrier ablation vs random ablation Δlogit comparison.
#
# Decision (M1/M2/M3):
#   high cos at final → γ̄ IS read-axis; paradox real (M2 mediation / M3 gating)
#   low cos at final → γ̄ WRONG axis (M1); recompute Stage-1 geometry on w_res
#
# PREREQUISITE: concept_directions.npz (γ̄, Σ⁻¹) from script 60_
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/runD_v2/geometry_stage1
#   sbatch jobs/run_concept_axis_diagnosis.sbatch

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/geometry_stage1

echo "========================================================"
echo "  Concept Axis Diagnosis (64 — is γ̄ the right axis?)"
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
nvidia-smi || true
echo ""

# ── verify prerequisites ────────────────────────────────────────────────────
GEOM_NPZ=data/analysis/runD_v2/geometry_stage1/concept_directions.npz
CLUSTERS=data/analysis/runD_v2/clustering_full/cluster_labels.csv
PROMPTS=data/prompts/physics_decay_type_probe_v2_train.jsonl

[ -f "$GEOM_NPZ" ] || { echo "ERROR: $GEOM_NPZ not found"; exit 1; }
[ -f "$CLUSTERS" ] || { echo "ERROR: $CLUSTERS not found"; exit 1; }
[ -f "$PROMPTS"  ] || { echo "ERROR: $PROMPTS not found"; exit 1; }
echo "Prerequisites verified."

# Use same carrier features as 62/63 (coimp_louvain: cluster 13=L24, cluster 8=L18)
# Sweep depths L14..L25 + final residual
# D2 enabled with 50 random sets on 100 prompts (cap so it finishes in time)
date
python -u scripts/64_concept_axis_diagnosis.py \
    --concept_npz    "$GEOM_NPZ" \
    --prompts        "$PROMPTS" \
    --cluster_labels "$CLUSTERS" \
    --cluster_col    coimp_louvain \
    --clusters       13,8 \
    --layers         24 18 \
    --sweep_min      14 \
    --sweep_max      25 \
    --train_frac     0.6 \
    --shrink         0.1 \
    --cos_hi         0.5 \
    --cos_lo         0.2 \
    --run_d2 \
    --d2_max         100 \
    --n_random_d2    50 \
    --out_dir        data/analysis/runD_v2/geometry_stage1 \
    --device         cuda \
    --model_size     4b \
    --seed           0

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Read: data/analysis/runD_v2/geometry_stage1/concept_axis_diagnosis.json"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/geometry_stage1/ data/analysis/runD_v2/geometry_stage1/"
