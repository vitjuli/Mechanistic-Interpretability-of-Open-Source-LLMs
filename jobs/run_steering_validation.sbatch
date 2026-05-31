#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=steering-validation
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/steering-validation-%j.out
#SBATCH --error=logs/steering-validation-%j.out
#
# 61b: Steering specificity validation.
# Tests whether steering along lbar = Σ⁻¹γ̄ is SPECIFIC to that direction
# vs random / S⊥ / format directions of the same raw-L2 norm. Closes the gap
# that 61's steering sweep had no controls.
#
# Four blocks:
#   (1) Specificity controls (random/sperp/format at same c)
#   (2) Symmetric directionality (α→β at c>0; β→α at c<0)
#   (3) Held-out + all pairs
#   (4) Dose-response curve + c_min
#
# PREREQUISITE: data/analysis/runD_v2/geometry_stage1/concept_directions.npz
#   (already exists from previous run of 60_)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/iia_failure_diagnosis
#   sbatch jobs/run_steering_validation.sbatch

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  Steering Validation (61b — specificity + symmetry)"
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
PROMPTS=data/prompts/physics_decay_type_probe_v2_train.jsonl

[ -f "$GEOM_NPZ" ] || { echo "ERROR: $GEOM_NPZ not found — run 60_ first"; exit 1; }
[ -f "$PROMPTS"  ] || { echo "ERROR: $PROMPTS not found";  exit 1; }
echo "Prerequisites verified."

date
python -u scripts/61b_steering_validation.py \
    --concept_npz   "$GEOM_NPZ" \
    --prompts       "$PROMPTS" \
    --model_name    Qwen/Qwen3-4B \
    --model_size    4b \
    --device        cuda \
    --layer_configs "24;18,24;22" \
    --c_grid        0.1 0.25 0.5 0.75 1.0 2.0 \
    --n_random      5 \
    --train_frac    0.6 \
    --flip_target   0.9 \
    --tau_ctrl      0.3 \
    --seed          0

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Read: data/analysis/iia_failure_diagnosis/steering_validation_summary.json (field 'verdict')"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/ data/analysis/iia_failure_diagnosis/"
