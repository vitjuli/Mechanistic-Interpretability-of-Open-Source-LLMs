#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ext-recon-grad
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ext-recon-grad-%j.out
#SBATCH --error=logs/ext-recon-grad-%j.out
#
# Extends scripts 51 and 53 from L10-L25 → L10-L35 (26 layers).
#
# Runtime estimate (A100-80GB):
#   Script 51 (L10-L35, 26 layers): ~3.5h  (extrapolated from 2:11 on 16 layers)
#   Script 53 (L10-L35, 26 layers): ~3.5h
#   Total: ~7h → 5h buffer within 12h limit
#
# Outputs:
#   data/analysis/recon_baseline/physics_decay_type_probe_v2/
#       recon_baseline_physics_decay_type_probe_v2_train.csv    ← OVERWRITES (extended)
#   data/analysis/gradient_sensitivity/physics_decay_type_probe_v2/
#       gradient_sensitivity_physics_decay_type_probe_v2_train.csv  ← OVERWRITES (extended)

set -uo pipefail
mkdir -p logs
echo "========================================================"
echo "  Extended recon baseline + gradient sensitivity (L10-L35)"
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
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"

# ── Script 51: Reconstruction noise baseline (L10-L35) ────────────────────────
echo "========================================================"
echo "  STEP 1: Script 51 — recon noise baseline"
echo "  Layers: ${LAYERS}"
echo "  Started: $(date)"
echo "========================================================"

python -u scripts/51_recon_iia_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --layers    $LAYERS \
    --output_dir data/analysis/recon_baseline

echo ""
echo "  Script 51 done: $(date)"
echo ""

# ── Script 53: Gradient sensitivity analysis (L10-L35) ───────────────────────
echo "========================================================"
echo "  STEP 2: Script 53 — gradient sensitivity"
echo "  Layers: ${LAYERS}"
echo "  Started: $(date)"
echo "========================================================"

python -u scripts/53_gradient_sensitivity_analysis.py \
    --behaviour   "$BEHAVIOUR" \
    --split       "$SPLIT" \
    --layers      $LAYERS \
    --output_dir  data/analysis/gradient_sensitivity

echo ""
echo "  Script 53 done: $(date)"
echo ""

echo "========================================================"
echo "  All steps complete: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/recon_baseline/ data/analysis/recon_baseline/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/gradient_sensitivity/ data/analysis/gradient_sensitivity/"
