#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=subspace-pilot
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/subspace-pilot-%j.out
#SBATCH --error=logs/subspace-pilot-%j.out
#
# Stage-2 subspace-intervention PILOT (~30 held-out within-family pairs).
# Decides Outcome 1/2/3/4: is α/β concept interventionally sufficient in a
# low-dim subspace, and under which definition (C / B / A)?
#
# PREREQUISITE: concept_directions.npz with (gbar, Sigma) from script 60_.
#   If it doesn't exist yet — see verification block below.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/iia_failure_diagnosis
#   sbatch jobs/run_subspace_pilot.sbatch

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  Subspace Intervention Pilot (Stage-2 Park-Veitch)"
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

# ── verify prerequisites + auto-run 60_ if concept_directions.npz missing ──
GEOM_NPZ=data/analysis/runD_v2/geometry_stage1/concept_directions.npz
CLUSTERS=data/analysis/runD_v2/clustering_full/cluster_labels.csv
PROMPTS=data/prompts/physics_decay_type_probe_v2_train.jsonl

[ -f "$CLUSTERS" ] || { echo "ERROR: $CLUSTERS not found"; exit 1; }
[ -f "$PROMPTS"  ] || { echo "ERROR: $PROMPTS not found";  exit 1; }

if [ ! -f "$GEOM_NPZ" ]; then
    echo "concept_directions.npz not found — running 60_ first (~1 min)..."
    python -u scripts/60_causal_inner_product_geometry.py --device cuda \
        || { echo "ERROR: script 60_ failed"; exit 1; }
fi
[ -f "$GEOM_NPZ" ] || { echo "ERROR: still no $GEOM_NPZ after 60_"; exit 1; }
echo "Prerequisites verified."

date
python -u scripts/61_subspace_interventions_pilot.py \
    --concept_npz    "$GEOM_NPZ" \
    --cluster_labels "$CLUSTERS" \
    --cluster_col    agglo_coimp_subgroup_k30 \
    --A3_cluster_col coimp_louvain \
    --A1_cluster     16 \
    --A3_cluster     13 \
    --prompts        "$PROMPTS" \
    --model_name     Qwen/Qwen3-4B \
    --model_size     4b \
    --device         cuda \
    --layer_configs  "18;22;24;18,24" \
    --r_values       1 3 8 11 \
    --n_pairs        30 \
    --train_frac     0.6 \
    --steer_sweep    -2 -1 -0.5 0.5 1 2 4 \
    --tau_flip       0.7 \
    --tau_ctrl       0.3 \
    --seed           0

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Read: data/analysis/iia_failure_diagnosis/subspace_pilot_summary.json (field 'verdict')"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/ data/analysis/iia_failure_diagnosis/"
