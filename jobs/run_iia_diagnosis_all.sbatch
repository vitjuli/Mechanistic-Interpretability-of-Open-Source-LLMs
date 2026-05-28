#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=iia-diag-now
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/iia-diag-now-%j.out
#SBATCH --error=logs/iia-diag-now-%j.out
#
# IIA failure diagnosis — RUN-NOW subset (independent of new clustering):
#   H1: patch top-K circuit features (from n538 attribution graph)
#   H4: early-only vs late-only layer split
#
# H2 (cluster pairs) and H3-reactive (cluster ablation) require the NEW
# clusters from Phase 3-4 (semantics scripts 19→22→23→26). Run those
# AFTER new cluster_semantics.json is generated.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/iia_failure_diagnosis
#   # The circuit_features_for_h1.json file must exist locally (committed)
#   sbatch jobs/run_iia_diagnosis_all.sbatch

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  IIA Failure Diagnosis — H1 + H2 + H4"
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

# Verify circuit features file
CF=/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json
[ -f "$CF" ] || { echo "ERROR: $CF not found — commit + push locally first"; exit 1; }
echo "Circuit features verified."

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== H1: patch top-K circuit features ===="
date
python -u scripts/53_iia_diagnosis.py \
    --mode h1_circuit \
    --behaviour physics_decay_type_probe_v2 \
    --split train \
    --max_pairs 200
echo "==== H1 done: $(date) ===="

# H2 SKIPPED — waits for new clusters from Phase 3-4 (semantics scripts).
# Will be in a separate sbatch jobs/run_iia_diagnosis_post_clustering.sbatch.

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== H4: layer split (early vs late only) ===="
date
python -u scripts/53_iia_diagnosis.py \
    --mode h4_layer_split \
    --behaviour physics_decay_type_probe_v2 \
    --split train \
    --max_pairs 200
echo "==== H4 done: $(date) ===="

echo ""
echo "==== All done: $(date) ===="
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/ data/analysis/iia_failure_diagnosis/"
