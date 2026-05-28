#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=iia-diag-post
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/iia-diag-post-%j.out
#SBATCH --error=logs/iia-diag-post-%j.out
#
# IIA failure diagnosis — POST-CLUSTERING subset (requires Phase 3-4 done):
#   H2: joint patching of detector+executor cluster pairs (NEW clusters on n538)
#
# H3-reactive runs separately in jobs/run_h3_reactive_backup.sbatch
# (since it has a different runtime profile).
#
# BEFORE SUBMITTING:
#   - Phase 3-4 of runD pipeline must be complete
#   - cluster_semantics.json must reflect new clusters on 538 prompts
#   - Update --cluster_json arg in scripts/53_iia_diagnosis.py if needed

set -uo pipefail
mkdir -p logs data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  IIA Failure Diagnosis — POST-CLUSTERING (H2 on n538)"
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

# Sanity: verify new cluster file exists and looks fresh
CS=dashboard_probe/public/data/cluster_semantics.json
[ -f "$CS" ] || { echo "ERROR: $CS not found"; exit 1; }
echo "Cluster file: $CS  ($(ls -la $CS | awk '{print $5, $6, $7, $8}'))"

date
python -u scripts/53_iia_diagnosis.py \
    --mode h2_pairs \
    --behaviour physics_decay_type_probe_v2 \
    --split train \
    --max_pairs 200
echo "==== H2 done: $(date) ===="

echo ""
echo "==== All done: $(date) ===="
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/iia_failure_diagnosis/ data/analysis/iia_failure_diagnosis/"
