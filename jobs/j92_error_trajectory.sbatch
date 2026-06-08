#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j92-etraj
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j92-%j.out
#SBATCH --error=logs/j92-%j.out
#
# 92 — Error-formation trajectory. Logit-lens-based diagnosis of WHERE the wrong α/β
# answer commits across layers, for failed-β vs succeeded-β vs correct-α groups.
# No transcoders; only the base model + tokenizer. ~10-15 min real on A100.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/error_trajectory

echo "========================================================"
echo "  92 — error-formation trajectory (logit-lens + Fisher)"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
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

# transcoders not needed here, but keep PYTHONPATH for uniformity
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
nvidia-smi || true
date

# ── smoke (~2 min) ─────────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (batch_size=8) ===="
date
python -u scripts/92_error_trajectory.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --batch_size 8 \
    --out_dir data/analysis/runD_v2/error_trajectory_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~10-15 min) ─────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN ===="
date
python -u scripts/92_error_trajectory.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/error_trajectory

echo ""
echo "==== Done: $(date) ===="
