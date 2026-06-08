#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j104b-esc
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j104b-%j.out
#SBATCH --error=logs/j104b-%j.out
#
# 104b — Escalation steering on late layers L18-L35 at c ∈ {4, 8, 16, 32}.
# For each (ℓ, c, direction ∈ {w_res, u, random}):
#   * intact_flip
#   * Δmargin (actual) vs c·σ·‖g‖·cos(g, v̂) (predicted by calculus)
#   * cos(g, v̂)
#   * AUC-B@35
# Print GAP(w_res − random) and GAP(u − random); flag rows where random
# intact > 0.2 as [ARTIFACT ZONE] (perturbation breaks forced format by sheer norm).
#
# Definitive answer to "why no w_res flip at any c": if GAP(w_res − random) stays
# ~0 across c ∈ {4, 8, 16, 32}, it is GEOMETRY (g ⊥ w_res) not SCALING. The
# calculus prediction column confirms this analytically.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/escalation

echo "========================================================"
echo "  104b — escalation steering (c=4,8,16,32 × L18-L35)"
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

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
nvidia-smi || true
date

echo ""
echo "==== Escalation run (c=4,8,16,32; layers L18-L35; n_eval=24) ===="
date
python -u scripts/104b_escalation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --c_list 4 8 16 32 \
    --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 \
    --n_eval 24 \
    --out_dir data/analysis/runD_v2/escalation

echo ""
echo "==== Done: $(date) ===="
