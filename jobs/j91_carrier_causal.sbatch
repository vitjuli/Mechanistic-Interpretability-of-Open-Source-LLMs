#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j91-carc
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j91-%j.out
#SBATCH --error=logs/j91-%j.out
#
# 91 — Carrier causal validation. The deciding test after j88e2's geometric finding
# that the transcoder dictionary aligns with u above the proper decoder-aware null at
# L21-L25. Here we ask whether u-aligned features actually CARRY the decision:
#   (geom)  apples-to-apples max|cos(d_f, u)| vs max|cos(d_f, w_res)| against ONE
#           decoder-null — closes the w_res-vs-u methodological caveat in one row.
#   (a)     do top-K carrier features fire on the prompts? do they encode α/β?
#   (b)     does error-free ablation of the carrier set move margin ABOVE the
#           frequency-matched random-feature null? (THE causal test)
#   (c)     does the ablation contribution direction align with u?
#
# Three presentable outcomes — see script docstring.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/carrier_causal

echo "========================================================"
echo "  91 — carrier causal validation (geom + activate + ablate)"
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

# transcoder_loader.py lives at src/transcoder/
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

echo "Python: $(which python)"
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

# ── smoke (~3-4 min) ──────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L23 L24, topk=10, 5 targets, 2 controls) ===="
date
python -u scripts/91_carrier_causal_validation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 23 24 \
    --topk 10 --n_control 2 --max_targets 5 \
    --out_dir data/analysis/runD_v2/carrier_causal_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run (~30-40 min) ─────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (L21-L25, topk=30, 80 targets, 8 controls) ===="
date
python -u scripts/91_carrier_causal_validation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 21 22 23 24 25 \
    --topk 30 --n_control 8 --max_targets 80 \
    --out_dir data/analysis/runD_v2/carrier_causal

echo ""
echo "==== Done: $(date) ===="
