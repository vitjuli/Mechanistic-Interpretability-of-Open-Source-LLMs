#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=B1-CONCEPT
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/B1-%x-%j.out
#SBATCH --error=logs/B1-%x-%j.out
#
# B1 single-concept pipeline: 121 (precheck gate) → 119 (field dump) →
# 122 tier1 (broad sweep) → 122 tier2 (dense grid).
# Auto-stops on precheck failure (non-zero exit from 121).
#
# Usage: set CONCEPT env var before sbatch, OR edit the CONCEPT= line below.
#   sbatch --export=CONCEPT=grammar_number jobs/B1_concept_template.sbatch
#
# Expected runtime per concept: ~3-3.5 hours on A100.
#   121 ~5 min, 119 ~12 min, 122 tier1 ~2 hours, 122 tier2 ~1 hour

CONCEPT="${CONCEPT:-grammar_number}"
set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_${CONCEPT}

echo "========================================================"
echo "  B1 pipeline for concept: ${CONCEPT}"
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

# ── Stage 1: precheck gate (forward-only, ~5 min) ──────────────────────────
echo ""
echo "==== STAGE 1: 121 precheck gate ===="
python -u scripts/121_b1_precheck.py \
    --prompts data/prompts/B1_${CONCEPT}.jsonl \
    --concept ${CONCEPT} \
    --out_dir data/analysis/runD_v2/B1_${CONCEPT}
PRECHECK=$?
if [ $PRECHECK -ne 0 ]; then
    echo "==== PRECHECK FAILED (exit=$PRECHECK) — aborting pipeline ===="
    exit $PRECHECK
fi

# ── Stage 2: field dump (forward+backward, ~12 min) ────────────────────────
echo ""
echo "==== STAGE 2: 119 field dump ===="
python -u scripts/119_capture_field_dump.py \
    --prompts data/prompts/B1_${CONCEPT}.jsonl \
    --out_dir data/analysis/runD_v2/B1_${CONCEPT}/field_dump

# ── Stage 3: steering sweep tier 1 (broad survey, ~2 hours) ────────────────
echo ""
echo "==== STAGE 3: 122 tier 1 sweep ===="
python -u scripts/122_b1_steering_sweep.py \
    --prompts data/prompts/B1_${CONCEPT}.jsonl \
    --dump_dir data/analysis/runD_v2/B1_${CONCEPT}/field_dump \
    --tier 1 \
    --out_dir data/analysis/runD_v2/B1_${CONCEPT}

# ── Stage 4: steering sweep tier 2 (dense grid on auto-selected layers, ~1h)
echo ""
echo "==== STAGE 4: 122 tier 2 sweep ===="
python -u scripts/122_b1_steering_sweep.py \
    --prompts data/prompts/B1_${CONCEPT}.jsonl \
    --dump_dir data/analysis/runD_v2/B1_${CONCEPT}/field_dump \
    --tier 2 \
    --auto_select_from data/analysis/runD_v2/B1_${CONCEPT}/steering_sweep_tier1.csv \
    --out_dir data/analysis/runD_v2/B1_${CONCEPT}

echo ""
echo "==== Done: $(date) ===="
echo "Next on Mac (CPU):"
echo "  python scripts/118_null_calibration.py --dump_dir data/analysis/runD_v2/B1_${CONCEPT}/field_dump"
echo "  python scripts/116_flip_law_retro.py --dump_dir data/analysis/runD_v2/B1_${CONCEPT}/field_dump --measured_86 data/analysis/runD_v2/B1_${CONCEPT}/steering_sweep_tier1.csv"
