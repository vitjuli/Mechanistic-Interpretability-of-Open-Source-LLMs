#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j96-inj
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j96-%j.out
#SBATCH --error=logs/j96-%j.out
#
# 96 — Injection contest: concept (w_res Fisher) vs surface (diff-of-means between
# process-framed and property-framed prompts) vs random null. Inject at layers
# L14/17/20/24, strengths 2σ/4σ/8σ, on BOTH groups:
#   (a) all failed-β prompts (149)
#   (b) property-framed subset (β-recall ≈ 0.27)
# Measure margin-flip AND intact-flip (whether top-1 actually becomes ' beta').
#
# Verdicts:
#   * surface heals (above null) & concept does not → "model listens to form, not meaning"
#   * both move → check |cos(w_res, d_surf)| geometry to detect tautology
#   * neither beats null → injection doesn't fix the failure mode at these strengths
#
# Tautology guard built-in: prints |cos(w_res, d_surf)| per layer.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/injection_contest

echo "========================================================"
echo "  96 — injection contest (concept vs surface vs null)"
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

# ── smoke (3-4 min) ───────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L17, alpha=4, n_random=2) ===="
date
python -u scripts/96_injection_contest.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 17 --alphas 4 --n_random 2 \
    --out_dir data/analysis/runD_v2/injection_contest_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (25-35 min) ──────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (L14/17/20/24, alphas 2/4/8) ===="
date
python -u scripts/96_injection_contest.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/injection_contest

echo ""
echo "==== Done: $(date) ===="
