#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j99-disa
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j99-%j.out
#SBATCH --error=logs/j99-%j.out
#
# 99 — Suppression disambiguation. Decides whether j97's L20 partial recovery
# (margin-flip 0.26 vs null 0.03 = 8×) is genuine concept-wake or a tautology
# (we erased the decision axis itself, not a separate competitor).
#
# Three guards run on the BASE format (n=149 failed-β, large sample, margin-metric
# robust to norm artefact of projection):
#   (1) Geometry: |cos(d_error, u)| and |cos(d_error, w_res)| per layer.
#       d_error ≈ u (cos > 0.6) ⇒ we erased the decision axis itself = tautology.
#   (2) α-control: do CORRECTLY-predicted α prompts also flip toward β under the
#       same projection? If yes ⇒ we just deflated α everywhere, not healed β.
#   (3) Placebo: build d_error from SHUFFLED failed/succ labels and rerun.
#       If placebo heals comparably ⇒ d_error is generic, not specifically the
#       failure-vs-success direction.
#
# Verdict per layer: SPECIFIC_REAL / TAUTOLOGY / MIXED.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/supp_disambig

echo "========================================================"
echo "  99 — suppression disambiguation (concept-wake vs tautology)"
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
echo "==== Suppression disambiguation run (L17 L20 L24) ===="
date
python -u scripts/99_suppression_disambiguation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 17 20 24 \
    --out_dir data/analysis/runD_v2/supp_disambig

echo ""
echo "==== Done: $(date) ===="
