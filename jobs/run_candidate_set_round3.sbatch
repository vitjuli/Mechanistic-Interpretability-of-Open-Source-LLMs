#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-r3
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cand-r3-%j.out
#SBATCH --error=logs/cand-r3-%j.out

# ── Candidate-set screening — Round 3 ────────────────────────────────────────
# Targeted redesigns based on rounds 1-2 diagnostics:
#
#   physics_spin_statistics_v3_mini (32 prompts, ' boson'/' fermion')
#     - F1 fix: replaced pion → gluon (force carrier, unambiguously a boson)
#     - F2 fix: restricted to s∈{0,1} bosons and s∈{1/2,3/2} fermions only
#     - F3/F4 unchanged (both 100% in round-2)
#     - Key comparison: F2 (no label) vs F4 (explicit "an integer" label)
#     - Expected: 90-96% sign accuracy
#
#   chemistry_acid_base_mini (32 prompts, ' acid'/' base')
#     - New domain; single-token symmetric answers; no prior bias
#     - F3_proton_transfer (verbal rule) expected ~100%
#     - Expected: 90%+ sign accuracy
#
#   physics_charge_sign_mini (32 prompts, ' positive'/' negative')
#     - New domain; single-token symmetric answers
#     - F1=named particle recall; F4=charge algebra
#     - Expected: 90%+ sign accuracy
#
# Token audit:
#   ' boson'/' fermion': 2-token each (confirmed); length-normalised
#   ' acid'/' base'/' positive'/' negative': check at runtime (expected 1-token)
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_spin_statistics_v3*_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_chemistry_acid_base*_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_charge_sign*_train.csv data/results/
#
# sbatch jobs/run_candidate_set_round3.sbatch

set -euo pipefail

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Candidate-set screening — Round 3"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results

# ── Token audit ───────────────────────────────────────────────────────────────
echo ""
echo "==== Token audit ===="
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
tokens = [' acid', ' base', ' positive', ' negative', ' boson', ' fermion']
print('  Round-3 answer tokens:')
all_ok = True
for a in tokens:
    n = len(tok.encode(a, add_special_tokens=False))
    status = 'OK (1-token)' if n == 1 else f'MULTI ({n} tokens, length-normalized)'
    if n > 1: all_ok = False
    print(f'    {repr(a)}: {n} token(s) — {status}')
if all_ok:
    print('  [OK] All answer tokens are single-token.')
else:
    print('  [WARN] Multi-token answers present (length normalization will be used).')
" || true

# ── Generate prompts ─────────────────────────────────────────────────────────
echo ""
echo "==== Generating round-3 screening prompts ===="
python -u scripts/35_generate_candidate_set_round3_prompts.py

for beh in physics_spin_statistics_v3_mini chemistry_acid_base_mini physics_charge_sign_mini; do
    f="data/prompts/${beh}_train.jsonl"
    [ -f "$f" ] && echo "  ${beh}: $(wc -l < $f) prompts — OK" \
      || { echo "  [FATAL] $f not found"; exit 1; }
done

# ── Run screening baseline ───────────────────────────────────────────────────
echo ""
echo "████  ROUND-3 SCREENING BASELINE  $(date)"
echo ""

python -u scripts/33_run_candidate_set_screening.py \
    --behaviours \
        physics_spin_statistics_v3_mini \
        chemistry_acid_base_mini \
        physics_charge_sign_mini \
    --split train \
    --device cuda

RC=$?
echo ""
echo "Screening exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Round-3 screening failed"; exit 1; }

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Round 3 — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Report: data/results/candidate_set_screening_report.md"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_spin_statistics_v3*_train.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_chemistry_acid_base*_train.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_charge_sign*_train.csv data/results/"
echo "========================================================"
