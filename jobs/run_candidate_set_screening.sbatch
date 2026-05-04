#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-screen
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cand-screen-%j.out
#SBATCH --error=logs/cand-screen-%j.out

# ── Candidate-set behaviour screening ────────────────────────────────────────
# Evaluates 4 mini behaviours in a single model load and ranks them.
#
# Behaviours tested:
#   physics_parity_rule_mini       (32 prompts, ' even'/' odd')
#   physics_spin_statistics_mini   (24 prompts, ' boson'/' fermion')
#   physics_approximation_regime_mini (24 prompts, ' classical'/' relativistic')
#   physics_e1_selection_mini      (16 prompts, ' allowed'/' forbidden')
#
# Outputs:
#   data/results/baseline_<behaviour>_train.csv   (per-behaviour)
#   data/results/candidate_set_screening_summary.csv
#   data/results/candidate_set_screening_report.md
#   logs/cand-screen-<JOBID>.out
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*_mini_train.csv data/results/
#
# sbatch jobs/run_candidate_set_screening.sbatch

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
echo "  Candidate-set behaviour screening"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results

# ── Step 0: Token audit ───────────────────────────────────────────────────────
echo ""
echo "==== Token audit ===="
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
answers = [' even', ' odd', ' boson', ' fermion', ' classical', ' relativistic', ' allowed', ' forbidden']
print('  Token lengths for answer tokens:')
multi = []
for a in answers:
    n = len(tok.encode(a, add_special_tokens=False))
    status = 'OK' if n == 1 else f'WARN ({n} tokens)'
    print(f'    {repr(a)}: {n} token(s) — {status}')
    if n > 1:
        multi.append(a)
if multi:
    print(f'  [WARN] Multi-token answers: {multi}')
    print('  Length-normalised scoring will correct for this, but single-token preferred.')
else:
    print('  [OK] All answer tokens are single-token.')
" || true

# ── Step 1: Generate prompts ──────────────────────────────────────────────────
echo ""
echo "==== Generating screening prompts ===="
python -u scripts/32_generate_candidate_set_screening_prompts.py

# Verify all prompt files exist
for beh in physics_parity_rule_mini physics_spin_statistics_mini physics_approximation_regime_mini physics_e1_selection_mini; do
    f="data/prompts/${beh}_train.jsonl"
    [ -f "$f" ] && echo "  ${beh}: $(wc -l < $f) prompts — OK" \
      || { echo "  [FATAL] $f not found"; exit 1; }
done

# ── Step 2: Run screening baseline ───────────────────────────────────────────
echo ""
echo "████  SCREENING BASELINE  $(date)"
echo ""

python -u scripts/33_run_candidate_set_screening.py \
    --split train \
    --device cuda

RC=$?
echo ""
echo "Screening exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Screening failed"; exit 1; }

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Summary CSV:  data/results/candidate_set_screening_summary.csv"
echo "  Report:       data/results/candidate_set_screening_report.md"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*_mini_train.csv data/results/"
echo "========================================================"
