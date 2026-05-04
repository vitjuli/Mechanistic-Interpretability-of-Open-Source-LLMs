#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-r2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cand-r2-%j.out
#SBATCH --error=logs/cand-r2-%j.out

# ── Candidate-set behaviour screening — Round 2 ──────────────────────────────
# Round-2 improvements over round-1:
#   - Spin statistics: novel spin values (s=3/2, s=2 etc.) test pure rule application;
#     F4_hybrid (value + explicit integer label) isolates bottleneck step.
#   - Regime: Yes/No framing removes systematic ' classical' token prior that caused 50%.
#   - Conductor/insulator: new behaviour, single-token Yes/No, strong expected signal.
#
# Behaviours:
#   physics_spin_statistics_v2_mini  (32 prompts, ' boson'/' fermion')
#   physics_relativistic_regime_v2_mini (24 prompts, ' Yes'/' No')
#   physics_conductor_insulator_mini    (24 prompts, ' Yes'/' No')
#
# Outputs:
#   data/results/baseline_<behaviour>_train.csv
#   data/results/candidate_set_screening_summary.csv
#   data/results/candidate_set_screening_report.md
#   logs/cand-r2-<JOBID>.out
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*v2*_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_conductor*_train.csv data/results/
#
# sbatch jobs/run_candidate_set_round2.sbatch

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
echo "  Candidate-set screening — Round 2"
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
answers = [' boson', ' fermion', ' Yes', ' No']
print('  Round-2 answer tokens:')
for a in answers:
    n = len(tok.encode(a, add_special_tokens=False))
    status = 'OK (1-token)' if n == 1 else f'MULTI ({n} tokens, length-normalized)'
    print(f'    {repr(a)}: {n} token(s) — {status}')
" || true

# ── Generate prompts ─────────────────────────────────────────────────────────
echo ""
echo "==== Generating round-2 screening prompts ===="
python -u scripts/34_generate_candidate_set_round2_prompts.py

for beh in physics_spin_statistics_v2_mini physics_relativistic_regime_v2_mini physics_conductor_insulator_mini; do
    f="data/prompts/${beh}_train.jsonl"
    [ -f "$f" ] && echo "  ${beh}: $(wc -l < $f) prompts — OK" \
      || { echo "  [FATAL] $f not found"; exit 1; }
done

# ── Run screening baseline ───────────────────────────────────────────────────
echo ""
echo "████  ROUND-2 SCREENING BASELINE  $(date)"
echo ""

python -u scripts/33_run_candidate_set_screening.py \
    --behaviours \
        physics_spin_statistics_v2_mini \
        physics_relativistic_regime_v2_mini \
        physics_conductor_insulator_mini \
    --split train \
    --device cuda

RC=$?
echo ""
echo "Screening exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Round-2 screening failed"; exit 1; }

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Round 2 — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Report: data/results/candidate_set_screening_report.md"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*v2*_train.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_conductor*_train.csv data/results/"
echo "========================================================"
