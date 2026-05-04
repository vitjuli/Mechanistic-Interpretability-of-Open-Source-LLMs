#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-state
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cand-state-%j.out
#SBATCH --error=logs/cand-state-%j.out

# ── Candidate-state pilot screening ──────────────────────────────────────────
# Tests whether the model can select a candidate from an explicit set using a filter.
# This is NOT binary label classification — the candidate set is given in the prompt.
#
# Structure: candidate_set + filter_property → selected_candidate
#
# Behaviours:
#   physics_particle_candidate_selection_mini (32 prompts)
#     "The options are proton, neutron, and electron. Which has negative charge? Answer: electron"
#     Candidate sets: {p,n,e}, {p,e,γ}, {e,e+,γ}
#     Filters: charge, lepton/boson, rest mass, decay role
#
#   physics_decay_product_selection_mini (32 prompts)
#     "In beta-minus decay (p, e, antineutrino produced). Which product is the lepton? Answer: electron"
#     Processes: beta-minus, beta-plus
#     Note: antineutrino/neutrino appear in prompt TEXT only — never as answer tokens
#
# Token audit:
#   ' electron':  1 token (expected)
#   ' proton':    1 token (expected)
#   ' neutron':   1 token (expected)
#   ' photon':    1 token (expected)
#   ' positron':  1 token (expected — confirm)
#   ' antineutrino': MULTI — prompt text only
#
# Report includes:
#   - family-level accuracy
#   - per-target-candidate accuracy
#   - per-filter-property accuracy
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*candidate*_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_decay_product*_train.csv data/results/
#
# sbatch jobs/run_candidate_state_pilot.sbatch

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
echo "  Candidate-state pilot screening"
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
answer_tokens  = [' electron', ' proton', ' neutron', ' photon', ' positron']
excluded_tokens = [' antineutrino', ' neutrino']
print('  Answer tokens (must be 1-token for clean scoring):')
for a in answer_tokens:
    n = len(tok.encode(a, add_special_tokens=False))
    status = 'OK' if n == 1 else f'WARN — {n} tokens (length-normalized)'
    print(f'    {repr(a)}: {status}')
print('  Excluded tokens (appear in prompt text only):')
for a in excluded_tokens:
    n = len(tok.encode(a, add_special_tokens=False))
    print(f'    {repr(a)}: {n} tokens — excluded from answer tokens as expected')
" || true

# ── Generate prompts ─────────────────────────────────────────────────────────
echo ""
echo "==== Generating candidate-state pilot prompts ===="
python -u scripts/36_generate_candidate_state_pilot_prompts.py

for beh in physics_particle_candidate_selection_mini physics_decay_product_selection_mini; do
    f="data/prompts/${beh}_train.jsonl"
    [ -f "$f" ] && echo "  ${beh}: $(wc -l < $f) prompts — OK" \
      || { echo "  [FATAL] $f not found"; exit 1; }
done

# ── Run screening baseline ───────────────────────────────────────────────────
echo ""
echo "████  CANDIDATE-STATE PILOT BASELINE  $(date)"
echo ""

python -u scripts/33_run_candidate_set_screening.py \
    --behaviours \
        physics_particle_candidate_selection_mini \
        physics_decay_product_selection_mini \
    --split train \
    --device cuda

RC=$?
echo ""
echo "Screening exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Pilot screening failed"; exit 1; }

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Candidate-state pilot — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Report: data/results/candidate_set_screening_report.md"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_screening_* data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_*candidate*_train.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_decay_product*_train.csv data/results/"
echo "========================================================"
