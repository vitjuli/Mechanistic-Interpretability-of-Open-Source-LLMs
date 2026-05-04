#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-large
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cand-large-%j.out
#SBATCH --error=logs/cand-large-%j.out

# ── Candidate-state large-scale evaluation ────────────────────────────────────
# Generates ~311 prompts with experimental structure designed to test whether
# the model forms an internal candidate-set representation.
#
# Experimental subsets:
#   core (253):               diverse (set, filter) pairs × 5 families incl. F5_no_set
#   distractor_sensitivity (21): same filter/correct, varying distractor difficulty
#   counterfactual (15):      reordered, minimal, extended candidate sets
#   set_size (22):            same filter across 2-5 candidate set sizes
#
# Key analysis outputs:
#   H1 vs H2 test:            with-set vs no-set logprob_diff (is the set being used?)
#   Distractor sensitivity:   does margin decrease with harder distractors? (H2 evidence)
#   Set-size effect:          does adding more candidates change confidence?
#   Competition analysis:     which wrong candidates get elevated probability?
#
# All answer tokens confirmed ≤2 tokens with length normalization.
# Multi-candidate logprobs saved for every prompt (6 tokens evaluated per prompt).
#
# Outputs: data/results/candidate_set_large/
#   candidate_state_results.csv  — per-prompt with all candidate logprobs
#   candidate_state_summary.md   — evaluation summary
#   h1_vs_h2_results.csv         — H1 vs H2 comparison
#   distractor_sensitivity.csv   — difficulty curve
#   set_size_effect.csv          — set-size trend
#   competition_analysis.csv     — which distractors compete most
#   analysis_report.md           — recommendation memo
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ \
#     data/results/candidate_set_large/
#
# sbatch jobs/run_candidate_state_large.sbatch

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
echo "  Candidate-state large evaluation"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results/candidate_set_large

# ── Token audit ───────────────────────────────────────────────────────────────
echo ""
echo "==== Token audit ===="
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
pool = [' electron',' proton',' neutron',' photon',' positron',' muon']
print('  Global candidate pool token lengths:')
for t in pool:
    n = len(tok.encode(t, add_special_tokens=False))
    status = 'OK' if n == 1 else f'MULTI ({n} toks — length-normalized)'
    print(f'    {repr(t)}: {status}')
" || true

# ── Generate prompts ─────────────────────────────────────────────────────────
echo ""
echo "==== Generating large candidate-state dataset ===="
python -u scripts/38_generate_candidate_state_large.py

f="data/prompts/physics_particle_candidate_selection_train.jsonl"
[ -f "$f" ] && echo "  physics_particle_candidate_selection: $(wc -l < $f) prompts — OK" \
  || { echo "  [FATAL] Prompt file not found"; exit 1; }

# ── Run multi-candidate evaluation ──────────────────────────────────────────
echo ""
echo "████  CANDIDATE-STATE EVALUATION  $(date)"
echo ""

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour physics_particle_candidate_selection \
    --split train \
    --device cuda

RC=$?
echo "Evaluation exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Evaluation failed"; exit 1; }

# ── Analysis ─────────────────────────────────────────────────────────────────
echo ""
echo "████  ANALYSIS  $(date)"
echo ""

python -u scripts/40_analyze_candidate_state.py

RC=$?
echo "Analysis exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Analysis failed — check manually"; }

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
echo "    data/results/candidate_set_large/analysis_report.md"
echo "    data/results/candidate_set_large/h1_vs_h2_results.csv"
echo "    data/results/candidate_set_large/distractor_sensitivity.csv"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ \\"
echo "      data/results/candidate_set_large/"
echo "========================================================"
