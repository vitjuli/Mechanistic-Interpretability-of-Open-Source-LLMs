#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_l22int
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/mc_b1_l22int-%j.out
#SBATCH --error=logs/mc_b1_l22int-%j.err
#
# multilingual_circuits_b1 — script 13: L22 causal intervention
#
# Scales L22_F108295 and L22_F32734 activations by alpha ∈ {2.0, 3.0} for
# the 18 incorrect FR prompts.  Tests whether L22 is a causal bottleneck:
#   - Type A: expect delta_logit > 0 (scaling helps), flips for large alpha
#   - Type B: expect delta_logit < 0 (scaling worsens, confirms causal role)
#
# Method: encode → feats[:, feat_idx] *= alpha → decode → patch L22 MLP output
# (mirrors ablation_zero from script 07/11, but scales instead of zeros)
#
# Requires: model + L22 transcoder.
# Total forward passes: 18 × (1 baseline + 2 alpha × 2 fwd each) = 90 passes ≈ 5 min.
#
# Key diagnostics to check in output:
#   "Target improved:  N/18"         ← for alpha=2.0
#   "Control improved: N/18"         ← control comparison (Fix 2)
#   "L22 vs control Δ: +X"           ← positive = L22 specific, not generic scaling
#   "Flipped:  N/18"                 ← expect >0 Type A for causal confirmation
#   "Dose-response: monotone"        ← Fix 1: gradual causal vs extreme forcing
#   "% active at baseline: XX%"      ← Fix 4: confirm features are non-zero
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_b1_13_l22_intervention.sbatch
#
# (Can run in parallel with script 12 — they are independent.)
#
# AFTER COMPLETION:
#   Sync outputs:
#     git add data/results/reasoning_traces/multilingual_circuits_b1/l22_intervention_results_train.json
#     git commit -m "script 13: L22 causal intervention — gateway bottleneck confirmed"

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — L22 Causal Intervention (Script 13)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

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

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train

# Verify prerequisites
ERROR_CASES="data/results/reasoning_traces/${BEHAVIOUR}/error_cases_${SPLIT}.json"
if [ ! -f "$ERROR_CASES" ]; then
    echo "ERROR: error_cases not found: $ERROR_CASES"
    echo "  Run scripts 10 + 11 (v2) first."
    exit 1
fi
echo "Input: $ERROR_CASES (OK)"

# ══════════════════════════════════════════════════════════════
# Script 13: L22 causal intervention
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Script 13: L22 Causal Intervention ===="
echo "  Targets: L22_F108295, L22_F32734"
echo "  Alphas:  2.0  3.0"
date

python -u scripts/13_l22_intervention.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --alphas 1.5 2.0 3.0

echo ""
echo "========================================"
echo "Script 13 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Output:"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/l22_intervention_results_${SPLIT}.json"
echo ""
echo "Hypothesis confirmed if:"
echo "  - n_improved >= 12/18 at alpha=2.0"
echo "  - n_type_b_worsened = 2/2"
echo "  - at least one Type-A prompt flipped to correct"
