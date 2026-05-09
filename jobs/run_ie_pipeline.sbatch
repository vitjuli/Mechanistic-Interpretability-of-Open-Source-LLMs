#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-pipeline
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ie-pipeline-%j.out
#SBATCH --error=logs/ie-pipeline-%j.out

# sbatch jobs/run_ie_pipeline.sbatch
# RUNS: 04 (features L10-L35) → 06 (attribution graph) → 07 (interventions)
# REQUIRES: baseline gate passed (sign_accuracy >= 0.85)
# LAYERS: L10-L35 (27 layers) — captures early form, mid-domain, late abstract phases

set -euo pipefail
BEHAVIOUR=physics_intensive_extensive_v1
N_PROMPTS=200
LAYERS=$(seq 10 35 | tr '\n' ' ')

cd /rds/user/iv294/hpc-work/thesis/project
module purge
module load rhel8/default-amp 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  IE Full Pipeline — ${BEHAVIOUR}"
echo "  N_PROMPTS=${N_PROMPTS}  LAYERS=L10-L35"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -4 || true

# ── Stage 1: Feature extraction ───────────────────────────────────────────────
echo ""
echo "── Stage 1: Extract transcoder features (L10-L35) ──"
python -u scripts/04_extract_transcoder_features.py \
    --behaviour    "$BEHAVIOUR" \
    --split        train \
    --model_size   4b \
    --layers       $LAYERS \
    --top_k        50 \
    --save_full_acts

echo "Feature extraction done: $(date)"

# ── Stage 2: Attribution graph ────────────────────────────────────────────────
echo ""
echo "── Stage 2: Build attribution graph ──"
python -u scripts/06_build_attribution_graph.py \
    --behaviour         "$BEHAVIOUR" \
    --split             train \
    --model_size        4b \
    --n_prompts         $N_PROMPTS \
    --layers            $LAYERS \
    --vw_threshold      0.03 \
    --graph_node_mode   role_aware \
    --top_k_per_layer   10 \
    --activation_weighted \
    --output_suffix     _roleaware

echo "Attribution graph done: $(date)"

# ── Stage 3: Interventions ────────────────────────────────────────────────────
echo ""
echo "── Stage 3: Run interventions ──"
python -u scripts/07_run_interventions.py \
    --behaviour     "$BEHAVIOUR" \
    --split         train \
    --graph_suffix  _roleaware \
    --graph_n_prompts $N_PROMPTS

echo ""
echo "── Pipeline complete: $(date) ──"
echo "Results: data/results/attribution_graphs/${BEHAVIOUR}/"
echo "         data/results/interventions/${BEHAVIOUR}/"
echo ""
echo "Next: sbatch jobs/run_ie_analysis.sbatch"
echo "========================================================"
