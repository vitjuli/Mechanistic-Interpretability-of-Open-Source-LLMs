#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-full
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ie-full-%j.out
#SBATCH --error=logs/ie-full-%j.out

# sbatch jobs/run_ie_full.sbatch
#
# Full pipeline for physics_intensive_extensive_v1:
#   S0: generate prompts
#   S1: baseline gate   (~30–60 min, Instruct model)
#   S2: feature extract  (~3–4h,    BASE model + transcoders, layers L10–L35)
#   S3: attribution graph (~1h,     BASE model)
#   S4: interventions    (~2h,      BASE model)
#   S5: layer transition (~1h,      BASE model, all 37 layers)
#   S6: cluster analysis (~1h,      BASE model, cross-domain transfer)
#   S7: cluster ablation (~1h,      BASE model, transcoders)
#
# GATE: if S1 sign_accuracy < 0.65 the job exits early.
# If 0.65 ≤ sign_accuracy < 0.85 a warning is printed but the pipeline continues.

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

TSTART=$(date +%s)
stamp() { echo "  [$(( ($(date +%s) - TSTART) / 60 ))m elapsed]  $(date)"; }

echo "========================================================"
echo "  IE Full Pipeline — ${BEHAVIOUR}"
echo "  N_PROMPTS=${N_PROMPTS}  LAYERS=L10-L35"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -6 || true
echo ""

# ── S0: Generate prompts ──────────────────────────────────────────────────────
echo "── S0: Generate prompts ──────────────────────"
python -u scripts/63_generate_ie_prompts.py
stamp

# ── S1: Baseline gate ─────────────────────────────────────────────────────────
echo ""
echo "── S1: Baseline gate ─────────────────────────"
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     train
stamp

# Gate check: read saved CSV and verify sign accuracy
python3 - <<GATE
import sys, glob, pandas as pd
from pathlib import Path

behaviour = "$BEHAVIOUR"
# Script 02 saves to data/results/baseline/{behaviour}/
candidates = sorted(Path("data/results/baseline").rglob(f"*{behaviour}*train*.csv"))
if not candidates:
    print("GATE: No baseline CSV found — cannot verify accuracy. Continuing.")
    sys.exit(0)

df = pd.read_csv(candidates[-1])
# Column may be 'sign_correct', 'correct', or contain logprob_diff info
acc_col = next((c for c in ['sign_correct','correct','sign_accuracy'] if c in df.columns), None)
if acc_col is None:
    print(f"GATE: Columns in CSV: {list(df.columns)}")
    print("GATE: Cannot find accuracy column — continuing.")
    sys.exit(0)

acc = float(df[acc_col].mean())
print(f"GATE: sign_accuracy = {acc:.3f}")

if acc < 0.65:
    print("GATE FAILED: accuracy < 0.65 — the ' intensive'/' extensive' token format")
    print("likely does not work with this model configuration.")
    print("Fix: regenerate prompts with Yes/No format (edit scripts/63_generate_ie_prompts.py).")
    sys.exit(1)
elif acc < 0.85:
    print(f"GATE WARNING: accuracy {acc:.3f} < 0.85 target. Pipeline continues.")
    print("Consider removing hard properties before the mechanistic run.")
else:
    print(f"GATE PASSED: accuracy {acc:.3f} >= 0.85")
GATE

echo ""
echo "── S2: Feature extraction (L10-L35, top_k=50) ───"
python -u scripts/04_extract_transcoder_features.py \
    --behaviour    "$BEHAVIOUR" \
    --split        train \
    --model_size   4b \
    --layers       $LAYERS \
    --top_k        50 \
    --save_full_acts
stamp

echo ""
echo "── S3: Attribution graph (n=${N_PROMPTS}, vw=0.03, top_k_per_layer=10) ──"
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
stamp

echo ""
echo "── S4: Interventions ─────────────────────────"
python -u scripts/07_run_interventions.py \
    --behaviour       "$BEHAVIOUR" \
    --split           train \
    --graph_suffix    _roleaware \
    --graph_n_prompts $N_PROMPTS
stamp

echo ""
echo "── S5: Layer transition ARI + transfer probe ─"
python -u scripts/70_ie_layer_transition.py \
    --behaviour "$BEHAVIOUR" \
    --device    cuda
stamp

echo ""
echo "── S6: Cluster analysis + cross-domain transfer ──"
python -u scripts/71_ie_cluster_analysis.py \
    --behaviour  "$BEHAVIOUR" \
    --device     cuda \
    --k          6
stamp

echo ""
echo "── S7: Cluster interventions ─────────────────"
python -u scripts/72_ie_cluster_interventions.py \
    --behaviour  "$BEHAVIOUR" \
    --device     cuda \
    --k          6 \
    --n_prompts  100
stamp

ELAPSED=$(( ($(date +%s) - TSTART) / 60 ))
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE — ${ELAPSED}m total — $(date)"
echo "  Results:"
echo "    data/results/baseline/${BEHAVIOUR}/"
echo "    data/results/attribution_graphs/${BEHAVIOUR}/"
echo "    data/results/interventions/${BEHAVIOUR}/"
echo "    data/results/abstraction_ie/${BEHAVIOUR}/"
echo ""
echo "  Sync locally:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/ data/results/"
echo "========================================================"
