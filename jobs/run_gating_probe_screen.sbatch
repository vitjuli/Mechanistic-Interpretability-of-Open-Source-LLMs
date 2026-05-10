#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=gating-screen
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gating-screen-%j.out
#SBATCH --error=logs/gating-screen-%j.out

# sbatch jobs/run_gating_probe_screen.sbatch
#
# Full gating-probe screening pipeline:
#   S0: generate prompts (no GPU — already done locally)
#   S1: behavioural baseline — logp(' Yes'/' No') for all 420 prompts (~20 min)
#   S2: analyse results — all metrics, ranking, CRS, report
#   S3: representation probe — hidden-state decodability by layer (~40 min)
#   S4: finalise report
#
# Model: Qwen/Qwen3-4B BASE throughout (logprob + hidden-state consistent)

set -euo pipefail

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
echo "  Gating Probe Screen"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -6 || true
echo ""

# ── S0: Ensure prompts exist ──────────────────────────────────────────────────
if [ ! -f data/prompts/gating/gating_probe_v1.jsonl ]; then
    echo "── S0: Generate prompts ─────────────────────────────"
    python -u scripts/80_generate_gating_probe_prompts.py
    stamp
else
    echo "── S0: Prompts already exist ($(wc -l < data/prompts/gating/gating_probe_v1.jsonl) lines)"
fi
echo ""

# ── S1: Behavioural baseline ──────────────────────────────────────────────────
echo "── S1: Behavioural baseline (420 prompts) ───────────"
python -u scripts/81_run_gating_probe_baseline.py \
    --device cuda \
    --dtype  bfloat16
stamp

echo ""

# ── S2: Analysis + metrics ────────────────────────────────────────────────────
echo "── S2: Analysis + CRS ranking ───────────────────────"
python -u scripts/82_analyze_gating_probe_results.py
stamp

echo ""

# ── S3: Representation probe (families A,F,G,H — allow/block) ────────────────
echo "── S3: Representation probe (hidden-state) ──────────"
python -u scripts/83_gating_representation_probe.py \
    --device  cuda \
    --dtype   bfloat16 \
    --families A,F,G,H
stamp

echo ""

# ── S4: Append probe results to report ────────────────────────────────────────
python3 - << 'APPEND_REPORT'
import json
from pathlib import Path

summary_path = Path("data/results/gating_probe_v1/probe_summary.json")
report_path  = Path("docs/gating_probe_screen_results.md")
xfer_path    = Path("data/results/gating_probe_v1/cross_family_transfer.csv")

if not summary_path.exists():
    print("probe_summary.json not found; skipping report append")
    exit(0)

with open(summary_path) as f:
    s = json.load(f)

import pandas as pd
xfer_df = pd.read_csv(xfer_path) if xfer_path.exists() else pd.DataFrame()

lines = [
    "",
    "---",
    "",
    "## Representation Probe Results (Families A, F, G, H)",
    "",
    f"- **Best probe layer**: L{s['best_probe_layer']} (CV = {s['best_probe_cv']:.4f})",
    f"- **Gate decodable from**: L{s['decodable_from_layer']}",
    f"- **Mean ARI(gate_label)**: {s['mean_ari_gate']:.4f}",
    f"- **Mean ARI(wording_family)**: {s['mean_ari_wording']:.4f}",
    f"- **Degenerate layers excluded**: {s['n_degenerate_layers']}",
    "",
    "### Cross-Family Transfer (train A, test F/G/H)",
    "",
    "| Train | Test | Probe Layer | Transfer Acc | n_train | n_test |",
    "|-------|------|-------------|-------------|---------|--------|",
]
for t in s.get("cross_family_transfer", []):
    lines.append(
        f"| {t['train_family']} | {t['test_family']} | L{t['probe_layer']} | "
        f"{t['transfer_acc']:.4f} | {t['n_train']} | {t['n_test']} |"
    )

if s["best_probe_cv"] > 0.80 and any(t["transfer_acc"] > 0.70 for t in s.get("cross_family_transfer",[])):
    conclusion = "**Domain-general gate**: gate direction transfers across physical domains."
elif s["best_probe_cv"] > 0.70:
    conclusion = "**Partially general gate**: representable but cross-domain transfer limited."
else:
    conclusion = "**Domain-specific gate**: gate is not strongly linearly encoded."

lines += ["", f"**Conclusion**: {conclusion}", ""]

existing = report_path.read_text() if report_path.exists() else ""
report_path.write_text(existing + "\n".join(lines))
print(f"Probe results appended to {report_path}")
APPEND_REPORT

stamp

ELAPSED=$(( ($(date +%s) - TSTART) / 60 ))
echo ""
echo "========================================================"
echo "  COMPLETE — ${ELAPSED}m total — $(date)"
echo ""
echo "  Results: data/results/gating_probe_v1/"
echo "  Report:  docs/gating_probe_screen_results.md"
echo ""
echo "  Sync:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/gating_probe_v1/ data/results/gating_probe_v1/"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/docs/gating_probe_screen_results.md docs/"
echo "========================================================"
