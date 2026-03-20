#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_gate
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/mc_b1_gate-%j.out
#SBATCH --error=logs/mc_b1_gate-%j.err
#
# multilingual_circuits_b1 — BASELINE GATE (script 02 only)
#
# Run this BEFORE the full pipeline rerun to verify the template fix worked.
# FR templates were rewritten (B1-v2, 2026-03-20) to remove '"{word}"' pattern.
#
# ACCEPTANCE CRITERIA (must ALL pass before submitting mc_b1_02_09.sbatch):
#   [1] EN sign_accuracy >= 0.90
#   [2] FR sign_accuracy >= 0.75       (up from 0.67 with old templates)
#   [3] argmax_is_quote_rate == 0.00   (printed by the Python check below)
#   [4] No other dominant punctuation artifact in generated column
#
# IF GATE FAILS:
#   - Stop. Do NOT submit the full pipeline.
#   - Check the gate analysis printed at the end of this log.
#   - If [3] fails: quotes still present — check prompt file for remaining '"{word}"'
#   - If [2] fails but [3] passes: new artifact (article attraction, colon bias, etc.)
#     → inspect the 'generated' column in the CSV for dominant wrong tokens
#     → revise colon-ending templates T6/T7 in _ML_ANT_TEMPLATES_B1 accordingly
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1
#   wc -l data/prompts/multilingual_circuits_b1_train.jsonl   # expect 96
#   grep '"' data/prompts/multilingual_circuits_b1_train.jsonl | grep '"fr"' | head -5
#   # ↑ verify: no prompts contain the pattern "word" (with double-quotes around the word)
#   sbatch jobs/multilingual_circuits_b1_02_gate.sbatch
#
# AFTER COMPLETION:
#   Inspect logs/mc_b1_gate-<jobid>.out
#   Look for: "GATE: PASS" or "GATE: FAIL"
#   If PASS: sbatch jobs/multilingual_circuits_b1_02_09.sbatch
#

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Baseline Gate"
echo "Template fix B1-v2 (2026-03-20)"
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
echo ""

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train

# Verify prompt file exists and has correct count
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found."
    echo "  Run: python scripts/01_generate_prompts.py --behaviour ${BEHAVIOUR}"
    exit 1
fi
ACTUAL=$(wc -l < "$PROMPT_FILE")
echo "Prompt file: ${ACTUAL} lines (expect 96)"
if [ "$ACTUAL" -ne 96 ]; then
    echo "ERROR: Expected 96 prompts, got ${ACTUAL}"
    exit 1
fi

# Verify no FR prompts contain the quoted-word pattern
QUOTED=$(python3 -c "
import json
count = 0
with open('${PROMPT_FILE}') as f:
    for line in f:
        p = json.loads(line)
        if p.get('language') == 'fr' and ('\"' + p.get('word','') + '\"') in p['prompt']:
            count += 1
print(count)
")
echo "FR prompts with quoted word pattern: ${QUOTED} (must be 0)"
if [ "$QUOTED" -ne 0 ]; then
    echo "ERROR: FR templates still contain '\"word\"' pattern. git pull and regenerate."
    exit 1
fi

echo ""
echo "==== Script 02: Baseline evaluation ===="
date

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ── Post-baseline artifact gate ────────────────────────────────────────────────
echo ""
echo "==== Baseline Gate Analysis ===="
python3 - <<'PYEOF'
import json, csv, sys
from pathlib import Path
from collections import Counter

behaviour = "multilingual_circuits_b1"
split     = "train"
csv_path  = Path(f"data/results/baseline/{behaviour}/{behaviour}_{split}_results.csv")

if not csv_path.exists():
    print(f"ERROR: Baseline CSV not found: {csv_path}")
    sys.exit(1)

rows = list(csv.DictReader(csv_path.open()))
fr_rows = [r for r in rows if r.get("language") == "fr"]
en_rows = [r for r in rows if r.get("language") == "en"]

# EN accuracy
en_success  = sum(1 for r in en_rows if r.get("success","").lower() in ("true","1","yes"))
en_acc      = en_success / len(en_rows) if en_rows else 0.0

# FR accuracy (sign: correct_log_prob > incorrect_log_prob)
fr_success  = sum(1 for r in fr_rows if r.get("success","").lower() in ("true","1","yes"))
fr_acc      = fr_success / len(fr_rows) if fr_rows else 0.0

# argmax_is_quote: count FR failing prompts (success=False) that generated '"'
fr_fail     = [r for r in fr_rows if r.get("success","").lower() not in ("true","1","yes")]
quote_count = sum(1 for r in fr_fail if r.get("generated","").strip() == '"')
quote_rate  = quote_count / len(fr_fail) if fr_fail else 0.0

# Top predicted tokens for failing FR prompts
gen_counter = Counter(r.get("generated","").strip() for r in fr_fail)
top5 = gen_counter.most_common(5)

print(f"EN accuracy:          {en_acc:.3f}  ({en_success}/{len(en_rows)})  [gate: >= 0.90]")
print(f"FR accuracy:          {fr_acc:.3f}  ({fr_success}/{len(fr_rows)})  [gate: >= 0.75]")
print(f"FR failing prompts:   {len(fr_fail)}")
print(f"argmax_is_quote_rate: {quote_rate:.4f}  ({quote_count}/{len(fr_fail)})  [gate: == 0.00]")
print(f"Top-5 predicted tokens for failing FR prompts:")
for tok, cnt in top5:
    print(f"  {repr(tok):20s}: {cnt}/{len(fr_fail)}")
print()

gate_pass = True
failures  = []

if en_acc < 0.90:
    gate_pass = False
    failures.append(f"EN accuracy {en_acc:.3f} < 0.90")
if fr_acc < 0.75:
    gate_pass = False
    failures.append(f"FR accuracy {fr_acc:.3f} < 0.75")
if quote_rate > 0.0:
    gate_pass = False
    failures.append(f"argmax_is_quote_rate {quote_rate:.4f} > 0.00  (template artifact persists)")

if gate_pass:
    print("GATE: PASS")
    print("  → Safe to submit: sbatch jobs/multilingual_circuits_b1_02_09.sbatch")
else:
    print("GATE: FAIL")
    for f in failures:
        print(f"  ✗ {f}")
    if quote_rate > 0.0:
        print("  → STOP. Do not submit full pipeline. Fix FR templates further.")
    elif fr_acc < 0.75:
        print("  → Inspect top-5 tokens above. Check for article attraction (le/la/l') or")
        print("    colon bias. Consider removing T6/T7 (colon-ending templates).")
    sys.exit(1)
PYEOF

echo ""
echo "========================================"
echo "Baseline Gate complete."
echo "Job: $SLURM_JOB_ID"
date
