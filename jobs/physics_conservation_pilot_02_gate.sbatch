#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=pc_pilot_gate
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=logs/pc_pilot_gate-%j.out
#SBATCH --error=logs/pc_pilot_gate-%j.err
#
# physics_conservation_pilot — BASELINE GATE (script 02 only)
#
# Purpose: Validate that Qwen3-4B performs the physics_conservation task before
# committing to full pipeline. 36 prompts, 9 diagnostic concepts × 4 templates.
#
# ACCEPTANCE CRITERIA:
#   [1] Overall accuracy (success rate) >= 0.75
#   [2] Named conservative (C00 gravity, C01 spring): accuracy = 1.0  (easy baseline)
#   [3] Named non-conservative (C14 friction, C15 air): accuracy = 1.0 (easy baseline)
#   [4] Adversarial cases: REPORT ONLY (no hard threshold at pilot stage)
#       — key concern: does C22 (divergence-free) fool the model? (TRUE confusion)
#       — key concern: does C23 (magnitude-only) fool the model? (TRUE confusion)
#   [5] No tokenization artifacts (both ' True' and ' False' must be single tokens)
#
# IF GATE FAILS:
#   [1] fails: Overall < 0.75 — model doesn't understand the task at all. Check prompt format.
#   [2/3] fails: Named forces wrong — severe problem with prompt phrasing. Redesign templates.
#   [5] fails: Tokenization problem — check correct_token_len and incorrect_token_len in CSV.
#
# IF ADVERSARIAL accuracy < 0.5:
#   → This is expected (hard cases). Proceed with note that adversarial is genuinely hard.
#   → Consider adding more intermediate adversarial cases to the full 200-prompt dataset.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   python scripts/01_generate_prompts.py --behaviour physics_conservation_pilot
#   wc -l data/prompts/physics_conservation_pilot_train.jsonl   # expect 36
#   sbatch jobs/physics_conservation_pilot_02_gate.sbatch
#
# AFTER COMPLETION:
#   Inspect logs/pc_pilot_gate-<jobid>.out
#   Look for: "PILOT GATE: PASS" or "PILOT GATE: FAIL"
#   Also look for "ADVERSARIAL ANALYSIS" section — key diagnostics.
#   If PASS: sbatch jobs/physics_conservation_02_09.sbatch
#

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "physics_conservation — Pilot Baseline Gate"
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

BEHAVIOUR=physics_conservation_pilot
SPLIT=train
EXPECTED_N=36

# Verify prompt file
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found."
    echo "  Run: python scripts/01_generate_prompts.py --behaviour ${BEHAVIOUR}"
    exit 1
fi
ACTUAL=$(wc -l < "$PROMPT_FILE")
echo "Prompt file: ${ACTUAL} lines (expect ${EXPECTED_N})"
if [ "$ACTUAL" -ne "$EXPECTED_N" ]; then
    echo "ERROR: Expected ${EXPECTED_N} prompts, got ${ACTUAL}"
    exit 1
fi

# Quick check: True/False tokens are single tokens
echo ""
echo "==== Tokenization check ===="
python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
for s in [" True", " False"]:
    ids = tok.encode(s, add_special_tokens=False)
    print(f"  {repr(s):10s} → {ids} ({len(ids)} token{'s' if len(ids)>1 else ''})")
    if len(ids) != 1:
        print(f"  ERROR: '{s}' is not a single token! Behavior cannot use teacher-forced logprob.")
        sys.exit(1)
print("  Tokenization check: OK (both are single tokens)")
PYEOF

echo ""
echo "==== Script 02: Baseline evaluation ===="
date

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

# ── Pilot gate analysis ────────────────────────────────────────────────────────
echo ""
echo "==== Pilot Gate Analysis ===="
python3 - <<'PYEOF'
import csv, sys
from pathlib import Path
from collections import defaultdict, Counter

behaviour = "physics_conservation_pilot"
split     = "train"
csv_path  = Path(f"data/results/baseline_{behaviour}_{split}.csv")

if not csv_path.exists():
    print(f"ERROR: Baseline CSV not found: {csv_path}")
    sys.exit(1)

rows = list(csv.DictReader(csv_path.open()))
n = len(rows)

def is_success(r):
    return str(r.get("success","")).lower() in ("true","1","yes")

def logdiff(r):
    try: return float(r.get("logprob_diff_normalized", 0))
    except: return 0.0

# ── Overall accuracy ──
overall_success = sum(1 for r in rows if is_success(r))
overall_acc = overall_success / n
print(f"\n{'─'*60}")
print(f"OVERALL ACCURACY:  {overall_acc:.3f}  ({overall_success}/{n})")
print(f"{'─'*60}")

# ── By concept family ──
print("\n── By concept_family ──")
by_family = defaultdict(list)
for r in rows:
    by_family[r.get("concept_family","unknown")].append(r)
for fam in sorted(by_family.keys()):
    rs = by_family[fam]
    acc = sum(1 for r in rs if is_success(r)) / len(rs)
    avg_diff = sum(logdiff(r) for r in rs) / len(rs)
    print(f"  {fam:32s}: {acc:.2f}  ({sum(is_success(r) for r in rs)}/{len(rs)})  "
          f"avg_logdiff={avg_diff:+.3f}")

# ── By difficulty ──
print("\n── By difficulty ──")
by_diff = defaultdict(list)
for r in rows:
    by_diff[r.get("difficulty","unknown")].append(r)
for diff in ["easy","medium","hard"]:
    rs = by_diff.get(diff, [])
    if rs:
        acc = sum(1 for r in rs if is_success(r)) / len(rs)
        print(f"  {diff:8s}: {acc:.2f}  ({sum(is_success(r) for r in rs)}/{len(rs)})")

# ── By template_idx ──
print("\n── By template_idx ──")
by_tidx = defaultdict(list)
for r in rows:
    by_tidx[r.get("template_idx","?")].append(r)
TIDX_NAMES = {"0": "T0 (direct path)",
              "3": "T3 (potential exist)",
              "4": "T4 (conservative label)",
              "5": "T5 (circulation integral)"}
for tidx in ["0","3","4","5"]:
    rs = by_tidx.get(tidx, [])
    if rs:
        acc = sum(1 for r in rs if is_success(r)) / len(rs)
        print(f"  {TIDX_NAMES.get(tidx, tidx):30s}: {acc:.2f}  "
              f"({sum(is_success(r) for r in rs)}/{len(rs)})")

# ── Adversarial deep-dive ──
print("\n── Adversarial near-miss cases ──")
adv_families = {"adversarial_divergence": "C22 divergence-free (∇·F=0)",
                "adversarial_magnitude":  "C23 magnitude-only",
                "adversarial_pathdep":    "C24 path-history"}
for fam, label in adv_families.items():
    rs = by_family.get(fam, [])
    if not rs:
        print(f"  {label}: no prompts found")
        continue
    acc = sum(1 for r in rs if is_success(r)) / len(rs)
    print(f"\n  {label}:  accuracy={acc:.2f}  ({sum(is_success(r) for r in rs)}/{len(rs)})")
    for r in sorted(rs, key=lambda x: x.get("template_idx","0")):
        ok = "✓" if is_success(r) else "✗"
        gen = r.get("generated","?")
        nd = logdiff(r)
        print(f"    T{r.get('template_idx','?')} [{ok}] gen='{gen}'  logdiff={nd:+.3f}  "
              f"{r['prompt'][:60]}...")

# ── Failures ──
fails = [r for r in rows if not is_success(r)]
print(f"\n── Failed prompts ({len(fails)}/{n}) ──")
for r in fails:
    fam = r.get("concept_family","?")
    tidx = r.get("template_idx","?")
    gen = r.get("generated","?")
    nd = logdiff(r)
    print(f"  [{fam:25s}] T{tidx}  gen='{gen}'  logdiff={nd:+.3f}")
    print(f"    {r['prompt'][:80]}...")

# ── Token length check ──
print("\n── Token length check (must be 1 for both) ──")
tl_issues = [r for r in rows if
    str(r.get("correct_token_len","1")) != "1" or
    str(r.get("incorrect_token_len","1")) != "1"]
if tl_issues:
    print(f"  WARNING: {len(tl_issues)} prompts have multi-token answers!")
    for r in tl_issues[:3]:
        print(f"    {r['prompt'][:60]} | "
              f"correct_len={r.get('correct_token_len')} "
              f"incorrect_len={r.get('incorrect_token_len')}")
else:
    print("  All prompts: single-token answers ✓")

# ── Gate decision ──
print(f"\n{'═'*60}")
gate_pass = True
failures_list = []

# Easy cases must be perfect
easy_rows = by_diff.get("easy", [])
easy_acc = sum(1 for r in easy_rows if is_success(r)) / len(easy_rows) if easy_rows else 0.0

if overall_acc < 0.75:
    gate_pass = False
    failures_list.append(f"Overall accuracy {overall_acc:.3f} < 0.75")

if easy_acc < 1.0:
    gate_pass = False
    failures_list.append(f"Easy accuracy {easy_acc:.3f} < 1.0  (named forces wrong — prompt format problem)")

if tl_issues:
    gate_pass = False
    failures_list.append(f"{len(tl_issues)} prompts have multi-token answers (tokenization artifact)")

adv_rows = [r for r in rows if r.get("concept_family","").startswith("adversarial")]
adv_acc = sum(1 for r in adv_rows if is_success(r)) / len(adv_rows) if adv_rows else 0.0

if gate_pass:
    print("PILOT GATE: PASS")
    print(f"  Overall={overall_acc:.3f}  Easy={easy_acc:.3f}  Adversarial={adv_acc:.3f}")
    if adv_acc < 0.5:
        print(f"  NOTE: Adversarial accuracy {adv_acc:.3f} < 0.5 — model struggles with near-misses.")
        print(f"        This is expected (hard cases) but confirms the adversarial design is working.")
        print(f"        → Proceed with full pipeline. These will be the most interesting prompts.")
    else:
        print(f"  NOTE: Adversarial accuracy {adv_acc:.3f} — model handles near-misses well.")
        print(f"        → Strengthen adversarial cases in the full 200-prompt dataset.")
    print(f"  → Next: sbatch jobs/physics_conservation_02_09.sbatch")
else:
    print("PILOT GATE: FAIL")
    for f in failures_list:
        print(f"  ✗ {f}")
    sys.exit(1)
PYEOF

echo ""
echo "========================================"
echo "Pilot baseline gate complete."
echo "Job: $SLURM_JOB_ID"
date
