#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=egate-s8
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/egate-s8-%j.out
#SBATCH --error=logs/egate-s8-%j.out
#
# physics_entropy_gate_v1 — S8 only (minimal-pair patching, rerun after hook fix)
#
# Fix: forward hooks in get_residuals and make_patch_hook now handle both
# tuple and bare-Tensor output from Qwen3DecoderLayer (newer transformers).
#
# Prereqs: all of S0-S7 already complete from job 29235566.
# Runtime: ~30 min (40 pairs × 9 layers × 2 directions).

set -uo pipefail

mkdir -p logs

echo "========================================================"
echo "  physics_entropy_gate_v1 — S8 minimal-pair patching"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

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

echo "==== [S8] Minimal-pair patching (L26-L34) ===="
date

python3 - <<'PATCH_SCRIPT'
"""
Minimal-pair activation patching for physics_entropy_gate_v1.

For each (pair_id, wording_family) pair:
  - Source: allow prompt (nd > 0)
  - Target: block prompt (nd < 0)
  - Patch: residual stream at layers 26-34 (covers L28 gate + margin)
  - Metric: does target nd flip from <0 to >0 after patching?

Also tests block→allow direction.
"""
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.model_utils import ModelWrapper

PROMPT_PATH  = Path("data/prompts/physics_entropy_gate_v1_train.jsonl")
OUT_DIR      = Path("data/results/physics_entropy_gate_v1")
MODEL_NAME   = "Qwen/Qwen3-4B"
PATCH_LAYERS = list(range(26, 35))   # L26-L34, covers L27 gate ± margin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=torch.bfloat16)
mdl = mw.model
tok = mw.tokenizer
mdl.eval()

yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
no_id  = tok(" No",  add_special_tokens=False).input_ids[0]

prompts = [json.loads(l) for l in open(PROMPT_PATH)]

# Group into (pair_id, wording_family) → {allow: prompt, block: prompt}
groups = defaultdict(dict)
for p in prompts:
    key = (p["pair_id"], p["wording_family"])
    groups[key][p["pair_role"]] = p

complete_pairs = [(k, v) for k, v in groups.items() if "allow" in v and "block" in v]
print(f"  Found {len(complete_pairs)} complete allow/block pairs")
if len(complete_pairs) == 0:
    print("  No pairs found — skipping patching.")
    sys.exit(0)

import random
random.seed(42)
random.shuffle(complete_pairs)
complete_pairs = complete_pairs[:40]

def get_nd(input_ids):
    with torch.no_grad():
        out = mdl(input_ids=input_ids, use_cache=False)
        lp  = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(lp[yes_id] - lp[no_id])

def get_residuals(input_ids, layers):
    """Return dict layer→residual_at_final_token (1D tensor)."""
    residuals = {}
    hooks = []
    def make_hook(l):
        def h(module, inp, out):
            # Newer transformers may return hidden_states as a bare Tensor rather
            # than a (Tensor, ...) tuple — handle both.
            hs = out[0] if isinstance(out, tuple) else out  # (batch, seq, hidden)
            residuals[l] = hs[0, -1].detach().clone()
        return h
    for l in layers:
        hk = mdl.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(hk)
    with torch.no_grad():
        mdl(input_ids=input_ids, use_cache=False)
    for hk in hooks:
        hk.remove()
    return residuals

def patch_nd(source_ids, target_ids, patch_layers):
    """Patch residual at patch_layers from source into target forward pass."""
    src_residuals = get_residuals(source_ids, patch_layers)
    result_nd = {}

    for patch_layer in patch_layers:
        def make_patch_hook(pl):
            def h(module, inp, out):
                if pl in src_residuals:
                    # Newer transformers may return hidden_states as a bare Tensor
                    # rather than a (Tensor, ...) tuple — handle both.
                    hs = out[0] if isinstance(out, tuple) else out
                    src_vec = src_residuals[pl].to(hs.dtype)
                    patched = hs.clone()
                    patched[0, -1] = src_vec
                    if isinstance(out, tuple):
                        return (patched,) + out[1:]
                    return patched
                return out
            return h
        hk = mdl.model.layers[patch_layer].register_forward_hook(make_patch_hook(patch_layer))
        with torch.no_grad():
            out = mdl(input_ids=target_ids, use_cache=False)
            lp  = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nd  = float(lp[yes_id] - lp[no_id])
        hk.remove()
        result_nd[patch_layer] = nd

    return result_nd

rows = []
for i, ((pid, wf), pair) in enumerate(complete_pairs):
    if i % 5 == 0:
        print(f"  Patching pair {i}/{len(complete_pairs)}", flush=True)

    allow_p = pair["allow"]
    block_p = pair["block"]

    allow_ids = tok(allow_p["prompt"], return_tensors="pt").to(device).input_ids
    block_ids = tok(block_p["prompt"], return_tensors="pt").to(device).input_ids

    nd_allow_base = get_nd(allow_ids)
    nd_block_base = get_nd(block_ids)

    nd_block_patched = patch_nd(allow_ids, block_ids, PATCH_LAYERS)
    nd_allow_patched = patch_nd(block_ids, allow_ids, PATCH_LAYERS)

    for layer, nd_bp in nd_block_patched.items():
        nd_ap = nd_allow_patched.get(layer, float("nan"))
        rows.append({
            "pair_id":          pid,
            "wording_family":   wf,
            "concept":          allow_p.get("concept", ""),
            "patch_layer":      layer,
            "nd_allow_base":    round(nd_allow_base, 4),
            "nd_block_base":    round(nd_block_base, 4),
            "nd_block_patched": round(nd_bp, 4),
            "nd_allow_patched": round(nd_ap, 4),
            "block_flipped":    (nd_block_base < 0) and (nd_bp > 0),
            "allow_flipped":    (nd_allow_base > 0) and (nd_ap < 0),
        })

patch_df = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
patch_df.to_csv(OUT_DIR / "minimal_pair_patching.csv", index=False)

print(f"\n  Minimal-pair patching results:")
print(f"  {'Layer':<8} {'block_flip_rate':>16} {'allow_flip_rate':>16}  n_pairs")
for layer, g in patch_df.groupby("patch_layer"):
    bf = float(g["block_flipped"].mean())
    af = float(g["allow_flipped"].mean())
    print(f"  L{layer:<7} {bf:>16.4f} {af:>16.4f}  {len(g)//2}")

print(f"\nSaved: {OUT_DIR}/minimal_pair_patching.csv")
PATCH_SCRIPT

echo ""
echo "========================================================"
echo "  S8 done: $(date)"
echo "========================================================"
echo ""
echo "Sync to local:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/physics_entropy_gate_v1/minimal_pair_patching.csv data/results/physics_entropy_gate_v1/"
