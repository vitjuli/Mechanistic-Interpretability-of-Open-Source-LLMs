#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=egate-s8c-s14
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/egate-s8c-s14-%j.out
#SBATCH --error=logs/egate-s8c-s14-%j.out
#
# physics_entropy_gate_v1 — S8c + S14
#
# S8c: Negative-control patching at L10–L25 (completes the layer sweep
#      started by S8 which covered L26–L34).  Expected: flip rates much
#      lower than the 87–92% seen at L26–L34, confirming that the gate
#      representation specifically emerges at/after L27, not earlier.
#      Saves to minimal_pair_patching_early_layers.csv (do not overwrite
#      the already-synced L26–L34 results).
#
# S14: Path validation for the top circuit paths via script 14.
#      Tests propagation consistency for each A→B causal transition:
#        L31_F28917→L32_F79442→L33_F143078→L34_F{15971,28807}→L35_*
#        L33_F17197→L34_F37728  (shortest path, rank-1 causal score)
#      Outputs propagation_consistency per path (target ≥ 0.70).
#
# Prereqs: S0–S8 complete (job 29235566 + 29289788).
# Runtime estimate:  S8c ~20 min  |  S14 ~60–80 min  |  total ≤ 2h

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_entropy_gate_v1 — S8c + S14"
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

# ─────────────────────────────────────────────────────────────────────────────
# S8c — Negative-control patching (L10–L25)
#
# Same 40 minimal pairs and same patch_nd logic as S8.
# Covers the 16 "early" layers to show flip rates are low before L27.
# Saves to a new file; combine with minimal_pair_patching.csv (L26–L34)
# for the full L10–L34 sweep in local analysis.
# ─────────────────────────────────────────────────────────────────────────────
echo "==== [S8c] Negative-control patching (L10–L25) ===="
date

python3 - <<'PATCH_EARLY'
"""
Negative-control patching for physics_entropy_gate_v1.

Patches residual stream from allow→block (and block→allow) at each of
L10–L25 for 40 sampled minimal pairs.  These are the layers BEFORE the
gate transition observed at L27; we expect substantially lower flip rates
than the 87–92% seen at L26–L34, confirming temporal specificity.

Output: data/results/physics_entropy_gate_v1/minimal_pair_patching_early_layers.csv
"""
import json, sys, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch

sys.path.insert(0, ".")
from src.model_utils import ModelWrapper

PROMPT_PATH  = Path("data/prompts/physics_entropy_gate_v1_train.jsonl")
OUT_DIR      = Path("data/results/physics_entropy_gate_v1")
MODEL_NAME   = "Qwen/Qwen3-4B"
PATCH_LAYERS = list(range(10, 26))   # L10–L25 (negative-control range)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=torch.bfloat16)
mdl = mw.model
tok = mw.tokenizer
mdl.eval()

yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
no_id  = tok(" No",  add_special_tokens=False).input_ids[0]

prompts = [json.loads(l) for l in open(PROMPT_PATH)]
groups  = defaultdict(dict)
for p in prompts:
    key = (p["pair_id"], p["wording_family"])
    groups[key][p["pair_role"]] = p

complete_pairs = [(k, v) for k, v in groups.items() if "allow" in v and "block" in v]
print(f"  Found {len(complete_pairs)} complete allow/block pairs")

# Use the SAME 40 pairs as S8 (same seed) for a direct comparison
random.seed(42)
random.shuffle(complete_pairs)
complete_pairs = complete_pairs[:40]

def get_nd(input_ids):
    with torch.no_grad():
        out = mdl(input_ids=input_ids, use_cache=False)
        lp  = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(lp[yes_id] - lp[no_id])

def get_residuals(input_ids, layers):
    residuals = {}
    hooks = []
    def make_hook(l):
        def h(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
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
    src_residuals = get_residuals(source_ids, patch_layers)
    result_nd = {}
    for patch_layer in patch_layers:
        def make_patch_hook(pl):
            def h(module, inp, out):
                if pl in src_residuals:
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
        print(f"  Patching pair {i}/{len(complete_pairs)}  (layers L10-L25)", flush=True)

    allow_p   = pair["allow"]
    block_p   = pair["block"]
    allow_ids = tok(allow_p["prompt"], return_tensors="pt").to(device).input_ids
    block_ids = tok(block_p["prompt"], return_tensors="pt").to(device).input_ids

    nd_allow_base    = get_nd(allow_ids)
    nd_block_base    = get_nd(block_ids)
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
patch_df.to_csv(OUT_DIR / "minimal_pair_patching_early_layers.csv", index=False)

print(f"\n  Patching results — early layers (negative control):")
print(f"  {'Layer':<8} {'block_flip_rate':>16} {'allow_flip_rate':>16}  n_pairs")
for layer, g in patch_df.groupby("patch_layer"):
    bf = float(g["block_flipped"].mean())
    af = float(g["allow_flipped"].mean())
    print(f"  L{layer:<7} {bf:>16.4f} {af:>16.4f}  {len(g)//2}")

print(f"\nFor reference — late layers from S8 (L26–L34):")
print(f"  L26:  block=0.8750  allow=0.9250")
print(f"  L27:  block=0.9000  allow=0.9250")
print(f"  L28-L33: block=0.9000  allow=0.9250")
print(f"  L34:  block=0.9250  allow=0.9250")
print(f"\nSaved: {OUT_DIR}/minimal_pair_patching_early_layers.csv")
PATCH_EARLY

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# S14 — Path validation
#
# For each unique A→B causal transition in the circuit, ablates feature A
# at layer L_A and measures the activation change at feature B at layer L_B.
# Reports propagation_consistency (fraction of prompts where |Δact_B| > 0.01).
#
# The two circuit branches to validate:
#   Branch 1 (short, strongest causal score):
#     input → L33_F17197 → L34_F37728 → output_correct
#   Branch 2 (long, star-path score):
#     input → L31_F28917 → L32_F79442 → L33_F143078 → L34_F{15971,28807} → L35_*
# ─────────────────────────────────────────────────────────────────────────────
echo "==== [S14] Path validation ===="
date

python -u scripts/14_path_validation.py \
    --behaviour   physics_entropy_gate_v1 \
    --split       train \
    --circuits_json data/results/causal_edges/physics_entropy_gate_v1/circuits_physics_entropy_gate_v1_train.json \
    --n_paths     10 \
    --epsilon     0.01 \
    --max_prompts 100

echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  S8c + S14 done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Sync to local:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/physics_entropy_gate_v1/minimal_pair_patching_early_layers.csv data/results/physics_entropy_gate_v1/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/path_validation/physics_entropy_gate_v1/ data/results/path_validation/physics_entropy_gate_v1/"
