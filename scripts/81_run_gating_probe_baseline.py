"""
Behavioural baseline for gating_probe_v1.

Uses Qwen3-4B BASE model with logprob evaluation:
  nd(x) = logp(' Yes'|x) - logp(' No'|x)
  nd > 0  → model predicts Yes (allowed/valid/possible)
  nd < 0  → model predicts No  (blocked/invalid/impossible)

Answer token IDs (Qwen3-4B tokenizer, confirmed single tokens):
  ' Yes' = 7414   ' No'  = 2308

Outputs:
  data/results/gating_probe_v1/baseline_results.csv

Usage:
    python scripts/81_run_gating_probe_baseline.py --device cuda
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper

PROMPT_PATH = Path("data/prompts/gating/gating_probe_v1.jsonl")
OUT_DIR     = Path("data/results/gating_probe_v1")
MODEL_NAME  = "Qwen/Qwen3-4B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",  default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--batch",  type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(l) for l in open(PROMPT_PATH)]
    print(f"Loaded {len(prompts)} prompts")

    print("Loading model…")
    mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    mdl = mw.model
    tok = mw.tokenizer
    mdl.eval()

    # Confirm single-token answer IDs
    yes_ids = tok(" Yes", add_special_tokens=False).input_ids
    no_ids  = tok(" No",  add_special_tokens=False).input_ids
    assert len(yes_ids) == 1 and len(no_ids) == 1, "Multi-token answers detected!"
    YES_ID, NO_ID = yes_ids[0], no_ids[0]
    print(f"Answer tokens: ' Yes'={YES_ID}  ' No'={NO_ID}")

    rows = []
    with torch.no_grad():
        for i, row in enumerate(prompts):
            if i % 50 == 0:
                print(f"  {i}/{len(prompts)}", end="\r", flush=True)

            inputs = tok(row["prompt"], return_tensors="pt").to(device)
            out    = mdl.model(**inputs, use_cache=False)
            lp     = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

            lp_yes = float(lp[YES_ID])
            lp_no  = float(lp[NO_ID])
            nd     = lp_yes - lp_no   # >0 → model says Yes

            pred_yes   = nd > 0
            correct    = (row["correct_answer"] == " Yes") == pred_yes

            rows.append({
                **{k: row[k] for k in [
                    "pair_id","family","family_name","concept_key","domain",
                    "wording_family","gate_label","pair_role","surface_direction",
                    "constraint_type","abstract_rule","difficulty",
                    "expected_mechanism","correct_answer",
                ]},
                "lp_yes":   round(lp_yes, 5),
                "lp_no":    round(lp_no,  5),
                "nd":       round(nd, 5),
                "pred_yes": pred_yes,
                "correct":  correct,
                "prompt":   row["prompt"],
            })

    print()
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "baseline_results.csv"
    df.to_csv(out_path, index=False)

    # Quick summary
    print(f"\n{'Family':5s} {'Name':30s} {'n':>5s} {'sign_acc':>9s}")
    print("-" * 55)
    for fam, grp in df.groupby("family"):
        print(f"  {fam:3s}  {grp['family_name'].iloc[0]:30s} {len(grp):>4d}  {grp['correct'].mean():.4f}")
    print("-" * 55)
    print(f"  {'ALL':3s}  {'':30s} {len(df):>4d}  {df['correct'].mean():.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
