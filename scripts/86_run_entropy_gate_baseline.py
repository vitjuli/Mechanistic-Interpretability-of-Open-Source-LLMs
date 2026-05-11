"""
Behavioural baseline + gating analysis for physics_entropy_gate_v1.

Evaluates sign_acc, AUC(gate_label), pair_flip_rate, direction_sensitivity,
wording robustness, and a Causal Readiness Score (CRS) — the same metrics
as gating_probe_v1 (scripts 81+82) but embedded in a single script for the
full behaviour dataset.

Model: Qwen/Qwen3-4B BASE.
Answer tokens (confirmed single-token):
  ' Yes' (token 7414) = physically allowed
  ' No'  (token 2308) = physically blocked

Gates (used by sbatch for pipeline go/no-go):
  overall_sign_acc   in [0.75, 0.95]  (warn if <0.75 or >0.95)
  pair_flip_rate     ≥ 0.70
  auc_gate_label     ≥ 0.85

Outputs:
  data/results/physics_entropy_gate_v1/
    baseline_results.csv
    baseline_metrics.json     ← machine-readable for gate check
    wording_breakdown.csv
    pair_analysis.csv

Usage:
    python scripts/86_run_entropy_gate_baseline.py --device cuda [--dtype bfloat16]
"""

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper

BEHAVIOUR    = "physics_entropy_gate_v1"
PROMPT_PATH  = Path(f"data/prompts/{BEHAVIOUR}_train.jsonl")
OUT_DIR      = Path(f"data/results/{BEHAVIOUR}")
MODEL_NAME   = "Qwen/Qwen3-4B"


def auc_score(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",  default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPT_PATH.exists():
        print(f"ERROR: {PROMPT_PATH} not found.")
        print("       Run: python scripts/85_generate_entropy_gate_prompts.py")
        sys.exit(1)

    prompts = [json.loads(l) for l in open(PROMPT_PATH)]
    print(f"Loaded {len(prompts)} prompts from {PROMPT_PATH}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model…")
    mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    mdl = mw.model
    tok = mw.tokenizer
    mdl.eval()

    yes_ids = tok(" Yes", add_special_tokens=False).input_ids
    no_ids  = tok(" No",  add_special_tokens=False).input_ids
    assert len(yes_ids) == 1, f"' Yes' is multi-token: {yes_ids}"
    assert len(no_ids)  == 1, f"' No' is multi-token: {no_ids}"
    YES_ID, NO_ID = yes_ids[0], no_ids[0]
    print(f"Answer tokens confirmed: ' Yes'={YES_ID}  ' No'={NO_ID}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    rows = []
    with torch.no_grad():
        for i, row in enumerate(prompts):
            if i % 20 == 0:
                print(f"  {i}/{len(prompts)}", end="\r", flush=True)

            inputs = tok(row["prompt"], return_tensors="pt").to(device)
            out    = mdl(**inputs, use_cache=False)
            lp     = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

            lp_yes = float(lp[YES_ID])
            lp_no  = float(lp[NO_ID])
            nd     = lp_yes - lp_no   # >0 → model says Yes (allowed)

            pred_allow = nd > 0
            gate_is_allow = (row["gate_label"] == "allow")
            correct = (pred_allow == gate_is_allow)

            rows.append({
                **{k: row[k] for k in [
                    "pair_id", "pair_role", "gate_label", "physical_direction",
                    "concept", "wording_family", "difficulty", "domain",
                    "constraint_type", "abstract_rule", "expected_mechanism",
                    "correct_answer",
                ] if k in row},
                "lp_yes":     round(lp_yes, 5),
                "lp_no":      round(lp_no,  5),
                "nd":         round(nd, 5),
                "pred_allow": pred_allow,
                "correct":    correct,
                "prompt":     row["prompt"],
            })
    print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_results.csv", index=False)
    print(f"Saved: {OUT_DIR}/baseline_results.csv")

    # ── Per-wording breakdown ────────────────────────────────────────────────
    wd_rows = []
    for wf, g in df.groupby("wording_family"):
        wd_rows.append({
            "wording_family": wf,
            "n":              len(g),
            "sign_acc":       round(float(g["correct"].mean()), 4),
            "mean_nd_allow":  round(float(g.loc[g["gate_label"]=="allow","nd"].mean()), 4),
            "mean_nd_block":  round(float(g.loc[g["gate_label"]=="block","nd"].mean()), 4),
        })
    wd_df = pd.DataFrame(wd_rows).sort_values("wording_family")
    wd_df.to_csv(OUT_DIR / "wording_breakdown.csv", index=False)

    # ── Pair analysis ────────────────────────────────────────────────────────
    pair_rows = []
    for pid, g in df.groupby("pair_id"):
        allow_g  = g[g["gate_label"] == "allow"]
        block_g  = g[g["gate_label"] == "block"]
        if allow_g.empty or block_g.empty:
            continue
        nd_all = float(allow_g["nd"].mean())
        nd_blk = float(block_g["nd"].mean())
        frac_ok_allow = float((allow_g["nd"] > 0).mean())
        frac_ok_block = float((block_g["nd"] < 0).mean())
        flip_correct  = frac_ok_allow > 0.5 and frac_ok_block > 0.5
        pair_rows.append({
            "pair_id":              pid,
            "concept":              g["concept"].iloc[0],
            "difficulty":           g["difficulty"].iloc[0],
            "nd_allow_mean":        round(nd_all, 4),
            "nd_block_mean":        round(nd_blk, 4),
            "direction_sensitivity":round(abs(nd_all - nd_blk), 4),
            "frac_correct_allow":   round(frac_ok_allow, 4),
            "frac_correct_block":   round(frac_ok_block, 4),
            "flip_correct":         flip_correct,
        })
    pair_df = pd.DataFrame(pair_rows).sort_values("pair_id")
    pair_df.to_csv(OUT_DIR / "pair_analysis.csv", index=False)

    # ── Global metrics ───────────────────────────────────────────────────────
    overall_sign_acc = float(df["correct"].mean())
    mean_nd          = float(df["nd"].mean())
    mean_abs_nd      = float(df["nd"].abs().mean())

    y_true  = (df["gate_label"] == "allow").astype(int).values
    auc_val = auc_score(y_true, df["nd"].values)

    pair_flip_rate = float(pair_df["flip_correct"].mean()) if len(pair_df) > 0 else float("nan")
    mean_dir_sens  = float(pair_df["direction_sensitivity"].mean()) if len(pair_df) > 0 else float("nan")

    wf_accs   = df.groupby("wording_family")["correct"].mean()
    wf_std    = float(wf_accs.std()) if len(wf_accs) > 1 else 0.0
    wf_min    = float(wf_accs.min())

    # CRS (same formula as script 82): pair_flip × sign_acc × norm_dir_sens × (1-wf_std)
    ds_norm   = min(mean_dir_sens / 5.0, 1.0)   # normalize: 5.0 nats ≈ ceiling in probe data
    wf_cons   = max(0.0, 1.0 - wf_std)
    crs       = pair_flip_rate * overall_sign_acc * ds_norm * wf_cons

    # Per-difficulty breakdown
    diff_summary = {}
    for diff, g in df.groupby("difficulty"):
        diff_summary[diff] = round(float(g["correct"].mean()), 4)

    metrics = {
        "behaviour":          BEHAVIOUR,
        "n_prompts":          len(df),
        "overall_sign_acc":   round(overall_sign_acc, 4),
        "mean_nd":            round(mean_nd, 4),
        "mean_abs_nd":        round(mean_abs_nd, 4),
        "auc_gate_label":     round(auc_val, 4),
        "pair_flip_rate":     round(pair_flip_rate, 4),
        "direction_sensitivity_mean": round(mean_dir_sens, 4),
        "wording_family_std": round(wf_std, 4),
        "wording_family_min_acc": round(wf_min, 4),
        "causal_readiness_score": round(crs, 4),
        "per_difficulty": diff_summary,
        "per_wording":    {k: round(v, 4) for k, v in wf_accs.items()},
    }

    with open(OUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {OUT_DIR}/baseline_metrics.json")

    # ── Gate assessment ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  BASELINE GATE — {BEHAVIOUR}")
    print("=" * 62)
    print(f"  n_prompts         : {len(df)}")
    print(f"  overall_sign_acc  : {overall_sign_acc:.4f}", end="")
    if overall_sign_acc > 0.95:
        print("  *** WARN: >0.95 — possible ceiling / fact retrieval")
    elif overall_sign_acc < 0.75:
        print("  *** WARN: <0.75 — model may not support this gate")
    else:
        print("  [OK]")
    print(f"  pair_flip_rate    : {pair_flip_rate:.4f}", end="")
    print("  [OK]" if pair_flip_rate >= 0.70 else "  *** WARN: <0.70")
    print(f"  auc_gate_label    : {auc_val:.4f}", end="")
    print("  [OK]" if auc_val >= 0.85 else "  *** WARN: <0.85")
    print(f"  dir_sensitivity   : {mean_dir_sens:.4f}")
    print(f"  wording_std       : {wf_std:.4f}")
    print(f"  CRS               : {crs:.4f}")

    print(f"\n  Per-wording sign accuracy:")
    for _, wr in wd_df.iterrows():
        flag = "OK  " if wr["sign_acc"] >= 0.70 else "WARN"
        print(f"    [{flag}] {wr['wording_family']:<22} {wr['sign_acc']:.4f}  (n={int(wr['n'])})")

    print(f"\n  Per-difficulty:")
    for diff in sorted(diff_summary):
        acc = diff_summary[diff]
        flag = "OK  " if acc >= 0.70 else "WARN"
        print(f"    [{flag}] {diff:<12} {acc:.4f}")

    print(f"\n  Worst pairs (flip_correct=False):")
    failed_pairs = pair_df[~pair_df["flip_correct"]].head(5)
    if failed_pairs.empty:
        print("    (all pairs flipped correctly)")
    else:
        for _, pr in failed_pairs.iterrows():
            print(f"    {pr['pair_id']} {pr['concept']:<30} "
                  f"nd_allow={pr['nd_allow_mean']:+.3f}  nd_block={pr['nd_block_mean']:+.3f}")

    # Gate decision for pipeline
    gate_pass = (
        0.75 <= overall_sign_acc <= 0.97
        and pair_flip_rate >= 0.70
        and auc_val >= 0.85
    )
    gate_warn = (
        (overall_sign_acc > 0.95)
        or (pair_flip_rate < 0.70)
        or (auc_val < 0.85)
    )

    print()
    if overall_sign_acc < 0.75:
        print("GATE: HARD_FAIL — sign_accuracy below 0.75, pipeline should abort.")
        sys.exit(2)
    elif gate_warn:
        print("GATE: WARN_PASS — one or more metrics borderline, inspect before continuing.")
    else:
        print("GATE: PASS — proceeding with full pipeline.")

    print("=" * 62)


if __name__ == "__main__":
    main()
