"""
55_late_layer_sanity.py — Late-layer ablation sanity check (L26–L35)

Answers: is L24 a genuine causal peak, or just the edge of the current analysis window?

Protocol for each target layer:
  1. Collect MLP-input activations for all prompts (post_attention_layernorm, last token).
  2. Encode with the trained transcoder.
  3. Rank features by discriminative activation: |mean_act_alpha − mean_act_beta|.
  4. Select top-K features.
  5. Run zero-ablation per-feature per-prompt; record effect_size + sign_flipped.
  6. Print comparison table vs. L24 benchmark.

Feature-selection note: |mean_alpha − mean_beta| is a direct proxy for what
gradient × activation attribution captures at the decision token. It identifies
features whose activation pattern distinguishes alpha from beta prompts.

Usage:
    python scripts/55_late_layer_sanity.py \\
        --behaviour  physics_decay_type_probe_v2 \\
        --split      train \\
        --layers     26 28 30 32 34 \\
        --top_k      5 \\
        --output_dir data/analysis/late_layer_sanity

    sbatch jobs/run_late_layer_sanity.sbatch
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set
import yaml

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    p.add_argument("--split",     default="train", choices=["train", "test"])
    p.add_argument("--layers",    type=int, nargs="+",
                   default=[26, 28, 30, 32, 34])
    p.add_argument("--top_k",     type=int, default=5,
                   help="Top-K features per layer by discriminative activation")
    p.add_argument("--output_dir", default="data/analysis/late_layer_sanity")
    return p.parse_args()

# ── helpers ───────────────────────────────────────────────────────────────────

def get_mlp_input(hf_model, tokenizer, prompt, layer_idx, device):
    """Capture post_attention_layernorm output at last token — (1, d)."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}
    block = hf_model.model.layers[layer_idx]

    def _hook(module, inp, out):
        captured["x"] = out[:, -1:, :].detach()   # (1, 1, d)

    h = block.post_attention_layernorm.register_forward_hook(_hook)
    with torch.no_grad():
        hf_model(**inputs, use_cache=False)
    h.remove()
    return captured["x"][0]    # (1, d)


def compute_logit_diff(hf_model, tokenizer, prompt, correct, incorrect, device):
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = hf_model(**inputs, use_cache=False)
    lp  = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct,   add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()


def compute_logit_diff_with_patch(hf_model, tokenizer, prompt, correct, incorrect,
                                   layer_idx, x_patch, device):
    """Forward pass with x_patch replacing post_attention_layernorm at last token."""
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    block = hf_model.model.layers[layer_idx]
    _p = x_patch.to(device)   # (1, d)

    def _hook(module, inp, out):
        mod = out.clone()
        mod[0, -1, :] = _p[0, :]
        return mod

    h = block.post_attention_layernorm.register_forward_hook(_hook)
    with torch.no_grad():
        out = hf_model(**inputs, use_cache=False)
    h.remove()
    lp  = torch.log_softmax(out.logits[0, -1, :], dim=0)
    cid = tokenizer.encode(correct,   add_special_tokens=False)[0]
    iid = tokenizer.encode(incorrect, add_special_tokens=False)[0]
    return (lp[cid] - lp[iid]).item()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    root   = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tc_cfg  = yaml.safe_load(open(root / "configs" / "transcoder_config.yaml"))
    tc_info = tc_cfg["transcoders"][tc_cfg["model_size"]]

    prompts = [json.loads(l) for l in open(
        root / "data" / "prompts" / f"{args.behaviour}_{args.split}.jsonl")]

    print(f"Loaded {len(prompts)} prompts.")
    print(f"Target layers: {args.layers}")
    print(f"Top-K per layer: {args.top_k}")
    print()

    print("Loading model…")
    model_wrap = ModelWrapper(tc_info["model_name"], device=device)
    hf_model   = model_wrap.model
    tokenizer  = model_wrap.tokenizer

    print(f"Loading transcoders for layers {args.layers}…")
    tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                                  layers=args.layers, device=device)

    # ── L24 benchmark from Run B/C (individual feature, fixed-axis α-β axis) ──
    # Source: three-mode ablation control; mean |effect| on fixed-axis per layer
    # We'll compute the same fixed-axis stats here for direct comparison.

    alpha_idxs = [i for i, p in enumerate(prompts) if p["correct_answer"].strip() == "alpha"]
    beta_idxs  = [i for i, p in enumerate(prompts) if p["correct_answer"].strip() == "beta"]
    print(f"Alpha prompts: {len(alpha_idxs)},  Beta prompts: {len(beta_idxs)}")
    print()

    out_dir = root / args.output_dir / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for layer in args.layers:
        tc = tc_set[layer]
        d_tc = tc.encode(torch.zeros(1, tc_info.get("d_model", 2560), dtype=tc.dtype, device=device)).shape[-1]
        print(f"{'='*60}")
        print(f"Layer {layer}  (d_transcoder={d_tc})")
        print(f"{'='*60}")

        # Step 1: collect activations for all prompts
        print(f"  Collecting activations for {len(prompts)} prompts…")
        act_matrix = torch.zeros(len(prompts), d_tc, dtype=torch.float32)

        for i, p in enumerate(prompts):
            x = get_mlp_input(hf_model, tokenizer, p["prompt"], layer, device)
            with torch.no_grad():
                a = tc.encode(x.to(tc.dtype))
            act_matrix[i] = a[0].float().cpu()

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(prompts)}")

        # Step 2: rank features by |mean_alpha − mean_beta|
        mean_alpha = act_matrix[alpha_idxs].mean(dim=0)   # (d_tc,)
        mean_beta  = act_matrix[beta_idxs].mean(dim=0)    # (d_tc,)
        disc_score = (mean_alpha - mean_beta).abs()       # (d_tc,)

        top_feats = disc_score.topk(args.top_k).indices.tolist()
        print(f"  Top-{args.top_k} discriminative features: {top_feats}")
        for fi in top_feats:
            print(f"    feat {fi:6d}: |Δmean|={disc_score[fi]:.4f}  "
                  f"mean_α={mean_alpha[fi]:.4f}  mean_β={mean_beta[fi]:.4f}")
        print()

        # Collect baselines and cache encoded activations in ONE pass.
        # This avoids re-running the full model for every (feature, prompt) pair.
        print(f"  Caching baselines + encoded activations…")
        baselines = {}
        act_encoded = []   # list of (x_orig, a_encoded) per prompt, on CPU

        for i, p in enumerate(prompts):
            b = compute_logit_diff(
                hf_model, tokenizer, p["prompt"],
                p["correct_answer"], p["incorrect_answer"], device)
            baselines[i] = b

            x = get_mlp_input(hf_model, tokenizer, p["prompt"], layer, device)
            with torch.no_grad():
                a = tc.encode(x.to(tc.dtype))
            act_encoded.append((x.cpu(), a.cpu()))   # keep on CPU to save VRAM

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(prompts)}")

        # Step 3: ablation per feature × prompt — reuse cached activations
        print(f"  Running ablations ({args.top_k} features × {len(prompts)} prompts)…")
        for feat_idx in top_feats:
            feat_id = f"L{layer}_F{feat_idx}"
            disc    = disc_score[feat_idx].item()
            dir_str = "pro_alpha" if (mean_alpha[feat_idx] > mean_beta[feat_idx]) else "anti_alpha"

            for i, p in enumerate(prompts):
                x_cpu, a_cpu = act_encoded[i]
                x = x_cpu.to(device)
                a = a_cpu.to(device)

                with torch.no_grad():
                    a_abl = a.clone()
                    a_abl[0, feat_idx] = 0.0
                    x_hat = tc.decode(a_abl).to(x.dtype)

                iv_margin = compute_logit_diff_with_patch(
                    hf_model, tokenizer, p["prompt"],
                    p["correct_answer"], p["incorrect_answer"],
                    layer, x_hat, device)

                b  = baselines[i]
                es = iv_margin - b
                sf = int((b > 0) != (iv_margin > 0))

                all_rows.append({
                    "layer":            layer,
                    "feature_id":       feat_id,
                    "feature_idx":      feat_idx,
                    "disc_score":       disc,
                    "feature_dir":      dir_str,
                    "prompt_idx":       i,
                    "correct_answer":   p["correct_answer"].strip(),
                    "baseline_logit":   b,
                    "intervened_logit": iv_margin,
                    "effect_size":      es,
                    "sign_flipped":     sf,
                })

            # Per-feature summary
            feat_rows = [r for r in all_rows if r["feature_id"] == feat_id]
            sfr  = np.mean([r["sign_flipped"] for r in feat_rows])
            meff = np.mean([r["effect_size"]  for r in feat_rows])
            alpha_eff = np.mean([r["effect_size"] for r in feat_rows if r["correct_answer"] == "alpha"])
            beta_eff  = np.mean([r["effect_size"] for r in feat_rows if r["correct_answer"] == "beta"])
            print(f"    {feat_id} ({dir_str:11s}): SFR={sfr:.4f}  "
                  f"mean_eff={meff:+.4f}  α_eff={alpha_eff:+.4f}  β_eff={beta_eff:+.4f}")

        print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / f"late_layer_sanity_{args.behaviour}_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows → {csv_path}")
    print()

    # ── Summary comparison table ──────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY: mean |fixed-axis effect| per layer")
    print("(fixed-axis = logit(α) − logit(β), to match three-mode control)")
    print()

    alpha_tok_id = tokenizer.encode(" alpha", add_special_tokens=False)[0]
    beta_tok_id  = tokenizer.encode(" beta",  add_special_tokens=False)[0]

    # compute fixed-axis effect per row: direction is always logit(α) - logit(β)
    # For α prompts (correct=α): effect_size = iv - base, fixed_axis_eff = +(iv - base) [same axis]
    # For β prompts (correct=β): effect_size = iv_β-axis - base_β-axis, fixed_axis_eff = -(iv - base)
    def fixed_axis_eff(row):
        if row["correct_answer"] == "alpha":
            return row["effect_size"]
        else:
            return -row["effect_size"]

    df["fixed_axis_eff"] = df.apply(fixed_axis_eff, axis=1)

    print(f"{'Layer':>6} {'Feature':>14} {'SFR':>8} {'mean_eff':>10} {'α_eff':>10} {'β_eff':>10} {'disc':>8}")
    print("-" * 72)
    for layer in args.layers:
        ldf = df[df["layer"] == layer]
        for feat_id, fdf in ldf.groupby("feature_id"):
            sfr     = fdf["sign_flipped"].mean()
            meff    = fdf["fixed_axis_eff"].mean()
            a_eff   = fdf[fdf["correct_answer"] == "alpha"]["fixed_axis_eff"].mean()
            b_eff   = fdf[fdf["correct_answer"] == "beta"]["fixed_axis_eff"].mean()
            disc    = fdf["disc_score"].iloc[0]
            print(f"  L{layer:>2d} {feat_id:>14}  {sfr:>8.4f}  {meff:>+10.4f}  "
                  f"{a_eff:>+10.4f}  {b_eff:>+10.4f}  {disc:>8.4f}")

    print()
    print("L24 benchmark (from three-mode ablation control / Run B):")
    print("  Best feature: SFR~0.30, mean fixed-axis eff~+0.61 (anti-α features)")
    print("  If any L26+ feature exceeds these, L24 is NOT the causal peak.")
    print()

    # Layer-level max SFR and max mean |eff|
    print(f"{'Layer':>6} {'max_SFR':>9} {'max|eff|':>10} {'n_features':>12}")
    print("-" * 42)
    for layer in args.layers:
        ldf = df[df["layer"] == layer]
        by_feat = ldf.groupby("feature_id").agg(
            sfr=("sign_flipped", "mean"),
            abs_eff=("fixed_axis_eff", lambda x: abs(x.mean()))
        )
        print(f"  L{layer:>2d}  {by_feat['sfr'].max():>9.4f}  "
              f"{by_feat['abs_eff'].max():>+10.4f}  {len(by_feat):>12}")

    print()
    # Save layer summary
    rows = []
    for layer in args.layers:
        ldf = df[df["layer"] == layer]
        for feat_id, fdf in ldf.groupby("feature_id"):
            rows.append({
                "layer":       layer,
                "feature_id":  feat_id,
                "disc_score":  fdf["disc_score"].iloc[0],
                "feature_dir": fdf["feature_dir"].iloc[0],
                "sfr":         fdf["sign_flipped"].mean(),
                "mean_fixed_axis_eff":  fdf["fixed_axis_eff"].mean(),
                "alpha_eff":   fdf[fdf["correct_answer"] == "alpha"]["fixed_axis_eff"].mean(),
                "beta_eff":    fdf[fdf["correct_answer"] == "beta"]["fixed_axis_eff"].mean(),
            })
    summary_df = pd.DataFrame(rows)
    summ_path  = out_dir / f"late_layer_sanity_summary_{args.behaviour}_{args.split}.csv"
    summary_df.to_csv(summ_path, index=False)
    print(f"Summary saved → {summ_path}")


if __name__ == "__main__":
    main()
