"""
53_gradient_sensitivity_analysis.py

Direct test: project reconstruction error onto logit-diff gradient.

For each (prompt, layer), measures:
  ε       = x̂ − x          (reconstruction error vector)
  g       = ∇_x logit_diff  (gradient of logit_correct − logit_incorrect
                              w.r.t. MLP input x at last token position)

Key metrics:
  projection         = |⟨ε, g⟩|               (raw directional sensitivity)
  projection_norm    = |⟨ε, g⟩| / ‖ε‖         (user's metric: per unit error)
  cosine             = |⟨ε, g⟩| / (‖ε‖ · ‖g‖) (cosine sim, dimensionless)

If L24 cosine >> L11/L12, reconstruction error on L24 is disproportionately
aligned with the task-sensitive direction — direct evidence for adversarially
directed error beyond what the norm correlation analysis shows.

Protocol:
  1. ONE forward pass with retain_grad() hooks → logit_diff.backward()
     → ∂logit_diff/∂x_layer for all layers simultaneously
  2. Separate no_grad pass: x̂ = dec(enc(x)), ε = x̂ − x
  3. Project ε onto g per layer

Usage (local test, CPU slow):
    python scripts/53_gradient_sensitivity_analysis.py --max_prompts 20

On CSD3 (GPU, full run):
    sbatch jobs/run_gradient_sensitivity.sbatch
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_single_token(tokenizer, tok: str) -> int:
    ids = tokenizer.encode(tok, add_special_tokens=False)
    return ids[0]


def compute_gradient_and_recon(
    model: ModelWrapper,
    transcoder_set,
    prompts: list,
    layers: list,
    device: str,
    max_prompts: int | None = None,
) -> pd.DataFrame:

    if max_prompts:
        prompts = prompts[:max_prompts]

    rows = []

    for pi, p in enumerate(prompts):
        prompt_text  = p["prompt"]
        correct      = p["correct_answer"]
        incorrect    = p["incorrect_answer"]
        prompt_id    = p.get("prompt_id", str(pi))
        group_id     = p.get("group_id", "")
        level_label  = p.get("level_label", "")

        correct_id   = ensure_single_token(model.tokenizer, correct)
        incorrect_id = ensure_single_token(model.tokenizer, incorrect)

        if pi % 20 == 0:
            logger.info(f"  prompt {pi+1}/{len(prompts)}")

        inputs = model.tokenizer([prompt_text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Step 1: ONE forward pass to get all gradients ────────────────────
        # Hook stores intermediate outputs (retained for backward)
        captured: dict[int, torch.Tensor] = {}

        def _make_fwd_hook(layer_idx):
            def _hook(module, inp, out):
                out.retain_grad()
                captured[layer_idx] = out      # shape (1, seq, d)
            return _hook

        handles = []
        for layer in layers:
            block = model.model.model.layers[layer]
            h = block.post_attention_layernorm.register_forward_hook(
                _make_fwd_hook(layer))
            handles.append(h)

        model.model.zero_grad()
        try:
            out_logits = model.model(**inputs, use_cache=False)
            lp = torch.log_softmax(out_logits.logits[0, -1, :], dim=0)
            logit_diff = lp[correct_id] - lp[incorrect_id]
            logit_diff.backward()
        except Exception as e:
            logger.warning(f"Prompt {prompt_id}: backward failed — {e}")
            for h in handles:
                h.remove()
            continue
        finally:
            for h in handles:
                h.remove()

        delta_orig = logit_diff.item()

        # ── Step 2: Reconstruction for each layer (no_grad) ─────────────────
        for layer in layers:
            if layer not in captured or layer not in transcoder_set._transcoders:
                continue

            x_full = captured[layer]          # (1, seq, d)  — in computation graph
            g_full = x_full.grad              # (1, seq, d)  — gradient

            if g_full is None:
                logger.debug(f"  L{layer} prompt {prompt_id}: grad is None")
                continue

            x = x_full[0, -1, :].detach()    # (d,)
            g = g_full[0, -1, :].detach()     # (d,)

            tc = transcoder_set[layer]

            try:
                with torch.no_grad():
                    x_tc  = x.to(tc.dtype).unsqueeze(0)      # (1, d)
                    a     = tc.encode(x_tc)
                    x_hat = tc.decode(a).squeeze(0)          # (d,)
                    x_hat = x_hat.to(x.dtype)
            except Exception as e:
                logger.debug(f"  L{layer} prompt {prompt_id}: recon failed — {e}")
                continue

            eps = x_hat - x                               # reconstruction error

            # ── Metrics ──────────────────────────────────────────────────────
            eps_norm = eps.norm().item()
            g_norm   = g.norm().item()

            if eps_norm < 1e-9 or g_norm < 1e-9:
                continue

            dot      = (eps * g).sum().item()
            proj     = abs(dot)                           # |⟨ε, g⟩|
            proj_n   = proj / eps_norm                   # |⟨ε, g⟩| / ‖ε‖
            cosine   = proj / (eps_norm * g_norm)        # cosine similarity

            # Logit diff with reconstruction patched in (sign flip check)
            # Reuse: patched_delta ≈ delta_orig + dot  (first-order approx)
            # Exact value via second forward pass (no_grad, cheap):
            block    = model.model.model.layers[layer]
            called   = {"n": 0}
            _xh      = x_hat.unsqueeze(0).to(device)    # (1, d)

            def _patch_hook(module, inp, out):
                called["n"] += 1
                mod = out.clone()
                mod[0, -1, :] = _xh[0, :]
                return mod

            ph = block.post_attention_layernorm.register_forward_hook(_patch_hook)
            try:
                with torch.no_grad():
                    out2 = model.model(**inputs, use_cache=False)
                lp2 = torch.log_softmax(out2.logits[0, -1, :], dim=0)
                delta_recon = (lp2[correct_id] - lp2[incorrect_id]).item()
            except Exception:
                delta_recon = float("nan")
            finally:
                ph.remove()

            sign_flipped = int(
                not (np.isnan(delta_recon))
                and np.sign(delta_recon) != np.sign(delta_orig)
            )

            rows.append({
                "prompt_idx":       pi,
                "prompt_id":        prompt_id,
                "group_id":         group_id,
                "level_label":      level_label,
                "correct_answer":   correct.strip(),
                "layer":            layer,
                "delta_orig":       delta_orig,
                "delta_recon":      delta_recon,
                "recon_err_norm":   eps_norm,
                "grad_norm":        g_norm,
                "projection":       proj,          # |⟨ε, g⟩|
                "projection_norm":  proj_n,        # |⟨ε, g⟩| / ‖ε‖
                "cosine_err_grad":  cosine,        # |⟨ε, g⟩| / (‖ε‖·‖g‖)
                "sign_flipped":     sign_flipped,
            })

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    layers = sorted(df["layer"].unique())
    print("\n=== GRADIENT SENSITIVITY ANALYSIS ===\n")
    print(f"{'L':>4}  {'n':>5}  {'flip%':>6}  {'recon_err':>10}  "
          f"{'grad_norm':>10}  {'proj_norm':>10}  {'cosine':>8}  "
          f"{'cosine(flip)':>13}  {'cosine(ok)':>11}")
    print("-" * 97)

    for layer in layers:
        g  = df[df["layer"] == layer]
        gf = g[g["sign_flipped"] == 1]
        go = g[g["sign_flipped"] == 0]

        print(f"  L{layer:>2d}  {len(g):>5}  "
              f"{g['sign_flipped'].mean()*100:>5.1f}%  "
              f"{g['recon_err_norm'].mean():>10.3f}  "
              f"{g['grad_norm'].mean():>10.3f}  "
              f"{g['projection_norm'].mean():>10.4f}  "
              f"{g['cosine_err_grad'].mean():>8.4f}  "
              f"{gf['cosine_err_grad'].mean() if len(gf) else float('nan'):>13.4f}  "
              f"{go['cosine_err_grad'].mean() if len(go) else float('nan'):>11.4f}")

    print()
    # Highlight L11, L12, L18, L24 for direct comparison
    print("=== KEY LAYER COMPARISON ===\n")
    for layer in [11, 12, 18, 24]:
        g = df[df["layer"] == layer]
        print(f"  L{layer:>2d}  cosine={g['cosine_err_grad'].mean():.4f}  "
              f"proj_norm={g['projection_norm'].mean():.4f}  "
              f"flip%={g['sign_flipped'].mean()*100:.1f}%")

    print()
    l11 = df[df["layer"] == 11]["cosine_err_grad"].mean()
    l24 = df[df["layer"] == 24]["cosine_err_grad"].mean()
    ratio = l24 / l11 if l11 > 0 else float("nan")
    print(f"  L24/L11 cosine ratio: {ratio:.2f}×")
    if ratio > 2:
        print("  CONFIRMED: L24 error is disproportionately aligned with "
              "task-sensitive direction.")
    elif ratio > 1.3:
        print("  PARTIAL: L24 shows elevated alignment but not strongly so.")
    else:
        print("  NOT CONFIRMED: L24 and L11 alignment are similar.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour",    default="physics_decay_type_probe_v2")
    p.add_argument("--split",        default="train")
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--layers", nargs="+", type=int, default=list(range(10, 26)))
    p.add_argument("--max_prompts",  type=int, default=None)
    p.add_argument("--output_dir",   default="data/analysis/gradient_sensitivity")
    p.add_argument("--model_name",   default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    root    = Path(__file__).parent.parent
    out_dir = root / args.output_dir / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = (Path(args.prompts_file) if args.prompts_file
                   else root / "data" / "prompts" /
                        f"{args.behaviour}_{args.split}.jsonl")
    assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"

    prompts = [json.loads(l) for l in open(prompt_path)]
    logger.info(f"Loaded {len(prompts)} prompts")

    import yaml
    tc_cfg     = yaml.safe_load(open(root / "configs" / "transcoder_config.yaml"))
    tc_info    = tc_cfg["transcoders"][tc_cfg["model_size"]]
    model_name = args.model_name or tc_info["model_name"]

    logger.info(f"Loading model: {model_name}")
    model = ModelWrapper(model_name, device=args.device)

    logger.info(f"Loading transcoders for layers {args.layers}")
    tc_set = load_transcoder_set(repo_id=tc_info["repo_id"],
                                 layers=args.layers, device=args.device)

    logger.info("Running gradient sensitivity analysis…")
    df = compute_gradient_and_recon(
        model=model, transcoder_set=tc_set,
        prompts=prompts, layers=args.layers,
        device=args.device, max_prompts=args.max_prompts,
    )

    if df.empty:
        logger.error("No results produced.")
        sys.exit(1)

    csv_path  = out_dir / f"gradient_sensitivity_{args.behaviour}_{args.split}.csv"
    json_path = out_dir / f"gradient_sensitivity_{args.behaviour}_{args.split}_summary.json"

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved → {csv_path}  ({len(df)} rows)")

    summarise(df)

    # Save numeric summary
    summary = {}
    for layer in sorted(df["layer"].unique()):
        g = df[df["layer"] == layer]
        summary[int(layer)] = {
            "n": len(g),
            "flip_pct": float(g["sign_flipped"].mean()),
            "mean_cosine_err_grad": float(g["cosine_err_grad"].mean()),
            "mean_projection_norm": float(g["projection_norm"].mean()),
            "mean_recon_err": float(g["recon_err_norm"].mean()),
            "mean_grad_norm": float(g["grad_norm"].mean()),
        }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {json_path}")


if __name__ == "__main__":
    main()
