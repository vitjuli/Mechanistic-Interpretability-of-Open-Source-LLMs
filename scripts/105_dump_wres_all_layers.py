"""
105 — Dump Fisher LDA discriminator w_res at all 36 residual-stream layers.

For each layer ℓ ∈ {0..35}:
  * capture h^(ℓ) at the answer position for 538 prompts
  * fit Fisher LDA on the train family split → w_res^(ℓ) in d=2560
  * save as w_res_L{ℓ:02d}.npy

This enables the dashboard's ROTATION mode to show w_res rotation across ALL
36 layers (currently limited to 13 layers from geometry_stage1).

Runtime: ~5 min on A100 (cached model). Output: 36 × 20 KB files + manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("dump_wres")


def fisher_axis(H, y, shrink=0.1):
    """Fisher LDA axis (unit vector pointing β-side), with shrinkage."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    n = np.linalg.norm(w)
    return w / n if n > 1e-30 else w


def self_test():
    rng = np.random.default_rng(0)
    d = 20
    H = rng.standard_normal((400, d))
    y = np.array([0] * 200 + [1] * 200)
    # Plant a strong signal along x
    H[y == 1] += np.array([5.0] + [0] * (d - 1))[None, :]
    w = fisher_axis(H, y, shrink=0.1)
    assert abs(w[0]) > 0.8, f"axis should point along x for planted signal, got w[0]={w[0]:.3f}"
    # Test classification quality
    proj = H @ w
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, proj)
        assert auc > 0.95, f"AUC should be high, got {auc:.3f}"
    except ImportError:
        # sklearn not available — manual AUC
        order = np.argsort(proj)
        ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(proj) + 1)
        n1 = (y == 1).sum()
        n0 = (y == 0).sum()
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        assert auc > 0.95, f"AUC should be high, got {auc:.3f}"
    print(f"[self_test] OK — Fisher LDA axis recovers planted signal "
          f"(w[0]={w[0]:.3f}, AUC={auc:.3f}).")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    base = model.model
    n_layers = len(base.layers)
    d = model.config.hidden_size
    logger.info("model: %d layers, d=%d", n_layers, d)

    prompts = [json.loads(line) for line in open(args.prompts)]
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = sorted({p.get("surface_family", str(i)) for i, p in enumerate(prompts)})
    rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p.get("surface_family", "") in train_fams for p in prompts])
    logger.info("prompts: %d (train families: %d)", nP, int(is_train.sum()))

    # ── Capture residuals at all 36 layers + final norm ─────────────────────
    n_total = n_layers + 1  # n hidden states including embeddings
    H = np.zeros((n_total, nP, d), np.float32)
    logger.info("capturing residuals at %d layer-taps over %d prompts...",
                n_total, nP)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
            o = model(**inp, output_hidden_states=True, use_cache=False)
            hs = o.hidden_states
            for L in range(n_total):
                H[L, i] = hs[L][0, -1, :].float().cpu().numpy()
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d", i + 1, nP)

    # ── Fit Fisher LDA per layer (train families only) ──────────────────────
    manifest = []
    for L in range(n_total):
        H_L = H[L].astype(np.float64)
        w = fisher_axis(H_L[is_train], y[is_train], args.shrink)
        # Save with name w_res_L{ℓ:02d}.npy where ℓ ∈ {0..36}
        # L=0 = embeddings; L=k = residual after k blocks; final tap = L=36 here
        if L < n_layers:
            name = f"w_res_L{L:02d}.npy"
        else:
            name = "w_res_final.npy"
        np.save(out / name, w)
        manifest.append({"layer": L, "file": name,
                         "is_final": L == n_layers,
                         "norm": float(np.linalg.norm(w))})
        if L % 6 == 0 or L == n_total - 1:
            logger.info("  L%02d: saved %s (norm=1.000)", L, name)

    with open(out / "wres_manifest.json", "w") as f:
        json.dump({"n_layers": n_layers,
                   "n_total_taps": n_total,
                   "d": d,
                   "shrink": args.shrink,
                   "train_frac": args.train_frac,
                   "n_train_prompts": int(is_train.sum()),
                   "vectors": manifest}, f, indent=2)

    print(f"\nDumped {n_total} w_res vectors to {out}/")
    print(f"Manifest: {out}/wres_manifest.json")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts",
                   default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/wres_all_layers")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    run_real(args)


if __name__ == "__main__":
    main()
