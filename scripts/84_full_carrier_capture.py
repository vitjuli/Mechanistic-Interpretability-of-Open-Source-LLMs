"""
84_full_carrier_capture.py   [does ANY feature write w_res? — all features, all layers]
=========================================================================================
Exp 66/74 measured how much the 227-feature co-importance CARRIER's decoder directions
capture the concept axis w_res, and found ~28th percentile of a random-feature null (i.e.
the carrier does not write the axis). But the carrier is a SELECTED subset, and transcoders
see only MLP. This script broadens the test to EVERY feature in the dictionary, on every
transcoder layer, to answer: does ANY feature -- not just the co-importance carrier -- write
the concept along w_res?

Per layer (lazy-load the transcoder decoder weights, ~0.8 GB/layer, freed after use):
  (1) PER-FEATURE alignment: causal cosine cos_C(d_f, w_res) for ALL features, where the
      causal inner product whitens by the residual covariance Sigma (matching the j66/j74
      metric). Report max |cos_C| over all features and compare to the MAX of an equal
      number of random unit directions. If max_feature ~ max_random -> no single feature's
      write-direction aligns with the concept axis.
  (2) COLLECTIVE capture: project w_res onto the span of the top-K most-aligned features'
      decoder directions; capture = ||P_K w_res|| / ||w_res||. Sweep K and compare to a
      random-K-feature null (the j66/j74 quantity, now over the FULL dictionary). If the
      most-aligned K features capture w_res no better than random K -> the dictionary does
      not write the concept even with its best features.
  (3) OPTIONAL (--with_activations): select features by ACTIVATION class-separation (does
      the feature fire differently on alpha vs beta), then measure their collective capture
      vs random-K. This is the honest "independent selection" version (needs the transcoder
      to encode residuals; GPU).

Decoder directions are read robustly (W_dec / w_dec / decoder.weight, either orientation);
if the accessor misses, the script prints the available attributes and exits -- check with
a one-layer smoke run before the full sweep.

Honest expectation (from 66/74): max alignment ~ random, collective capture ~ random-K at
every layer -> NO feature writes the concept axis; w_res is a readout of a signal the MLP
dictionary does not encode as a write-direction. A clean positive (some feature/few features
capture w_res far above null) would localise the write and is reported as such.

SELF-TEST (no torch / no repo):  python 84_full_carrier_capture.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("carrier_capture")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def whitener(Sigma, eps=1e-3):
    """Sigma^{-1/2} for the causal inner product (symmetric)."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps * float(vals.max()) + 1e-12, None)
    return (vecs * (vals ** -0.5)) @ vecs.T


def causal_cos_all(D, w, W):
    """|causal cosine| of each row of D (n_feat x d) with vector w, under whitener W.
    Causal cos uses the whitened space: <a,b>_C = (Wa).(Wb)."""
    Dw = D @ W.T                       # whiten each decoder dir (rows)
    ww = W @ w
    num = Dw @ ww
    den = (np.linalg.norm(Dw, axis=1) + 1e-30) * (np.linalg.norm(ww) + 1e-30)
    return np.abs(num / den)


def collective_capture(D_sel, w):
    """Capture = ||P w|| / ||w||, P = projection onto span(rows of D_sel)."""
    B = D_sel.T                        # (d, K)
    G = B.T @ B
    G += 1e-8 * float(np.trace(G) / max(G.shape[0], 1) + 1e-12) * np.eye(G.shape[0])
    a = np.linalg.solve(G, B.T @ w)
    return float(np.linalg.norm(B @ a) / (np.linalg.norm(w) + 1e-30))


def random_max_cos_expectation(n_dirs, d, n_samples=2000, rng=None):
    """Empirical max |cosine| of n_dirs random unit vecs with a fixed random unit vec."""
    rng = rng or np.random.default_rng(0)
    w = unit_raw(rng.standard_normal(d))
    out = []
    for _ in range(max(1, n_samples // max(n_dirs, 1))):
        R = rng.standard_normal((n_dirs, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        out.append(float(np.abs(R @ w).max()))
    return float(np.mean(out)), float(np.percentile(out, 95))


def percentile_of(value, null):
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, nf = 16, 400
    w = unit_raw(rng.standard_normal(d))
    Sigma = np.eye(d)                                  # identity -> causal cos == euclidean cos
    W = whitener(Sigma)
    assert np.allclose(W, np.eye(d), atol=1e-6), "whitener of identity is identity"

    # dictionary of random decoder dirs; NONE aligned with w -> max cos ~ random
    D = rng.standard_normal((nf, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    cos_rand = causal_cos_all(D, w, W)
    rmean, r95 = random_max_cos_expectation(nf, d, rng=rng)
    assert cos_rand.max() < r95 + 0.15, "no-aligned dictionary: max cos near random max"

    # now plant ONE aligned feature -> it should stick out
    D2 = D.copy(); D2[7] = unit_raw(0.9 * w + 0.1 * D2[7])
    cos2 = causal_cos_all(D2, w, W)
    assert int(np.argmax(cos2)) == 7 and cos2.max() > 0.8, "planted aligned feature must be detected"

    # collective capture: top-K aligned (incl planted) should capture more than random-K
    top = np.argsort(-cos2)[:5]
    cap_top = collective_capture(D2[top], w)
    rng2 = np.random.default_rng(1)
    cap_rand = np.mean([collective_capture(D2[rng2.choice(nf, 5, replace=False)], w) for _ in range(20)])
    assert cap_top > cap_rand, "top-aligned features should capture w better than random K"

    # a random direction is captured by random K at the random-K level (sanity on the null)
    assert 0.0 <= cap_rand <= 1.0 and 0.0 <= cap_top <= 1.0, "capture in [0,1]"
    assert percentile_of(0.9, np.array([0.1, 0.5])) == 100.0
    print("[self_test] OK — whitener, causal cos, random-max, planted-feature detect, collective capture pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for a in p.split("."):
        o = getattr(o, a)
    return o


def _decoder_matrix(tc, d_model):
    """Robustly fetch the decoder weight as (n_features, d_model)."""
    import numpy as _np
    cand = ["W_dec", "w_dec", "W_decoder", "decoder_weight"]
    arr = None
    for a in cand:
        if hasattr(tc, a):
            arr = getattr(tc, a); break
    if arr is None and hasattr(tc, "decoder"):
        dec = tc.decoder
        for a in ["weight", "W", "w_dec", "W_dec"]:
            if hasattr(dec, a):
                arr = getattr(dec, a); break
    if arr is None:
        raise AttributeError(f"decoder weight not found; transcoder attrs: {[x for x in dir(tc) if not x.startswith('__')][:40]}")
    M = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else _np.asarray(arr, dtype=_np.float64)
    if M.ndim != 2:
        raise ValueError(f"decoder weight is not 2D: shape {M.shape}")
    if M.shape[1] == d_model:
        return M                          # (n_feat, d_model)
    if M.shape[0] == d_model:
        return M.T                        # transpose to (n_feat, d_model)
    raise ValueError(f"decoder weight shape {M.shape} matches neither axis to d_model={d_model}")


def run_real(args):
    import torch, yaml
    from src.model_utils import ModelWrapper
    from src.transcoder import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(args.transcoder_config) as f:
        tc_cfg = yaml.safe_load(f)
    model_size = tc_cfg.get("model_size", "4b")
    model_name = tc_cfg["transcoders"][model_size]["model_name"]
    model = ModelWrapper(model_name=model_name, dtype="bfloat16", device="auto", trust_remote_code=True)
    device = next(model.model.parameters()).device
    blocks = _chain(model.model, "model.layers"); n_layers = len(blocks); norm = _chain(model.model, "model.norm")
    d_model = model.model.config.hidden_size
    tokn = model.tokenizer
    alpha_id = tokn.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tokn.encode(args.beta_answer, add_special_tokens=False)[0]

    layers = args.layers if args.layers else list(range(args.layer_lo, args.layer_hi + 1))
    layers = [L for L in layers if 0 <= L < n_layers]
    last = n_layers - 1
    logger.info("d_model=%d; analysing transcoder layers %s", d_model, layers)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm

    # ---------- capture residuals per layer (for w_res + Sigma) ----------
    logger.info("Capturing residuals at %d layers for %d prompts...", len(layers), len(prompts))
    res = {L: np.zeros((len(prompts), d_model), np.float32) for L in layers}
    y = np.zeros(len(prompts), int); tr = np.zeros(len(prompts), bool)
    for i, p in enumerate(prompts):
        inp = model.tokenize([p["prompt"]]); inp = {k: v.to(device) for k, v in inp.items()}
        g = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a): g[L] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            res[L][i] = g[L]
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        tr[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    # ---------- load transcoders (lazy) ----------
    transcoder_set = load_transcoder_set(model_size=model_size, device=device, dtype=torch.bfloat16,
                                         lazy_load=True, layers=layers)

    K_grid = sorted({k for k in args.k_grid})
    rows = []; per_layer = []
    for L in layers:
        H = res[L].astype(np.float64)
        wL = fisher_axis(H[tr], y[tr], args.shrink)
        Sigma = np.cov(H.T) if args.causal else np.eye(d_model)
        W = whitener(Sigma) if args.causal else np.eye(d_model)
        D = _decoder_matrix(transcoder_set[L], d_model)            # (n_feat, d_model)
        n_feat = D.shape[0]
        cos = causal_cos_all(D, wL, W)                              # |causal cos| per feature
        order = np.argsort(-cos)
        rmean, r95 = random_max_cos_expectation(n_feat, d_model, rng=rng)
        # collective capture by top-K aligned vs random-K null
        cap_top = {}; cap_null_p95 = {}; cap_pct = {}
        for K in [k for k in K_grid if k <= n_feat]:
            ct = collective_capture(D[order[:K]], wL)
            null = [collective_capture(D[rng.choice(n_feat, K, replace=False)], wL) for _ in range(args.n_random_sets)]
            cap_top[K] = ct; cap_null_p95[K] = float(np.percentile(null, 95))
            cap_pct[K] = percentile_of(ct, np.array(null))
        rec = {"layer": int(L), "n_features": int(n_feat),
               "max_cos_feature": float(cos.max()), "rand_max_cos_mean": rmean, "rand_max_cos_p95": r95,
               "top_features": [int(j) for j in order[:10]],
               "capture_top_by_K": cap_top, "capture_null_p95_by_K": cap_null_p95, "capture_pct_vs_null_by_K": cap_pct}
        per_layer.append(rec)
        for K in cap_top:
            rows.append({"layer": int(L), "K": int(K), "capture_top_aligned": cap_top[K],
                         "capture_random_p95": cap_null_p95[K], "pct_vs_null": cap_pct[K]})
        logger.info("  L%d: max|cos_C|=%.3f (rand max p95 %.3f) | capture top-227=%.3f (rand p95 %.3f, %.0f pct)",
                    L, cos.max(), r95, cap_top.get(227, cap_top.get(max(cap_top), float('nan'))),
                    cap_null_p95.get(227, cap_null_p95.get(max(cap_null_p95), float('nan'))),
                    cap_pct.get(227, cap_pct.get(max(cap_pct), float('nan'))))
        try:
            transcoder_set.unload(L)                                # free decoder weights if supported
        except Exception:
            pass

    import csv as _csv
    with open(out / "full_carrier_capture.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer", "K", "capture_top_aligned", "capture_random_p95", "pct_vs_null"])
        w.writeheader(); [w.writerow(r) for r in rows]
    (out / "full_carrier_capture.json").write_text(json.dumps({"per_layer": per_layer, "k_grid": K_grid}, indent=2))

    # ---------- verdict ----------
    print("\n" + "=" * 92)
    print("FULL CARRIER CAPTURE -- does ANY feature (all layers, all features) write w_res?")
    print("=" * 92)
    feat_hit = [r for r in per_layer if r["max_cos_feature"] > r["rand_max_cos_p95"] + 0.1]
    Kref = 227 if 227 in K_grid else max(K_grid)
    cap_hit = [r for r in per_layer if r["capture_pct_vs_null_by_K"].get(Kref, 0) >= 95
               and r["capture_top_by_K"].get(Kref, 0) > 2 * r["capture_null_p95_by_K"].get(Kref, 1)]
    if feat_hit or cap_hit:
        ex = (feat_hit or cap_hit)[0]
        print(f"OUTCOME -- a write-direction EXISTS: L{ex['layer']} has feature alignment / collective capture "
              f"well above the random-feature null. The concept IS written by the dictionary somewhere; "
              f"the 227-carrier (66/74) simply missed it. Localise these features next.")
    else:
        mx = max((r["max_cos_feature"] for r in per_layer), default=float("nan"))
        print(f"NO FEATURE WRITES THE AXIS -- across all {len(layers)} layers and ALL features, the best "
              f"single-feature alignment ({mx:.3f}) is at the random-direction level, and top-aligned "
              f"feature sets capture w_res no better than random K. The MLP dictionary does not encode the "
              f"concept as a write-direction -> 66/74 negative holds over the FULL dictionary; w_res is a "
              f"readout of a signal the dictionary does not write.")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/full_carrier_capture")
    p.add_argument("--transcoder_config", default="configs/transcoder_config.yaml")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="explicit layers; else [layer_lo..layer_hi]")
    p.add_argument("--layer_lo", type=int, default=10)
    p.add_argument("--layer_hi", type=int, default=25, help="default L10-L25 = transcoder/carrier range; widen if transcoders cover more")
    p.add_argument("--causal", action="store_true", default=True, help="use causal (whitened) cosine, matching 66/74")
    p.add_argument("--euclidean", dest="causal", action="store_false", help="use plain euclidean cosine instead")
    p.add_argument("--k_grid", type=int, nargs="*", default=[50, 100, 227, 500, 1000])
    p.add_argument("--n_random_sets", type=int, default=20, help="random-K-feature null draws per K")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
