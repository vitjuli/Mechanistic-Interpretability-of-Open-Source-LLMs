"""
64_concept_axis_diagnosis.py
===================================================================
Tests the PREMISE that every prior script (60-63) silently assumed:
that the concept axis is gbar = gamma(beta) - gamma(alpha), the unembedding
contrast of the answer tokens.

The paradox forcing this check:
  * SFR = 1.000 on 24 contrastive pairs  => the L24 carrier is CAUSALLY NECESSARY
    (ablating it breaks the answer). A mere correlate cannot do that.
  * along-gbar = 0.000, |cos_C(d_f, gbar)| ~ null (script 63), and steering along
    gbar / lbar does not flip (61b)  => the carrier does NOT write along gbar.
A causally-necessary mechanism that is orthogonal to gbar means one of:
  (M1) gbar is NOT the concept axis -- the naive unembedding contrast is wrong,
       and the concept lives along a DIFFERENT residual-stream direction w_res.
  (M2) downstream mediation -- the carrier writes early (L18/L24) into directions
       that later blocks transform into the gbar direction by the final layer.
  (M3) gating / attention routing -- non-additive control that neither steering
       along gbar nor along-gbar projection can detect.

D1 (CORE) discriminates M1 from M2/M3 by finding the concept axis EMPIRICALLY in
the RESIDUAL STREAM (not in transcoder feature-activations, which is what the 82%
probe / cos=0.82 / CS-ICC were actually computed on) and comparing it to gbar:

  At several depths (post-block L for L in a sweep, plus the final residual that
  the unembedding reads), capture h(p) on the decision token for all prompts,
  split TRAIN/HELD-OUT by surface_family, fit a regularised Fisher / LDA axis
  w_res that separates alpha vs beta IN RESIDUAL SPACE, verify it generalises on
  held-out, then measure cos_C(w_res, gbar) and cos_C(w_res, carrier-span).

  * cos_C(w_res, gbar) HIGH at the final residual  => gbar IS the read axis (M1
    rejected). If it is also high post-L24, the carrier paradox is real
    (necessary, axis right, but does not write along it -> M2/M3).
  * cos_C(w_res, gbar) LOW                          => gbar is the WRONG axis (M1).
    The concept is linearly carried by w_res, not by the answer-token contrast.
    Then Stage-1 geometry must be recomputed on w_res, and the negative results
    of 61/61b/62/63 are artefacts of the axis choice, not properties of the model.
  * cos_C(w_res, gbar) RISING with depth           => downstream mediation (M2):
    the gbar component is assembled in later layers.
  * cos_C(w_res, carrier-span) HIGH                 => the carrier writes toward
    the TRUE residual concept axis even though it is ~orthogonal to gbar.

D2 (secondary -- causal breadth, NOT existence; SFR already proved existence):
  Ablate the 37 carrier features (zero their activations) vs N random 37-feature
  sets from other layers, on a clean forward over all prompts, and compare the
  Delta logit distributions. carrier >> random => broadly causal; carrier ~ random
  => causal only on the narrow SFR subset.

INPUTS: concept_directions.npz (gbar, Sigma_inv) from 60_, the prompt jsonl, the
transcoders (for carrier decoder rows / ablation). Captures residuals on the fly
like 61_. Residual taps via register_forward_pre_hook on model.layers[L] (input
to block L = output of block L-1) and on model.norm (final residual).

SELF-TEST (no torch / no repo): python 64_concept_axis_diagnosis.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("concept_axis")

FEATURE_ID_RE = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")


# =====================================================================
# Geometry core (pure numpy; unit-tested by --self_test)
# =====================================================================

def causal_cos(a: np.ndarray, b: np.ndarray, Sigma_inv: np.ndarray) -> float:
    num = float(a @ Sigma_inv @ b)
    na = float(np.sqrt(max(a @ Sigma_inv @ a, 1e-30)))
    nb = float(np.sqrt(max(b @ Sigma_inv @ b, 1e-30)))
    return num / (na * nb)


def fisher_axis_residual(H: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """
    Regularised Fisher/LDA axis in residual space separating two classes.
      w = Sigma_within^{-1} (mu_1 - mu_0),  Sigma_within shrunk toward its diagonal.
    H: (n, d) residual activations; y: (n,) in {0,1}. Returns unit-raw w (d,).
    The shrinkage makes Sigma_within invertible when n < d (always, here: d=2560).
    """
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mu0 = H[y == 0].mean(0)
    mu1 = H[y == 1].mean(0)
    # pooled within-class covariance
    X0 = H[y == 0] - mu0
    X1 = H[y == 1] - mu1
    n = H.shape[0]
    Sw = (X0.T @ X0 + X1.T @ X1) / max(n - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    # shrink toward diagonal (Ledoit-Wolf-style) for invertibility at n<<d
    diag = np.diag(np.diag(Sw))
    Sw_reg = (1 - shrink) * Sw + shrink * diag
    # add a tiny ridge in case some diag entries are ~0
    ridge = 1e-6 * float(np.mean(np.diag(Sw_reg)) + 1e-12)
    Sw_reg = Sw_reg + ridge * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw_reg, (mu1 - mu0))
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 0 else w


def project_separation(H: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """How well does axis w separate the classes on this (held-out) set?"""
    s = H @ w
    s0, s1 = s[y == 0], s[y == 1]
    # point-biserial-ish: standardized mean difference (Cohen's d) + AUC
    pooled = np.sqrt(0.5 * (s0.var(ddof=1) + s1.var(ddof=1)) + 1e-30)
    d = float((s1.mean() - s0.mean()) / pooled) if pooled > 0 else float("nan")
    # AUC via Mann-Whitney U
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(s1), len(s0)
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 > 0 else float("nan")
    acc = float(max(np.mean((s > np.median(s)) == y), np.mean((s <= np.median(s)) == y)))
    return {"cohens_d": d, "auc": auc, "median_split_acc": acc}


def subspace_capture_cos(w: np.ndarray, basis: np.ndarray, Sigma_inv: np.ndarray) -> float:
    """
    cos of the angle between unit axis w and the carrier subspace span(basis rows),
    in the causal metric. 1 => w lies in the carrier span; 0 => orthogonal to it.
    """
    A = _whitener(Sigma_inv)
    Bw = (A @ basis.T)              # (d, k) whitened columns
    ww = A @ w
    if Bw.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(Bw)
    proj = Q @ (Q.T @ ww)
    denom = np.linalg.norm(ww)
    return float(np.linalg.norm(proj) / denom) if denom > 0 else 0.0


def _whitener(Sigma_inv: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(Sigma_inv)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def parse_feature_id(fid: str) -> Tuple[int, int]:
    m = FEATURE_ID_RE.match(str(fid).strip())
    if not m:
        raise ValueError(f"cannot parse feature_id {fid!r}")
    return int(m.group(1)), int(m.group(2))


# =====================================================================
# Self-test: planted residual concept axis, gbar aligned vs misaligned
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(64)
    d = 200
    evals = np.concatenate([np.linspace(15, 3, 15), np.linspace(2, 0.05, d - 15)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (Q * evals) @ Q.T; Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_inv = np.linalg.inv(Sigma + 1e-3 * np.mean(np.diag(Sigma)) * np.eye(d))

    # TRUE residual concept axis
    w_true = rng.standard_normal(d); w_true /= np.linalg.norm(w_true)
    # residuals: class signal along w_true + big nuisance variance elsewhere
    n = 400
    y = rng.integers(0, 2, n)
    nuis = rng.standard_normal((n, d)) @ np.linalg.cholesky(Sigma).T * 0.5
    H = nuis + np.outer((y * 2 - 1.0) * 2.0, w_true)

    # Case A: gbar ALIGNED with the true axis (M1 rejected)
    gbar_aligned = w_true + 0.1 * rng.standard_normal(d)
    # Case B: gbar MISALIGNED (M1 -- wrong axis)
    gbar_wrong = rng.standard_normal(d)

    # train/held-out
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
    w_res = fisher_axis_residual(H[tr], y[tr])
    sep_tr = project_separation(H[tr], y[tr], w_res)
    sep_te = project_separation(H[te], y[te], w_res)

    print("\n--- SELF TEST -------------------------------------------------")
    print(f"d={d}  recovered residual axis: train AUC={sep_tr['auc']:.3f} held-out AUC={sep_te['auc']:.3f}")
    print(f"  cos_C(w_res, w_true)      = {causal_cos(w_res, w_true, Sigma_inv):+.3f}  (expect high)")
    cosA = causal_cos(w_res, gbar_aligned, Sigma_inv)
    cosB = causal_cos(w_res, gbar_wrong, Sigma_inv)
    print(f"  cos_C(w_res, gbar_ALIGNED) = {cosA:+.3f}  (expect high -> gbar correct)")
    print(f"  cos_C(w_res, gbar_WRONG)   = {cosB:+.3f}  (expect ~0  -> gbar wrong axis, M1)")

    # carrier subspace capture: build a carrier that writes along w_true (+ortho)
    carrier_aligned = np.array([w_true + 0.3 * rng.standard_normal(d) for _ in range(8)])
    carrier_ortho = rng.standard_normal((8, d))
    capA = subspace_capture_cos(w_res, carrier_aligned, Sigma_inv)
    capB = subspace_capture_cos(w_res, carrier_ortho, Sigma_inv)
    print(f"  capture cos(w_res in carrier_ALIGNED span) = {capA:.3f} (expect high)")
    print(f"  capture cos(w_res in carrier_ORTHO span)   = {capB:.3f} (8 random dirs in {d}-d)")

    assert sep_te["auc"] > 0.8, f"residual axis must generalise (held-out AUC {sep_te['auc']:.3f})"
    assert causal_cos(w_res, w_true, Sigma_inv) > 0.7, "must recover the true axis"
    assert cosA > 0.6, f"aligned gbar must read high (got {cosA:.3f})"
    assert abs(cosB) < 0.4, f"wrong gbar must read ~0 (got {cosB:.3f})"
    assert capA > capB, "carrier writing along axis must capture more than orthogonal carrier"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(D1 can detect whether gbar is the residual concept axis, and whether")
    print(" the carrier span contains it)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Feature resolution (same options as 62/63)
# =====================================================================

def _resolve_features(args) -> Dict[int, List[int]]:
    feats: Dict[int, List[int]] = {}

    def add(L, fi): feats.setdefault(L, []).append(fi)

    if args.features:
        for tok in args.features.split(","):
            L, fi = parse_feature_id(tok); add(L, fi)
        return feats
    if args.feature_file:
        for line in open(args.feature_file):
            line = line.strip()
            if line:
                L, fi = parse_feature_id(line); add(L, fi)
        return feats
    if args.cluster_labels:
        import pandas as pd
        cl = pd.read_csv(args.cluster_labels)
        if "feature_id" not in cl.columns:
            cl = cl.rename(columns={cl.columns[0]: "feature_id"})
        cl["feature_id"] = cl["feature_id"].astype(str)

        def norm_id(c):
            s = str(c).strip().lstrip("Cc")
            return s[:-2] if s.endswith(".0") else s
        wanted = {norm_id(c) for c in args.clusters.split(",")} if args.clusters else None
        for fid, c in zip(cl["feature_id"], cl[args.cluster_col].apply(norm_id)):
            if wanted is None or c in wanted:
                try:
                    L, fi = parse_feature_id(fid)
                except ValueError:
                    continue
                if args.layers and L not in args.layers:
                    continue
                add(L, fi)
        return feats
    raise SystemExit("provide --features, --feature_file, or --cluster_labels. See --help.")


# =====================================================================
# Real run
# =====================================================================

def _getattr_chain(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def run_real(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cd = np.load(args.concept_npz)
    gbar = cd["gbar"].astype(np.float64)
    Sigma_inv = cd["Sigma_inv"].astype(np.float64)
    d = gbar.shape[0]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _getattr_chain(model, "model.layers")
    n_layers = len(blocks)
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts})
    rng.shuffle(fams)
    n_tr = int(round(len(fams) * args.train_frac))
    train_fams = set(fams[:n_tr])

    # depth taps: post-block L (= input to block L+1) for L in sweep; + final residual
    sweep = args.sweep_layers or list(range(args.sweep_min, args.sweep_max + 1))
    # tap "post-block L" by pre-hooking block L+1; final residual by pre-hooking model.norm
    tap_layers = sorted(set(sweep))
    want_final = True

    def capture_all(ptext: str) -> Dict[str, np.ndarray]:
        inputs = tok([ptext], return_tensors="pt").to(args.device)
        grabbed: Dict[str, np.ndarray] = {}
        handles = []
        for L in tap_layers:
            if L + 1 < n_layers:
                def _mk(L=L):
                    def _pre(mod, a_in):
                        grabbed[f"postL{L}"] = a_in[0][0, -1, :].detach().float().cpu().numpy()
                        return None
                    return _pre
                handles.append(blocks[L + 1].register_forward_pre_hook(_mk(), with_kwargs=False))
        if want_final:
            norm_mod = _getattr_chain(model, "model.norm")
            def _mkf():
                def _pre(mod, a_in):
                    grabbed["final"] = a_in[0][0, -1, :].detach().float().cpu().numpy()
                    return None
                return _pre
            handles.append(norm_mod.register_forward_pre_hook(_mkf(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    # ---- capture residuals for all prompts ----
    logger.info("Capturing residuals at %d depths + final for %d prompts...",
                len(tap_layers), len(prompts))
    taps = [f"postL{L}" for L in tap_layers] + ["final"]
    H = {t: [] for t in taps}
    y, is_train = [], []
    for i, p in enumerate(prompts):
        if i > 0 and i % 50 == 0:
            logger.info("  captured residuals: %d/%d prompts", i, len(prompts))
        g = capture_all(p["prompt"])
        for t in taps:
            H[t].append(g[t])
        y.append(1 if p["correct_answer"].strip() == "beta" else 0)  # 1=beta,0=alpha
        is_train.append(p["surface_family"] in train_fams)
    y = np.array(y); is_train = np.array(is_train)
    for t in taps:
        H[t] = np.array(H[t], dtype=np.float64)
    logger.info("  done: %d alpha, %d beta; %d train families, %d held-out families",
                int((y==0).sum()), int((y==1).sum()),
                len(train_fams), len(fams)-len(train_fams))

    # ---- carrier decoder span (for capture-cos) ----
    feats = _resolve_features(args)
    layers = sorted(feats.keys())
    from src.transcoder import load_transcoder_set
    # For D2 we'll need random rows from many layers too (not just carrier layers).
    d2_pool_layers = list(range(args.sweep_min, args.sweep_max + 1)) if args.run_d2 else []
    all_tc_layers = sorted(set(layers) | set(d2_pool_layers))
    tset = load_transcoder_set(model_size=args.model_size, device=args.device,
                               lazy_load=True, layers=all_tc_layers)

    # Cache W_dec per layer as numpy — eliminates slow _get_decoder_vectors per call.
    # Critical for D2 which would otherwise do 50×538×2 ≈ 54000 decoder calls.
    W_dec_cache: Dict[int, np.ndarray] = {}

    def _ensure_wdec_cached(layer: int) -> np.ndarray:
        if layer not in W_dec_cache:
            tc = tset[layer]
            if hasattr(tc, "W_dec"):
                logger.info("Caching W_dec for layer %d (shape=%s)...", layer, tc.W_dec.shape)
                W_dec_cache[layer] = tc.W_dec.detach().float().cpu().numpy().astype(np.float64)
                logger.info("  cached %.1f MB", W_dec_cache[layer].nbytes / 1e6)
            else:
                raise RuntimeError(f"transcoder for L{layer} has no W_dec")
        return W_dec_cache[layer]

    def decoder_rows(L, idxs):
        W = _ensure_wdec_cached(L)
        return W[np.asarray(idxs, dtype=np.int64)]

    carrier_basis = np.vstack([decoder_rows(L, feats[L]) for L in layers])

    # ---- D1: residual Fisher axis per depth, CV, cos with gbar and carrier span ----
    # Null band: cos_C of w_res against random directions (so cos thresholds are
    # calibrated, not absolute -- random dirs have nonzero cos in finite d).
    A_white = _whitener(Sigma_inv)
    def null_cos_band(w_res, n=500):
        ww = A_white @ w_res; ww /= (np.linalg.norm(ww) + 1e-30)
        cwith = []
        for _ in range(n):
            r = rng.standard_normal(d); rw = A_white @ r; rw /= (np.linalg.norm(rw) + 1e-30)
            cwith.append(abs(float(ww @ rw)))
        a = np.array(cwith)
        return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99))}

    d1 = []
    for t in taps:
        Htr, ytr = H[t][is_train], y[is_train]
        Hte, yte = H[t][~is_train], y[~is_train]
        w_res = fisher_axis_residual(Htr, ytr, shrink=args.shrink)
        sep_te = project_separation(Hte, yte, w_res)
        cos_g = causal_cos(w_res, gbar, Sigma_inv)
        cap = subspace_capture_cos(w_res, carrier_basis, Sigma_inv)
        nb = null_cos_band(w_res)
        d1.append({
            "tap": t,
            "heldout_auc": sep_te["auc"], "heldout_d": sep_te["cohens_d"],
            "heldout_acc": sep_te["median_split_acc"],
            "cos_C_w_gbar": cos_g,
            "cos_C_w_gbar_null_p95": nb["p95"],
            "cos_C_w_gbar_excess_over_null": abs(cos_g) - nb["p95"],
            "carrier_capture_cos": cap,
        })
        np.save(out / f"w_res_{t}.npy", w_res)
        logger.info("D1 %6s: heldout AUC=%.3f  cos_C(w_res,gbar)=%+.3f (null p95=%.3f)  carrier-capture=%.3f",
                    t, sep_te["auc"], cos_g, nb["p95"], cap)

    # ---- D2: carrier vs random ablation Delta logit (clean forward) ----
    def clean_and_ablated_dl(ptext, abl: Dict[int, List[int]]):
        inputs = tok([ptext], return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = model(**inputs, use_cache=False)
        lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
        dl_clean = float(lp[beta_id] - lp[alpha_id])
        # ablate: zero chosen feature activations by subtracting their decoder
        # contribution from the block output (post_attention_layernorm path as 52/53)
        handles = []
        for L, idxs in abl.items():
            rows = torch.tensor(decoder_rows(L, idxs), dtype=torch.float32, device=args.device)
            tc = tset[L]
            def _mk(L=L, rows=rows, tc=tc):
                def _hook(mod, inp, outp):
                    # outp is MLP-output add; recompute feature acts and remove their writes
                    return outp
                return _hook
            # NOTE: exact ablation requires the transcoder forward; we approximate by
            # projecting the chosen decoder directions out of the block's residual add.
            block = blocks[L]
            def _mkpre(L=L, rows=rows):
                def _pre(mod, a_in):
                    hs = a_in[0].clone()
                    h = hs[0, -1, :]
                    # remove components along each decoder row (orthogonal projection-out)
                    for r in rows:
                        h = h - (torch.dot(h, r) / (torch.dot(r, r) + 1e-8)) * r
                    hs[0, -1, :] = h
                    return (hs,)
                return _pre
            handles.append(block.register_forward_pre_hook(_mkpre(), with_kwargs=False))
        try:
            with torch.no_grad():
                o2 = model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), 0)
        dl_abl = float(lp2[beta_id] - lp2[alpha_id])
        return dl_clean, dl_abl

    d2 = None
    if args.run_d2:
        sub = prompts if not args.d2_max else prompts[: args.d2_max]
        carrier_abl = {L: feats[L] for L in layers}
        carrier_eff = []
        for p in sub:
            dlc, dla = clean_and_ablated_dl(p["prompt"], carrier_abl)
            carrier_eff.append(abs(dlc - dla))
        # random sets of same total size and layer profile
        sizes = {L: len(feats[L]) for L in layers}
        rand_eff_means = []
        for s in range(args.n_random_d2):
            abl = {}
            for L, k in sizes.items():
                tc = tset[L]
                nfeat = tc.W_dec.shape[0] if hasattr(tc, "W_dec") else args.d_transcoder
                abl[L] = [int(rng.integers(nfeat)) for _ in range(k)]
            eff = []
            for p in sub:
                dlc, dla = clean_and_ablated_dl(p["prompt"], abl)
                eff.append(abs(dlc - dla))
            rand_eff_means.append(float(np.mean(eff)))
        carrier_mean = float(np.mean(carrier_eff))
        rand_arr = np.array(rand_eff_means)
        pctl = float((rand_arr < carrier_mean).mean())
        d2 = {
            "carrier_mean_abs_dlogit": carrier_mean,
            "random_mean_abs_dlogit_band": {
                "mean": float(rand_arr.mean()), "p05": float(np.percentile(rand_arr, 5)),
                "p95": float(np.percentile(rand_arr, 95))},
            "carrier_percentile_vs_random": pctl,
            "n_prompts": len(sub), "n_random_sets": args.n_random_d2,
        }
        logger.info("D2: carrier |Δlogit|=%.4f vs random band mean=%.4f (carrier at %.0f%% of null)",
                    carrier_mean, rand_arr.mean(), 100 * pctl)

    # ---- verdict ----
    final_rec = next(r for r in d1 if r["tap"] == "final")
    final_cos = final_rec["cos_C_w_gbar"]
    final_auc = final_rec["heldout_auc"]
    final_excess = final_rec["cos_C_w_gbar_excess_over_null"]
    final_null = final_rec["cos_C_w_gbar_null_p95"]
    post_cos = [r["cos_C_w_gbar"] for r in d1 if r["tap"].startswith("postL")]
    rising = len(post_cos) >= 2 and post_cos[-1] > post_cos[0] + 0.15

    if final_auc < 0.65:
        verdict = ("NO LINEAR RESIDUAL AXIS at final residual (held-out AUC < 0.65): the "
                   "concept may not be linearly readable in the residual stream at all; "
                   "revisit whether alpha/beta is linearly represented.")
    elif abs(final_cos) >= args.cos_hi and final_excess > 0:
        verdict = (f"gbar IS the residual concept axis (cos_C={final_cos:+.2f}, null p95={final_null:.2f}, "
                   "at final). M1 rejected. Carrier is necessary (SFR=1.0) yet does not write along "
                   "gbar (63) -> mechanism is DOWNSTREAM-MEDIATED or GATING (M2/M3); "
                   + ("depth sweep shows the gbar component RISING -> supports M2 (mediation)."
                      if rising else
                      "depth sweep does NOT show a rising gbar component -> favours M3 (gating/routing)."))
    elif final_excess <= 0 or abs(final_cos) <= args.cos_lo:
        verdict = (f"gbar is the WRONG axis (cos_C={final_cos:+.2f} vs null p95={final_null:.2f} at final, "
                   f"held-out AUC={final_auc:.2f}): w_res is NOT distinguishable from random alignment with "
                   "gbar. The residual concept axis differs from the answer-token contrast. RECOMPUTE "
                   "Stage-1 geometry on w_res; the negatives of 61/61b/62/63 are likely axis-choice artefacts.")
    else:
        verdict = (f"AMBIGUOUS: cos_C(w_res,gbar)={final_cos:+.2f} (null p95={final_null:.2f}, excess "
                   f"{final_excess:+.2f}) at final. Partial alignment; report both and recompute key tests "
                   "on w_res as a robustness check.")

    cap_final = next(r["carrier_capture_cos"] for r in d1 if r["tap"] == "final")
    results = {
        "params": {"concept_npz": str(args.concept_npz), "taps": taps,
                   "train_frac": args.train_frac, "shrink": args.shrink,
                   "carrier_features": {str(L): feats[L] for L in layers}},
        "D1": d1, "D2": d2,
        "final_cos_C_w_gbar": final_cos, "final_heldout_auc": final_auc,
        "carrier_capture_cos_final": cap_final, "gbar_component_rising": rising,
        "verdict": verdict,
    }
    with open(out / "concept_axis_diagnosis.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- console ----
    print("\n" + "=" * 86)
    print("CONCEPT-AXIS DIAGNOSIS  --  is gbar the residual concept axis?")
    print("=" * 86)
    print(f"{'tap':>8} {'heldoutAUC':>11} {'cos_C(w,gbar)':>14} {'null_p95':>9} {'excess':>8} {'carrier_cap':>12}")
    for r in d1:
        print(f"{r['tap']:>8} {r['heldout_auc']:>11.3f} {r['cos_C_w_gbar']:>+14.3f} "
              f"{r['cos_C_w_gbar_null_p95']:>9.3f} {r['cos_C_w_gbar_excess_over_null']:>+8.3f} "
              f"{r['carrier_capture_cos']:>12.3f}")
    if d2:
        print(f"\nD2 ablation: carrier |Δlogit|={d2['carrier_mean_abs_dlogit']:.4f}  "
              f"random band=[{d2['random_mean_abs_dlogit_band']['p05']:.4f}, "
              f"{d2['random_mean_abs_dlogit_band']['p95']:.4f}]  "
              f"carrier at {100*d2['carrier_percentile_vs_random']:.0f}%ile")
    print("\nVERDICT: " + verdict)
    print(f"\nwrote: {out}/concept_axis_diagnosis.json + w_res_*.npy")
    print("=" * 86)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concept_npz", type=str,
                   default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--prompts", type=str, default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", type=str, default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_size", type=str, default="4b")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alpha_answer", type=str, default=" alpha")
    p.add_argument("--beta_answer", type=str, default=" beta")
    # feature selection
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--feature_file", type=str, default=None)
    p.add_argument("--cluster_labels", type=str, default=None)
    p.add_argument("--cluster_col", type=str, default="coimp_louvain")
    p.add_argument("--clusters", type=str, default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    # depth sweep
    p.add_argument("--sweep_layers", type=int, nargs="*", default=None)
    p.add_argument("--sweep_min", type=int, default=14)
    p.add_argument("--sweep_max", type=int, default=25)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1, help="LDA shrinkage toward diagonal")
    p.add_argument("--cos_hi", type=float, default=0.5, help="cos above this => gbar is the axis")
    p.add_argument("--cos_lo", type=float, default=0.2, help="cos below this => gbar is wrong axis")
    # D2
    p.add_argument("--run_d2", action="store_true", help="run carrier-vs-random ablation (slower)")
    p.add_argument("--d2_max", type=int, default=None, help="cap prompts for D2")
    p.add_argument("--n_random_d2", type=int, default=50)
    p.add_argument("--d_transcoder", type=int, default=163840)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
