"""
65_concept_axis_causality.py   [JOB 1 / 4 -- TERMINAL ARBITER]
===================================================================
Decides the ONE remaining binary question: is the residual concept axis w_res
(which 64 showed is linearly decodable at held-out AUC 0.99, yet ~orthogonal to
gbar) actually CAUSAL, or only DECODABLE?

  decodable  = a probe can read alpha/beta off the residual stream (AUC 0.99).
  causal     = pushing the residual along that axis CHANGES the model's output.
These are different. AUC 0.99 proves the first. Only steering proves the second.

THE DECISIVE CONTROL (this is why the run matters): w_res is the direction of
MAXIMUM class separation (Fisher/LDA). Steering along it is almost guaranteed to
move SOMETHING -- that is nearly tautological, exactly as gbar would have been.
So the real question is not "does w_res steering flip?" but "does it flip MORE
than an ARBITRARY separating direction of the same norm?". We build that null
explicitly: SHUFFLED-LABEL Fisher (N=10) -- fit the same LDA on PERMUTED labels,
giving directions that separate the data by chance. If real w_res flips no more
than shuffled-label directions, the flip is an artefact of pushing along any
high-variance separating axis, NOT causality of the concept.

CONTROLS (all unit-raw-L2, norm-matched per c): w_res vs
  - shuffled-label Fisher x10   (DECISIVE: arbitrary separating directions)
  - random x N seeds            (generic energy)
  - S_perp                      (M-orthogonal complement of span{w_res})
  - format                      (top raw-activation PC = the 17%-variance axis)

DESIGN (E2a): per-layer steering. The concept is decodable at every depth, but
causal control may live only in a window. We steer at the residual AFTER block L
(= input to block L+1) using the w_res FOUND AT THAT DEPTH, with the per-depth
sigma. Steering layers default {post-L18, post-L21, post-L24, final-pre-norm}.
A measurement-only tap at final-POST-norm answers the LayerNorm caveat: is gbar
aligned with the readout axis AFTER normalisation (which the unembedding reads)?

Also computes:
  * LOGIT-LENS  W_U @ w_res  top-k tokens -- if w_res is the model's readout
    direction, its top logits should be meaningful (alpha/beta/physics), not noise.
  * per-depth held-out AUC of w_res (re-confirms decodability), and cos_C(w_res, gbar).

VERDICT (locked before the run):
  OUTCOME 1 (CAUSAL): w_res flips >= tau on some layer AND beats the shuffled-label
     band (p95) AND random/sperp/format -> the concept axis is causal. Positive
     result on the RIGHT axis; carrier is a statistical readout, not the mechanism.
     => recompute Stage-1 geometry on w_res; write a positive chapter.
  OUTCOME 2 (DECODABLE-NOT-CAUSAL): w_res flips but NOT above the shuffled-label
     band -> flip is generic to any separating direction. No causal low-dim axis.
     => strong negative result.
  OUTCOME 3 (NO CONTROL): w_res does not flip on any layer (like gbar) -> the
     concept is committed before the residual can be pushed. Strongest negative.

NORMALISATION fixed exactly as 61b: unit-raw-L2 directions; sigma = std<h, dir>_2
on clean TRAIN residuals at the SAME depth; step of "c sigma" has raw L2 = |c| sigma,
identical across all directions -> exact norm-match.

INPUTS: concept_directions.npz (gbar, Sigma_inv) from 60_, prompts jsonl, base
model Qwen/Qwen3-4B. Recomputes w_res from captured residuals (self-contained;
also writes them so 66/68 could reuse).

SELF-TEST (no torch/repo): python 65_concept_axis_causality.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("axis_causality")


# =====================================================================
# Geometry / control helpers (pure numpy; unit-tested by --self_test)
# =====================================================================

def whitener(Sigma_inv: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(Sigma_inv)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def causal_cos(a, b, Sigma_inv) -> float:
    num = float(a @ Sigma_inv @ b)
    na = float(np.sqrt(max(a @ Sigma_inv @ a, 1e-30)))
    nb = float(np.sqrt(max(b @ Sigma_inv @ b, 1e-30)))
    return num / (na * nb)


def unit_raw(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def fisher_axis(H: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Regularised LDA axis separating classes in residual space; unit raw-L2."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    n = H.shape[0]
    Sw = (X0.T @ X0 + X1.T @ X1) / max(n - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    return unit_raw(w)


def auc_of_axis(H, y, w) -> float:
    s = H @ w
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def sperp_unit(w_raw, A, Ainv, rng) -> np.ndarray:
    """Unit-raw direction in the M-orthogonal complement of span{w}, M=Sigma_inv."""
    qw = unit_raw(A @ w_raw)
    v = rng.standard_normal(A.shape[0]); vw = v - qw * (qw @ v)
    nv = np.linalg.norm(vw)
    if nv < 1e-12:
        vw = rng.standard_normal(A.shape[0]); vw = vw - qw * (qw @ vw); nv = np.linalg.norm(vw)
    return unit_raw(Ainv @ (vw / nv))


def directional_flip(dl_clean, dl_steered, toward) -> int:
    if toward == "beta":
        return int(dl_clean < 0 and dl_steered > 0)
    return int(dl_clean > 0 and dl_steered < 0)


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


# =====================================================================
# Self-test: causal vs decodable-only, with the shuffled-label control
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(65)
    d = 160
    evals = np.concatenate([np.linspace(12, 3, 12), np.linspace(2, 0.05, d - 12)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (Q * evals) @ Q.T; Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_inv = np.linalg.inv(Sigma + 1e-3 * np.mean(np.diag(Sigma)) * np.eye(d))
    A = whitener(Sigma_inv); Ainv = np.linalg.inv(A)

    g_read = unit_raw(rng.standard_normal(d))           # the model's readout direction
    w_sep = unit_raw(rng.standard_normal(d))            # a separating-but-not-readout dir
    n = 500
    y = rng.integers(0, 2, n)

    def make_H(sep_dir, amp=1.0):
        nuis = rng.standard_normal((n, d)) @ np.linalg.cholesky(Sigma).T * 0.5
        return nuis + np.outer((y * 2 - 1.0) * amp, sep_dir)

    # CAUSAL case: classes separate ALONG the readout direction (modest amplitude)
    H_causal = make_H(g_read, amp=1.0)
    # DECODABLE-ONLY case: classes separate along w_sep, ~orthogonal to g_read
    w_sep_o = unit_raw(w_sep - g_read * (g_read @ w_sep))
    H_decod = make_H(w_sep_o, amp=1.0)

    def dl(h):  # linear readout the "model" uses
        return float(h @ g_read)

    def steer_flip_rate(H, alpha_idx, w_dir, c):
        flips = []
        for i in alpha_idx:
            h = H[i].copy()
            if dl(h) >= 0:
                continue
            flips.append(directional_flip(dl(h), dl(h + c * w_dir), "beta"))
        return float(np.mean(flips)) if flips else 0.0

    print("\n--- SELF TEST -------------------------------------------------")
    C_TEST = 6.0   # steer magnitude large enough to cross the margin if causal
    for name, H in [("CAUSAL (sep along readout)", H_causal),
                    ("DECODABLE-ONLY (sep ⟂ readout)", H_decod)]:
        w_res = fisher_axis(H, y)
        auc = auc_of_axis(H, y, w_res)
        alpha_idx = [i for i in range(n) if dl(H[i]) < 0][:60]
        fr_w = steer_flip_rate(H, alpha_idx, unit_raw(w_res), C_TEST)
        shuf = np.array([steer_flip_rate(H, alpha_idx, unit_raw(fisher_axis(H, rng.permutation(y))), C_TEST)
                         for _ in range(10)])
        print(f"\n{name}:")
        print(f"  Fisher AUC = {auc:.3f}  (decodable either way)")
        print(f"  flip(w_res)        = {fr_w:.2f}")
        print(f"  flip(shuffled) band= mean {shuf.mean():.2f}  p95 {np.percentile(shuf,95):.2f}")
        print(f"  cos_C(w_res, g_read) = {causal_cos(w_res, g_read, Sigma_inv):+.3f}")

    # Assertions
    wc = fisher_axis(H_causal, y); idx_c = [i for i in range(n) if dl(H_causal[i]) < 0][:60]
    fr_c = steer_flip_rate(H_causal, idx_c, unit_raw(wc), C_TEST)
    shuf_c = np.array([steer_flip_rate(H_causal, idx_c, unit_raw(fisher_axis(H_causal, rng.permutation(y))), C_TEST)
                       for _ in range(10)])
    wd = fisher_axis(H_decod, y); idx_d = [i for i in range(n) if dl(H_decod[i]) < 0][:60]
    fr_d = steer_flip_rate(H_decod, idx_d, unit_raw(wd), C_TEST)

    assert auc_of_axis(H_decod, y, wd) > 0.8, "decodable-only must still be decodable (high AUC)"
    assert fr_c > np.percentile(shuf_c, 95), "CAUSAL: w_res must beat shuffled-label band"
    assert fr_c > 0.5, f"CAUSAL: w_res should flip (got {fr_c:.2f})"
    assert fr_d < 0.2, f"DECODABLE-ONLY: w_res must NOT flip (got {fr_d:.2f})"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(test distinguishes causal axis from decodable-only via the shuffled-label control)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def _chain(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cd = np.load(args.concept_npz)
    gbar = cd["gbar"].astype(np.float64)
    Sigma_inv = cd["Sigma_inv"].astype(np.float64)
    d = gbar.shape[0]
    A = whitener(Sigma_inv); Ainv = np.linalg.inv(A)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks)
    norm_mod = _chain(model, "model.norm")
    W_U = (model.lm_head.weight if hasattr(model, "lm_head")
           else model.get_output_embeddings().weight).detach().float().cpu().numpy()
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    n_tr = int(round(len(fams) * args.train_frac)); train_fams = set(fams[:n_tr])

    steer_layers = args.steer_layers or [18, 21, 24]    # residual AFTER block L (=input to L+1)
    # taps: postL{L} via pre-hook block L+1; 'final' via pre-hook model.norm (pre-norm);
    #       'final_pn' via post-hook model.norm (post-norm, measurement only)
    def capture(ptext):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in steer_layers:
            if L + 1 < n_layers:
                def mk(L=L):
                    def pre(m, a): g[f"postL{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                    return pre
                handles.append(blocks[L + 1].register_forward_pre_hook(mk(), with_kwargs=False))
        def mkf():
            def pre(m, a): g["final"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
            return pre
        handles.append(norm_mod.register_forward_pre_hook(mkf(), with_kwargs=False))
        def mkpn():
            def post(m, i, o):
                t = o[0] if isinstance(o, tuple) else o
                g["final_pn"] = t[0, -1, :].detach().float().cpu().numpy()
            return post
        handles.append(norm_mod.register_forward_hook(mkpn()))
        try:
            with torch.no_grad():
                model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return g

    logger.info("Capturing residuals for %d prompts...", len(prompts))
    taps = [f"postL{L}" for L in steer_layers] + ["final", "final_pn"]
    H = {t: [] for t in taps}; y = []; tr_mask = []
    for i, p in enumerate(prompts):
        g = capture(p["prompt"])
        for t in taps:
            H[t].append(g[t])
        y.append(1 if p["correct_answer"].strip() == "beta" else 0)
        tr_mask.append(p["surface_family"] in train_fams)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))
    for t in taps:
        H[t] = np.array(H[t], np.float64)
    y = np.array(y); tr_mask = np.array(tr_mask)

    # per-tap: w_res (real), shuffled-label wres x N, format dir, sigma, AUC, cos with gbar, logit-lens
    geom = {}
    for t in taps:
        Htr, ytr = H[t][tr_mask], y[tr_mask]
        w_res = fisher_axis(Htr, ytr, args.shrink)
        auc_te = auc_of_axis(H[t][~tr_mask], y[~tr_mask], w_res)
        shuf_dirs = [fisher_axis(Htr, rng.permutation(ytr), args.shrink) for _ in range(args.n_shuffle)]
        Hc = Htr - Htr.mean(0); _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        format_dir = unit_raw(Vt[0])
        sigma = float(np.std(Htr @ w_res))
        ll = W_U @ w_res
        topk = np.argsort(ll)[::-1][:args.logit_lens_k]
        botk = np.argsort(ll)[:args.logit_lens_k]
        geom[t] = {
            "w_res": w_res, "shuf_dirs": shuf_dirs, "format_dir": format_dir,
            "sigma": sigma, "heldout_auc": auc_te, "cos_C_gbar": causal_cos(w_res, gbar, Sigma_inv),
            "sperp": sperp_unit(w_res, A, Ainv, rng),
            "logit_lens_top": [tok.decode([int(i)]) for i in topk],
            "logit_lens_bot": [tok.decode([int(i)]) for i in botk],
        }
        np.save(out / f"w_res65_{t}.npy", w_res)
        logger.info("tap %9s: heldout AUC=%.3f  cos_C(w_res,gbar)=%+.3f  sigma=%.4f  top-logits=%s",
                    t, auc_te, geom[t]["cos_C_gbar"], sigma, geom[t]["logit_lens_top"][:5])

    # ---- steering machinery (pre-hook add at the tap's module) ----
    def module_for(tap):
        if tap == "final":
            return norm_mod
        L = int(tap.replace("postL", ""))
        return blocks[L + 1]

    def clean_dl(ptext):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = model(**inp, use_cache=False)
        lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
        return float(lp[beta_id] - lp[alpha_id])

    def steered_dl(ptext, tap, delta):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
        def pre(m, a):
            hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
        h = module_for(tap).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
        finally:
            h.remove()
        lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
        return float(lp[beta_id] - lp[alpha_id])

    held = [p for p in prompts if p["surface_family"] not in train_fams]
    held_alpha = [p for p in held if p["correct_answer"].strip() == "alpha"]
    held_beta = [p for p in held if p["correct_answer"].strip() == "beta"]
    if args.max_targets:
        held_alpha, held_beta = held_alpha[:args.max_targets], held_beta[:args.max_targets]
    pos = sorted([c for c in args.c_grid if c > 0])

    # steering taps = only pre-norm points (final_pn is measurement-only)
    steer_taps = [f"postL{L}" for L in steer_layers] + ["final"]
    rows = []

    def run(tap, targets, toward, cvals):
        sig = geom[tap]["sigma"]
        dirs = {"w_res": unit_raw(geom[tap]["w_res"]),
                "sperp": unit_raw(geom[tap]["sperp"]),
                "format": unit_raw(geom[tap]["format_dir"])}
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        for j, sd in enumerate(geom[tap]["shuf_dirs"]):
            dirs[f"shuffled{j}"] = unit_raw(sd)
        for tgt in targets:
            dlc = clean_dl(tgt["prompt"])
            if toward == "beta" and not (dlc < 0):
                continue
            if toward == "alpha" and not (dlc > 0):
                continue
            for c in cvals:
                for name, vec in dirs.items():
                    step = (c * sig) * vec
                    dls = steered_dl(tgt["prompt"], tap, step)
                    rows.append({"tap": tap, "toward": toward, "dir": name, "c": c,
                                 "dl_clean": dlc, "dl_steered": dls,
                                 "flip": directional_flip(dlc, dls, toward),
                                 "step_norm": float(np.linalg.norm(step))})

    for tap in steer_taps:
        logger.info("steering at %s (alpha->beta, %d targets)...", tap, len(held_alpha))
        run(tap, held_alpha, "beta", pos)
        logger.info("steering at %s (beta->alpha, %d targets)...", tap, len(held_beta))
        run(tap, held_beta, "alpha", [-c for c in pos])

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "axis_causality_curve.csv", index=False)

    def grp(k): return ("random" if k.startswith("random")
                        else "shuffled" if k.startswith("shuffled") else k)
    df["dg"] = df["dir"].map(grp)

    summary = {"params": {"steer_taps": steer_taps, "c_grid_pos": pos,
                          "n_shuffle": args.n_shuffle, "n_random": args.n_random,
                          "train_frac": args.train_frac},
               "geometry": {t: {"heldout_auc": geom[t]["heldout_auc"],
                                "cos_C_gbar": geom[t]["cos_C_gbar"],
                                "logit_lens_top": geom[t]["logit_lens_top"],
                                "logit_lens_bot": geom[t]["logit_lens_bot"]} for t in taps},
               "dose_response": [], "specificity": [], "verdict": None}

    # dose-response + specificity (w_res vs shuffled band, the decisive comparison)
    best = {"tap": None, "flip": -1, "shuf_p95": None, "beats_shuffled": False, "c": None}
    for tap in steer_taps:
        for arm in ("beta", "alpha"):
            sub = df[(df.tap == tap) & (df.toward == arm)]
            if sub.empty:
                continue
            for grpname in ["w_res", "shuffled", "random", "sperp", "format"]:
                gg = sub[sub.dg == grpname].groupby("c")["flip"].mean().sort_index()
                if gg.empty:
                    continue
                summary["dose_response"].append({
                    "tap": tap, "arm": arm, "dir": grpname,
                    "flip_by_c": {str(c): round(float(v), 3) for c, v in gg.items()},
                    "flip_overall": round(float(sub[sub.dg == grpname]["flip"].mean()), 3),
                })
            # specificity per c: w_res vs shuffled p95
            for c in sorted(sub["c"].unique()):
                sc = sub[sub.c == c]
                wf = float(sc[sc.dg == "w_res"]["flip"].mean()) if (sc.dg == "w_res").any() else float("nan")
                shuf_rates = sc[sc.dg == "shuffled"].groupby(sc[sc.dg == "shuffled"]["dir"])["flip"].mean().values \
                    if (sc.dg == "shuffled").any() else np.array([])
                shuf_p95 = float(np.percentile(shuf_rates, 95)) if shuf_rates.size else float("nan")
                rand_max = float(sc[sc.dg == "random"]["flip"].mean()) if (sc.dg == "random").any() else float("nan")
                sperp_v = float(sc[sc.dg == "sperp"]["flip"].mean()) if (sc.dg == "sperp").any() else float("nan")
                fmt_v = float(sc[sc.dg == "format"]["flip"].mean()) if (sc.dg == "format").any() else float("nan")
                beats = (not np.isnan(wf)) and (not np.isnan(shuf_p95)) and wf > shuf_p95 \
                    and wf > max([v for v in [rand_max, sperp_v, fmt_v] if not np.isnan(v)] or [0])
                summary["specificity"].append({
                    "tap": tap, "arm": arm, "c": c, "flip_w_res": round(wf, 3),
                    "shuffled_p95": round(shuf_p95, 3) if not np.isnan(shuf_p95) else None,
                    "random": round(rand_max, 3) if not np.isnan(rand_max) else None,
                    "sperp": round(sperp_v, 3) if not np.isnan(sperp_v) else None,
                    "format": round(fmt_v, 3) if not np.isnan(fmt_v) else None,
                    "w_res_beats_all_controls": bool(beats),
                })
                if (not np.isnan(wf)) and wf > best["flip"]:
                    best = {"tap": tap, "flip": wf, "shuf_p95": shuf_p95,
                            "beats_shuffled": bool(beats), "c": c, "arm": arm}

    max_wres_flip = max((s["flip_w_res"] for s in summary["specificity"]), default=0.0)
    any_beats = any(s["w_res_beats_all_controls"] and s["flip_w_res"] >= args.tau_flip
                    for s in summary["specificity"])

    if max_wres_flip < args.tau_flip:
        verdict = (f"OUTCOME 3 (NO CONTROL): w_res does not reach tau_flip={args.tau_flip} at any "
                   f"layer (max {max_wres_flip:.2f}). Like gbar, the decodable axis is not steerable "
                   "-> concept committed before residual can be pushed. STRONGEST NEGATIVE result.")
    elif any_beats:
        verdict = (f"OUTCOME 1 (CAUSAL): w_res reaches tau_flip AND beats the shuffled-label band "
                   f"and all controls (best: {best['tap']} arm={best.get('arm')} flip={best['flip']:.2f} "
                   f"vs shuffled p95={best['shuf_p95']:.2f}). The residual concept axis is CAUSAL. "
                   "Positive result on the RIGHT axis; carrier (cap 0.13, D2 68%ile) is a statistical "
                   "readout, not the mechanism. => recompute Stage-1 geometry on w_res.")
    else:
        verdict = (f"OUTCOME 2 (DECODABLE-NOT-CAUSAL): w_res flips (max {max_wres_flip:.2f}) but NOT above "
                   "the shuffled-label band -> the flip is generic to ANY separating direction of the "
                   "same norm, not causality of the concept. No low-dim causal axis exists. STRONG NEGATIVE.")
    summary["verdict"] = verdict
    summary["max_wres_flip"] = max_wres_flip

    with open(out / "axis_causality_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # console
    print("\n" + "=" * 88)
    print("CONCEPT-AXIS CAUSALITY  --  is the residual axis w_res causal? (E2a)")
    print("=" * 88)
    print("Per-tap geometry (decodability + alignment + logit-lens):")
    for t in taps:
        g = geom[t]
        print(f"  {t:9s} heldoutAUC={g['heldout_auc']:.3f}  cos_C(w_res,gbar)={g['cos_C_gbar']:+.3f}  "
              f"top-logits={g['logit_lens_top'][:5]}")
    print("\nSteering dose-response  flip-rate(w_res) vs shuffled p95 (the decisive control):")
    for tap in steer_taps:
        for arm in ("beta", "alpha"):
            specs = [s for s in summary["specificity"] if s["tap"] == tap and s["arm"] == arm]
            if not specs:
                continue
            print(f"  {tap} [{ 'a->b' if arm=='beta' else 'b->a'}]:")
            for s in sorted(specs, key=lambda r: abs(r["c"])):
                flag = " <<beats" if s["w_res_beats_all_controls"] else ""
                print(f"     |c|={abs(s['c']):.2f}  w_res={s['flip_w_res']:.2f}  "
                      f"shuf_p95={s['shuffled_p95']}  rand={s['random']}  "
                      f"sperp={s['sperp']}  fmt={s['format']}{flag}")
    print("\nVERDICT: " + verdict)
    print(f"\nwrote: {out}/axis_causality_summary.json, axis_causality_curve.csv, w_res65_*.npy")
    print("=" * 88)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--steer_layers", type=int, nargs="*", default=None, help="post-block layers (default 18 21 24)")
    p.add_argument("--c_grid", type=float, nargs="*", default=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0])
    p.add_argument("--n_shuffle", type=int, default=10, help="shuffled-label Fisher directions (DECISIVE control)")
    p.add_argument("--n_random", type=int, default=5)
    p.add_argument("--max_targets", type=int, default=None, help="cap held-out targets per class")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--tau_flip", type=float, default=0.7)
    p.add_argument("--logit_lens_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
