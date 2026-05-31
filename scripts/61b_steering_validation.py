"""
61b_steering_validation.py
===================================================================
VALIDATION of the steering result from 61_subspace_interventions_pilot.py.

61 found: additive steering h <- h + c*sigma*lhat along lbar = Sigma^{-1} gbar
flips 100% of alpha-target prompts at c >= +0.5 sigma, while subspace INTERCHANGE
(any S) does not flip. That dissociation is real -- BUT the steering sweep in 61
has no control direction, so a 100% flip is, on its own, logically
indistinguishable from "I added energy along the very axis that Delta logit
measures" (recall Delta logit = lambda(x)^T gbar, and lbar is gbar's Riesz dual,
so a component along lbar moving Delta logit is nearly tautological).

This script settles whether the steering effect is SPECIFIC to lbar. It runs the
four blocks that turn a promising number into a defensible one:

  (1) SPECIFICITY (the core).  Steer along lbar vs norm-matched control
      directions on the SAME c-grid:
        - random directions (N seeds),
        - S_perp (M-orthogonal complement of span{lbar}, full-Sigma metric),
        - format direction (top raw-activation PC = the 17%-variance,
          label-free direction your PCA flagged).
      If controls flip as often as lbar at the same c, the steering claim is
      void and the theorem honestly weakens to representational-only.

  (2) SYMMETRIC DIRECTIONALITY.  Steer alpha-targets toward beta (c>0) AND
      beta-targets toward alpha (c<0), with a donor-free DIRECTIONAL flip
      criterion that can detect BOTH (61's criterion only detected alpha->beta,
      which is why its c<0 column read 0). Each side must flip toward its target.

  (3) HELD-OUT + ALL PAIRS.  lbar is built from the unembedding (no activation
      labels), but we still confirm it flips prompts from surface_family values
      NOT used to estimate Sigma / the format PC. Uses every within-family
      contrastive prompt, not 10.

  (4) DOSE-RESPONSE + c_min.  Flip-rate as a smooth function of c for lbar vs
      each control (the figure: S-curve for lbar, flat ~0 for controls), plus
      c_min = smallest |c| with flip-rate >= --flip_target (default 0.9). A small
      c_min is the quantitative rebuttal to "unbounded perturbation": the
      sufficient push is small and bounded, not arbitrary.

NORMALISATION (fixed cleanly here; 61 mixed conventions).
  lhat is UNIT RAW-L2. sigma_lam is the std of the scalar projection of clean
  TRAIN residuals onto that same unit lhat:  sigma = std_p <h(p), lhat>_2.
  Then a step of "c sigma" has raw L2 norm |c| sigma, and EVERY control
  direction is unit raw-L2 too, so "norm-matched at c" means identical raw L2
  step ||Delta h|| = |c| sigma for lbar and every control. This makes the
  specificity comparison exact.

PATCH POINT identical to 61: residual-stream input to block ell
(register_forward_pre_hook on model.layers[ell]); base model Qwen/Qwen3-4B.

OUTPUTS (data/analysis/iia_failure_diagnosis/):
  steering_validation_curve.csv     long: (direction, arm, c, layer_config, flip, dl_clean, dl_steered, step_norm)
  steering_validation_summary.json  per-(arm,layer) dose-response, c_min, lbar-vs-control gaps, verdict
  steering_validation_cmin.csv      c_min per arm per layer_config

SELF-TEST (no torch / no repo): python 61b_steering_validation.py --self_test
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
logger = logging.getLogger("steering_validation")


# =====================================================================
# Geometry: reuse 61's validated helpers if present, else define locally
# =====================================================================
try:
    # When run from the repo, 61 sits beside this file.
    import importlib.util as _ilu
    _p = Path(__file__).resolve().parent / "61_subspace_interventions_pilot.py"
    if _p.exists():
        _spec = _ilu.spec_from_file_location("_p61", str(_p))
        _p61 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_p61)
        whitener_from_cov = _p61.whitener_from_cov
        bootstrap_ci = _p61.bootstrap_ci
        _REUSED = True
    else:
        raise ImportError
except Exception:  # pragma: no cover - fallback keeps 61b standalone for --self_test
    _REUSED = False

    def whitener_from_cov(Sigma: np.ndarray, ridge: float = 1e-3):
        d = Sigma.shape[0]
        Sigma = 0.5 * (Sigma + Sigma.T)
        ridge_abs = ridge * float(np.mean(np.diag(Sigma)))
        evals, evecs = np.linalg.eigh(Sigma + ridge_abs * np.eye(d))
        evals = np.clip(evals, 1e-30, None)
        cond = float(evals.max() / evals.min())
        A = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
        Sigma_inv = (evecs * (1.0 / evals)) @ evecs.T
        return A, Sigma_inv, cond

    def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
        return float(x.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# =====================================================================
# Steering-specific geometry (clean unit-raw-L2 convention)
# =====================================================================

def unit_raw(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def sperp_unit_direction(lhat_raw: np.ndarray, A_full: np.ndarray, Ainv_full: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    A unit-raw-L2 direction lying in the M-orthogonal complement of span{lbar},
    M = Sigma^{-1} (full). Built in whitened coords (where M-orthogonality is
    Euclidean orthogonality), then mapped back and renormalised to unit raw L2.
    """
    qw = unit_raw(A_full @ lhat_raw)            # whitened lbar direction (unit in whitened)
    v = rng.standard_normal(A_full.shape[0])
    vw = v - qw * (qw @ v)                       # remove whitened-lbar component
    nv = np.linalg.norm(vw)
    if nv < 1e-12:
        vw = rng.standard_normal(A_full.shape[0]); vw = vw - qw * (qw @ vw); nv = np.linalg.norm(vw)
    vw = vw / nv
    return unit_raw(Ainv_full @ vw)              # back to raw, unit raw-L2


def directional_flip(dl_clean: float, dl_steered: float, toward: str) -> int:
    """
    Donor-free directional flip: the steer moved sign(Delta logit) = sign(logit_beta
    - logit_alpha) from its clean value to the side of `toward`.
      toward='beta': flip iff dl_clean < 0 (was alpha) and dl_steered > 0 (now beta).
      toward='alpha': flip iff dl_clean > 0 (was beta) and dl_steered < 0 (now alpha).
    Counts only genuine sign CHANGES toward the intended target (so it cannot be
    inflated by prompts already on the target side -- those are excluded from the
    denominator by the caller).
    """
    if toward == "beta":
        return int(dl_clean < 0 and dl_steered > 0)
    return int(dl_clean > 0 and dl_steered < 0)


def cmin_from_curve(cs: np.ndarray, rates: np.ndarray, target: float) -> Optional[float]:
    """Smallest |c| (among swept, positive magnitudes) whose flip-rate >= target."""
    mags = np.abs(cs)
    ok = rates >= target
    if not ok.any():
        return None
    return float(np.min(mags[ok]))


# =====================================================================
# Self-test: a synthetic model where lbar is causal and controls are not
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(23)
    d = 48
    e_c = np.zeros(d); e_c[1] = 1.0           # concept axis
    e_f = np.zeros(d); e_f[0] = 1.0           # format axis (huge variance, label-free)
    n = 800
    labels = rng.integers(0, 2, size=n)
    H = rng.standard_normal((n, d)) * 0.3
    H[:, 0] += rng.standard_normal(n) * 3.0   # format dominates variance
    H[:, 1] += (labels * 2.0 - 1.0) * 1.0     # concept tracks label
    Sigma = np.cov(H, rowvar=False)
    A_full, Sigma_inv, cond = whitener_from_cov(Sigma)
    Ainv_full = np.linalg.inv(A_full)
    gbar = 2.0 * e_c
    lbar = Sigma_inv @ gbar
    lhat = unit_raw(lbar)

    # the model reads the concept along gbar (logits = unembedding . residual)
    ghat = unit_raw(gbar)
    def dl(h): return float(h @ ghat)

    # format PC = top raw-activation PC (should be ~e_f)
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    format_dir = unit_raw(Vt[0])

    # sigma along lhat on clean activations
    sigma_lam = float(np.std(H @ lhat))

    # alpha-target prompt (concept negative), with lots of format.
    # Keep the clean margin modest relative to sigma_lam so c=1 sigma is enough
    # (the realistic regime the assertion encodes: a small bounded push suffices).
    h_alpha = -0.6 * e_c + 4.0 * e_f + 0.2 * rng.standard_normal(d)
    cs = np.array([0.25, 0.5, 1.0, 2.0])
    arms = {"lbar": lhat, "format": format_dir,
            "sperp": sperp_unit_direction(lhat, A_full, Ainv_full, rng)}
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"d={d} cond(Sigma)={cond:.2e} sigma_lam={sigma_lam:.3f}")
    print(f"clean dl(h_alpha)={dl(h_alpha):+.3f} (negative => alpha, correct)")
    rates = {}
    for arm, vec in arms.items():
        rr = []
        for c in cs:
            step = (c * sigma_lam) * vec
            rr.append(directional_flip(dl(h_alpha), dl(h_alpha + step), "beta"))
        rates[arm] = np.array(rr)
        print(f"  steer {arm:7s}: flips by c={list(cs)} -> {rates[arm].tolist()}  "
              f"(step raw-L2 at c=1: {sigma_lam:.3f})")

    # lbar must flip by c=1; format & sperp must NOT flip at the same c (specificity)
    assert rates["lbar"][cs == 1.0][0] == 1, "lbar steering must flip the concept by c=1 sigma"
    assert rates["format"][cs == 1.0][0] == 0, "format steering must NOT flip (specificity)"
    assert rates["sperp"][cs == 1.0][0] == 0, "S_perp steering must NOT flip (specificity)"
    # norm-match: every arm uses the SAME step norm at a given c
    step_norms = {arm: float(np.linalg.norm((1.0 * sigma_lam) * vec)) for arm, vec in arms.items()}
    assert max(step_norms.values()) - min(step_norms.values()) < 1e-9, "arms must be norm-matched"
    cmin = cmin_from_curve(cs, rates["lbar"], 0.9)
    print(f"norm-match across arms at c=1: max-min = {max(step_norms.values())-min(step_norms.values()):.2e}")
    print(f"c_min(lbar, target=0.9) = {cmin}")
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("---------------------------------------------------------------\n")


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
    device = args.device
    rng = np.random.default_rng(args.seed)

    # ---- concept directions from 60_ ----
    cd = np.load(args.concept_npz)
    gbar = cd["gbar"].astype(np.float64)
    if "Sigma" not in cd:
        raise SystemExit("concept_npz must contain 'Sigma' (re-run 60_ to dump it)")
    A_full, Sigma_inv, _ = whitener_from_cov(cd["Sigma"].astype(np.float64), ridge=args.ridge)
    Ainv_full = np.linalg.inv(A_full)
    lbar = cd["lbar"].astype(np.float64) if "lbar" in cd else (Sigma_inv @ gbar)
    lhat = unit_raw(lbar)                              # UNIT RAW-L2 (clean convention)
    d = gbar.shape[0]

    # ---- model + tokenizer (base model!) ----
    logger.info("Loading %s (base)", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    blocks = _getattr_chain(model, "model.layers")

    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)
    alpha_id, beta_id = a_id[0], b_id[0]

    # ---- prompts + split by surface_family (same convention as 61) ----
    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts})
    rng.shuffle(fams)
    n_train = int(round(len(fams) * args.train_frac))
    train_fams = set(fams[:n_train]); held_fams = set(fams[n_train:])
    train_prompts = [p for p in prompts if p["surface_family"] in train_fams]
    held_prompts = [p for p in prompts if p["surface_family"] in held_fams]
    logger.info("families: %d train / %d held-out", len(train_fams), len(held_fams))

    layer_set = sorted(set(sum([cfg for cfg in args.layer_configs], [])))

    # ---- hooks: capture residual input; steered forward (add delta at last token) ----
    def capture_resid(ptext: str, layers: List[int]) -> Dict[int, np.ndarray]:
        inputs = tok([ptext], return_tensors="pt").to(device)
        grabbed: Dict[int, np.ndarray] = {}
        handles = []
        for L in layers:
            def _make(L=L):
                def _pre(module, args_in):
                    grabbed[L] = args_in[0][0, -1, :].detach().float().cpu().numpy()
                    return None
                return _pre
            handles.append(blocks[L].register_forward_pre_hook(_make(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    def clean_dl(ptext: str) -> float:
        inputs = tok([ptext], return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inputs, use_cache=False)
        lp = torch.log_softmax(o.logits[0, -1, :].float(), dim=0)
        return float(lp[beta_id] - lp[alpha_id])

    def steered_dl(ptext: str, delta_by_layer: Dict[int, np.ndarray]) -> float:
        inputs = tok([ptext], return_tensors="pt").to(device)
        handles = []
        for L, delta in delta_by_layer.items():
            dt = torch.tensor(delta, dtype=torch.float32, device=device)
            def _make(dt=dt):
                def _pre(module, args_in):
                    hs = args_in[0].clone()
                    hs[0, -1, :] = hs[0, -1, :] + dt
                    return (hs,)
                return _pre
            handles.append(blocks[L].register_forward_pre_hook(_make(), with_kwargs=False))
        try:
            with torch.no_grad():
                o = model(**inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(o.logits[0, -1, :].float(), dim=0)
        return float(lp[beta_id] - lp[alpha_id])

    # ---- sigma along lhat (per layer) from clean TRAIN residuals ----
    logger.info("Capturing TRAIN residuals for sigma_lam and the format PC (layers %s)", layer_set)
    train_resid = {L: [] for L in layer_set}
    for p in train_prompts:
        g = capture_resid(p["prompt"], layer_set)
        for L in layer_set:
            train_resid[L].append(g[L])
    sigma_by_L: Dict[int, float] = {}
    format_dir_by_L: Dict[int, np.ndarray] = {}
    for L in layer_set:
        H = np.array(train_resid[L])                  # (n, d)
        sigma_by_L[L] = float(np.std(H @ lhat))       # std of scalar projection onto unit lhat
        Hc = H - H.mean(0)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        format_dir_by_L[L] = unit_raw(Vt[0])          # top raw-activation PC (the format direction)
        logger.info("  L%d: sigma_lam=%.4f  format-PC cos with lhat=%.3f",
                    L, sigma_by_L[L], abs(float(format_dir_by_L[L] @ lhat)))

    # ---- control direction bank (all UNIT RAW-L2) ----
    def control_dirs_for_layer(L: int) -> Dict[str, np.ndarray]:
        dirs = {"lbar": lhat}
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        dirs["sperp"] = sperp_unit_direction(lhat, A_full, Ainv_full, rng)
        dirs["format"] = format_dir_by_L[L]
        return dirs

    # ---- targets: held-out prompts, split by their correct answer ----
    held_alpha = [p for p in held_prompts if p["correct_answer"].strip() == "alpha"]
    held_beta = [p for p in held_prompts if p["correct_answer"].strip() == "beta"]
    if args.max_targets:
        held_alpha = held_alpha[: args.max_targets]
        held_beta = held_beta[: args.max_targets]
    logger.info("held-out targets: %d alpha (steer->beta, c>0), %d beta (steer->alpha, c<0)",
                len(held_alpha), len(held_beta))

    # ---- sweep ----
    # arm 'beta'  : steer alpha-correct targets toward beta with c in +grid
    # arm 'alpha' : steer beta-correct targets toward alpha with c in -grid
    pos_grid = sorted([c for c in args.c_grid if c > 0])
    neg_grid = sorted([-c for c in pos_grid])         # mirror for the alpha arm
    rows = []

    def run_arm(targets, toward, c_values):
        for tgt in targets:
            dlc = clean_dl(tgt["prompt"])
            # exclude prompts already on the target side (nothing to flip) from denominator
            if toward == "beta" and not (dlc < 0):
                continue
            if toward == "alpha" and not (dlc > 0):
                continue
            ha = capture_resid(tgt["prompt"], layer_set)  # not used for steering, but keeps parity
            for cfg in args.layer_configs:
                cfg_name = "+".join(f"L{L}" for L in cfg)
                ctrl = {L: control_dirs_for_layer(L) for L in cfg}
                arm_dirs = list(ctrl[cfg[0]].keys())
                for c in c_values:
                    for dname in arm_dirs:
                        delta_by_L = {}
                        step_norm = 0.0
                        for L in cfg:
                            step = (c * sigma_by_L[L]) * ctrl[L][dname]
                            delta_by_L[L] = step
                            step_norm += float(np.linalg.norm(step))
                        dls = steered_dl(tgt["prompt"], delta_by_L)
                        rows.append({
                            "arm": toward, "toward": toward, "direction_kind": dname,
                            "c": c, "layer_config": cfg_name,
                            "fam": tgt["surface_family"], "tgt_id": tgt.get("prompt_id"),
                            "dl_clean": dlc, "dl_steered": dls,
                            "flip": directional_flip(dlc, dls, toward),
                            "step_norm": step_norm,
                        })

    run_arm(held_alpha, "beta", pos_grid)
    run_arm(held_beta, "alpha", neg_grid)

    df = pd.DataFrame(rows)
    df.to_csv(out / "steering_validation_curve.csv", index=False)

    # ---- collapse random* into one 'random' kind (mean over seeds per row-key) ----
    def kind_group(k): return "random" if str(k).startswith("random") else k
    df["dir_group"] = df["direction_kind"].map(kind_group)

    # ---- dose-response per (arm, layer_config, dir_group, c) ----
    summary = {
        "n_alpha_targets": len(held_alpha), "n_beta_targets": len(held_beta),
        "c_grid_pos": pos_grid, "flip_target": args.flip_target,
        "reused_61_helpers": bool(_REUSED),
        "sigma_lam_by_layer": {str(k): round(v, 4) for k, v in sigma_by_L.items()},
        "format_cos_lhat_by_layer": {str(L): round(abs(float(format_dir_by_L[L] @ lhat)), 4) for L in layer_set},
        "dose_response": [], "cmin": [], "specificity_gap": [], "verdict": None,
    }
    cmin_rows = []
    for (arm, cfg, grp), g in df.groupby(["arm", "layer_config", "dir_group"]):
        gg = g.groupby("c")["flip"].mean().sort_index()
        cs = np.array([abs(c) for c in gg.index.values])
        rates = gg.values.astype(float)
        rate_mean, lo, hi = bootstrap_ci(g["flip"].values, seed=args.seed)
        summary["dose_response"].append({
            "arm": arm, "layer_config": cfg, "direction": grp,
            "flip_by_c": {str(c): round(float(r), 3) for c, r in zip(gg.index.values, rates)},
            "flip_overall": round(rate_mean, 3), "flip_CI": [round(lo, 3), round(hi, 3)],
        })
        cm = cmin_from_curve(cs, rates, args.flip_target)
        cmin_rows.append({"arm": arm, "layer_config": cfg, "direction": grp, "c_min": cm})
        summary["cmin"].append({"arm": arm, "layer_config": cfg, "direction": grp, "c_min": cm})

    pd.DataFrame(cmin_rows).to_csv(out / "steering_validation_cmin.csv", index=False)

    # ---- specificity gap: lbar flip-rate minus best control, per (arm, layer, c) ----
    for (arm, cfg, c), g in df.groupby(["arm", "layer_config", "c"]):
        by = g.groupby("dir_group")["flip"].mean()
        lb = float(by.get("lbar", float("nan")))
        ctrl_best = float(max([v for k, v in by.items() if k != "lbar"], default=float("nan")))
        summary["specificity_gap"].append({
            "arm": arm, "layer_config": cfg, "c": c,
            "lbar": round(lb, 3), "best_control": round(ctrl_best, 3),
            "gap": round(lb - ctrl_best, 3) if not (np.isnan(lb) or np.isnan(ctrl_best)) else None,
        })

    # ---- verdict: lbar must (a) reach flip_target on both arms and (b) beat every
    #      control by a clear margin at the c where lbar first hits the target. ----
    def arm_curve(arm, grp):
        recs = [r for r in summary["dose_response"] if r["arm"] == arm and r["direction"] == grp]
        return recs

    def lbar_hits(arm) -> bool:
        return any(r["flip_overall"] >= args.flip_target or
                   max(map(float, r["flip_by_c"].values()), default=0) >= args.flip_target
                   for r in arm_curve(arm, "lbar"))

    # max control flip-rate anywhere (per arm) at matched c
    def max_control_gap(arm) -> float:
        gaps = [s["gap"] for s in summary["specificity_gap"]
                if s["arm"] == arm and s["gap"] is not None]
        return float(max(gaps)) if gaps else float("nan")

    beta_ok = lbar_hits("beta")
    alpha_ok = lbar_hits("alpha")
    # control suppression: at the strongest lbar setting, best control stays well below
    worst_control = max(
        [s["best_control"] for s in summary["specificity_gap"]
         if (s["arm"] == "beta" and s["lbar"] >= args.flip_target)
         or (s["arm"] == "alpha" and s["lbar"] >= args.flip_target)],
        default=0.0,
    )
    specific = (worst_control <= args.tau_ctrl)

    if beta_ok and alpha_ok and specific:
        verdict = ("STEERING VALIDATED: lbar flips BOTH directions to target while every "
                   "norm-matched control (random / S_perp / format) stays <= tau_ctrl. "
                   "Concept direction is SPECIFIC -> Theorem 1(2) holds in steering form; "
                   "report c_min as the (small, bounded) sufficient magnitude.")
    elif (beta_ok or alpha_ok) and specific:
        verdict = ("STEERING PARTIALLY VALIDATED: specific to lbar but asymmetric across "
                   "directions -> state the working direction precisely; investigate the "
                   "weaker arm (decision-margin asymmetry) before strengthening Theorem 1(2).")
    elif (beta_ok or alpha_ok) and not specific:
        verdict = ("NOT SPECIFIC: a norm-matched control flips comparably to lbar -> the "
                   "steering effect is (at least partly) generic energy along the readout "
                   "axis. Do NOT claim interventional sufficiency; weaken to "
                   "representational-only (Cond I + necessity), drop/qualify Thm 1(2).")
    else:
        verdict = ("STEERING NOT SUPPORTED at flip_target on held-out: weaken to "
                   "representational-only (Cond I + necessity); drop/qualify Thm 1(2).")
    summary["verdict"] = verdict

    with open(out / "steering_validation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # ---- console ----
    print("\n" + "=" * 82)
    print("STEERING VALIDATION  --  specificity, symmetry, dose-response")
    print("=" * 82)
    for arm in ("beta", "alpha"):
        arm_label = "alpha->beta (c>0)" if arm == "beta" else "beta->alpha (c<0)"
        print(f"\n[{arm_label}]  dose-response  flip-rate by |c| (rows = direction):")
        recs = [r for r in summary["dose_response"] if r["arm"] == arm]
        cfgs = sorted({r["layer_config"] for r in recs})
        for cfg in cfgs:
            print(f"  layer {cfg}:")
            for grp in ["lbar", "random", "sperp", "format"]:
                rr = [r for r in recs if r["layer_config"] == cfg and r["direction"] == grp]
                if not rr:
                    continue
                fb = rr[0]["flip_by_c"]
                order = sorted(fb.items(), key=lambda kv: abs(float(kv[0])))
                cells = "  ".join(f"|c|={abs(float(k)):.2f}:{v:.2f}" for k, v in order)
                star = " *" if grp == "lbar" else "  "
                print(f"   {star}{grp:8s} {cells}")
    print("\nc_min (smallest |c| reaching flip_target):")
    for rec in summary["cmin"]:
        if rec["direction"] == "lbar":
            print(f"  {rec['arm']:5s} {rec['layer_config']:10s} c_min={rec['c_min']}")
    print("\nVERDICT: " + summary["verdict"])
    print(f"\nwrote: {out}/steering_validation_curve.csv, steering_validation_summary.json, "
          f"steering_validation_cmin.csv")
    print("=" * 82)


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")

    p.add_argument("--concept_npz", type=str, help="concept_directions.npz from 60_ (needs gbar, Sigma)")
    p.add_argument("--prompts", type=str, default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", type=str, default="data/analysis/iia_failure_diagnosis")

    p.add_argument("--model_size", type=str, default="4b")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alpha_answer", type=str, default=" alpha")
    p.add_argument("--beta_answer", type=str, default=" beta")

    p.add_argument("--layer_configs",
                   type=lambda s: [[int(x) for x in grp.split(",")] for grp in s.split(";")],
                   default=[[24], [18, 24], [22]],
                   help="semicolon-separated; comma within a config. Default '24;18,24;22'")
    p.add_argument("--c_grid", type=float, nargs="*",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
                   help="POSITIVE magnitudes; mirrored to negative for the alpha arm")
    p.add_argument("--n_random", type=int, default=5, help="random control directions (seeds)")
    p.add_argument("--max_targets", type=int, default=None,
                   help="cap held-out targets per class (None = all)")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--flip_target", type=float, default=0.9, help="flip-rate defining c_min / success")
    p.add_argument("--tau_ctrl", type=float, default=0.3, help="max allowed control flip-rate for specificity")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    if not args.concept_npz:
        raise SystemExit("provide --concept_npz (from 60_) or use --self_test. See --help.")
    run_real(args)


if __name__ == "__main__":
    main()
