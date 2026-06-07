"""
89_intervention_calculus.py   [predictable interventions + the minimal flip and its geometry]
==============================================================================================
Two computations that turn the usage gradient into quantitative laws:

(A) INTERVENTION CALCULUS (validation-grade, AtP*-adjacent). Local linearity predicts the
    response to ANY push: Delta_margin ~ g_i . delta. We fire a battery of directions
    (w_res, u_bar, random) at a grid of norm-matched amplitudes and plot predicted vs
    measured margin change: R^2 and slope per (layer, amplitude). If the law holds up to
    some amplitude, steering outcomes are predictable from ONE backward pass without
    running the intervention; the breakdown amplitude is the validity radius. (Note: the
    log-softmax margin lp[beta]-lp[alpha] equals the raw logit difference exactly, so
    predicted and measured live on the same scale.)

(B) MINIMAL FLIPPING PERTURBATION (headline). Per layer, per held-out target, find the
    smallest latent shift delta* that flips the margin sign (Newton iterations on the
    locally-linear model, each step re-measured with a real forward+backward). Then
    decompose it:
      s_concept = ||P_Wc delta*||^2 / ||delta*||^2   (share inside the readable concept map)
      s_usage   = ||P_Uu delta*||^2 / ||delta*||^2   (share inside the usage subspace)
    s_concept ~ 0 everywhere = the decision-boundary normal AVOIDS the readable concept
    subspace -- the sharpest geometric statement of the bypass. The set {delta*_i} also
    gives the LEVER-CONE dimension (participation ratio of its Gram): an empirical measure
    of the non-identifiability class of working interventions.

SELF-TEST (no torch / no repo):  python 89_intervention_calculus.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("calculus")


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


def inlp_subspace(H, y, k, shrink=0.1):
    Hc = H.copy().astype(np.float64); ws = []
    for _ in range(k):
        w = fisher_axis(Hc, y, shrink)
        for prev in ws:
            w = w - (w @ prev) * prev
        nrm = np.linalg.norm(w)
        if nrm < 1e-8:
            break
        w = w / nrm; ws.append(w); Hc = Hc - np.outer(Hc @ w, w)
    Q, _ = np.linalg.qr(np.stack(ws, axis=1))
    return Q[:, : len(ws)]


def gram_top(G, k):
    _, s, Vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    return Vt[:k].T, s


def participation_ratio(sing_vals):
    lam = np.asarray(sing_vals, float) ** 2
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-30))


def r_squared(pred, meas):
    pred = np.asarray(pred, float); meas = np.asarray(meas, float)
    ss_res = float(((meas - pred) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum()) + 1e-30
    return 1.0 - ss_res / ss_tot


def fit_slope(pred, meas):
    pred = np.asarray(pred, float); meas = np.asarray(meas, float)
    return float((pred @ meas) / ((pred @ pred) + 1e-30))


def newton_flip_step(delta, g, m_cur, m_target):
    """One Newton step for the locally-linear margin: choose delta so that m -> m_target."""
    g = np.asarray(g, float)
    return delta + (m_target - m_cur) * g / (float(g @ g) + 1e-30)


def subspace_share(delta, W):
    delta = np.asarray(delta, float)
    if W is None or W.size == 0:
        return float("nan")
    p = W.T @ delta
    return float((p @ p) / (float(delta @ delta) + 1e-30))


def random_orthonormal(d, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q[:, :k]


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d = 25
    a = unit_raw(rng.standard_normal(d)) * 2.0          # linear toy margin m(h) = h @ a
    h0 = rng.standard_normal(d)
    m0 = float(h0 @ a)
    # (A) calculus exact in the linear toy
    preds, meas = [], []
    for _ in range(30):
        delta = 0.3 * rng.standard_normal(d)
        preds.append(float(a @ delta)); meas.append(float((h0 + delta) @ a - m0))
    assert r_squared(preds, meas) > 0.999 and abs(fit_slope(preds, meas) - 1) < 1e-6
    # (B) minimal flip: one Newton step lands exactly on the target margin
    tau = 0.4; m_target = -np.sign(m0) * tau
    delta = newton_flip_step(np.zeros(d), a, m0, m_target)
    m1 = float((h0 + delta) @ a)
    assert abs(m1 - m_target) < 1e-9 and np.sign(m1) != np.sign(m0)
    assert abs(np.linalg.norm(delta) - abs(m_target - m0) / np.linalg.norm(a)) < 1e-9, "minimal-norm property"
    # shares: subspace containing a -> 1, orthogonal subspace -> 0
    Wa = unit_raw(a)[:, None]
    Wo = random_orthonormal(d, 3, rng); Wo = Wo - Wa @ (Wa.T @ Wo); Wo, _ = np.linalg.qr(Wo); Wo = Wo[:, :3]
    assert subspace_share(delta, Wa) > 0.999 and subspace_share(delta, Wo) < 1e-6
    # lever-cone dimension: identical levers -> PR 1; orthogonal levers -> PR k
    D1 = np.tile(unit_raw(rng.standard_normal(d)), (10, 1))
    _, s1 = gram_top(D1, 5); assert abs(participation_ratio(s1) - 1.0) < 1e-6
    Dk = random_orthonormal(d, 5, rng).T
    _, sk = gram_top(Dk, 5); assert abs(participation_ratio(sk) - 5.0) < 1e-6
    print("[self_test] OK — exact calculus on linear toy, Newton minimal flip, subspace shares, lever-cone PR pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size; last = n_layers - 1
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in (args.layers or [8, 16, 21, 24, 30, 35]) if 0 <= L < n_layers})
    logger.info("model: %d layers; calculus + min-flip over taps %s", n_layers, layers)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    # ---------- capture: residual + gradient at answer position on chosen taps ----------
    nP = len(prompts)
    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    grad = {L: np.zeros((nP, d), np.float32) for L in layers}
    y = np.zeros(nP, int); trm = np.zeros(nP, bool); clean_margin = np.zeros(nP)
    logger.info("capturing residuals + gradients for %d prompts at %d taps...", nP, len(layers))
    for p_ in model.parameters():
        p_.requires_grad_(True)
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        keep = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            row = model(**inp, use_cache=False).logits[0, -1, :]
            (row[beta_id] - row[alpha_id]).backward()
            for L in layers:
                t = keep[L]
                res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                grad[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
            lp = torch.log_softmax(row.detach().float(), 0)
            clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        model.zero_grad(set_to_none=True)
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        trm[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, nP)

    held = [i for i in range(nP) if not trm[i]]

    def steer_eval(ptext, L, delta, need_grad=False):
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
        inp = tok([ptext], return_tensors="pt").to(args.device)
        keep = {}
        def pre(m, a):
            if need_grad and a[0].requires_grad:
                a[0].retain_grad(); keep["t"] = a[0]
            hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt
            return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            if need_grad:
                row = model(**inp, use_cache=False).logits[0, -1, :]
                (row[beta_id] - row[alpha_id]).backward()
                g = keep["t"].grad[0, -1, :].float().cpu().numpy() if "t" in keep else None
                m = float(row[beta_id].item() - row[alpha_id].item())
                model.zero_grad(set_to_none=True)
                return m, g
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            return float(row[beta_id] - row[alpha_id]), None
        finally:
            h.remove()

    # ---------- pipeline sanity (CHEAP: 1 prompt × both paths) ----------
    # Catches retain_grad / hook / shape bugs BEFORE the long capture phase.
    logger.info("(sanity) testing steer_eval on 1 prompt × no_grad / with_grad paths...")
    sanity_L = layers[0]
    sanity_p = prompts[0]["prompt"]
    sanity_d = np.zeros(d, np.float32)
    try:
        m_ng, _ = steer_eval(sanity_p, sanity_L, sanity_d, need_grad=False)
        m_g, g_g = steer_eval(sanity_p, sanity_L, sanity_d, need_grad=True)
        assert g_g is not None and g_g.shape == (d,), f"grad shape {None if g_g is None else g_g.shape}"
        assert abs(m_ng - m_g) < 1e-2, f"margin mismatch no_grad={m_ng} with_grad={m_g}"
        logger.info("(sanity) OK — both paths produce consistent margins (%.4f vs %.4f), grad shape %s",
                    m_ng, m_g, g_g.shape)
    except Exception as e:
        logger.error("(sanity) FAILED before main loop: %s", e)
        raise

    # ---------- (A) calculus: predicted vs measured ----------
    ca = [i for i in held if y[i] == 0][: args.calc_targets]
    cb = [i for i in held if y[i] == 1][: args.calc_targets]
    calc_targets = ca + cb
    pts = []
    logger.info("(A) calculus battery: %d taps x dirs x c=%s x %d targets ...", len(layers), args.c_grid, len(calc_targets))
    for L in layers:
        H = res[L].astype(np.float64)
        w_res = fisher_axis(H[trm], y[trm], args.shrink)
        u_bar = unit_raw(grad[L].astype(np.float64).mean(0))
        sig = float(np.std(H[trm] @ w_res))
        dirs = {"w_res": w_res, "usage": u_bar}
        for r in range(args.n_random):
            dirs[f"random{r}"] = unit_raw(rng.standard_normal(d))
        for c in args.c_grid:
            for name, v in dirs.items():
                for i in calc_targets:
                    s = +1.0 if y[i] == 0 else -1.0          # push toward the opposite class
                    delta = (s * c * sig) * unit_raw(v)
                    pred = float(grad[L][i].astype(np.float64) @ delta)
                    m1, _ = steer_eval(prompts[i]["prompt"], L, delta)
                    pts.append({"layer": int(L), "dir": name, "c": float(c),
                                "pred": pred, "meas": m1 - clean_margin[i]})
        for c in args.c_grid:
            sel = [q for q in pts if q["layer"] == L and q["c"] == c]
            r2 = r_squared([q["pred"] for q in sel], [q["meas"] for q in sel])
            sl = fit_slope([q["pred"] for q in sel], [q["meas"] for q in sel])
            logger.info("  L%d c=%g: R2=%.3f slope=%.2f (n=%d)", L, c, r2, sl, len(sel))

    # ---------- (B) minimal flip ----------
    fa = [i for i in held if y[i] == 0 and clean_margin[i] < 0][: args.flip_targets]
    fb = [i for i in held if y[i] == 1 and clean_margin[i] > 0][: args.flip_targets]
    flip_targets = fa + fb
    logger.info("(B) minimal flip on %d baseline-correct targets x %d taps ...", len(flip_targets), len(layers))
    flip_rows = []; per_layer = []
    for L in layers:
        H = res[L].astype(np.float64)
        Wc = inlp_subspace(H[trm], y[trm], args.k_concept, args.shrink)
        Uu, _ = gram_top(grad[L], args.k_usage)
        dmu = np.linalg.norm(H[trm][y[trm] == 1].mean(0) - H[trm][y[trm] == 0].mean(0))
        deltas = []
        for i in flip_targets:
            m0 = clean_margin[i]; m_target = -np.sign(m0) * args.tau
            delta = np.zeros(d); m_cur = m0; g = grad[L][i].astype(np.float64)
            flipped = False
            for it in range(args.flip_iters):
                delta = newton_flip_step(delta, g, m_cur, m_target)
                need_g = (it < args.flip_iters - 1)
                m_cur, gnew = steer_eval(prompts[i]["prompt"], L, delta, need_grad=need_g)
                if gnew is not None:
                    g = gnew.astype(np.float64)
                if np.sign(m_cur) != np.sign(m0) and abs(m_cur) >= 0.25 * args.tau:
                    flipped = True
                    break
            nrm = float(np.linalg.norm(delta))
            row = {"layer": int(L), "idx": int(i), "flipped": int(flipped), "iters": it + 1,
                   "norm": nrm, "norm_over_dmu": nrm / (dmu + 1e-12),
                   "s_concept": subspace_share(delta, Wc), "s_usage": subspace_share(delta, Uu),
                   "cos_delta_g0": float(unit_raw(delta) @ unit_raw(grad[L][i].astype(np.float64)))}
            flip_rows.append(row)
            if flipped:
                deltas.append(unit_raw(delta))
        ok = [r for r in flip_rows if r["layer"] == L and r["flipped"]]
        if ok:
            _, sd = gram_top(np.stack(deltas), min(len(deltas), 20))
            per_layer.append({"layer": int(L), "flip_rate": len(ok) / len(flip_targets),
                              "med_norm_over_dmu": float(np.median([r["norm_over_dmu"] for r in ok])),
                              "med_s_concept": float(np.median([r["s_concept"] for r in ok])),
                              "med_s_usage": float(np.median([r["s_usage"] for r in ok])),
                              "lever_cone_pr": participation_ratio(sd)})
        else:
            per_layer.append({"layer": int(L), "flip_rate": 0.0, "med_norm_over_dmu": float("nan"),
                              "med_s_concept": float("nan"), "med_s_usage": float("nan"),
                              "lever_cone_pr": float("nan")})
        pl = per_layer[-1]
        logger.info("  L%d: flip-rate=%.2f | ||delta||/||dmu||=%.2f | s_concept=%.3f s_usage=%.3f | lever-cone PR=%.1f",
                    L, pl["flip_rate"], pl["med_norm_over_dmu"], pl["med_s_concept"], pl["med_s_usage"], pl["lever_cone_pr"])

    # ---------- save + verdict ----------
    import csv as _csv
    def wcsv(name, rws):
        if not rws:
            return
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rws[0].keys())); w.writeheader(); [w.writerow(r) for r in rws]
    wcsv("calculus_points.csv", pts); wcsv("minflip_per_target.csv", flip_rows); wcsv("minflip_per_layer.csv", per_layer)

    lowc = min(args.c_grid)
    r2_low = r_squared([q["pred"] for q in pts if q["c"] == lowc], [q["meas"] for q in pts if q["c"] == lowc])
    sc = [p_["med_s_concept"] for p_ in per_layer if not np.isnan(p_["med_s_concept"])]
    su = [p_["med_s_usage"] for p_ in per_layer if not np.isnan(p_["med_s_usage"])]
    print("\n" + "=" * 96)
    print("INTERVENTION CALCULUS + MINIMAL FLIP")
    print("=" * 96)
    print(f"(A) at c={lowc}: global R2(pred, meas) = {r2_low:.3f} "
          f"({'one backward pass PREDICTS steering outcomes' if r2_low > 0.8 else 'local linearity weak at this scale'})")
    if sc:
        print(f"(B) decision-boundary normal: median s_concept = {np.median(sc):.3f}, median s_usage = {np.median(su):.3f} "
              f"-> {'the normal AVOIDS the readable concept map' if np.median(sc) < 0.1 else 'the normal partially lies in the concept map'}; "
              f"lever-cone PR per layer in CSV (empirical non-identifiability dimension).")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/intervention_calculus")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = 8 16 21 24 30 35")
    p.add_argument("--c_grid", type=float, nargs="*", default=[0.5, 1, 4, 16])
    p.add_argument("--n_random", type=int, default=2)
    p.add_argument("--calc_targets", type=int, default=30, help="per class for part A")
    p.add_argument("--flip_targets", type=int, default=20, help="per class for part B")
    p.add_argument("--flip_iters", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.5, help="target |margin| beyond zero after the flip")
    p.add_argument("--k_concept", type=int, default=13)
    p.add_argument("--k_usage", type=int, default=13)
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
