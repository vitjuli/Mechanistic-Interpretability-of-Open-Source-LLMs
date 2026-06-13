"""
124_curvature_probe.py   [directional second derivative at the ignition zone]
==============================================================================
The dose-response join (123) localized the first-order law's failure to
L19-25 x usage x c>=4-8 with an UNDER-prediction (superlinear measured
response). This script turns that hypothesis into a number.

For each target prompt i, probe layer L, direction v in {usage, w_res}:
push direction is the actual steering push  v_i = s_i * v_hat  (s_i toward the
opposite class, exactly as in 122). Measure the directional second derivative
by finite difference of the FIRST directional derivative:

    kappa_i(L, v) = [ <g(h + eps*sigma*v_i), v_i> - <g(h), v_i> ] / (eps*sigma)

g(h) comes FREE from the 119 field dump; only g(h + delta) needs a GPU pass
(steered forward + backward). Two eps values check finite-difference stability.

Outputs:
  curvature_per_prompt.csv   per (prompt, layer, dir, eps): dm0, dm1, kappa
  curvature_summary.csv      per (layer, dir, class): kappa stats + the
                             second-order extrapolated dose curves
  quad_join.csv              tier-2 cells at probe layers re-predicted with the
                             quadratic term; MAE before vs after the correction
                             (does 2nd order pull the c=4-8 cells back into tol?)

Interpretation guide (printed): superlinearity TOWARD the target on both
classes cannot come from a single global quadratic (kappa is sign-invariant);
it requires kappa_i whose SIGN tracks the push side — i.e. an S-shaped margin
profile steepening away from the boundary. The summary therefore reports kappa
split by class/push sign; "ignition" = kappa consistently amplifying the push.

Budget: n_targets x len(layers) x 2 dirs x len(eps) fwd+bwd
        (188 x 7 x 2 x 2 ~ 5.3k passes ~ 15-20 min A100).

SELF-TEST (no torch / no repo):  python 124_curvature_probe.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("curv124")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def kappa_fd(dm0, dm1, step):
    """finite-difference directional second derivative from two first derivatives."""
    return (dm1 - dm0) / step


def quad_predict_flip_norm(G, m0, y, idx, v, c, sigma, kappa):
    """flip_norm prediction with the quadratic term:
    m1 = m0 + t*<g, v_i> + 0.5*t^2*kappa_i, t = c*sigma, v_i = s_i*v_hat.
    kappa: dict prompt->kappa (per this layer/dir) or scalar."""
    vu = unit_raw(np.asarray(v, float))
    fln = []
    t = c * sigma
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        k_i = kappa[i] if hasattr(kappa, "__getitem__") else float(kappa)
        dm = t * float(G[i].astype(np.float64) @ (s * vu)) + 0.5 * t * t * k_i
        m1 = m0[i] + dm
        if y[i] == 0 and m0[i] < 0:
            fln.append(int(m1 > 0))
        elif y[i] == 1 and m0[i] > 0:
            fln.append(int(m1 < 0))
    return float(np.mean(fln)) if fln else float("nan")


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 30, 300
    a = unit_raw(rng.standard_normal(d))
    y = (np.arange(n) % 2).astype(int)
    H = rng.standard_normal((n, d)) * 0.4 + np.outer(2 * y - 1.0, 0.6 * a)

    # S-shaped margin: m(h) = A*tanh(h@a / A) with A controlling saturation.
    A = 2.0
    m_fn = lambda h: A * np.tanh((h @ a) / A)
    g_fn = lambda h: (1 - np.tanh((h @ a) / A) ** 2) * a
    # analytic second derivative along +a: m'' = -2/A * tanh(z)(1-tanh^2 z), z=h@a/A
    kpp = lambda h: float(-2.0 / A * np.tanh((h @ a) / A) * (1 - np.tanh((h @ a) / A) ** 2))

    # (1) finite difference recovers the analytic curvature (small eps)
    eps = 1e-3
    errs = []
    for i in range(0, n, 17):
        v = a  # push along +a
        dm0 = float(g_fn(H[i]) @ v)
        dm1 = float(g_fn(H[i] + eps * v) @ v)
        errs.append(abs(kappa_fd(dm0, dm1, eps) - kpp(H[i])))
    assert max(errs) < 1e-3, f"finite difference must recover analytic curvature: {max(errs)}"

    # (2) sign structure of the S-shape: pushing TOWARD the boundary from either
    # side, the push-aligned curvature is POSITIVE for alpha (z<0, push +a:
    # kappa>0) and, along the beta push v=-a, kappa(v)=kappa(+a) is NEGATIVE at
    # z>0 — i.e. curvature AMPLIFIES the push toward the boundary on both sides:
    # s_i * (d/dt)<g, v_i> consistent with steepening.
    i_a = int(np.where(y == 0)[0][0]); i_b = int(np.where(y == 1)[0][0])
    assert kpp(H[i_a]) > 0 and kpp(H[i_b]) < 0, "tanh curvature sign must track the side"

    # (3) quadratic correction restores flip prediction where linear fails
    sigma = 1.0; c = 1.2
    G = np.stack([g_fn(H[i]) for i in range(n)])
    m0 = np.array([m_fn(H[i]) for i in range(n)])
    idx = list(range(n))
    kappas = {i: kpp(H[i]) for i in idx}      # push-invariant (kappa(v)=kappa(-v))
    # measured
    fln_meas = []
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        m1 = m_fn(H[i] + s * c * sigma * a)
        if y[i] == 0 and m0[i] < 0:
            fln_meas.append(int(m1 > 0))
        elif y[i] == 1 and m0[i] > 0:
            fln_meas.append(int(m1 < 0))
    meas = float(np.mean(fln_meas))
    lin = quad_predict_flip_norm(G, m0, y, idx, a, c, sigma, 0.0)
    quad = quad_predict_flip_norm(G, m0, y, idx, a, c, sigma, kappas)
    assert abs(quad - meas) < abs(lin - meas), \
        f"quadratic term must improve the prediction: lin={lin:.3f} quad={quad:.3f} meas={meas:.3f}"
    print("[self_test] OK — FD recovers analytic curvature, S-shape sign structure, "
          "quadratic correction improves flips.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    prompts = [json.loads(l) for l in open(args.corpus)]
    y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
    n_layers = int(meta["n_layers"]); d = int(meta["d"])
    nP = len(y)
    id0 = meta["id_class0"].astype(int) if "id_class0" in meta else np.full(nP, int(meta["alpha_id"]))
    id1 = meta["id_class1"].astype(int) if "id_class1" in meta else np.full(nP, int(meta["beta_id"]))
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    held = np.where(~trm)[0]
    correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
    targets = [int(i) for i in held if correct[i]][: args.max_targets or None]
    logger.info("targets: %d (baseline-correct held) | layers %s | eps %s",
                len(targets), args.layers, args.eps)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); norm_mod = _chain(model, "model.norm")
    last = n_layers - 1
    for p_ in model.parameters():
        p_.requires_grad_(True)

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    def grad_at(i, L, delta):
        """steered forward + backward; returns <g(h+delta), unit(delta)> per the SAME tap."""
        inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
        keep = {}
        def pre(m_, a_):
            hs = a_[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt
            hs.retain_grad(); keep["h"] = hs
            return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            o = model(**inp, use_cache=False)
            row = o.logits[0, -1, :]
            (row[int(id1[i])] - row[int(id0[i])]).backward()
            g = keep["h"].grad[0, -1, :].float().cpu().numpy()
        finally:
            h.remove(); model.zero_grad(set_to_none=True)
        du = unit_raw(delta)
        return float(g.astype(np.float64) @ du)

    rows = []
    for L in args.layers:
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        w = fisher_axis(H[trm], y[trm], args.shrink)
        u = unit_raw(G.mean(0))
        sigma = float(np.std(H[trm] @ w))
        for dname, v in (("usage", u), ("w_res", w)):
            vu = unit_raw(v)
            for i in targets:
                s = +1.0 if y[i] == 0 else -1.0
                vi = s * vu
                dm0 = float(G[i] @ vi)                  # free, from the dump
                for eps in args.eps:
                    step = eps * sigma
                    dm1 = grad_at(i, L, step * vi)
                    rows.append({"layer": L, "dir": dname, "eps": eps, "idx": int(i),
                                 "y": int(y[i]), "sigma": sigma,
                                 "dm0": dm0, "dm1": dm1,
                                 "kappa": kappa_fd(dm0, dm1, step)})
            done = [r for r in rows if r["layer"] == L and r["dir"] == dname and r["eps"] == args.eps[0]]
            ks = np.array([r["kappa"] for r in done])
            amp = np.array([r["kappa"] * (1 if r["y"] == 0 else -1) >= 0 for r in done])
            logger.info("  L%02d %s: median kappa=%+.4f | push-amplifying frac=%.2f (n=%d)",
                        L, dname, float(np.median(ks)), float(amp.mean()), len(done))

    with open(out / "curvature_per_prompt.csv", "w", newline="") as f:
        wf = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); wf.writeheader()
        [wf.writerow(r) for r in rows]

    # ---------- summary + quadratic re-join against tier2 ----------
    summ = []
    for L in args.layers:
        for dname in ("usage", "w_res"):
            for eps in args.eps:
                for cls in (0, 1):
                    sel = [r for r in rows if r["layer"] == L and r["dir"] == dname
                           and r["eps"] == eps and r["y"] == cls]
                    if not sel:
                        continue
                    ks = np.array([r["kappa"] for r in sel])
                    summ.append({"layer": L, "dir": dname, "eps": eps, "class": cls,
                                 "kappa_median": float(np.median(ks)),
                                 "kappa_p25": float(np.quantile(ks, .25)),
                                 "kappa_p75": float(np.quantile(ks, .75)),
                                 "n": len(sel)})
    with open(out / "curvature_summary.csv", "w", newline="") as f:
        wf = _csv.DictWriter(f, fieldnames=list(summ[0].keys())); wf.writeheader()
        [wf.writerow(r) for r in summ]

    quad_rows = []
    if args.tier2_csv and Path(args.tier2_csv).exists():
        with open(args.tier2_csv) as f:
            meas = [{"layer": int(r["layer"]), "c": float(r["c"]), "dir": r["dir"],
                     "flip_norm": float(r["flip_norm"]) if r["flip_norm"] not in ("", "nan") else float("nan")}
                    for r in _csv.DictReader(f)]
        eps0 = args.eps[0]
        for L in args.layers:
            H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
            G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
            w = fisher_axis(H[trm], y[trm], args.shrink)
            u = unit_raw(G.mean(0))
            sigma = float(np.std(H[trm] @ w))
            for dname, v in (("usage", u), ("w_res", w)):
                kmap = {r["idx"]: r["kappa"] for r in rows
                        if r["layer"] == L and r["dir"] == dname and r["eps"] == eps0}
                for r in meas:
                    if r["layer"] != L or r["dir"] != dname or np.isnan(r["flip_norm"]):
                        continue
                    lin = quad_predict_flip_norm(G, m0, y, targets, v, r["c"], sigma, 0.0)
                    qd = quad_predict_flip_norm(G, m0, y, targets, v, r["c"], sigma,
                                                {i: kmap.get(i, 0.0) for i in targets})
                    quad_rows.append({"layer": L, "c": r["c"], "dir": dname,
                                      "meas": r["flip_norm"], "pred_lin": lin, "pred_quad": qd,
                                      "ae_lin": abs(lin - r["flip_norm"]),
                                      "ae_quad": abs(qd - r["flip_norm"])})
        if quad_rows:
            with open(out / "quad_join.csv", "w", newline="") as f:
                wf = _csv.DictWriter(f, fieldnames=list(quad_rows[0].keys())); wf.writeheader()
                [wf.writerow(r) for r in quad_rows]

    print("\n" + "=" * 92)
    print("CURVATURE PROBE — ignition-zone second derivative")
    print("=" * 92)
    for L in args.layers:
        s_u = [r for r in summ if r["layer"] == L and r["dir"] == "usage" and r["eps"] == args.eps[0]]
        if s_u:
            by_cls = {r["class"]: r["kappa_median"] for r in s_u}
            print(f"  L{L:02d} usage: median kappa class0={by_cls.get(0, float('nan')):+.4f} "
                  f"class1={by_cls.get(1, float('nan')):+.4f}  "
                  f"(S-shape signature: opposite signs tracking push side)")
    if quad_rows:
        mid = [r for r in quad_rows if 4 <= r["c"] <= 16 and r["dir"] == "usage"]
        if mid:
            print(f"quadratic correction on usage c=4-16 cells: "
                  f"MAE {float(np.mean([r['ae_lin'] for r in mid])):.3f} -> "
                  f"{float(np.mean([r['ae_quad'] for r in mid])):.3f}")
    print("outputs: curvature_per_prompt.csv | curvature_summary.csv | quad_join.csv")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/B1_alpha_beta.jsonl")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--tier2_csv", default="data/analysis/runD_v2/B1_alpha_beta/steering_sweep_tier2.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/curvature")
    p.add_argument("--layers", type=int, nargs="*", default=[16, 19, 21, 22, 23, 24, 25])
    p.add_argument("--eps", type=float, nargs="*", default=[0.5, 1.0],
                   help="finite-difference steps in sigma units (small but above noise)")
    p.add_argument("--max_targets", type=int, default=0, help="0 = full baseline-correct held pool")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
