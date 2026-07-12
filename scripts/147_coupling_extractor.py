"""
147_coupling_extractor.py   [per-layer read/write/use coupling curve, raw vs scaffold]
=======================================================================================
Extracts the ONE thing the dynamic-coupling test needs and nothing else: per layer, in each
regime (raw, scaffold), the geometric relationship among the three concept axes and the output.

Axes per layer L (from the field dump's res_L / grad_L):
  w_res (reading) = Fisher/LDA direction  unit( Sigma_w_shrunk^{-1} (mu_beta - mu_alpha) )
  delta (writing) = class-mean difference unit( mu_beta - mu_alpha )
  u     (using)   = mean gradient         unit( mean_i grad_L[i] )         # grad = d(logit_b - logit_a)/dh
  gamma (output)  = unembedding diff       unit( meta['wU_diff'] )          # same for both regimes

Output per (regime, layer): signed cosines
  cos(w_res,u)  cos(delta,u)  cos(w_res,delta)            # the three pairwise couplings
  cos(w_res,gamma)  cos(delta,gamma)  cos(u,gamma)        # output alignment of each axis
  auc_wres   (decodability of w_res at this layer)
  norm_u  norm_delta  (where the using / writing signal concentrates)

The coupling-curve test then asks: is there a layer where cos(delta,u) RISES (writing couples to
using) BEFORE cos(u,gamma) saturates (trivial output convergence)? And does scaffold shift WHERE
or HOW MUCH this happens vs raw? That dynamic, ICL-modulated coupling is the one angle the static
read!=use literature (readout-mediator angle, Dyck, counting-ViTs) does not cover.

This file is small to RUN (CPU, seconds) and produces a tiny CSV (~72 rows) that is easy to send.

SELF-TEST:  python 147_coupling_extractor.py --self_test
RUN:        python 147_coupling_extractor.py \
              --raw_dump      data/analysis/runD_v2/field_dump \
              --scaffold_dump data/analysis/runD_v2/B1_alpha_beta/field_dump \
              --out coupling_curve.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("coupling147")


# ---------------------------------------------------------------------
# Core (exercised by --self_test)
# ---------------------------------------------------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def fisher_direction(res, y, shrink=0.1):
    """w_res = unit( Sigma_w_shrunk^{-1} (mu1 - mu0) ); Ledoit-Wolf-style diagonal shrinkage."""
    res = res.astype(np.float64)
    x0, x1 = res[y == 0], res[y == 1]
    mu0, mu1 = x0.mean(0), x1.mean(0)
    d = res.shape[1]
    Sw = (np.cov(x0, rowvar=False) * (len(x0) - 1) + np.cov(x1, rowvar=False) * (len(x1) - 1)) / (len(res) - 2)
    tau = np.trace(Sw) / d
    Sw_shr = (1 - shrink) * Sw + shrink * tau * np.eye(d)
    w = np.linalg.solve(Sw_shr, mu1 - mu0)
    return unit(w)


def delta_direction(res, y):
    res = res.astype(np.float64)
    return unit(res[y == 1].mean(0) - res[y == 0].mean(0))


def usage_direction(grad):
    return unit(grad.astype(np.float64).mean(0))


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(0)
    assert abs(np.linalg.norm(unit(rng.standard_normal(20))) - 1) < 1e-9
    e1 = np.array([1.0, 0, 0]); e2 = np.array([0, 1.0, 0])
    assert abs(cos(e1, e2)) < 1e-9 and abs(cos(e1, e1) - 1) < 1e-9
    assert cos(e1, -e1) < -0.99, "signed cosine keeps sign"

    # Fisher / delta: planted class separation along a known axis
    d, n = 12, 400
    y = np.array([i % 2 for i in range(n)])
    sep = unit(rng.standard_normal(d))
    res = rng.standard_normal((n, d)) * 0.3 + (y[:, None] * 2.0) * sep[None, :]
    dlt = delta_direction(res, y)
    assert abs(cos(dlt, sep)) > 0.95, f"delta recovers separation axis ({cos(dlt, sep):.2f})"
    wr = fisher_direction(res, y)
    assert abs(cos(wr, sep)) > 0.8, f"w_res aligns to separation ({cos(wr, sep):.2f})"

    # usage direction = mean gradient
    g = rng.standard_normal((n, d)) * 0.2 + sep[None, :]
    u = usage_direction(g)
    assert abs(cos(u, sep)) > 0.9, "u recovers mean-gradient direction"

    # AUC: w_res projection separates classes here (since sep drives both)
    a = auc(res @ wr, y)
    assert a > 0.9, f"w_res decodes ({a:.2f})"

    # planted orthogonality: if grad mean is orthogonal to sep, cos(u, delta) ~ 0
    g_orth = rng.standard_normal((n, d)) * 0.05 + unit(np.array([0.0] + [0] * (d - 2) + [1.0]))[None, :]
    # make it truly orthogonal to sep
    perp = np.zeros(d); perp[np.argmin(np.abs(sep))] = 1.0
    perp = unit(perp - (perp @ sep) * sep)
    g_orth = rng.standard_normal((n, d)) * 0.05 + perp[None, :]
    u_orth = usage_direction(g_orth)
    assert abs(cos(u_orth, dlt)) < 0.3, f"orthogonal grad -> low cos(u,delta) ({cos(u_orth, dlt):.2f})"

    print("[self_test] OK — unit/signed-cos, Fisher & delta recover separation, u recovers mean grad, "
          "AUC decodes, orthogonal grad gives low coupling. pass.")


# ---------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------
def process_regime(regime, dump_dir, shrink, intact_only=False):
    dump = Path(dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    y = np.asarray(meta["y"]).astype(int)
    n_layers = int(meta["n_layers"])
    gamma = unit(np.asarray(meta["wU_diff"], dtype=np.float64))
    # optional: keep only prompts where the model actually commits to alpha/beta (argmax in {a,b})
    keep = np.ones(len(y), dtype=bool)
    if intact_only:
        if "baseline_intact" not in meta:
            logger.warning("[%s] baseline_intact absent in meta; ignoring --intact_only", regime)
        else:
            keep = np.asarray(meta["baseline_intact"]).astype(bool)
            logger.info("[%s] intact_only: keeping %d/%d committed prompts", regime, int(keep.sum()), len(y))
            if keep.sum() < 20:
                logger.warning("[%s] only %d committed prompts — axes may be noisy", regime, int(keep.sum()))
    y = y[keep]
    rows = []
    for L in range(n_layers):
        res = np.load(dump / f"res_L{L:02d}.npy")[keep]
        grad = np.load(dump / f"grad_L{L:02d}.npy")[keep]
        w_res = fisher_direction(res, y, shrink)
        delta = delta_direction(res, y)
        u = usage_direction(grad)
        rows.append({
            "regime": regime, "layer": L,
            "cos_wres_u": cos(w_res, u),
            "cos_delta_u": cos(delta, u),
            "cos_wres_delta": cos(w_res, delta),
            "cos_wres_gamma": cos(w_res, gamma),
            "cos_delta_gamma": cos(delta, gamma),
            "cos_u_gamma": cos(u, gamma),
            "auc_wres": auc(res.astype(np.float64) @ w_res, y),
            "norm_u": float(np.linalg.norm(grad.astype(np.float64).mean(0))),
            "norm_delta": float(np.linalg.norm(res[y == 1].astype(np.float64).mean(0) - res[y == 0].astype(np.float64).mean(0))),
        })
        logger.info("[%s] L%02d  cos(wres,u)=%+.3f cos(delta,u)=%+.3f cos(u,gamma)=%+.3f auc=%.3f",
                    regime, L, rows[-1]["cos_wres_u"], rows[-1]["cos_delta_u"],
                    rows[-1]["cos_u_gamma"], rows[-1]["auc_wres"])
    return rows


def run_real(args):
    all_rows = []
    all_rows += process_regime("raw", args.raw_dump, args.shrink, args.intact_only)
    if args.scaffold_dump:
        all_rows += process_regime("scaffold", args.scaffold_dump, args.shrink, args.intact_only)

    cols = ["regime", "layer", "cos_wres_u", "cos_delta_u", "cos_wres_delta",
            "cos_wres_gamma", "cos_delta_gamma", "cos_u_gamma", "auc_wres", "norm_u", "norm_delta"]
    out = Path(args.out)
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(f"{r[c]:.5f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    logger.info("wrote %d rows (%d layers x regimes) to %s", len(all_rows), len(all_rows), out)
    logger.info("this CSV is small — send it back for the coupling-curve analysis.")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--raw_dump", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--scaffold_dump", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--intact_only", action="store_true",
                   help="compute axes only on prompts where the model commits to alpha/beta (baseline_intact==1)")
    p.add_argument("--out", default="coupling_curve.csv")
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
