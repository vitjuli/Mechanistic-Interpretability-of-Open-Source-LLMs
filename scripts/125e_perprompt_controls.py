"""
125e_perprompt_controls.py   [is the per-prompt r=0.99 real, or a baseline-margin artifact?]
=============================================================================================
125d found per-prompt r ~ 0.99 between predicted and realized Δmargin on usage cells
within radius. Before trusting it, two controls -- the same discipline that collapsed
the old R²=1.000:

(A) PARTIAL CORRELATION removing the baseline-margin confound. Within a cell, both
    predicted Δm_i and realized Δm_i depend on how far the prompt sits from the
    decision boundary (m0_i). If both merely inherit that dependence, they correlate
    through m0, not through the law capturing per-prompt DIRECTION structure. We
    compute partial_r = corr(pred, realized | m0): residualize both pred and realized
    on [m0, 1] by OLS, correlate the residuals. If partial_r stays high -> the law
    captures structure beyond margin. If it collapses -> 0.99 was the margin confound.

(B) RANDOM-DIRECTION NULL predictor. Replace u with a random unit direction of the
    same norm, recompute the predicted Δm_i, and measure its per-prompt r (raw and
    partial). A random direction should give r ~ 0 once margin is controlled. If the
    null also scores high, the high r is a property of the data, not of u.

Outcome:
  - usage partial_r >> null partial_r (and usage partial_r not tiny) -> the law is
    CONTENTFUL as a per-prompt ranker beyond margin. State: "ranks per-prompt response
    above what baseline margin alone explains (partial r = X)".
  - usage partial_r ~ null partial_r ~ 0 -> the law adds nothing beyond margin;
    DOWNGRADE fully: "first order fixes only sign and rough margin-driven scale".

CPU only. Needs the dump (g, m0) + cells_tier2.csv per concept.

SELF-TEST (no torch / no repo):  python 125e_perprompt_controls.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("controls125e")


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


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def residualize(v, covars):
    """OLS-residualize vector v on covariate matrix covars (with intercept added)."""
    v = np.asarray(v, float)
    X = np.column_stack([np.asarray(covars, float), np.ones(len(v))])
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ coef


def partial_corr(pred, realized, control):
    """corr(pred, realized | control): residualize both on control, correlate residuals."""
    rp = residualize(pred, np.asarray(control).reshape(-1, 1))
    rr = residualize(realized, np.asarray(control).reshape(-1, 1))
    return pearson(rp, rr)


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    n = 200
    m0 = rng.standard_normal(n) * 2.0                       # baseline margins

    # CASE 1: realized = law structure (along a real signal) + margin dependence + noise.
    # The law signal is INDEPENDENT of m0. Partial corr should stay high.
    law_signal = rng.standard_normal(n)
    realized = 1.0 * law_signal + 1.5 * m0 + 0.05 * rng.standard_normal(n)
    pred = 1.0 * law_signal + 1.5 * m0                      # predictor tracks both
    raw_r = pearson(pred, realized)
    par_r = partial_corr(pred, realized, m0)
    assert raw_r > 0.95, f"raw r high: {raw_r}"
    assert par_r > 0.9, f"partial r should stay high when law signal is real: {par_r}"

    # CASE 2: realized and pred share ONLY the margin dependence (no law structure).
    # Raw r high (both driven by m0), partial r ~ 0.
    realized2 = 1.5 * m0 + 0.3 * rng.standard_normal(n)
    pred2 = 1.5 * m0 + 0.3 * rng.standard_normal(n)         # independent noise, shared m0
    raw_r2 = pearson(pred2, realized2)
    par_r2 = partial_corr(pred2, realized2, m0)
    assert raw_r2 > 0.9, f"raw r2 high via shared margin: {raw_r2}"
    assert abs(par_r2) < 0.3, f"partial r2 must collapse (no structure beyond margin): {par_r2}"

    # residualize removes the control exactly
    z = residualize(2.0 * m0 + 3.0, np.asarray(m0).reshape(-1, 1))
    assert np.std(z) < 1e-9, "residualizing an exact linear function of control -> ~0"
    print("[self_test] OK — partial corr keeps real structure, collapses margin-only, "
          "residualize exact. pass.")


# =====================================================================
# Real run
# =====================================================================
def load_dump(dpath):
    dump = Path(dpath)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    return dump, meta, fams, int(meta["n_layers"])


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_concept(name, dpath, cells_csvs, c_star, shrink, split_seed, train_frac, n_null, seed):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
    d = int(meta["d"])
    trm = reconstruct_split(fams, split_seed, train_frac)
    rng = np.random.default_rng(seed)

    cells = []
    for c in cells_csvs:
        if Path(c).exists():
            with open(c) as f:
                cells += list(_csv.DictReader(f))
    if not cells:
        return []

    by_cell = defaultdict(dict)
    for r in cells:
        if r["dir"] != "usage":
            continue
        by_cell[(int(r["layer"]), float(r["c"]))][int(r["idx"])] = float(r["m1"])

    cache = {}
    def layer_objs(L):
        if L not in cache:
            G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
            H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
            u = unit_raw(G.mean(0)); sigma = float(np.std(H[trm] @ fisher_axis(H[trm], y[trm], shrink)))
            cache[L] = (G, u, sigma)
        return cache[L]

    rows = []
    for (L, c), realized_m1 in sorted(by_cell.items()):
        if c > c_star:
            continue                                   # within radius only
        G, u, sigma = layer_objs(L)
        idx = sorted(realized_m1.keys())
        s = np.array([+1.0 if y[i] == 0 else -1.0 for i in idx])
        m0_i = np.array([m0[i] for i in idx])
        realized = np.array([realized_m1[i] - m0[i] for i in idx])

        pred_u = np.array([float(G[i].astype(np.float64) @ (s[k] * c * sigma * u)) for k, i in enumerate(idx)])
        raw_u = pearson(pred_u, realized)
        par_u = partial_corr(pred_u, realized, m0_i)

        # random-direction null predictors (same norm as the push), averaged
        raw_null, par_null = [], []
        for _ in range(n_null):
            r_dir = unit_raw(rng.standard_normal(d))
            pred_r = np.array([float(G[i].astype(np.float64) @ (s[k] * c * sigma * r_dir)) for k, i in enumerate(idx)])
            raw_null.append(pearson(pred_r, realized))
            par_null.append(partial_corr(pred_r, realized, m0_i))
        rows.append({"concept": name, "layer": L, "c": c, "n": len(idx),
                     "raw_r_u": raw_u, "partial_r_u": par_u,
                     "raw_r_null_mean": float(np.nanmean(raw_null)),
                     "raw_r_null_p95": float(np.nanquantile(np.abs(raw_null), 0.95)),
                     "partial_r_null_mean": float(np.nanmean(par_null)),
                     "partial_r_null_p95": float(np.nanquantile(np.abs(par_null), 0.95)),
                     "u_above_null_partial": int(abs(par_u) > float(np.nanquantile(np.abs(par_null), 0.95)))})
    return rows


def run_real(args):
    concepts = json.load(open(args.concepts))
    all_rows = []
    for cdef in concepts:
        rows = run_concept(cdef["name"], cdef["dump"], cdef.get("cells_csvs", []),
                           float(cdef.get("c_star", args.c_star_default)),
                           args.shrink, args.split_seed, args.train_frac, args.n_null, args.seed)
        all_rows += rows
        logger.info("%s: %d within-radius usage cells", cdef["name"], len(rows))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in all_rows]

    print("\n" + "=" * 100)
    print("PER-PROMPT r CONTROLS — does r=0.99 survive margin-control and beat a random direction?")
    print("=" * 100)
    print(f"{'concept':<15}{'raw r(u)':>10}{'partial r(u)':>14}{'partial r null p95':>20}{'verdict':>16}")
    for cdef in concepts:
        nm = cdef["name"]
        sel = [r for r in all_rows if r["concept"] == nm and not np.isnan(r["partial_r_u"])]
        if not sel:
            continue
        raw_med = float(np.median([r["raw_r_u"] for r in sel]))
        par_med = float(np.median([r["partial_r_u"] for r in sel]))
        null_med = float(np.median([r["partial_r_null_p95"] for r in sel]))
        above = sum(r["u_above_null_partial"] for r in sel)
        verdict = "CONTENTFUL" if par_med > max(0.3, null_med) else "margin-artifact"
        print(f"{nm:<15}{raw_med:>10.3f}{par_med:>14.3f}{null_med:>20.3f}"
              f"{verdict:>16}  ({above}/{len(sel)} cells u>null)")

    print("\nINTERPRETATION:")
    print("  - partial r(u) stays well above 0 AND above the random-direction null -> the law")
    print("    ranks per-prompt response beyond what baseline margin explains: CONTENTFUL.")
    print("    State: 'first-order ranks per-prompt response above margin alone (partial r=X),")
    print("    but underestimates absolute scale (curvature).'")
    print("  - partial r(u) ~ 0 or ~ null -> the 0.99 was the baseline-margin confound;")
    print("    DOWNGRADE fully: first order gives only sign + margin-driven scale.")
    print(f"per cell: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest with dump, cells_csvs, c_star per concept")
    p.add_argument("--out", default="data/analysis/runD_v2/perprompt_controls.csv")
    p.add_argument("--c_star_default", type=float, default=4.0)
    p.add_argument("--n_null", type=int, default=20)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.concepts, "--concepts manifest required"
    run_real(args)


if __name__ == "__main__":
    main()
