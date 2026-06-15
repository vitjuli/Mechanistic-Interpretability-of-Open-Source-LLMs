"""
125d_honest_law_recompute.py   [turn R2=1.000 into an honest, sliced verdict]
==============================================================================
The headline "R2=1.000 up to c=4" mixes everything: both directions, both
regimes, all amplitudes. This script reads the per-target tier-2 cells (idx, y,
m0, m1, layer, c, dir -- already on disk) and recomputes the law's accuracy in
the slices that actually matter, so we know whether the claim survives once the
trivial cells are removed.

For the predicted side it recomputes the first-order Δmargin from the dump
(g_i . delta) on EXACTLY the cells the sweep measured, then:

(A) R2 of predicted vs realized Δmargin, split four ways:
      direction in {usage, w_res} x regime in {within radius c<=c*, beyond c>c*}.
    The w_res cells are the trivial ones (predicted ~0, realized ~0): a high
    pooled R2 can be carried by them alone. Splitting exposes that.

(B) PER-PROMPT correlation: for a fixed (layer, c, dir), correlate predicted
    Δm_i against realized Δm_i across prompts (not just cell means). Cell-mean
    R2 can look perfect while per-prompt structure is wrong; this checks the
    law at the individual level. Realized Δm_i = m1 - m0 is in the cells.

(C) DIRECTION-DISCRIMINATION vs null: does the predictor correctly rank usage
    above w_res (and above shuffled/random dirs) in realized effect? Report the
    predicted-vs-realized effect ordering and whether the gap exceeds what a
    sign-shuffled predictor would give. If the law cannot tell u from w_res
    beyond chance, "predicts steering" is hollow.

Outcome:
  - usage-cell R2 stays high within radius AND per-prompt corr is real AND
    direction-discrimination beats null -> the law is CONTENTFUL; state it
    precisely (first-order, within radius, on the causal direction).
  - usage-cell R2 collapses without the w_res cells -> HONESTLY downgrade to
    "first-order tangent + measured radius", no more.

CPU only. Needs the field dump (for g, m0) + cells_tier2.csv per concept.

SELF-TEST (no torch / no repo):  python 125d_honest_law_recompute.py --self_test
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
logger = logging.getLogger("honest125d")


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


def r_squared(pred, meas):
    pred = np.asarray(pred, float); meas = np.asarray(meas, float)
    if len(pred) < 2:
        return float("nan")
    ss = float(((meas - pred) ** 2).sum())
    st = float(((meas - meas.mean()) ** 2).sum()) + 1e-30
    return 1.0 - ss / st


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def predicted_dm_perprompt(g_rows, vhat, c, sigma, y, idx):
    """first-order predicted Δmargin toward the actual push direction, per prompt.
    push s_i = +1 for y=0 (toward beta), -1 for y=1. realized Δm sign convention in
    cells is (m1 - m0) with the SAME push, so predicted must match: dm_i = g_i . (s c σ vhat)."""
    out = {}
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        out[i] = float(g_rows[i].astype(np.float64) @ (s * c * sigma * unit_raw(vhat)))
    return out


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 40, 200
    y = (np.arange(n) % 2).astype(int)
    a = unit_raw(rng.standard_normal(d))
    # linear world: margin = h.a, gradient = a everywhere -> predicted == realized exactly
    H = rng.standard_normal((n, d)) * 0.5 + np.outer(2 * y - 1.0, a)
    g = np.tile(a, (n, 1)); m0 = H @ a
    idx = list(range(n)); sigma = 1.0; c = 2.0
    pred = predicted_dm_perprompt(g, a, c, sigma, y, idx)
    realized = {}
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        realized[i] = float((H[i] + s * c * sigma * a) @ a) - m0[i]
    p = [pred[i] for i in idx]; r = [realized[i] for i in idx]
    assert r_squared(p, r) > 0.999 and pearson(p, r) > 0.999, "linear world: perfect recovery"

    # w_res-like direction orthogonal to a -> predicted ~0, realized ~0: R2 ill-defined but
    # both near zero; the per-prompt corr should be ~nan/low, NOT high.
    w = rng.standard_normal(d); w -= (w @ a) * a; w = unit_raw(w)
    predw = predicted_dm_perprompt(g, w, c, sigma, y, idx)
    realw = {}
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        realw[i] = float((H[i] + s * c * sigma * w) @ a) - m0[i]
    assert np.mean(np.abs(list(predw.values()))) < 1e-9, "orthogonal predicted ~0"
    assert np.mean(np.abs(list(realw.values()))) < 1e-9, "orthogonal realized ~0"

    # direction discrimination: usage realized effect >> w_res realized effect
    eff_u = np.mean(np.abs(list(realized.values())))
    eff_w = np.mean(np.abs(list(realw.values())))
    assert eff_u > 10 * eff_w, "usage must dominate w_res in realized effect"

    # r_squared sanity: a constant-offset wrong prediction gives low/negative R2
    bad = [realized[i] + 5.0 for i in idx]
    assert r_squared(bad, [realized[i] for i in idx]) < 0, "offset prediction -> negative R2"
    print("[self_test] OK — linear recovery, orthogonal near-zero, direction dominance, R2 sanity pass.")


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


def run_concept(name, dpath, cells_csvs, c_star, shrink, split_seed, train_frac):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
    trm = reconstruct_split(fams, split_seed, train_frac)

    cells = []
    for c in cells_csvs:
        if Path(c).exists():
            with open(c) as f:
                cells += list(_csv.DictReader(f))
    if not cells:
        logger.warning("%s: no cells", name); return [], [], []

    # group realized Δm by (layer, c, dir) -> {idx: m1}
    by_cell = defaultdict(dict)
    for r in cells:
        by_cell[(int(r["layer"]), float(r["c"]), r["dir"])][int(r["idx"])] = float(r["m1"])

    # per-layer axes + sigma cache
    cache = {}
    def axes_for(L):
        if L not in cache:
            H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
            G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
            w = fisher_axis(H[trm], y[trm], shrink)
            u = unit_raw(G.mean(0))
            sigma = float(np.std(H[trm] @ w))
            cache[L] = (G, w, u, sigma)
        return cache[L]

    cell_rows, pp_rows = [], []
    for (L, c, dname), realized_m1 in by_cell.items():
        if dname not in ("usage", "w_res"):
            continue
        G, w, u, sigma = axes_for(L)
        vhat = u if dname == "usage" else w
        idx = sorted(realized_m1.keys())
        pred = predicted_dm_perprompt(G, vhat, c, sigma, y, idx)
        realized = {i: realized_m1[i] - m0[i] for i in idx}
        p = np.array([pred[i] for i in idx]); rr = np.array([realized[i] for i in idx])
        cell_rows.append({"concept": name, "layer": L, "c": c, "dir": dname,
                          "regime": "within" if c <= c_star else "beyond",
                          "n": len(idx),
                          "pred_mean": float(p.mean()), "realized_mean": float(rr.mean()),
                          "abs_realized_mean": float(np.abs(rr).mean())})
        pp_rows.append({"concept": name, "layer": L, "c": c, "dir": dname,
                        "regime": "within" if c <= c_star else "beyond",
                        "n": len(idx), "perprompt_pearson": pearson(p, rr),
                        "perprompt_r2": r_squared(p, rr)})
    return cell_rows, pp_rows, []


def run_real(args):
    concepts = json.load(open(args.concepts))
    all_cells, all_pp = [], []
    for cdef in concepts:
        cr, pp, _ = run_concept(cdef["name"], cdef["dump"], cdef.get("cells_csvs", []),
                                float(cdef.get("c_star", args.c_star_default)),
                                args.shrink, args.split_seed, args.train_frac)
        all_cells += cr; all_pp += pp
        logger.info("%s: %d cells recomputed", cdef["name"], len(cr))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_cells[0].keys())); w.writeheader()
        [w.writerow(r) for r in all_cells]
    ppout = out.with_name("law_perprompt.csv")
    with open(ppout, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_pp[0].keys())); w.writeheader()
        [w.writerow(r) for r in all_pp]

    print("\n" + "=" * 100)
    print("HONEST FLIP-LAW RECOMPUTE — R2 sliced by direction x regime (cell means)")
    print("=" * 100)
    print(f"{'concept':<15}{'dir':<7}{'regime':<9}{'n_cells':>8}{'R2(Δm)':>10}{'|realized|':>12}")
    for cdef in concepts:
        nm = cdef["name"]
        for dname in ("usage", "w_res"):
            for regime in ("within", "beyond"):
                sel = [r for r in all_cells if r["concept"] == nm and r["dir"] == dname
                       and r["regime"] == regime]
                if not sel:
                    continue
                r2 = r_squared([r["pred_mean"] for r in sel], [r["realized_mean"] for r in sel])
                eff = float(np.mean([r["abs_realized_mean"] for r in sel]))
                print(f"{nm:<15}{dname:<7}{regime:<9}{len(sel):>8}{r2:>10.3f}{eff:>12.4f}")

    print("\n" + "-" * 100)
    print("THE KEY CONTRAST — pooled R2 vs usage-only-within-radius R2 (does the claim survive?):")
    for cdef in concepts:
        nm = cdef["name"]
        pooled = [r for r in all_cells if r["concept"] == nm]
        uw = [r for r in all_cells if r["concept"] == nm and r["dir"] == "usage" and r["regime"] == "within"]
        r2_pool = r_squared([r["pred_mean"] for r in pooled], [r["realized_mean"] for r in pooled])
        r2_uw = r_squared([r["pred_mean"] for r in uw], [r["realized_mean"] for r in uw]) if uw else float("nan")
        # how much of the pooled fit is w_res trivial cells?
        wres = [r for r in pooled if r["dir"] == "w_res"]
        frac_trivial = float(np.mean([abs(r["realized_mean"]) < 0.05 for r in wres])) if wres else float("nan")
        print(f"  {nm:<15} pooled R2={r2_pool:.3f} | usage-within-radius R2={r2_uw:.3f} | "
              f"w_res cells near-zero: {frac_trivial:.0%}")

    print("\nPER-PROMPT correlation (law at the individual level, usage cells within radius):")
    for cdef in concepts:
        nm = cdef["name"]
        sel = [r for r in all_pp if r["concept"] == nm and r["dir"] == "usage"
               and r["regime"] == "within" and not np.isnan(r["perprompt_pearson"])]
        if sel:
            med = float(np.median([r["perprompt_pearson"] for r in sel]))
            lo = float(np.min([r["perprompt_pearson"] for r in sel]))
            print(f"  {nm:<15} median per-prompt r = {med:.3f} (min {lo:.3f}, n_cells {len(sel)})")
    print("\nVERDICT GUIDE:")
    print("  - usage-within-radius R2 high AND per-prompt r high -> law is CONTENTFUL on the")
    print("    causal direction; state precisely (first-order, within radius, direction u).")
    print("  - pooled R2 high but usage-within R2 low / per-prompt r low -> the perfect pooled")
    print("    number was carried by trivial w_res cells; DOWNGRADE the claim honestly.")
    print(f"cells: {out} | per-prompt: {ppout}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest with dump, cells_csvs, c_star per concept")
    p.add_argument("--out", default="data/analysis/runD_v2/law_cells_recompute.csv")
    p.add_argument("--c_star_default", type=float, default=4.0)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.concepts, "--concepts manifest required"
    run_real(args)


if __name__ == "__main__":
    main()
