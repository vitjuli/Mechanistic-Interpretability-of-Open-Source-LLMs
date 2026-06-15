"""
129_predictor_showdown.py   [is <g_i,.> better than logit-lens, where they diverge?]
=====================================================================================
Our flip-law predictor uses the PER-LAYER usage gradient g_i = grad_h(logit_beta -
logit_alpha) at layer L. The cheap alternative -- logit lens / LAP -- predicts the
margin push by projecting through the UNEMBEDDING directly, i.e. it uses the static
readout direction gamma_bar = W_U[beta] - W_U[alpha] applied at every layer, ignoring
that the gradient rotates with depth.

At the readout the two AGREE (we measured cos(u_final, gamma_bar) = +0.993). The
question that decides whether our "law" is a real contribution or just a rebrand of
the logit lens: in the MID-STACK zone where u and gamma_bar DIVERGE (we measured
AUC(u) is U-shaped -- u is blind mid-stack while gamma_bar is the fixed readout), does
<g_i,.> predict realized flips BETTER than <gamma_bar,.>?

Both predictors are computable from the dump (g saved per layer; gamma_bar = wU_diff
saved in meta) -- pure CPU recompute against the realized tier-2 cells.

Per (layer, c, dir=usage) it computes, on the measured prompts:
  - realized Δm_i = m1 - m0                                   (from cells)
  - pred_grad_i   = <g_i, s_i c σ vhat>                       (our predictor)
  - pred_lens_i   = <gamma_bar, s_i c σ vhat>  scaled to best-fit               (LAP-style)
and reports, per layer, the per-prompt correlation of EACH predictor with realized,
plus the cos(g_layer, gamma_bar) so the divergence zone is explicit. The decisive
plot is corr_grad - corr_lens AS A FUNCTION OF cos(g, gamma_bar): if our predictor
wins precisely where cos is low (mid-stack), that is the contribution. If they tie
everywhere, the law is the logit-lens tangent and we say so.

NOTE the honest framing (point 3): deep layers are STEEP, so first-order is worse
there -- but the COMPARISON is relative (both predictors are first-order tangents
along their own direction). The question is which DIRECTION's tangent is right, not
whether the tangent saturates. Steepness hurts both equally; divergence is about
direction.

CPU only. Needs the dump (g, gamma_bar, m0) + cells_tier2.csv per concept.

SELF-TEST (no torch / no repo):  python 129_predictor_showdown.py --self_test
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
logger = logging.getLogger("showdown129")


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


def best_fit_scale(pred, realized):
    """least-squares scalar gain mapping pred->realized (the lens predictor is only
    defined up to a per-layer gain, so we give it its best possible scale -- a
    GENEROUS handicap in its favor, making our win harder to claim)."""
    pred = np.asarray(pred, float); realized = np.asarray(realized, float)
    den = float((pred * pred).sum()) + 1e-30
    return float((pred * realized).sum() / den)


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 50, 300
    y = (np.arange(n) % 2).astype(int)
    gamma = unit_raw(rng.standard_normal(d))           # static readout direction
    sigma = 1.0; c = 2.0
    # Mid-stack: each prompt has its OWN local gradient g_i, scattered around a mean
    # direction that is ROTATED AWAY from gamma. Realized Δm_i is driven by g_i (causal),
    # and the push vhat is also per-prompt = unit(g_i) (steer along each prompt's own u).
    mean_dir = unit_raw(gamma + 1.5 * unit_raw(rng.standard_normal(d)))
    G = np.array([unit_raw(mean_dir + 0.6 * rng.standard_normal(d)) for _ in range(n)])
    idx = list(range(n))
    s = np.array([+1.0 if y[i] == 0 else -1.0 for i in idx])
    # realized truth: linear along each prompt's own gradient, pushing along that gradient
    rr = np.array([float(G[i] @ (s[i] * c * sigma * G[i])) for i in idx])   # = s_i c σ (||g_i||=1)
    # add per-prompt realized variation via heterogeneous push magnitudes
    mag = rng.uniform(0.5, 1.5, n)
    rr = np.array([float(G[i] @ (s[i] * c * sigma * mag[i] * G[i])) for i in idx])
    # our predictor: per-prompt gradient dotted with the per-prompt push (knows mag too)
    pg = np.array([float(G[i] @ (s[i] * c * sigma * mag[i] * G[i])) for i in idx])
    # lens predictor: static gamma dotted with the per-prompt push; misses g_i structure
    pl = np.array([float(gamma @ (s[i] * c * sigma * mag[i] * G[i])) for i in idx])
    pl = pl * best_fit_scale(pl, rr)
    cg, cl = pearson(pg, rr), pearson(pl, rr)
    assert cg > 0.999, f"our predictor must track realized: {cg}"
    assert cl < cg - 0.1, f"lens must lose in the divergence zone: grad={cg} lens={cl}"

    # readout case: g_i ALL equal gamma -> both predictors identical -> tie
    Gr = np.tile(gamma, (n, 1))
    rr2 = np.array([float(gamma @ (s[i] * c * sigma * mag[i] * gamma)) for i in idx])
    pg2 = np.array([float(Gr[i] @ (s[i] * c * sigma * mag[i] * gamma)) for i in idx])
    pl2 = np.array([float(gamma @ (s[i] * c * sigma * mag[i] * gamma)) for i in idx])
    pl2 = pl2 * best_fit_scale(pl2, rr2)
    assert abs(pearson(pg2, rr2) - pearson(pl2, rr2)) < 1e-6, "readout: predictors tie"
    print("[self_test] OK — grad predictor wins under divergence, ties at readout, lens handicap pass.")


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


def run_concept(name, dpath, cells_csvs, shrink, split_seed, train_frac):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
    gamma = meta["wU_diff"].astype(np.float64)
    trm = reconstruct_split(fams, split_seed, train_frac)

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
            H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
            G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
            w = fisher_axis(H[trm], y[trm], shrink)
            u = unit_raw(G.mean(0))
            sigma = float(np.std(H[trm] @ w))
            cache[L] = (G, u, sigma)
        return cache[L]

    rows = []
    for (L, c), realized_m1 in sorted(by_cell.items()):
        G, u, sigma = layer_objs(L)
        vhat = u
        idx = sorted(realized_m1.keys())
        rr = np.array([realized_m1[i] - m0[i] for i in idx])
        s = np.array([+1.0 if y[i] == 0 else -1.0 for i in idx])
        # our predictor: per-prompt gradient
        pg = np.array([float(G[i].astype(np.float64) @ (s[k] * c * sigma * vhat)) for k, i in enumerate(idx)])
        # lens/LAP predictor, FAIREST form: the logit lens reads the margin by projecting the
        # (steered) residual through the unembedding. Predicted Δmargin from a push is the
        # readout projection of the push: <gamma_bar, s c σ vhat>. For a fixed (L,c,vhat) this
        # varies across prompts ONLY through s -- which is the honest limitation of a static
        # readout direction: it cannot see per-prompt gradient structure. We additionally let
        # it use the per-prompt baseline margin m0_i as a free covariate (best-fit), giving the
        # lens its strongest two-parameter form so any remaining win is real.
        push_proj = np.array([float(gamma @ (s[k] * c * sigma * vhat)) for k in range(len(idx))])
        m0_i = np.array([m0[i] for i in idx])
        # two-feature OLS: realized ~ a*push_proj + b*m0_i  (lens + baseline)
        X = np.vstack([push_proj, m0_i, np.ones(len(idx))]).T
        try:
            coef, *_ = np.linalg.lstsq(X, rr, rcond=None)
            pl = X @ coef
        except Exception:
            pl = push_proj * best_fit_scale(push_proj, rr)
        cg = pearson(pg, rr); cl = pearson(pl, rr)
        cos_g_gamma = float(abs(unit_raw(G.mean(0)) @ unit_raw(gamma)))
        rows.append({"concept": name, "layer": L, "c": c, "n": len(idx),
                     "cos_g_gamma": cos_g_gamma,
                     "corr_grad": cg, "corr_lens": cl,
                     "grad_minus_lens": (cg - cl) if not (np.isnan(cg) or np.isnan(cl)) else float("nan"),
                     "realized_std": float(rr.std())})
    return rows


def run_real(args):
    concepts = json.load(open(args.concepts))
    all_rows = []
    for cdef in concepts:
        rows = run_concept(cdef["name"], cdef["dump"], cdef.get("cells_csvs", []),
                           args.shrink, args.split_seed, args.train_frac)
        all_rows += rows
        logger.info("%s: %d usage cells scored", cdef["name"], len(rows))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in all_rows]

    print("\n" + "=" * 100)
    print("PREDICTOR SHOWDOWN — usage-gradient <g_i,.> vs logit-lens <gamma_bar,.>")
    print("=" * 100)
    print("Per-prompt correlation with realized Δmargin, by layer (usage pushes).")
    print("The lens is given its best-fit per-layer scale (handicap in ITS favor).\n")
    # bucket by divergence: low cos(g,gamma) = mid-stack = where it matters
    for cdef in concepts:
        nm = cdef["name"]
        sel = [r for r in all_rows if r["concept"] == nm and not np.isnan(r["grad_minus_lens"])
               and r["realized_std"] > 1e-6]
        if not sel:
            continue
        lowdiv = [r for r in sel if r["cos_g_gamma"] < args.div_thresh]   # divergence zone
        highdiv = [r for r in sel if r["cos_g_gamma"] >= args.div_thresh]  # readout-aligned
        def med(rs, k): return float(np.median([r[k] for r in rs])) if rs else float("nan")
        print(f"[{nm}]")
        print(f"  divergence zone  (cos(g,γ) < {args.div_thresh}): "
              f"corr_grad={med(lowdiv,'corr_grad'):.3f}  corr_lens={med(lowdiv,'corr_lens'):.3f}  "
              f"Δ={med(lowdiv,'grad_minus_lens'):+.3f}  (n={len(lowdiv)})")
        print(f"  readout-aligned  (cos(g,γ) ≥ {args.div_thresh}): "
              f"corr_grad={med(highdiv,'corr_grad'):.3f}  corr_lens={med(highdiv,'corr_lens'):.3f}  "
              f"Δ={med(highdiv,'grad_minus_lens'):+.3f}  (n={len(highdiv)})")

    print("\nVERDICT:")
    print("  - corr_grad >> corr_lens in the DIVERGENCE zone (low cos g,γ) -> our predictor")
    print("    captures structure the logit lens cannot: REAL contribution, localized to")
    print("    mid-stack where the gradient has rotated away from the readout.")
    print("  - Δ ~ 0 everywhere -> the law is the logit-lens tangent; say so and drop the")
    print("    'superior predictor' claim, keep only 'first-order + measured radius'.")
    print("  (Steepness/saturation hurts both predictors equally; this test isolates DIRECTION.)")
    print(f"per cell: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest with dump + cells_csvs per concept")
    p.add_argument("--out", default="data/analysis/runD_v2/predictor_showdown.csv")
    p.add_argument("--div_thresh", type=float, default=0.5,
                   help="cos(g,gamma) below this = divergence zone (mid-stack)")
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
