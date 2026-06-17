"""
132_geometric_atlas.py   [the full geometry in one place — find where OUR signal is]
=====================================================================================
The "clouds separate and axes align to the output" picture is published
(2505.18752, 2603.12760). This atlas consolidates the FULL per-layer geometry in
both regimes so we can see the whole thing at once AND foreground the parts the ICL-
geometry literature does NOT cover. Per layer, per regime (raw / scaffold), it computes:

CLOUD GEOMETRY (mostly known backdrop):
  sep_snr      ||mu_beta - mu_alpha|| / sqrt(mean within-class variance)   separation SNR
  within_spread mean within-class std                                       cloud size
  auc_fisher   AUC of the best linear separator (Fisher)                    separability
  d_eff        (tr S)^2 / tr(S^2) of pooled centered cloud                  effective dim

TRIAD PAIRWISE ANGLES across depth (OURS — read≠use≠write as a depth profile):
  cos_wres_u, cos_wres_delta, cos_u_delta   |cos| between the three axes per layer

PER-AXIS ALIGNMENT TO OUTPUT gamma_bar (OURS — refines "alignment increases"):
  cos_wres_gamma, cos_u_gamma, cos_delta_gamma
  -> the key question: does ONLY u go to the output while w_res stays misaligned?
     That is read≠use via output-alignment, sharper than "alignment increases".

ANISOTROPY (OURS — connects to rogue dimensions / why w_res steering is inert):
  cos_sep_topPC  |cos(separation direction, top principal component of pooled cloud)|
  -> HIGH: classes separate along the dominant variance (cheap separation).
     LOW:  concept lives in a LOW-variance direction orthogonal to the spread.

ROTATION per layer (OURS — where in depth the axes turn):
  rot_wres, rot_u, rot_delta = 1 - |cos(axis_L, axis_{L+1})|   local rotation speed

All cosines carry a random-direction null p95 (high-d => tiny). All CPU, existing dumps.

Output: one tidy CSV (layer x regime x all quantities) + an "events" summary locating
the depth where separability emerges, where u aligns to gamma, where rotation peaks.

SELF-TEST (no torch / no repo):  python 132_geometric_atlas.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("atlas132")


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


def abs_cos(a, b):
    return float(abs(unit_raw(a) @ unit_raw(b)))


def auc_from_scores(s, y):
    """AUC of scores s for labels y in {0,1} via rank statistic."""
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(s))
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def d_eff_of(H):
    """participation ratio of the pooled centered cloud covariance spectrum."""
    Hc = H - H.mean(0)
    s = np.linalg.svd(Hc, compute_uv=False); ev = s ** 2 / max(len(H) - 1, 1)
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-30))


def sep_snr(H, y):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    within = 0.5 * (H[y == 0].var(0).mean() + H[y == 1].var(0).mean())
    return float(np.linalg.norm(mu1 - mu0) / (np.sqrt(within) + 1e-30))


def top_pc(H):
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    return unit_raw(Vt[0])


def random_cos_p95(d, vhat, rng, n=400):
    vhat = unit_raw(vhat)
    cs = [abs(float(unit_raw(rng.standard_normal(d)) @ vhat)) for _ in range(n)]
    return float(np.quantile(cs, 0.95))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 50, 400
    y = (np.arange(n) % 2).astype(int)

    # well-separated clouds along a known direction sep_dir; separation SNR grows with gap
    sep_dir = unit_raw(rng.standard_normal(d))
    H_close = 0.5 * np.outer(2 * y - 1.0, sep_dir) + rng.standard_normal((n, d))
    H_far = 3.0 * np.outer(2 * y - 1.0, sep_dir) + rng.standard_normal((n, d))
    assert sep_snr(H_far, y) > sep_snr(H_close, y), "separation SNR grows with centroid gap"
    assert auc_from_scores(H_far @ fisher_axis(H_far, y), y) > 0.9, "separable -> high AUC"

    # alignment cosine correctness
    a = unit_raw(rng.standard_normal(d)); b = unit_raw(rng.standard_normal(d))
    assert abs(abs_cos(a, a) - 1.0) < 1e-9 and abs_cos(a, b) < 0.6

    # anisotropy: if separation is ALONG the dominant variance, cos(sep, topPC) high;
    # if separation is along a LOW-variance direction, cos low.
    big = unit_raw(rng.standard_normal(d))                       # dominant variance direction
    small = rng.standard_normal(d); small -= (small @ big) * big; small = unit_raw(small)
    # cloud with big variance along `big`, tiny separation along `small`
    base = 5.0 * np.outer(rng.standard_normal(n), big)           # huge spread along big
    sepd = 0.5 * np.outer(2 * y - 1.0, small)                    # small separation along small
    H_lowvar = base + sepd + 0.1 * rng.standard_normal((n, d))
    pc = top_pc(H_lowvar); sep_vec = unit_raw(H_lowvar[y == 1].mean(0) - H_lowvar[y == 0].mean(0))
    assert abs_cos(sep_vec, pc) < 0.4, "separation orthogonal to dominant variance -> low cos"
    assert abs_cos(big, pc) > 0.9, "top PC recovers the dominant variance direction"

    # d_eff: isotropic ~ d; rank-1-dominated ~ small
    assert d_eff_of(rng.standard_normal((n, d))) > 0.5 * d, "isotropic d_eff ~ d"
    assert d_eff_of(np.outer(rng.standard_normal(n), big)) < 3, "rank-1 cloud d_eff ~ 1"

    p95 = random_cos_p95(d, sep_dir, rng, n=200); assert p95 < 0.45
    print("[self_test] OK — sep SNR, AUC, alignment, anisotropy (low/high), d_eff, null. pass.")


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


def axes_at(dump, L, y, held, shrink):
    H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
    G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
    Hh = H[held]; yh = y[held]
    w = fisher_axis(Hh, yh, shrink)
    u = unit_raw(G[held].mean(0))
    delta = unit_raw(Hh[yh == 1].mean(0) - Hh[yh == 0].mean(0))
    return Hh, yh, w, u, delta


def run_regime(name, dpath, gamma, shrink, split_seed, train_frac, rng):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int)
    d = int(meta["d"])
    held = ~reconstruct_split(fams, split_seed, train_frac)

    # cache axes for per-layer rotation
    cache = {}
    for L in range(n_layers):
        cache[L] = axes_at(dump, L, y, held, shrink)

    rows = []
    for L in range(n_layers):
        Hh, yh, w, u, delta = cache[L]
        sep = sep_snr(Hh, yh)
        spread = float(0.5 * (Hh[yh == 0].std(0).mean() + Hh[yh == 1].std(0).mean()))
        auc = auc_from_scores(Hh @ w, yh)
        deff = d_eff_of(Hh)
        pc = top_pc(Hh)
        sep_vec = unit_raw(Hh[yh == 1].mean(0) - Hh[yh == 0].mean(0))
        p95 = random_cos_p95(d, w, rng, n=200)

        # per-layer rotation (to next layer)
        def rot(axis_idx):
            if L + 1 not in cache:
                return float("nan")
            cur = (w, u, delta)[axis_idx]
            nxt = (cache[L + 1][2], cache[L + 1][3], cache[L + 1][4])[axis_idx]
            return float(1.0 - abs_cos(cur, nxt))

        rows.append({
            "regime": name, "layer": L,
            # cloud (backdrop)
            "sep_snr": sep, "within_spread": spread, "auc_fisher": auc, "d_eff": deff,
            # triad pairwise (ours)
            "cos_wres_u": abs_cos(w, u), "cos_wres_delta": abs_cos(w, delta),
            "cos_u_delta": abs_cos(u, delta),
            # alignment to output (ours)
            "cos_wres_gamma": abs_cos(w, gamma), "cos_u_gamma": abs_cos(u, gamma),
            "cos_delta_gamma": abs_cos(delta, gamma),
            # anisotropy (ours)
            "cos_sep_topPC": abs_cos(sep_vec, pc),
            # rotation (ours)
            "rot_wres": rot(0), "rot_u": rot(1), "rot_delta": rot(2),
            "null_cos_p95": p95,
        })
    return rows


def run_real(args):
    # gamma from whichever dump has it (same model)
    g_dump = Path(args.scaffold_dump)
    meta = np.load(g_dump / "meta.npz", allow_pickle=True)
    gamma = unit_raw(meta["wU_diff"].astype(np.float64))
    rng = np.random.default_rng(args.seed)

    rows = []
    rows += run_regime("raw", args.raw_dump, gamma, args.shrink, args.split_seed, args.train_frac, rng)
    rows += run_regime("scaffold", args.scaffold_dump, gamma, args.shrink, args.split_seed, args.train_frac, rng)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    # --- summaries the picture should answer ---
    def series(regime, key):
        return [(r["layer"], r[key]) for r in rows if r["regime"] == regime]

    def first_layer_above(regime, key, thr):
        for L, v in series(regime, key):
            if not np.isnan(v) and v > thr:
                return L
        return None

    def peak_layer(regime, key):
        s = [(L, v) for L, v in series(regime, key) if not np.isnan(v)]
        return max(s, key=lambda t: t[1])[0] if s else None

    print("\n" + "=" * 100)
    print("GEOMETRIC ATLAS — full per-layer geometry, raw vs scaffold")
    print("=" * 100)
    for regime in ("raw", "scaffold"):
        print(f"\n[{regime}]  'events' by depth:")
        print(f"  separability AUC>0.9 first at layer : {first_layer_above(regime,'auc_fisher',0.9)}")
        print(f"  u aligns to output cos(u,γ)>0.5 at  : {first_layer_above(regime,'cos_u_gamma',0.5)}")
        print(f"  w_res aligns to output cos>0.5 at   : {first_layer_above(regime,'cos_wres_gamma',0.5)}")
        print(f"  peak w_res rotation at layer        : {peak_layer(regime,'rot_wres')}")
        print(f"  peak u rotation at layer            : {peak_layer(regime,'rot_u')}")

    print("\nKEY NOVEL CONTRASTS (where OUR signal would show), ignition band L19-24 medians:")
    for regime in ("raw", "scaffold"):
        band = [r for r in rows if r["regime"] == regime and 19 <= r["layer"] <= 24]
        def m(k): return float(np.median([r[k] for r in band])) if band else float("nan")
        print(f"  [{regime}] cos(u,γ)={m('cos_u_gamma'):.3f}  cos(wres,γ)={m('cos_wres_gamma'):.3f}  "
              f"<- if u>>wres: USED axis goes to output, READABLE does not (read≠use via output)")
        print(f"           cos(sep,topPC)={m('cos_sep_topPC'):.3f}  "
              f"<- if LOW: concept separates along a LOW-variance direction (rogue-dim link)")
        print(f"           triad: wres-u={m('cos_wres_u'):.3f} wres-δ={m('cos_wres_delta'):.3f} "
              f"u-δ={m('cos_u_delta'):.3f} (null≈{m('null_cos_p95'):.3f})")
    print("\nINTERPRETATION GUIDE:")
    print("  - cos(u,γ) HIGH but cos(wres,γ) ~ null in late layers -> the USED axis aligns to the")
    print("    output while the READABLE axis does not. Sharper than 'alignment increases' (known).")
    print("  - cos(sep,topPC) LOW -> separation lives off the dominant-variance axis -> explains")
    print("    why steering along high-variance w_res is inert (rogue-dimension geometry).")
    print("  - the four 'events' coinciding in depth -> one ignition zone; staggered -> a pipeline.")
    print(f"saved: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--raw_dump", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--scaffold_dump", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--out", default="data/analysis/runD_v2/geometric_atlas.csv")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
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
