"""
125_headline_assemble.py   [cross-concept headline: predicted vs realized steerability]
=========================================================================================
Builds the B1 headline-graph data across concepts using the INTACT-CONDITIONED
metric (so concepts that break format at different depths are comparable).

Per concept it needs:
  - the 119 field dump (for w_res, u, sigma, margins, baseline-correct pool)
  - the 122 sweep CSVs (measured). flip_norm_intact is read directly if present
    (new 122 output); otherwise, if per-target cells_tier*.csv exist, it is
    recomputed on the baseline-correct pool; otherwise the concept is reported
    with measured=NaN and a note to re-run tier2 with the new metric.

For each concept it reports, at a fixed reporting amplitude c0 and a layer rule:
  PREDICTED steerability:
     F_hat_margin(c0)   first-order predicted flip_norm (margin) from the dump
     (intact is not first-order predictable from the margin gradient, so the
      predicted axis is margin-flip; the MEASURED axis carries the intact
      condition — the asymmetry is stated, not hidden)
  MEASURED steerability:
     flip_norm          measured margin-flip among baseline-correct
     flip_norm_intact   measured intact-conditioned flip (headline)
     intact_given_flip  how clean the flips are
  GEOMETRY:
     |cos(u, w_res)|, its within-span null p95, percentile, and excess
     (does the reading axis align with usage ABOVE the calibrated null?)
  along {w_res, usage}.

Layer rule: 'argmax_usage' (per-concept layer of max measured usage
flip_norm_intact) and a fixed late control (default 35). Both reported.

Output: headline_table.csv (one row per concept x dir x layer-rule) and a
console summary ranking concepts by realized w_res steerability — the spectrum.

SELF-TEST (no torch / no repo):  python 125_headline_assemble.py --self_test
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
logger = logging.getLogger("headline125")


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


def predict_flip_norm_margin(G, m0, y, idx, v, c, sigma):
    vu = unit_raw(np.asarray(v, float))
    fln = []
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        m1 = m0[i] + float(G[i].astype(np.float64) @ (s * c * sigma * vu))
        if y[i] == 0 and m0[i] < 0:
            fln.append(int(m1 > 0))
        elif y[i] == 1 and m0[i] > 0:
            fln.append(int(m1 < 0))
    return float(np.mean(fln)) if fln else float("nan")


def span_null_p95(Hc, anchor, rng, n=2000):
    _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    r = int((s > s.max() * 1e-10).sum())
    V = Vt[:r].T
    R = rng.standard_normal((n, r)) @ V.T
    cc = np.abs(R @ unit_raw(anchor)) / (np.linalg.norm(R, axis=1) + 1e-30)
    return float(np.quantile(cc, 0.95))


def recompute_flip_norm_intact(cell_rows):
    """from per-target records (idx,y,m0,m1,intact): flip_norm_intact + intact_given_flip
    on the baseline-correct pool, per (layer,c,dir)."""
    from collections import defaultdict
    grp = defaultdict(list)
    for r in cell_rows:
        grp[(int(r["layer"]), float(r["c"]), r["dir"])].append(r)
    out = {}
    for k, recs in grp.items():
        fln_it, nflip = [], 0
        for r in recs:
            y = int(r["y"]); m0 = float(r["m0"]); m1 = float(r["m1"]); it = int(r["intact"])
            corr = (m0 < 0) if y == 0 else (m0 > 0)
            if not corr:
                continue
            f = int(m1 > 0) if y == 0 else int(m1 < 0)
            nflip += f
            fln_it.append(f and it)
        out[k] = {"flip_norm_intact": float(np.mean(fln_it)) if fln_it else float("nan"),
                  "intact_given_flip": float(sum(fln_it) / nflip) if nflip else float("nan")}
    return out


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 30, 200
    a = unit_raw(rng.standard_normal(d)); y = (np.arange(n) % 2).astype(int)
    H = rng.standard_normal((n, d)) * 0.4 + np.outer(2 * y - 1.0, 0.8 * a)
    G = np.tile(a, (n, 1)); m0 = H @ a
    idx = list(range(n)); sigma = 1.0
    f = predict_flip_norm_margin(G, m0, y, idx, a, 2.0, sigma)
    assert 0.0 <= f <= 1.0 and f > 0.3, "usage-aligned push should flip a real fraction"
    f0 = predict_flip_norm_margin(G, m0, y, idx, unit_raw(rng.standard_normal(d)) - (rng.standard_normal(d) @ a) * a, 2.0, sigma)
    assert f0 < f, "orthogonal direction flips fewer"

    # recompute_flip_norm_intact on hand data
    cells = [
        {"layer": 5, "c": 4.0, "dir": "usage", "idx": 0, "y": 0, "m0": -1.0, "m1": 0.5, "intact": 1},
        {"layer": 5, "c": 4.0, "dir": "usage", "idx": 1, "y": 1, "m0": 0.8, "m1": -0.3, "intact": 0},
        {"layer": 5, "c": 4.0, "dir": "usage", "idx": 2, "y": 0, "m0": 0.3, "m1": 1.0, "intact": 1},  # incorrect -> excluded
    ]
    rc = recompute_flip_norm_intact(cells)[(5, 4.0, "usage")]
    assert abs(rc["flip_norm_intact"] - 0.5) < 1e-12 and abs(rc["intact_given_flip"] - 0.5) < 1e-12

    # span null p95 in a low-rank space is well above isotropic
    V = np.linalg.qr(rng.standard_normal((d, 5)))[0]
    Hc = rng.standard_normal((n, 5)) @ V.T
    p95 = span_null_p95(Hc, V[:, 0], rng)
    assert p95 > 0.3, f"low-rank span null should be large: {p95}"
    print("[self_test] OK — margin flip prediction, intact recompute, span null pass.")


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


def load_sweep(path):
    rows = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            def g(k):
                v = r.get(k, "")
                return float(v) if v not in ("", "nan", None) else float("nan")
            rows.append({"layer": int(r["layer"]), "c": float(r["c"]), "dir": r["dir"],
                         "flip_norm": g("flip_norm"),
                         "flip_norm_intact": g("flip_norm_intact"),
                         "intact_given_flip": g("intact_given_flip")})
    return rows


def run_concept(name, dpath, sweep_csvs, cells_csvs, c0, late_layer, shrink, split_seed, train_frac, rng):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
    trm = reconstruct_split(fams, split_seed, train_frac)
    held = np.where(~trm)[0]
    correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
    pool = [int(i) for i in held if correct[i]]

    meas = []
    for c in sweep_csvs:
        if Path(c).exists():
            meas += load_sweep(c)
    # backfill flip_norm_intact from per-target dumps if missing
    need_backfill = any(np.isnan(r["flip_norm_intact"]) for r in meas)
    if need_backfill:
        cell_rows = []
        for c in cells_csvs:
            if Path(c).exists():
                with open(c) as f:
                    cell_rows += list(_csv.DictReader(f))
        if cell_rows:
            rc = recompute_flip_norm_intact(cell_rows)
            for r in meas:
                k = (r["layer"], r["c"], r["dir"])
                if np.isnan(r["flip_norm_intact"]) and k in rc:
                    r["flip_norm_intact"] = rc[k]["flip_norm_intact"]
                    r["intact_given_flip"] = rc[k]["intact_given_flip"]

    def meas_at(L, c, dname, key):
        for r in meas:
            if r["layer"] == L and abs(r["c"] - c) < 1e-9 and r["dir"] == dname:
                return r[key]
        return float("nan")

    # layer rule: argmax measured usage flip_norm_intact at c0
    usage_at_c0 = [(r["layer"], r["flip_norm_intact"]) for r in meas
                   if r["dir"] == "usage" and abs(r["c"] - c0) < 1e-9
                   and not np.isnan(r["flip_norm_intact"])]
    argmax_layer = max(usage_at_c0, key=lambda t: t[1])[0] if usage_at_c0 else late_layer

    rows = []
    for rule, L in (("argmax_usage", argmax_layer), ("late_fixed", late_layer)):
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        Hc = H[trm] - H[trm].mean(0)
        w = fisher_axis(H[trm], y[trm], shrink)
        u = unit_raw(G.mean(0))
        sigma = float(np.std(H[trm] @ w))
        for dname, v in (("w_res", w), ("usage", u)):
            cos_uw = float(abs(u @ w))
            p95 = span_null_p95(Hc, w if dname == "w_res" else u, rng)
            rows.append({
                "concept": name, "layer_rule": rule, "layer": int(L), "dir": dname,
                "c0": c0,
                "pred_flip_norm_margin": predict_flip_norm_margin(G, m0, y, pool, v, c0, sigma),
                "meas_flip_norm": meas_at(L, c0, dname, "flip_norm"),
                "meas_flip_norm_intact": meas_at(L, c0, dname, "flip_norm_intact"),
                "meas_intact_given_flip": meas_at(L, c0, dname, "intact_given_flip"),
                "abs_cos_u_wres": cos_uw, "span_p95": p95,
                "cos_excess": cos_uw / (p95 + 1e-30),
                "cos_above_null": int(cos_uw > p95),
                "n_correct_pool": len(pool),
            })
    return rows


def run_real(args):
    rng = np.random.default_rng(args.seed)
    concepts = json.load(open(args.concepts))
    all_rows = []
    for cdef in concepts:
        logger.info("concept: %s", cdef["name"])
        all_rows += run_concept(cdef["name"], cdef["dump"],
                                cdef.get("sweep_csvs", []), cdef.get("cells_csvs", []),
                                args.c0, args.late_layer, args.shrink,
                                args.split_seed, args.train_frac, rng)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in all_rows]

    print("\n" + "=" * 100)
    print("CROSS-CONCEPT HEADLINE — intact-conditioned steerability vs geometry (layer rule: argmax_usage)")
    print("=" * 100)
    print(f"{'concept':<16}{'dir':<7}{'L':>3}  {'pred_fnm':>9}{'meas_fn':>9}{'meas_fn_it':>11}"
          f"{'int|flip':>9}  {'|cos|':>7}{'p95':>7}{'>null':>6}")
    for r in [x for x in all_rows if x["layer_rule"] == "argmax_usage"]:
        def fmt(v): return f"{v:.2f}" if not (isinstance(v, float) and np.isnan(v)) else "  nan"
        print(f"{r['concept']:<16}{r['dir']:<7}{r['layer']:>3}  {fmt(r['pred_flip_norm_margin']):>9}"
              f"{fmt(r['meas_flip_norm']):>9}{fmt(r['meas_flip_norm_intact']):>11}"
              f"{fmt(r['meas_intact_given_flip']):>9}  {r['abs_cos_u_wres']:>7.4f}{r['span_p95']:>7.3f}"
              f"{'yes' if r['cos_above_null'] else 'no':>6}")
    # spectrum ranking by measured w_res intact steerability
    wres = [r for r in all_rows if r["dir"] == "w_res" and r["layer_rule"] == "argmax_usage"]
    wres = sorted(wres, key=lambda r: -(r["meas_flip_norm_intact"] if not np.isnan(r["meas_flip_norm_intact"]) else -1))
    print("\nSPECTRUM (concepts by realized w_res intact-flip at c0 — the reading-axis steerability):")
    for r in wres:
        v = r["meas_flip_norm_intact"]
        print(f"  {r['concept']:<16} w_res flip_norm_intact = {v:.2f}" if not np.isnan(v)
              else f"  {r['concept']:<16} w_res flip_norm_intact =  nan (re-run tier2 with new metric)")
    print(f"\nNOTE: predicted axis is MARGIN-flip (first-order); measured axis is INTACT-conditioned.")
    print(f"headline table: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="json list: [{name, dump, sweep_csvs:[...], cells_csvs:[...]}]")
    p.add_argument("--out", default="data/analysis/runD_v2/headline_table.csv")
    p.add_argument("--c0", type=float, default=4.0, help="reporting amplitude (strict validity radius)")
    p.add_argument("--late_layer", type=int, default=35)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.concepts, "--concepts json required"
    run_real(args)


if __name__ == "__main__":
    main()
