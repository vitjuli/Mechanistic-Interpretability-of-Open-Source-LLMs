"""
116_flip_law_retro.py   [GO/NO-GO: does one backward pass predict steering outcomes?]
=======================================================================================
THE gate of the research programme. The candidate law:

    margin after push      m1_i  ~=  m_i + <g_i, delta>          (first order)
    per-instance flip      flip_i =  [sign(m1_i) crosses toward the target]
    predicted flip-rate    F_hat(c, v, L) = mean_i flip_i  for delta = s*c*sigma_L*unit(v)

computable from the 119 field dump WITHOUT any new GPU intervention. If F_hat calibrates
against the steering sweeps ALREADY RUN (86 steering_efficiency.csv, 89 calculus_points.csv,
optionally 85), the cross-concept law programme is GO; if it fails even on alpha/beta where
our data is richest, the predictor form must be revised BEFORE committing weeks of GPU to
the concept battery.

What this script does (pure CPU):

(A) PREDICTED SURFACE, full corpus x ALL layers: for every layer L, directions
    {w_res, u_bar, shuffled-Fisher x k, random x k} and a saturating c-grid, compute
    predicted mean margin movement and predicted flip-rate F_hat on (i) the FULL corpus,
    (ii) held-out prompts only, (iii) the reconstructed 86 target subset (first
    max_targets per class of the held set under the seed-0 family split) — so the retro
    comparison is apples-to-apples.   -> flip_law_predicted_surface.csv

(B) RETRO-CALIBRATION against measured CSVs (whatever exists is used; columns are
    auto-detected):
      86 steering_efficiency.csv : (layer, c, dir, mean_dmargin_toward, margin_flip)
      89 calculus_points.csv     : per-point (pred, meas) margin changes
      85 csv (optional)          : per-(layer, c) flip / intact under w_res + nulls
    Metrics per amplitude c: R^2 and slope of predicted-vs-measured margin movement;
    mean |F_hat - F_meas| for flip-rates; the EMPIRICAL VALIDITY RADIUS = largest c
    whose calibration stays within tolerance.   -> flip_law_calibration.csv

(C) VERDICT: GO / REVISE with the exact numbers.

Predicted INTACT-flip is *not* first-order computable from the margin gradient alone
(it needs the full top-1 logit field); the law is stated and tested for margin-flip,
and intact comparisons from measured CSVs are reported alongside as context.

SELF-TEST (no torch / no repo):  python 116_flip_law_retro.py --self_test
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
logger = logging.getLogger("fliplaw116")


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
    ss_res = float(((meas - pred) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum()) + 1e-30
    return 1.0 - ss_res / ss_tot


def fit_slope(pred, meas):
    pred = np.asarray(pred, float); meas = np.asarray(meas, float)
    return float((pred @ meas) / ((pred @ pred) + 1e-30))


def predict_battery(G, m0, y, idx, v, c, sigma):
    """First-order prediction for the 86 steering protocol on prompt subset idx.
    Push TOWARD the opposite class: s=+1 (toward beta) for y=0, s=-1 for y=1.
    Returns (mean signed margin movement toward target,
             predicted flip-rate          [86 definition: ceiling = baseline accuracy],
             predicted normalized flip    [among baseline-correct prompts; ceiling 1.0 —
                                           the cross-concept-comparable metric],
             n_correct)."""
    vu = unit_raw(np.asarray(v, float))
    dm_t, fl, fl_n = [], [], []
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        dm = float(G[i].astype(np.float64) @ (s * c * sigma * vu))
        m1 = m0[i] + dm
        dm_t.append(s * dm)
        if y[i] == 0:    # alpha prompt steered toward beta
            f = int(m0[i] < 0 and m1 > 0); corr = m0[i] < 0
        else:            # beta prompt steered toward alpha
            f = int(m0[i] > 0 and m1 < 0); corr = m0[i] > 0
        fl.append(f)
        if corr:
            fl_n.append(f)
    return (float(np.mean(dm_t)), float(np.mean(fl)),
            float(np.mean(fl_n)) if fl_n else float("nan"), len(fl_n))


def binom_ci_halfwidth(p, n, z=1.96):
    return float(z * np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))


def detect_columns(header):
    """Map heterogeneous CSV headers (85/86 variants) onto canonical names."""
    h = [c.strip().lower() for c in header]
    def find(*cands):
        for c in cands:
            if c in h:
                return header[h.index(c)]
        return None
    return {"layer": find("layer", "l"),
            "c": find("c", "amp", "amplitude", "sigma_mult"),
            "dir": find("dir", "direction", "name", "vector"),
            "dmargin": find("mean_dmargin_toward", "mean_dmargin", "dmargin", "delta_margin"),
            "flip": find("margin_flip", "flip", "flip_rate"),
            "intact": find("intact_rate", "intact_flip", "intact")}


def load_measured(path):
    rows = []
    with open(path) as f:
        rd = _csv.DictReader(f)
        cols = detect_columns(rd.fieldnames)
        if cols["layer"] is None or cols["c"] is None:
            raise ValueError(f"{path}: cannot detect layer/c columns in {rd.fieldnames}")
        for r in rd:
            rows.append({"layer": int(float(r[cols["layer"]])),
                         "c": float(r[cols["c"]]),
                         "dir": (r[cols["dir"]] if cols["dir"] else "w_res"),
                         "dmargin": float(r[cols["dmargin"]]) if cols["dmargin"] and r[cols["dmargin"]] != "" else None,
                         "flip": float(r[cols["flip"]]) if cols["flip"] and r[cols["flip"]] != "" else None,
                         "intact": float(r[cols["intact"]]) if cols["intact"] and r[cols["intact"]] != "" else None})
    return rows


# =====================================================================
# Self-test: exact in a linear world, breaks detectably in a saturating one
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 30, 400
    a = 2.0 * unit_raw(rng.standard_normal(d))            # usage covector
    w = unit_raw(rng.standard_normal(d)); w -= (w @ unit_raw(a)) * unit_raw(a); w = unit_raw(w)
    y = (np.arange(n) % 2).astype(int)
    H = rng.standard_normal((n, d)) + np.outer(y * 2 - 1.0, 0.8 * unit_raw(a))   # classes split along a
    m_lin = H @ a
    G = np.tile(a, (n, 1))

    # (1) LINEAR world: predictor must match brute-force exactly at every amplitude
    idx = list(range(n)); sigma = 1.0
    for c in (0.5, 1, 4, 16):
        for v in (unit_raw(a), w):
            p_dm, p_fl, p_fln, _ = predict_battery(G, m_lin, y, idx, v, c, sigma)
            dm_t, fl = [], []
            for i in idx:
                s = +1.0 if y[i] == 0 else -1.0
                m1 = float((H[i] + s * c * sigma * unit_raw(v)) @ a)
                dm_t.append(s * (m1 - m_lin[i]))
                fl.append(int(m_lin[i] < 0 and m1 > 0) if y[i] == 0 else int(m_lin[i] > 0 and m1 < 0))
            assert abs(p_dm - np.mean(dm_t)) < 1e-9 and abs(p_fl - np.mean(fl)) < 1e-12, \
                f"linear-world mismatch at c={c}"
    # along w (orthogonal to usage) the predicted flip-rate must be ~0 at any c
    _, p_fl_w, _, _ = predict_battery(G, m_lin, y, idx, w, 16, sigma)
    assert p_fl_w == 0.0, "orthogonal direction must predict zero flips"

    # (2) SATURATING world: m(h) = tanh(h@a/4)*4 — predictor good at small c, breaks at large
    m_sat = np.tanh(m_lin / 4.0) * 4.0
    G_sat = np.stack([(1 - np.tanh(m_lin[i] / 4.0) ** 2) * a for i in range(n)])
    r2 = {}
    for c in (0.5, 8.0):
        preds, meas = [], []
        for i in idx[:150]:
            s = +1.0 if y[i] == 0 else -1.0
            delta = s * c * unit_raw(a)
            preds.append(float(G_sat[i] @ delta))
            meas.append(float(np.tanh((H[i] + delta) @ a / 4.0) * 4.0 - m_sat[i]))
        r2[c] = r_squared(preds, meas)
    assert r2[0.5] > 0.98 and r2[8.0] < 0.9, f"validity radius must be detectable: {r2}"

    # (3) measured-CSV column auto-detect
    import io, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "m.csv")
        with open(pth, "w", newline="") as f:
            wcsv = _csv.DictWriter(f, fieldnames=["layer", "c", "dir", "mean_dmargin_toward", "margin_flip", "intact_rate"])
            wcsv.writeheader(); wcsv.writerow({"layer": 5, "c": 4, "dir": "usage",
                                               "mean_dmargin_toward": 1.2, "margin_flip": 0.4, "intact_rate": 0.9})
        rows = load_measured(pth)
        assert rows[0]["layer"] == 5 and rows[0]["flip"] == 0.4 and rows[0]["dmargin"] == 1.2

    print("[self_test] OK — exact linear-world prediction, zero-flip orthogonality, "
          "validity-radius detection, CSV auto-detect pass.")


# =====================================================================
# Real run
# =====================================================================
def load_dump(dump_dir):
    dump = Path(dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    n_layers = int(meta["n_layers"])
    res = {L: np.load(dump / f"res_L{L:02d}.npy") for L in range(n_layers)}
    grad = {L: np.load(dump / f"grad_L{L:02d}.npy") for L in range(n_layers)}
    return meta, fams, res, grad, n_layers


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train_fams = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train_fams for f in fams], bool)


def run_real(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    meta, fams, res, grad, n_layers = load_dump(args.dump_dir)
    y = meta["y"].astype(int)
    m0 = meta["clean_margin"].astype(np.float64)
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    nP, d = res[0].shape
    held = [i for i in range(nP) if not trm[i]]
    ha = [i for i in held if y[i] == 0][: args.max_targets_86]
    hb = [i for i in held if y[i] == 1][: args.max_targets_86]
    sub86 = ha + hb
    subsets = {"full": list(range(nP)), "held": held, "sub86": sub86}
    logger.info("dump: %d prompts, %d layers | held=%d | reconstructed 86-subset=%d",
                nP, n_layers, len(held), len(sub86))

    # ---------- (A) predicted surface ----------
    pred_rows = []
    for L in range(n_layers):
        H = res[L].astype(np.float64); G = grad[L]
        w = fisher_axis(H[trm], y[trm], args.shrink)
        u = unit_raw(G.astype(np.float64).mean(0))
        sigma = float(np.std(H[trm] @ w))
        dirs = {"w_res": w, "usage": u}
        for k in range(args.n_shuffled):
            yp = y[trm].copy(); rng.shuffle(yp)
            dirs[f"shuffled{k}"] = fisher_axis(H[trm], yp, args.shrink)
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        for c in args.c_grid:
            for name, v in dirs.items():
                for sub_name, idx in subsets.items():
                    dm, fl, fln, n_corr = predict_battery(G, m0, y, idx, v, c, sigma)
                    pred_rows.append({"layer": L, "c": float(c), "dir": name, "subset": sub_name,
                                      "n": len(idx), "pred_dmargin": dm, "pred_flip": fl,
                                      "pred_flip_norm": fln, "n_correct": n_corr,
                                      "pred_flip_ci": binom_ci_halfwidth(fl, len(idx)),
                                      "sigma": sigma})
        if L % 4 == 0 or L == n_layers - 1:
            pf = {r["c"]: r["pred_flip"] for r in pred_rows
                  if r["layer"] == L and r["dir"] == "usage" and r["subset"] == "full"}
            pw = {r["c"]: r["pred_flip"] for r in pred_rows
                  if r["layer"] == L and r["dir"] == "w_res" and r["subset"] == "full"}
            logger.info("  L%02d predicted flip-rate (full corpus): usage %s | w_res %s",
                        L, {k: round(v, 2) for k, v in pf.items()}, {k: round(v, 2) for k, v in pw.items()})

    with open(out / "flip_law_predicted_surface.csv", "w", newline="") as f:
        wf = _csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); wf.writeheader()
        [wf.writerow(r) for r in pred_rows]

    # ---------- (B) retro-calibration ----------
    cal_rows, verdict = [], {}
    pred_idx = {(r["layer"], r["c"], r["dir"], r["subset"]): r for r in pred_rows}

    if args.measured_86 and Path(args.measured_86).exists():
        meas = load_measured(args.measured_86)
        # 86 ran on the sub86 subset; only deterministic directions are reproducible
        joined = []
        for r in meas:
            if r["dir"] not in ("usage", "w_res"):
                continue
            key = (r["layer"], r["c"], r["dir"], "sub86")
            if key in pred_idx:
                joined.append((pred_idx[key], r))
        if joined:
            for c in sorted({r["c"] for _, r in joined}):
                sel = [(p, r) for p, r in joined if r["c"] == c]
                pd = [p["pred_dmargin"] for p, r in sel if r["dmargin"] is not None]
                md = [r["dmargin"] for p, r in sel if r["dmargin"] is not None]
                pf = [p["pred_flip"] for p, r in sel if r["flip"] is not None]
                mf = [r["flip"] for p, r in sel if r["flip"] is not None]
                row = {"source": "86", "c": c, "n_cells": len(sel),
                       "r2_dmargin": r_squared(pd, md), "slope_dmargin": fit_slope(pd, md),
                       "mae_flip": float(np.mean(np.abs(np.array(pf) - np.array(mf)))) if pf else float("nan"),
                       "max_ae_flip": float(np.max(np.abs(np.array(pf) - np.array(mf)))) if pf else float("nan")}
                cal_rows.append(row)
                logger.info("  cal vs 86  c=%g: R2(dmargin)=%.3f slope=%.2f | flip MAE=%.3f max=%.3f (%d cells)",
                            c, row["r2_dmargin"], row["slope_dmargin"], row["mae_flip"], row["max_ae_flip"], len(sel))
        else:
            logger.warning("86 CSV found but no joinable (layer,c,dir) cells — check seed/split args")

    if args.measured_89 and Path(args.measured_89).exists():
        # 89 stores per-point (pred, meas) margin changes — re-derive R2/slope per c as
        # an independent consistency anchor for the same first-order law.
        with open(args.measured_89) as f:
            rd = _csv.DictReader(f)
            pts = [{"layer": int(float(r["layer"])), "c": float(r["c"]), "dir": r["dir"],
                    "pred": float(r["pred"]), "meas": float(r["meas"])} for r in rd]
        for c in sorted({q["c"] for q in pts}):
            sel = [q for q in pts if q["c"] == c]
            row = {"source": "89", "c": c, "n_cells": len(sel),
                   "r2_dmargin": r_squared([q["pred"] for q in sel], [q["meas"] for q in sel]),
                   "slope_dmargin": fit_slope([q["pred"] for q in sel], [q["meas"] for q in sel]),
                   "mae_flip": float("nan"), "max_ae_flip": float("nan")}
            cal_rows.append(row)
            logger.info("  cal vs 89  c=%g: R2=%.3f slope=%.2f (n=%d points)",
                        c, row["r2_dmargin"], row["slope_dmargin"], len(sel))

    if args.measured_85 and Path(args.measured_85).exists():
        try:
            meas = load_measured(args.measured_85)
            joined = [(pred_idx.get((r["layer"], r["c"], "w_res", "sub86")), r)
                      for r in meas if r["dir"].lower() in ("w_res", "wres")]
            joined = [(p, r) for p, r in joined if p is not None and r["flip"] is not None]
            if joined:
                pf = [p["pred_flip"] for p, r in joined]; mf = [r["flip"] for p, r in joined]
                row = {"source": "85", "c": -1.0, "n_cells": len(joined),
                       "r2_dmargin": float("nan"), "slope_dmargin": float("nan"),
                       "mae_flip": float(np.mean(np.abs(np.array(pf) - np.array(mf)))),
                       "max_ae_flip": float(np.max(np.abs(np.array(pf) - np.array(mf))))}
                cal_rows.append(row)
                logger.info("  cal vs 85 (w_res, all c pooled): flip MAE=%.3f max=%.3f (%d cells)",
                            row["mae_flip"], row["max_ae_flip"], len(joined))
        except Exception as e:
            logger.warning("85 CSV present but not joinable: %s", e)

    if cal_rows:
        with open(out / "flip_law_calibration.csv", "w", newline="") as f:
            wf = _csv.DictWriter(f, fieldnames=list(cal_rows[0].keys())); wf.writeheader()
            [wf.writerow(r) for r in cal_rows]

    # ---------- (C) verdict ----------
    print("\n" + "=" * 96)
    print("FLIP-LAW GO/NO-GO — does one backward pass predict steering outcomes on alpha/beta?")
    print("=" * 96)
    cal86 = [r for r in cal_rows if r["source"] == "86"]
    cal89 = [r for r in cal_rows if r["source"] == "89"]
    if not cal86 and not cal89:
        print("NO measured CSVs joined — predicted surface written; provide --measured_86/--measured_89 paths.")
    else:
        ok_c = []
        for r in sorted(cal86 + cal89, key=lambda q: q["c"]):
            good_r2 = (not np.isnan(r["r2_dmargin"])) and r["r2_dmargin"] >= args.r2_tol
            good_fl = np.isnan(r["mae_flip"]) or r["mae_flip"] <= args.flip_tol
            if good_r2 and good_fl:
                ok_c.append(r["c"])
            print(f"  src={r['source']:>2} c={r['c']:>5}: R2={r['r2_dmargin']:.3f} "
                  f"slope={r['slope_dmargin'] if not np.isnan(r['slope_dmargin']) else float('nan'):.2f} "
                  f"flipMAE={r['mae_flip']:.3f}" if not np.isnan(r["mae_flip"]) else
                  f"  src={r['source']:>2} c={r['c']:>5}: R2={r['r2_dmargin']:.3f}")
        radius = max(ok_c) if ok_c else 0.0
        go = radius >= args.go_radius
        print(f"\nempirical validity radius (R2>={args.r2_tol}, flip MAE<={args.flip_tol}): c* = {radius}")
        verdict_txt = ("GO — commit the cross-concept battery; lock the single-u_i predictor form"
                       if go else
                       "REVISE — predictor form fails inside the working amplitude range; "
                       "try usage-subspace U_k or second-order correction before the battery")
        print(f"VERDICT: {verdict_txt}")
    print("predicted surface: flip_law_predicted_surface.csv (full corpus x all layers x all dirs)")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/flip_law")
    p.add_argument("--measured_86", default="data/analysis/runD_v2/usage_direction/steering_efficiency.csv")
    p.add_argument("--measured_89", default="data/analysis/runD_v2/intervention_calculus/calculus_points.csv")
    p.add_argument("--measured_85", default=None, help="optional path to the 85 per-layer sweep CSV")
    p.add_argument("--c_grid", type=float, nargs="*", default=[0.5, 1, 2, 4, 8, 16, 32])
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--n_shuffled", type=int, default=3)
    p.add_argument("--max_targets_86", type=int, default=40, help="86's per-class target cap (subset reconstruction)")
    p.add_argument("--r2_tol", type=float, default=0.8)
    p.add_argument("--flip_tol", type=float, default=0.10)
    p.add_argument("--go_radius", type=float, default=4.0)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0, help="must match 86/89")
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
