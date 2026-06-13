"""
123_ab_scaffold_closure.py   [anchor §3.6 closure: 3 CPU analyses on the two dumps]
====================================================================================
Closes the three open items from the alpha_beta scaffold run. Pure CPU.

(A) RAW vs SCAFFOLD per-layer table — does the 2-shot scaffold change the geometry?
    For each layer and each dump: |cos(u, w_res)|, d_eff, sigma, held-out AUC along
    w_res and along u, mean |margin|, margin accuracy, per-class recall.
    -> ab_scaffold_comparison.csv

(B) BETA-CLOUD LEAK — is the representational asymmetry format-contingent like the
    behavioral one? Per layer, per dump, per axis (u, w_res): fit the class-midpoint
    threshold on train projections, measure on held-out
        leak_beta  = frac(beta  on the alpha side)
        leak_alpha = frac(alpha on the beta  side)
    If the raw-dump beta leak shrinks on the scaffold dump => geometry follows
    context; if it persists while behavior equalizes => downstream compensation.
    -> beta_cloud_leak.csv

(C) PER-CELL DOSE-RESPONSE JOIN — same metric, same pool. Recomputes first-order
    predictions of flip_norm and mean_dmargin_toward on EXACTLY the pools 122 used:
        tier1: first --t1_per_class held per class (86-compatible selection)
        tier2: all baseline-correct held prompts
    and joins per (layer, c, dir in {w_res, usage}) with the measured sweep CSVs.
    Flags cells where |pred - meas| flip exceeds tolerance; prints the breakdown
    map (where the first-order law fails => candidate nonlinear-ignition cells).
    -> dose_response_join.csv

SELF-TEST (no torch / no repo):  python 123_ab_scaffold_closure.py --self_test
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
logger = logging.getLogger("closure123")


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


def auc_scalar(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def d_eff_from_centered(Hc):
    n = Hc.shape[0]
    G = (Hc @ Hc.T) / n
    tr1 = float(np.trace(G)); tr2 = float((G * G).sum())
    return tr1 * tr1 / (tr2 + 1e-30)


def cloud_leak(proj_tr, y_tr, m_tr, proj_he, y_he):
    """Leak of each class across TWO thresholds on the axis:
    (1) 'dec' — the DECISION boundary pulled onto the axis: regress train margin on
        train projection, threshold = projection where predicted margin = 0. This is
        the threshold w.r.t. which the original beta-leak finding is defined.
        Guarded: if the margin-projection coupling is weak (|corr| < 0.1) the
        calibration is meaningless — returns nan for dec leaks and flags it.
    (2) 'mid' — class-midpoint threshold (NOTE: symmetric by construction for
        equal-variance classes; kept only as a variance-asymmetry probe).
    Orientation: axis oriented so class-1 train mean > class-0 train mean.
    Returns dict(leak_alpha_dec, leak_beta_dec, leak_alpha_mid, leak_beta_mid,
                 dec_calibrated)."""
    proj_tr = np.asarray(proj_tr, float); proj_he = np.asarray(proj_he, float)
    m_tr = np.asarray(m_tr, float)
    mu0, mu1 = proj_tr[y_tr == 0].mean(), proj_tr[y_tr == 1].mean()
    sgn = 1.0 if mu1 >= mu0 else -1.0
    p_tr, p_he = sgn * proj_tr, sgn * proj_he
    out = {}
    # midpoint threshold
    thr_mid = sgn * (mu0 + mu1) / 2.0
    out["leak_beta_mid"] = float((p_he[y_he == 1] < thr_mid).mean()) if (y_he == 1).any() else float("nan")
    out["leak_alpha_mid"] = float((p_he[y_he == 0] > thr_mid).mean()) if (y_he == 0).any() else float("nan")
    # decision-calibrated threshold: fit m ~ a*p + b on train, thr = -b/a
    sp, sm = p_tr.std() + 1e-30, m_tr.std() + 1e-30
    corr = float(np.corrcoef(p_tr, m_tr)[0, 1]) if sp > 1e-20 and sm > 1e-20 else 0.0
    if abs(corr) < 0.1:
        out["leak_beta_dec"] = float("nan"); out["leak_alpha_dec"] = float("nan")
        out["dec_calibrated"] = 0
    else:
        a = corr * sm / sp
        b = float(m_tr.mean() - a * p_tr.mean())
        thr_dec = -b / a
        out["leak_beta_dec"] = float((p_he[y_he == 1] < thr_dec).mean()) if (y_he == 1).any() else float("nan")
        out["leak_alpha_dec"] = float((p_he[y_he == 0] > thr_dec).mean()) if (y_he == 0).any() else float("nan")
        out["dec_calibrated"] = 1
    return out


def predict_cell(G, m0, y, idx, v, c, sigma):
    """First-order flip_norm + mean_dmargin_toward on pool idx (same defs as 122)."""
    vu = unit_raw(np.asarray(v, float))
    dm_t, fln, fl = [], [], []
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        dm = float(G[i].astype(np.float64) @ (s * c * sigma * vu))
        m1 = m0[i] + dm
        dm_t.append(s * dm)
        if y[i] == 0:
            f = int(m0[i] < 0 and m1 > 0); corr = m0[i] < 0
        else:
            f = int(m0[i] > 0 and m1 < 0); corr = m0[i] > 0
        fl.append(f)
        if corr:
            fln.append(f)
    return {"pred_dmargin": float(np.mean(dm_t)),
            "pred_flip": float(np.mean(fl)),
            "pred_flip_norm": float(np.mean(fln)) if fln else float("nan")}


def r_squared(pred, meas):
    pred = np.asarray(pred, float); meas = np.asarray(meas, float)
    if len(pred) < 2:
        return float("nan")
    ss = float(((meas - pred) ** 2).sum()); st = float(((meas - meas.mean()) ** 2).sum()) + 1e-30
    return 1.0 - ss / st


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 40, 600
    y = (np.arange(n) % 2).astype(int)
    a = unit_raw(rng.standard_normal(d))

    # (1) cloud_leak with a TRUE decision boundary: margin = h@a (boundary at proj 0)
    H_asym = rng.standard_normal((n, d)) * 0.5
    H_asym[y == 0] -= 1.0 * a            # alpha well left of the boundary
    H_asym[y == 1] += 0.1 * a            # beta barely right => leaks across proj=0
    m_lin = H_asym @ a                   # margin field, zero exactly at the boundary
    tr = np.arange(n) < n // 2
    lk = cloud_leak(H_asym[tr] @ a, y[tr], m_lin[tr], H_asym[~tr] @ a, y[~tr])
    assert lk["dec_calibrated"] == 1
    assert lk["leak_beta_dec"] > 0.3 and lk["leak_alpha_dec"] < 0.05, \
        f"asymmetric toy must show beta leak across the DECISION boundary: {lk}"
    # midpoint threshold is symmetric by construction for equal variances:
    assert abs(lk["leak_beta_mid"] - lk["leak_alpha_mid"]) < 0.08, "midpoint leak ~symmetric"
    H_sym = rng.standard_normal((n, d)) * 0.5 + np.outer(2 * y - 1, a)
    m_sym = H_sym @ a
    lk2 = cloud_leak(H_sym[tr] @ a, y[tr], m_sym[tr], H_sym[~tr] @ a, y[~tr])
    assert lk2["leak_beta_dec"] < 0.05 and lk2["leak_alpha_dec"] < 0.05, "symmetric toy: no leak"
    # orientation invariance
    lk3 = cloud_leak(H_sym[tr] @ (-a), y[tr], m_sym[tr], H_sym[~tr] @ (-a), y[~tr])
    assert abs(lk3["leak_beta_dec"] - lk2["leak_beta_dec"]) < 1e-12, "leak must be sign-invariant"
    # weak-coupling guard: random axis carries no margin information
    r_ax = unit_raw(rng.standard_normal(d))
    r_ax -= (r_ax @ a) * a; r_ax = unit_raw(r_ax)
    lk4 = cloud_leak(H_sym[tr] @ r_ax, y[tr], m_sym[tr], H_sym[~tr] @ r_ax, y[~tr])
    assert lk4["dec_calibrated"] == 0 and np.isnan(lk4["leak_beta_dec"]), "guard must trip"

    # (2) predict_cell matches a brute-force linear world incl. flip_norm
    G = np.tile(a, (n, 1)); m0 = H_sym @ a
    idx = list(np.where(~tr)[0]); sigma = 1.0
    pc = predict_cell(G, m0, y, idx, a, 2.0, sigma)
    fln, dm = [], []
    for i in idx:
        s = +1.0 if y[i] == 0 else -1.0
        m1 = float((H_sym[i] + s * 2.0 * a) @ a)
        dm.append(s * (m1 - m0[i]))
        if (y[i] == 0 and m0[i] < 0) or (y[i] == 1 and m0[i] > 0):
            fln.append(int(m0[i] < 0 and m1 > 0) if y[i] == 0 else int(m0[i] > 0 and m1 < 0))
    assert abs(pc["pred_dmargin"] - np.mean(dm)) < 1e-9
    assert abs(pc["pred_flip_norm"] - np.mean(fln)) < 1e-12

    # (3) AUC sane
    assert auc_scalar(H_sym @ a, y) > 0.95 and 0.4 < auc_scalar(rng.standard_normal(n), y) < 0.6
    print("[self_test] OK — leak metric (asym/sym/sign-invariance), matched-pool prediction, AUC pass.")


# =====================================================================
# Real run
# =====================================================================
def load_dump(dump_dir):
    dump = Path(dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    return dump, meta, fams, int(meta["n_layers"])


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def layer_objects(dump, L, y, trm, shrink):
    H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
    G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
    w = fisher_axis(H[trm], y[trm], shrink)
    u = unit_raw(G.mean(0))
    sigma = float(np.std(H[trm] @ w))
    return H, G, w, u, sigma


def load_sweep(path):
    rows = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            rows.append({"layer": int(r["layer"]), "c": float(r["c"]), "dir": r["dir"],
                         "mean_dmargin_toward": float(r["mean_dmargin_toward"]),
                         "flip_norm": float(r["flip_norm"]) if r["flip_norm"] not in ("", "nan") else float("nan"),
                         "margin_flip": float(r["margin_flip"])})
    return rows


def run_real(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- (A) + (B): per-layer comparison over both dumps ----------------
    comp_rows, leak_rows = [], []
    dumps = {"raw": args.raw_dump, "scaffold": args.scaffold_dump}
    for tag, dpath in dumps.items():
        if not dpath or not Path(dpath).exists():
            logger.warning("dump '%s' missing (%s) — section A/B will be one-sided", tag, dpath)
            continue
        dump, meta, fams, n_layers = load_dump(dpath)
        y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        he = ~trm
        correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
        for L in range(n_layers):
            H, G, w, u, sigma = layer_objects(dump, L, y, trm, args.shrink)
            Hc = H[trm] - H[trm].mean(0)
            comp_rows.append({
                "dump": tag, "layer": L,
                "abs_cos_u_wres": float(abs(u @ w)),
                "d_eff": d_eff_from_centered(Hc),
                "sigma": sigma,
                "auc_wres_held": auc_scalar(H[he] @ w, y[he]),
                "auc_u_held": auc_scalar(H[he] @ u, y[he]),
                "mean_abs_margin": float(np.abs(m0).mean()),
                "margin_acc": float(correct.mean()),
                "recall_c0": float(correct[y == 0].mean()),
                "recall_c1": float(correct[y == 1].mean()),
            })
            for axis_name, axis in (("u", u), ("w_res", w)):
                lk = cloud_leak(H[trm] @ axis, y[trm], m0[trm], H[he] @ axis, y[he])
                leak_rows.append({"dump": tag, "layer": L, "axis": axis_name, **lk})
            if L % 8 == 0 or L == n_layers - 1:
                logger.info("[%s] L%02d cos=%.4f d_eff=%.0f AUC(w)=%.3f AUC(u)=%.3f",
                            tag, L, comp_rows[-1]["abs_cos_u_wres"], comp_rows[-1]["d_eff"],
                            comp_rows[-1]["auc_wres_held"], comp_rows[-1]["auc_u_held"])

    # ---------------- (C): per-cell dose-response join on the scaffold dump ----------
    join_rows = []
    if args.scaffold_dump and Path(args.scaffold_dump).exists():
        dump, meta, fams, n_layers = load_dump(args.scaffold_dump)
        y = meta["y"].astype(int); m0 = meta["clean_margin"].astype(np.float64)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        held = np.where(~trm)[0]
        correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
        pools = {}
        t1 = [i for i in held if y[i] == 0][: args.t1_per_class] + \
             [i for i in held if y[i] == 1][: args.t1_per_class]
        pools["tier1"] = t1
        pools["tier2"] = [int(i) for i in held if correct[i]]
        sweeps = {}
        if args.tier1_csv and Path(args.tier1_csv).exists():
            sweeps["tier1"] = load_sweep(args.tier1_csv)
        if args.tier2_csv and Path(args.tier2_csv).exists():
            sweeps["tier2"] = load_sweep(args.tier2_csv)
        cache = {}
        for tier, meas_rows in sweeps.items():
            pool = pools[tier]
            for r in meas_rows:
                if r["dir"] not in ("w_res", "usage"):
                    continue
                L = r["layer"]
                if L not in cache:
                    cache[L] = layer_objects(dump, L, y, trm, args.shrink)
                H, G, w, u, sigma = cache[L]
                v = w if r["dir"] == "w_res" else u
                pc = predict_cell(G, m0, y, pool, v, r["c"], sigma)
                err = (abs(pc["pred_flip_norm"] - r["flip_norm"])
                       if not (np.isnan(pc["pred_flip_norm"]) or np.isnan(r["flip_norm"])) else float("nan"))
                join_rows.append({"tier": tier, "layer": L, "c": r["c"], "dir": r["dir"],
                                  "pred_flip_norm": pc["pred_flip_norm"], "meas_flip_norm": r["flip_norm"],
                                  "abs_err_flip_norm": err,
                                  "pred_dmargin": pc["pred_dmargin"],
                                  "meas_dmargin": r["mean_dmargin_toward"],
                                  "exceeds_tol": int(not np.isnan(err) and err > args.flip_tol)})
        # summaries
        for tier in sweeps:
            for c in sorted({r["c"] for r in join_rows if r["tier"] == tier}):
                sel = [r for r in join_rows if r["tier"] == tier and r["c"] == c]
                r2 = r_squared([r["pred_dmargin"] for r in sel], [r["meas_dmargin"] for r in sel])
                errs = [r["abs_err_flip_norm"] for r in sel if not np.isnan(r["abs_err_flip_norm"])]
                bad = [(r["layer"], r["dir"]) for r in sel if r["exceeds_tol"]]
                logger.info("[join %s] c=%g: R2(dmargin)=%.3f | flip_norm MAE=%.3f max=%.3f | "
                            "cells>tol: %s", tier, c, r2,
                            float(np.mean(errs)) if errs else float("nan"),
                            float(np.max(errs)) if errs else float("nan"),
                            bad if bad else "none")

    # ---------------- write ----------------
    def wcsv(name, rows):
        if not rows:
            return
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            [w.writerow(r) for r in rows]
    wcsv("ab_scaffold_comparison.csv", comp_rows)
    wcsv("beta_cloud_leak.csv", leak_rows)
    wcsv("dose_response_join.csv", join_rows)

    # ---------------- verdict ----------------
    print("\n" + "=" * 96)
    print("ANCHOR §3.6 CLOSURE — raw vs scaffold, beta-cloud leak, dose-response join")
    print("=" * 96)
    if comp_rows:
        for tag in dumps:
            sel = [r for r in comp_rows if r["dump"] == tag]
            if not sel:
                continue
            med = lambda k: float(np.median([r[k] for r in sel]))
            late = [r for r in sel if r["layer"] >= 28]
            print(f"[{tag:>8}] median |cos(u,w)|={med('abs_cos_u_wres'):.4f} | median d_eff={med('d_eff'):.0f} "
                  f"(late layers: {float(np.median([r['d_eff'] for r in late])):.0f}) | "
                  f"acc={sel[0]['margin_acc']:.3f} recall c0/c1={sel[0]['recall_c0']:.3f}/{sel[0]['recall_c1']:.3f}")
    if leak_rows:
        for tag in dumps:
            for ax in ("u", "w_res"):
                base = [r for r in leak_rows if r["dump"] == tag and r["axis"] == ax
                        and 19 <= r["layer"] <= 26]
                sel = [r for r in base if r["dec_calibrated"] == 1]
                if sel:
                    print(f"[{tag:>8}] beta-leak across DECISION boundary along {ax:>5} "
                          f"(L19-26 median): {float(np.median([r['leak_beta_dec'] for r in sel])):.3f} "
                          f"(alpha {float(np.median([r['leak_alpha_dec'] for r in sel])):.3f}; "
                          f"calibrated {len(sel)}/{len(base)} layers)")
                elif base:
                    print(f"[{tag:>8}] {ax}: decision calibration failed on all L19-26 "
                          f"(weak margin-projection coupling) — see mid columns in the CSV")
    if join_rows:
        bad = [r for r in join_rows if r["exceeds_tol"]]
        print(f"dose-response cells joined: {len(join_rows)} | exceeding flip tol {args.flip_tol}: {len(bad)}")
        if bad:
            from collections import Counter
            cnt = Counter((r["layer"], r["dir"]) for r in bad)
            print("  breakdown map (layer, dir) -> n cells over tol:", dict(cnt))
            print("  -> these are the candidate nonlinear-ignition cells; inspect before any claim.")
    print("outputs: ab_scaffold_comparison.csv | beta_cloud_leak.csv | dose_response_join.csv")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--raw_dump", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--scaffold_dump", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--tier1_csv", default="data/analysis/runD_v2/B1_alpha_beta/steering_sweep_tier1.csv")
    p.add_argument("--tier2_csv", default="data/analysis/runD_v2/B1_alpha_beta/steering_sweep_tier2.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/closure")
    p.add_argument("--t1_per_class", type=int, default=40)
    p.add_argument("--flip_tol", type=float, default=0.10)
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
