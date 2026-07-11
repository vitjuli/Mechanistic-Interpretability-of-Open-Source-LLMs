"""
132_flip_law_assembly.py   [end-to-end flip law: front-end + CDF, all levels, one manifest]
========================================================================================
Assembles the complete flip-law evidence from a 119 field dump plus tier-2 per-prompt
sweep cells (122's --dump_cells output and 131's cells_tier2_delta.csv). No GPU.

Levels assembled per (layer, c, dir[, class]) cell:
  L0  exact identity        measured flip  <=>  realized adversarial drive a > |m0|
  L1  weak-coupling law     flip = E_a[F_|m|(a)]     (dispersion-aware, U-statistic)
  L2  mean-field law        flip = F_|m|(a_bar)      (valid iff drive concentrated)
  FE  front-end             a_pred = c * sigma * <g(p), v_hat>   (one backward pass; see predicted_drives for why s_y cancels)
      optional kappa fix    a_pred2 = a_pred + 0.5 * kappa_y * (c*sigma)^2

Every level is evaluated on BOTH realized drives (from the sweep m1-m0) and predicted
drives (from the dump gradients), so the end-to-end chain predicted-x -> measured-flip
is one table, and the total deviation decomposes into: first-order error (FE vs realized),
Jensen/dispersion (L2 vs L1), and coupling (L1 vs exact).

Outputs in --out_dir:
  flip_law_master.csv        one row per (cell x class-slice x drive-source x level)
  flip_law_premises.csv      A1 (CV of drive) and A2 (spearman(a,|m0|)) per cell
  flip_law_calibration.png   2x2 identity-line figure (realized|predicted x L2|L1)
  numbers_for_thesis.json    every headline number, keyed, for direct insertion
  heldout_F.json             prompt-split transfer test (KT3) per concept

Discipline options:
  --F_split {pool,train}     pool: F from the same baseline-correct pool (in-sample);
                             train: F from train-split prompts only (held-out F)
  --heldout_reps N           KT3: N stratified prompt splits, F on one half,
                             evaluation on the other, symmetrized

SELF-TEST (no repo data):  python 132_flip_law_assembly.py --self_test

Typical CSD3 run (per concept):
  python 132_flip_law_assembly.py \
      --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
      --cells    data/analysis/runD_v2/B1_alpha_beta/cells_tier2.csv \
                 data/analysis/runD_v2/B1_alpha_beta/cells_tier2_delta.csv \
      --concept  alpha_beta \
      --out_dir  data/analysis/runD_v2/B1_alpha_beta/flip_law \
      --split_seed <SAME AS 122> --train_frac <SAME AS 122> --heldout_reps 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("assembly132")

INFO_LO, INFO_HI = 0.05, 0.95      # informative-cell band on the predicted probability


# =====================================================================
# Shared conventions (identical to 122/131)
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


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


# =====================================================================
# Law primitives
# =====================================================================
def F_of(sample):
    """Empirical CDF of |m0| as a callable on arrays."""
    s = np.sort(np.asarray(sample, float))
    return lambda x: np.searchsorted(s, np.asarray(x, float), side="left") / len(s)


def level_values(a, absm, F):
    """Given per-prompt drives a and margins absm on the eval pool, and a CDF F:
       returns (mean-field, dispersion-aware, exact)."""
    mf = float(F(np.array([a.mean()]))[0])
    disp = float(F(a).mean())
    exact = float((a > absm).mean())
    return mf, disp, exact


def premises(a, absm):
    from scipy.stats import spearmanr
    abar = a.mean()
    cv = float(a.std() / abs(abar)) if abs(abar) > 1e-12 else float("inf")
    rho = float(spearmanr(a, absm).statistic) if len(a) > 5 else float("nan")
    return cv, rho


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(1)
    absm = np.abs(rng.normal(0, 1, 4000))
    F = F_of(absm)
    # (1) constant drive: all three levels coincide and equal the analytic CDF
    a = np.full(4000, 0.8)
    mf, disp, exact = level_values(a, absm, F)
    assert abs(mf - disp) < 1e-9 and abs(exact - mf) < 0.02
    # (2) dispersed independent drive: exact ~ disp, mean-field biased; Jensen sign
    a = np.abs(rng.normal(0.8, 0.6, 4000))
    mf, disp, exact = level_values(a, absm, F)
    assert abs(exact - disp) < 0.02, "independence level must match exact for independent drives"
    assert abs(exact - mf) > abs(exact - disp), "dispersion correction must improve on mean-field"
    # (3) coupled drive (a tracks the margin from below): exact collapses to ~0
    # while the independence level stays near 0.5 - coupling detected as exact != disp
    a = 0.8 * absm + np.abs(rng.normal(0, 0.001, 4000))
    mf, disp, exact = level_values(a, absm, F)
    assert exact < 0.1 and disp > 0.3 and (exact - disp) < -0.2, \
        "sub-threshold coupled drive must separate exact from the independence level"
    # (4) premises detect what they should
    cv, rho = premises(a, absm)
    assert rho > 0.5 and cv > 0.3
    print("[self_test] OK - CDF levels, Jensen ordering, coupling detection, premises pass.")


# =====================================================================
# Assembly
# =====================================================================
def load_cells(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        need = {"layer", "c", "dir", "idx", "y", "m0", "m1", "intact"}
        assert need.issubset(df.columns), f"{p}: missing columns {need - set(df.columns)}"
        frames.append(df)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["layer", "c", "dir", "idx"])
    d["correct"] = ((d.y == 0) & (d.m0 < 0)) | ((d.y == 1) & (d.m0 > 0))
    d = d[d.correct].copy()
    d["a_real"] = -np.sign(d.m0) * (d.m1 - d.m0)
    d["absm"] = d.m0.abs()
    d["flip"] = (d.a_real > d.absm).astype(int)
    d["iflip"] = d.flip * d.intact
    chk = np.where(d.y == 0, (d.m0 < 0) & (d.m1 > 0), (d.m0 > 0) & (d.m1 < 0)).astype(int)
    assert (d.flip.values == chk).all(), "flip identity violated - margin sign convention mismatch"
    return d


def predicted_drives(dump, cells, split_seed, train_frac, shrink, kappa=None):
    """Per (layer, dir): map idx -> per-prompt gradient projection <g(p), v_hat>.
       The first-order ADVERSARIAL drive on the baseline-correct pool is
           a_pred = c * sigma * <g(p), v_hat>          (NO s_y factor):
       a = -sign(m0)*dm and dm ~ c*sigma*s_y*<g,v>, and on baseline-correct
       prompts -sign(m0)*s_y == +1 identically for both classes, so the label
       sign cancels. Including s_y here clamps the y==1 half of the pool to
       zero and caps predicted flip rates at the class share (the 0.52/0.57
       ceiling bug). Genuinely unfavorable projections (<g,v> < 0) remain
       clamped downstream - those prompts cannot flip at first order."""
    fams = json.load(open(Path(dump) / "families.json"))
    meta = np.load(Path(dump) / "meta.npz", allow_pickle=True)
    y_all = meta["y"].astype(int)
    trm = reconstruct_split(fams, split_seed, train_frac)
    out = {}
    for L in sorted(cells.layer.unique()):
        H = np.load(Path(dump) / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(Path(dump) / f"grad_L{L:02d}.npy").astype(np.float64)
        w = fisher_axis(H[trm], y_all[trm], shrink)
        sigma = float(np.std(H[trm] @ w))
        vecs = {"w_res": w,
                "usage": unit_raw(G.mean(0)),
                "delta": unit_raw(H[trm][y_all[trm] == 1].mean(0) - H[trm][y_all[trm] == 0].mean(0))}
        for dname, v in vecs.items():
            out[(int(L), dname)] = (G @ unit_raw(v), sigma)
    return out


def assemble(cells, pred, concept, F_train=None, kappa_df=None):
    """Build the master table. F is estimated on the eval pool, or, if F_train is
    given (dict cls -> |m0| array of TRAIN-split baseline-correct prompts from the
    dump meta), on the train split (held-out-F discipline). |m0| is cell-independent,
    so the train CDF is global per class slice."""
    rows, prem = [], []
    for (L, c, dr), g in cells.groupby(["layer", "c", "dir"]):
        slices = [("pooled", g)] + [(int(k), gg) for k, gg in g.groupby("y")]
        for cls, gc in slices:
            absm = gc.absm.values
            Fsrc = absm if F_train is None else F_train[cls]
            if len(Fsrc) < 10 or len(gc) < 10:
                continue
            F = F_of(Fsrc)
            entry = dict(concept=concept, layer=int(L), c=float(c), dir=dr, cls=cls, n=len(gc))
            # realized drives
            mf, disp, exact = level_values(gc.a_real.values, absm, F)
            rows.append({**entry, "drive": "realized", "meanfield": mf, "disp": disp,
                         "exact": exact, "intact_exact": float(gc.iflip.mean())})
            cv, rho = premises(gc.a_real.values, absm)
            prem.append({**entry, "drive": "realized", "cv": cv, "rho": rho})
            # predicted drives (front-end), if gradients cover this (layer, dir)
            key = (int(L), dr if dr in ("w_res", "usage", "delta") else None)
            if key in pred:
                proj, sigma = pred[key]
                a_pred = float(c) * sigma * proj[gc.idx.values]
                if kappa_df is not None:
                    kk = kappa_df[(kappa_df.layer == L)]
                    if len(kk):
                        ky = np.where(gc.y.values == 0,
                                      float(kk.kappa_c0.iloc[0]), float(kk.kappa_c1.iloc[0]))
                        a_pred = a_pred + 0.5 * ky * (float(c) * sigma) ** 2
                # linearity diagnostics on favourable-sign prompts: the ignition map
                fav = a_pred > 1e-6
                lin = dict(ratio_med=np.nan, lin_rho=np.nan, lin_slope=np.nan,
                           n_fav=int(fav.sum()), cv_areal=np.nan)
                if fav.sum() >= 10:
                    ar, ap = gc.a_real.values[fav], a_pred[fav]
                    from scipy.stats import spearmanr
                    lin["ratio_med"] = float(np.median(ar / ap))
                    lin["lin_rho"] = float(spearmanr(ar, ap).statistic)
                    lin["lin_slope"] = float(np.polyfit(ap, ar, 1)[0])
                    med = np.median(ar)
                    lin["cv_areal"] = float((np.percentile(ar, 75) - np.percentile(ar, 25)) /
                                            (abs(med) + 1e-30))
                a_pred = np.maximum(a_pred, 0.0)     # unfavorable sign cannot flip (first order)
                mf, disp, exact_pred = level_values(a_pred, absm, F)
                rows.append({**entry, **lin, "drive": "predicted", "meanfield": mf, "disp": disp,
                             "exact": float((a_pred > absm).mean()),
                             "measured_ref": float(gc.flip.mean())})
                cv, rho = premises(a_pred, absm)
                prem.append({**entry, "drive": "predicted", "cv": cv, "rho": rho})
    return pd.DataFrame(rows), pd.DataFrame(prem)


def mae_block(T, drive, level, measured_col="exact"):
    """MAE on informative cells for a given drive source and law level.
       For drive='predicted', measured flips come from measured_ref (the sweep)."""
    t = T[(T.cls == "pooled") & (T.drive == drive)].copy()
    meas = t["measured_ref"] if drive == "predicted" else t["exact"]
    predv = t[level]
    sel = predv.between(INFO_LO, INFO_HI)
    if sel.sum() == 0:
        return {"n": 0}
    ae = (meas[sel] - predv[sel]).abs()
    return {"n": int(sel.sum()), "mae": float(ae.mean()), "max": float(ae.max()),
            "bias": float((meas[sel] - predv[sel]).mean())}


def heldout_F_test(cells, n_reps, seed=0):
    rng = np.random.default_rng(seed)
    prompts = cells[["idx", "y"]].drop_duplicates()
    res_ho, res_half = [], []
    groups = list(cells.groupby(["layer", "c", "dir"]))
    for _ in range(n_reps):
        A = set()
        for cl in (0, 1):
            ids = prompts[prompts.y == cl]["idx"].values
            A |= set(rng.choice(ids, size=len(ids) // 2, replace=False))
        for _, g in groups:
            inA = g["idx"].isin(A).values
            for tr, ev in ((inA, ~inA), (~inA, inA)):
                if tr.sum() < 10 or ev.sum() < 10:
                    continue
                F = F_of(g.absm.values[tr])
                aE, mE = g.a_real.values[ev], g.absm.values[ev]
                res_ho.append((float(F(aE).mean()), float(g.flip.values[ev].mean())))
                Fi = F_of(mE)
                res_half.append((float(Fi(aE).mean()), float(g.flip.values[ev].mean())))
    def mae(res):
        arr = np.array(res)
        sel = (arr[:, 0] >= INFO_LO) & (arr[:, 0] <= INFO_HI)
        return (float(np.abs(arr[sel, 1] - arr[sel, 0]).mean()) if sel.sum() else float("nan"),
                int(sel.sum()))
    (m_ho, n_ho), (m_hf, n_hf) = mae(res_ho), mae(res_half)
    return {"heldout_mae": m_ho, "heldout_n": n_ho,
            "matched_half_mae": m_hf, "matched_half_n": n_hf,
            "degradation": m_ho - m_hf}


def figure(T, out_png, concept):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"usage": "#1f77b4", "w_res": "#d62728", "delta": "#2ca02c",
              "random0": "#7f7f7f", "shuffled0": "#bcbd22"}
    P = T[T.cls == "pooled"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11), sharex=True, sharey=True)
    for i, drive in enumerate(["realized", "predicted"]):
        for j, level in enumerate(["meanfield", "disp"]):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], "k--", lw=1.4)
            t = P[P.drive == drive]
            meas = t["measured_ref"] if drive == "predicted" else t["exact"]
            for dr, gcol in colors.items():
                m = t.dir == dr
                ax.scatter(t[level][m], meas[m], s=40, color=gcol, alpha=0.85,
                           label=dr if (i == 0 and j == 0) else None)
            sel = t[level].between(INFO_LO, INFO_HI)
            mae = float((meas[sel] - t[level][sel]).abs().mean()) if sel.sum() else float("nan")
            ax.set_title("%s drive | %s law | MAE=%.3f (n=%d)"
                         % (drive, {"meanfield": "mean-field", "disp": "dispersion-aware"}[level],
                            mae, int(sel.sum())), fontsize=10)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("predicted flip probability")
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("measured flip rate")
    fig.suptitle("Flip-law calibration, end-to-end - %s" % concept, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--dump_dir"); ap.add_argument("--cells", nargs="+")
    ap.add_argument("--concept", default="concept"); ap.add_argument("--out_dir")
    ap.add_argument("--F_split", choices=["pool", "train"], default="pool")
    ap.add_argument("--heldout_reps", type=int, default=0)
    ap.add_argument("--kappa_csv", default=None,
                    help="optional CSV with columns layer,kappa_c0,kappa_c1 (from j124)")
    ap.add_argument("--shrink", type=float, default=0.1, help="MUST match the 122 run")
    ap.add_argument("--split_seed", type=int, default=0, help="MUST match the 122 run")
    ap.add_argument("--train_frac", type=float, default=0.6, help="MUST match the 122 run")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    assert args.dump_dir and args.cells and args.out_dir
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.cells)
    logger.info("cells: %d rows, dirs=%s, layers=%s",
                len(cells), sorted(cells["dir"].unique()), sorted(cells.layer.unique()))
    kappa_df = pd.read_csv(args.kappa_csv) if args.kappa_csv else None
    pred = predicted_drives(args.dump_dir, cells, args.split_seed, args.train_frac,
                            args.shrink, kappa_df)

    F_train = None
    if args.F_split == "train":
        fams = json.load(open(Path(args.dump_dir) / "families.json"))
        meta = np.load(Path(args.dump_dir) / "meta.npz", allow_pickle=True)
        y_all = meta["y"].astype(int); m_all = meta["clean_margin"].astype(np.float64)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        corr = ((y_all == 1) & (m_all > 0)) | ((y_all == 0) & (m_all < 0))
        sel = trm & corr
        F_train = {"pooled": np.abs(m_all[sel]),
                   0: np.abs(m_all[sel & (y_all == 0)]),
                   1: np.abs(m_all[sel & (y_all == 1)])}
        logger.info("held-out-F discipline: train-split CDF sizes pooled/0/1 = %d/%d/%d",
                    *(len(F_train[k]) for k in ("pooled", 0, 1)))
    T, PR = assemble(cells, pred, args.concept, F_train, kappa_df)
    T.to_csv(out / "flip_law_master.csv", index=False)
    PR.to_csv(out / "flip_law_premises.csv", index=False)

    numbers = {"concept": args.concept, "F_split": args.F_split,
               "n_pool": int(cells.groupby(["layer", "c", "dir"]).size().max())}
    for drive in ("realized", "predicted"):
        for level in ("meanfield", "disp"):
            numbers[f"{drive}_{level}"] = mae_block(T, drive, level)
    # predicted metrics restricted to the LINEAR RADIUS (ratio_med in [0.8,1.3], rho>=0.9):
    # outside it the first-order front-end measures ignition, not the law.
    if "ratio_med" in T.columns:
        linmask = T.ratio_med.between(0.8, 1.3) & (T.lin_rho >= 0.9)
        Tlin = T[(T.drive != "predicted") | linmask]
        for level in ("meanfield", "disp"):
            numbers[f"predicted_{level}_linear"] = mae_block(Tlin, "predicted", level)
        ign = T[(T.drive == "predicted") & (T.cls == "pooled") & (T.dir == "usage")]
        numbers["ignition_map"] = {
            f"L{int(r.layer)}_c{r.c:g}": {"ratio": round(float(r.ratio_med), 2),
                                          "rho": round(float(r.lin_rho), 2),
                                          "slope": round(float(r.lin_slope), 2),
                                          "cv_areal": round(float(r.cv_areal), 2)}
            for _, r in ign.iterrows() if np.isfinite(r.ratio_med)}
    u = PR[(PR.cls == "pooled") & (PR.dir == "usage") & (PR.drive == "realized")]
    numbers["A1_cv_usage_median"] = float(u.cv.median())
    numbers["A2_rho_usage_median"] = float(u.rho.median())
    if args.heldout_reps:
        numbers["heldout_F"] = heldout_F_test(cells, args.heldout_reps)
        json.dump(numbers["heldout_F"], open(out / "heldout_F.json", "w"), indent=2)
    json.dump(numbers, open(out / "numbers_for_thesis.json", "w"), indent=2)
    logger.info("numbers_for_thesis.json:\n%s", json.dumps(numbers, indent=2))

    figure(T, out / "flip_law_calibration.png", args.concept)
    logger.info("wrote %s", out / "flip_law_calibration.png")


if __name__ == "__main__":
    main()
