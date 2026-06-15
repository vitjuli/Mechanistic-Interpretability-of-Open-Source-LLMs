"""
127_rotation_control.py   [is the monotonic rotation law concept-specific or generic drift?]
=============================================================================================
The thesis chapter's signature quantitative result is the monotonic rotation law:
the concept axis turns across depth, so cos between the concept subspace at layer L
and at layer L+dk DECAYS with dk. The open question (chapter §7.1): is this decay a
property of THIS concept, or does ANY linearly-decodable attribute's subspace rotate
the same way (generic representational drift)? Answer needs a CONTROL concept.

We now have two concept dumps on disk (alpha_beta, grammar_number). This script
computes, for each concept, on the held-out set:

(A) per-layer concept axis  w_L = Fisher/LDA(residual_L, y)  (the reading direction),
    its unit form; the rotation curve is  rho(L, dk) = |cos(w_L, w_{L+dk})|  averaged
    over L within a band, as a function of dk. A concept-specific law shows a clean
    monotone decay; generic drift would look the same for ANY separating axis.

(B) THE CONTROL, two independent kinds so the claim is not one-sided:
    - CROSS-CONCEPT: compare the rotation curve of alpha_beta vs grammar_number. If
      both decay at the SAME rate, rotation is generic; if the RATES differ, rotation
      carries concept-specific structure.
    - SHUFFLED-LABEL NULL within each concept: w_L^shuf = Fisher on shuffled y. This
      is "any high-variance separating axis of the same norm" -- the strong null from
      the chapter. If the real rotation curve sits clearly below the shuffled curve
      (real axes stay MORE self-aligned across depth than random separators), the
      law is real structure, not an artifact of LDA on noise.

(C) Two summary scalars per concept: the decay constant (fit rho(dk)=exp(-dk/tau) ->
    tau, the rotation length-scale in layers) and the participation ratio of the
    stacked per-layer axes (PR of {w_L}: low => axes live in a small rotating plane;
    high => axes scatter). These are the comparable numbers for the chapter.

All CPU, both dumps already captured. Uses the same Fisher/split conventions as the
rest of the pipeline.

SELF-TEST (no torch / no repo):  python 127_rotation_control.py --self_test
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
logger = logging.getLogger("rot127")


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


def rotation_curve(axes, dks):
    """axes: dict L->unit vector. Returns {dk: mean_L |cos(w_L, w_{L+dk})|}."""
    Ls = sorted(axes)
    out = {}
    for dk in dks:
        vals = [abs(float(axes[L] @ axes[L + dk])) for L in Ls if (L + dk) in axes]
        out[dk] = float(np.mean(vals)) if vals else float("nan")
    return out


def fit_tau(dks, rhos):
    """fit rho = exp(-dk/tau) by least squares on log(rho); returns tau (layers)."""
    dks = np.asarray(dks, float); rhos = np.asarray(rhos, float)
    m = (rhos > 1e-6) & np.isfinite(rhos) & (dks > 0)
    if m.sum() < 2:
        return float("nan")
    # log rho = -dk/tau  -> slope = -1/tau through the data (allow intercept for robustness)
    A = np.vstack([dks[m], np.ones(m.sum())]).T
    slope, _ = np.linalg.lstsq(A, np.log(rhos[m]), rcond=None)[0]
    return float(-1.0 / slope) if slope < 0 else float("inf")


def participation_ratio_axes(axes):
    """PR of the stacked unit axes: low => they span a small rotating subspace."""
    M = np.stack([axes[L] for L in sorted(axes)])
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-30))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d = 80
    # Construct axes that rotate at a KNOWN rate within a 2D plane: w_L = cos(aL)e1 + sin(aL)e2.
    e1 = unit_raw(rng.standard_normal(d)); e2 = rng.standard_normal(d)
    e2 -= (e2 @ e1) * e1; e2 = unit_raw(e2)
    step = 0.12  # radians per layer
    axes = {L: unit_raw(np.cos(step * L) * e1 + np.sin(step * L) * e2) for L in range(24)}
    dks = [1, 2, 4, 8]
    cur = rotation_curve(axes, dks)
    # |cos| between layers L and L+dk = |cos(step*dk)|, independent of L -> exact
    for dk in dks:
        assert abs(cur[dk] - abs(np.cos(step * dk))) < 1e-9, f"rotation curve wrong at dk={dk}"
    # PR of a 2D-plane rotation is ~2
    assert 1.8 < participation_ratio_axes(axes) < 2.2, "planar rotation has PR ~2"

    # tau recovery on a true exponential
    dks2 = np.array([1, 2, 3, 4, 6, 8]); tau_true = 5.0
    rhos = np.exp(-dks2 / tau_true)
    assert abs(fit_tau(dks2, rhos) - tau_true) < 0.3, "tau fit must recover the decay constant"

    # rate discrimination on genuine DECAY curves (real axes decay, they don't rigidly rotate):
    # a faster-decaying concept has a SHORTER tau.
    rho_slow = np.exp(-dks2 / 8.0); rho_fast = np.exp(-dks2 / 3.0)
    t_slow = fit_tau(dks2, rho_slow); t_fast = fit_tau(dks2, rho_fast)
    assert t_fast < t_slow, f"faster decay must give shorter tau: {t_fast} vs {t_slow}"

    # random-walk baseline: step_corr=0 (any vector) collapses immediately -> tiny tau;
    # high persistence -> larger tau. The baseline must be MONOTONE in persistence.
    rng2 = np.random.default_rng(1)
    ax0 = random_walk_axes(30, 200, rng2, step_corr=0.0)
    c0 = rotation_curve(ax0, [1, 2, 4, 8])
    assert c0[1] < 0.2, "structureless walk: adjacent axes near-orthogonal (high-d)"
    ax9 = random_walk_axes(30, 200, np.random.default_rng(2), step_corr=0.9)
    c9 = rotation_curve(ax9, [1, 2, 4, 8])
    assert c9[1] > c0[1], "persistent walk stays more self-aligned at dk=1 than structureless"

    # u/delta axis builders: produce unit vectors; rotation curve well-defined.
    # toy "dump" in memory via a tiny shim is overkill — test the math directly:
    d2, n2 = 30, 60
    yy = (np.arange(n2) % 2).astype(int)
    Hm = np.random.default_rng(3).standard_normal((n2, d2))
    delta_vec = unit_raw(Hm[yy == 1].mean(0) - Hm[yy == 0].mean(0))
    assert abs(np.linalg.norm(delta_vec) - 1.0) < 1e-9, "delta axis is unit"
    Gm = np.random.default_rng(4).standard_normal((n2, d2))
    u_vec = unit_raw(Gm.mean(0))
    assert abs(np.linalg.norm(u_vec) - 1.0) < 1e-9, "u axis is unit"
    print("[self_test] OK — rotation curve exact, planar PR~2, tau recovery, rate discrimination, "
          "random-walk baseline monotonicity, u/delta unit axes pass.")


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


def concept_axes(dump, n_layers, y, mask, shrink, shuffle_rng=None):
    """per-layer Fisher axis (w_res, the READING direction) on the masked rows;
    if shuffle_rng given, labels shuffled."""
    axes = {}
    for L in range(n_layers):
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)[mask]
        yy = y[mask].copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(yy)
        if yy.min() == yy.max():
            continue
        axes[L] = fisher_axis(H, yy, shrink)
    return axes


def usage_axes(dump, n_layers, mask):
    """per-layer USAGE axis u = unit(mean gradient) on the masked rows. The gradient
    is grad_h(logit_beta - logit_alpha), stored per prompt; u is label-independent in
    construction (it's the output-sensitivity direction), so its proper null is the
    random-walk baseline, NOT shuffled labels."""
    axes = {}
    for L in range(n_layers):
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)[mask]
        axes[L] = unit_raw(G.mean(0))
    return axes


def writing_axes(dump, n_layers, y, mask, shuffle_rng=None):
    """per-layer WRITING axis delta = unit(mean_class1 - mean_class0) of the residual
    on the masked rows. Like Fisher it is class-difference based, so shuffled-label is
    a valid null (a random class split)."""
    axes = {}
    for L in range(n_layers):
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)[mask]
        yy = y[mask].copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(yy)
        if yy.min() == yy.max():
            continue
        axes[L] = unit_raw(H[yy == 1].mean(0) - H[yy == 0].mean(0))
    return axes


def random_walk_axes(n_layers, d, rng, step_corr=0.0):
    """Baseline axes with NO concept structure: each layer's axis is an independent
    random unit vector (step_corr=0) or a partially-correlated random walk (step_corr in
    [0,1)) where w_{L} = unit(step_corr*w_{L-1} + sqrt(1-step_corr^2)*noise). step_corr=0
    is the 'any vector' null; tuning it shows what rotation rate looks like for a
    structureless walk of a given persistence."""
    axes = {}
    prev = unit_raw(rng.standard_normal(d))
    for L in range(n_layers):
        if L == 0 or step_corr <= 0:
            cur = unit_raw(rng.standard_normal(d)) if step_corr <= 0 else prev
        else:
            noise = rng.standard_normal(d)
            cur = unit_raw(step_corr * prev + np.sqrt(1 - step_corr ** 2) * noise)
        axes[L] = cur; prev = cur
    return axes


def run_real(args):
    concepts = json.load(open(args.concepts))
    dks = args.dks
    rows, summ = [], []
    d_model = None
    for cdef in concepts:
        name = cdef["name"]
        dump, meta, fams, n_layers = load_dump(cdef["dump"])
        d_model = int(meta["d"])
        y = meta["y"].astype(int)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        held = ~trm

        # three axes: w_res (reading), u (using), delta (writing)
        axes_by_kind = {
            "w_res": concept_axes(dump, n_layers, y, held, args.shrink),
            "u": usage_axes(dump, n_layers, held),
            "delta": writing_axes(dump, n_layers, y, held),
        }

        for kind, axes in axes_by_kind.items():
            cur = rotation_curve(axes, dks)
            tau = fit_tau(dks, [cur[k] for k in dks])
            pr = participation_ratio_axes(axes)

            # null: shuffled-label for class-difference axes (w_res, delta); for u the
            # gradient is label-independent, so shuffled-label is not meaningful — we
            # compare u against the random-walk baseline (computed once below) instead,
            # and report tau_shuffled as nan for u.
            if kind in ("w_res", "delta"):
                null_curves, null_taus = [], []
                srng = np.random.default_rng(args.seed)
                builder = concept_axes if kind == "w_res" else writing_axes
                for _ in range(args.n_shuffle):
                    if kind == "w_res":
                        ax_s = concept_axes(dump, n_layers, y, held, args.shrink, shuffle_rng=srng)
                    else:
                        ax_s = writing_axes(dump, n_layers, y, held, shuffle_rng=srng)
                    nc = rotation_curve(ax_s, dks)
                    null_curves.append(nc)
                    null_taus.append(fit_tau(dks, [nc[k] for k in dks]))
                null_mean = {dk: float(np.mean([nc[dk] for nc in null_curves if not np.isnan(nc[dk])])) for dk in dks}
                null_p95 = {dk: float(np.quantile([nc[dk] for nc in null_curves if not np.isnan(nc[dk])], 0.95)) for dk in dks}
                tau_shuf = float(np.nanmean(null_taus))
            else:
                null_mean = {dk: float("nan") for dk in dks}
                null_p95 = {dk: float("nan") for dk in dks}
                tau_shuf = float("nan")

            for dk in dks:
                rows.append({"concept": name, "axis": kind, "dk": dk,
                             "rho_real": cur[dk], "rho_null_mean": null_mean[dk],
                             "rho_null_p95": null_p95[dk],
                             "real_above_null": int(cur[dk] > null_p95[dk]) if not np.isnan(null_p95[dk]) else -1})
            summ.append({"concept": name, "axis": kind, "tau_layers": tau, "tau_shuffled": tau_shuf,
                         "pr_axes": pr, "n_layers_used": len(axes),
                         "rho_dk1": cur.get(1, float("nan")), "rho_dk4": cur.get(4, float("nan")),
                         "rho_dk8": cur.get(8, float("nan"))})
            logger.info("[%s/%-5s] rho(dk1)=%.3f rho(dk8)=%.3f | tau=%.1f (shuf %.1f) | PR=%.1f",
                        name, kind, cur.get(1, float("nan")), cur.get(8, float("nan")), tau, tau_shuf, pr)

    # ---- third source: structureless random-walk baselines across a range of persistence ----
    rw_rng = np.random.default_rng(args.seed + 1)
    nL = summ_n_layers = None
    # use the max n_layers seen (dumps share 36)
    nL = 36
    rw_rows = []
    for sc in args.rw_step_corr:
        taus_rw = []
        for _ in range(args.n_rw):
            ax = random_walk_axes(nL, d_model, rw_rng, step_corr=sc)
            c = rotation_curve(ax, dks)
            taus_rw.append(fit_tau(dks, [c[k] for k in dks]))
        rw_rows.append({"step_corr": sc, "tau_mean": float(np.nanmean(taus_rw)),
                        "tau_p05": float(np.nanquantile(taus_rw, 0.05)),
                        "tau_p95": float(np.nanquantile(taus_rw, 0.95))})
        logger.info("random-walk baseline step_corr=%.2f -> tau=%.1f [%.1f, %.1f]",
                    sc, rw_rows[-1]["tau_mean"], rw_rows[-1]["tau_p05"], rw_rows[-1]["tau_p95"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    sout = out.with_name("rotation_summary.csv")
    with open(sout, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader()
        [w.writerow(r) for r in summ]
    rwout = out.with_name("rotation_rw_baseline.csv")
    with open(rwout, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rw_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rw_rows]

    print("\n" + "=" * 100)
    print("ROTATION OF THE TRIAD — do reading (w_res), using (u), writing (delta) axes rotate alike?")
    print("=" * 100)
    print(f"{'concept':<16}{'axis':<7}{'rho(dk1)':>9}{'rho(dk4)':>9}{'rho(dk8)':>9}{'tau':>7}{'tau_shuf':>10}{'PR':>7}")
    for s in summ:
        ts = f"{s['tau_shuffled']:>10.1f}" if not np.isnan(s['tau_shuffled']) else f"{'n/a':>10}"
        print(f"{s['concept']:<16}{s['axis']:<7}{s['rho_dk1']:>9.3f}{s['rho_dk4']:>9.3f}{s['rho_dk8']:>9.3f}"
              f"{s['tau_layers']:>7.1f}{ts}{s['pr_axes']:>7.1f}")
    print("\nStructureless random-walk tau (the shared null for ALL axes, esp. u):")
    for r in rw_rows:
        print(f"  step_corr={r['step_corr']:.2f}: tau = {r['tau_mean']:.1f} [{r['tau_p05']:.1f}, {r['tau_p95']:.1f}]")

    print("\nCROSS-AXIS COMPARISON (tau by axis, per concept):")
    for cdef in concepts:
        nm = cdef["name"]
        byax = {s["axis"]: s["tau_layers"] for s in summ if s["concept"] == nm}
        print(f"  {nm:<16} w_res={byax.get('w_res', float('nan')):.1f}  "
              f"u={byax.get('u', float('nan')):.1f}  delta={byax.get('delta', float('nan')):.1f}")
    print("\nVERDICT (read the numbers):")
    print("  - all three taus ~equal (and all << random-walk) -> rotation is a UNIFORM property")
    print("    of the residual stream: reading, using, and writing axes all rotate at ~the same")
    print("    characteristic rate. Strengthens 'geometry of computation is a network property'.")
    print("  - u rotates at a DIFFERENT rate than w_res/delta -> the triad is distinguished not")
    print("    only in direction but in DYNAMICS across depth. Strengthens read != use != write.")
    print("  - watch for u: likely fast mid-stack, slowing near readout (it glues to gamma_bar) —")
    print("    if so, u's rotation is NON-uniform across depth, unlike the steadier w_res.")
    print(f"saved: {out} | {sout} | {rwout}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest [{name, dump}] (reuse concepts_manifest.json)")
    p.add_argument("--out", default="data/analysis/runD_v2/rotation_control.csv")
    p.add_argument("--dks", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12])
    p.add_argument("--n_shuffle", type=int, default=10)
    p.add_argument("--rw_step_corr", type=float, nargs="*", default=[0.0, 0.5, 0.8, 0.9, 0.95],
                   help="persistence of the structureless random-walk baseline (0=any vector)")
    p.add_argument("--n_rw", type=int, default=20, help="random-walk repetitions per step_corr")
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
