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
    print("[self_test] OK — rotation curve exact, planar PR~2, tau recovery, rate discrimination pass.")


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
    """per-layer Fisher axis on the masked rows; if shuffle_rng given, labels shuffled."""
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


def run_real(args):
    concepts = json.load(open(args.concepts))
    dks = args.dks
    rows, summ = [], []
    curves = {}
    for cdef in concepts:
        name = cdef["name"]
        dump, meta, fams, n_layers = load_dump(cdef["dump"])
        y = meta["y"].astype(int)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        held = ~trm
        axes = concept_axes(dump, n_layers, y, held, args.shrink)
        cur = rotation_curve(axes, dks)
        tau = fit_tau(dks, [cur[k] for k in dks])
        pr = participation_ratio_axes(axes)
        curves[name] = cur

        # shuffled-label null (average several shuffles)
        null_curves = []
        srng = np.random.default_rng(args.seed)
        for _ in range(args.n_shuffle):
            ax_s = concept_axes(dump, n_layers, y, held, args.shrink, shuffle_rng=srng)
            null_curves.append(rotation_curve(ax_s, dks))
        null_mean = {dk: float(np.mean([nc[dk] for nc in null_curves if not np.isnan(nc[dk])])) for dk in dks}
        null_p95 = {dk: float(np.quantile([nc[dk] for nc in null_curves if not np.isnan(nc[dk])], 0.95)) for dk in dks}

        for dk in dks:
            rows.append({"concept": name, "dk": dk,
                         "rho_real": cur[dk], "rho_null_mean": null_mean[dk],
                         "rho_null_p95": null_p95[dk],
                         "real_above_null": int(cur[dk] > null_p95[dk])})
        summ.append({"concept": name, "tau_layers": tau, "pr_axes": pr,
                     "n_layers_used": len(axes),
                     "rho_dk1": cur.get(1, float("nan")),
                     "rho_dk8": cur.get(8, float("nan"))})
        logger.info("[%s] rotation: rho(dk=1)=%.3f rho(dk=8)=%.3f | tau=%.1f layers | PR(axes)=%.1f",
                    name, cur.get(1, float("nan")), cur.get(8, float("nan")), tau, pr)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    sout = out.with_name("rotation_summary.csv")
    with open(sout, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader()
        [w.writerow(r) for r in summ]

    print("\n" + "=" * 92)
    print("ROTATION CONTROL — is the monotonic rotation law concept-specific or generic drift?")
    print("=" * 92)
    print(f"{'concept':<16}{'rho(dk1)':>9}{'rho(dk8)':>9}{'tau(layers)':>13}{'PR(axes)':>10}")
    for s in summ:
        print(f"{s['concept']:<16}{s['rho_dk1']:>9.3f}{s['rho_dk8']:>9.3f}"
              f"{s['tau_layers']:>13.1f}{s['pr_axes']:>10.1f}")
    print("\nReal rotation vs shuffled-label null (does the real axis stay MORE self-aligned?):")
    for cdef in concepts:
        name = cdef["name"]
        sel = [r for r in rows if r["concept"] == name]
        above = [r["dk"] for r in sel if r["real_above_null"]]
        print(f"  {name:<16} real > null p95 at dk = {above if above else 'NONE'}")
    if len(summ) >= 2:
        taus = [s["tau_layers"] for s in summ]
        rng_tau = max(taus) - min(taus)
        print(f"\nCROSS-CONCEPT: tau range across concepts = {rng_tau:.1f} layers")
        print("  - taus DIFFER (range large) -> rotation RATE is concept-specific structure,")
        print("    not generic drift: the rotation law carries information. STRONG for the chapter.")
        print("  - taus ~equal -> rotation is generic; the law describes drift common to any axis.")
    print("  - real curve clearly above shuffled null -> rotation is real structure, not LDA-on-noise.")
    print(f"per (concept,dk): {out} | summary: {sout}")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest [{name, dump}] (reuse concepts_manifest.json)")
    p.add_argument("--out", default="data/analysis/runD_v2/rotation_control.csv")
    p.add_argument("--dks", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12])
    p.add_argument("--n_shuffle", type=int, default=10)
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
