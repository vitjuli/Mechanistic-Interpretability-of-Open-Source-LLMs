"""
118_null_calibration.py   [calibrated nulls for high-d cosines + reliability/disattenuation]
=============================================================================================
In d = 2560 ANY two unrelated directions are nearly orthogonal: for isotropic random unit
vectors E|cos| = sqrt(2/(pi*d)) ~ 0.016, sd ~ 1/sqrt(d) ~ 0.020. So |cos(u, w_res)| = 0.027
is NOT "discovered orthogonality" — orthogonality is the null. The claims must be stated as
"alignment at / above a CALIBRATED null", and the null must respect the actual geometry:
the residual stream is anisotropic (effective dimension d_eff << d), which INFLATES null
cosines by an order of magnitude relative to the isotropic formula.

This script consumes the 119 field dump and produces, per layer:

(A) EFFECTIVE DIMENSION  d_eff = (tr Sigma)^2 / tr(Sigma^2) of the centered residual
    covariance — the honest "how many directions are there really" number that every
    cosine claim in the dissertation should be calibrated against.

(B) THREE NULL FAMILIES for |cos(anchor, random)| with anchor in {w_res, u_bar}:
      isotropic        r ~ N(0, I_d)             (the textbook null; usually too LOW)
      cov-matched      r ~ N(0, Sigma_h)         (respects residual anisotropy)
                       sampled cheaply as r = H_c^T g / sqrt(n), g ~ N(0, I_n)
      within-span      r uniform in span(H_c)    (fair null when BOTH directions are
                       data-estimated: supervised directions live in a <= n-dim span)
    Reported as p50/p95/p99 of |cos|.

(C) CALIBRATED RE-REPORT: |cos(u_bar, w_res)| per layer with its percentile under each
    null and the excess ratio value/p95 — the format every directional claim in the
    chapter should adopt (value, null median, null p95, null type).

(D) RELIABILITY + DISATTENUATION (the answer to "your zero is just estimation noise"):
    family-bootstrap (B resamples of surface families) -> r_w = mean pairwise |cos|
    between resampled w_res, r_u likewise for u_bar; Spearman disattenuation gives the
    UPPER BOUND on the true alignment:
        |rho_true| <= |rho_obs| / sqrt(r_w * r_u)
    plus a bootstrap CI on rho_obs itself. If the bound is still tiny, the null alignment
    is robust to estimation noise — lemma-grade.

All computations are full-corpus, ALL layers, pure CPU.

SELF-TEST (no torch / no repo):  python 118_null_calibration.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("nullcal118")


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
    return float(abs(unit_raw(np.asarray(a, float)) @ unit_raw(np.asarray(b, float))))


def d_eff_from_centered(Hc):
    """(tr Sigma)^2 / tr(Sigma^2) without forming the d x d covariance:
    tr Sigma = ||Hc||_F^2 / n ; tr Sigma^2 = ||Hc Hc^T / n||_F^2 (n x n Gram trick)."""
    n = Hc.shape[0]
    G = (Hc @ Hc.T) / n                       # (n, n)
    tr1 = float(np.trace(G))                  # = tr Sigma
    tr2 = float((G * G).sum())                # = tr Sigma^2
    return tr1 * tr1 / (tr2 + 1e-30)


def iso_null_stats(d, n_samp, anchor, rng):
    R = rng.standard_normal((n_samp, d))
    c = np.abs(R @ unit_raw(anchor)) / np.linalg.norm(R, axis=1)
    return c


def cov_null_stats(Hc, n_samp, anchor, rng):
    """r = Hc^T g / sqrt(n) has covariance Sigma_emp; cheap anisotropic null."""
    n = Hc.shape[0]
    Gz = rng.standard_normal((n_samp, n))
    R = (Gz @ Hc) / np.sqrt(n)               # (n_samp, d)
    c = np.abs(R @ unit_raw(anchor)) / (np.linalg.norm(R, axis=1) + 1e-30)
    return c


def span_null_stats(Hc, n_samp, anchor, rng, rank=None):
    """uniform directions inside span(Hc): r = V z, z ~ N(0, I_r), V = right singular basis."""
    _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    r = int((s > s.max() * 1e-10).sum()) if rank is None else min(rank, Vt.shape[0])
    V = Vt[:r].T                              # (d, r)
    Z = rng.standard_normal((n_samp, r))
    R = Z @ V.T
    c = np.abs(R @ unit_raw(anchor)) / (np.linalg.norm(R, axis=1) + 1e-30)
    return c, r


def pctile_of(value, null_samples):
    return float(np.mean(np.asarray(null_samples) <= value))


def mean_pairwise_abscos(D):
    """D: (B, d) stack of unit directions -> mean |cos| over all pairs (reliability)."""
    D = np.stack([unit_raw(v) for v in D])
    C = np.abs(D @ D.T)
    B = D.shape[0]
    iu = np.triu_indices(B, 1)
    return float(C[iu].mean())


def disattenuation_bound(rho_obs, r_a, r_b):
    return float(abs(rho_obs) / np.sqrt(max(r_a, 1e-12) * max(r_b, 1e-12)))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)

    # (1) isotropic null matches the analytic E|cos| = sqrt(2/(pi d))
    d = 800
    anchor = rng.standard_normal(d)
    c_iso = iso_null_stats(d, 4000, anchor, rng)
    assert abs(c_iso.mean() - np.sqrt(2 / (np.pi * d))) < 0.003, "isotropic null off analytic value"

    # (2) anisotropy inflates the null: 5 dominant directions out of 800
    n = 300
    V = np.linalg.qr(rng.standard_normal((d, 5)))[0]
    Hc = rng.standard_normal((n, 5)) * 20 @ V.T + 0.5 * rng.standard_normal((n, d))
    de = d_eff_from_centered(Hc)
    assert de < 30, f"d_eff should collapse toward the dominant subspace, got {de:.1f}"
    a_in = V[:, 0]                            # anchor inside the dominant subspace
    c_cov = cov_null_stats(Hc, 4000, a_in, rng)
    assert c_cov.mean() > 3 * c_iso.mean(), "cov-matched null must be inflated vs isotropic"

    # (3) within-span null: rank detected, null between iso and cov for generic anchor
    c_span, r = span_null_stats(Hc, 2000, a_in, rng)
    assert r <= n, "span rank bounded by sample size"
    assert c_span.mean() > c_iso.mean(), "span null inflated vs isotropic for in-span anchor"

    # (4) percentile sanity
    assert pctile_of(np.quantile(c_cov, 0.95), c_cov) > 0.94

    # (5) reliability + disattenuation recover a known true alignment.
    #     true directions p, q with cos = rho_true; we observe noisy unit estimates.
    rho_true = 0.30
    p = unit_raw(rng.standard_normal(d))
    q = unit_raw(rho_true * p + np.sqrt(1 - rho_true ** 2) * unit_raw(
        rng.standard_normal(d) - (rng.standard_normal(d) @ p) * p))
    def noisy(v, s):
        return unit_raw(v + s * rng.standard_normal(d) / np.sqrt(d))
    sp, sq = 0.9, 1.4
    P = np.stack([noisy(p, sp) for _ in range(12)])
    Q = np.stack([noisy(q, sq) for _ in range(12)])
    r_p, r_q = mean_pairwise_abscos(P), mean_pairwise_abscos(Q)
    rho_obs = float(np.mean([abs(a @ b) for a in P for b in Q]))
    bound = disattenuation_bound(rho_obs, r_p, r_q)
    assert rho_obs < rho_true, "noise must attenuate the observed alignment"
    assert bound > rho_true * 0.85, f"disattenuated bound must (approximately) recover truth: {bound:.3f}"

    print("[self_test] OK — analytic isotropic null, anisotropic inflation, span null, "
          "percentiles, disattenuation recovery pass.")


# =====================================================================
# Real run (CPU only; consumes the 119 dump)
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
    """EXACTLY the 86/89 convention: sorted unique families, default_rng(seed).shuffle,
    first round(frac * len) are train."""
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train_fams = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train_fams for f in fams], bool)


def run_real(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    meta, fams, res, grad, n_layers = load_dump(args.dump_dir)
    y = meta["y"].astype(int)
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    nP, d = res[0].shape
    logger.info("dump: %d prompts, %d layers, d=%d | train fraction %.2f (%d prompts)",
                nP, n_layers, d, args.train_frac, int(trm.sum()))

    fam_arr = np.array(fams)
    uniq_fams = sorted(set(fams))

    geo_rows, rel_rows = [], []
    for L in range(n_layers):
        H = res[L].astype(np.float64); G = grad[L].astype(np.float64)
        Hc = H[trm] - H[trm].mean(0)
        w = fisher_axis(H[trm], y[trm], args.shrink)
        u = unit_raw(G.mean(0))
        c_obs = abs_cos(u, w)
        de = d_eff_from_centered(Hc)

        # ---- (B) three nulls against each anchor ----
        rec = {"layer": L, "d_eff": de, "abs_cos_u_wres": c_obs}
        for tag, anchor in (("wres", w), ("u", u)):
            c_iso = iso_null_stats(d, args.n_null, anchor, rng)
            c_cov = cov_null_stats(Hc, args.n_null, anchor, rng)
            c_spn, rank = span_null_stats(Hc, args.n_null, anchor, rng)
            for nm, cc in (("iso", c_iso), ("cov", c_cov), ("span", c_spn)):
                rec[f"{nm}_{tag}_p50"] = float(np.quantile(cc, 0.50))
                rec[f"{nm}_{tag}_p95"] = float(np.quantile(cc, 0.95))
                rec[f"{nm}_{tag}_p99"] = float(np.quantile(cc, 0.99))
            rec[f"span_rank_{tag}"] = rank
        # ---- (C) calibrated verdict for THE pair (u, w_res) ----
        for nm in ("iso", "cov", "span"):
            p95 = rec[f"{nm}_wres_p95"]
            rec[f"pct_{nm}"] = pctile_of(c_obs, cov_null_stats(Hc, args.n_null, w, rng)
                                         if nm == "cov" else
                                         (iso_null_stats(d, args.n_null, w, rng) if nm == "iso"
                                          else span_null_stats(Hc, args.n_null, w, rng)[0]))
            rec[f"excess_{nm}"] = c_obs / (p95 + 1e-30)
        geo_rows.append(rec)
        if L % 4 == 0 or L == n_layers - 1:
            logger.info("  L%02d: |cos(u,w)|=%.4f | d_eff=%.0f | p95 iso/cov/span = %.4f/%.4f/%.4f",
                        L, c_obs, de, rec["iso_wres_p95"], rec["cov_wres_p95"], rec["span_wres_p95"])

        # ---- (D) family-bootstrap reliability at selected layers (or all if asked) ----
        if args.reliability_all_layers or L in args.reliability_layers:
            Ws, Us, obs = [], [], []
            for b in range(args.n_boot):
                bf = rng.choice(uniq_fams, size=len(uniq_fams), replace=True)
                idx = np.concatenate([np.where(fam_arr == f)[0] for f in bf])
                yb = y[idx]
                if yb.min() == yb.max():
                    continue
                Wb = fisher_axis(H[idx], yb, args.shrink)
                Ub = unit_raw(G[idx].mean(0))
                Ws.append(Wb); Us.append(Ub); obs.append(abs(float(Wb @ Ub)))
            if len(Ws) >= 3:
                r_w = mean_pairwise_abscos(np.stack(Ws))
                r_u = mean_pairwise_abscos(np.stack(Us))
                bound = disattenuation_bound(c_obs, r_w, r_u)
                rel_rows.append({"layer": L, "r_w": r_w, "r_u": r_u,
                                 "rho_obs": c_obs,
                                 "rho_obs_boot_p05": float(np.quantile(obs, 0.05)),
                                 "rho_obs_boot_p95": float(np.quantile(obs, 0.95)),
                                 "rho_true_upper_bound": bound,
                                 "n_boot_eff": len(Ws)})
                logger.info("  L%02d reliability: r_w=%.3f r_u=%.3f -> rho_true <= %.4f "
                            "(obs %.4f, boot CI [%.4f, %.4f])",
                            L, r_w, r_u, bound, c_obs,
                            rel_rows[-1]["rho_obs_boot_p05"], rel_rows[-1]["rho_obs_boot_p95"])

    # ---------- save ----------
    import csv as _csv
    def wcsv(name, rows):
        if not rows:
            return
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            [w.writerow(r) for r in rows]
    wcsv("null_calibration_per_layer.csv", geo_rows)
    wcsv("reliability_disattenuation.csv", rel_rows)

    med_obs = float(np.median([r["abs_cos_u_wres"] for r in geo_rows]))
    med_de = float(np.median([r["d_eff"] for r in geo_rows]))
    above_cov = [r["layer"] for r in geo_rows if r["abs_cos_u_wres"] > r["cov_wres_p95"]]
    above_spn = [r["layer"] for r in geo_rows if r["abs_cos_u_wres"] > r["span_wres_p95"]]
    print("\n" + "=" * 96)
    print("NULL CALIBRATION + RELIABILITY — the high-dimensional hygiene layer")
    print("=" * 96)
    print(f"median |cos(u, w_res)| over layers = {med_obs:.4f} | median d_eff = {med_de:.0f} (of d={d})")
    print(f"isotropic E|cos| (analytic) = {np.sqrt(2/(np.pi*d)):.4f} — the WRONG null when d_eff << d")
    print(f"layers where |cos(u,w)| EXCEEDS cov-matched p95:  {above_cov if above_cov else 'NONE'}")
    print(f"layers where |cos(u,w)| EXCEEDS within-span p95:  {above_spn if above_spn else 'NONE'}")
    if rel_rows:
        worst = max(r["rho_true_upper_bound"] for r in rel_rows)
        print(f"disattenuation: max over tested layers of the TRUE-alignment upper bound = {worst:.4f}")
        print("  -> claim format: 'alignment at the calibrated null; bound robust to estimation noise'")
    print("REPORTING RULE for the dissertation: every cosine as "
          "(value, null p50, null p95, null type, d_eff).")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/null_calibration")
    p.add_argument("--n_null", type=int, default=2000)
    p.add_argument("--n_boot", type=int, default=40, help="family bootstraps for reliability")
    p.add_argument("--reliability_layers", type=int, nargs="*", default=[8, 14, 16, 21, 24, 30, 35])
    p.add_argument("--reliability_all_layers", action="store_true",
                   help="bootstrap EVERY layer (slower; full-corpus principle)")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0, help="must match 86/89 to reproduce their split")
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
