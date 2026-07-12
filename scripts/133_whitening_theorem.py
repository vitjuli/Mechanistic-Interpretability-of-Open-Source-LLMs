"""
133_whitening_theorem.py   [why the optimal probe must turn away from use: validation]
========================================================================================
Validates the whitening attenuation theorem on a 119 field dump (CPU only, no GPU).

Theorem (attenuation identity, Appendix F). Let Sigma be the (shrunken) within-class
covariance actually used by the pipeline, w = Sigma^{-1} dmu / ||.||, u the unit mean
margin-gradient. If u is an eigenvector of Sigma with eigenvalue sigma_u^2, then

    cos(w, u) = cos(dmu, u) * lam_eff / sigma_u^2,
    lam_eff  := ||dmu|| / ||Sigma^{-1} dmu||   (order -2 weighted power mean of the
                spectrum, weighted by dmu's eigencomponents -> dominated by the SMALL
                eigenvalues dmu touches).

For general u the identity holds with additive error <= eps(u) = ||(Sigma - sigma_u^2 I) u|| / sigma_u^2.

Mechanism prediction: sigma_u^2 = u' Sigma u is LARGE because within-class margin
fluctuations live along u; under a locally linear readout,
    Var_within(m) ~= ||g_bar||^2 * sigma_u^2.

Counterfactual prediction: margin-stratified subsampling (keep prompts with |m - median_c|
small) deflates Var_within(m), hence sigma_u^2, hence rotates w TOWARD u by the amount
the identity predicts -- checked as a dose curve without any GPU.

Per layer this script reports:
  cos_wu_measured, cos_dmu_u, lam_eff, sigma_u2, attenuation = lam_eff/sigma_u2,
  cos_wu_predicted, eigen_residual eps, rayleigh_pctl (percentile of sigma_u2 in the
  spectrum), margin_identity_ratio = Var_within(m) / (||g_bar||^2 sigma_u2),
  and the subsampling dose curve q in {1.0, 0.8, 0.6, 0.4, 0.2}.

Outputs: whitening_per_layer.csv, whitening_counterfactual.csv,
         whitening_numbers.json, whitening_theorem.png

SELF-TEST (no data):  python 133_whitening_theorem.py --self_test

Typical run:
  python 133_whitening_theorem.py \
      --dump_dir data/analysis/runD_v2/B1_alpha_beta/field_dump \
      --concept alpha_beta --layers 16,20,21,22,23,24,28,32,35 \
      --out_dir data/analysis/runD_v2/B1_alpha_beta/whitening \
      --split_seed <SAME AS 122> --train_frac <SAME AS 122> --shrink <SAME AS 122>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("whitening133")


# ---- pipeline conventions (identical to 122/131/132) ----
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def shrunken_cov(H, y, shrink=0.1):
    """EXACTLY the covariance inside 122's fisher_axis (diagonal shrinkage + tiny ridge).
    NOTE: thesis section 2.5.2 describes Ledoit-Wolf identity shrinkage; the pipeline's
    fisher_axis uses diagonal shrinkage with fixed intensity. The theorem holds for any
    SPD Sigma; we validate against the Sigma that actually produced w_LDA."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return Sw, (mu1 - mu0)


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


# ---- theorem quantities ----
def layer_quantities(H, G, y, m, shrink, do_spectrum=True):
    Sw, dmu = shrunken_cov(H, y, shrink)
    winv_dmu = np.linalg.solve(Sw, dmu)
    w = unit_raw(winv_dmu)
    u = unit_raw(G.mean(0))
    cos_wu = float(w @ u)
    cos_du = float(unit_raw(dmu) @ u)
    lam_eff = float(np.linalg.norm(dmu) / np.linalg.norm(winv_dmu))
    Su = Sw @ u
    sig_u2 = float(u @ Su)
    eps = float(np.linalg.norm(Su - sig_u2 * u) / sig_u2)
    # exact identity error (not the loose CS bound): |<w_hat, r>| / sigma_u^2
    r = Su - sig_u2 * u
    eps_eff = float(abs(w @ r) / sig_u2)
    pred = cos_du * lam_eff / sig_u2
    out = dict(cos_wu_measured=cos_wu, cos_dmu_u=cos_du, lam_eff=lam_eff,
               sigma_u2=sig_u2, attenuation=lam_eff / sig_u2,
               cos_wu_predicted=float(pred), eigen_residual=eps,
               eigen_residual_effective=eps_eff)
    # margin-fluctuation mechanism: Var_within(m) vs ||g_bar||^2 sigma_u^2
    gbar = G.mean(0)
    var_m = float(np.mean([m[y == c].var() for c in (0, 1)]))
    out["margin_identity_ratio"] = var_m / (float(np.linalg.norm(gbar) ** 2) * sig_u2 + 1e-30)
    out["var_within_margin"] = var_m
    if do_spectrum:
        lam = np.linalg.eigvalsh(Sw)
        out["rayleigh_pctl"] = float(100.0 * (lam < sig_u2).mean())
        out["lam_median"] = float(np.median(lam)); out["lam_max"] = float(lam[-1])
    return out


def counterfactual(H, G, y, m, shrink, fams=None, qs=(1.0, 0.8, 0.6, 0.4, 0.2), n_null=50, seed=0):
    """Margin-stratified subsampling: keep the q-fraction of prompts of each class
    closest to the class median margin; recompute Sigma and w; compare measured vs
    predicted rotation of cos(w,u). u is held fixed (full-sample gradient mean).

    CONTROL: size-matched random-subsample null (n_null draws per q, default 50). At
    small q the LDA estimate is noise-dominated in high dimension and |cos(w,u)| of a
    noisy w drifts UP toward the anisotropic-null level; any rise must be read against
    random subsamples of the SAME size. Columns null_cos_median / null_cos_p95 report
    the PROMPT-level null. When family labels are supplied, null_cos_*_fam report a
    FAMILY-level null (whole families drawn to size-match), which respects the
    within-family correlation (prompts of one family are near-duplicates) and gives a
    more honest, wider null; n_fam is the count of distinct families in the keep set."""
    u = unit_raw(G.mean(0))
    rng = np.random.default_rng(seed)
    fam_arr = np.asarray(fams) if fams is not None else None
    rows = []
    for q in qs:
        keep = np.zeros(len(y), bool)
        for c in (0, 1):
            idx = np.where(y == c)[0]
            dist = np.abs(m[idx] - np.median(m[idx]))
            keep[idx[np.argsort(dist)[: max(int(round(q * len(idx))), 8)]]] = True
        Sw, dmu = shrunken_cov(H[keep], y[keep], shrink)
        winv = np.linalg.solve(Sw, dmu)
        sig_u2 = float(u @ Sw @ u)
        lam_eff = float(np.linalg.norm(dmu) / np.linalg.norm(winv))
        cos_meas = abs(float(unit_raw(winv) @ u))
        # prompt-level size-matched null
        null_cos = []
        for _ in range(n_null):
            rkeep = np.zeros(len(y), bool)
            for c in (0, 1):
                idx = np.where(y == c)[0]
                rkeep[rng.choice(idx, size=max(int(round(q * len(idx))), 8), replace=False)] = True
            Sn, dn = shrunken_cov(H[rkeep], y[rkeep], shrink)
            null_cos.append(abs(float(unit_raw(np.linalg.solve(Sn, dn)) @ u)))
        null_cos = np.array(null_cos)
        # family-level null (draw whole families to size-match) + family count of keep
        n_fam = int(len(set(fam_arr[keep]))) if fam_arr is not None else -1
        if fam_arr is not None:
            null_cos_fam = []
            for _ in range(n_null):
                rkeep = np.zeros(len(y), bool)
                for c in (0, 1):
                    idx = np.where(y == c)[0]
                    target = max(int(round(q * len(idx))), 8)
                    cfams = list(set(fam_arr[idx])); rng.shuffle(cfams)
                    cnt = 0
                    for fm in cfams:
                        sel = idx[fam_arr[idx] == fm]
                        rkeep[sel] = True; cnt += len(sel)
                        if cnt >= target:
                            break
                Sn, dn = shrunken_cov(H[rkeep], y[rkeep], shrink)
                null_cos_fam.append(abs(float(unit_raw(np.linalg.solve(Sn, dn)) @ u)))
            null_cos_fam = np.array(null_cos_fam)
        else:
            null_cos_fam = null_cos  # fallback = prompt-level when no families given
        rows.append(dict(q=q, n=int(keep.sum()), n_fam=n_fam,
                         var_within_margin=float(np.mean([m[keep & (y == c)].var() for c in (0, 1)])),
                         sigma_u2=sig_u2, lam_eff=lam_eff,
                         cos_wu_measured=float(unit_raw(winv) @ u),
                         cos_wu_predicted=float((unit_raw(dmu) @ u) * lam_eff / sig_u2),
                         cos_dmu_u=float(unit_raw(dmu) @ u),
                         null_cos_median=float(np.median(null_cos)),
                         null_cos_p95=float(np.percentile(null_cos, 95)),
                         excess_over_null=float(cos_meas - np.median(null_cos)),
                         null_cos_median_fam=float(np.median(null_cos_fam)),
                         null_cos_p95_fam=float(np.percentile(null_cos_fam, 95)),
                         excess_over_null_fam=float(cos_meas - np.median(null_cos_fam))))
    return pd.DataFrame(rows)


# ---- self-test ----
def self_test():
    rng = np.random.default_rng(0)
    n, d = 400, 40
    # planted model: within-class variance is LARGE along a known axis u_true
    u_true = unit_raw(rng.normal(0, 1, d))
    B = rng.normal(0, 1, (d, d)) * 0.2
    y = np.repeat([0, 1], n // 2)
    latm = rng.normal(0, 6.0, n)                       # within-class margin fluctuation
    H = (latm[:, None] * u_true[None, :]) + rng.normal(0, 1, (n, d)) @ B
    H[y == 1] += 2.0 * unit_raw(rng.normal(0, 1, d))   # class separation elsewhere
    G = np.tile(u_true * 2.0, (n, 1)) + rng.normal(0, 0.05, (n, d))
    m = H @ (u_true * 2.0)
    out = layer_quantities(H, G, y, m, shrink=0.1)
    # (1) identity accuracy within the eigen-residual bound
    err = abs(out["cos_wu_measured"] - out["cos_wu_predicted"])
    assert err <= out["eigen_residual"] + 1e-9, (err, out["eigen_residual"])
    # (2) mechanism: u is a top-variance direction, attenuation < 1, whitening suppresses
    assert out["rayleigh_pctl"] >= 95 and out["attenuation"] < 0.5
    assert abs(out["cos_wu_measured"]) < abs(out["cos_dmu_u"]), "whitening must suppress alignment"
    # (3) margin identity ~ 1 under a linear readout
    assert 0.5 < out["margin_identity_ratio"] < 2.0, out["margin_identity_ratio"]
    # (4) counterfactual: subsampling by margin deflates sigma_u2 and rotates w toward u
    cf = counterfactual(H, G, y, m, shrink=0.1, n_null=10)  # small n_null, synthetic (no families)
    assert cf.sigma_u2.iloc[-1] < 0.5 * cf.sigma_u2.iloc[0]
    assert abs(cf.cos_wu_measured.iloc[-1]) > abs(cf.cos_wu_measured.iloc[0])
    # predicted rotation agrees with measured at the endpoints of the dose curve
    d_meas = abs(cf.cos_wu_measured.iloc[-1]) - abs(cf.cos_wu_measured.iloc[0])
    d_pred = abs(cf.cos_wu_predicted.iloc[-1]) - abs(cf.cos_wu_predicted.iloc[0])
    assert d_meas > 0 and d_pred > 0, "both measured and predicted |cos| must rise as variance deflates"
    print("[self_test] OK - identity+bound, variance mechanism, margin identity, counterfactual rotation pass.")


# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--dump_dir"); ap.add_argument("--out_dir"); ap.add_argument("--concept", default="concept")
    ap.add_argument("--layers", default="16,20,21,22,23,24,28,32,35")
    ap.add_argument("--counterfactual_layers", default="22,24")
    ap.add_argument("--shrink", type=float, default=0.1, help="MUST match the 122 run")
    ap.add_argument("--split_seed", type=int, default=0, help="MUST match the 122 run")
    ap.add_argument("--train_frac", type=float, default=0.6, help="MUST match the 122 run")
    ap.add_argument("--n_null", type=int, default=50, help="random/family subsamples per q in the counterfactual null")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    assert args.dump_dir and args.out_dir
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    y_all = meta["y"].astype(int); m_all = meta["clean_margin"].astype(np.float64)
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)

    per, cfs = [], []
    for L in [int(x) for x in args.layers.split(",")]:
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        q = layer_quantities(H[trm], G[trm], y_all[trm], m_all[trm], args.shrink)
        q.update(layer=L, concept=args.concept)
        per.append(q)
        logger.info("L%02d cos(w,u): measured %+0.4f predicted %+0.4f (eps %.3f) | "
                    "atten %.3f  rayleigh pctl %.0f  margin-ratio %.2f",
                    L, q["cos_wu_measured"], q["cos_wu_predicted"], q["eigen_residual"],
                    q["attenuation"], q.get("rayleigh_pctl", -1), q["margin_identity_ratio"])
        if L in [int(x) for x in args.counterfactual_layers.split(",")]:
            cf = counterfactual(H[trm], G[trm], y_all[trm], m_all[trm], args.shrink,
                                fams=[f for f, t in zip(fams, trm) if t], n_null=args.n_null)
            cf["layer"] = L; cf["concept"] = args.concept
            cfs.append(cf)

    per = pd.DataFrame(per); per.to_csv(out / "whitening_per_layer.csv", index=False)
    CF = pd.concat(cfs, ignore_index=True) if cfs else pd.DataFrame()
    if len(CF):
        CF.to_csv(out / "whitening_counterfactual.csv", index=False)

    zone = per[per.layer.between(20, 26)]
    numbers = {
        "concept": args.concept,
        "attenuation_zone_median": float(zone.attenuation.median()) if len(zone) else None,
        "rayleigh_pctl_zone_median": float(zone.rayleigh_pctl.median()) if len(zone) else None,
        "eigen_residual_zone_median": float(zone.eigen_residual.median()) if len(zone) else None,
        "eigen_residual_effective_zone_median": float(zone.eigen_residual_effective.median()) if len(zone) else None,
        "margin_identity_ratio_zone_median": float(zone.margin_identity_ratio.median()) if len(zone) else None,
        "pred_vs_measured_cos_mae": float((per.cos_wu_measured - per.cos_wu_predicted).abs().mean()),
        "counterfactual": (
            {str(L): {"cos_start": float(g[g.q == 1.0].cos_wu_measured.iloc[0]),
                      "cos_end": float(g[g.q == g.q.min()].cos_wu_measured.iloc[0]),
                      "pred_end": float(g[g.q == g.q.min()].cos_wu_predicted.iloc[0]),
                      "sigma_deflation": float(g[g.q == g.q.min()].sigma_u2.iloc[0] /
                                               g[g.q == 1.0].sigma_u2.iloc[0]),
                      "null_cos_end_median": float(g[g.q == g.q.min()].null_cos_median.iloc[0]),
                      "null_cos_end_p95": float(g[g.q == g.q.min()].null_cos_p95.iloc[0]),
                      "excess_over_null_end": float(g[g.q == g.q.min()].excess_over_null.iloc[0]),
                      "n_fam_end": int(g[g.q == g.q.min()].n_fam.iloc[0]),
                      "null_cos_end_median_fam": float(g[g.q == g.q.min()].null_cos_median_fam.iloc[0]),
                      "null_cos_end_p95_fam": float(g[g.q == g.q.min()].null_cos_p95_fam.iloc[0]),
                      "excess_over_null_end_fam": float(g[g.q == g.q.min()].excess_over_null_fam.iloc[0])}
             for L, g in CF.groupby("layer")} if len(CF) else None),
    }
    json.dump(numbers, open(out / "whitening_numbers.json", "w"), indent=2)
    logger.info("whitening_numbers.json:\n%s", json.dumps(numbers, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.plot(per.layer, per.cos_dmu_u.abs(), "o-", color="#2ca02c", label="|cos(dmu, u)| before whitening")
    ax.plot(per.layer, per.cos_wu_measured.abs(), "o-", color="#d62728", label="|cos(w, u)| measured")
    ax.plot(per.layer, per.cos_wu_predicted.abs(), "s--", color="#1f77b4", label="|cos(w, u)| predicted (identity)")
    ax.fill_between(per.layer, np.maximum(per.cos_wu_predicted.abs() - per.eigen_residual, 0),
                    per.cos_wu_predicted.abs() + per.eigen_residual, color="#1f77b4", alpha=0.15,
                    label="eigen-residual bound")
    ax.set_xlabel("layer"); ax.set_ylabel("|cos|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Whitening attenuation: predicted vs measured", fontsize=10)
    ax = axes[1]
    if len(CF):
        for L, g in CF.groupby("layer"):
            ax.plot(g.var_within_margin, g.cos_wu_measured.abs(), "o-", label=f"L{L} measured")
            ax.plot(g.var_within_margin, g.cos_wu_predicted.abs(), "s--", alpha=0.6, label=f"L{L} predicted")
            ax.fill_between(g.var_within_margin, 0, g.null_cos_p95, alpha=0.12, color="gray",
                            label=f"L{L} size-matched null (95th pctl)")
            ax.plot(g.var_within_margin, g.null_cos_median, ":", color="gray", alpha=0.8)
        ax.set_xlabel("within-class margin variance (subsampled corpus)")
        ax.set_ylabel("|cos(w, u)|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("Counterfactual: deflating margin variance rotates the probe toward use", fontsize=10)
    fig.suptitle("Whitening theorem - %s" % args.concept, fontsize=12)
    fig.tight_layout(); fig.savefig(out / "whitening_theorem.png", dpi=150, bbox_inches="tight")
    logger.info("wrote %s", out / "whitening_theorem.png")


if __name__ == "__main__":
    main()
