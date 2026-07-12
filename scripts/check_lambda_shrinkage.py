"""
Shrinkage justification for w_LDA (Sections 2.5.2 / 2.7 appendix):
  (1) Ledoit-Wolf data-driven optimal lambda per layer (within-class covariance).
  (2) cos(w_LDA, u) robustness to lambda in {0.04, 0.1, 0.2}.
  (3) per-direction projection std (sigma_delta / sigma_u / sigma_wLDA) and the neutral
      random-direction scale sigma_typ, to justify the shared LDA reference in steering.

Input : data/analysis/runD_v2/field_dump/{res_L*.npy, grad_L*.npy, meta.npz}
Output: data/analysis/27c/lambda_shrinkage.csv
"""
import os, csv, numpy as np
from sklearn.covariance import ledoit_wolf

DD = "data/analysis/runD_v2/field_dump"


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(H, y, lam):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    d = mu1 - mu0
    H0, H1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (H0.T @ H0 + H1.T @ H1) / max(1, len(H) - 2)
    Sw = (1 - lam) * Sw + lam * (np.trace(Sw) / Sw.shape[0]) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, d))


def main():
    y = np.load(f"{DD}/meta.npz")["y"].astype(int)
    rows = []
    for L in range(36):
        rf, gf = f"{DD}/res_L{L:02d}.npy", f"{DD}/grad_L{L:02d}.npy"
        if not (os.path.exists(rf) and os.path.exists(gf)):
            continue
        H = np.load(rf).astype(np.float64)
        G = np.load(gf).astype(np.float64)
        # (1) Ledoit-Wolf optimal shrinkage on within-class residuals
        Hc = H.copy()
        for c in (0, 1):
            Hc[y == c] -= H[y == c].mean(0)
        _, lw_lambda = ledoit_wolf(Hc, assume_centered=True)
        # directions
        dh = unit(H[y == 1].mean(0) - H[y == 0].mean(0))
        uh = unit((G[y == 0].mean(0) + G[y == 1].mean(0)) / 2)
        # (2) cos(w_LDA, u) at three lambdas
        cos = {lam: abs(float(fisher(H, y, lam) @ uh)) for lam in (0.04, 0.1, 0.2)}
        # (3) projection stds + neutral scale
        sig_d, sig_u = (H @ dh).std(), (H @ uh).std()
        sig_w = (H @ fisher(H, y, 0.1)).std()
        sig_typ = float(np.sqrt(H.var(axis=0).mean()))
        rows.append({"layer": L, "lw_optimal_lambda": round(float(lw_lambda), 4),
                     "cos_wu_l04": round(cos[0.04], 4), "cos_wu_l10": round(cos[0.1], 4),
                     "cos_wu_l20": round(cos[0.2], 4),
                     "sigma_delta": round(sig_d, 3), "sigma_u": round(sig_u, 3),
                     "sigma_wlda": round(sig_w, 3), "sigma_typ": round(sig_typ, 3)})
    os.makedirs("data/analysis/27c", exist_ok=True)
    out = "data/analysis/27c/lambda_shrinkage.csv"
    with open(out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        cw.writerows(rows)
    zone = [r for r in rows if 20 <= r["layer"] <= 30]
    m = lambda k: np.mean([r[k] for r in zone])
    print(f"wrote {out} ({len(rows)} layers)")
    print(f"zone L20-30: LW-optimal λ={m('lw_optimal_lambda'):.3f} (we use 0.1)")
    print(f"  cos(w,u) robustness: λ=0.04 {m('cos_wu_l04'):.3f} | 0.1 {m('cos_wu_l10'):.3f} | 0.2 {m('cos_wu_l20'):.3f}")
    print(f"  σ_δ={m('sigma_delta'):.2f}  σ_u={m('sigma_u'):.2f}  σ_wLDA={m('sigma_wlda'):.2f}  σ_typ={m('sigma_typ'):.2f}")


if __name__ == "__main__":
    main()
