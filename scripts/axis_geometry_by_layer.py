"""
Per-layer axis geometry + causal effect (Section 3.1.3), alpha/beta scaffold.
Columns: cos(w_LDA,u), cos(Delta_mu,u), cos(u,gamma), cos(w_LDA,gamma), cos(Delta_mu,gamma),
         and per-layer margin flip for w_LDA / Delta_mu / u at c=16.

Shows that the causal effect tracks alignment: w stays orthogonal to u and gamma and is inert,
the use direction converges to gamma and is causal earliest, the writing direction aligns with
use around L21-22 and becomes causal there.

Inputs : data/analysis/runD_v2/B1_alpha_beta/field_dump/{res,grad}_L*.npy, meta.npz (gamma = wU_diff)
         data/analysis/runD_v2/B1_alpha_beta/steering_delta/steering_sweep_tier1.csv (margin_flip, c=16)
Output : data/analysis/27c/axis_geometry_by_layer.csv
"""
import os, csv, numpy as np, pandas as pd

DD = "data/analysis/runD_v2/B1_alpha_beta/field_dump"
ST = "data/analysis/runD_v2/B1_alpha_beta/steering_delta/steering_sweep_tier1.csv"


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(H, y, lam=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    d = mu1 - mu0
    H0, H1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (H0.T @ H0 + H1.T @ H1) / max(1, len(H) - 2)
    Sw = (1 - lam) * Sw + lam * (np.trace(Sw) / Sw.shape[0]) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, d))


def main():
    m = np.load(f"{DD}/meta.npz")
    y = m["y"].astype(int)
    gamma = unit(m["wU_diff"].astype(np.float64))
    st = pd.read_csv(ST)
    st = st[st.c == 16]

    def margin(L, d):
        s = st[(st.layer == L) & (st.dir == d)]
        return round(float(s.margin_flip.iloc[0]), 3) if len(s) else ""

    rows = []
    for L in range(36):
        rf, gf = f"{DD}/res_L{L:02d}.npy", f"{DD}/grad_L{L:02d}.npy"
        if not (os.path.exists(rf) and os.path.exists(gf)):
            continue
        H = np.load(rf).astype(np.float64)
        G = np.load(gf).astype(np.float64)
        d = unit(H[y == 1].mean(0) - H[y == 0].mean(0))
        u = unit((G[y == 0].mean(0) + G[y == 1].mean(0)) / 2)
        w = fisher(H, y)
        rows.append({"layer": L,
                     "cos_w_u": round(abs(float(w @ u)), 3), "cos_delta_u": round(abs(float(d @ u)), 3),
                     "cos_u_gamma": round(abs(float(u @ gamma)), 3), "cos_w_gamma": round(abs(float(w @ gamma)), 3),
                     "cos_delta_gamma": round(abs(float(d @ gamma)), 3),
                     "margin_w": margin(L, "w_res"), "margin_delta": margin(L, "delta"), "margin_u": margin(L, "usage")})
    os.makedirs("data/analysis/27c", exist_ok=True)
    out = "data/analysis/27c/axis_geometry_by_layer.csv"
    with open(out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        cw.writerows(rows)
    print(f"wrote {out} ({len(rows)} layers)")
    for r in rows:
        if r["layer"] in (16, 20, 22, 24, 28, 32, 35):
            print(f"  L{r['layer']:02d} w·u={r['cos_w_u']} Δμ·u={r['cos_delta_u']} u·γ={r['cos_u_gamma']} | "
                  f"w_m={r['margin_w']} Δμ_m={r['margin_delta']} u_m={r['margin_u']}")


if __name__ == "__main__":
    main()
