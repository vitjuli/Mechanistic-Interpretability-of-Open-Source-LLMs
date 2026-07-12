"""
Held-out decodability (AUC) per layer for the three direction axes (w_LDA / Delta_mu / u).
Reproduces the Section 3.1.1 table. Pure numpy + sklearn, runs on the local field_dump.

Axes (per layer, fit on TRAIN split, AUC measured on TEST split):
  w_LDA = Fisher with shrinkage (eq 7)        Delta_mu = mu_beta - mu_alpha (eq 4)
  u     = class-symmetric mean grad (eq 9)

Input : data/analysis/runD_v2/field_dump/{res_L*.npy, grad_L*.npy, meta.npz}
Output: data/analysis/27c/decodability_by_layer.csv  (layer, auc_wlda, auc_delta, auc_u)
"""
import os, csv, numpy as np
from sklearn.metrics import roc_auc_score

DD = "data/analysis/runD_v2/field_dump"
LAM = 0.1
SEED = 0


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(H, y, lam=LAM):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    d = mu1 - mu0
    H0, H1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (H0.T @ H0 + H1.T @ H1) / max(1, len(H) - 2)
    Sw = (1 - lam) * Sw + lam * (np.trace(Sw) / Sw.shape[0]) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, d))


def auc(scores, y):
    a = roc_auc_score(y, scores)
    return max(a, 1 - a)


def main():
    y = np.load(f"{DD}/meta.npz")["y"].astype(int)
    n = len(y)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    rows = []
    for L in range(36):
        rf, gf = f"{DD}/res_L{L:02d}.npy", f"{DD}/grad_L{L:02d}.npy"
        if not os.path.exists(rf):
            continue
        H = np.load(rf).astype(np.float64)
        Htr, Hte, ytr, yte = H[tr], H[te], y[tr], y[te]
        dh = unit(Htr[ytr == 1].mean(0) - Htr[ytr == 0].mean(0))
        wh = fisher(Htr, ytr)
        row = {"layer": L, "auc_wlda": round(auc(Hte @ wh, yte), 4),
               "auc_delta": round(auc(Hte @ dh, yte), 4)}
        if os.path.exists(gf):
            G = np.load(gf).astype(np.float64)
            uh = unit((G[tr][ytr == 0].mean(0) + G[tr][ytr == 1].mean(0)) / 2)
            row["auc_u"] = round(auc(Hte @ uh, yte), 4)
        else:
            row["auc_u"] = ""
        rows.append(row)
    os.makedirs("data/analysis/27c", exist_ok=True)
    out = "data/analysis/27c/decodability_by_layer.csv"
    with open(out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["layer", "auc_wlda", "auc_delta", "auc_u"])
        cw.writeheader()
        cw.writerows(rows)
    print(f"wrote {out} ({len(rows)} layers)")
    for r in rows:
        if r["layer"] % 4 == 0 or r["layer"] in (18, 22, 35):
            print(f"  L{r['layer']:02d}  w_LDA={r['auc_wlda']}  Δμ={r['auc_delta']}  u={r['auc_u']}")


if __name__ == "__main__":
    main()
