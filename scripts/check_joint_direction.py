"""
Joint direction (Section 3.1.4): a direction that both decodes the concept (high AUC) and
aligns with the use direction u. Reproduces the claim that the reading/use separation is a
choice, not a necessity: such a direction exists, yet the actual w_LDA is orthogonal to u.

We build it in the span of w_LDA and u: v(t) = unit((1-t) w_LDA + t u), and report the
intermediate t=0.4 alongside the endpoints. Held-out AUC, cos(.,u), cos(.,gamma) at layer 24.

Input : data/analysis/runD_v2/{B1_alpha_beta,.}/field_dump/{res,grad}_L24.npy, meta.npz (gamma=wU_diff)
Output: data/analysis/27c/joint_direction.csv
"""
import os, csv, numpy as np
from sklearn.metrics import roc_auc_score

LAYER = 24
TJOINT = 0.4
CONCEPTS = [("alpha_beta_scaffold", "data/analysis/runD_v2/B1_alpha_beta/field_dump"),
            ("alpha_beta_probe", "data/analysis/runD_v2/field_dump")]


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


def auc(s, y):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def main():
    rows = []
    for name, dd in CONCEPTS:
        m = np.load(f"{dd}/meta.npz")
        y = m["y"].astype(int)
        gamma = unit(m["wU_diff"].astype(np.float64))
        n = len(y)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        H = np.load(f"{dd}/res_L{LAYER:02d}.npy").astype(np.float64)
        G = np.load(f"{dd}/grad_L{LAYER:02d}.npy").astype(np.float64)
        w = fisher(H[tr], y[tr])
        u = unit((G[tr][y[tr] == 0].mean(0) + G[tr][y[tr] == 1].mean(0)) / 2)
        j = unit((1 - TJOINT) * w + TJOINT * u)
        for lbl, v in [("w_LDA", w), ("joint", j), ("u", u)]:
            rows.append({"concept": name, "direction": lbl, "layer": LAYER,
                         "auc": round(auc(H[te] @ v, y[te]), 3),
                         "cos_u": round(abs(float(v @ u)), 3),
                         "cos_gamma": round(abs(float(v @ gamma)), 3)})
    os.makedirs("data/analysis/27c", exist_ok=True)
    out = "data/analysis/27c/joint_direction.csv"
    with open(out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["concept", "direction", "layer", "auc", "cos_u", "cos_gamma"])
        cw.writeheader()
        cw.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['concept']:20} {r['direction']:6} AUC={r['auc']} cos_u={r['cos_u']} cos_γ={r['cos_gamma']}")


if __name__ == "__main__":
    main()
