"""
70_wres_subspace.py   [LAPTOP / CPU-ONLY -- seconds, from synced arrays]
===================================================================
Sharpens the "rotating axis" finding from 69. Uses ONLY the 13 w_res_*.npy
(from 64) + Sigma_inv/gbar/lbar (concept_directions.npz from 60). No model.

Three questions 69 left open:

  (Q1) Does the concept axis rotate WITHIN A SMALL SUBSPACE, or all over the
       place? -> effective dimensionality (participation ratio) of the 13-axis
       stack, in the causal metric, vs a random-direction null.
         PR ~ 2-3  => "the concept occupies a low-dim residual subspace; the
                      optimal 1-D readout rotates within it" (clean, publishable).
         PR ~ k    => the rotation fills many dimensions (messier).

  (Q2) Is gbar orthogonal to the WHOLE concept subspace, or only to each
       individual axis? -> projection capture of gbar (and lbar) onto
       span{w_res_L}. Capture ~ 0 with per-axis cos ~ 0 => gbar is orthogonal to
       the entire subspace (the strongest "wrong axis" statement). Capture larger
       than per-axis cos => gbar has small components spread across the subspace.

  (Q3) Is the rotation genuine or Fisher estimation noise? -> mean cos_C as a
       function of layer-distance |Li - Lj|. MONOTONE decay => systematic
       rotation (noise would be distance-independent / flat).

OUTPUT (default ./wres_subspace_out/):
  wres_subspace.json   PR, null PR, gbar/lbar capture, distance-decay curve, verdicts

SELF-TEST: python 70_wres_subspace.py --self_test
"""

from __future__ import annotations
import argparse, glob, json, logging, re, sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wres_subspace")
TAP_RE = re.compile(r"w_res(?:65)?_(postL(\d+)|final)\.npy$")


def whitener(Si):
    w, V = np.linalg.eigh(Si); w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def participation_ratio(M: np.ndarray) -> float:
    """PR of a symmetric PSD matrix: (sum λ)^2 / sum λ^2, in [1, k]."""
    w = np.clip(np.linalg.eigvalsh(M), 0, None)
    s = float(w.sum())
    return float(s * s / np.sum(w ** 2)) if s > 0 else float("nan")


def capture_cos(axis, basis, Si, A=None) -> float:
    """cos angle between `axis` and span(basis rows) in the causal metric, [0,1]."""
    if A is None:
        A = whitener(Si)
    Bw = A @ basis.T
    aw = A @ axis
    if Bw.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(Bw)
    proj = Q @ (Q.T @ aw)
    den = np.linalg.norm(aw)
    return float(np.linalg.norm(proj) / den) if den > 0 else 0.0


def whitened_unit(W, Si, A=None):
    if A is None:
        A = whitener(Si)
    Ww = W @ A.T
    return Ww / np.clip(np.linalg.norm(Ww, axis=1, keepdims=True), 1e-30, None)


def load_axes(folder: Path) -> Tuple[List[str], List[int], np.ndarray]:
    found = []
    for p in folder.glob("w_res*_*.npy"):
        m = TAP_RE.search(p.name)
        if not m:
            continue
        if m.group(1) == "final":
            order, label = 10_000, "final"
        else:
            order, label = int(m.group(2)), f"L{int(m.group(2))}"
        found.append((order, label, p))
    found.sort(key=lambda t: t[0])
    if not found:
        raise SystemExit(f"no w_res_*.npy in {folder}")
    labels = [l for _, l, _ in found]
    orders = [o for o, _, _ in found]
    W = np.array([np.load(p).astype(np.float64) for _, _, p in found])
    W = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-30, None)
    return labels, orders, W


# =====================================================================
# Self-test
# =====================================================================

def self_test():
    rng = np.random.default_rng(70); d = 220
    ev = np.concatenate([np.linspace(12, 3, 12), np.linspace(2, .05, d - 12)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Si = np.linalg.inv((Q * ev) @ Q.T + 1e-3 * np.eye(d))
    A = whitener(Si); Ai = np.linalg.inv(A)
    k = 13

    w0 = rng.standard_normal(d); w0 /= np.linalg.norm(w0)
    w1 = rng.standard_normal(d); w1 -= w0 * (w0 @ w1); w1 /= np.linalg.norm(w1)

    W_stable = np.array([w0 + 0.01 * rng.standard_normal(d) for _ in range(k)])
    th = np.linspace(0, np.pi / 2, k)
    W_rot2d = np.array([np.cos(t) * w0 + np.sin(t) * w1 for t in th])   # spans exactly 2D
    W_rand = rng.standard_normal((k, d))

    def PR(W):
        U = whitened_unit(W, Si); return participation_ratio(U @ U.T)

    print("\n--- SELF TEST -------------------------------------------------")
    for nm, W in [("stable(1D)", W_stable), ("rotating(2D)", W_rot2d), ("random", W_rand)]:
        print(f"  {nm:13s}: PR = {PR(W):.2f}  (of k={k})")

    # gbar in the 2D plane vs orthogonal to it
    g_in = (0.6 * w0 + 0.8 * w1); g_in_raw = Ai @ (A @ g_in)
    g_out = rng.standard_normal(d); g_out -= w0 * (w0 @ g_out); g_out -= w1 * (w1 @ g_out)
    cap_in = capture_cos(g_in, W_rot2d, Si)
    cap_out = capture_cos(g_out, W_rot2d, Si)
    print(f"  gbar IN-plane capture  = {cap_in:.2f}  (expect high)")
    print(f"  gbar OUT-plane capture = {cap_out:.2f}  (expect low)")

    pr_s, pr_r2, pr_rand = PR(W_stable), PR(W_rot2d), PR(W_rand)
    assert pr_s < 1.5, f"stable must be ~1D (PR={pr_s:.2f})"
    assert 1.5 < pr_r2 < 4.0, f"rotating-2D must be ~2-3 (PR={pr_r2:.2f})"
    assert pr_rand > 7.0, f"random must fill many dims (PR={pr_rand:.2f})"
    assert cap_in > 0.8 and cap_out < 0.4, "gbar capture must separate in-plane vs orthogonal"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(PR distinguishes 1D / low-dim-rotation / spread; capture locates gbar vs subspace)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def run_real(args):
    folder = Path(args.geom_dir)
    labels, orders, W = load_axes(folder)
    k, d = W.shape
    logger.info("loaded %d axes: %s", k, labels)

    cd = np.load(args.concept_npz)
    Si = cd["Sigma_inv"].astype(np.float64)
    gbar = cd["gbar"].astype(np.float64) if "gbar" in cd else None
    lbar = cd["lbar"].astype(np.float64) if "lbar" in cd else None

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Compute whitener ONCE (was being recomputed inside every capture_cos/whitened_unit
    # call, which means n_null*2 eigh(2560x2560) calls — minutes-to-hours).
    logger.info("computing whitener A = Sigma_inv^{1/2} (one-time eigh d=%d)...", Si.shape[0])
    A_white = whitener(Si)
    logger.info("  whitener ready.")

    # (Q1) effective dimensionality of the axis stack
    U = whitened_unit(W, Si, A=A_white)
    M = U @ U.T
    PR = participation_ratio(M)
    # eigen-spectrum (how the concept variance distributes across orthogonal dirs)
    eig = np.sort(np.clip(np.linalg.eigvalsh(M), 0, None))[::-1]
    eig_frac = (eig / eig.sum()).tolist()
    rank90 = int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.90) + 1)
    logger.info("Q1: PR=%.2f (frac of k=%d: %.3f), rank90=%d", PR, k, PR/k, rank90)

    # null PR: k random unit directions
    rng = np.random.default_rng(args.seed)
    logger.info("Q1 null: %d random %d-axis stacks...", args.n_null, k)
    null_PR = []
    for i in range(args.n_null):
        if i > 0 and i % 50 == 0:
            logger.info("  null PR %d/%d", i, args.n_null)
        R = rng.standard_normal((k, d))
        Ur = whitened_unit(R, Si, A=A_white)
        null_PR.append(participation_ratio(Ur @ Ur.T))
    null_PR = np.array(null_PR)

    # (Q2) gbar / lbar projection onto the concept subspace
    gbar_cap = capture_cos(gbar, W, Si, A=A_white) if gbar is not None else None
    lbar_cap = capture_cos(lbar, W, Si, A=A_white) if lbar is not None else None
    logger.info("Q2: gbar_capture=%.4f  lbar_capture=%s",
                gbar_cap if gbar_cap else float("nan"),
                f"{lbar_cap:.4f}" if lbar_cap else "n/a")
    # null capture: random direction onto the same subspace
    logger.info("Q2 null: %d random axes onto subspace...", args.n_null)
    null_cap = []
    for i in range(args.n_null):
        if i > 0 and i % 50 == 0:
            logger.info("  null cap %d/%d", i, args.n_null)
        r = rng.standard_normal(d)
        null_cap.append(capture_cos(r, W, Si, A=A_white))
    null_cap = np.array(null_cap)

    # (Q3) distance-decay of cos_C
    dist_curve = {}
    maxd = max(orders[i] for i in range(k) if orders[i] < 9999) - min(orders) if k > 1 else 0
    # use index distance for 'final' robustness
    by_idx = {}
    for delta in range(1, k):
        vals = [M[i, i + delta] for i in range(k - delta)]
        by_idx[delta] = float(np.mean(vals))
    monotone = all(by_idx[i] >= by_idx[i + 1] - args.mono_tol for i in range(1, k - 1)) if k > 2 else True

    # Null-calibrated verdict (the right comparison: PR vs null, not PR vs absolute cutoff).
    # Three regimes:
    #   PR ≤ 0.5 × null_PR  → CONCENTRATED (significantly compressed vs random)
    #   PR ≥ 0.9 × null_PR  → INDISTINGUISHABLE FROM RANDOM (no structure)
    #   else                → STRUCTURED MID-DIM (between compressed and random)
    pr_ratio_null = PR / float(null_PR.mean())
    pr_below_null_p05 = PR < float(np.percentile(null_PR, 5))
    if pr_ratio_null <= 0.5:
        if PR < 2.5:
            verdict_q1 = (f"LOW-DIM concept subspace (PR={PR:.2f}≈plane; "
                          f"{pr_ratio_null:.2f}× null mean; rotation within few dims)")
        else:
            verdict_q1 = (f"STRUCTURED MID-DIM (PR={PR:.2f}, ~{rank90}D for 90% mass; "
                          f"{pr_ratio_null:.2f}× null mean — significantly compressed vs random "
                          f"but not a single plane)")
    elif pr_ratio_null < 0.9:
        verdict_q1 = (f"STRUCTURED MID-DIM (PR={PR:.2f}; {pr_ratio_null:.2f}× null mean — "
                      f"compressed vs random but moderately spread)")
    else:
        verdict_q1 = (f"RANDOM-LIKE (PR={PR:.2f}; {pr_ratio_null:.2f}× null — "
                      f"indistinguishable from independent random directions)")

    res = {
        "geom_dir": str(folder), "n_axes": k, "labels": labels,
        "Q1_effective_dim": {
            "participation_ratio": PR, "PR_frac_of_k": PR / k, "rank90": rank90,
            "eigfrac_top5": eig_frac[:5],
            "null_PR": {"mean": float(null_PR.mean()), "p05": float(np.percentile(null_PR, 5))},
            "PR_over_null_mean": float(pr_ratio_null),
            "PR_below_null_p05": bool(pr_below_null_p05),
            "verdict": verdict_q1,
        },
        "Q2_gbar_vs_subspace": {
            "gbar_capture": gbar_cap, "lbar_capture": lbar_cap,
            "null_capture": {"mean": float(null_cap.mean()), "p95": float(np.percentile(null_cap, 95))},
            "verdict": (None if gbar_cap is None else
                        "gbar ORTHOGONAL to the whole concept subspace (strongest wrong-axis result)"
                        if gbar_cap <= np.percentile(null_cap, 95) else
                        "gbar lies PARTLY in the subspace (small spread components, off every axis)"),
        },
        "Q3_rotation_vs_noise": {
            "cos_by_index_distance": by_idx,
            "monotone_decay": bool(monotone),
            "verdict": ("MONOTONE decay -> genuine systematic rotation (not estimation noise)"
                        if monotone else "non-monotone -> rotation claim weaker; could be noise"),
        },
    }
    with open(out / "wres_subspace.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print("w_res SUBSPACE  (laptop/CPU)  --  how the concept rotates")
    print("=" * 78)
    q1 = res["Q1_effective_dim"]
    print(f"\n  (Q1) effective dim of the {k}-axis stack:")
    print(f"       participation ratio = {q1['participation_ratio']:.2f}  ({q1['PR_frac_of_k']:.2f} of k)  "
          f"rank90 = {q1['rank90']}/{k}")
    print(f"       random-null PR = {q1['null_PR']['mean']:.2f} (p05 {q1['null_PR']['p05']:.2f})")
    print(f"       top eigen-fractions = {[round(x,3) for x in q1['eigfrac_top5']]}")
    print(f"       => {q1['verdict']}")
    q2 = res["Q2_gbar_vs_subspace"]
    if q2["gbar_capture"] is not None:
        print(f"\n  (Q2) gbar capture by concept subspace = {q2['gbar_capture']:.3f}  "
              f"(null p95 {q2['null_capture']['p95']:.3f})")
        if q2["lbar_capture"] is not None:
            print(f"       lbar capture = {q2['lbar_capture']:.3f}")
        print(f"       => {q2['verdict']}")
    q3 = res["Q3_rotation_vs_noise"]
    print(f"\n  (Q3) cos_C by layer-index distance:")
    print("       " + "  ".join(f"Δ{dd}:{v:+.2f}" for dd, v in list(q3["cos_by_index_distance"].items())[:8]))
    print(f"       => {q3['verdict']}")
    print(f"\n  wrote: {out}/wres_subspace.json")
    print("=" * 78)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--geom_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--out_dir", default="wres_subspace_out")
    p.add_argument("--n_null", type=int, default=500)
    p.add_argument("--lowdim_frac", type=float, default=0.35, help="PR < this*k => low-dim subspace")
    p.add_argument("--mono_tol", type=float, default=0.03, help="tolerance for monotone-decay check")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
