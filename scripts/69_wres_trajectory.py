"""
69_wres_trajectory.py   [LAPTOP / CPU-ONLY -- no GPU, no model, no transcoders]
===================================================================
Cross-layer geometry of the residual concept axis w_res, from data ALREADY on
disk (the small arrays synced from CSD3). Needs ONLY:
  * w_res_*.npy            (saved by 64: w_res_postL14..25.npy + w_res_final.npy;
                            or w_res65_*.npy from 65 if present)
  * concept_directions.npz (from 60: Sigma_inv, gbar, lbar)
No transformers, no torch, no transcoders. Runs in seconds on a laptop CPU.

THE QUESTION (so far unanswered): is the alpha/beta concept a SINGLE stable
residual direction maintained across depth, or does the representation ROTATE
through the network? And does w_res drift TOWARD gbar near the output -- which
would be direct evidence for downstream mediation (M2: the concept gets
"translated" into the unembedding readout basis by later layers)?

This is reference material for the thesis regardless of what the GPU batch (65-68)
later shows:
  * w_res stable + orthogonal to gbar throughout (incl. final)
      -> the concept lives in a CONSISTENT residual direction distinct from the
         unembedding contrast; strengthens the "wrong axis / not localizable" story.
  * w_res rotates toward gbar near the end
      -> downstream mediation (M2): gbar is partly rehabilitated as the LATE-layer
         readout axis; reframes the interpretation. Flag this to your advisor.

OUTPUTS (default ./wres_trajectory_out/):
  wres_trajectory.json   pairwise cos_C matrix, gbar/lbar trajectories, consecutive-cos
  wres_trajectory.csv    per-layer: cos_C(w_res_L, gbar), cos_C(w_res_L, lbar),
                          cos_C(w_res_L, w_res_final), cos to previous layer
  wres_pairwise_cosC.npy  the LxL matrix (for your own plotting)
  wres_trajectory.png     (if matplotlib present) heatmap + trajectory curves

SELF-TEST (synthetic; verifies stable-vs-rotating discrimination):
  python 69_wres_trajectory.py --self_test
"""

from __future__ import annotations
import argparse, glob, json, logging, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wres_trajectory")

TAP_RE = re.compile(r"w_res(?:65)?_(postL(\d+)|final)\.npy$")


# =====================================================================
# Geometry (pure numpy; causal metric + Euclidean sibling)
# =====================================================================

def whitener(Si: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(Si); w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def causal_cos(a: np.ndarray, b: np.ndarray, Si: np.ndarray) -> float:
    num = float(a @ Si @ b)
    na = float(np.sqrt(max(a @ Si @ a, 1e-30))); nb = float(np.sqrt(max(b @ Si @ b, 1e-30)))
    return num / (na * nb)


def euclid_cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na * nb > 0 else float("nan")


def pairwise_causal(W: np.ndarray, Si: np.ndarray) -> np.ndarray:
    """W: (L, d) stack of axes (each beta-oriented). Returns (L, L) causal-cos matrix."""
    A = whitener(Si)
    Ww = W @ A.T                                   # whiten rows
    n = np.linalg.norm(Ww, axis=1, keepdims=True)
    Ww = Ww / np.clip(n, 1e-30, None)
    return Ww @ Ww.T


# =====================================================================
# Loading
# =====================================================================

def load_axes(folder: Path) -> Tuple[List[str], np.ndarray]:
    """Find w_res_*.npy, order by depth (postL14..25, then final). Return labels + (L,d)."""
    found = []
    for p in folder.glob("w_res*_*.npy"):
        m = TAP_RE.search(p.name)
        if not m:
            continue
        if m.group(1) == "final":
            order = 10_000
            label = "final"
        else:
            order = int(m.group(2))
            label = f"L{order}"
        found.append((order, label, p))
    found.sort(key=lambda t: t[0])
    if not found:
        raise SystemExit(f"no w_res_*.npy found in {folder} (expected files saved by script 64/65)")
    labels = [lbl for _, lbl, _ in found]
    W = np.array([np.load(p).astype(np.float64) for _, _, p in found])
    # normalise to unit raw-L2 (Fisher axes are already unit, but be safe)
    W = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-30, None)
    logger.info("loaded %d axes: %s (dim=%d)", len(labels), labels, W.shape[1])
    return labels, W


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(69)
    d = 200
    ev = np.concatenate([np.linspace(12, 3, 12), np.linspace(2, 0.05, d - 12)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Si = np.linalg.inv((Q * ev) @ Q.T + 1e-3 * np.eye(d))
    gbar = rng.standard_normal(d)

    L = 13
    w0 = rng.standard_normal(d); w0 /= np.linalg.norm(w0)
    w1 = rng.standard_normal(d); w1 -= w0 * (w0 @ w1); w1 /= np.linalg.norm(w1)

    # STABLE: same direction + tiny per-layer noise
    W_stable = np.array([w0 + 0.01 * rng.standard_normal(d) for _ in range(L)])
    W_stable /= np.linalg.norm(W_stable, axis=1, keepdims=True)

    # ROTATING: rotate from w0 toward w1 across depth
    thetas = np.linspace(0, np.pi / 2, L)
    W_rot = np.array([np.cos(t) * w0 + np.sin(t) * w1 for t in thetas])

    # gbar-APPROACH: orthogonal to gbar early, rotating toward gbar late
    ghat = gbar / np.linalg.norm(gbar)
    g_perp = w0 - ghat * (ghat @ w0); g_perp /= np.linalg.norm(g_perp)
    W_appr = np.array([np.cos(t) * g_perp + np.sin(t) * ghat for t in thetas])

    print("\n--- SELF TEST -------------------------------------------------")
    for nm, W in [("stable", W_stable), ("rotating", W_rot), ("gbar-approach", W_appr)]:
        M = pairwise_causal(W, Si)
        corner = M[0, -1]                     # first vs last layer
        consec = np.mean([M[i, i + 1] for i in range(L - 1)])
        cg_first = causal_cos(W[0], gbar, Si); cg_last = causal_cos(W[-1], gbar, Si)
        print(f"  {nm:14s}: corner cos(first,last)={corner:+.2f}  consec={consec:+.2f}  "
              f"cos(.,gbar) {cg_first:+.2f}->{cg_last:+.2f}")

    Ms = pairwise_causal(W_stable, Si); Mr = pairwise_causal(W_rot, Si)
    assert Ms[0, -1] > 0.8, "stable axis: first/last must stay aligned"
    assert Mr[0, -1] < 0.4, "rotating axis: first/last must diverge"
    assert np.mean([Mr[i, i + 1] for i in range(L - 1)]) > 0.8, "rotating: consecutive still close"
    cg = [causal_cos(W_appr[i], gbar, Si) for i in range(L)]
    assert cg[-1] > cg[0] + 0.3, "gbar-approach: cos with gbar must rise across depth"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(detects stable vs rotating axis, and drift toward gbar = M2 signature)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def run_real(args) -> None:
    folder = Path(args.geom_dir)
    labels, W = load_axes(folder)
    L, d = W.shape

    cd = np.load(args.concept_npz)
    if "Sigma_inv" not in cd:
        raise SystemExit("concept_npz needs 'Sigma_inv' (from 60_).")
    Si = cd["Sigma_inv"].astype(np.float64)
    gbar = cd["gbar"].astype(np.float64) if "gbar" in cd else None
    lbar = cd["lbar"].astype(np.float64) if "lbar" in cd else None

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    M = pairwise_causal(W, Si)
    np.save(out / "wres_pairwise_cosC.npy", M)
    Meu = np.array([[euclid_cos(W[i], W[j]) for j in range(L)] for i in range(L)])

    final_idx = labels.index("final") if "final" in labels else L - 1
    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            "tap": lab,
            "cos_C_gbar": causal_cos(W[i], gbar, Si) if gbar is not None else None,
            "cos_C_lbar": causal_cos(W[i], lbar, Si) if lbar is not None else None,
            "cos_C_to_final": float(M[i, final_idx]),
            "cos_C_to_prev": float(M[i, i - 1]) if i > 0 else None,
            "euclid_cos_to_final": float(Meu[i, final_idx]),
        })

    # summary diagnostics
    consec = [float(M[i, i + 1]) for i in range(L - 1)]
    offdiag = M[np.triu_indices(L, 1)]
    gbar_traj = [r["cos_C_gbar"] for r in rows] if gbar is not None else None
    gbar_rise = (gbar_traj is not None and len(gbar_traj) >= 2
                 and gbar_traj[final_idx] - min(gbar_traj) > args.gbar_rise_thresh)

    summary = {
        "geom_dir": str(folder), "n_axes": L, "labels": labels,
        "pairwise_causal_cos": {
            "mean_offdiag": float(np.mean(offdiag)),
            "min_offdiag": float(np.min(offdiag)),
            "first_vs_last": float(M[0, final_idx]),
            "mean_consecutive": float(np.mean(consec)),
        },
        "stability_verdict": (
            "STABLE (single axis across depth)" if float(np.min(offdiag)) > args.stable_thresh
            else "ROTATING (axis changes across depth)" if float(M[0, final_idx]) < args.rotate_thresh
            else "PARTIALLY STABLE (drifts but related)"),
        "gbar_trajectory": gbar_traj,
        "gbar_drift_toward_final": gbar_rise,
        "gbar_interpretation": (
            "w_res DRIFTS toward gbar by the final layer -> evidence for DOWNSTREAM "
            "MEDIATION (M2): concept translated into the readout basis late. gbar partly "
            "rehabilitated as the late-layer axis." if gbar_rise
            else "w_res stays ~orthogonal to gbar at all depths incl. final -> the concept "
            "axis is distinct from the unembedding contrast throughout (no M2 rescue of gbar)."),
        "per_layer": rows,
    }
    with open(out / "wres_trajectory.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    import csv
    with open(out / "wres_trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # optional figure
    fig_made = False
    if not args.no_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            im = ax[0].imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
            ax[0].set_xticks(range(L)); ax[0].set_yticks(range(L))
            ax[0].set_xticklabels(labels, rotation=90, fontsize=7)
            ax[0].set_yticklabels(labels, fontsize=7)
            ax[0].set_title("pairwise cos_C(w_res_Li, w_res_Lj)")
            fig.colorbar(im, ax=ax[0], fraction=0.046)
            xs = range(L)
            if gbar_traj is not None:
                ax[1].plot(xs, gbar_traj, "o-", label="cos_C(w_res, gbar)")
            ax[1].plot(xs, [r["cos_C_to_final"] for r in rows], "s--", label="cos_C(w_res, w_res_final)")
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(labels, rotation=90, fontsize=7)
            ax[1].set_ylim(-1.05, 1.05); ax[1].legend(fontsize=8)
            ax[1].set_title("concept-axis trajectory across depth")
            fig.tight_layout(); fig.savefig(out / "wres_trajectory.png", dpi=130)
            fig_made = True
        except Exception as e:  # matplotlib missing or headless issue -> skip gracefully
            logger.info("figure skipped (%s); JSON/CSV still written", e)

    # console
    print("\n" + "=" * 78)
    print("w_res TRAJECTORY ACROSS DEPTH  (laptop/CPU; from synced arrays)")
    print("=" * 78)
    print(f"  axes: {labels}")
    s = summary["pairwise_causal_cos"]
    print(f"\n  pairwise cos_C: mean off-diag={s['mean_offdiag']:+.3f}  min={s['min_offdiag']:+.3f}  "
          f"first-vs-last={s['first_vs_last']:+.3f}  consecutive={s['mean_consecutive']:+.3f}")
    print(f"  STABILITY: {summary['stability_verdict']}")
    if gbar_traj is not None:
        print(f"\n  cos_C(w_res, gbar) by depth:")
        print("    " + "  ".join(f"{lab}:{c:+.2f}" for lab, c in zip(labels, gbar_traj)))
        print(f"  {summary['gbar_interpretation']}")
    print(f"\n  wrote: {out}/wres_trajectory.json, .csv, wres_pairwise_cosC.npy"
          + (", wres_trajectory.png" if fig_made else ""))
    print("=" * 78)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--geom_dir", default="data/analysis/runD_v2/geometry_stage1",
                   help="folder with w_res_*.npy (from 64/65)")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--out_dir", default="wres_trajectory_out")
    p.add_argument("--stable_thresh", type=float, default=0.7,
                   help="min off-diagonal cos_C to call the axis STABLE across depth")
    p.add_argument("--rotate_thresh", type=float, default=0.4,
                   help="first-vs-last cos_C below this => ROTATING")
    p.add_argument("--gbar_rise_thresh", type=float, default=0.3,
                   help="rise in cos_C(w_res,gbar) toward final to flag M2 mediation")
    p.add_argument("--no_fig", action="store_true", help="skip the matplotlib figure")
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
