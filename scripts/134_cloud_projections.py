"""
134 — Per-layer cloud projections (α / β residual clouds in 4 informative 2D planes).

Precompute for an interactive dashboard tab. For each layer L (0..35) in each regime
(raw / scaffold) we:

  1. Load residual H[L] from field_dump and compute axes on held rows:
       w_res = Fisher LDA, u = unit(mean grad), δ = unit(mean_class1 - mean_class0)
     plus geometric helpers from j132 atlas:
       sep_dir  = δ            (separation direction by construction)
       top_PC   = first SVD direction of centered residual
  2. Center the full residual cloud, then PROJECT all 538 points onto 4 planes:
       sep_topPC   — the anisotropy plane (where 2603.12760 / our atlas live)
       wres_u      — the read-use plane (canonical bypass picture)
       u_delta     — the use-write plane (raw-vs-scaffold contrast was sharpest here)
       wres_delta  — read-write plane
     Each plane is given as 2 orthonormal unit vectors (Gram–Schmidt on the
     two source axes); we project the cloud and also project the three triad
     axes (so we can draw little arrows from origin in the scatter).
  3. Save one .npz per regime with coords, arrows, labels and metadata.

Output: data/analysis/runD_v2/cloud_projections/{regime}_projections.npz
  y                : (n_prompts,) int labels
  families         : (n_prompts,) <U-… family strings
  layers           : (n_layers,) int
  coords_<plane>   : (n_layers, n_prompts, 2) cloud coordinates
  arrows_<plane>   : (n_layers, 3, 2) projections of (w_res, u, δ) in the plane
  axis_labels      : ["w_res", "u", "delta"]
  plane_names      : ["sep_topPC", "wres_u", "u_delta", "wres_delta"]

Pure CPU / numpy. Per layer per regime: load 2 npy + a few SVDs. Few seconds.

SELF-TEST (no dump):  python 134_cloud_projections.py --self_test
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
logger = logging.getLogger("clouds134")


# ─────────────────────────── pure-numpy core ──────────────────────────────────
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


def top_pc_of(Hc):
    """Top right singular vector of (already-centered) Hc."""
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    return unit_raw(Vt[0])


def orthonormal_plane(a, b):
    """Gram-Schmidt: e1 = unit(a); e2 = unit(b - <b,e1>e1). If b ≈ ∥ a, e2 falls
    back to a random orthogonal direction so the plane is well-defined."""
    e1 = unit_raw(a)
    proj = (b @ e1) * e1
    rem = b - proj
    if np.linalg.norm(rem) < 1e-10:
        # b is collinear with a — pick any orthogonal direction
        rnd = np.random.default_rng(0).standard_normal(a.shape[0])
        rnd -= (rnd @ e1) * e1
        e2 = unit_raw(rnd)
    else:
        e2 = unit_raw(rem)
    return e1, e2


def project_2d(M, e1, e2):
    """Project rows of M onto orthonormal pair (e1, e2). Returns (N, 2)."""
    return np.column_stack([M @ e1, M @ e2])


# ─────────────────────────── self-test ────────────────────────────────────────
def self_test():
    rng = np.random.default_rng(0)
    d, n = 60, 200
    y = (np.arange(n) % 2).astype(int)

    # synthetic two-cloud with separation along known sep direction, plus a
    # high-variance noise direction.
    big = unit_raw(rng.standard_normal(d))           # high-variance axis
    sep = rng.standard_normal(d); sep -= (sep @ big) * big; sep = unit_raw(sep)
    H = (3.0 * np.outer(2 * y - 1.0, sep)            # class separation along sep
         + 5.0 * np.outer(rng.standard_normal(n), big)  # big variance along big
         + 0.1 * rng.standard_normal((n, d)))         # tiny isotropic

    Hc = H - H.mean(0)
    e_big = top_pc_of(Hc)
    assert abs(e_big @ big) > 0.95, "topPC recovers high-variance direction"

    # plane (sep, topPC) — orthonormalize
    e1, e2 = orthonormal_plane(sep, e_big)
    assert abs(e1 @ e2) < 1e-9, "plane vectors orthogonal"
    assert abs(np.linalg.norm(e1) - 1) < 1e-9 and abs(np.linalg.norm(e2) - 1) < 1e-9, "unit"

    # project clouds onto (sep, topPC) — class means split along e1 (sep), not along e2
    coords = project_2d(Hc, e1, e2)
    m0, m1 = coords[y == 0].mean(0), coords[y == 1].mean(0)
    gap_along_e1 = abs(m1[0] - m0[0])
    gap_along_e2 = abs(m1[1] - m0[1])
    assert gap_along_e1 > 5 * gap_along_e2, "separation is along e1 (sep), not e2 (topPC)"

    # arrow projection: project the sep axis itself into the plane — should be (1, 0)
    arrow = np.array([sep @ e1, sep @ e2])
    assert abs(arrow[0] - 1.0) < 1e-6 and abs(arrow[1]) < 1e-6, "sep direction in plane is (1, 0)"

    # collinear b — fallback path
    e1b, e2b = orthonormal_plane(sep, sep)
    assert abs(e1b @ e2b) < 1e-9, "collinear-fallback still produces orthogonal plane"

    print("[self_test] OK — topPC recovery, plane orthonormality, separation alignment, "
          "arrow projection, collinear fallback. pass.")


# ─────────────────────────── real run ─────────────────────────────────────────
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


def process_regime(name, dpath, args):
    dump, meta, fams, n_layers = load_dump(dpath)
    y = meta["y"].astype(int)
    d_model = int(meta["d"])
    n_prompts = len(y)
    held = ~reconstruct_split(fams, args.split_seed, args.train_frac)
    families = np.array(fams)

    plane_names = ["sep_topPC", "wres_u", "u_delta", "wres_delta"]
    axis_labels = ["w_res", "u", "delta"]

    coords = {p: np.zeros((n_layers, n_prompts, 2), np.float32) for p in plane_names}
    arrows = {p: np.zeros((n_layers, 3, 2), np.float32) for p in plane_names}

    for L in range(n_layers):
        H_full = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)   # (n_prompts, d)
        G_held = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)[held]

        H_held = H_full[held]
        y_held = y[held]

        # axes on held rows
        w = fisher_axis(H_held, y_held, args.shrink)
        u = unit_raw(G_held.mean(0))
        delta = unit_raw(H_held[y_held == 1].mean(0) - H_held[y_held == 0].mean(0))

        # centered full cloud and its topPC
        Hc = H_full - H_full.mean(0)
        pc = top_pc_of(Hc)

        sep_dir = delta  # convention: class-mean diff IS the separation direction

        plane_defs = {
            "sep_topPC":  (sep_dir, pc),
            "wres_u":     (w, u),
            "u_delta":    (u, delta),
            "wres_delta": (w, delta),
        }

        for pname, (a, b) in plane_defs.items():
            e1, e2 = orthonormal_plane(a, b)
            coords[pname][L] = project_2d(Hc, e1, e2).astype(np.float32)
            arrows[pname][L, 0] = np.array([w @ e1, w @ e2], np.float32)
            arrows[pname][L, 1] = np.array([u @ e1, u @ e2], np.float32)
            arrows[pname][L, 2] = np.array([delta @ e1, delta @ e2], np.float32)

        if (L + 1) % 6 == 0:
            logger.info("  [%s] layer %d/%d processed", name, L + 1, n_layers)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "y": y.astype(np.int8),
        "families": families,
        "layers": np.arange(n_layers, dtype=np.int32),
        "axis_labels": np.array(axis_labels),
        "plane_names": np.array(plane_names),
    }
    for pname in plane_names:
        payload[f"coords_{pname}"] = coords[pname]
        payload[f"arrows_{pname}"] = arrows[pname]

    out_path = out / f"{name}_projections.npz"
    np.savez_compressed(out_path, **payload)
    logger.info("[%s] saved %s  size=%.1f MB", name, out_path, out_path.stat().st_size / 1e6)
    return n_layers, n_prompts


def run_real(args):
    n_raw, p_raw = process_regime("raw", args.raw_dump, args)
    n_scf, p_scf = process_regime("scaffold", args.scaffold_dump, args)

    print("\n" + "=" * 100)
    print("CLOUD PROJECTIONS — per-layer 2D coords for 4 planes, both regimes")
    print("=" * 100)
    print(f"  raw:      n_layers={n_raw}  n_prompts={p_raw}")
    print(f"  scaffold: n_layers={n_scf}  n_prompts={p_scf}")
    print(f"  planes:   sep_topPC | wres_u | u_delta | wres_delta")
    print(f"  output:   {args.out_dir}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--raw_dump", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--scaffold_dump", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/cloud_projections")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
