"""
132 — Dump u and delta per-layer trajectories + SVD-3 projections.

j105 already dumped w_res per layer (`wres_all_layers/`); the same trajectory
matrix never got saved for the using axis u or the writing axis delta — they
were computed in memory by 127 and thrown away.

This script reconstructs them from the existing field_dump using the same
convention as 127 (held-out family split, label-based class diff, mean gradient):

    u[L]     = unit(mean grad[L] over held rows)
    delta[L] = unit(mean_class1(res[L]) - mean_class0(res[L]) over held rows)

For each axis we save:
  - per-layer unit vector files  (axis_trajectories/<kind>_L00..L35.npy)
  - stacked (36, 2560) trajectory (axis_trajectories/<kind>_all.npy)
  - per-axis SVD-3                (axis_trajectories/<kind>_svd3.npz)

And one shared basis for OVERLAY rendering:
  - triad_svd3.npz  with proj_w, proj_u, proj_d each (36, 3) in one common basis,
    obtained by SVD-3 of the stacked (108, 2560) triad matrix.

Pure CPU / numpy. Walks field_dump only — no model, no forward pass. Seconds.

SELF-TEST (no dump required):  python 132_dump_axis_trajectories.py --self_test
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
logger = logging.getLogger("traj132")


def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def svd3(matrix):
    """SVD on (n_rows, d) matrix; return (n_rows, 3) projection, singular vals, basis."""
    M = np.asarray(matrix, dtype=np.float64)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    proj = (U[:, :3] * S[:3])  # equivalent to M @ Vt[:3].T
    return proj, S, Vt[:3]


def self_test():
    rng = np.random.default_rng(0)
    d = 60

    # construct two known directions e1, e2 in d-dim
    e1 = unit_raw(rng.standard_normal(d))
    e2 = rng.standard_normal(d); e2 -= (e2 @ e1) * e1; e2 = unit_raw(e2)

    # build a 36-layer trajectory that rotates in the (e1, e2) plane
    n = 36
    theta = np.linspace(0, 1.5 * np.pi, n)
    traj = np.stack([np.cos(t) * e1 + np.sin(t) * e2 for t in theta])

    proj, S, basis = svd3(traj)
    assert proj.shape == (n, 3), "proj shape"
    # 99% of variance in top 2 components (planar rotation)
    var_top2 = (S[:2] ** 2).sum() / (S ** 2).sum()
    assert var_top2 > 0.99, f"planar trajectory should have ~all variance in 2 components: {var_top2}"

    # class_diff axis test
    y = (np.arange(40) % 2).astype(int)
    H = np.outer(2 * y - 1.0, e1) + 0.05 * rng.standard_normal((40, d))
    delta = unit_raw(H[y == 1].mean(0) - H[y == 0].mean(0))
    assert abs(delta @ e1) > 0.95, "delta recovers e1"

    # mean grad test
    G = np.outer(np.ones(40), e2) + 0.05 * rng.standard_normal((40, d))
    u = unit_raw(G.mean(0))
    assert abs(u @ e2) > 0.95, "u recovers e2"

    # triad stack SVD
    triad = np.concatenate([traj, traj * -1, traj * 0.5])
    p, _, _ = svd3(triad)
    assert p.shape == (3 * n, 3), "triad stacked proj"
    print("[self_test] OK — svd3 shape/variance, delta/u recovery, triad stacking. pass.")


def run_real(args):
    dump = Path(args.dump_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    y = meta["y"].astype(int)
    d_model = int(meta["d"])
    n_layers = int(meta["n_layers"])

    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    held = ~trm
    logger.info("dump=%s d=%d n_layers=%d n_held=%d (train_frac=%.2f, split_seed=%d)",
                dump, d_model, n_layers, held.sum(), args.train_frac, args.split_seed)

    # ── compute u and delta per layer ───────────────────────────────────────
    u_all = np.zeros((n_layers, d_model), np.float64)
    delta_all = np.zeros((n_layers, d_model), np.float64)
    yo = y[held]

    for L in range(n_layers):
        # u = unit(mean grad over held)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)[held]
        u_all[L] = unit_raw(G.mean(0))

        # delta = unit(mean_class1 - mean_class0) of residual over held
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)[held]
        delta_all[L] = unit_raw(H[yo == 1].mean(0) - H[yo == 0].mean(0))

        if (L + 1) % 6 == 0:
            logger.info("  layers %d/%d processed", L + 1, n_layers)

    # ── save per-layer + stacked ────────────────────────────────────────────
    for L in range(n_layers):
        np.save(out / f"u_L{L:02d}.npy", u_all[L])
        np.save(out / f"delta_L{L:02d}.npy", delta_all[L])
    np.save(out / "u_all.npy", u_all)
    np.save(out / "delta_all.npy", delta_all)
    logger.info("wrote per-layer u/delta files (n_layers=%d) and stacked arrays", n_layers)

    # ── per-axis SVD-3 projections ──────────────────────────────────────────
    u_proj, u_S, u_basis = svd3(u_all)
    delta_proj, delta_S, delta_basis = svd3(delta_all)

    u_var3 = float((u_S[:3] ** 2).sum() / (u_S ** 2).sum())
    d_var3 = float((delta_S[:3] ** 2).sum() / (delta_S ** 2).sum())
    logger.info("u SVD-3 explains %.1f%% | delta SVD-3 explains %.1f%%",
                u_var3 * 100, d_var3 * 100)

    np.savez(out / "u_svd3.npz",
             proj=u_proj, singular=u_S, basis=u_basis,
             labels=np.array([f"L{L:02d}" for L in range(n_layers)]),
             var_explained=u_var3)
    np.savez(out / "delta_svd3.npz",
             proj=delta_proj, singular=delta_S, basis=delta_basis,
             labels=np.array([f"L{L:02d}" for L in range(n_layers)]),
             var_explained=d_var3)

    # ── shared triad SVD-3 for overlay (load existing w_res too) ───────────
    wres_dir = Path(args.wres_dir)
    if wres_dir.exists():
        wres_all = np.zeros((n_layers, d_model), np.float64)
        n_found = 0
        for L in range(n_layers):
            p = wres_dir / f"w_res_L{L:02d}.npy"
            if p.exists():
                v = np.load(p).astype(np.float64)
                wres_all[L] = unit_raw(v) if np.linalg.norm(v) > 0 else v
                n_found += 1
            else:
                logger.warning("missing w_res file: %s", p)
        logger.info("loaded w_res from %d/%d layers", n_found, n_layers)

        triad = np.concatenate([wres_all, u_all, delta_all], axis=0)  # (3*n_layers, d)
        triad_proj, triad_S, triad_basis = svd3(triad)
        triad_var3 = float((triad_S[:3] ** 2).sum() / (triad_S ** 2).sum())
        logger.info("triad shared SVD-3 explains %.1f%% of the joint rotation", triad_var3 * 100)

        proj_w = triad_proj[:n_layers]
        proj_u = triad_proj[n_layers:2 * n_layers]
        proj_d = triad_proj[2 * n_layers:]

        np.savez(out / "triad_svd3.npz",
                 proj_w=proj_w, proj_u=proj_u, proj_d=proj_d,
                 singular=triad_S, basis=triad_basis,
                 labels=np.array([f"L{L:02d}" for L in range(n_layers)]),
                 var_explained=triad_var3)
        logger.info("wrote triad_svd3.npz (shared overlay basis)")
    else:
        logger.warning("w_res dir not found at %s — skipping shared triad SVD-3", wres_dir)

    # ── summary print ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("AXIS TRAJECTORIES — per-layer u and delta (saved to disk for dashboard / Tab 9)")
    print("=" * 100)
    print(f"  layers:               {n_layers}")
    print(f"  d_model:              {d_model}")
    print(f"  n_held:               {held.sum()}  (train_frac={args.train_frac})")
    print(f"  u SVD-3 var explained:     {u_var3*100:5.1f}%")
    print(f"  delta SVD-3 var explained: {d_var3*100:5.1f}%")
    if wres_dir.exists():
        print(f"  triad SVD-3 var explained: {triad_var3*100:5.1f}%  (shared basis for overlay)")
    print(f"  output dir: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--wres_dir", default="data/analysis/runD_v2/wres_all_layers")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/axis_trajectories")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
