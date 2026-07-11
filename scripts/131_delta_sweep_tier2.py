"""
131_delta_sweep_tier2.py   [missing-direction sweep: writing axis delta = mu1 - mu0]
========================================================================================
Closes the last coverage gap of the measured flip-law calibration: the tier-2 sweeps
(122) covered {w_res, usage, random*, shuffled*} but never the writing direction.
This script runs the SAME tier-2 protocol for delta (and optionally re-runs usage as
a continuity cross-check), on the SAME layers / c grid / baseline-correct held-out
pool, and emits cells_tier2_delta.csv with columns identical to 122's --dump_cells
output (layer, c, dir, idx, y, m0, m1, intact), so 132_flip_law_assembly.py and the
local calibration notebooks consume it unchanged.

Conventions replicated from 122 (MUST match the original sweep run):
  * split:   reconstruct_split(families.json, --split_seed, --train_frac)
  * sigma:   std of train-split residual projections onto w_res at that layer
             (direction-independent norm-matching, as in 122)
  * sign:    s = +1 if y==0 else -1  (push toward the opposite class by label)
  * margin:  log-softmax difference logit[id_class1] - logit[id_class0]
  * pool:    full baseline-correct held-out prompts
  * delta:   unit(mu_{y==1} - mu_{y==0}) computed on the TRAIN split residuals

Layer / c grid: pass --match_cells <existing cells_tier2.csv> to replicate exactly
the layers and c values of the original sweep (recommended), or set --layers/--c_grid.

SELF-TEST (no torch / no GPU):  python 131_delta_sweep_tier2.py --self_test

Typical CSD3 run (per concept):
  python 131_delta_sweep_tier2.py \
      --dump_dir  data/analysis/runD_v2/B1_alpha_beta/field_dump \
      --corpus    data/corpora/B1_alpha_beta.jsonl \
      --match_cells data/analysis/runD_v2/B1_alpha_beta/cells_tier2.csv \
      --out_dir   data/analysis/runD_v2/B1_alpha_beta \
      --dirs delta,usage --split_seed <SAME AS 122> --train_frac <SAME AS 122>
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sweep131")


# =====================================================================
# Pure-numpy core (shared conventions with 122; exercised by --self_test)
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


def delta_axis(H, y):
    """Writing direction on the train split: unit(mu1 - mu0)."""
    return unit_raw(H[y == 1].mean(0) - H[y == 0].mean(0))


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def grid_from_cells(path):
    layers, cs = set(), set()
    with open(path) as f:
        for row in _csv.DictReader(f):
            layers.add(int(row["layer"])); cs.add(float(row["c"]))
    return sorted(layers), sorted(cs)


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    # delta axis points from class0 mean to class1 mean
    H = np.vstack([rng.normal(0, 1, (50, 8)), rng.normal(0, 1, (50, 8)) + np.array([3] + [0] * 7)])
    y = np.array([0] * 50 + [1] * 50)
    d = delta_axis(H, y)
    assert d[0] > 0.95, "delta must align with the injected class-mean offset"
    # split reconstruction is deterministic in (fams, seed, frac)
    fams = [f"f{i%7}" for i in range(100)]
    a = reconstruct_split(fams, 13, 0.6); b = reconstruct_split(fams, 13, 0.6)
    assert (a == b).all() and 0.3 < a.mean() < 0.9
    # grid round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("layer,c,dir,idx,y,m0,m1,intact\n16,0.5,w_res,0,0,-1,-1,1\n24,16.0,usage,1,1,2,1,1\n")
        p = f.name
    L, C = grid_from_cells(p); os.unlink(p)
    assert L == [16, 24] and C == [0.5, 16.0]
    print("[self_test] OK - delta axis, split reconstruction, grid match pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    prompts = [json.loads(l) for l in open(args.corpus)]
    nP = len(prompts); assert nP == len(fams), "corpus/dump mismatch"
    y = meta["y"].astype(int)
    m0 = meta["clean_margin"].astype(np.float64)
    n_layers = int(meta["n_layers"]); d = int(meta["d"])
    id0 = meta["id_class0"].astype(int) if "id_class0" in meta else np.full(nP, int(meta["alpha_id"]))
    id1 = meta["id_class1"].astype(int) if "id_class1" in meta else np.full(nP, int(meta["beta_id"]))
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)

    if args.match_cells:
        layers, c_grid = grid_from_cells(args.match_cells)
        logger.info("grid matched to %s: layers=%s c=%s", args.match_cells, layers, c_grid)
    else:
        layers = [int(x) for x in args.layers.split(",")]
        c_grid = [float(x) for x in args.c_grid.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); norm_mod = _chain(model, "model.norm")
    last = n_layers - 1

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    def steer_eval(i, L, delta_vec):
        inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta_vec, dtype=torch.float32, device=args.device)
        def pre(m_, a):
            hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(row, 0)
            m1 = float(lp[int(id1[i])] - lp[int(id0[i])])
            t1 = int(row.argmax().item())
            return m1, int(t1 in (int(id0[i]), int(id1[i])))
        finally:
            h.remove()

    correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
    held = np.where(~trm)[0]
    targets = [int(i) for i in held if correct[i]]           # full baseline-correct held-out pool
    logger.info("pool: %d baseline-correct held-out targets", len(targets))

    want = [s.strip() for s in args.dirs.split(",") if s.strip()]
    cell_rows = []
    for L in layers:
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        w = fisher_axis(H[trm], y[trm], args.shrink)
        sigma = float(np.std(H[trm] @ w))                    # 122's norm-matching sigma
        dd = {}
        if "delta" in want:
            dd["delta"] = delta_axis(H[trm], y[trm])
        if "usage" in want:
            dd["usage"] = unit_raw(G.mean(0))
        for c in c_grid:
            for dname, vec in dd.items():
                flips = 0
                for i in targets:
                    s = +1.0 if y[i] == 0 else -1.0
                    m1, intact = steer_eval(i, L, (s * c * sigma) * unit_raw(vec))
                    cell_rows.append({"layer": int(L), "c": float(c), "dir": dname, "idx": int(i),
                                      "y": int(y[i]), "m0": float(m0[i]), "m1": float(m1),
                                      "intact": int(intact)})
                    if (y[i] == 0 and m0[i] < 0 and m1 > 0) or (y[i] == 1 and m0[i] > 0 and m1 < 0):
                        flips += 1
                logger.info("L%02d c=%-5g %-6s flip_norm=%.3f (sigma=%.3g)",
                            L, c, dname, flips / max(len(targets), 1), sigma)

    outfile = out / "cells_tier2_delta.csv"
    with open(outfile, "w", newline="") as f:
        wtr = _csv.DictWriter(f, fieldnames=["layer", "c", "dir", "idx", "y", "m0", "m1", "intact"])
        wtr.writeheader()
        for r in cell_rows:
            wtr.writerow(r)
    logger.info("wrote %s  (%d rows)", outfile, len(cell_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--dump_dir"); ap.add_argument("--corpus"); ap.add_argument("--out_dir")
    ap.add_argument("--match_cells", default=None,
                    help="existing cells_tier2.csv to replicate layers/c grid")
    ap.add_argument("--layers", default="16,22,23,24,35")
    ap.add_argument("--c_grid", default="0.5,1,2,4,8,16,32")
    ap.add_argument("--dirs", default="delta",
                    help="comma list from {delta,usage}; usage = continuity cross-check vs 122")
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shrink", type=float, default=0.1, help="MUST match the 122 run")
    ap.add_argument("--split_seed", type=int, default=0, help="MUST match the 122 run")
    ap.add_argument("--train_frac", type=float, default=0.6, help="MUST match the 122 run")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    assert args.dump_dir and args.corpus and args.out_dir, "--dump_dir/--corpus/--out_dir required"
    run_real(args)


if __name__ == "__main__":
    main()
