"""
122_b1_steering_sweep.py   [B1 measured steering sweep — the realized side of the law]
========================================================================================
Per-concept steering harness (spec §4 step 6). Consumes the concept's 119 field
dump (directions come from the dump — w_res recomputed on the train split, u_bar
from gradients; no recapture) and the rendered corpus, runs norm-matched pushes,
and measures the FULL metric set per (layer, c, dir):

  mean_dmargin_toward   signed margin movement toward the target class
  margin_flip           86-compatible flip (ceiling = baseline accuracy)
  flip_norm             flip among baseline-correct targets (ceiling 1.0 —
                        the cross-concept headline metric)
  flip_c0->c1 / c1->c0  direction-resolved normalized flips
  intact_rate           top-1 in the prompt's two class tokens (post-push)
  intact_flip           margin_flip AND intact (now meaningful: scaffolded corpus)

Two tiers:
  tier1 (default): ALL layers x c in {1,4,16} x {w_res, usage, shuffled*, random*}
                   x up to --t1_per_class targets/class  (~35k forwards)
  tier2: dense c grid {0.5,1,2,4,8,16,32} on --tier2_layers (default: auto =
         top-3 usage-flip layers from a tier1 CSV + fixed controls 16,24,35)
         x the FULL baseline-correct held-out pool

Output CSVs are column-compatible with 116's auto-detect loader, so the
predicted-vs-measured calibration and dose-response plots come for free:
  python 116_flip_law_retro.py --dump_dir <dump> --measured_86 steering_sweep_tier1.csv

SELF-TEST (no torch / no repo):  python 122_b1_steering_sweep.py --self_test
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
logger = logging.getLogger("sweep122")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
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


def aggregate_cell(recs):
    """recs: list of per-target dicts with keys y, m0, m1, intact.
    Returns the full metric set for one (layer, c, dir) cell."""
    dm_t, fl, fln, f01, f10, it, itf = [], [], [], [], [], [], []
    for r in recs:
        s = +1.0 if r["y"] == 0 else -1.0
        dm_t.append(s * (r["m1"] - r["m0"]))
        if r["y"] == 0:
            f = int(r["m0"] < 0 and r["m1"] > 0); corr = r["m0"] < 0
        else:
            f = int(r["m0"] > 0 and r["m1"] < 0); corr = r["m0"] > 0
        fl.append(f); it.append(r["intact"]); itf.append(int(f and r["intact"]))
        if corr:
            fln.append(f)
            (f01 if r["y"] == 0 else f10).append(f)
    n = len(recs)
    return {"n_targets": n, "n_correct": len(fln),
            "mean_dmargin_toward": float(np.mean(dm_t)),
            "margin_flip": float(np.mean(fl)),
            "flip_norm": float(np.mean(fln)) if fln else float("nan"),
            "flip_c0_to_c1": float(np.mean(f01)) if f01 else float("nan"),
            "flip_c1_to_c0": float(np.mean(f10)) if f10 else float("nan"),
            "intact_rate": float(np.mean(it)),
            "intact_flip": float(np.mean(itf))}


def pick_tier2_layers(tier1_rows, n_top=3, controls=(16, 24, 35)):
    """argmax layers by usage flip_norm at the largest c, plus fixed controls."""
    usage = [r for r in tier1_rows if r["dir"] == "usage"]
    if not usage:
        return sorted(set(controls))
    cmax = max(r["c"] for r in usage)
    sel = sorted([r for r in usage if r["c"] == cmax],
                 key=lambda r: -(r["flip_norm"] if not np.isnan(r["flip_norm"]) else -1))
    top = [int(r["layer"]) for r in sel[:n_top]]
    return sorted(set(top) | set(int(c) for c in controls))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    # (1) aggregate_cell exact on a hand-built cell
    recs = [
        {"y": 0, "m0": -1.0, "m1": +0.5, "intact": 1},   # correct alpha, flipped, intact
        {"y": 0, "m0": -1.0, "m1": -0.2, "intact": 1},   # correct alpha, no flip
        {"y": 0, "m0": +0.4, "m1": +1.0, "intact": 0},   # baseline-INcorrect alpha (excluded from norm)
        {"y": 1, "m0": +0.8, "m1": -0.3, "intact": 0},   # correct beta, flipped, broken output
        {"y": 1, "m0": -0.5, "m1": -1.0, "intact": 1},   # baseline-incorrect beta
    ]
    m = aggregate_cell(recs)
    assert m["n_targets"] == 5 and m["n_correct"] == 3
    assert abs(m["margin_flip"] - 2 / 5) < 1e-12
    assert abs(m["flip_norm"] - 2 / 3) < 1e-12
    assert abs(m["flip_c0_to_c1"] - 1 / 2) < 1e-12 and abs(m["flip_c1_to_c0"] - 1.0) < 1e-12
    assert abs(m["intact_flip"] - 1 / 5) < 1e-12
    # signed movement toward target: alpha rows s=+1, beta rows s=-1
    exp_dm = np.mean([+1.5, +0.8, +0.6, +1.1, +0.5])
    assert abs(m["mean_dmargin_toward"] - exp_dm) < 1e-12

    # (2) tier-2 layer picker
    rows = [{"dir": "usage", "c": 16.0, "layer": L,
             "flip_norm": fn} for L, fn in ((20, 0.9), (24, 0.8), (8, 0.7), (3, 0.1))]
    rows += [{"dir": "usage", "c": 4.0, "layer": 20, "flip_norm": 0.99}]
    sel = pick_tier2_layers(rows, n_top=2, controls=(16, 35))
    assert sel == [16, 20, 24, 35], sel
    print("[self_test] OK — cell aggregation (all 8 metrics), tier-2 layer selection pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    prompts = [json.loads(l) for l in open(args.corpus)]
    nP = len(prompts); assert nP == len(fams), "corpus/dump mismatch"
    concept = prompts[0].get("concept", "unknown")
    y = meta["y"].astype(int)
    m0 = meta["clean_margin"].astype(np.float64)
    n_layers = int(meta["n_layers"]); d = int(meta["d"])
    id0 = meta["id_class0"].astype(int) if "id_class0" in meta else np.full(nP, int(meta["alpha_id"]))
    id1 = meta["id_class1"].astype(int) if "id_class1" in meta else np.full(nP, int(meta["beta_id"]))
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    held = np.where(~trm)[0]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); norm_mod = _chain(model, "model.norm")
    last = n_layers - 1

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    def steer_eval(i, L, delta):
        inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
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

    # ---------- directions per layer (from the dump; GPU not touched) ----------
    def dirs_for_layer(L):
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        w = fisher_axis(H[trm], y[trm], args.shrink)
        u = unit_raw(G.mean(0))
        sigma = float(np.std(H[trm] @ w))
        dd = {"w_res": w, "usage": u}
        for k in range(args.n_shuffled):
            yp = y[trm].copy(); rng.shuffle(yp)
            dd[f"shuffled{k}"] = fisher_axis(H[trm], yp, args.shrink)
        for k in range(args.n_random):
            dd[f"random{k}"] = unit_raw(rng.standard_normal(d))
        return dd, sigma

    # ---------- target pools ----------
    correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
    def tier_targets(per_class):
        c0 = [i for i in held if y[i] == 0][:per_class]
        c1 = [i for i in held if y[i] == 1][:per_class]
        return c0 + c1                                  # 86-compatible (no correctness filter)
    t1_targets = tier_targets(args.t1_per_class)
    t2_targets = [int(i) for i in held if correct[i]]   # full baseline-correct pool

    def run_tier(name, layers, c_grid, targets, dir_filter=None):
        rows = []
        total = len(layers) * len(c_grid) * len(targets)
        logger.info("%s: %d layers x c=%s x %d targets (x dirs) ...", name, len(layers), c_grid, len(targets))
        done = 0
        for L in layers:
            dd, sigma = dirs_for_layer(L)
            if dir_filter:
                dd = {k: v for k, v in dd.items() if k in dir_filter}
            for c in c_grid:
                for dname, vec in dd.items():
                    recs = []
                    for i in targets:
                        s = +1.0 if y[i] == 0 else -1.0
                        m1, intact = steer_eval(i, L, (s * c * sigma) * unit_raw(vec))
                        recs.append({"y": int(y[i]), "m0": m0[i], "m1": m1, "intact": intact})
                    cell = aggregate_cell(recs)
                    cell.update({"layer": int(L), "c": float(c), "dir": dname, "sigma": sigma})
                    rows.append(cell)
                done += len(targets)
            lu = [r for r in rows if r["layer"] == L and r["dir"] == "usage"]
            lw = [r for r in rows if r["layer"] == L and r["dir"] == "w_res"]
            logger.info("  L%02d done | usage flip_norm by c: %s | w_res: %s | intact_rate(usage,max c)=%.2f",
                        L, {r["c"]: round(r["flip_norm"], 2) for r in lu},
                        {r["c"]: round(r["flip_norm"], 2) for r in lw},
                        (lu[-1]["intact_rate"] if lu else float("nan")))
        return rows

    fields = ["layer", "c", "dir", "sigma", "n_targets", "n_correct", "mean_dmargin_toward",
              "margin_flip", "flip_norm", "flip_c0_to_c1", "flip_c1_to_c0",
              "intact_rate", "intact_flip"]
    def wcsv(name, rows):
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields); w.writeheader()
            [w.writerow({k: r.get(k) for k in fields}) for r in rows]

    t1_rows = []
    if args.tier in ("1", "both"):
        t1_rows = run_tier("tier1", list(range(n_layers)), args.t1_c_grid, t1_targets)
        wcsv("steering_sweep_tier1.csv", t1_rows)
        logger.info("tier1 written: steering_sweep_tier1.csv (%d cells)", len(t1_rows))

    if args.tier in ("2", "both"):
        if args.tier2_layers:
            t2_layers = args.tier2_layers
        else:
            src = t1_rows
            if not src and args.tier1_csv and Path(args.tier1_csv).exists():
                with open(args.tier1_csv) as f:
                    src = [{**r, "c": float(r["c"]), "layer": int(r["layer"]),
                            "flip_norm": float(r["flip_norm"]) if r["flip_norm"] not in ("", "nan") else float("nan")}
                           for r in _csv.DictReader(f)]
            t2_layers = pick_tier2_layers(src, controls=tuple(args.tier2_controls))
        logger.info("tier2 layers: %s | full correct held pool: %d targets", t2_layers, len(t2_targets))
        t2_rows = run_tier("tier2", t2_layers, args.t2_c_grid, t2_targets,
                           dir_filter={"w_res", "usage", "shuffled0", "random0"})
        wcsv("steering_sweep_tier2.csv", t2_rows)
        logger.info("tier2 written: steering_sweep_tier2.csv (%d cells)", len(t2_rows))

    json.dump({"concept": concept, "n_prompts": nP,
               "baseline_accuracy": float(correct.mean()),
               "t1_targets": len(t1_targets), "t2_targets": len(t2_targets)},
              open(out / "sweep_summary.json", "w"), indent=2)
    print("\n" + "=" * 88)
    print(f"B1 STEERING SWEEP — {concept}")
    print("=" * 88)
    print(f"outputs in {out}: steering_sweep_tier1.csv / steering_sweep_tier2.csv")
    print("next: python 116_flip_law_retro.py --dump_dir <dump> "
          "--measured_86 <tier1 csv> (calibration + dose-response join)")
    print("=" * 88 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", help="rendered corpus jsonl from 120")
    p.add_argument("--dump_dir", help="field dump from 119 on this corpus")
    p.add_argument("--out_dir", default=None, help="default: <dump_dir>/../steering_sweep_<concept>")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tier", choices=["1", "2", "both"], default="both")
    p.add_argument("--t1_c_grid", type=float, nargs="*", default=[1, 4, 16])
    p.add_argument("--t2_c_grid", type=float, nargs="*", default=[0.5, 1, 2, 4, 8, 16, 32])
    p.add_argument("--t1_per_class", type=int, default=40)
    p.add_argument("--tier2_layers", type=int, nargs="*", default=None)
    p.add_argument("--tier2_controls", type=int, nargs="*", default=[16, 24, 35])
    p.add_argument("--tier1_csv", default=None, help="reuse an existing tier1 CSV for layer pick")
    p.add_argument("--n_shuffled", type=int, default=2)
    p.add_argument("--n_random", type=int, default=2)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.corpus and args.dump_dir, "--corpus and --dump_dir required"
    if args.out_dir is None:
        args.out_dir = str(Path(args.dump_dir).parent / "steering_sweep")
    run_real(args)


if __name__ == "__main__":
    main()
