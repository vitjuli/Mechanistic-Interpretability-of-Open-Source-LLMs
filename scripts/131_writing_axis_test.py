"""
131_writing_axis_test.py   [does delta deserve the name 'writing axis'? decoded != written != used]
===================================================================================================
We have been calling delta = unit(mean_beta(h) - mean_alpha(h)) the "writing axis".
But that is the class-difference of the residual STATE (accumulated), not of what
components WRITE. This script introduces the GENUINE writing direction and compares
all four objects, settling the name and measuring the decoded != written != used
framework directly.

At each layer the residual stream receives additive writes:
    h[L+1] = h[L] + attn_out[L] + mlp_out[L]
The TRUE per-layer writing direction for the concept is the class-difference of what
the components ADD at that layer:
    delta_write[L]      = unit( mean_beta(attn_out+mlp_out) - mean_alpha(attn_out+mlp_out) )
    delta_write_mlp[L]  = unit( mean_beta(mlp_out)         - mean_alpha(mlp_out) )
    delta_write_attn[L] = unit( mean_beta(attn_out)        - mean_alpha(attn_out) )
all measured at the decision (last) position, on held-out families.

We compare delta_write against the three STATE axes (recomputed from the existing dump,
same definitions as script 127):
    delta_state = unit(mean_beta(h) - mean_alpha(h))   the OLD "delta" (residual center-diff)
    w_res       = Fisher/LDA(h, y)                     the READING axis
    u           = unit(mean grad)                      the USING axis (output sensitivity)

Reported per layer (and summarized), with a random-direction null p95 for every cosine:
    cos(delta_write, delta_state)  -> is the old "delta" what is actually written?
    cos(delta_write, w_res)        -> do components write along the readable axis?
    cos(delta_write, u)            -> do components write along the USED axis? (the key one)
plus the WRITE MAGNITUDE per component (||class-diff of mlp/attn write||) -> who writes.

VERDICTS:
  - cos(delta_write, delta_state) HIGH (>> null) -> the old "delta" tracks what is written;
    calling it (a proxy for) the writing axis is defensible. Keep the name, note it's a
    state-side proxy for the true delta_write.
  - cos(delta_write, delta_state) ~ null -> the old "delta" is an emergent center-diff, NOT
    what components write; rename it "separation axis" and use delta_write as the writing axis.
  - cos(delta_write, u) LOW while cos(delta_write, w_res) HIGH -> components WRITE a READABLE
    concept the model does NOT USE: decoded != written, and written != used. Direct evidence
    for the four-way dissociation (this is the headline this script can produce).

Needs the model (forward pass + hooks on attn/mlp outputs). GPU. Loads u/res from the
existing field dump (same corpus/order) and captures component writes fresh.

SELF-TEST (no torch / no repo):  python 131_writing_axis_test.py --self_test
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
logger = logging.getLogger("write131")


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


def class_diff_axis(M, y):
    """unit(mean over class1 - mean over class0) of rows of M."""
    return unit_raw(M[y == 1].mean(0) - M[y == 0].mean(0))


def abs_cos(a, b):
    return float(abs(unit_raw(a) @ unit_raw(b)))


def random_cos_null_p95(d, vhat, rng, n=500):
    """p95 of |cos(random_unit, vhat)| in d dimensions."""
    vhat = unit_raw(vhat)
    cs = [abs(float(unit_raw(rng.standard_normal(d)) @ vhat)) for _ in range(n)]
    return float(np.quantile(cs, 0.95))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 60, 200
    y = (np.arange(n) % 2).astype(int)

    # construct component writes whose class-difference points along a KNOWN direction w_dir
    w_dir = unit_raw(rng.standard_normal(d))
    attn = np.outer(2 * y - 1.0, 0.5 * w_dir) + 0.05 * rng.standard_normal((n, d))
    mlp = np.outer(2 * y - 1.0, 1.0 * w_dir) + 0.05 * rng.standard_normal((n, d))
    dwrite = class_diff_axis(attn + mlp, y)
    assert abs_cos(dwrite, w_dir) > 0.99, "write class-diff recovers the write direction"

    # state = cumulative; if state center-diff is along w_dir too, cos high
    state = np.outer(2 * y - 1.0, w_dir) + 0.1 * rng.standard_normal((n, d))
    dstate = class_diff_axis(state, y)
    assert abs_cos(dwrite, dstate) > 0.9, "aligned state matches write"

    # a state center-diff along an ORTHOGONAL direction -> low cos (the 'rename' case)
    o_dir = rng.standard_normal(d); o_dir -= (o_dir @ w_dir) * w_dir; o_dir = unit_raw(o_dir)
    state_o = np.outer(2 * y - 1.0, o_dir) + 0.1 * rng.standard_normal((n, d))
    dstate_o = class_diff_axis(state_o, y)
    assert abs_cos(dwrite, dstate_o) < 0.3, "orthogonal state must NOT match write"

    # null calibration: random cosine in high-d is small
    p95 = random_cos_null_p95(d, dwrite, rng, n=300)
    assert p95 < 0.4, f"random-direction null p95 should be modest in d={d}: {p95}"
    assert abs_cos(dwrite, w_dir) > p95, "true alignment beats null"

    # who-writes magnitude: mlp write magnitude > attn here (1.0 vs 0.5 loading)
    mag_mlp = np.linalg.norm(mlp[y == 1].mean(0) - mlp[y == 0].mean(0))
    mag_attn = np.linalg.norm(attn[y == 1].mean(0) - attn[y == 0].mean(0))
    assert mag_mlp > mag_attn, "magnitude ranks the writing component"

    # Fisher axis sanity
    w = fisher_axis(state, y); assert abs(np.linalg.norm(w) - 1.0) < 1e-9
    print("[self_test] OK — write recovery, state-match high/low cases, null calibration, "
          "component magnitude, Fisher unit. pass.")


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

    prompts = [json.loads(l) for l in open(args.corpus)]
    fams = [p["surface_family"] for p in prompts]
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    y = meta["y"].astype(int)
    d_model = int(meta["d"])
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    held = ~trm
    n_layers = int(meta["n_layers"])
    layers = args.layers if args.layers else list(range(n_layers))

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers")

    # capture attn_out and mlp_out at the last position, per layer, per prompt.
    # Robust pattern: a shared mutable index holder the closures read at call time.
    attn_store = {L: np.zeros((len(prompts), d_model), np.float64) for L in layers}
    mlp_store = {L: np.zeros((len(prompts), d_model), np.float64) for L in layers}
    cur = {"i": 0}

    def mk_hook(L, store):
        def hook(mod, inp, output):
            t = output[0] if isinstance(output, tuple) else output
            store[L][cur["i"]] = t[0, -1, :].detach().float().cpu().numpy()
        return hook

    handles = []
    for L in layers:
        handles.append(blocks[L].self_attn.register_forward_hook(mk_hook(L, attn_store)))
        handles.append(blocks[L].mlp.register_forward_hook(mk_hook(L, mlp_store)))

    for i, p in enumerate(prompts):
        cur["i"] = i
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        with torch.no_grad():
            model(**inp, use_cache=False)
        if (i + 1) % 100 == 0:
            logger.info("captured %d/%d prompts", i + 1, len(prompts))

    for h in handles:
        h.remove()

    # load state axes inputs from the existing dump (same corpus/order)
    def res_at(L):
        return np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)

    def grad_u(L):
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        return unit_raw(G[held].mean(0))

    rows = []
    yo = y[held]
    for L in layers:
        H = res_at(L)[held]
        attn = attn_store[L][held]
        mlp = mlp_store[L][held]

        # writing axes
        dwrite = class_diff_axis(attn + mlp, yo)
        dwrite_mlp = class_diff_axis(mlp, yo)
        dwrite_attn = class_diff_axis(attn, yo)
        mag_mlp = float(np.linalg.norm(mlp[yo == 1].mean(0) - mlp[yo == 0].mean(0)))
        mag_attn = float(np.linalg.norm(attn[yo == 1].mean(0) - attn[yo == 0].mean(0)))

        # state axes
        dstate = class_diff_axis(H, yo)
        w_res = fisher_axis(H, yo, args.shrink)
        u = grad_u(L)

        p95 = random_cos_null_p95(d_model, dwrite, rng, n=args.n_null)
        rows.append({
            "layer": L,
            "cos_write_state": abs_cos(dwrite, dstate),
            "cos_write_wres": abs_cos(dwrite, w_res),
            "cos_write_u": abs_cos(dwrite, u),
            "cos_writeMLP_state": abs_cos(dwrite_mlp, dstate),
            "cos_writeMLP_u": abs_cos(dwrite_mlp, u),
            "cos_writeAttn_state": abs_cos(dwrite_attn, dstate),
            "cos_writeAttn_u": abs_cos(dwrite_attn, u),
            "cos_writeMLP_writeAttn": abs_cos(dwrite_mlp, dwrite_attn),
            "null_p95": p95,
            "mag_write_mlp": mag_mlp, "mag_write_attn": mag_attn,
            "mlp_attn_mag_ratio": mag_mlp / (mag_attn + 1e-30),
            # cross-check the three STATE axes against each other (context)
            "cos_state_wres": abs_cos(dstate, w_res),
            "cos_state_u": abs_cos(dstate, u),
            "cos_wres_u": abs_cos(w_res, u),
        })
        logger.info("L%02d | write-state=%.3f write-wres=%.3f write-u=%.3f (null %.3f) | "
                    "mlp/attn mag=%.2f | state-u=%.3f wres-u=%.3f",
                    L, rows[-1]["cos_write_state"], rows[-1]["cos_write_wres"],
                    rows[-1]["cos_write_u"], rows[-1]["null_p95"], rows[-1]["mlp_attn_mag_ratio"],
                    rows[-1]["cos_state_u"], rows[-1]["cos_wres_u"])

    with open(out / "writing_axis_test.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    # summaries over ignition band (where the concept is written)
    band = [r for r in rows if args.band_lo <= r["layer"] <= args.band_hi]
    def med(rs, k): return float(np.median([r[k] for r in rs])) if rs else float("nan")
    def above(rs, k): return sum(r[k] > r["null_p95"] for r in rs)

    print("\n" + "=" * 100)
    print("WRITING-AXIS TEST — does the old 'delta' track what components WRITE? (decoded≠written≠used)")
    print("=" * 100)
    print(f"ignition band L{args.band_lo}-{args.band_hi} medians (null p95 ~ {med(band,'null_p95'):.3f}):")
    print(f"  cos(write, OLD-delta/state) = {med(band,'cos_write_state'):.3f}   "
          f"[above null on {above(band,'cos_write_state')}/{len(band)} layers]")
    print(f"  cos(write, w_res / reading) = {med(band,'cos_write_wres'):.3f}   "
          f"[above null on {above(band,'cos_write_wres')}/{len(band)} layers]")
    print(f"  cos(write, u / using)       = {med(band,'cos_write_u'):.3f}   "
          f"[above null on {above(band,'cos_write_u')}/{len(band)} layers]")
    print(f"  MLP/attn write-magnitude ratio = {med(band,'mlp_attn_mag_ratio'):.2f} "
          f"(>1: MLP writes the concept; <1: attention writes it)")
    print(f"  context — cos(state,u)={med(band,'cos_state_u'):.3f}  cos(wres,u)={med(band,'cos_wres_u'):.3f} "
          f"(both should be ~null: read≠use)")
    print("\nVERDICTS:")
    print("  1) cos(write, OLD-delta) HIGH >> null -> the old 'delta' is a faithful state-side")
    print("     proxy for what's written. Keep calling it (a proxy for) the writing axis.")
    print("  1') cos(write, OLD-delta) ~ null -> the old 'delta' is an emergent center-diff, NOT")
    print("      what's written. Rename it 'separation axis'; use delta_write as THE writing axis.")
    print("  2) cos(write, w_res) HIGH but cos(write, u) ~ null -> components WRITE a READABLE")
    print("     concept the model does NOT USE: decoded≠written≠used, measured. Headline.")
    print(f"saved: {out/'writing_axis_test.csv'}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/B1_alpha_beta.jsonl")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/writing_axis")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default: all layers")
    p.add_argument("--band_lo", type=int, default=19)
    p.add_argument("--band_hi", type=int, default=24)
    p.add_argument("--n_null", type=int, default=500)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
