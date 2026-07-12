"""
155_realized_write_decomposition.py  [v4] MLP-vs-attention on the TRUE MLP output (no transcoder recon)
======================================================================================================
The transcoder does NOT reconstruct the class-difference well (rel-error ~0.87, verified on CSD3 -- not
our bug: no skip connection, truncation to active features loses <1%). So we do NOT use the transcoder
reconstruction for the main result. Instead we use the TRUE MLP output (captured by a forward hook,
already in the 154-v3 dump) and split each axis's layer-to-layer change into:
    MLP   = mlp_true_beta - mlp_true_alpha      (the REAL differential MLP write, exact)
    attn  = Delta_axis - MLP                    (attention contribution, exact remainder)
Two-way, exact: MLP + attn = Delta (no transcoder error term -- we never reconstruct).

Each part split into LENGTH (radial, SIGNED -- delta shrinks as well as grows) and ROTATION (projected
onto the ACTUAL rotation direction; high-dim safe).

MECHANICAL VALIDITY caveat (printed):
  * delta: the residual literally receives attn+mlp each layer, so Delta(delta) = attn_diff + mlp_diff
    is an EXACT MECHANICAL decomposition. "X% of delta's rotation is carried by the MLP write" is literal.
  * u, w_res: these are DERIVED directions (gradient / Fisher), NOT built additively from attn+mlp.
    The same projection is reported but it is a GEOMETRIC ALIGNMENT of the MLP residual-write with the
    axis change, not a mechanical cause. Interpret as correlation, not attribution.

Transcoder reliability is reported SEPARATELY (rel-error per layer) as a gauge; the per-feature
rotators/amplifiers breakdown is gated by it and printed only as a flagged appendix.

SELF-TEST: python 155_realized_write_decomposition.py --self_test
RUN:       python 155_realized_write_decomposition.py --dump <.../realized_writes> --out realized_<c>.csv
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("realized155")

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def fisher_dir(Sw, mu_diff, shrink=0.1):
    d = Sw.shape[0]
    Sw_s = (1 - shrink) * Sw + shrink * (np.trace(Sw) / d) * np.eye(d)
    return unit(np.linalg.solve(Sw_s, mu_diff))

def decompose(axis_prev, axis_cur, mlp_true):
    """split Delta = axis_cur-axis_prev into {length signed, rotation} x {MLP-true, attention}."""
    radial = unit(axis_prev)
    Delta = axis_cur - axis_prev
    len_change = float(Delta @ radial)
    Delta_perp = Delta - len_change * radial
    nperp = np.linalg.norm(Delta_perp)
    rot_dir = unit(Delta_perp)
    attn = Delta - mlp_true                       # exact remainder
    resid = np.linalg.norm((mlp_true + attn) - Delta)   # == 0 by construction
    return {"len_change": len_change, "rot_mag": nperp,
            "rot_frac": float(nperp / (np.linalg.norm(Delta) + 1e-12)),
            "len_mlp": float(mlp_true @ radial), "len_attn": float(attn @ radial),
            "rot_mlp": float(mlp_true @ rot_dir), "rot_attn": float(attn @ rot_dir),
            "invariant_resid": float(resid)}

def self_test():
    rng = np.random.default_rng(0); d = 200
    prev = rng.standard_normal(d); radial = unit(prev)
    rdir = rng.standard_normal(d); rdir -= (rdir @ radial) * radial; rdir = unit(rdir)
    cur = np.linalg.norm(prev) * (np.cos(0.3) * radial + np.sin(0.3) * rdir) * 1.4
    mlp_true = 0.6 * (cur - prev) + 0.05 * rng.standard_normal(d)
    r = decompose(prev, cur, mlp_true)
    assert r["invariant_resid"] < 1e-9, "MLP+attn must equal Delta exactly"
    assert r["len_change"] > 0
    cur2 = np.linalg.norm(prev) * radial * 0.5
    r2 = decompose(prev, cur2, np.zeros(d))
    assert r2["len_change"] < 0, "shrink -> negative length"
    # genuine Fisher with anisotropic cov differs from mean-diff
    Sw = np.diag(np.linspace(0.1, 5.0, 20)); mu = rng.standard_normal(20)
    w = fisher_dir(Sw, mu, 0.1)
    assert abs(unit(w) @ unit(mu)) < 0.999 and abs(w[0]) > abs(w[-1])
    print("[self_test] OK - MLP+attn=Delta exact, signed length, genuine Fisher w_res. pass.")

def build_axis(dump, L, which):
    return (np.load(dump / f"{which}_alpha_L{L:02d}.npy").astype(np.float64),
            np.load(dump / f"{which}_beta_L{L:02d}.npy").astype(np.float64))

def run_real(args):
    dump = Path(args.dump)
    layers = sorted(int(p.stem.split("L")[-1]) for p in dump.glob("res_beta_L*.npy"))
    if len(layers) < 2: raise SystemExit(f"need >=2 layers in {dump}")
    meta = json.load(open(dump / "dump_meta.json")) if (dump / "dump_meta.json").exists() else {}
    shrink = float(meta.get("shrink", 0.1))

    axes = {"delta": {}, "u": {}, "wres": {}}; mlp_true_d = {}; wres_genuine = True
    for L in layers:
        ra, rb = build_axis(dump, L, "res"); ga, gb = build_axis(dump, L, "grad"); ma, mb = build_axis(dump, L, "mlp_true")
        axes["delta"][L] = rb - ra
        axes["u"][L] = (ga + gb) / 2.0
        swp = dump / f"Sw_L{L:02d}.npy"
        if swp.exists():
            axes["wres"][L] = fisher_dir(np.load(swp).astype(np.float64), rb - ra, shrink)
        else:
            axes["wres"][L] = unit(rb - ra); wres_genuine = False
        mlp_true_d[L] = mb - ma

    # transcoder reliability gauge (reconstruction vs TRUE MLP) -- NOT used in the split
    tc_rel = {}
    for L in layers:
        try:
            Wdec = np.load(dump / f"W_dec_active_L{L:02d}.npy").astype(np.float64)
            da = (np.load(dump / f"act_beta_L{L:02d}.npy").astype(np.float64)
                  - np.load(dump / f"act_alpha_L{L:02d}.npy").astype(np.float64))
            recon = (da[:, None] * Wdec).sum(0)
            tc_rel[L] = float(np.linalg.norm(mlp_true_d[L] - recon) / (np.linalg.norm(mlp_true_d[L]) + 1e-12))
        except FileNotFoundError:
            tc_rel[L] = float("nan")

    print("\n" + "=" * 100)
    print("MLP (true) vs ATTENTION: change of each axis -> {length signed, rotation} (transcoder NOT used)")
    print("=" * 100)
    rows = []
    for ax in ("delta", "u", "wres"):
        valid = "EXACT mechanical (residual = attn+mlp)" if ax == "delta" else \
                "GEOMETRIC alignment only (derived axis, NOT attn+mlp additive) -- read as correlation"
        extra = "" if (ax != "wres" or wres_genuine) else "  [WARNING: Sw missing, w_res fell back to delta]"
        print(f"\n  AXIS = {ax}   [{valid}]{extra}")
        print(f"  {'L':>3} {'dLen':>8} {'rotMag':>7} {'rotFrac':>7} | {'len:MLP':>8} {'len:attn':>9} | "
              f"{'rot:MLP':>8} {'rot:attn':>9}")
        for L in layers:
            if (L - 1) not in axes[ax]: continue
            d = decompose(axes[ax][L - 1], axes[ax][L], mlp_true_d[L])
            rows.append({"axis": ax, "layer": L, **d})
            print(f"  {L:>3} {d['len_change']:>+8.3f} {d['rot_mag']:>7.3f} {d['rot_frac']:>7.3f} | "
                  f"{d['len_mlp']:>+8.3f} {d['len_attn']:>+9.3f} | {d['rot_mlp']:>+8.3f} {d['rot_attn']:>+9.3f}")
        rz = [r for r in rows if r["axis"] == ax and r["rot_frac"] > 0.3]
        if rz:
            def share(part, comp):
                num = np.mean([abs(r[f"{comp}_{part}"]) for r in rz])
                den = np.mean([abs(r[f"{comp}_mlp"]) + abs(r[f"{comp}_attn"]) for r in rz]) + 1e-12
                return num / den
            print(f"    rotation: MLP {share('mlp','rot')*100:.0f}%  attention {share('attn','rot')*100:.0f}%   "
                  f"|  length: MLP {share('mlp','len')*100:.0f}%  attention {share('attn','len')*100:.0f}%   "
                  f"(avg over rotation-zone layers)")

    inv = max((r["invariant_resid"] for r in rows), default=0.0)
    print(f"\n  invariant max||MLP+attn - Delta|| = {inv:.2e} (exact two-way split)")
    tcvals = [v for v in tc_rel.values() if v == v]
    print(f"\n  TRANSCODER RELIABILITY (separate gauge, NOT used above): mean rel-error = {np.mean(tcvals):.3f}")
    print(f"    (high => transcoder doesn't reconstruct the class-difference; per-feature rotator/amplifier")
    print(f"     breakdown via the transcoder is therefore UNRELIABLE here and is omitted. Cause not")
    print(f"     established -- possibly OOD answer-position or low between-prompt variance of the concept signal.)")

    out = Path(args.out)
    with open(out, "w") as f:
        f.write("axis,layer,len_change,rot_mag,rot_frac,len_mlp,len_attn,rot_mlp,rot_attn,tc_rel_error\n")
        for r in rows:
            f.write(f"{r['axis']},{r['layer']},{r['len_change']:.5f},{r['rot_mag']:.5f},{r['rot_frac']:.5f},"
                    f"{r['len_mlp']:.5f},{r['len_attn']:.5f},{r['rot_mlp']:.5f},{r['rot_attn']:.5f},"
                    f"{tc_rel.get(r['layer'], float('nan')):.5f}\n")
    logger.info("wrote %s", out); print("=" * 100 + "\n")

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true"); p.add_argument("--dump"); p.add_argument("--out", default="realized.csv")
    return p

def main():
    args = build_parser().parse_args()
    if args.self_test: self_test(); return
    if not args.dump: raise SystemExit("need --dump")
    run_real(args)

if __name__ == "__main__":
    main()
