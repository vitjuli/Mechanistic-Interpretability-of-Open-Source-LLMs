"""
66_carrier_vs_wres_geometry.py   [JOB 2 / 4]
===================================================================
63 asked whether the carrier projects onto gbar (it did not). But 64 showed gbar
is the WRONG axis; the true residual concept axis is w_res. So the meaningful
question is: does the carrier project onto w_res -- the axis that actually
separates the classes in the residual stream?

This re-runs the 63 convergence test using w_res (from 64 / 65) instead of gbar:
for each carrier decoder direction d_f, the signed causal cosine cos_C(d_f, w_res),
its sign-concentration, magnitude vs a random-direction null, and the carrier
subspace's capture of w_res. It answers, at both outcomes of 65:

  * If 65 says w_res is CAUSAL: does the carrier write toward the causal axis?
    (carrier-capture high => carrier IS a mechanism along w_res after all;
     low => carrier is a correlate even of the true axis.)
  * If 65 says NOT causal: this still characterises how unrelated the carrier is
    to the only linearly-separating axis, strengthening the negative result.

Pure geometry: needs only w_res (npy from 64/65), Sigma_inv (60_ npz), and the
carrier decoder rows. No prompts, no forward pass.

SELF-TEST: python 66_carrier_vs_wres_geometry.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("carrier_vs_wres")
FID = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")


def whitener(Si):
    w, V = np.linalg.eigh(Si); w = np.clip(w, 0, None); return (V * np.sqrt(w)) @ V.T

def signed_cos(D, axis, Si):
    Dg = D @ Si @ axis
    dC = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, Si, D), 1e-30, None))
    aC = float(np.sqrt(axis @ Si @ axis))
    return Dg / (dC * aC)

def sign_conc(cs):
    s = np.sign(cs); s = s[s != 0]
    if s.size == 0:
        return {"abs_mean_sign": float("nan"), "maj_frac": float("nan")}
    maj = 1.0 if s.sum() >= 0 else -1.0
    return {"abs_mean_sign": float(abs(np.mean(s))), "maj_frac": float(np.mean(s == maj))}

def capture_cos(axis, basis, Si):
    A = whitener(Si); Bw = A @ basis.T; aw = A @ axis
    if Bw.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(Bw); proj = Q @ (Q.T @ aw)
    den = np.linalg.norm(aw)
    return float(np.linalg.norm(proj) / den) if den > 0 else 0.0

def parse_fid(f):
    m = FID.match(str(f).strip())
    if not m:
        raise ValueError(f"bad feature_id {f!r}")
    return int(m.group(1)), int(m.group(2))


def self_test():
    rng = np.random.default_rng(66); d = 200
    ev = np.concatenate([np.linspace(15, 3, 15), np.linspace(2, .05, d - 15)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); S = (Q * ev) @ Q.T; S = .5 * (S + S.T)
    Si = np.linalg.inv(S + 1e-3 * np.mean(np.diag(S)) * np.eye(d))
    A = whitener(Si); Ai = np.linalg.inv(A)
    wg = rng.standard_normal(d); wg /= np.linalg.norm(wg); w_res = Ai @ wg
    k = 12
    # carrier ALIGNED to w_res (each writes a same-sign component along it)
    B = rng.standard_normal((k, d)) @ A.T; B, _ = np.linalg.qr(B.T); B = B.T[:k]
    a = 0.3; conv = np.array([np.sqrt(1 - a * a) * B[i] + a * wg for i in range(k)]) @ Ai.T
    # carrier UNRELATED to w_res
    unrel = rng.standard_normal((k, d)) @ Ai.T
    print("\n--- SELF TEST -------------------------------------------------")
    for nm, D in [("aligned", conv), ("unrelated", unrel)]:
        cs = signed_cos(D, w_res, Si); sc = sign_conc(cs); cap = capture_cos(w_res, D, Si)
        print(f"  {nm:9s}: signed-mean={cs.mean():+.3f} maj={sc['maj_frac']:.2f} "
              f"|cos|={np.mean(np.abs(cs)):.3f} capture={cap:.3f}")
    cs_a = signed_cos(conv, w_res, Si); cs_u = signed_cos(unrel, w_res, Si)
    assert sign_conc(cs_a)["maj_frac"] > 0.9, "aligned carrier must be sign-concentrated on w_res"
    assert np.mean(np.abs(cs_a)) > 3 * np.mean(np.abs(cs_u)), "aligned |cos| must exceed unrelated"
    assert capture_cos(w_res, conv, Si) > capture_cos(w_res, unrel, Si), "aligned must capture more"
    print("\nALL SELF-TEST ASSERTIONS PASSED \n--------------------------------------------\n")


def resolve(args):
    feats = {}
    def add(L, fi): feats.setdefault(L, []).append(fi)
    if args.features:
        for t in args.features.split(","):
            L, fi = parse_fid(t); add(L, fi)
    elif args.feature_file:
        for ln in open(args.feature_file):
            ln = ln.strip()
            if ln:
                L, fi = parse_fid(ln); add(L, fi)
    elif args.cluster_labels:
        import pandas as pd
        cl = pd.read_csv(args.cluster_labels)
        if "feature_id" not in cl.columns:
            cl = cl.rename(columns={cl.columns[0]: "feature_id"})
        def nid(c):
            s = str(c).strip().lstrip("Cc"); return s[:-2] if s.endswith(".0") else s
        want = {nid(c) for c in args.clusters.split(",")} if args.clusters else None
        for fid, c in zip(cl["feature_id"].astype(str), cl[args.cluster_col].apply(nid)):
            if want is None or c in want:
                try:
                    L, fi = parse_fid(fid)
                except ValueError:
                    continue
                if args.layers and L not in args.layers:
                    continue
                add(L, fi)
    else:
        raise SystemExit("need --features/--feature_file/--cluster_labels")
    return feats


def run_real(args):
    import torch
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cd = np.load(args.concept_npz); Si = cd["Sigma_inv"].astype(np.float64); d = Si.shape[0]
    w_res = np.load(args.wres_npy).astype(np.float64); w_res = w_res / (np.linalg.norm(w_res) + 1e-30)

    feats = resolve(args); layers = sorted(feats)
    from src.transcoder import load_transcoder_set
    null_layers = args.null_layers or list(range(args.null_min, args.null_max + 1))
    tset = load_transcoder_set(model_size=args.model_size, device=args.device, lazy_load=True,
                               layers=sorted(set(layers) | set(null_layers)))

    # W_dec cache (eliminates slow _get_decoder_vectors; mandatory for null_band)
    W_dec_cache: Dict[int, np.ndarray] = {}
    def _ensure_wdec(L):
        if L not in W_dec_cache:
            tc = tset[L]
            if not hasattr(tc, "W_dec"):
                raise RuntimeError(f"transcoder L{L} has no W_dec")
            logger.info("Caching W_dec for layer %d (shape=%s)...", L, tc.W_dec.shape)
            W_dec_cache[L] = tc.W_dec.detach().float().cpu().numpy().astype(np.float64)
            logger.info("  cached %.1f MB", W_dec_cache[L].nbytes / 1e6)
        return W_dec_cache[L]

    def rows(L, idx):
        W = _ensure_wdec(L)
        return W[np.asarray(idx, dtype=np.int64)]

    _pool_done = {"v": False}
    def null_band(k, n):
        if not _pool_done["v"]:
            logger.info("Pre-caching W_dec for %d null layers...", len(null_layers))
            for L in null_layers:
                _ensure_wdec(L)
            _pool_done["v"] = True
        majs, absm = [], []
        pool = null_layers
        logger.info("  null band: %d draws × %d directions...", n, k)
        for di in range(n):
            if di > 0 and di % 50 == 0:
                logger.info("    null draw %d/%d", di, n)
            R = np.empty((k, Si.shape[0]), dtype=np.float64)
            for slot in range(k):
                L = pool[rng.integers(len(pool))]
                W = W_dec_cache[L]
                R[slot] = W[int(rng.integers(W.shape[0]))]
            cs = signed_cos(R, w_res, Si)
            majs.append(sign_conc(cs)["maj_frac"]); absm.append(float(np.mean(np.abs(cs))))
        def b(x):
            a = np.array(x); return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}
        return b(majs), b(absm)

    res = {"wres_npy": str(args.wres_npy), "layers": {}, "cross_layer": None}
    for L in layers:
        D = rows(L, feats[L]); cs = signed_cos(D, w_res, Si)
        sc = sign_conc(cs); cap = capture_cos(w_res, D, Si)
        mb, ab = null_band(len(feats[L]), args.n_null) if len(feats[L]) >= 2 else (None, None)
        res["layers"][str(L)] = {"k": len(feats[L]), "signed_mean": float(cs.mean()),
                                 "maj_frac": sc["maj_frac"], "mean_abs_cos": float(np.mean(np.abs(cs))),
                                 "carrier_capture_wres": cap,
                                 "null_maj_frac": mb, "null_mean_abs_cos": ab}
        logger.info("L%d: signed-mean=%+.3f maj=%.2f (null~%.2f) |cos|=%.3f (null~%.3f) capture=%.3f",
                    L, cs.mean(), sc["maj_frac"], mb["mean"] if mb else float("nan"),
                    np.mean(np.abs(cs)), ab["mean"] if ab else float("nan"), cap)
    allD = np.vstack([rows(L, feats[L]) for L in layers])
    cs = signed_cos(allD, w_res, Si)
    res["cross_layer"] = {"k": allD.shape[0], "signed_mean": float(cs.mean()),
                          "maj_frac": sign_conc(cs)["maj_frac"],
                          "mean_abs_cos": float(np.mean(np.abs(cs))),
                          "carrier_capture_wres": capture_cos(w_res, allD, Si)}
    with open(out / "carrier_vs_wres.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\n" + "=" * 76)
    print("CARRIER vs TRUE AXIS w_res  (63-test on the right axis)")
    print("=" * 76)
    print(f"{'layer':>6} {'k':>4} {'signed_mean':>12} {'maj_frac':>9} {'|cos|':>7} {'capture_wres':>13}")
    for L in layers:
        r = res["layers"][str(L)]
        print(f"{L:>6} {r['k']:>4} {r['signed_mean']:>+12.3f} {r['maj_frac']:>9.2f} "
              f"{r['mean_abs_cos']:>7.3f} {r['carrier_capture_wres']:>13.3f}")
    cl = res["cross_layer"]
    print(f"{'ALL':>6} {cl['k']:>4} {cl['signed_mean']:>+12.3f} {cl['maj_frac']:>9.2f} "
          f"{cl['mean_abs_cos']:>7.3f} {cl['carrier_capture_wres']:>13.3f}")
    print(f"\nwrote: {out}/carrier_vs_wres.json")
    print("=" * 76)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--wres_npy", default="data/analysis/runD_v2/geometry_stage1/w_res_final.npy",
                   help="true residual axis from 64 (w_res_final.npy) or 65 (w_res65_final.npy)")
    p.add_argument("--features", default=None)
    p.add_argument("--feature_file", default=None)
    p.add_argument("--cluster_labels", default=None)
    p.add_argument("--cluster_col", default="coimp_louvain")
    p.add_argument("--clusters", default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--d_tc", type=int, default=163840)
    p.add_argument("--n_null", type=int, default=200)
    p.add_argument("--null_layers", type=int, nargs="*", default=None)
    p.add_argument("--null_min", type=int, default=10)
    p.add_argument("--null_max", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
