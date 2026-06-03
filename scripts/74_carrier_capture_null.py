"""
74_carrier_capture_null.py   [CSD3 / GPU-or-CPU, mirrors 66's I/O]
===================================================================
Closes "Hole 3": exp 66 reports the carrier's COLLECTIVE capture of w_res = 0.303
(227 features span ~30% of the concept axis in the causal metric). A skeptic says
"so the concept IS partly in the dictionary." This script asks the right question:

   do 227 RANDOM transcoder features -- matched per-layer to the carrier's counts --
   capture w_res just as much?

If carrier-capture sits inside the random-feature null band, then 0.303 is NULL-LEVEL:
the co-importance carrier captures w_res no better than the same number of randomly
chosen features from the same layers. The dictionary does not contain the concept
axis in any privileged way; 0.30 is a dimensional artefact (k features in d dims span
~sqrt(k/d) of any vector). If carrier-capture is ABOVE the band, 0.30 is meaningful
and the framing must change.

WHY MATCHED PER-LAYER: the carrier draws features from 16 layers with specific counts
(L10:8, ..., L24:20, L25:12). The null samples the SAME counts from the SAME layers,
so the only difference from the carrier is WHICH features (co-importance-selected vs
random), not the layer distribution. This isolates the selection effect.

Capture is computed by the IDENTICAL routine as 66 (capture_cos): whiten by Sigma^-1/2,
QR the decoder-dir span, report ||proj(w_res)|| / ||w_res|| in the whitened space.

DIMENSIONAL EXPECTATION (stated for the writeup): for isotropic directions,
E[capture] = sqrt(k/d) = sqrt(227/2560) = 0.298. The carrier's 0.303 already matches
this; the real-feature null tells us whether trained decoder dirs (non-isotropic)
shift it.

INPUTS (same as 66):
  --geom_npz   concept_directions.npz   (Sigma_inv)
  --wres_npy   w_res_final.npy          (concept axis; or w_res65_final.npy)
  --carrier    carrier feature list (json/txt of "L{layer}_F{feat}") for per-layer counts
  transcoder W_dec for the carrier's layers (loaded via the project's TranscoderSet)

OUTPUT:
  carrier_capture_null.json   { carrier_capture, null_mean, null_p95, null_p99,
                                percentile_of_carrier, verdict, per_layer_counts }

SELF-TEST: python 74_carrier_capture_null.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("capture_null")
FID = re.compile(r"[Ll](\d+)[_-]?[Ff](\d+)")


# ---- geometry (identical to 66) -------------------------------------------------
def whitener(Si):
    w, V = np.linalg.eigh(Si)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def capture_cos(axis, basis, Si):
    A = whitener(Si); Bw = A @ basis.T; aw = A @ axis
    if Bw.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(Bw); proj = Q @ (Q.T @ aw)
    den = np.linalg.norm(aw)
    return float(np.linalg.norm(proj) / den) if den > 0 else 0.0


def parse_fid(s):
    m = FID.search(str(s).strip())
    if not m:
        raise ValueError(f"bad feature id {s!r}")
    return int(m.group(1)), int(m.group(2))


def load_carrier_counts(path):
    """Return {layer: n_features} from a carrier list (json list, or lines 'L18_F1234')."""
    p = Path(path)
    txt = p.read_text()
    ids = []
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            obj = obj.get("features") or obj.get("carrier") or []
        ids = [str(x) for x in obj]
    except json.JSONDecodeError:
        ids = [ln for ln in txt.splitlines() if ln.strip()]
    counts = {}
    for s in ids:
        L, _ = parse_fid(s)
        counts[L] = counts.get(L, 0) + 1
    return counts


# ---- self-test (planted: carrier aligned vs random; random ~ sqrt(k/d)) ---------
def self_test():
    rng = np.random.default_rng(74); d = 256
    ev = np.concatenate([np.linspace(12, 3, 12), np.linspace(2, .05, d - 12)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); Si = (Q * ev) @ Q.T; Si = .5 * (Si + Si.T)
    Ai = whitener(np.linalg.inv(Si))
    wg = rng.standard_normal(d); wg /= np.linalg.norm(wg); w_res = Ai @ wg

    k = 60
    # pool of "decoder dirs": mostly unrelated, isotropic
    pool = rng.standard_normal((4000, d))
    # random-feature null
    null = []
    for _ in range(150):
        idx = rng.choice(len(pool), k, replace=False)
        null.append(capture_cos(w_res, pool[idx], Si))
    null = np.array(null)
    # an ALIGNED carrier: k dirs each carrying a w_res component -> should capture MORE
    aligned = rng.standard_normal((k, d)) * 0.5 + w_res / np.linalg.norm(w_res)
    cap_aligned = capture_cos(w_res, aligned, Si)
    # a RANDOM carrier (same as null draws) -> should sit in band
    cap_rand = capture_cos(w_res, pool[rng.choice(len(pool), k, replace=False)], Si)

    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  random-feature null: mean={null.mean():.3f}  p95={np.percentile(null,95):.3f}  "
          f"(isotropic sqrt(k/d)={np.sqrt(k/d):.3f})")
    print(f"  ALIGNED carrier capture = {cap_aligned:.3f}  (expect >> null)")
    print(f"  RANDOM carrier capture  = {cap_rand:.3f}  (expect in band)")
    assert cap_aligned > np.percentile(null, 95), "aligned carrier must beat the null band"
    assert abs(null.mean() - np.sqrt(k / d)) < 0.05, "isotropic null must match sqrt(k/d)"
    assert cap_rand < np.percentile(null, 99) + 0.05, "random carrier must sit in band"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(null distinguishes an aligned carrier from random features from the same pool)")
    print("---------------------------------------------------------------\n")


# ---- real run (needs W_dec via project loader; mirrors 66 exactly) --------------
def run_real(args):
    import torch
    from src.transcoder import load_transcoder_set            # same import as 66

    geom = np.load(args.geom_npz)
    Si = geom["Sigma_inv"].astype(np.float64)
    w_res = np.load(args.wres_npy).astype(np.float64); w_res /= (np.linalg.norm(w_res) + 1e-30)

    counts = load_carrier_counts(args.carrier)                # {layer: n_features}
    car_ids = load_carrier_feature_ids(args.carrier)          # [(layer, feat), ...]
    layers = sorted(counts)
    k_total = sum(counts.values())
    logger.info("carrier: %d features across %d layers %s", k_total, len(layers), counts)

    tset = load_transcoder_set(model_size=args.model_size, device=args.device,
                               lazy_load=True, layers=layers)

    def rows(L, idx):                                         # exact 66 accessor
        return (tset[L]._get_decoder_vectors(torch.tensor(idx, dtype=torch.long))
                .detach().float().cpu().numpy().astype(np.float64))

    def d_tc(L):
        return tset[L].W_dec.shape[0] if hasattr(tset[L], "W_dec") and tset[L].W_dec is not None else args.d_tc

    # carrier capture (reproduce 66's 0.303)
    by_layer = {}
    for (L, f) in car_ids:
        by_layer.setdefault(L, []).append(f)
    car_rows = np.vstack([rows(L, by_layer[L]) for L in layers])
    carrier_capture = capture_cos(w_res, car_rows, Si)
    logger.info("carrier collective capture (reproduced) = %.4f  (66 reported 0.303)", carrier_capture)

    # null: pre-sample all random indices per layer, ONE slice read per layer
    rng = np.random.default_rng(args.seed)
    sampled = {}                                              # L -> (n_null, counts[L], d)
    for L in layers:
        n = d_tc(L)
        idx = rng.integers(0, n, size=(args.n_null, counts[L]))
        vecs = rows(L, idx.reshape(-1))                       # (n_null*counts[L], d) in one read
        sampled[L] = vecs.reshape(args.n_null, counts[L], vecs.shape[-1])
        logger.info("sampled random features L%d: %d draws x %d feats (d_tc=%d)", L, args.n_null, counts[L], n)

    null = np.empty(args.n_null)
    for j in range(args.n_null):
        stacked = np.vstack([sampled[L][j] for L in layers])  # (227, d)
        null[j] = capture_cos(w_res, stacked, Si)
    logger.info("null done: mean=%.4f p95=%.4f", null.mean(), np.percentile(null, 95))

    pctile = float((null < carrier_capture).mean() * 100)
    res = {
        "carrier_capture": carrier_capture,
        "isotropic_sqrt_k_over_d": float(np.sqrt(k_total / w_res.shape[0])),
        "null_mean": float(null.mean()), "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)), "null_p99": float(np.percentile(null, 99)),
        "null_min": float(null.min()), "null_max": float(null.max()),
        "percentile_of_carrier": pctile, "per_layer_counts": counts, "n_null": args.n_null,
        "verdict": (
            f"NULL-LEVEL: carrier capture {carrier_capture:.3f} sits at the {pctile:.0f}th percentile "
            f"of the matched random-feature null (mean {null.mean():.3f}, p95 {np.percentile(null,95):.3f}). "
            "The co-importance carrier captures w_res no better than random features from the same "
            "layers; the dictionary does not contain the concept axis in any privileged way. The 0.30 "
            "is a dimensional artefact (~sqrt(k/d))."
            if carrier_capture <= np.percentile(null, 95) else
            f"ABOVE NULL: carrier capture {carrier_capture:.3f} exceeds the random-feature null p95 "
            f"({np.percentile(null,95):.3f}) at the {pctile:.0f}th percentile. The carrier captures w_res "
            "more than random features from the same layers; 0.30 is meaningful -- reframe accordingly.")}

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "carrier_capture_null.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 76)
    print("CARRIER CAPTURE vs MATCHED RANDOM-FEATURE NULL  (Hole 3)")
    print("=" * 76)
    print(f"  carrier collective capture : {carrier_capture:.4f}")
    print(f"  isotropic sqrt(k/d)        : {res['isotropic_sqrt_k_over_d']:.4f}")
    print(f"  random-feature null        : mean {null.mean():.4f}  p95 {np.percentile(null,95):.4f}  "
          f"p99 {np.percentile(null,99):.4f}")
    print(f"  carrier at percentile      : {pctile:.0f}%")
    print("\nVERDICT: " + res["verdict"])
    print(f"\nwrote: {args.out_dir}/carrier_capture_null.json")
    print("=" * 76)


def load_carrier_feature_ids(path):
    p = Path(path); txt = p.read_text()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            obj = obj.get("features") or obj.get("carrier") or []
        ids = [str(x) for x in obj]
    except json.JSONDecodeError:
        ids = [ln for ln in txt.splitlines() if ln.strip()]
    return [parse_fid(s) for s in ids]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--geom_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--wres_npy", default="data/analysis/runD_v2/geometry_stage1/w_res_final.npy")
    p.add_argument("--carrier", default="data/analysis/runD_v2/geometry_stage1/carrier_features.json")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--n_null", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_size", default="4b", choices=["0.6b", "1.7b", "4b", "8b", "14b"])
    p.add_argument("--d_tc", type=int, default=163840, help="transcoder dict size fallback")
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
