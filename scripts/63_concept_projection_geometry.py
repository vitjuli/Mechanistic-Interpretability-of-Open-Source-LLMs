"""
63_concept_projection_geometry.py
===================================================================
ARBITER for the CONVERGENCE hypothesis.

62_carrier_gram_spectrum.py refuted uniform redundancy: the carrier's decoder
directions are nearly ORTHOGONAL to each other (mean |cos_C| ~ 0.04, PR/k ~ 0.93,
statistically like random unrelated directions). Yet earlier experiments showed
graded-patching saturation (one feature ~ full IIA) and per-pair Jaccard 0.92-0.99
(the same pairs flip). Those two facts look incompatible.

The convergence hypothesis reconciles them:

  the k write-directions d_f are mutually ~orthogonal (different inputs), BUT
  each has a non-trivial, SAME-SIGN projection onto the single read-axis gbar
  (= the alpha/beta concept direction). I.e. many orthogonal write-directions
  -> one scalar read-out. Not "one axis copied k times" (redundancy, refuted)
  and not "k independent functions" (naive combinatorial); a MANY-TO-ONE map.

This script measures, in the causal inner product, whether that is true.

PREDICTIONS the convergence hypothesis must pass (and the controls that falsify):
  (P1) SAME-SIGN concentration. The signed causal cosines cos_C(d_f, gbar) are
       concentrated on ONE sign (all push toward the carrier's polarity), far more
       than the random-null directions, whose signed cosines are symmetric ~0.
  (P2) NON-TRIVIAL magnitude. |cos_C(d_f, gbar)| is materially larger than null.
       (If the d_f were truly random, |cos_C| ~ 1/sqrt(d) ~ 0.02; convergence
       requires a clear excess.)
  (P3) MUTUAL orthogonality co-exists with (P1)-(P2): the SAME directions are
       ~orthogonal to each other (re-confirmed here from 62's Gram) while sharing
       the gbar component. This is the signature of convergence, not redundancy.
  (P4) READOUT rank. Projecting the carrier onto gbar collapses a ~k-dim
       orthogonal set onto ~1 behavioural axis: the variance of the d_f along
       gbar relative to their orthogonal residual quantifies the many-to-one
       compression.

WHAT IT COMPUTES (per layer + cross-layer, causal metric ⟨a,b⟩_C = a^T Σ⁻¹ b):
  * signed cosine cos_C(d_f, gbar) for every carrier feature
  * sign-concentration = |mean(sign)| and fraction on the majority sign
  * one-sample test that signed cosines have non-zero mean (sign-consistent push)
  * mean |cos_C| vs the random-null band (200 draws from other layers)
  * gbar-aligned vs orthogonal-residual energy split of the carrier directions
  * a "convergence index": same-sign fraction × (mean|cos_C| / null mean|cos_C|)

VERDICT per layer:
  CONVERGENCE  if signed cosines are sign-concentrated AND mean|cos_C| exceeds
               the null band (orthogonal-to-each-other but aligned-to-gbar);
  DIFFUSE      if cos_C(d_f, gbar) ~ null (no shared readout axis) -> the carrier
               does NOT converge on gbar and the reconciliation fails;
  MIXED        otherwise (report numbers).

INPUTS: only concept_directions.npz (gbar, Sigma_inv) from 60_ + transcoder
decoder rows for the chosen features. NO prompts, NO model forward pass.

SELF-TEST (no torch / no repo): python 63_concept_projection_geometry.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("concept_projection")

FEATURE_ID_RE = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")


# =====================================================================
# Geometry core (pure numpy; unit-tested by --self_test)
# =====================================================================

def signed_causal_cosines(D: np.ndarray, gbar: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """
    cos_C(d_f, gbar) = <d_f, gbar>_C / (||d_f||_C ||gbar||_C), per row of D.
    Sign is meaningful: + = toward the carrier's beta-polarity end, - = alpha.
    """
    Dg = D @ Sigma_inv @ gbar                                # <d_f, gbar>_C, (k,)
    dC = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, Sigma_inv, D), 1e-30, None))
    gC = float(np.sqrt(gbar @ Sigma_inv @ gbar))
    return Dg / (dC * gC)


def sign_concentration(cos_signed: np.ndarray) -> Dict[str, float]:
    """How concentrated on one sign the signed cosines are."""
    s = np.sign(cos_signed)
    s = s[s != 0]
    if s.size == 0:
        return {"abs_mean_sign": float("nan"), "majority_sign_frac": float("nan"),
                "majority_sign": 0}
    maj = 1.0 if s.sum() >= 0 else -1.0
    return {
        "abs_mean_sign": float(abs(np.mean(s))),               # 1.0 = all same sign
        "majority_sign_frac": float(np.mean(s == maj)),         # >=0.5
        "majority_sign": int(maj),
    }


def one_sample_sign_test(cos_signed: np.ndarray) -> Dict[str, float]:
    """
    Is the mean signed cosine non-zero? Report mean, std-error, t-like z, and a
    sign-test p (binomial, two-sided) on majority sign. Convergence => mean != 0.
    """
    x = np.asarray(cos_signed, dtype=float)
    k = x.size
    mean = float(np.mean(x)) if k else float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
    z = float(mean / se) if (se and se > 0) else float("nan")
    # exact two-sided sign test
    from math import comb
    s = np.sign(x); s = s[s != 0]
    n = s.size
    if n == 0:
        p = float("nan")
    else:
        kmaj = int(max((s > 0).sum(), (s < 0).sum()))
        tail = sum(comb(n, i) for i in range(kmaj, n + 1)) / (2.0 ** n)
        p = float(min(1.0, 2.0 * tail))
    return {"mean_signed_cos": mean, "se": se, "z": z, "sign_test_p": p, "n_nonzero": int(n)}


def energy_split(D: np.ndarray, gbar: np.ndarray, Sigma_inv: np.ndarray) -> Dict[str, float]:
    """
    Split the carrier directions' energy (in the causal metric) into the part
    along gbar vs the orthogonal residual. Quantifies many-to-one compression:
    along_frac near 0 with sign-concentration ~1 => many ortho dirs each with a
    small but consistent gbar component (classic convergence/superposition).
    """
    A = None  # whitened coords
    # whiten: <a,b>_C = (A a)·(A b); build A via eigh of Sigma_inv
    w, V = np.linalg.eigh(Sigma_inv)
    w = np.clip(w, 0.0, None)
    A = (V * np.sqrt(w)) @ V.T                               # Sigma_inv^{1/2}
    Dw = D @ A.T
    gw = A @ gbar
    ghat = gw / (np.linalg.norm(gw) + 1e-30)
    along = Dw @ ghat                                        # (k,)
    along_energy = float(np.sum(along ** 2))
    total_energy = float(np.sum(Dw ** 2))
    return {
        "along_gbar_energy_frac": along_energy / total_energy if total_energy > 0 else float("nan"),
        "orth_residual_energy_frac": 1.0 - (along_energy / total_energy) if total_energy > 0 else float("nan"),
        "mean_abs_along": float(np.mean(np.abs(along))),
    }


def parse_feature_id(fid: str) -> Tuple[int, int]:
    m = FEATURE_ID_RE.match(str(fid).strip())
    if not m:
        raise ValueError(f"cannot parse feature_id {fid!r} (expected like 'L24_123' or 'L24:123')")
    return int(m.group(1)), int(m.group(2))


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(63)
    d = 256
    evals = np.concatenate([np.linspace(20, 4, 20), np.linspace(2, 0.05, d - 20)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (Q * evals) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_inv = np.linalg.inv(Sigma + 1e-3 * np.mean(np.diag(Sigma)) * np.eye(d))

    # whitened concept axis
    w, V = np.linalg.eigh(Sigma_inv); w = np.clip(w, 0, None)
    A = (V * np.sqrt(w)) @ V.T
    Ainv = np.linalg.inv(A)
    gw = rng.standard_normal(d); gw /= np.linalg.norm(gw)
    gbar = Ainv @ gw                                         # so whitened gbar = gw

    k = 20

    # CONVERGENCE set: orthonormal-in-whitened directions, EACH given a small
    # same-sign component along gw. Mutually ~orthogonal, but all project onto gbar.
    B = rng.standard_normal((k, d)) @ A.T                    # random whitened
    B, _ = np.linalg.qr(B.T); B = B.T[:k]                   # orthonormal rows (whitened)
    a = 0.30                                                 # shared gbar loading
    conv_w = np.array([np.sqrt(1 - a*a) * B[i] + a * gw for i in range(k)])
    D_conv = conv_w @ np.linalg.inv(A).T                    # back to raw

    # DIFFUSE set: random directions with NO shared gbar component
    D_diff = (rng.standard_normal((k, d))) @ Ainv.T

    print("\n--- SELF TEST -------------------------------------------------")
    for name, D in [("convergence", D_conv), ("diffuse", D_diff)]:
        cs = signed_causal_cosines(D, gbar, Sigma_inv)
        sc = sign_concentration(cs)
        st = one_sample_sign_test(cs)
        es = energy_split(D, gbar, Sigma_inv)
        # mutual orthogonality (mean |cos_C| between the directions)
        Kc = D @ Sigma_inv @ D.T
        dd = np.sqrt(np.clip(np.diag(Kc), 1e-30, None))
        C = Kc / np.outer(dd, dd)
        iu = np.triu_indices(k, 1)
        print(f"\n{name} set (k={k}):")
        print(f"  signed cos_C(d,gbar): mean={st['mean_signed_cos']:+.3f}  "
              f"|mean sign|={sc['abs_mean_sign']:.2f}  maj-sign frac={sc['majority_sign_frac']:.2f}")
        print(f"  mean |cos_C(d,gbar)| = {np.mean(np.abs(cs)):.3f}   "
              f"sign-test p={st['sign_test_p']:.1e}")
        print(f"  mutual mean |cos_C| (orthogonality) = {np.mean(np.abs(C[iu])):.3f}")
        print(f"  along-gbar energy frac = {es['along_gbar_energy_frac']:.3f}")

    cs_c = signed_causal_cosines(D_conv, gbar, Sigma_inv)
    cs_d = signed_causal_cosines(D_diff, gbar, Sigma_inv)
    sc_c = sign_concentration(cs_c); sc_d = sign_concentration(cs_d)
    # convergence: sign-concentrated AND larger |cos|; diffuse: symmetric, small
    assert sc_c["majority_sign_frac"] > 0.9, f"convergence must be sign-concentrated (got {sc_c['majority_sign_frac']:.2f})"
    assert np.mean(np.abs(cs_c)) > 3 * np.mean(np.abs(cs_d)), "convergence |cos_C| must exceed diffuse"
    assert sc_d["majority_sign_frac"] < 0.8, f"diffuse should be sign-symmetric (got {sc_d['majority_sign_frac']:.2f})"
    # and convergence directions remain ~mutually orthogonal
    Kc = D_conv @ Sigma_inv @ D_conv.T
    dd = np.sqrt(np.clip(np.diag(Kc), 1e-30, None)); C = Kc / np.outer(dd, dd)
    mo = np.mean(np.abs(C[np.triu_indices(k, 1)]))
    assert mo < 0.30, f"convergence directions should stay ~mutually orthogonal (got {mo:.2f})"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(convergence: sign-concentrated onto gbar AND mutually ~orthogonal;")
    print(" diffuse: cosines symmetric around 0)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Feature resolution (same options as 62)
# =====================================================================

def _resolve_features(args) -> Dict[int, List[int]]:
    feats: Dict[int, List[int]] = {}

    def add(L, fi): feats.setdefault(L, []).append(fi)

    if args.features:
        for tok in args.features.split(","):
            L, fi = parse_feature_id(tok); add(L, fi)
        return feats
    if args.feature_file:
        for line in open(args.feature_file):
            line = line.strip()
            if line:
                L, fi = parse_feature_id(line); add(L, fi)
        return feats
    if args.cluster_labels:
        import pandas as pd
        cl = pd.read_csv(args.cluster_labels)
        if "feature_id" not in cl.columns:
            cl = cl.rename(columns={cl.columns[0]: "feature_id"})
        cl["feature_id"] = cl["feature_id"].astype(str)
        if args.cluster_col not in cl.columns:
            raise SystemExit(f"--cluster_col {args.cluster_col!r} not in {list(cl.columns)}")

        def norm_id(c):
            s = str(c).strip().lstrip("Cc")
            return s[:-2] if s.endswith(".0") else s
        wanted = {norm_id(c) for c in args.clusters.split(",")} if args.clusters else None
        for fid, c in zip(cl["feature_id"], cl[args.cluster_col].apply(norm_id)):
            if wanted is None or c in wanted:
                try:
                    L, fi = parse_feature_id(fid)
                except ValueError:
                    continue
                if args.layers and L not in args.layers:
                    continue
                add(L, fi)
        return feats
    raise SystemExit("provide --features, --feature_file, or --cluster_labels. See --help.")


# =====================================================================
# Real run
# =====================================================================

def run_real(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cd = np.load(args.concept_npz)
    for key in ("gbar", "Sigma_inv"):
        if key not in cd:
            raise SystemExit(f"concept_npz must contain '{key}' (from 60_).")
    gbar = cd["gbar"].astype(np.float64)
    Sigma_inv = cd["Sigma_inv"].astype(np.float64)
    d = gbar.shape[0]

    feats = _resolve_features(args)
    layers = sorted(feats.keys())
    logger.info("carrier features: %s", {L: len(v) for L, v in feats.items()})

    from src.transcoder import load_transcoder_set
    null_layers = args.null_layers or list(range(args.null_layer_min, args.null_layer_max + 1))
    need = sorted(set(layers) | set(null_layers))
    tset = load_transcoder_set(model_size=args.model_size, device=args.device,
                               lazy_load=True, layers=need)

    # Cache full W_dec per layer as numpy (one GPU→CPU transfer, then instant indexing).
    # Same fix as in 62 — eliminates the slow per-call _get_decoder_vectors overhead.
    W_dec_cache: Dict[int, np.ndarray] = {}

    def _ensure_wdec_cached(layer: int) -> np.ndarray:
        if layer not in W_dec_cache:
            tc = tset[layer]
            if hasattr(tc, "W_dec"):
                logger.info("Caching W_dec for layer %d (shape=%s)...", layer, tc.W_dec.shape)
                W_dec_cache[layer] = tc.W_dec.detach().float().cpu().numpy().astype(np.float64)
                logger.info("  cached %.1f MB", W_dec_cache[layer].nbytes / 1e6)
            else:
                raise RuntimeError(f"transcoder for L{layer} has no W_dec attribute")
        return W_dec_cache[layer]

    def decoder_rows(layer: int, idxs: List[int]) -> np.ndarray:
        W = _ensure_wdec_cached(layer)
        return W[np.asarray(idxs, dtype=np.int64)]

    _pool_precached = {"done": False}
    def null_directions(k: int, exclude: set) -> np.ndarray:
        """k random decoder rows from layers NOT in `exclude` — W_dec cached."""
        pool = [L for L in null_layers if L not in exclude] or null_layers
        # pre-cache all pool layers once (idempotent)
        if not _pool_precached["done"]:
            logger.info("Pre-caching W_dec for %d null layers (one-time)...", len(pool))
            for L in pool:
                _ensure_wdec_cached(L)
            _pool_precached["done"] = True
        d = Sigma_inv.shape[0]
        D = np.empty((k, d), dtype=np.float64)
        for slot in range(k):
            L = pool[rng.integers(len(pool))]
            W = W_dec_cache[L]
            D[slot] = W[int(rng.integers(W.shape[0]))]
        return D

    results = {"layers": {}, "cross_layer": None, "params": {
        "concept_npz": str(args.concept_npz), "n_null_draws": args.n_null_draws,
        "null_layers": null_layers}}

    def null_band_for(k: int, exclude: set, n: int):
        majfracs, absmeans = [], []
        logger.info("  computing null band: %d draws of %d random directions...", n, k)
        for di in range(n):
            if di > 0 and di % 50 == 0:
                logger.info("    null draw %d/%d", di, n)
            Dn = null_directions(k, exclude)
            cs = signed_causal_cosines(Dn, gbar, Sigma_inv)
            majfracs.append(sign_concentration(cs)["majority_sign_frac"])
            absmeans.append(float(np.mean(np.abs(cs))))
        def band(x):
            a = np.array(x); return {"mean": float(a.mean()),
                                     "p05": float(np.percentile(a, 5)),
                                     "p95": float(np.percentile(a, 95))}
        return band(majfracs), band(absmeans)

    for L in layers:
        D = decoder_rows(L, feats[L]); k = len(feats[L])
        cs = signed_causal_cosines(D, gbar, Sigma_inv)
        sc = sign_concentration(cs)
        st = one_sample_sign_test(cs)
        es = energy_split(D, gbar, Sigma_inv)
        mean_abs = float(np.mean(np.abs(cs)))
        np.save(out / f"concept_cosines_{L}.npy", cs)

        maj_band, abs_band = null_band_for(k, {L}, args.n_null_draws) if k >= 2 else (None, None)

        verdict = "MIXED"
        if maj_band and abs_band:
            sign_concentrated = sc["majority_sign_frac"] > maj_band["p95"]
            magnitude_excess = mean_abs > abs_band["p95"]
            if sign_concentrated and magnitude_excess:
                verdict = "CONVERGENCE"
            elif (not sign_concentrated) and (not magnitude_excess):
                verdict = "DIFFUSE"
        conv_index = (sc["majority_sign_frac"] * (mean_abs / abs_band["mean"])
                      if (abs_band and abs_band["mean"] > 0) else float("nan"))

        results["layers"][str(L)] = {
            "k": k,
            "signed_cosine": {"mean": st["mean_signed_cos"], "z": st["z"],
                              "sign_test_p": st["sign_test_p"]},
            "sign_concentration": sc,
            "mean_abs_cos_C_gbar": mean_abs,
            "energy_split": es,
            "null_majority_sign_frac": maj_band,
            "null_mean_abs_cos": abs_band,
            "convergence_index": conv_index,
            "verdict": verdict,
        }
        logger.info("L%d: k=%d  signed-mean=%+.3f  maj-sign=%.2f (null~%.2f)  "
                    "|cosC|=%.3f (null~%.3f)  along-gbar=%.3f  -> %s",
                    L, k, st["mean_signed_cos"], sc["majority_sign_frac"],
                    maj_band["mean"] if maj_band else float("nan"),
                    mean_abs, abs_band["mean"] if abs_band else float("nan"),
                    es["along_gbar_energy_frac"], verdict)

    # cross-layer
    all_D = [decoder_rows(L, feats[L]) for L in layers]
    if all_D:
        D = np.vstack(all_D); k = D.shape[0]
        cs = signed_causal_cosines(D, gbar, Sigma_inv)
        results["cross_layer"] = {
            "k": k,
            "signed_cosine": one_sample_sign_test(cs),
            "sign_concentration": sign_concentration(cs),
            "mean_abs_cos_C_gbar": float(np.mean(np.abs(cs))),
            "energy_split": energy_split(D, gbar, Sigma_inv),
        }
        np.save(out / "concept_cosines_ALL.npy", cs)

    with open(out / "concept_projection.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- console ----
    print("\n" + "=" * 84)
    print("CONCEPT PROJECTION GEOMETRY  --  convergence arbiter (causal metric)")
    print("=" * 84)
    print(f"{'layer':>6} {'k':>4} {'signed_mean':>12} {'maj_sign':>9} {'null_maj':>9} "
          f"{'|cosC|':>7} {'null':>7} {'along_g':>8} {'p_sign':>9} verdict")
    for L in layers:
        r = results["layers"][str(L)]
        sc, es = r["sign_concentration"], r["energy_split"]
        mb, ab = r["null_majority_sign_frac"], r["null_mean_abs_cos"]
        print(f"{L:>6} {r['k']:>4} {r['signed_cosine']['mean']:>+12.3f} "
              f"{sc['majority_sign_frac']:>9.2f} {(mb['mean'] if mb else float('nan')):>9.2f} "
              f"{r['mean_abs_cos_C_gbar']:>7.3f} {(ab['mean'] if ab else float('nan')):>7.3f} "
              f"{es['along_gbar_energy_frac']:>8.3f} {r['signed_cosine']['sign_test_p']:>9.1e} "
              f"{r['verdict']}")
    if results["cross_layer"]:
        cl = results["cross_layer"]; sc, es = cl["sign_concentration"], cl["energy_split"]
        print(f"{'ALL':>6} {cl['k']:>4} {cl['signed_cosine']['mean_signed_cos']:>+12.3f} "
              f"{sc['majority_sign_frac']:>9.2f} {'-':>9} "
              f"{cl['mean_abs_cos_C_gbar']:>7.3f} {'-':>7} "
              f"{es['along_gbar_energy_frac']:>8.3f} "
              f"{cl['signed_cosine']['sign_test_p']:>9.1e} (cross-layer)")
    print("\nINTERPRETATION:")
    print("  maj_sign >> null AND |cosC| >> null  ==>  CONVERGENCE")
    print("     (write-dirs mutually ~orthogonal [from 62] BUT all project onto gbar,")
    print("      same sign -> many-to-one read-out; reconciles ortho-dirs with Jaccard 0.92)")
    print("  maj_sign ~ null, |cosC| ~ null       ==>  DIFFUSE (no shared read-out axis)")
    print(f"\nwrote: {out}/concept_projection.json + per-layer concept_cosines_*.npy")
    print("=" * 84)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concept_npz", type=str,
                   default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--features", type=str, default=None,
                   help="comma list like 'L24:123,L18:789'")
    p.add_argument("--feature_file", type=str, default=None)
    p.add_argument("--cluster_labels", type=str, default=None)
    p.add_argument("--cluster_col", type=str, default="agglo_coimp_subgroup_k30")
    p.add_argument("--clusters", type=str, default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", type=str, default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_size", type=str, default="4b")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--d_transcoder", type=int, default=163840)
    p.add_argument("--n_null_draws", type=int, default=200)
    p.add_argument("--null_layers", type=int, nargs="*", default=None)
    p.add_argument("--null_layer_min", type=int, default=10)
    p.add_argument("--null_layer_max", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
