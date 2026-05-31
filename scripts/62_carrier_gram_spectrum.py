"""
62_carrier_gram_spectrum.py
===================================================================
ARBITER between two competing architectures for the alpha/beta carrier:

  H_redundant  (uniform redundancy): the k decoder directions of a layer's
      carrier are near-collinear -- many near-copies of ONE causal axis. Then
      the causal Gram K_ij = <d_i, d_j>_C has ONE dominant eigenvalue and the
      rest ~0  =>  effective rank ~ 1, participation ratio ~ 1.
      This is what graded-patching saturation (1 feature ~= full IIA) and
      per-pair Jaccard 0.92-0.99 imply, and it is CONSISTENT with the four
      negative interventional results (copies compensate any subset removal).

  H_combinatorial (rich continuous space): the directions are diverse and
      span many independent axes, so their combinations reach exponentially
      many distinct points. Then K has a FLAT spectrum  =>  effective rank ~ k,
      participation ratio ~ k. This is the picture required by a
      "continuous space of linear combinations / reasoning trajectories".

These make OPPOSITE predictions about the SAME object (the carrier's decoder
Gram). This script measures it, in the causal inner product, so the choice of
which theory to develop is made on the spectrum, not on narrative appeal.

WHAT IT COMPUTES (per layer, and for any cross-layer set):
  * Causal Gram        K_ij = <d_i, d_j>_C = d_i^T Sigma^{-1} d_j   (Sigma from 60_)
  * Also the COSINE Gram in the causal metric (scale-free collinearity)
  * Eigenvalue spectrum of the cosine Gram (scale-free) and of K (scale-aware)
  * effective rank metrics:
      - participation ratio PR = (sum lambda)^2 / sum(lambda^2)   in [1, k]
      - entropy effective rank  exp(H), H = -sum p_i log p_i, p_i = lambda_i/sum
      - rank90 / rank99 = #eigs to reach 90% / 99% of total spectral mass
      - lambda_1 share = lambda_1 / sum lambda
  * Null calibration: the SAME metrics for K random decoder directions drawn
    from OTHER layers' transcoders (matched count), so "rank ~ 1" is read
    against what random unrelated directions give. (Random high-dim directions
    are near-orthogonal => high PR; a low PR for the carrier is therefore a
    strong, calibrated redundancy signal.)
  * Mean |cosine_C| among carrier directions (signed and unsigned) -- the
    direct collinearity number behind the Jaccard 0.92-0.99 observation.

VERDICT (per layer): REDUNDANT if PR/k and rank90/k are both small and
lambda_1-share is large, well separated from the random-null band;
COMBINATORIAL if the carrier's spectrum is statistically like random
near-orthogonal directions; INTERMEDIATE otherwise (report the number, don't
force a label).

FEATURE SELECTION (robust to wherever your cluster CSV lives):
  --features "L24:123,L24:456,L18:789"     explicit list (fastest, unambiguous)
  --feature_file ids.txt                    one feature_id per line
  --cluster_labels cl.csv --cluster_col COL --clusters 16,19  (filter by cluster)
  --layers 18 24                            if no list/CSV: uses the union of
                                            features named in --features per layer,
                                            else requires one of the above.

OUTPUT (data/analysis/runD_v2/geometry_stage1/ by default):
  carrier_gram_spectrum.json     per-layer metrics + null band + verdict
  carrier_gram_<layer>.npy       the causal Gram K for each layer (for figures)
  carrier_cosine_<layer>.npy     the causal-cosine Gram for each layer

SELF-TEST (no torch / no repo): python 62_carrier_gram_spectrum.py --self_test
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
logger = logging.getLogger("gram_spectrum")

FEATURE_ID_RE = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")


# =====================================================================
# Geometry core (pure numpy; unit-tested by --self_test)
# =====================================================================

def causal_cosine_gram(D: np.ndarray, Sigma_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given decoder directions D (k, d) and Sigma_inv (d, d), return:
      K       : causal Gram, K_ij = d_i^T Sigma_inv d_j           (scale-aware)
      C       : causal-cosine Gram, C_ij = K_ij / sqrt(K_ii K_jj)  (scale-free)
    """
    K = D @ Sigma_inv @ D.T
    K = 0.5 * (K + K.T)
    diag = np.clip(np.diag(K), 1e-30, None)
    inv_sqrt = 1.0 / np.sqrt(diag)
    C = K * np.outer(inv_sqrt, inv_sqrt)
    C = 0.5 * (C + C.T)
    return K, C


def spectrum_metrics(M: np.ndarray) -> Dict[str, float]:
    """
    Effective-rank metrics from the eigenvalues of a symmetric PSD matrix M.
    For the cosine Gram, trace = k, so these are directly comparable across sets.
    """
    k = M.shape[0]
    w = np.linalg.eigvalsh(M)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0 or k == 0:
        return {"k": k, "participation_ratio": float("nan"), "entropy_eff_rank": float("nan"),
                "rank90": k, "rank99": k, "lambda1_share": float("nan"),
                "lambda1": 0.0, "lambda2": 0.0}
    p = w / total
    pr = float((w.sum() ** 2) / np.sum(w ** 2))
    nz = p[p > 0]
    H = float(-np.sum(nz * np.log(nz)))
    eff = float(np.exp(H))
    w_desc = np.sort(w)[::-1]
    csum = np.cumsum(w_desc) / total
    rank90 = int(np.searchsorted(csum, 0.90) + 1)
    rank99 = int(np.searchsorted(csum, 0.99) + 1)
    return {
        "k": int(k),
        "participation_ratio": pr,
        "participation_ratio_frac": pr / k,
        "entropy_eff_rank": eff,
        "entropy_eff_rank_frac": eff / k,
        "rank90": rank90, "rank90_frac": rank90 / k,
        "rank99": rank99, "rank99_frac": rank99 / k,
        "lambda1_share": float(w_desc[0] / total),
        "lambda1": float(w_desc[0]),
        "lambda2": float(w_desc[1]) if k > 1 else 0.0,
    }


def mean_offdiag_abs(C: np.ndarray) -> Dict[str, float]:
    """Mean signed and unsigned off-diagonal causal cosine (collinearity number)."""
    k = C.shape[0]
    if k < 2:
        return {"mean_cos_signed": float("nan"), "mean_cos_abs": float("nan"),
                "min_cos": float("nan"), "max_offdiag": float("nan")}
    iu = np.triu_indices(k, k=1)
    off = C[iu]
    return {
        "mean_cos_signed": float(np.mean(off)),
        "mean_cos_abs": float(np.mean(np.abs(off))),
        "min_cos": float(np.min(off)),
        "max_offdiag": float(np.max(off)),
    }


def parse_feature_id(fid: str) -> Tuple[int, int]:
    m = FEATURE_ID_RE.match(str(fid).strip())
    if not m:
        raise ValueError(f"cannot parse feature_id {fid!r} (expected like 'L24_123' or 'L24:123')")
    return int(m.group(1)), int(m.group(2))


# =====================================================================
# Self-test: planted redundant vs diverse sets give the expected verdict
# =====================================================================

def self_test() -> None:
    rng = np.random.default_rng(31)
    d = 256
    # anisotropic Sigma (like an unembedding covariance)
    evals = np.concatenate([np.linspace(20, 4, 20), np.linspace(2, 0.05, d - 20)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (Q * evals) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_inv = np.linalg.inv(Sigma + 1e-3 * np.mean(np.diag(Sigma)) * np.eye(d))

    k = 12
    # REDUNDANT set: one axis + small jitter (near-copies)
    axis = rng.standard_normal(d)
    axis = axis / np.linalg.norm(axis)
    D_red = np.array([axis + 0.02 * rng.standard_normal(d) for _ in range(k)])
    # COMBINATORIAL set: k independent random directions (near-orthogonal in high-d)
    D_comb = rng.standard_normal((k, d))

    print("\n--- SELF TEST -------------------------------------------------")
    for name, D in [("redundant", D_red), ("combinatorial", D_comb)]:
        K, C = causal_cosine_gram(D, Sigma_inv)
        sm = spectrum_metrics(C)
        od = mean_offdiag_abs(C)
        print(f"\n{name} set (k={k}):")
        print(f"  participation_ratio = {sm['participation_ratio']:.2f}  "
              f"(frac of k: {sm['participation_ratio_frac']:.3f})")
        print(f"  entropy_eff_rank    = {sm['entropy_eff_rank']:.2f}  "
              f"(frac: {sm['entropy_eff_rank_frac']:.3f})")
        print(f"  rank90 = {sm['rank90']}/{k}   lambda1_share = {sm['lambda1_share']:.3f}")
        print(f"  mean |cos_C| off-diag = {od['mean_cos_abs']:.3f}")

    Kr, Cr = causal_cosine_gram(D_red, Sigma_inv)
    Kc, Cc = causal_cosine_gram(D_comb, Sigma_inv)
    pr_red = spectrum_metrics(Cr)["participation_ratio"]
    pr_comb = spectrum_metrics(Cc)["participation_ratio"]
    l1_red = spectrum_metrics(Cr)["lambda1_share"]
    l1_comb = spectrum_metrics(Cc)["lambda1_share"]

    assert pr_red < 0.35 * k, f"redundant set must have PR << k (got {pr_red:.2f}, k={k})"
    assert pr_comb > 0.6 * k, f"combinatorial set must have PR ~ k (got {pr_comb:.2f})"
    assert l1_red > 0.6, f"redundant set: lambda1 should dominate (got {l1_red:.3f})"
    assert l1_comb < 0.3, f"combinatorial set: no dominant eig (got {l1_comb:.3f})"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(redundant -> PR~1, lambda1 dominant; combinatorial -> PR~k, flat)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def _resolve_features(args, cl_layers_available: Optional[set]) -> Dict[int, List[int]]:
    """Return {layer: [feature_idx,...]} from --features / --feature_file / cluster CSV."""
    feats: Dict[int, List[int]] = {}

    def add(fid_layer, fid_idx):
        feats.setdefault(fid_layer, []).append(fid_idx)

    if args.features:
        for tok in args.features.split(","):
            L, fi = parse_feature_id(tok)
            add(L, fi)
        return feats

    if args.feature_file:
        for line in open(args.feature_file):
            line = line.strip()
            if line:
                L, fi = parse_feature_id(line)
                add(L, fi)
        return feats

    if args.cluster_labels:
        import pandas as pd
        cl = pd.read_csv(args.cluster_labels)
        if "feature_id" not in cl.columns:
            cl = cl.rename(columns={cl.columns[0]: "feature_id"})
        cl["feature_id"] = cl["feature_id"].astype(str)
        if args.cluster_col not in cl.columns:
            raise SystemExit(f"--cluster_col {args.cluster_col!r} not in {list(cl.columns)}")

        def norm_id(cid):
            s = str(cid).strip().lstrip("C").lstrip("c")
            return s[:-2] if s.endswith(".0") else s
        wanted = {norm_id(c) for c in args.clusters.split(",")} if args.clusters else None
        col_norm = cl[args.cluster_col].apply(norm_id)
        for fid, c in zip(cl["feature_id"], col_norm):
            if wanted is None or c in wanted:
                try:
                    L, fi = parse_feature_id(fid)
                except ValueError:
                    continue
                if args.layers and L not in args.layers:
                    continue
                add(L, fi)
        return feats

    raise SystemExit("provide --features, --feature_file, or --cluster_labels (+--clusters). See --help.")


def run_real(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cd = np.load(args.concept_npz)
    if "Sigma_inv" not in cd:
        raise SystemExit("concept_npz must contain 'Sigma_inv' (from 60_).")
    Sigma_inv = cd["Sigma_inv"].astype(np.float64)
    d = Sigma_inv.shape[0]

    feats = _resolve_features(args, None)
    layers = sorted(feats.keys())
    logger.info("carrier features: %s", {L: len(v) for L, v in feats.items()})

    # transcoders
    from src.transcoder import load_transcoder_set
    null_layers = args.null_layers or [L for L in range(args.null_layer_min, args.null_layer_max + 1)]
    need_layers = sorted(set(layers) | set(null_layers))
    tset = load_transcoder_set(model_size=args.model_size, device=args.device,
                               lazy_load=True, layers=need_layers)

    def decoder_rows(layer: int, idxs: List[int]) -> np.ndarray:
        tc = tset[layer]
        t = torch.tensor(idxs, dtype=torch.long)
        return tc._get_decoder_vectors(t).detach().float().cpu().numpy().astype(np.float64)

    # ── BATCH null draws ────────────────────────────────────────────────────
    # OPTIMIZATION: instead of 200 × k single-feature decoder calls, we
    # pre-generate ALL (layer, idx) pairs for ALL null draws, group by layer,
    # do ONE batched decoder_rows() call per layer, then redistribute.
    # For k=20, n_null_draws=200, null_layers=16 → 200*20=4000 random rows total,
    # distributed across 16 layers ≈ 250 rows/layer. One call per layer instead of 4000.
    def batched_null_draws(k: int, n_draws: int, exclude: set) -> List[np.ndarray]:
        """Return n_draws Gram-input matrices D_i of shape (k, d), all from random
        unrelated layers, batched per layer for speed."""
        pool_layers = [L for L in null_layers if L not in exclude]
        if not pool_layers:
            pool_layers = null_layers
        # 1) sample all (draw_idx, slot, layer, fidx) tuples
        plan = []
        for di in range(n_draws):
            for slot in range(k):
                L = pool_layers[rng.integers(len(pool_layers))]
                tc = tset[L]
                nfeat = tc.W_dec.shape[0] if hasattr(tc, "W_dec") else args.d_transcoder
                fi = int(rng.integers(nfeat))
                plan.append((di, slot, L, fi))
        # 2) group by layer and do ONE batched extraction per layer
        from collections import defaultdict
        by_layer = defaultdict(list)
        for (di, slot, L, fi) in plan:
            by_layer[L].append((di, slot, fi))
        # cache (layer, fi) → row
        rows_cache: Dict[Tuple[int, int], np.ndarray] = {}
        for L, triples in by_layer.items():
            fis = [fi for (_, _, fi) in triples]
            # unique idxs only (in case of duplicates)
            unique_fis = sorted(set(fis))
            D_batch = decoder_rows(L, unique_fis)  # (n_unique, d)
            for j, fi in enumerate(unique_fis):
                rows_cache[(L, fi)] = D_batch[j]
        # 3) reassemble per-draw matrices
        Ds = [np.zeros((k, Sigma_inv.shape[0]), dtype=np.float64) for _ in range(n_draws)]
        for (di, slot, L, fi) in plan:
            Ds[di][slot] = rows_cache[(L, fi)]
        return Ds

    results = {"layers": {}, "cross_layer": None,
               "null": {}, "params": {
                   "n_null_draws": args.n_null_draws, "null_layers": null_layers,
                   "concept_npz": str(args.concept_npz)}}

    # ---- per-layer carrier spectrum + null calibration ----
    for L in layers:
        idxs = feats[L]
        if len(idxs) < 2:
            logger.warning("layer %d has <2 features; spectrum trivial, skipping null", L)
        D = decoder_rows(L, idxs)
        K, C = causal_cosine_gram(D, Sigma_inv)
        np.save(out / f"carrier_gram_{L}.npy", K)
        np.save(out / f"carrier_cosine_{L}.npy", C)
        sm = spectrum_metrics(C)
        od = mean_offdiag_abs(C)

        # null: random unrelated directions of the same count k (BATCHED)
        k = len(idxs)
        null_pr, null_l1, null_cosabs = [], [], []
        if k >= 2:
            logger.info("L%d: drawing %d null sets (k=%d each) with batched decoder lookup...",
                        L, args.n_null_draws, k)
            null_Ds = batched_null_draws(k, args.n_null_draws, exclude={L})
            logger.info("L%d: null draws assembled, computing %d Gram spectra...",
                        L, args.n_null_draws)
            for Dn in null_Ds:
                _, Cn = causal_cosine_gram(Dn, Sigma_inv)
                smn = spectrum_metrics(Cn)
                odn = mean_offdiag_abs(Cn)
                null_pr.append(smn["participation_ratio"])
                null_l1.append(smn["lambda1_share"])
                null_cosabs.append(odn["mean_cos_abs"])

        def band(x):
            if not x:
                return None
            a = np.array(x)
            return {"mean": float(a.mean()), "p05": float(np.percentile(a, 5)),
                    "p95": float(np.percentile(a, 95))}

        # verdict for this layer
        verdict = "INTERMEDIATE"
        if k >= 2 and null_pr:
            pr_null_lo = np.percentile(null_pr, 5)
            redundant = (sm["participation_ratio"] < 0.4 * k
                         and sm["lambda1_share"] > 0.5
                         and sm["participation_ratio"] < pr_null_lo)
            combinatorial = (sm["participation_ratio"] >= pr_null_lo
                             and sm["lambda1_share"] < 0.3)
            if redundant:
                verdict = "REDUNDANT"
            elif combinatorial:
                verdict = "COMBINATORIAL"

        results["layers"][str(L)] = {
            "k": k, "spectrum": sm, "offdiag_cosine": od,
            "null_participation_ratio": band(null_pr),
            "null_lambda1_share": band(null_l1),
            "null_mean_cos_abs": band(null_cosabs),
            "verdict": verdict,
        }
        logger.info("L%d: k=%d PR=%.2f (%.2f of k) lambda1=%.3f mean|cosC|=%.3f -> %s",
                    L, k, sm["participation_ratio"], sm["participation_ratio_frac"],
                    sm["lambda1_share"], od["mean_cos_abs"], verdict)

    # ---- cross-layer set (all carrier features together) ----
    all_D, all_ids = [], []
    for L in layers:
        D = decoder_rows(L, feats[L])
        all_D.append(D)
        all_ids += [(L, fi) for fi in feats[L]]
    if all_D:
        D = np.vstack(all_D)
        K, C = causal_cosine_gram(D, Sigma_inv)
        np.save(out / "carrier_gram_ALL.npy", K)
        np.save(out / "carrier_cosine_ALL.npy", C)
        results["cross_layer"] = {
            "k": D.shape[0], "spectrum": spectrum_metrics(C),
            "offdiag_cosine": mean_offdiag_abs(C),
        }

    with open(out / "carrier_gram_spectrum.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- console summary ----
    print("\n" + "=" * 80)
    print("CARRIER GRAM SPECTRUM  --  redundancy vs combinatorial arbiter (causal metric)")
    print("=" * 80)
    print(f"{'layer':>6} {'k':>4} {'PR':>7} {'PR/k':>6} {'rank90':>7} {'lam1':>7} "
          f"{'|cosC|':>7} {'null PR/k':>11} verdict")
    for L in layers:
        r = results["layers"][str(L)]
        sm, od = r["spectrum"], r["offdiag_cosine"]
        npr = r["null_participation_ratio"]
        null_frac = f"{npr['mean']/sm['k']:.2f}" if (npr and sm['k']) else "  -  "
        print(f"{L:>6} {sm['k']:>4} {sm['participation_ratio']:>7.2f} "
              f"{sm['participation_ratio_frac']:>6.2f} "
              f"{str(sm['rank90'])+'/'+str(sm['k']):>7} {sm['lambda1_share']:>7.3f} "
              f"{od['mean_cos_abs']:>7.3f} {null_frac:>11} {r['verdict']}")
    if results["cross_layer"]:
        cl = results["cross_layer"]; sm = cl["spectrum"]
        print(f"{'ALL':>6} {sm['k']:>4} {sm['participation_ratio']:>7.2f} "
              f"{sm['participation_ratio_frac']:>6.2f} "
              f"{str(sm['rank90'])+'/'+str(sm['k']):>7} {sm['lambda1_share']:>7.3f} "
              f"{cl['offdiag_cosine']['mean_cos_abs']:>7.3f} {'(n/a)':>11} (cross-layer)")
    print("\nINTERPRETATION:")
    print("  PR/k -> 0, lambda1 -> 1, |cosC| -> 1, below null band  ==>  UNIFORM REDUNDANCY")
    print("       (carrier ~ one causal axis copied k times; continuum hypothesis disfavoured)")
    print("  PR/k ~ null, lambda1 small                              ==>  COMBINATORIAL/diverse")
    print("       (directions independent; rich-linear-combination hypothesis viable)")
    print(f"\nwrote: {out}/carrier_gram_spectrum.json + per-layer Gram/cosine .npy")
    print("=" * 80)


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")

    p.add_argument("--concept_npz", type=str,
                   default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz",
                   help="npz from 60_ (needs Sigma_inv)")
    # feature selection (one of)
    p.add_argument("--features", type=str, default=None,
                   help="comma list like 'L24:123,L24:456,L18:789'")
    p.add_argument("--feature_file", type=str, default=None, help="one feature_id per line")
    p.add_argument("--cluster_labels", type=str, default=None)
    p.add_argument("--cluster_col", type=str, default="agglo_coimp_subgroup_k30")
    p.add_argument("--clusters", type=str, default=None,
                   help="comma list of cluster IDs to include (e.g. '16,19'); default all")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="restrict to these layers when using a cluster CSV")

    p.add_argument("--out_dir", type=str, default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_size", type=str, default="4b")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--d_transcoder", type=int, default=163840,
                   help="fallback feature count if tc.W_dec shape is unavailable")

    # null calibration
    p.add_argument("--n_null_draws", type=int, default=200)
    p.add_argument("--null_layers", type=int, nargs="*", default=None,
                   help="layers to draw random control directions from (default range)")
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
