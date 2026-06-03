"""
72_feature_acts_read_wres.py   [LAPTOP / CPU-ONLY -- from synced npz, seconds]
===================================================================
The closing mechanism test (chapter sec 4.4). Resolves WHY feature-activations
correlate with the answer (71: CV-AUC 0.974) even though their decoder WRITE
directions are orthogonal to the concept axis (62/63/64: carrier-capture ~0.13).

Hypothesis (convergence/readout): the features do not WRITE the concept into the
residual stream; they READ it. If so, the 227 feature-activations a_f(p) should
linearly PREDICT the projection of the residual state onto the concept axis,
  t(p) = <h(p), w_res>,
with high held-out R^2 -- i.e. the features carry a readout of the concept
subspace. This turns "the carrier is a correlate" from an inference-by-elimination
into a MEASURED fact: features encode the concept's position, they don't cause it.

WHAT IT DOES (all from data already on disk; NO model, NO GPU):
  * builds w_res at each depth by Fisher/LDA on the TRAIN split (same procedure
    as 64), from h_residual_per_depth.npz;
  * forms the target t(p) = <h(p), w_res> (concept-axis projection per prompt);
  * regresses the 227 feature-activations onto t(p) with RIDGE, family-grouped
    CV (held-out R^2) -- do activations linearly predict the concept projection?
  * CONTROLS that make a high R^2 meaningful, not mechanical:
      - shuffled-prompt target (break a_f<->t pairing): R^2 -> 0 expected;
      - random-direction target <h(p), r>: features should predict the CONCEPT
        projection better than a random residual direction (else "features
        predict any residual projection", which would be trivial);
      - per-depth, since w_res rotates (69): is the readout strongest at some depth?
  * also: direct AUC of t(p) (sanity: the projection itself separates classes).

INTERPRETATION:
  high held-out R^2 for w_res, >> shuffled and >> random-direction target
      => features READ the concept subspace (encode its projection). Mechanism for
         71's decodability that is consistent with 62/63's write-orthogonality.
  R^2 ~ random-direction target
      => features predict residual projections generically; no special readout of
         the concept axis (weaker claim; report honestly).

INPUTS:
  --npz   h_residual_per_depth.npz   (postL14..25, final, y, is_train)  [synced]
  --act   activation_matrix.npy      (227 x 538 or 538 x 227)           [local]
  --prompts ...jsonl                 (for surface_family grouping)       [local]

SELF-TEST: python 72_feature_acts_read_wres.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("feat_read_wres")


# =====================================================================
# Core (pure numpy; ridge regression + grouped CV)
# =====================================================================

def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(len(y) - 2, 1); Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    return w / (np.linalg.norm(w) + 1e-30)


def auc_of(t, y):
    o = np.argsort(t); r = np.empty_like(o, float); r[o] = np.arange(1, len(t) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def group_kfold(groups, n_splits, seed=0):
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    for f in np.array_split(uniq, n_splits):
        te = np.array([g in set(f.tolist()) for g in groups])
        yield np.where(~te)[0], np.where(te)[0]


def ridge_cv_r2(X, t, groups, n_splits=5, lam=10.0, seed=0):
    """Family-grouped CV held-out R^2 of ridge regression X -> t (X standardized on train)."""
    r2s, preds = [], np.zeros_like(t)
    for tr, te in group_kfold(groups, n_splits, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        tm = t[tr].mean()
        ttr = t[tr] - tm
        d = Xtr.shape[1]
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ttr)
        pred = Xte @ w + tm
        preds[te] = pred
        ss_res = np.sum((t[te] - pred) ** 2)
        ss_tot = np.sum((t[te] - t[tr].mean()) ** 2) + 1e-30
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s)), float(np.std(r2s)), preds


# =====================================================================
# Self-test: planted readout vs no readout
# =====================================================================

def self_test():
    rng = np.random.default_rng(72)
    nfam, per = 40, 12
    n = nfam * per; d = 2560; nf = 60
    groups = np.repeat(np.arange(nfam), per)
    famlab = rng.integers(0, 2, nfam); y = famlab[groups]

    w_true = rng.standard_normal(d); w_true /= np.linalg.norm(w_true)
    H = rng.standard_normal((n, d)) * 0.5 + np.outer((y * 2 - 1.0) * 2.0, w_true)
    t = H @ w_true                                  # concept projection

    # features that READ t (linear in t) + noise; plus features unrelated to t
    A_read = np.outer(t, rng.standard_normal(nf) * 0.5) + rng.standard_normal((n, nf)) * 0.3
    A_noise = rng.standard_normal((n, nf))
    A = np.hstack([A_read, A_noise])               # 120 features, half read t

    print("\n--- SELF TEST -------------------------------------------------")
    r2, sd, _ = ridge_cv_r2(A, t, groups)
    print(f"  features -> concept projection t:  held-out R^2 = {r2:.3f} +/- {sd:.3f}  (expect high)")
    # shuffled target
    r2_sh, _, _ = ridge_cv_r2(A, rng.permutation(t), groups)
    print(f"  shuffled target:                   held-out R^2 = {r2_sh:.3f}  (expect ~0)")
    # random-direction target
    r = rng.standard_normal(d); t_rand = H @ (r / np.linalg.norm(r))
    r2_rand, _, _ = ridge_cv_r2(A, t_rand, groups)
    print(f"  random-direction target:           held-out R^2 = {r2_rand:.3f}  (expect < concept)")
    assert r2 > 0.5, f"features must predict the concept projection (R2={r2:.3f})"
    assert r2_sh < 0.1, f"shuffled target must give ~0 (R2={r2_sh:.3f})"
    assert r2 > r2_rand + 0.1, "concept projection must be more predictable than a random direction"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(features predict the CONCEPT-axis projection specifically -> they read it)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def run_real(args):
    z = np.load(args.npz)
    y = z["y"]; tr = z["is_train"]
    taps = [k for k in z.keys() if k.startswith("postL")] + (["final"] if "final" in z.keys() else [])
    taps = sorted(taps, key=lambda s: (9999 if s == "final" else int(s.replace("postL", ""))))

    A = np.load(args.act).astype(np.float64)
    if A.shape[0] < A.shape[1]:
        A = A.T                                       # -> (prompts, features)
    prompts = [json.loads(l) for l in open(args.prompts)]
    if len(prompts) != A.shape[0]:
        raise SystemExit(f"prompts {len(prompts)} != activation rows {A.shape[0]}")
    if A.shape[0] != len(y):
        raise SystemExit(f"activation rows {A.shape[0]} != npz prompts {len(y)}")
    groups = np.array([p[args.group_field] for p in prompts])
    logger.info("A=%s  h taps=%s  %d groups", A.shape, taps, len(set(groups.tolist())))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    res = {"taps": taps, "n_features": A.shape[1], "per_depth": []}

    for k in taps:
        H = z[k].astype(np.float64)
        w = fisher_axis(H[tr], y[tr], args.shrink)
        t = H @ w                                     # concept-axis projection
        t_auc = auc_of(t[~tr], y[~tr])                # sanity: projection separates classes

        r2, r2sd, _ = ridge_cv_r2(A, t, groups, args.n_splits, args.lam, args.seed)
        r2_sh, _, _ = ridge_cv_r2(A, rng.permutation(t), groups, args.n_splits, args.lam, args.seed)
        # random-direction target band
        rand_r2 = []
        for _ in range(args.n_rand):
            r = rng.standard_normal(H.shape[1]); r /= np.linalg.norm(r)
            rand_r2.append(ridge_cv_r2(A, H @ r, groups, args.n_splits, args.lam, args.seed)[0])
        rand_r2 = np.array(rand_r2)

        rec = {
            "tap": k, "projection_heldout_auc": t_auc,
            "feat_to_proj_r2": r2, "feat_to_proj_r2_std": r2sd,
            "shuffled_r2": r2_sh,
            "random_dir_r2_mean": float(rand_r2.mean()),
            "random_dir_r2_p95": float(np.percentile(rand_r2, 95)),
            "reads_concept": bool(r2 > 0.3 and r2 > np.percentile(rand_r2, 95) and r2 > r2_sh + 0.1),
        }
        res["per_depth"].append(rec)
        logger.info("%8s: proj-AUC=%.3f  feat->proj R2=%.3f+/-%.3f  shuffled=%.3f  randdir=%.3f  reads=%s",
                    k, t_auc, r2, r2sd, r2_sh, rand_r2.mean(), rec["reads_concept"])

    # best depth + verdict
    best = max(res["per_depth"], key=lambda r: r["feat_to_proj_r2"])
    res["best_tap"] = best["tap"]
    res["verdict"] = (
        f"FEATURES READ THE CONCEPT SUBSPACE: at {best['tap']}, the 227 feature-activations "
        f"predict the concept-axis projection <h,w_res> with held-out R^2={best['feat_to_proj_r2']:.3f} "
        f"(shuffled {best['shuffled_r2']:.3f}, random-direction {best['random_dir_r2_mean']:.3f}). "
        "The features ENCODE the concept's position in the residual stream -- a readout -- which "
        "explains their decodability (71) despite their write-directions being orthogonal to the "
        "concept axis (62/63). Carrier-as-correlate is now a measured fact, not inference-by-elimination."
        if best["reads_concept"] else
        f"NO SPECIFIC READOUT: best R^2={best['feat_to_proj_r2']:.3f} at {best['tap']} is not clearly "
        "above the random-direction band; features do not specifically encode the concept-axis "
        "projection. Report the negative honestly.")

    with open(out / "feature_acts_read_wres.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 84)
    print("DO FEATURE-ACTIVATIONS READ THE CONCEPT AXIS?  (chapter sec 4.4)")
    print("=" * 84)
    print(f"{'tap':>8} {'projAUC':>8} {'feat->proj R2':>14} {'shuffled':>9} {'randdir':>8} {'reads':>6}")
    for r in res["per_depth"]:
        print(f"{r['tap']:>8} {r['projection_heldout_auc']:>8.3f} {r['feat_to_proj_r2']:>14.3f} "
              f"{r['shuffled_r2']:>9.3f} {r['random_dir_r2_mean']:>8.3f} {str(r['reads_concept']):>6}")
    print("\nVERDICT: " + res["verdict"])
    print(f"\nwrote: {out}/feature_acts_read_wres.json")
    print("=" * 84)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--npz", default="data/analysis/runD_v2/geometry_stage1/h_residual_per_depth.npz")
    p.add_argument("--act", default="data/analysis/runD_v2/activations/activation_matrix.npy")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--group_field", default="surface_family")
    p.add_argument("--out_dir", default="feat_read_out")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--lam", type=float, default=10.0, help="ridge regularization")
    p.add_argument("--n_rand", type=int, default=20, help="random-direction target controls")
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
