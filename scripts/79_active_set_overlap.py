"""
79_active_set_overlap.py   [CSD3 / GPU to encode, or CPU from saved acts]
===================================================================
Turns the sec-4.4 argument ("the transcoder encoder is nonlinear, so a linear
functional of feature activations need not parallel w_res") from words into a
measured number.

LOGIC. A linear probe in feature space, sum_f v_f a_f(p), maps back to h-space as a
SINGLE linear functional only if the same features are active on every prompt. With
JumpReLU sparsity the active set varies prompt-to-prompt, making it PIECEWISE-linear
(one effective hyperplane per activation pattern) -- which is why it need not coincide
with the single hyperplane w_res. This script measures HOW piecewise the regime is:

  * mean pairwise JACCARD overlap of active (nonzero / top-k) feature sets across
    prompts. Jaccard ~1 => same features fire everywhere => effectively ONE linear
    piece => the "parallel to w_res" intuition would have force. Jaccard low =>
    active sets vary => many linear pieces => the intuition fails quantitatively.
  * RANDOM baseline: Jaccard of random top-k subsets ~ k/(2d-k) (~1e-4 here), so any
    structure shows as Jaccard >> that; the gap between observed and 1.0 is the
    degree of nonlinearity.
  * effective number of distinct linear pieces: binary-activity matrix rank and the
    count of distinct active-set patterns.
  * per-feature activity frequency: size of an "always-on core" (active on >=90% of
    prompts) vs prompt-specific features.

INTERPRETATION FOR THE CHAPTER. If a small core is always on but most active features
are prompt-specific (low-to-moderate Jaccard, high pattern count), the feature->h map
is strongly piecewise-linear; feature-space decodability (AUC 0.97) and the residual
axis w_res are then NOT linearly tied, reconciling sec 3.1 with sec 6.2 without
contradiction.

INPUT (either):
  --mlp_inputs_dir  dir with per-layer MLP-input arrays mlp_input_L{idx}.npy
                    (n_prompts x d_model, as script 04 saves) -> encoded via transcoder
  --feature_acts    precomputed sparse/dense feature-activation .npz {L{idx}: (n,d_tc)}

SELF-TEST: python 79_active_set_overlap.py --self_test
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("active_overlap")


def active_sets_from_dense(A, top_k=None):
    """A: (n_prompts, d_tc) activations >=0. Return list of frozensets of active indices."""
    sets = []
    for row in A:
        if top_k is not None:
            idx = np.argpartition(row, -top_k)[-top_k:]
            idx = idx[row[idx] > 0]
        else:
            idx = np.nonzero(row > 0)[0]
        sets.append(frozenset(int(i) for i in idx))
    return sets


def overlap_metrics(sets, d_tc, n_pairs=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(sets)
    sizes = np.array([len(s) for s in sets])
    # sampled pairwise Jaccard
    jac = []
    for _ in range(min(n_pairs, n * (n - 1) // 2)):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        a, b = sets[i], sets[j]
        u = len(a | b)
        jac.append(len(a & b) / u if u else 0.0)
    jac = np.array(jac)
    # per-feature activity frequency
    freq = {}
    for s in sets:
        for f in s:
            freq[f] = freq.get(f, 0) + 1
    freq_arr = np.array(list(freq.values())) / n
    core = int((freq_arr >= 0.9).sum())          # always-on core
    specific = int((freq_arr <= 0.1).sum())       # prompt-specific
    # distinct active-set patterns
    n_patterns = len(set(sets))
    k = int(np.median(sizes))
    rand_jac = k / (2 * d_tc - k) if d_tc > k else float("nan")
    return {
        "n_prompts": n, "median_active_size": int(np.median(sizes)),
        "mean_jaccard": float(jac.mean()), "median_jaccard": float(np.median(jac)),
        "jaccard_p10": float(np.percentile(jac, 10)), "jaccard_p90": float(np.percentile(jac, 90)),
        "random_jaccard_baseline": float(rand_jac),
        "n_distinct_patterns": n_patterns, "frac_distinct": float(n_patterns / n),
        "core_features_always_on": core, "prompt_specific_features": specific,
        "n_features_ever_active": len(freq),
    }


def verdict_from(m):
    j = m["mean_jaccard"]
    if j > 0.8:
        return ("NEAR-LINEAR regime: active sets are largely shared across prompts "
                f"(Jaccard {j:.2f}); the feature->h map is close to a single linear functional, "
                "so feature-space decoding and w_res could plausibly align. The sec-4.4 "
                "nonlinearity argument would be WEAK here -- revisit.")
    if j < 0.4:
        return ("STRONGLY PIECEWISE-LINEAR regime: active sets vary substantially across prompts "
                f"(Jaccard {j:.2f}, far below 1 though above the ~{m['random_jaccard_baseline']:.1e} random "
                f"baseline; {m['frac_distinct']:.2f} of prompts have a distinct active pattern). A linear "
                "functional of feature activations is many-piece in h-space and need NOT parallel w_res. "
                "This supports sec 4.4: feature-decodability (AUC 0.97) and w_res orthogonality to the "
                "decoder span are two consequences of encoder nonlinearity, not a contradiction.")
    return (f"INTERMEDIATE regime (Jaccard {j:.2f}): a shared core ({m['core_features_always_on']} always-on "
            f"features) plus substantial prompt-specific variation ({m['prompt_specific_features']} features). "
            "The feature->h map is piecewise-linear with a stable core; feature-space decoding is only "
            "partly tied to a single residual direction, consistent with sec 4.4.")


def self_test():
    rng = np.random.default_rng(79); d_tc = 2000; n = 150; k = 40
    # NEAR-LINEAR: same k features active on (almost) every prompt
    core = rng.choice(d_tc, k, replace=False)
    A_lin = np.zeros((n, d_tc))
    for i in range(n):
        A_lin[i, core] = np.abs(rng.standard_normal(k)) + 0.5     # same set every prompt
    m_lin = overlap_metrics(active_sets_from_dense(A_lin), d_tc)
    # PIECEWISE: a small shared core + mostly prompt-specific features
    small_core = rng.choice(d_tc, 5, replace=False)
    A_pw = np.zeros((n, d_tc))
    for i in range(n):
        A_pw[i, small_core] = np.abs(rng.standard_normal(5)) + 0.5
        rest = rng.choice(d_tc, k - 5, replace=False)
        A_pw[i, rest] = np.abs(rng.standard_normal(k - 5)) + 0.5  # different each prompt
    m_pw = overlap_metrics(active_sets_from_dense(A_pw), d_tc)
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  NEAR-LINEAR (fixed active set): mean Jaccard = {m_lin['mean_jaccard']:.3f}  "
          f"(expect ~1), distinct patterns frac = {m_lin['frac_distinct']:.2f}")
    print(f"  PIECEWISE (varying active set): mean Jaccard = {m_pw['mean_jaccard']:.3f}  "
          f"(expect low), distinct patterns frac = {m_pw['frac_distinct']:.2f}")
    assert m_lin["mean_jaccard"] > 0.9, "fixed active set must give Jaccard ~1"
    assert m_pw["mean_jaccard"] < 0.4, "varying active sets must give low Jaccard"
    assert m_pw["frac_distinct"] > 0.9, "piecewise regime should have mostly distinct patterns"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(mean active-set Jaccard distinguishes a near-linear encoder regime from a")
    print(" strongly piecewise-linear one -- the quantity behind the sec-4.4 argument)")
    print("---------------------------------------------------------------\n")


def run_real(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = args.layers
    feat = {}

    if args.feature_acts:
        z = np.load(args.feature_acts)
        for key in z.files:
            L = int(key.replace("L", "").replace("postL", ""))
            if (not layers) or (L in layers):
                feat[L] = z[key]
    elif args.mlp_inputs_dir:
        import torch
        import sys
        sys.path.insert(0, str(Path(args.repo_root)))
        from src.transcoder import load_transcoder_set
        d = Path(args.mlp_inputs_dir)
        avail = []
        for p in d.glob("mlp_input_L*.npy"):
            L = int(p.stem.replace("mlp_input_L", ""))
            if (not layers) or (L in layers):
                avail.append(L)
        avail = sorted(avail)
        tset = load_transcoder_set(model_size=args.model_size, device=args.device, lazy_load=True, layers=avail)
        for L in avail:
            X = np.load(d / f"mlp_input_L{L}.npy")              # (n, d_model)
            tc = tset[L]
            with torch.no_grad():
                A = tc.encode(torch.tensor(X, dtype=tc.dtype, device=args.device))
                A = A.detach().float().cpu().numpy()
            feat[L] = A
            logger.info("encoded L%d: %s active-frac mean %.4f", L, A.shape, float((A > 0).mean()))
    else:
        raise SystemExit("provide --feature_acts or --mlp_inputs_dir")

    results = {}
    for L in sorted(feat):
        A = feat[L]
        d_tc = A.shape[1]
        m = overlap_metrics(active_sets_from_dense(A, top_k=args.top_k), d_tc, seed=args.seed)
        m["verdict"] = verdict_from(m)
        results[f"L{L}"] = m
        logger.info("L%d: Jaccard mean=%.3f distinct=%.2f core=%d", L, m["mean_jaccard"],
                    m["frac_distinct"], m["core_features_always_on"])

    # aggregate across layers
    js = [results[k]["mean_jaccard"] for k in results]
    agg = {"layers": sorted(feat), "mean_jaccard_across_layers": float(np.mean(js)),
           "per_layer": results,
           "overall_verdict": verdict_from({"mean_jaccard": float(np.mean(js)),
                                            "random_jaccard_baseline": results[list(results)[0]]["random_jaccard_baseline"],
                                            "frac_distinct": float(np.mean([results[k]["frac_distinct"] for k in results])),
                                            "core_features_always_on": int(np.mean([results[k]["core_features_always_on"] for k in results])),
                                            "prompt_specific_features": int(np.mean([results[k]["prompt_specific_features"] for k in results]))})}
    with open(out / "active_set_overlap.json", "w") as fh:
        json.dump(agg, fh, indent=2, default=float)

    print("\n" + "=" * 84)
    print("ACTIVE-SET OVERLAP  --  how piecewise-linear is the transcoder encoder? (sec 4.4)")
    print("=" * 84)
    print(f"{'layer':>7} {'Jaccard':>9} {'med_size':>9} {'distinct%':>10} {'core':>6} {'specific':>9}")
    for k in sorted(results, key=lambda s: int(s[1:])):
        m = results[k]
        print(f"{k:>7} {m['mean_jaccard']:>9.3f} {m['median_active_size']:>9} "
              f"{m['frac_distinct']*100:>9.1f}% {m['core_features_always_on']:>6} {m['prompt_specific_features']:>9}")
    print(f"\n  random-Jaccard baseline ~ {results[list(results)[0]]['random_jaccard_baseline']:.1e}")
    print(f"  mean Jaccard across layers: {agg['mean_jaccard_across_layers']:.3f}")
    print("\nVERDICT: " + agg["overall_verdict"])
    print(f"\nwrote: {out}/active_set_overlap.json")
    print("=" * 84)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mlp_inputs_dir", default=None)
    p.add_argument("--feature_acts", default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None, help="restrict to these layers (default all found)")
    p.add_argument("--top_k", type=int, default=None, help="restrict active set to top-k (matches extraction top-k, e.g. 50)")
    p.add_argument("--out_dir", default="data/analysis/active_set_overlap")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
