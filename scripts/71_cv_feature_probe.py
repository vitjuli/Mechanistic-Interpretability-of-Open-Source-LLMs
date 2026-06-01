"""
71_cv_feature_probe.py   [LAPTOP / CPU-ONLY -- sklearn, seconds]
===================================================================
Validates the central REPRESENTATIONAL result on transcoder feature-activations:
the 82% linear-probe / cos=0.82 Fisher separation of alpha vs beta. Those were
reported WITHOUT cross-validation; a reviewer's first question is whether they
are in-sample artefacts (a probe overfitting 227 features to 538 prompts).

This answers it from data already on disk -- NO model, NO transcoders, NO GPU:
  activation_matrix.npy  (227 features x 538 prompts)  [transpose -> (538, 227)]
  feature_ids.txt        (227 ids; for optional per-feature reporting)
  prompts jsonl          (correct_answer, surface_family; line i == column i)

WHAT IT DOES
  * FAMILY-GROUPED k-fold CV (GroupKFold on surface_family): paraphrases of the
    same physical situation NEVER straddle train/test, so held-out AUC is honest.
    (A naive prompt-level split would leak paraphrases and inflate the score.)
  * Reports held-out AUC + accuracy (mean +/- std across folds) for:
      - Fisher/LDA (the cos=0.82 method) and Logistic Regression (the 82% probe).
  * SHUFFLED-LABEL control: same family-grouped CV on permuted labels. Real AUC
    should be high; shuffled ~0.5. If shuffled is also high => split leakage and
    the real number cannot be trusted.
  * Fisher DIRECTION STABILITY across folds (cosine between per-fold weight
    vectors) -- is there one stable separating direction, or fold-dependent?
  * Permutation p-value: where the real CV-AUC sits vs the shuffled distribution.

OUTPUT (default ./cv_probe_out/):
  cv_feature_probe.json   per-model CV AUC/acc, shuffled band, p-value, stability

SELF-TEST: python 71_cv_feature_probe.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from itertools import combinations
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cv_probe")


# =====================================================================
# Minimal, dependency-light implementations (work even without sklearn)
# =====================================================================

def group_kfold_indices(groups: np.ndarray, n_splits: int, seed: int = 0):
    """Yield (train_idx, test_idx) with whole groups held out (no group on both sides)."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for f in folds:
        test_groups = set(f.tolist())
        test_mask = np.array([g in test_groups for g in groups])
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def fisher_lda(X, y, shrink=0.1):
    """LDA direction w = Sigma_within^{-1}(mu1-mu0), shrunk; returns w (unit), threshold."""
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    X0, X1 = X[y == 0] - mu0, X[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(len(y) - 2, 1)
    Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    w = w / (np.linalg.norm(w) + 1e-30)
    thr = 0.5 * (mu0 @ w + mu1 @ w)
    return w, float(thr)


def logreg(X, y, l2=1.0, iters=300, lr=0.1):
    """Tiny L2 logistic regression via gradient descent (standardized X)."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = float(np.mean(p - y))
        w -= lr * gw; b -= lr * gb
    return w, b


def auc_score(scores, y) -> float:
    o = np.argsort(scores); r = np.empty_like(o, float); r[o] = np.arange(1, len(scores) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


# =====================================================================
# Core CV routine
# =====================================================================

def run_cv(X, y, groups, n_splits, seed, shrink, model="fisher"):
    """Return per-fold held-out AUC/acc and the per-fold weight vectors."""
    aucs, accs, ws = [], [], []
    for tr, te in group_kfold_indices(groups, n_splits, seed):
        Xtr, Xte = standardize(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
            continue
        if model == "fisher":
            w, thr = fisher_lda(Xtr, ytr, shrink)
            s = Xte @ w
            acc = max(np.mean((s > np.median(Xtr @ w)) == yte),
                      np.mean((s <= np.median(Xtr @ w)) == yte))
        else:
            w, b = logreg(Xtr, ytr)
            s = Xte @ w + b
            acc = np.mean((s > 0).astype(int) == yte)
        aucs.append(auc_score(s, yte)); accs.append(float(acc)); ws.append(w / (np.linalg.norm(w) + 1e-30))
    return np.array(aucs), np.array(accs), ws


def shuffled_band(X, y, groups, n_splits, seed, shrink, model, n_perm):
    rng = np.random.default_rng(seed + 1)
    means = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a, _, _ = run_cv(X, yp, groups, n_splits, rng.integers(1 << 30), shrink, model)
        if a.size:
            means.append(float(a.mean()))
    return np.array(means)


# =====================================================================
# Self-test
# =====================================================================

def self_test():
    rng = np.random.default_rng(71)
    nfam, per = 40, 10
    n = nfam * per; d = 60
    groups = np.repeat(np.arange(nfam), per)
    # family-level label so paraphrases share a class (realistic); signal in 5 features
    fam_label = rng.integers(0, 2, nfam); y = fam_label[groups]
    X = rng.standard_normal((n, d)) * 1.0
    X[:, :5] += (y[:, None] * 2 - 1) * 1.2          # real signal
    # a leaky feature that is constant within family (would inflate a naive split)
    X[:, 5] = groups * 0.01 + rng.standard_normal(n) * 0.01

    print("\n--- SELF TEST -------------------------------------------------")
    for model in ("fisher", "logreg"):
        a, acc, ws = run_cv(X, y, groups, 5, 0, 0.1, model)
        cos = [abs(float(ws[i] @ ws[j])) for i, j in combinations(range(len(ws)), 2)]
        sh = shuffled_band(X, y, groups, 5, 0, 0.1, model, 30)
        print(f"  {model:7s}: CV-AUC={a.mean():.3f}+/-{a.std():.3f}  acc={acc.mean():.3f}  "
              f"shuffled={sh.mean():.3f}  dir-stability cos={np.mean(cos):.2f}")
        assert a.mean() > 0.75, f"{model}: real CV-AUC must be high (got {a.mean():.3f})"
        assert sh.mean() < 0.62, f"{model}: shuffled must be ~chance (got {sh.mean():.3f})"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(family-grouped CV recovers real signal; shuffled-label ~0.5; no leakage)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run
# =====================================================================

def run_real(args):
    X = np.load(args.act).astype(np.float64)
    if X.shape[0] < X.shape[1]:
        # stored as (features, prompts) -> transpose to (prompts, features)
        X = X.T
    prompts = [json.loads(l) for l in open(args.prompts)]
    if len(prompts) != X.shape[0]:
        raise SystemExit(f"prompt count {len(prompts)} != activation rows {X.shape[0]}")
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    groups = np.array([p[args.group_field] for p in prompts])
    logger.info("X=%s  %d prompts  %d groups (%s)  balance: %d/%d",
                X.shape, len(prompts), len(set(groups.tolist())), args.group_field,
                int((y == 0).sum()), int((y == 1).sum()))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {"act_shape": list(X.shape), "n_prompts": len(prompts),
           "n_groups": len(set(groups.tolist())), "group_field": args.group_field,
           "n_splits": args.n_splits, "models": {}}

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa
        from sklearn.linear_model import LogisticRegression  # noqa
        from sklearn.model_selection import GroupKFold  # noqa
        from sklearn.metrics import roc_auc_score  # noqa
        from sklearn.preprocessing import StandardScaler  # noqa
        have_sklearn = True
    except Exception:
        have_sklearn = False
    logger.info("sklearn available: %s (falling back to builtin if not)", have_sklearn)

    def sklearn_cv(model_name):
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        gkf = GroupKFold(n_splits=args.n_splits)
        aucs, accs, ws = [], [], []
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if model_name == "fisher":
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[tr])
                s = m.decision_function(Xte)
                ws.append(m.coef_.ravel() / (np.linalg.norm(m.coef_.ravel()) + 1e-30))
            else:
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
                s = m.decision_function(Xte)
                ws.append(m.coef_.ravel() / (np.linalg.norm(m.coef_.ravel()) + 1e-30))
            aucs.append(roc_auc_score(y[te], s)); accs.append(m.score(Xte, y[te]))
        return np.array(aucs), np.array(accs), ws

    def sklearn_shuffled(model_name, n_perm):
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(args.seed + 7); means = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            gkf = GroupKFold(n_splits=args.n_splits); aa = []
            for tr, te in gkf.split(X, yp, groups):
                sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                if model_name == "fisher":
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, yp[tr])
                else:
                    from sklearn.linear_model import LogisticRegression
                    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, yp[tr])
                aa.append(roc_auc_score(yp[te], m.decision_function(Xte)))
            means.append(float(np.mean(aa)))
        return np.array(means)

    for model_name in ("fisher", "logreg"):
        if have_sklearn:
            aucs, accs, ws = sklearn_cv(model_name)
            shuf = sklearn_shuffled(model_name, args.n_perm)
        else:
            aucs, accs, ws = run_cv(X, y, groups, args.n_splits, args.seed, args.shrink, model_name)
            shuf = shuffled_band(X, y, groups, args.n_splits, args.seed, args.shrink, model_name, args.n_perm)
        cos = [abs(float(ws[i] @ ws[j])) for i, j in combinations(range(len(ws)), 2)] if len(ws) > 1 else [float("nan")]
        pval = float((shuf >= aucs.mean()).mean()) if shuf.size else float("nan")
        res["models"][model_name] = {
            "cv_auc_mean": float(aucs.mean()), "cv_auc_std": float(aucs.std()),
            "cv_acc_mean": float(accs.mean()), "cv_acc_std": float(accs.std()),
            "fold_aucs": aucs.tolist(),
            "shuffled_auc_mean": float(shuf.mean()) if shuf.size else None,
            "shuffled_auc_p95": float(np.percentile(shuf, 95)) if shuf.size else None,
            "permutation_pvalue": pval,
            "direction_stability_cos_mean": float(np.mean(cos)),
            "validated": bool(aucs.mean() > 0.7 and (shuf.size == 0 or aucs.mean() > np.percentile(shuf, 95))),
        }
        logger.info("%s: CV-AUC=%.3f+/-%.3f acc=%.3f shuffled=%.3f p=%.3g dir-stab=%.2f",
                    model_name, aucs.mean(), aucs.std(), accs.mean(),
                    shuf.mean() if shuf.size else float("nan"), pval, np.mean(cos))

    f = res["models"]["fisher"]
    res["verdict"] = (
        f"VALIDATED: family-grouped CV AUC={f['cv_auc_mean']:.3f}+/-{f['cv_auc_std']:.3f} "
        f"(shuffled {f['shuffled_auc_mean']:.3f}, p={f['permutation_pvalue']:.3g}). The linear "
        "separability of alpha/beta in feature-activation space is REAL (not in-sample/leakage). "
        "Note: this is observational decodability of feature activations, consistent with the "
        "residual-stream finding (decodable but not localizable/causal)."
        if f["validated"] else
        f"NOT VALIDATED: CV AUC={f['cv_auc_mean']:.3f} not clearly above shuffled "
        f"{f['shuffled_auc_mean']:.3f} -> the 82%/0.82 figure may be in-sample; report with caution.")

    with open(out / "cv_feature_probe.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print("CV FEATURE-ACTIVATION PROBE  (family-grouped; validates 82%/cos=0.82)")
    print("=" * 78)
    print(f"  X={tuple(res['act_shape'])}  {res['n_groups']} {args.group_field} groups, {args.n_splits}-fold")
    for mn, m in res["models"].items():
        print(f"\n  {mn.upper()}:")
        print(f"    held-out AUC = {m['cv_auc_mean']:.3f} +/- {m['cv_auc_std']:.3f}   "
              f"acc = {m['cv_acc_mean']:.3f}")
        print(f"    shuffled-label AUC = {m['shuffled_auc_mean']:.3f} (p95 {m['shuffled_auc_p95']:.3f})  "
              f"perm p = {m['permutation_pvalue']:.3g}")
        print(f"    direction stability across folds: cos = {m['direction_stability_cos_mean']:.2f}")
    print(f"\n  {res['verdict']}")
    print(f"\n  wrote: {out}/cv_feature_probe.json")
    print("=" * 78)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--act", default="data/analysis/.../activation_matrix.npy",
                   help="activation_matrix.npy (features x prompts or prompts x features)")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--group_field", default="surface_family")
    p.add_argument("--out_dir", default="cv_probe_out")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_perm", type=int, default=50, help="shuffled-label permutations")
    p.add_argument("--shrink", type=float, default=0.1, help="LDA shrinkage (builtin fallback only)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
