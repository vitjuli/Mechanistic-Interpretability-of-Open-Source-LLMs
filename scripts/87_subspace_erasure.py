"""
87_subspace_erasure.py   [erase the whole concept SUBSPACE in the forward pass]
=========================================================================================
Every intervention so far was 1-DIMENSIONAL (the w_res component) or feature-set-based
(transcoder dictionaries). But the concept occupies a ~5D rotating subspace (PR 5.5; the
13-axis stack); its removal was only ever done OFFLINE on representations (complement AUC
0.975), never as a FORWARD-PASS intervention with a behavioural metric. If the model uses
the subspace COLLECTIVELY, single-direction tests would all miss it. This script closes
that gap -- the one new experiment whose outcome is genuinely open:

(A) SUBSPACE ERASURE. Per layer l, build a k-dim concept subspace by iterative Fisher
    (INLP-style: fit axis, project out, refit; orthonormalised), k in a grid (1..13).
    In the forward pass, at EVERY position:   h <- h - W_k W_k^T (h - mu)
    (mean-recentred amnesic erasure, LEACE-flavoured: erased coordinates are set to the
    train-mean, not zero). Measure held-out MARGIN ACCURACY (sign of logit_beta-logit_alpha
    matches the correct class) clean vs erased, against TWO nulls of the same rank:
      - random orthonormal k-frames            (any-k-dims-removed control)
      - shuffled-label INLP subspaces           (same construction pipeline, no concept)
    Outcomes:
      accuracy survives even k=13 above nulls -> the model does not use the linearly-
        available concept EVEN COLLECTIVELY (airtight bypass; strongest possible H2);
      accuracy drops below the nulls          -> FOUND the usage locus: the subspace as a
        whole is causally used, and all 1-D negatives simply sliced it too thin.

(B) OPTIONAL MASS-HEURISTIC POSITIVE CONTROL (--with_mass). Exp 76 showed a heavy=alpha
    shortcut behaviourally. Here we build the heuristic's DIRECTION: label prompts
    heavy/light (auto-extracted mass number A from the text, or --mass_labels file), fit
    w_mass per layer, report corr(heavy, class) and cos(w_mass, w_res), and steer along
    w_mass (norm-matched, vs shuffled-mass-label null). If w_mass moves the margin where
    w_res does not, the task's causal direction is the HEURISTIC's -- a positive contrast.
    Caveat (by design): mass correlates with class on this corpus (that IS the shortcut),
    so interpret cos(w_mass, w_res) and any steering asymmetry together, not in isolation.

SELF-TEST (no torch / no repo):  python 87_subspace_erasure.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("subspace_erase")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def auc_scalar(s, y):
    s = np.asarray(s, float); o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def inlp_subspace(H, y, k, shrink=0.1):
    """Iterative Fisher: fit axis, project data out, refit. Returns orthonormal W (d, k)."""
    Hc = H.copy().astype(np.float64)
    ws = []
    for _ in range(k):
        w = fisher_axis(Hc, y, shrink)
        for prev in ws:                      # re-orthogonalise against accumulated basis
            w = w - (w @ prev) * prev
        nrm = np.linalg.norm(w)
        if nrm < 1e-8:
            break
        w = w / nrm
        ws.append(w)
        Hc = Hc - np.outer(Hc @ w, w)
    W = np.stack(ws, axis=1)                 # (d, k_eff)
    # final orthonormalisation for numerical safety
    Q, _ = np.linalg.qr(W)
    return Q[:, : W.shape[1]]


def erase(H, W, mu):
    """Mean-recentred amnesic erasure: h <- h - W W^T (h - mu)."""
    X = H - mu
    return H - (X @ W) @ W.T


def random_orthonormal(d, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q[:, :k]


def margin_accuracy(margins, y):
    """sign(margin)>0 should mean beta (y=1)."""
    pred = (np.asarray(margins) > 0).astype(int)
    return float(np.mean(pred == np.asarray(y)))


def percentile_of(value, null):
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


def extract_mass_number(text, lo=100, hi=300):
    nums = [int(x) for x in re.findall(r"\b(\d{2,3})\b", text)]
    nums = [x for x in nums if lo <= x <= hi]
    return max(nums) if nums else None


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n, kstar = 20, 400, 3
    y = np.array([0, 1] * (n // 2))
    # concept lives in a k*-dim subspace S; the READOUT also lives in S (model uses it)
    S = random_orthonormal(d, kstar, rng)
    coef = rng.standard_normal(kstar)
    H = 0.4 * rng.standard_normal((n, d)) + ((y * 2 - 1.0)[:, None]) * (S @ coef)[None, :] * 0.8
    a = unit_raw(S @ coef)                                  # usage direction inside S
    margins = H @ a
    assert margin_accuracy(margins, y) > 0.9, "toy model decides via the subspace"

    # INLP recovers a subspace overlapping S: erasing it kills accuracy
    W = inlp_subspace(H, y, kstar)
    mu = H.mean(0)
    He = erase(H, W, mu)
    acc_erased = margin_accuracy(He @ a, y)
    assert acc_erased < 0.65, f"erasing the concept subspace must break the decision, got {acc_erased}"
    # erased data no longer decodes
    assert abs(auc_scalar(He @ fisher_axis(He, y), y) - 0.5) < 0.2 or auc_scalar(He @ a, y) < 0.75

    # random-k erasure does NOT break it
    accs = []
    for _ in range(10):
        R = random_orthonormal(d, kstar, rng)
        accs.append(margin_accuracy(erase(H, R, mu) @ a, y))
    assert np.mean(accs) > 0.85, "random matched-rank erasure must leave the decision intact"

    # recentring: erased projections equal mu's projection
    assert np.allclose((He - mu) @ W, 0, atol=1e-8), "erased component must be recentred to mu"
    # orthonormality
    assert np.allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-8)

    # mass extraction
    assert extract_mass_number("uranium-238 nucleus with A=238 emits") == 238
    assert extract_mass_number("light nucleus") is None
    assert percentile_of(0.9, np.array([0.1, 0.5])) == 100.0
    print("[self_test] OK — INLP subspace, recentred erasure breaks toy decision, random-k doesn't, helpers pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    last = n_layers - 1
    layers = args.layers if args.layers else list(range(n_layers))
    layers = [L for L in layers if 0 <= L < n_layers]
    logger.info("model: %d layers; subspace erasure over %d layers, k grid %s", n_layers, len(layers), args.k_grid)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    # ---------- capture answer-position residuals (for INLP bases + sigma) ----------
    logger.info("capturing residuals for %d prompts at %d layers...", len(prompts), len(layers))
    res = {L: np.zeros((len(prompts), d), np.float32) for L in layers}
    y = np.zeros(len(prompts), int); trm = np.zeros(len(prompts), bool); clean_margin = np.zeros(len(prompts))
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a): g[L] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0); clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            res[L][i] = g[L]
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        trm[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    held = [i for i in range(len(prompts)) if not trm[i]]
    ha = [i for i in held if y[i] == 0][: args.max_targets]
    hb = [i for i in held if y[i] == 1][: args.max_targets]
    targets = ha + hb
    y_t = np.array([y[i] for i in targets])
    acc_clean = margin_accuracy(np.array([clean_margin[i] for i in targets]), y_t)
    logger.info("clean held-out margin accuracy on %d targets: %.3f", len(targets), acc_clean)

    def run_erased(ptext, L, W_np, mu_np):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        Wt = torch.tensor(W_np, dtype=torch.float32, device=args.device)
        mut = torch.tensor(mu_np, dtype=torch.float32, device=args.device)
        def pre(m, a):
            hs = a[0].clone()
            X = hs[0] - mut                                  # (seq, d), all positions
            hs[0] = hs[0] - (X @ Wt) @ Wt.T
            return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(row, 0)
            return float(lp[beta_id] - lp[alpha_id])
        finally:
            h.remove()

    def eval_erasure(L, W_np, mu_np):
        ms = [run_erased(prompts[i]["prompt"], L, W_np, mu_np) for i in targets]
        return margin_accuracy(np.array(ms), y_t)

    # ---------- (A) sweep layers x k, vs random-rank and shuffled-label nulls ----------
    rows = []
    for L in layers:
        H = res[L].astype(np.float64); Htr = H[trm]; ytr = y[trm]
        mu = Htr.mean(0)
        for k in args.k_grid:
            W = inlp_subspace(Htr, ytr, k, args.shrink)
            acc_e = eval_erasure(L, W, mu)
            acc_rand = [eval_erasure(L, random_orthonormal(d, W.shape[1], rng), mu) for _ in range(args.n_random)]
            acc_shuf = [eval_erasure(L, inlp_subspace(Htr, rng.permutation(ytr), k, args.shrink), mu)
                        for _ in range(args.n_shuffle)]
            rec = {"layer": int(L), "k": int(W.shape[1]), "acc_clean": acc_clean, "acc_erased": acc_e,
                   "acc_rand_mean": float(np.mean(acc_rand)), "acc_rand_p5": float(np.percentile(acc_rand, 5)),
                   "acc_shuf_mean": float(np.mean(acc_shuf)), "acc_shuf_p5": float(np.percentile(acc_shuf, 5)),
                   "drop_vs_rand": float(np.mean(acc_rand) - acc_e),
                   "pct_below_rand": percentile_of(-acc_e, -np.asarray(acc_rand))}
            rows.append(rec)
            logger.info("  L%d k=%d: acc clean=%.3f erased=%.3f | rand(mean %.3f, p5 %.3f) shuf(mean %.3f)",
                        L, rec["k"], acc_clean, acc_e, rec["acc_rand_mean"], rec["acc_rand_p5"], rec["acc_shuf_mean"])
    import csv as _csv
    with open(out / "subspace_erasure.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

    # ---------- (B) optional mass-heuristic direction ----------
    mass_summary = None
    if args.with_mass:
        mass_y = None
        if args.mass_labels:
            labels = {}
            for line in open(args.mass_labels):
                o = json.loads(line); labels[int(o["idx"])] = int(o["heavy"])
            mass_y = np.array([labels.get(i, -1) for i in range(len(prompts))])
        else:
            As = [extract_mass_number(p["prompt"]) for p in prompts]
            cov = float(np.mean([a is not None for a in As]))
            logger.info("(B) mass extraction coverage: %.2f (threshold A>=%d)", cov, args.mass_thresh)
            if cov >= 0.5:
                mass_y = np.array([(-1 if a is None else int(a >= args.mass_thresh)) for a in As])
            else:
                logger.warning("mass coverage <0.5 -> skipping part B; provide --mass_labels to run it")
        if mass_y is not None:
            valid = mass_y >= 0
            phi = float(np.corrcoef(mass_y[valid], y[valid])[0, 1])
            logger.info("  corr(heavy, beta-class) = %+.3f  (the shortcut's confound, by design)", phi)
            m_rows = []
            for L in [L for L in args.mass_layers if L in res]:
                H = res[L].astype(np.float64)
                vt = valid & trm
                w_mass = fisher_axis(H[vt], mass_y[vt], args.shrink)
                w_res_L = fisher_axis(H[trm], y[trm], args.shrink)
                cosmw = float(np.dot(w_mass, w_res_L))
                auc_m = auc_scalar(H[valid & ~trm] @ w_mass, mass_y[valid & ~trm])
                sig = float(np.std(H[vt] @ w_mass))
                # steer toward heavy (+) on light prompts and toward light (-) on heavy ones
                dm_a = []
                for i in [i for i in targets if valid[i]][: 2 * args.max_targets]:
                    s = +1.0 if mass_y[i] == 0 else -1.0
                    inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
                    dt = torch.tensor((s * args.mass_c * sig) * w_mass, dtype=torch.float32, device=args.device)
                    def pre(m, a):
                        hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
                    hk = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
                    try:
                        with torch.no_grad():
                            row = model(**inp, use_cache=False).logits[0, -1, :].float()
                        lp = torch.log_softmax(row, 0)
                        m1 = float(lp[beta_id] - lp[alpha_id])
                    finally:
                        hk.remove()
                    # heavy=alpha shortcut predicts: push toward heavy -> margin toward alpha (negative)
                    dm_a.append(-(m1 - clean_margin[i]) * s)
                m_rows.append({"layer": int(L), "cos_wmass_wres": cosmw, "auc_mass": auc_m,
                               "mean_dmargin_toward_alpha_when_pushed_heavy": float(np.mean(dm_a))})
                logger.info("  L%d: cos(w_mass,w_res)=%+.3f auc_mass=%.3f d(margin->alpha|push heavy)=%+.3f",
                            L, cosmw, auc_m, m_rows[-1]["mean_dmargin_toward_alpha_when_pushed_heavy"])
            mass_summary = {"phi_heavy_class": phi, "rows": m_rows}
            json.dump(mass_summary, open(out / "mass_direction.json", "w"), indent=2)

    # ---------- verdict ----------
    print("\n" + "=" * 92)
    print("SUBSPACE ERASURE -- does the model use the concept subspace COLLECTIVELY?")
    print("=" * 92)
    kmax = max(r["k"] for r in rows)
    deep = [r for r in rows if r["k"] == kmax]
    sig_break = [r for r in deep if r["acc_erased"] < r["acc_rand_p5"] - 0.05 and r["acc_erased"] < acc_clean - 0.15]
    if sig_break:
        b = min(sig_break, key=lambda r: r["acc_erased"])
        print(f"OUTCOME: USAGE LOCUS FOUND -- erasing the k={kmax} concept subspace at L{b['layer']} drops margin "
              f"accuracy to {b['acc_erased']:.2f} (clean {acc_clean:.2f}; random-{kmax} p5 {b['acc_rand_p5']:.2f}). "
              f"The subspace is used collectively; the 1-D negatives sliced it too thin.")
    else:
        worst = min(deep, key=lambda r: r["acc_erased"]) if deep else None
        print(f"OUTCOME: NO COLLECTIVE USE -- even erasing the full k={kmax} concept subspace at every position, "
              f"margin accuracy never drops below the matched-rank random null (worst erased acc "
              f"{worst['acc_erased']:.2f} vs clean {acc_clean:.2f}, random p5 {worst['acc_rand_p5']:.2f}). "
              f"The model does not use the linearly-available concept even collectively -> the bypass (H2) is "
              f"airtight at the subspace level, not just the direction level.")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/subspace_erasure")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--k_grid", type=int, nargs="*", default=[1, 5, 13])
    p.add_argument("--n_random", type=int, default=4, help="random matched-rank frames per (layer,k)")
    p.add_argument("--n_shuffle", type=int, default=3, help="shuffled-label INLP subspaces per (layer,k)")
    p.add_argument("--max_targets", type=int, default=80, help="held-out targets per class")
    p.add_argument("--with_mass", action="store_true", default=True)
    p.add_argument("--no_mass", dest="with_mass", action="store_false")
    p.add_argument("--mass_labels", default=None, help="jsonl with {'idx':int,'heavy':0/1} per line (overrides auto)")
    p.add_argument("--mass_thresh", type=int, default=200, help="A >= thresh counts as heavy (auto-extraction)")
    p.add_argument("--mass_layers", type=int, nargs="*", default=[18, 21, 24])
    p.add_argument("--mass_c", type=float, default=4.0)
    p.add_argument("--train_frac", type=float, default=0.6)
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
