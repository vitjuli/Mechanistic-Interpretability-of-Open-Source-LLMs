"""
86_usage_direction.py   [build the USAGE direction: what does the model itself read?]
=========================================================================================
Sections 6.1-6.4 established what the concept axis w_res is NOT (not a lever anywhere).
This script CONSTRUCTS the missing third object: the USAGE direction -- the direction the
model's own downstream computation reads when forming the answer -- and measures its
geometry against the reading axis w_res and the unembedding contrast gamma_bar.

Two independent constructions:

(A) GRADIENT usage direction (the linearisation behind attribution patching / AtP*,
    Kramar et al. 2024):  u_i^(l) = d(logit_beta - logit_alpha)/d h^(l)  at the answer
    position, per prompt, per layer (one forward+backward per prompt captures ALL layers).
    Diagnostics per layer:
      - cos(u_bar, w_res) raw + causal      <- THE reading/usage angle (headline number)
      - does u decode alpha/beta? (held-out AUC of h.u_bar)  <- if NO, the model's local
        readout is CONCEPT-BLIND: the formal signature of a shortcut-driven decision
      - cos(u_alpha_bar, u_beta_bar)        <- class-conditional structure of the readout
      - adjacent-layer rotation of u vs rotation of w_res
      - cos(u_final, gamma_bar)             <- sanity: at the final tap the gradient must
        align with the unembedding contrast (validates the machinery)

(B) DLA answer-writers vs axis-writers: decompose the final margin into per-head direct
    contributions DLA_{i,L,h} = ((v_{L,h} * rmsnorm_weight)/rms_i) . (W_U[beta]-W_U[alpha])
    (frozen-rms approximation, standard DLA practice), rank all heads by class-separation
    of DLA (answer-writing), and compare with the axis-writing ranking (d_along from exp
    82). If the two top-50 sets are (near-)disjoint and rank-correlation is low, then THE
    HEADS THAT WRITE THE ANSWER ARE NOT THE HEADS THAT WRITE THE READABLE AXIS -- the
    component-level statement of reading != usage.

(C) Steering efficiency: norm-matched pushes along unit(u_bar) vs unit(w_res) vs random,
    same absolute magnitudes (c * sigma_res). u moving the margin more per unit norm is
    EXPECTED BY CONSTRUCTION (it is the steepest direction); the informative quantity is
    the efficiency RATIO eff(w_res)/eff(u): how much of the locally-steerable margin
    movement the reading axis captures. Near 0 = full bypass. intact-flip reported too.

Honest framing: (A) is guaranteed to produce a new OBJECT (the usage vector with angles);
the discoveries to look for are (i) the angle cos(u, w_res), (ii) whether u decodes the
concept, (iii) the writer-ranking overlap. None of these are foregone.

SELF-TEST (no torch / no repo):  python 86_usage_direction.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("usage_dir")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
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


def cohens_d(s, y):
    a, b = s[y == 0], s[y == 1]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) + 1e-12
    return float((b.mean() - a.mean()) / pooled)


def cosine(a, b):
    return float(np.dot(unit_raw(np.asarray(a, float)), unit_raw(np.asarray(b, float))))


def whitener(Sigma, eps=1e-3):
    Sigma = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps * float(vals.max()) + 1e-12, None)
    return (vecs * (vals ** -0.5)) @ vecs.T


def causal_cos(a, b, W):
    return cosine(W @ np.asarray(a, float), W @ np.asarray(b, float))


def spearman(a, b):
    """Rank correlation, pure numpy."""
    def rank(x):
        o = np.argsort(x); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-30))


def finite_diff_grad(f, h, eps=1e-5):
    g = np.zeros_like(h)
    for j in range(len(h)):
        e = np.zeros_like(h); e[j] = eps
        g[j] = (f(h + e) - f(h - e)) / (2 * eps)
    return g


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 14, 200
    # usage direction a; reading direction w nearly orthogonal to a
    a = unit_raw(rng.standard_normal(d))
    w = rng.standard_normal(d); w -= (w @ a) * a; w = unit_raw(w); w = unit_raw(w + 0.05 * a)
    # classes separated along w ONLY; margin readout = h.a  (concept-blind usage)
    y = np.array([0, 1] * (n // 2))
    H = 0.4 * rng.standard_normal((n, d)) + np.outer((y * 2 - 1.0) * 1.5, w)

    margin = lambda h: float(h @ a)
    g = finite_diff_grad(margin, H[0])
    assert cosine(g, a) > 0.999, "gradient must recover the usage direction"
    assert abs(cosine(g, w)) < 0.2, "usage nearly orthogonal to reading in this toy"

    # usage direction does NOT decode the concept (classes not separated along a)
    auc_u = auc_scalar(H @ a, y); auc_w = auc_scalar(H @ w, y)
    assert auc_w > 0.95 and abs(auc_u - 0.5) < 0.15, f"reading decodes ({auc_w}), usage doesn't ({auc_u})"

    # steering efficiency: pushing along a moves the margin fully; along w barely
    eff_a = margin(H[0] + a) - margin(H[0]); eff_w = margin(H[0] + w) - margin(H[0])
    assert eff_a > 0.99 and abs(eff_w) < 0.2, "per-unit-norm margin efficiency separates a from w"

    # DLA toy: head 0 writes along the answer contrast, head 1 writes along w_res
    wU_diff = a  # in the toy the unembedding contrast IS the usage direction
    n_heads = 4
    V = 0.1 * rng.standard_normal((n, n_heads, d))
    V[:, 0] += np.outer((y * 2 - 1.0), wU_diff)       # answer-writer
    V[:, 1] += np.outer((y * 2 - 1.0), w)             # axis-writer
    dla = V @ wU_diff                                  # (n, heads)
    along = V @ w
    answer_d = np.array([abs(cohens_d(dla[:, h], y)) for h in range(n_heads)])
    axis_d = np.array([abs(cohens_d(along[:, h], y)) for h in range(n_heads)])
    assert int(np.argmax(answer_d)) == 0 and int(np.argmax(axis_d)) == 1, "rankings must separate the two writers"
    assert spearman(np.array([1, 2, 3, 4.0]), np.array([1, 2, 3, 4.0])) > 0.99
    print("[self_test] OK — gradient recovery, concept-blind usage, efficiency, DLA ranking separation pass.")


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
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", d // n_heads))
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    W_U = _chain(model, "lm_head").weight.detach().float().cpu().numpy()
    wU_diff = (W_U[beta_id] - W_U[alpha_id]).astype(np.float64)
    norm_w = norm_mod.weight.detach().float().cpu().numpy().astype(np.float64)
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
    last = n_layers - 1
    layers = args.layers if args.layers else list(range(n_layers))
    layers = [L for L in layers if 0 <= L < n_layers]
    o_proj = {L: _chain(blocks[L], "self_attn.o_proj") for L in layers}
    W_O = {L: o_proj[L].weight.detach().float().cpu().numpy() for L in layers}
    logger.info("model: %d layers, %d heads; usage-direction over %d taps", n_layers, n_heads, len(layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    # ---------- capture: residual + GRADIENT of raw margin + o_proj inputs, one fwd+bwd ----------
    logger.info("capturing residuals + margin-gradients + head outputs for %d prompts...", len(prompts))
    for p_ in model.parameters():
        p_.requires_grad_(True)  # graph needed for activation grads
    nP = len(prompts)
    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    grad = {L: np.zeros((nP, d), np.float32) for L in layers}
    zin = {L: np.zeros((nP, n_heads * head_dim), np.float32) for L in layers}
    rms_final = np.zeros(nP, np.float32)
    y = np.zeros(nP, int); trm = np.zeros(nP, bool); clean_margin = np.zeros(nP)
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        keep = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            def mkz(L=L):
                def pre(m, a):
                    keep[f"z{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(o_proj[L].register_forward_pre_hook(mkz(), with_kwargs=False))
        try:
            o = model(**inp, use_cache=False)
            row = o.logits[0, -1, :]
            raw_margin = row[beta_id] - row[alpha_id]
            raw_margin.backward()
            lp = torch.log_softmax(row.detach().float(), 0)
            clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            t = keep[L]
            res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
            grad[L][i] = (t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0)
            zin[L][i] = keep[f"z{L}"]
        hf = res[layers[-1]][i] if layers[-1] == last else None
        if last in layers:
            hf = res[last][i]
        rms_final[i] = float(np.sqrt(np.mean(hf.astype(np.float64) ** 2) + rms_eps)) if hf is not None else 1.0
        model.zero_grad(set_to_none=True)
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        trm[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    # ---------- (A) per-layer usage geometry ----------
    logger.info("(A) usage-direction geometry per layer...")
    geo = []; u_bar = {}; w_res = {}
    prev_u = None; prev_w = None
    for L in layers:
        H = res[L].astype(np.float64); G = grad[L].astype(np.float64)
        wL = fisher_axis(H[trm], y[trm], args.shrink); w_res[L] = wL
        uL = unit_raw(G.mean(0)); u_bar[L] = uL
        u_a = unit_raw(G[y == 0].mean(0)); u_b = unit_raw(G[y == 1].mean(0))
        Sig = np.cov(H.T); Wwh = whitener(Sig)
        per_prompt_cos = np.array([cosine(G[i], wL) for i in range(nP)])
        rec = {"layer": int(L),
               "cos_u_wres": cosine(uL, wL), "causal_cos_u_wres": causal_cos(uL, wL, Wwh),
               "auc_along_u": auc_scalar((H[~trm]) @ uL, y[~trm]),
               "auc_along_wres": auc_scalar((H[~trm]) @ wL, y[~trm]),
               "cos_u_alpha_beta": cosine(u_a, u_b),
               "mean_perprompt_cos_g_wres": float(per_prompt_cos.mean()),
               "grad_norm_mean": float(np.linalg.norm(G, axis=1).mean()),
               "cos_u_prev": cosine(uL, prev_u) if prev_u is not None else None,
               "cos_wres_prev": cosine(wL, prev_w) if prev_w is not None else None}
        prev_u, prev_w = uL, wL
        geo.append(rec)
        if (L % 4 == 0) or (L == layers[-1]):
            logger.info("  L%d: cos(u,w_res)=%+.3f (causal %+.3f) | AUC along u=%.3f (w_res %.3f) | cos(u_a,u_b)=%.3f",
                        L, rec["cos_u_wres"], rec["causal_cos_u_wres"], rec["auc_along_u"],
                        rec["auc_along_wres"], rec["cos_u_alpha_beta"])
    if last in layers:
        cf = cosine(u_bar[last], wU_diff)
        logger.info("  sanity: cos(u_final, gamma_bar) = %+.3f (machinery valid if high)", cf)
    else:
        cf = float("nan")

    # ---------- (B) DLA answer-writers vs axis-writers ----------
    logger.info("(B) DLA: ranking all heads as ANSWER-writers vs AXIS-writers...")
    rows = []
    for L in layers:
        Z = zin[L].astype(np.float64).reshape(nP, n_heads, head_dim)
        Wt = W_O[L].T.reshape(n_heads, head_dim, -1)
        V = np.einsum("phk,hkd->phd", Z, Wt)                    # head contributions (n, H, d)
        dla = ((V * norm_w[None, None, :]) @ wU_diff) / rms_final[:, None]   # (n, H)
        along = V @ w_res[L]
        for h in range(n_heads):
            rows.append({"layer": int(L), "head": int(h),
                         "answer_d": abs(cohens_d(dla[:, h], y)),
                         "axis_d": abs(cohens_d(along[:, h], y))})
    ans = np.array([r["answer_d"] for r in rows]); axd = np.array([r["axis_d"] for r in rows])
    rho = spearman(ans, axd)
    topA = set(int(i) for i in np.argsort(-ans)[:50]); topX = set(int(i) for i in np.argsort(-axd)[:50])
    overlap = len(topA & topX)
    rsort = sorted(rows, key=lambda r: -r["answer_d"])
    from collections import Counter
    hist_ans = Counter(r["layer"] for r in rsort[:50])
    logger.info("  spearman(answer_d, axis_d) over %d heads = %.3f | top-50 overlap = %d/50", len(rows), rho, overlap)
    logger.info("  top-10 ANSWER-writers: %s", ", ".join(f"L{r['layer']}.H{r['head']}({r['answer_d']:.2f})" for r in rsort[:10]))
    logger.info("  answer-writer top-50 layer histogram: %s", dict(sorted(hist_ans.items(), key=lambda t: -t[1])))

    # ---------- (C) steering efficiency u vs w_res vs random ----------
    logger.info("(C) steering efficiency (norm-matched) along u vs w_res vs random...")
    held = [i for i in range(nP) if not trm[i]]
    ha = [i for i in held if y[i] == 0][: args.max_targets]
    hb = [i for i in held if y[i] == 1][: args.max_targets]
    targets = [(i, "beta") for i in ha] + [(i, "alpha") for i in hb]
    answer_ids = {alpha_id, beta_id}

    def run_steer(ptext, L, delta):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
        def pre(m, a):
            hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(row, 0)
            return float(lp[beta_id] - lp[alpha_id]), int(row.argmax().item())
        finally:
            h.remove()

    eff_rows = []
    steer_layers = args.steer_layers if args.steer_layers else layers
    for L in [L for L in steer_layers if L in w_res]:
        sig = float(np.std(res[L][trm].astype(np.float64) @ w_res[L]))
        dirs = {"usage": u_bar[L], "w_res": w_res[L]}
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        for c in args.c_grid:
            for name, vec in dirs.items():
                dm, fl, it = [], [], []
                for i, toward in targets:
                    s = +1.0 if toward == "beta" else -1.0
                    m1, t1 = run_steer(prompts[i]["prompt"], L, (s * c * sig) * unit_raw(vec))
                    dm.append((m1 - clean_margin[i]) * s)
                    fl.append(int(clean_margin[i] < 0 and m1 > 0) if toward == "beta"
                              else int(clean_margin[i] > 0 and m1 < 0))
                    it.append(int(t1 in answer_ids))
                eff_rows.append({"layer": int(L), "c": float(c), "dir": name,
                                 "mean_dmargin_toward": float(np.mean(dm)),
                                 "margin_flip": float(np.mean(fl)), "intact_rate": float(np.mean(it))})
        e_u = [r for r in eff_rows if r["layer"] == L and r["dir"] == "usage"]
        e_w = [r for r in eff_rows if r["layer"] == L and r["dir"] == "w_res"]
        ratio = (np.mean([r["mean_dmargin_toward"] for r in e_w]) /
                 (np.mean([r["mean_dmargin_toward"] for r in e_u]) + 1e-12))
        logger.info("  L%d: efficiency ratio eff(w_res)/eff(usage) = %.3f", L, ratio)

    # ---------- save + verdict ----------
    import csv as _csv
    with open(out / "usage_geometry.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(geo[0].keys())); w.writeheader(); [w.writerow(r) for r in geo]
    with open(out / "head_answer_vs_axis.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer", "head", "answer_d", "axis_d"]); w.writeheader(); [w.writerow(r) for r in rows]
    with open(out / "steering_efficiency.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(eff_rows[0].keys())); w.writeheader(); [w.writerow(r) for r in eff_rows]
    json.dump({"cos_u_final_gamma_bar": cf, "spearman_answer_axis": rho, "top50_overlap": overlap},
              open(out / "usage_summary.json", "w"), indent=2)

    med_cos = float(np.median([abs(r["cos_u_wres"]) for r in geo]))
    med_auc_u = float(np.median([r["auc_along_u"] for r in geo]))
    print("\n" + "=" * 92)
    print("USAGE DIRECTION -- what the model itself reads, vs the axis we decode")
    print("=" * 92)
    print(f"sanity: cos(u_final, gamma_bar) = {cf:+.3f} (must be high; validates gradient machinery)")
    print(f"READING vs USAGE angle: median |cos(u, w_res)| over layers = {med_cos:.3f} "
          f"({'nearly orthogonal -> explicit reading!=usage geometry' if med_cos < 0.2 else 'partially aligned'})")
    print(f"DOES USAGE DECODE THE CONCEPT? median AUC along u = {med_auc_u:.3f} "
          f"({'NO -> the local readout is CONCEPT-BLIND (shortcut signature)' if med_auc_u < 0.65 else 'yes -> readout sees the concept'})")
    print(f"COMPONENT LEVEL: spearman(answer-writing, axis-writing) = {rho:.3f}, top-50 overlap = {overlap}/50 "
          f"({'disjoint -> answer-writers are not the axis-writers' if overlap <= 10 else 'substantial overlap'})")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/usage_direction")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--steer_layers", type=int, nargs="*", default=None, help="default = same as --layers")
    p.add_argument("--c_grid", type=float, nargs="*", default=[1, 4, 16])
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--max_targets", type=int, default=40)
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
