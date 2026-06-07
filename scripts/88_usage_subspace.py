"""
88_usage_subspace.py   [the usage SUBSPACE: spectrum, angles to the concept map, dual erasure]
==============================================================================================
Exp 86 built the mean usage direction u (gradient of the answer margin). This script upgrades
u to a full SUBSPACE and runs the mirror image of exp 87:

(A) USAGE-GRAM. Per layer, M = E[g_i g_i^T] over per-prompt margin gradients. Its spectrum
    gives the participation ratio (how many directions the model's readout really spans) and
    its top-k eigenvectors U_k = the usage subspace.

(B) PRINCIPAL-ANGLE SPECTRUM between U_k and the readable concept map W_k (INLP, as in 87),
    per layer, vs a random-frame null. Not one cosine -- the whole spectrum theta_1..theta_k.
    This is the quantitative law behind the bypass: behavioural inertness <-> vanishing
    alignment of the causally-validated usage subspace with the readable concept subspace.

(C) DUAL ERASURE -- the headline. Exp 87 erased what WE can read (task survived everywhere).
    Here we erase what the MODEL uses: recentred removal of U_k at all positions, same
    machinery, same metric (held-out margin accuracy), nulls = random matched-rank frames +
    a CONSTRUCTION control (usage subspace of an unrelated contrast: logit(" the")-logit(" of"),
    same pipeline, no task). Expected outcome: collapse at small k -> the causal dimension
    k*(l) of the decision channel, disjoint from the 13-D readable map = double dissociation
    with dimensions on both sides. Built-in sanity: at the final tap usage-erasure MUST
    collapse accuracy (u_final ~ gamma_bar) -- if it doesn't, the machinery is broken.
    CUMULATIVE mode guards the known risk that a single-layer erasure is routed around via
    depth redundancy: erase U_k at whole bands of layers simultaneously (--cum_windows).

(D) USAGE SALIENCY. The same backward pass gives gradients at EVERY position: per layer, the
    share of usage mass on the final token vs earlier tokens, and which tokens carry the
    peak -- a map of WHERE the decision is read from (anatomy of the shortcut).

(E) CARRIER-CAPTURE FOR u. Does ANY of the 163,840 transcoder features write the usage
    direction? max |cos(d_f, u^l)| (plain + causal) vs random-max null, decoders only,
    layers 10-25. (Optional: --no_carrier to skip; failures degrade gracefully.)

Verdict wording is deliberately three-tier (lesson from exp 87's verdict bug): COLLAPSE cells,
MODEST below-null cells (coarse nulls!), and the final-layer sanity -- reported separately.

SELF-TEST (no torch / no repo):  python 88_usage_subspace.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("usage_subspace")


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
    Hc = H.copy().astype(np.float64); ws = []
    for _ in range(k):
        w = fisher_axis(Hc, y, shrink)
        for prev in ws:
            w = w - (w @ prev) * prev
        nrm = np.linalg.norm(w)
        if nrm < 1e-8:
            break
        w = w / nrm; ws.append(w); Hc = Hc - np.outer(Hc @ w, w)
    Q, _ = np.linalg.qr(np.stack(ws, axis=1))
    return Q[:, : len(ws)]


def random_orthonormal(d, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q[:, :k]


def erase(H, W, mu):
    X = H - mu
    return H - (X @ W) @ W.T


def margin_accuracy(margins, y):
    return float(np.mean((np.asarray(margins) > 0).astype(int) == np.asarray(y)))


def percentile_of(value, null):
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


def participation_ratio(sing_vals):
    lam = np.asarray(sing_vals, float) ** 2
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-30))


def gram_top(G, k):
    """Top-k eigenvectors of E[g g^T] via economy SVD of the (n,d) gradient matrix."""
    _, s, Vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    return Vt[:k].T, s


def principal_cosines(A, B):
    """A (d,ka), B (d,kb) orthonormal columns -> singular values = cos of principal angles."""
    return np.linalg.svd(A.T @ B, compute_uv=False)


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 30, 400
    basis = random_orthonormal(d, 6, rng)
    S_c = basis[:, :3]                       # readable concept subspace
    a = basis[:, 3]                          # usage direction (the model's readout)
    b2 = basis[:, 4]
    y = np.array([0, 1] * (n // 2)); y2 = (y * 2 - 1.0)
    wc = unit_raw(rng.standard_normal(3))
    H = 0.5 * rng.standard_normal((n, d)) + np.outer(y2, 3.0 * (S_c @ wc)) + np.outer(y2, 0.8 * a)
    margins = H @ a
    assert margin_accuracy(margins, y) > 0.9, "toy model must solve the task via a"

    # Gram recovers the planted usage subspace span(a, b2)
    G = a[None, :] + 0.5 * np.outer(rng.standard_normal(n), b2)
    U2, s = gram_top(G, 2)
    assert principal_cosines(U2, np.stack([a, b2], 1)).min() > 0.9, "Gram top-2 must recover usage span"
    assert principal_cosines(U2, S_c).max() < 0.3, "usage span must be disjoint from concept map"
    assert 1.0 <= participation_ratio(s) <= 3.0

    mu = H.mean(0)
    acc_base = margin_accuracy(margins, y)
    acc_u = margin_accuracy(erase(H, U2[:, :1], mu) @ a, y)          # erase what is USED -> collapse
    acc_c = margin_accuracy(erase(H, S_c, mu) @ a, y)                 # erase the disjoint readable map -> intact
    assert acc_u < 0.62, f"usage erasure must collapse the toy decision, got {acc_u}"
    assert acc_c > acc_base - 0.03 and (acc_c - acc_u) > 0.2, f"erasing a disjoint subspace must leave it intact, got c={acc_c} u={acc_u}"
    W3 = inlp_subspace(H, y, 3)
    assert W3.shape[1] == 3 and np.allclose(W3.T @ W3, np.eye(3), atol=1e-8)
    # note: in a LINEAR toy any class signal reaching the margin is itself linearly readable,
    # so INLP-erasure also collapses it -- the real-model dissociation requires a nonlinear
    # usage channel; the self-test therefore validates the machinery on planted subspaces.

    I3 = random_orthonormal(d, 3, rng)
    assert np.allclose(principal_cosines(I3, I3), 1.0, atol=1e-8)
    assert percentile_of(0.9, np.array([0.1, 0.2])) == 100.0
    print("[self_test] OK — Gram recovery, PR, principal angles, dual dissociation (use-erase collapses, read-erase doesn't) pass.")


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
    d = model.config.hidden_size; last = n_layers - 1
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    ctlA = tok.encode(args.control_a, add_special_tokens=False)[0]
    ctlB = tok.encode(args.control_b, add_special_tokens=False)[0]
    layers = args.layers if args.layers else list(range(n_layers))
    layers = sorted({L for L in layers if 0 <= L < n_layers})
    sal_layers = [L for L in args.saliency_layers if L in layers]
    logger.info("model: %d layers; usage-subspace over %d taps; k grid %s", n_layers, len(layers), args.k_grid)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    # ---------- capture: residuals + TWO gradients (margin, control) + saliency norms ----------
    nP = len(prompts)
    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    grad = {L: np.zeros((nP, d), np.float32) for L in layers}
    gctl = {L: np.zeros((nP, d), np.float32) for L in layers}
    sal_share = {L: np.zeros(nP, np.float32) for L in sal_layers}
    sal_tok = {L: [] for L in sal_layers}
    y = np.zeros(nP, int); trm = np.zeros(nP, bool); clean_margin = np.zeros(nP)
    logger.info("capturing residuals + margin/control gradients for %d prompts...", nP)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        keep = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            row = model(**inp, use_cache=False).logits[0, -1, :]
            (row[beta_id] - row[alpha_id]).backward(retain_graph=True)
            for L in layers:
                t = keep[L]
                res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                gfull = t.grad[0] if t.grad is not None else None
                grad[L][i] = gfull[-1, :].float().cpu().numpy() if gfull is not None else 0.0
                if L in sal_layers and gfull is not None:
                    nrm = gfull.norm(dim=1).float().cpu().numpy()
                    tot = float(nrm.sum()) + 1e-12
                    sal_share[L][i] = float(nrm[-1]) / tot
                    sal_tok[L].append(tok.convert_ids_to_tokens(int(inp.input_ids[0, int(np.argmax(nrm))])))
                if t.grad is not None:
                    t.grad = None
            (row[ctlA] - row[ctlB]).backward()
            for L in layers:
                t = keep[L]
                gctl[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
            lp = torch.log_softmax(row.detach().float(), 0)
            clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        model.zero_grad(set_to_none=True)
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        trm[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, nP)

    held = [i for i in range(nP) if not trm[i]]
    ha = [i for i in held if y[i] == 0][: args.max_targets]
    hb = [i for i in held if y[i] == 1][: args.max_targets]
    targets = ha + hb; y_t = np.array([y[i] for i in targets])
    acc_clean = margin_accuracy(np.array([clean_margin[i] for i in targets]), y_t)
    logger.info("clean held-out margin accuracy on %d targets: %.3f", len(targets), acc_clean)

    # ---------- (A)+(B): Gram spectrum, usage subspaces, principal angles ----------
    kmax = max(args.k_grid)
    Wc = {}; Uu = {}; Uc = {}; mu_tr = {}
    null_maxcos = [principal_cosines(random_orthonormal(d, kmax, rng), random_orthonormal(d, kmax, rng)).max()
                   for _ in range(12)]
    geo = []
    for L in layers:
        H = res[L].astype(np.float64)
        mu_tr[L] = H[trm].mean(0)
        Wc[L] = inlp_subspace(H[trm], y[trm], kmax, args.shrink)
        U, s = gram_top(grad[L], kmax); Uu[L] = U
        Ucl, _ = gram_top(gctl[L], kmax); Uc[L] = Ucl
        pc = principal_cosines(Wc[L], U)
        rec = {"layer": int(L), "pr_usage": participation_ratio(s),
               "angle_maxcos": float(pc.max()), "angle_meancos": float(pc.mean()),
               "angle_top3": ";".join(f"{v:.3f}" for v in np.sort(pc)[::-1][:3]),
               "null_maxcos_p95": float(np.percentile(null_maxcos, 95)),
               "auc_wres": auc_scalar(H[~trm] @ fisher_axis(H[trm], y[trm], args.shrink), y[~trm])}
        geo.append(rec)
        if (L % 4 == 0) or (L == layers[-1]):
            logger.info("  L%d: PR(usage)=%.1f | principal cos(concept,usage): max=%.3f mean=%.3f (rand-frame p95=%.3f)",
                        L, rec["pr_usage"], rec["angle_maxcos"], rec["angle_meancos"], rec["null_maxcos_p95"])

    # ---------- (C) dual erasure: per layer ----------
    def run_erased(ptext, hooks_spec):
        """hooks_spec: list of (layer, W_np, mu_np) applied simultaneously, all positions."""
        import torch as _t
        inp = tok([ptext], return_tensors="pt").to(args.device)
        handles = []
        for (L, W_np, mu_np) in hooks_spec:
            Wt = _t.tensor(W_np, dtype=_t.float32, device=args.device)
            mut = _t.tensor(mu_np, dtype=_t.float32, device=args.device)
            def pre(m, a, Wt=Wt, mut=mut):
                hs = a[0].clone(); X = hs[0] - mut
                hs[0] = hs[0] - (X @ Wt) @ Wt.T
                return (hs,)
            handles.append(tap(L).register_forward_pre_hook(pre, with_kwargs=False))
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(row, 0)
            return float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()

    def eval_spec(spec):
        ms = [run_erased(prompts[i]["prompt"], spec) for i in targets]
        return margin_accuracy(np.array(ms), y_t)

    rows = []
    logger.info("(C) dual erasure per layer, k grid %s ...", args.k_grid)
    for L in layers:
        for k in args.k_grid:
            U = Uu[L][:, :k]
            acc_e = eval_spec([(L, U, mu_tr[L])])
            acc_rand = [eval_spec([(L, random_orthonormal(d, k, rng), mu_tr[L])]) for _ in range(args.n_random)]
            acc_ctl = eval_spec([(L, Uc[L][:, :k], mu_tr[L])])
            rec = {"layer": int(L), "k": int(k), "acc_clean": acc_clean, "acc_erased": acc_e,
                   "acc_rand_mean": float(np.mean(acc_rand)), "acc_rand_min": float(np.min(acc_rand)),
                   "acc_control_contrast": acc_ctl}
            rows.append(rec)
            logger.info("  L%d k=%d: USE-erased=%.3f | rand(mean %.3f, min %.3f) control-contrast=%.3f (clean %.3f)",
                        L, k, acc_e, rec["acc_rand_mean"], rec["acc_rand_min"], acc_ctl, acc_clean)

    # ---------- (C2) cumulative windows ----------
    cum_rows = []
    for w in args.cum_windows:
        for start in range(0, n_layers, w):
            Ls = [L for L in range(start, min(start + w, n_layers)) if L in Uu]
            if not Ls:
                continue
            spec = [(L, Uu[L][:, : args.k_cum], mu_tr[L]) for L in Ls]
            acc_e = eval_spec(spec)
            accs_r = []
            for _ in range(args.n_random):
                accs_r.append(eval_spec([(L, random_orthonormal(d, args.k_cum, rng), mu_tr[L]) for L in Ls]))
            cum_rows.append({"window": f"L{Ls[0]}-L{Ls[-1]}", "k": args.k_cum, "acc_erased": acc_e,
                             "acc_rand_mean": float(np.mean(accs_r)), "acc_rand_min": float(np.min(accs_r)),
                             "acc_clean": acc_clean})
            logger.info("  cum L%d-L%d (k=%d): USE-erased=%.3f | rand(mean %.3f, min %.3f)",
                        Ls[0], Ls[-1], args.k_cum, acc_e, np.mean(accs_r), np.min(accs_r))

    # ---------- (D) saliency ----------
    sal_rows = []
    for L in sal_layers:
        cnt = Counter(sal_tok[L]).most_common(8)
        sal_rows.append({"layer": int(L), "mean_last_pos_share": float(sal_share[L].mean()),
                         "top_argmax_tokens": " | ".join(f"{t}:{c}" for t, c in cnt)})
        logger.info("(D) L%d: last-pos usage share=%.2f | peak tokens: %s", L,
                    sal_rows[-1]["mean_last_pos_share"], sal_rows[-1]["top_argmax_tokens"])

    # ---------- (E) carrier capture for u ----------
    car_rows = []
    if args.carrier:
        try:
            from transcoder_loader import load_transcoder_set
            ts = load_transcoder_set("4b", device=torch.device(args.device), dtype=torch.bfloat16, lazy_load=True)
            for L in [L for L in args.carrier_layers if L in layers]:
                u = unit_raw(grad[L].astype(np.float64).mean(0))
                Sig = np.cov(res[L].astype(np.float64).T)
                vals, vecs = np.linalg.eigh(0.5 * (Sig + Sig.T))
                vals = np.clip(vals, 1e-3 * vals.max(), None)
                Wh = (vecs * (vals ** -0.5)) @ vecs.T
                Wd = ts[L].W_dec
                Wd = (Wd.detach() if hasattr(Wd, "detach") else Wd).to(args.device, torch.float32)
                ut = torch.tensor(u, dtype=torch.float32, device=args.device)
                cos_plain = (Wd @ ut) / (Wd.norm(dim=1) + 1e-12)
                wh_t = torch.tensor(Wh, dtype=torch.float32, device=args.device)
                WW = Wd @ wh_t
                whu = wh_t @ ut
                cos_c = (WW @ whu) / ((WW.norm(dim=1) * whu.norm()) + 1e-12)
                F = Wd.shape[0]
                rmax = []
                for _ in range(3):
                    Rg = torch.randn(20000, d, device=args.device)
                    rmax.append(float(((Rg @ ut) / Rg.norm(dim=1)).abs().max()))
                rand_max = float(np.sqrt(2 * np.log(F) / d))
                car_rows.append({"layer": int(L),
                                 "max_abs_cos_plain": float(cos_plain.abs().max()),
                                 "max_abs_cos_causal": float(cos_c.abs().max()),
                                 "rand_max_analytic": rand_max,
                                 "rand_max_sampled20k": float(np.max(rmax))})
                logger.info("(E) L%d: max|cos(d_f,u)| plain=%.3f causal=%.3f (rand-max ~%.3f)",
                            L, car_rows[-1]["max_abs_cos_plain"], car_rows[-1]["max_abs_cos_causal"], rand_max)
                del Wd, WW
                torch.cuda.empty_cache()
                if hasattr(ts, "unload_layer"):
                    ts.unload_layer(L)
        except Exception as e:
            logger.warning("(E) carrier capture skipped: %s", e)

    # ---------- save + verdict ----------
    import csv as _csv
    def wcsv(name, rws):
        if not rws:
            return
        with open(out / name, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rws[0].keys())); w.writeheader(); [w.writerow(r) for r in rws]
    wcsv("usage_angles.csv", geo); wcsv("dual_erasure.csv", rows)
    wcsv("cumulative_erasure.csv", cum_rows); wcsv("usage_saliency.csv", sal_rows); wcsv("carrier_u.csv", car_rows)

    med_max = float(np.median([r["angle_maxcos"] for r in geo]))
    nonfinal = [r for r in rows if r["layer"] != last]
    final_rows = [r for r in rows if r["layer"] == last]
    collapse = [r for r in nonfinal if r["acc_erased"] <= 0.55]
    strong = [r for r in nonfinal if r["acc_erased"] < acc_clean - 0.15 and r["acc_erased"] < r["acc_rand_min"] - 0.05]
    modest = [r for r in nonfinal if r["acc_erased"] < r["acc_rand_min"] and r not in strong and r not in collapse]
    print("\n" + "=" * 96)
    print("USAGE SUBSPACE -- spectrum, angles to the readable map, and DUAL erasure")
    print("=" * 96)
    print(f"ANGLES: median max principal cos(concept_map, usage_subspace) over layers = {med_max:.3f} "
          f"(random-frame p95 ~ {float(np.percentile(null_maxcos,95)):.3f})")
    if final_rows:
        fr = min(final_rows, key=lambda r: r["k"])
        ok = fr["acc_erased"] <= 0.6
        print(f"SANITY (final tap, k={fr['k']}): usage-erase acc={fr['acc_erased']:.2f} -> "
              f"{'collapses as required (machinery valid)' if ok else 'DID NOT collapse -> CHECK MACHINERY'}")
    if collapse or strong:
        cells = sorted(set((r["layer"], r["k"]) for r in collapse + strong))
        print(f"COLLAPSE/STRONG cells (usage causally necessary): {cells}")
        ks = [k for (_, k) in cells]
        print(f"-> causal channel dimension candidates k* as low as {min(ks)}; readable-map erasure (exp 87) left the task intact"
              f" -> DOUBLE DISSOCIATION: the decision flows through a usage channel disjoint from the readable concept map.")
    else:
        print("No single-layer collapse: the decision channel is depth-redundant at the per-layer level.")
    if cum_rows:
        worst = min(cum_rows, key=lambda r: r["acc_erased"])
        print(f"CUMULATIVE: worst window {worst['window']} (k={worst['k']}): acc={worst['acc_erased']:.2f} "
              f"(rand mean {worst['acc_rand_mean']:.2f}, clean {acc_clean:.2f})")
    if modest:
        print(f"MODEST below-null cells (coarse nulls, n_random={args.n_random} -> treat as candidates only): "
              f"{sorted(set((r['layer'], r['k'])) for r in [modest][0]) if False else sorted(set((r['layer'], r['k']) for r in modest))}")
    print("=" * 96 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/usage_subspace")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--control_a", default=" the")
    p.add_argument("--control_b", default=" of")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--k_grid", type=int, nargs="*", default=[1, 5, 13])
    p.add_argument("--k_cum", type=int, default=5)
    p.add_argument("--cum_windows", type=int, nargs="*", default=[4, 8], help="window widths; empty disables")
    p.add_argument("--n_random", type=int, default=4)
    p.add_argument("--max_targets", type=int, default=80)
    p.add_argument("--saliency_layers", type=int, nargs="*", default=[8, 16, 21, 24, 30, 35])
    p.add_argument("--carrier", dest="carrier", action="store_true", default=True)
    p.add_argument("--no_carrier", dest="carrier", action="store_false")
    p.add_argument("--carrier_layers", type=int, nargs="*", default=list(range(10, 26)))
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
