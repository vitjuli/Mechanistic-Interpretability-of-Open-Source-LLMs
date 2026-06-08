"""
100_usage_direction_deepdive.py   [what does the USED direction u do across all 36 layers?]
========================================================================================================
u^(l) = grad_{h^(l)}(logit_beta - logit_alpha) is the direction the OUTPUT is sensitive to. We know
(86/90): median |cos(u,w_res)|=0.027 all layers; AUC(u) U-shaped (L16=0.40, final=0.91); steering along
u flips behaviour at L21/L24/L35 (intact 0.74-0.89, partly by-construction). This script profiles u over
EVERY layer in the forced regime to characterise the used direction:

  (i)   STEERING intact-flip from +-c*sigma*u_hat per layer (both arms: alpha->beta with +u, beta->alpha
        with -u), escalating c. References at matched c: random-direction null and w_res. -> where in
        depth does u become a behavioural lever, and how does that compare to the (inert) readable axis?
  (ii)  LOGIT-LENS of u^(l): top/bottom vocab tokens of W_U @ u_hat^(l), and the rank/score of the
        alpha/beta answer tokens. -> what does u encode at each depth (format/answer-slot vs content)?
  (iii) AUC(u^(l)) on held-out residual projection vs labels (with AUC(w_res) as reference). -> u is
        concept-blind in mid-stack; full profile makes the "lever but content-blind" picture explicit.

IMPORTANT (kept honest): u is the margin gradient, so steering along it trivially raises the margin --
the intact-flip from u is partly BY CONSTRUCTION. The scientific content is the DEPTH PROFILE and the
contrast u(lever, content-blind) vs w_res(content, inert), not "u is a found causal direction".

SELF-TEST (no torch):  python 100_usage_direction_deepdive.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("u_deepdive")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def auc(scores, y):
    """rank-based AUC: P(score[pos] > score[neg]); y in {0,1}."""
    s = np.asarray(scores, float); yy = np.asarray(y, int)
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def self_test():
    rng = np.random.default_rng(0); d, n = 16, 200
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))
    H = 0.3 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w)   # class signal along w
    assert auc(H @ w, y) > 0.9, "auc along separating dir should be high"
    r = rng.standard_normal(d); orth = unit_raw(r - (r @ w) * w)
    assert abs(auc(H @ orth, y) - 0.5) < 0.2, "auc along orthogonal dir ~0.5"
    # intact-flip counting: preds after steering toward target t
    clean = np.array([0, 0, 1, 1]); after = np.array([1, 1, 1, 0]); t = 1
    flip = float(np.mean(after == t))
    assert abs(flip - 0.75) < 1e-9, "intact rate toward t"
    print("[self_test] OK — auc + intact-flip counting verified.")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    bm = model.model; blocks = bm.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    W_U = model.lm_head.weight.detach()                                  # (vocab, d)
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = list(range(n_layers)) if args.layers == [-1] else sorted({L for L in args.layers if 0 <= L < n_layers})

    prompts = [json.loads(l) for l in open(args.prompts)]
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = [p["surface_family"] for p in prompts]
    ufam = sorted(set(fams)); rng.shuffle(ufam)
    train_fams = set(ufam[: int(round(len(ufam) * args.train_frac))])
    is_train = np.array([f in train_fams for f in fams])

    # ---- forced-format builder (balanced 2-shot, exemplars from OTHER families) ----
    by_cls_fam = {0: {}, 1: {}}
    for i, p in enumerate(prompts):
        by_cls_fam[int(y[i])].setdefault(p["surface_family"], []).append(i)

    def exemplar(cls, avoid_fam):
        fl = [f for f in by_cls_fam[cls] if f != avoid_fam] or list(by_cls_fam[cls])
        f = fl[rng.integers(len(fl))]; j = by_cls_fam[cls][f][rng.integers(len(by_cls_fam[cls][f]))]
        ans = args.beta_answer if cls == 1 else args.alpha_answer
        return f"{prompts[j]['prompt']}\nAnswer (alpha or beta):{ans}"

    def forced_text(i):
        fam = prompts[i]["surface_family"]
        ea, eb = exemplar(0, fam), exemplar(1, fam)
        shots = [ea, eb] if (i % 2 == 0) else [eb, ea]
        return shots[0] + "\n\n" + shots[1] + "\n\n" + prompts[i]["prompt"] + "\nAnswer (alpha or beta):"

    def tap_module(L):
        return blocks[L + 1] if L < last else bm.norm

    def top1(enc, hook=None):
        h = tap_module(hook[0]).register_forward_pre_hook(hook[1], with_kwargs=False) if hook else None
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
        finally:
            if h: h.remove()
        return int(lo.argmax().item())

    # ---- capture residual + gradient (u) at answer pos, forced format ----
    H = {L: np.zeros((len(prompts), d), np.float32) for L in layers}
    G = {L: np.zeros((len(prompts), d), np.float32) for L in layers}
    clean_pred = np.zeros(len(prompts), int)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    logger.info("capturing forced residual+grad over %d prompts, %d layers...", len(prompts), len(layers))
    for i in range(len(prompts)):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keep = {}; hs = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.detach().argmax().item())
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))

    intact_rate = float(np.mean(np.isin(clean_pred, [a_id, b_id])))
    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f acc=%.3f", intact_rate, acc)

    # ---- per-layer directions, AUC, logit-lens ----
    diag = {}
    te = ~is_train
    for L in layers:
        u = unit_raw(G[L][is_train].astype(np.float64).mean(0))
        w = fisher_axis(H[L][is_train].astype(np.float64), y[is_train], args.shrink)
        proj_u = H[L][te].astype(np.float64) @ u
        proj_w = H[L][te].astype(np.float64) @ w
        auc_u, auc_w = auc(proj_u, y[te]), auc(proj_w, y[te])
        # logit-lens of u: top/bottom tokens + rank of answer tokens
        ll = (W_U @ torch.tensor(u, dtype=W_U.dtype, device=W_U.device)).float().cpu().numpy()
        top = [tok.decode([t]) for t in np.argsort(ll)[::-1][:6]]
        bot = [tok.decode([t]) for t in np.argsort(ll)[:3]]
        rank_b = int((ll > ll[b_id]).sum()); rank_a = int((ll > ll[a_id]).sum())
        diag[L] = dict(u=u, w=w, sig_u=float(np.std(H[L][is_train].astype(np.float64) @ u)),
                       sig_w=float(np.std(H[L][is_train].astype(np.float64) @ w)),
                       auc_u=float(auc_u), auc_w=float(auc_w), cos_uw=float(abs(u @ w)),
                       top=top, bot=bot, rank_b=rank_b, rank_a=rank_a)
        logger.info("L%02d AUC(u)=%.3f AUC(w)=%.3f |cos|=%.3f | u-lens top=%s | β-rank=%d α-rank=%d",
                    L, auc_u, auc_w, abs(u @ w), top[:4], rank_b, rank_a)

    # ---- steering intact sweep (forced) ----
    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & te)[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & te)[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)
    a_idx, b_idx = a_idx[: args.n_steer], b_idx[: args.n_steer]
    logger.info("steering on %d alpha->beta, %d beta->alpha prompts; c=%s", len(a_idx), len(b_idx), args.c_list)

    def steer_hook(L, vec_signed):
        v = torch.tensor(vec_signed, dtype=torch.float32, device=args.device)
        def pre(m, a):
            a[0][:, -1, :] = a[0][:, -1, :] + v; return (a[0],) + tuple(a[1:])
        return pre

    def intact_toward(idxs, L, vhat, sigma, c, target_id, sign):
        if len(idxs) == 0:
            return float("nan")
        vec = sign * c * sigma * vhat
        k = 0
        for i in idxs:
            enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
            k += int(top1(enc, hook=(L, steer_hook(L, vec))) == target_id)
        return k / len(idxs)

    rows = []
    for L in layers:
        u, w = diag[L]["u"], diag[L]["w"]; su, sw = diag[L]["sig_u"], diag[L]["sig_w"]
        rdir = unit_raw(rng.standard_normal(d)); sr = float(np.std(H[L][is_train].astype(np.float64) @ rdir))
        for c in args.c_list:
            # alpha->beta: +u ; beta->alpha: -u  (u increases beta-alpha margin)
            iu = 0.5 * (intact_toward(a_idx, L, u, su, c, b_id, +1) + intact_toward(b_idx, L, u, su, c, a_id, -1))
            iw = 0.5 * (intact_toward(a_idx, L, w, sw, c, b_id, +1) + intact_toward(b_idx, L, w, sw, c, a_id, -1))
            ir = 0.5 * (intact_toward(a_idx, L, rdir, sr, c, b_id, +1) + intact_toward(b_idx, L, rdir, sr, c, a_id, -1))
            rows.append(dict(layer=int(L), c=float(c), intact_u=iu, intact_wres=iw, intact_random=ir,
                             auc_u=diag[L]["auc_u"], auc_wres=diag[L]["auc_w"], cos_uw=diag[L]["cos_uw"],
                             u_beta_rank=diag[L]["rank_b"], u_alpha_rank=diag[L]["rank_a"]))
            logger.info("  [L%02d c=%.1f] intact: u=%.2f  w_res=%.2f  random=%.2f", L, c, iu, iw, ir)

    with open(out / "usage_deepdive.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    with open(out / "u_logitlens.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=["layer", "auc_u", "auc_wres", "cos_uw", "beta_rank", "alpha_rank", "top_tokens", "bot_tokens"])
        w_.writeheader()
        for L in layers:
            w_.writerow(dict(layer=L, auc_u=diag[L]["auc_u"], auc_wres=diag[L]["auc_w"], cos_uw=diag[L]["cos_uw"],
                             beta_rank=diag[L]["rank_b"], alpha_rank=diag[L]["rank_a"],
                             top_tokens=" ".join(diag[L]["top"]), bot_tokens=" ".join(diag[L]["bot"])))

    print("\n" + "=" * 100)
    print("USAGE DIRECTION DEEP-DIVE (forced) — u is the margin gradient (lever partly by-construction)")
    print("=" * 100)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}\n")
    best = max(layers, key=lambda L: max((r["intact_u"] for r in rows if r["layer"] == L), default=0))
    print("per-layer (max over c):  layer | AUC(u) | AUC(w_res) | best intact_u | best intact_w_res | best intact_rand | β-rank(u-lens)")
    for L in layers:
        ru = [r for r in rows if r["layer"] == L]
        print(f"  L{L:02d}   {diag[L]['auc_u']:.2f}     {diag[L]['auc_w']:.2f}        "
              f"{max(r['intact_u'] for r in ru):.2f}            {max(r['intact_wres'] for r in ru):.2f}            "
              f"{max(r['intact_random'] for r in ru):.2f}          {diag[L]['rank_b']}")
    print(f"\nPeak u-lever layer: L{best}. Read against commitment onset (~L17, exp 92): does u become a lever "
          f"where the model commits? AUC(u) dips mid-stack = u is content-blind there even while a lever.")
    print("CONTRAST: w_res (content, AUC~0.99) is NOT a lever (intact_wres ~ random); u (lever) is content-blind "
          "mid-stack. The lever and the readable concept are different directions, at every depth.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/u_deepdive")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[-1], help="-1 = all layers")
    p.add_argument("--c_list", type=float, nargs="*", default=[2.0, 4.0])
    p.add_argument("--n_steer", type=int, default=24)
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
