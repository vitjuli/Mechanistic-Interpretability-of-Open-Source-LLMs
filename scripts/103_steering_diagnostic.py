"""
103_steering_diagnostic.py   [is w_res-steering a SCALE BUG or a genuine null?]
========================================================================================================
Claim under scrutiny: h^(l) <- h^(l) + c*sigma*w_res_hat does NOT change the model's answer. Two very
different explanations:
  (A) BUG / tiny addition: c*sigma is negligible vs ||h||, so h barely moves -> nothing happens.
  (B) GENUINE null: the addition is sizeable along the axis, but the OUTPUT is not sensitive to that
      axis (g = grad(margin) is ~orthogonal to w_res), so the margin does not move.
This script measures, on ONE prompt set, for layers {17,21,24,30} and directions {w_res, u, random},
everything needed to distinguish them:

  ||w_hat||_2            must be exactly 1.0           (rules out a normalisation mismatch, cf exp 61b)
  sigma                  std of <h, w_hat> on train    (the steering scale)
  c*sigma  vs  ||h||     ratio = c*sigma / ||h||        (<-- the 'tiny addition' hypothesis (A))
  proj shift (sigma)     <h+vec, w_hat> - <h, w_hat>, in sigma units; MUST equal c  (hook applied + arithmetic)
  cos(g, dir)            ~0.03 for w_res, ~1 for u       (<-- why margin moves or not, hypothesis (B))
  dmargin_actual         margin_steered - margin_clean   (the real effect)
  dmargin_pred           <g, c*sigma*w_hat>  (exp-89)    (should match actual at small c)
  intact-flip            top-1 becomes the target answer

DECISIVE sanity:
  * SAME code along u must give large dmargin + intact-flip -> proves the hook + arithmetic are correct
    and isolates DIRECTION (not code) as the cause if w_res stays flat at the same c*sigma.
  * ZERO-ablation h<-0 at the layer must break the answer (intact-rate drops) -> proves the hook
    physically modifies the forward pass.

SELF-TEST (no torch):  python 103_steering_diagnostic.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("steer_diag")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def self_test():
    rng = np.random.default_rng(0); d, n = 2560, 200
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))                       # readable/concept axis
    ru = rng.standard_normal(d); u = unit_raw(ru - (ru @ w) * w)             # used axis ~|_ w
    H = 30.0 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w) * 5.0     # ||h|| ~ 30*sqrt(d) big
    # margin depends on u (used), NOT on w (readable): g points along u
    g = u * 4.0
    sig_w = float(np.std(H @ w)); sig_u = float(np.std(H @ u))
    c = 4.0
    # hypothesis A check: c*sigma vs ||h|| -> ratio ~ c/sqrt(d) ~ 0.08 (small in TOTAL norm, the concern)
    hn = float(np.mean(np.linalg.norm(H, axis=1)))
    ratio = c * sig_w / hn
    assert ratio < 0.1, f"in high-d the addition is small in total norm (ratio={ratio:.3f}~c/sqrt(d))"
    # but the sigma-unit shift is exactly c
    assert abs(((H[0] + c * sig_w * w) @ w - H[0] @ w) / sig_w - c) < 1e-6, "proj shift must equal c"
    # hypothesis B: dmargin_pred along w ~ 0 (g _|_ w), along u large
    dpred_w = float(np.dot(g, c * sig_w * w)); dpred_u = float(np.dot(g, c * sig_u * u))
    assert abs(dpred_w) < 1e-6 and dpred_u > 1.0, "margin moves along u (used), not along w (readable)"
    print("[self_test] OK — toy reproduces: small-norm addition, exact c-sigma shift, dmargin~0 along w, large along u.")


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
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})

    prompts = [json.loads(l) for l in open(args.prompts)]
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = [p["surface_family"] for p in prompts]
    ufam = sorted(set(fams)); rng.shuffle(ufam)
    train_fams = set(ufam[: int(round(len(ufam) * args.train_frac))])
    is_train = np.array([f in train_fams for f in fams])

    by_cls_fam = {0: {}, 1: {}}
    for i, p in enumerate(prompts):
        by_cls_fam[int(y[i])].setdefault(p["surface_family"], []).append(i)

    def exemplar(cls, avoid):
        fl = [f for f in by_cls_fam[cls] if f != avoid] or list(by_cls_fam[cls])
        f = fl[rng.integers(len(fl))]; j = by_cls_fam[cls][f][rng.integers(len(by_cls_fam[cls][f]))]
        ans = args.beta_answer if cls == 1 else args.alpha_answer
        return f"{prompts[j]['prompt']}\nAnswer (alpha or beta):{ans}"

    def forced_text(i):
        fam = prompts[i]["surface_family"]; ea, eb = exemplar(0, fam), exemplar(1, fam)
        shots = [ea, eb] if (i % 2 == 0) else [eb, ea]
        return shots[0] + "\n\n" + shots[1] + "\n\n" + prompts[i]["prompt"] + "\nAnswer (alpha or beta):"

    def tap_module(L):
        return blocks[L + 1] if L < last else bm.norm

    # ---- capture residual + grad + clean margin/pred (forced) over all prompts ----
    H = {L: np.zeros((len(prompts), d), np.float32) for L in layers}
    G = {L: np.zeros((len(prompts), d), np.float32) for L in layers}
    margin_clean = np.zeros(len(prompts)); clean_pred = np.zeros(len(prompts), int)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    logger.info("capturing forced residual+grad over %d prompts, layers %s...", len(prompts), layers)
    for i in range(len(prompts)):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keep, hs = {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.detach().argmax().item())
            margin_clean[i] = float(lo[b_id].detach() - lo[a_id].detach())
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

    tr = is_train
    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)
    a_idx, b_idx = a_idx[: args.n], b_idx[: args.n]
    logger.info("diagnostic on %d alpha + %d beta prompts", len(a_idx), len(b_idx))

    def add_hook(vec, verify=None):
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        vt = torch.tensor(verify[0], dtype=torch.float32, device=args.device) if verify else None
        box = {}
        def pre(m, a):
            if vt is not None:
                box["before"] = float((a[0][0, -1, :] @ vt).item())
            a[0][:, -1, :] = a[0][:, -1, :] + v
            if vt is not None:
                box["after"] = float((a[0][0, -1, :] @ vt).item())
            return (a[0],) + tuple(a[1:])
        return pre, box

    def zero_hook():
        def pre(m, a):
            a[0][:, -1, :] = 0.0; return (a[0],) + tuple(a[1:])
        return pre

    def fwd_margin_pred(i, L, hook):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        h = tap_module(L).register_forward_pre_hook(hook, with_kwargs=False)
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
        finally:
            h.remove()
        return int(lo.argmax().item()), float(lo[b_id] - lo[a_id])

    # ---- per-layer clean stats + zero-ablation sanity ----
    print("\n" + "=" * 110)
    print("STEERING DIAGNOSTIC — scale bug (A) vs genuine null (B)?")
    print("=" * 110)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}\n")
    for L in layers:
        hn = float(np.mean(np.linalg.norm(H[L][np.r_[a_idx, b_idx]].astype(np.float64), axis=1)))
        # zero-ablation: how many of a_idx+b_idx still have top-1 in {alpha,beta} after h<-0
        zk = 0
        for i in np.r_[a_idx, b_idx]:
            pr, _ = fwd_margin_pred(i, L, zero_hook()); zk += int(pr in (a_id, b_id))
        logger.info("L%02d: mean||h||=%.1f (rms=%.1f) | ZERO-ablation intact-rate=%.2f (clean 1.00 -> hook works if <1)",
                    L, hn, hn / np.sqrt(d), zk / len(np.r_[a_idx, b_idx]))

    # ---- main: per (layer, direction, c) ----
    dirs = {}
    rows = []
    for L in layers:
        w_res = fisher_axis(H[L][tr].astype(np.float64), y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        rdir = unit_raw(rng.standard_normal(d))
        dirs[L] = {"w_res": w_res, "u": u, "random": rdir}
        hn = float(np.mean(np.linalg.norm(H[L][a_idx].astype(np.float64), axis=1)))
        for name, vhat in dirs[L].items():
            sigma = float(np.std(H[L][tr].astype(np.float64) @ vhat))
            wnorm = float(np.linalg.norm(vhat))
            # cos(g, dir) on alpha arm
            cos_gv = float(np.mean([abs(unit_raw(G[L][i].astype(np.float64)) @ vhat) for i in a_idx]))
            for c in args.c_list:
                vec = (c * sigma) * vhat                      # alpha->beta arm uses +vec
                # one-off proj-shift verification on first alpha prompt
                hook, box = add_hook(vec, verify=(vhat,))
                _ = fwd_margin_pred(int(a_idx[0]), L, hook)
                shift_sigma = (box["after"] - box["before"]) / sigma if sigma > 0 else float("nan")
                # actual dmargin + intact on alpha arm (+vec) and beta arm (-vec)
                dms, fa = [], 0
                for i in a_idx:
                    hk, _b = add_hook(vec); pr, m = fwd_margin_pred(i, L, hk)
                    dms.append(m - margin_clean[i]); fa += int(pr == b_id)
                fb = 0
                for i in b_idx:
                    hk, _b = add_hook(-vec); pr, m = fwd_margin_pred(i, L, hk)
                    fb += int(pr == a_id)
                dmargin_actual = float(np.mean(dms))
                dmargin_pred = float(np.mean([np.dot(G[L][i].astype(np.float64), vec) for i in a_idx]))
                intact = 0.5 * (fa / len(a_idx) + fb / len(b_idx))
                rows.append(dict(layer=int(L), direction=name, c=float(c),
                                 w_norm=wnorm, sigma=sigma, c_sigma=c * sigma, h_norm=hn,
                                 ratio_csigma_hnorm=c * sigma / hn, proj_shift_in_sigma=shift_sigma,
                                 cos_g_dir=cos_gv, dmargin_actual=dmargin_actual,
                                 dmargin_pred=dmargin_pred, intact_flip=intact))
                logger.info("  L%02d %-7s c=%.0f | ||w||=%.3f sigma=%.2f c*sigma=%.1f ||h||=%.0f ratio=%.4f "
                            "| proj_shift=%.2fσ cos(g,dir)=%.3f | Δm act=%.2f pred=%.2f | intact=%.2f",
                            L, name, c, wnorm, sigma, c * sigma, hn, c * sigma / hn,
                            shift_sigma, cos_gv, dmargin_actual, dmargin_pred, intact)

    with open(out / "steering_diagnostic.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "-" * 110)
    print("VERDICT GUIDE (read the rows above):")
    print("  ||w||=1.000 everywhere                          -> no normalisation mismatch (rules out the 61b-class bug)")
    print("  ratio c*sigma/||h|| small (~1e-2..1e-3)          -> the addition IS small in TOTAL norm (your concern, hypothesis A)")
    print("  BUT proj_shift ~= c (e.g. 4.00 sigma)            -> along the axis the shift is c std-devs (large in axis units)")
    print("  cos(g,w_res) ~0.03  while cos(g,u) ~1.0          -> the OUTPUT is ~blind to w_res, fully sensitive to u")
    print("  Δm_actual ~ Δm_pred, and ~0 for w_res, large for u -> margin moves along u, not w_res: a GENUINE null, not a bug")
    print("  intact: u flips (->~1), w_res ~0 at the SAME c*sigma -> SAME code, only direction differs => not a code error")
    print("  ZERO-ablation intact-rate < 1.0                  -> the hook physically changes the forward (mechanism works)")
    print("Conclusion: if ||w||=1, proj_shift=c, zero-ablation breaks, and u flips while w_res does not at equal c*sigma,")
    print("then w_res-steering doing nothing is HYPOTHESIS B (output not sensitive to the readable axis), NOT a scale bug.")
    print("=" * 110 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/steer_diag")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[17, 21, 24, 30])
    p.add_argument("--c_list", type=float, nargs="*", default=[2.0, 4.0, 8.0])
    p.add_argument("--n", type=int, default=16)
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
