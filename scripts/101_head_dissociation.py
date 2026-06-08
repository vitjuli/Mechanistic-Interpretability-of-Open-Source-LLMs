"""
101_head_dissociation.py   [are ANSWER-writer heads the lever, while AXIS-writer heads are not?]
========================================================================================================
Exp 80 ablated the 10 heads ALIGNED WITH w_res (axis-writers) -> intact-flip 0 (not necessary). Exp 86
found axis-writers and answer-writers are nearly disjoint head sets (Spearman 0.144, top-50 overlap 11).
The missing test: ablate the ANSWER-writers (heads that write toward the beta-alpha answer contrast) and
see if THEY are the behavioural lever. If answer-writers break the answer (intact>0) while axis-writers
do not (exp 80, =0), that is the clean head-level form of decoded != used.

Two rankings of every head (L,h), by Cohen's |d| of the head's residual contribution at the answer pos:
  answer-writer score : projection onto gamma_bar = unit(W_U[beta] - W_U[alpha])    (DLA, answer slot)
  axis-writer  score  : projection onto w_res^(L)                                    (= exp 80)
Head contribution = W_O^(L)[:, h-slice] @ (o_proj input)_h  (o_proj is linear -> exact decomposition).

Interventions (forced regime, intact-flip both arms, vs random-head null of matched size K):
  ABLATION (zero head slice at answer pos) -- necessity
  NEGATION (x -1 at answer pos)            -- active sufficiency
Selection scopes: (a) global top-K over all layers; (b) MID-STACK only (layers [lo,hi]) -- the
non-trivial scope, since late answer-writers ~ the readout (u ~ gamma_bar near output) and are trivially
important. The interesting question is the middle.

SELF-TEST (no torch):  python 101_head_dissociation.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("head_dissoc")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def cohend(x, y):
    x = np.asarray(x, float)
    m1, m0 = x[y == 1].mean(), x[y == 0].mean()
    s = np.sqrt(0.5 * (x[y == 1].var(ddof=1) + x[y == 0].var(ddof=1))) + 1e-12
    return (m1 - m0) / s


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def self_test():
    rng = np.random.default_rng(0)
    nH, hd, P, d = 8, 4, 120, 12
    y = (rng.random(P) > 0.5).astype(int)
    X = 0.2 * rng.standard_normal((P, nH, hd))            # o_proj inputs per head
    WO = rng.standard_normal((d, nH, hd))                 # W_O reshaped
    gamma = unit_raw(rng.standard_normal(d))
    # make head 3 an answer-writer: its contribution along gamma tracks the label
    X[:, 3, :] += np.outer(2 * y - 1, np.ones(hd)) * 0.8
    # head contribution projected on gamma: c_h(p)
    tmp = np.einsum("d,dhk->hk", gamma, WO)               # (nH,hd)
    c = np.einsum("hk,phk->ph", tmp, X)                   # (P,nH)
    scores = np.array([abs(cohend(c[:, h], y)) for h in range(nH)])
    assert scores.argmax() == 3, f"head 3 should rank top as answer-writer, got {scores.argmax()}"
    print("[self_test] OK — per-head DLA projection + Cohen's d ranking recovers the planted head.")


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
    n_heads = model.config.num_attention_heads
    W_U = model.lm_head.weight.detach()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    gamma = unit_raw((W_U[b_id] - W_U[a_id]).float().cpu().numpy().astype(np.float64))
    layers = list(range(n_layers))

    def oproj(L):
        return blocks[L].self_attn.o_proj
    in_feat = oproj(0).in_features
    head_dim = in_feat // n_heads
    logger.info("n_heads=%d head_dim=%d in_feat=%d d=%d", n_heads, head_dim, in_feat, d)

    prompts = [json.loads(l) for l in open(args.prompts)]
    P = len(prompts)
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

    # ---- one capture pass: residual (for w_res) + o_proj inputs (for head DLA), answer pos ----
    H = np.zeros((P, n_layers, d), np.float32)
    X = np.zeros((P, n_layers, in_feat), np.float32)
    clean_pred = np.zeros(P, int)
    logger.info("capturing forced residual + o_proj inputs over %d prompts...", P)
    for i in range(P):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keepH, keepX, hs = {}, {}, []
        for L in layers:
            def mkH(L=L):
                def pre(m, a): keepH[L] = a[0].detach()[0, -1, :]; return None
                return pre
            def mkX(L=L):
                def pre(m, a): keepX[L] = a[0].detach()[0, -1, :]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mkH(), with_kwargs=False))
            hs.append(oproj(L).register_forward_pre_hook(mkX(), with_kwargs=False))
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.argmax().item())
            for L in layers:
                H[i, L] = keepH[L].float().cpu().numpy(); X[i, L] = keepX[L].float().cpu().numpy()
        finally:
            for h in hs: h.remove()
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, P)

    intact_rate = float(np.mean(np.isin(clean_pred, [a_id, b_id])))
    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f acc=%.3f", intact_rate, acc)

    # ---- per-head scores ----
    tr = is_train
    WO = oproj  # closure
    ans_score = np.zeros((n_layers, n_heads)); axis_score = np.zeros((n_layers, n_heads))
    for L in layers:
        Wre = oproj(L).weight.detach().float().cpu().numpy().reshape(d, n_heads, head_dim)  # (d,nH,hd)
        w_res = fisher_axis(H[tr, L].astype(np.float64), y[tr], args.shrink)
        Xr = X[:, L, :].reshape(P, n_heads, head_dim).astype(np.float64)
        tg = np.einsum("d,dhk->hk", gamma, Wre); cg = np.einsum("hk,phk->ph", tg, Xr)        # onto gamma
        tw = np.einsum("d,dhk->hk", w_res, Wre); cw = np.einsum("hk,phk->ph", tw, Xr)        # onto w_res
        for h in range(n_heads):
            ans_score[L, h] = abs(cohend(cg[tr, h], y[tr]))
            axis_score[L, h] = abs(cohend(cw[tr, h], y[tr]))

    def topK(score, K, lo=None, hi=None):
        mask = np.ones_like(score, bool)
        if lo is not None:
            mask[:lo] = False; mask[hi + 1:] = False
        flat = [(score[L, h], L, h) for L in layers for h in range(n_heads) if mask[L, h]]
        flat.sort(reverse=True)
        return [(L, h) for _, L, h in flat[:K]]

    # report overlap of the two rankings (Spearman-ish via top-set)
    K = args.topk
    ans_all = topK(ans_score, K); axis_all = topK(axis_score, K)
    overlap = len(set(ans_all) & set(axis_all))
    logger.info("global top-%d: answer-writers=%s", K, ans_all)
    logger.info("global top-%d: axis-writers  =%s", K, axis_all)
    logger.info("overlap(answer,axis) top-%d = %d/%d", K, overlap, K)

    # ---- intervention machinery ----
    def edit_hook(heads_here, mode):
        idx = []
        for h in heads_here:
            idx.extend(range(h * head_dim, (h + 1) * head_dim))
        idx = torch.tensor(idx, device=args.device)
        def pre(m, a):
            if mode == "ablate": a[0][:, -1, idx] = 0.0
            else: a[0][:, -1, idx] = -a[0][:, -1, idx]
            return (a[0],) + tuple(a[1:])
        return pre

    def run_with(heads, mode, idxs, target_id):
        if len(idxs) == 0: return float("nan")
        per_layer = {}
        for (L, h) in heads: per_layer.setdefault(L, []).append(h)
        k = 0
        for i in idxs:
            enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
            hs = [oproj(L).register_forward_pre_hook(edit_hook(hh, mode), with_kwargs=False) for L, hh in per_layer.items()]
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                for hk in hs: hk.remove()
            k += int(int(lo.argmax().item()) == target_id)
        return k / len(idxs)

    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)
    a_idx, b_idx = a_idx[: args.n_eval], b_idx[: args.n_eval]

    def both_arms(heads, mode):
        # ablation should not have a target direction; we just measure if answer is destroyed ->
        # report intact-flip-to-OTHER (alpha-prompts becoming beta and vice versa) as "breaks answer"
        ab = run_with(heads, mode, a_idx, b_id)   # alpha prompt -> became beta?
        ba = run_with(heads, mode, b_idx, a_id)   # beta prompt  -> became alpha?
        return 0.5 * (ab + ba), ab, ba

    def random_null(K, mode, lo=None, hi=None):
        vals = []
        for _ in range(args.n_random):
            pool = [(L, h) for L in layers for h in range(n_heads)
                    if (lo is None or (lo <= L <= hi))]
            sel = [pool[j] for j in rng.choice(len(pool), size=K, replace=False)]
            vals.append(both_arms(sel, mode)[0])
        return float(np.mean(vals)), float(np.percentile(vals, 95))

    scopes = [("global", None, None)]
    if args.mid_lo >= 0:
        scopes.append((f"mid[{args.mid_lo}-{args.mid_hi}]", args.mid_lo, args.mid_hi))

    rows = []
    for sc_name, lo, hi in scopes:
        ans_sel = topK(ans_score, K, lo, hi); axis_sel = topK(axis_score, K, lo, hi)
        for mode in ["ablate", "negate"]:
            ia, ab_a, ba_a = both_arms(ans_sel, mode)
            ix, ab_x, ba_x = both_arms(axis_sel, mode)
            rn_mean, rn_p95 = random_null(K, mode, lo, hi)
            rows.append(dict(scope=sc_name, mode=mode,
                             answer_writers=ia, axis_writers=ix, random_mean=rn_mean, random_p95=rn_p95,
                             ans_arm_ab=ab_a, ans_arm_ba=ba_a))
            logger.info("[%s %s] answer-writers=%.2f  axis-writers=%.2f  | random mean=%.2f p95=%.2f",
                        sc_name, mode, ia, ix, rn_mean, rn_p95)

    with open(out / "head_dissociation.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    # also dump the rankings
    with open(out / "head_scores.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=["layer", "head", "answer_writer_score", "axis_writer_score"]); w_.writeheader()
        for L in layers:
            for h in range(n_heads):
                w_.writerow(dict(layer=L, head=h, answer_writer_score=float(ans_score[L, h]), axis_writer_score=float(axis_score[L, h])))

    print("\n" + "=" * 100)
    print("HEAD DISSOCIATION (forced, intact-flip) — are answer-writers the lever, axis-writers not?")
    print("=" * 100)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}; overlap(answer,axis) top-{K} = {overlap}/{K}\n")
    for r in rows:
        verdict = ""
        if r["answer_writers"] > max(r["random_p95"], 0.1) and r["axis_writers"] <= max(r["random_p95"], 0.1):
            verdict = " <= DISSOCIATION (answer-writers break it, axis-writers don't)"
        elif r["answer_writers"] <= r["random_p95"] and r["axis_writers"] <= r["random_p95"]:
            verdict = " <= neither beats null"
        print(f"  [{r['scope']:>10} {r['mode']:>6}] answer-writers {r['answer_writers']:.2f}  "
              f"axis-writers {r['axis_writers']:.2f}  (random {r['random_mean']:.2f}/p95 {r['random_p95']:.2f}){verdict}")
    print("\nReading: 'breaks it' = alpha-prompt top-1 becomes beta (or vice versa) under the edit. The mid-stack scope "
          "is the non-trivial one (late answer-writers ~ readout, trivially important). axis-writers reproduce exp 80 "
          "(intact 0). If answer-writers >> null in mid-stack, the lever is the answer-writing heads, not the axis-writers.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/head_dissoc")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--mid_lo", type=int, default=12)
    p.add_argument("--mid_hi", type=int, default=28)
    p.add_argument("--n_eval", type=int, default=24)
    p.add_argument("--n_random", type=int, default=3)
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
