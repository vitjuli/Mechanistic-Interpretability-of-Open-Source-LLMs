"""
114_head_triad_bridge.py   [which channel does each attention head WRITE into: read / use / write / readout?]
========================================================================================================
Connects the attention-writing story (exp 101: answer-writers vs axis-writers) to the decoded/written/used
triad. Each head h at layer L adds a vector to the residual stream = (its slice of the o_proj input) mapped
through its slice of W_O. We reconstruct that per-head write at the answer position and decompose it into
the triad channels, reporting the FRACTION of the head's output norm along each unit direction:
    frac_wres  (read)   frac_u  (used)   frac_delta  (written)   frac_gamma  (readout contrast)
Then we classify heads -- answer-writers (high frac_gamma) vs axis-writers (high frac_wres) -- and ask:
are they DIFFERENT heads, and does each write into a different channel? E.g. do axis-writers feed the
read channel (w_res) while answer-writers feed the readout (gamma ~ u late)? This is the head-level face of
decoded != used, per layer.

W_O = o_proj.weight is (d_model, n_heads*head_dim); per-head write = oproj_in[:, h-slice] @ W_O[:, h-slice].T.
Light-ish: capture residual+grad (for w_res/u/delta) + o_proj input (for head writes), then offline algebra.
SELF-TEST (no torch):  python 114_head_triad_bridge.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("head_triad")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def frac_along(vecs, uhat):
    """mean over rows of <vec, uhat>^2 / ||vec||^2  (fraction of output energy along uhat)."""
    num = (vecs @ uhat) ** 2
    den = (vecs ** 2).sum(1) + 1e-30
    return float(np.mean(num / den)), float(np.mean(vecs @ uhat))   # frac, signed mean proj


def self_test():
    rng = np.random.default_rng(0); d, nh, hd = 256, 8, 32
    w = unit_raw(rng.standard_normal(d)); g = unit_raw(rng.standard_normal(d - 0))
    g = unit_raw(g - (g @ w) * w)                                   # gamma _|_ w for clean test
    WO = rng.standard_normal((d, nh * hd)) * 0.1
    # head 0 writes along w (axis-writer), head 1 writes along g (answer-writer)
    WO[:, 0 * hd:1 * hd] = np.outer(w, rng.standard_normal(hd))
    WO[:, 1 * hd:2 * hd] = np.outer(g, rng.standard_normal(hd))
    oin = rng.standard_normal((50, nh * hd))
    fr_w, fr_g = [], []
    for h in range(nh):
        sl = slice(h * hd, (h + 1) * hd)
        hv = oin[:, sl] @ WO[:, sl].T
        fr_w.append(frac_along(hv, w)[0]); fr_g.append(frac_along(hv, g)[0])
    assert np.argmax(fr_w) == 0 and np.argmax(fr_g) == 1, "axis-writer=head0, answer-writer=head1 recovered"
    print(f"[self_test] OK — per-head channel decomposition recovers axis-writer (h0 frac_w={fr_w[0]:.2f}) vs answer-writer (h1 frac_g={fr_g[1]:.2f}).")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True, trust_remote_code=True).to(args.device).eval()
    bm = model.model; blocks = bm.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    nh = model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", d // nh)
    W_U = model.lm_head.weight.detach()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    gamma = unit_raw((W_U[b_id] - W_U[a_id]).float().cpu().numpy().astype(np.float64))
    layers = sorted({L for L in args.layers if 0 <= L < n_layers}) if args.layers != [-1] else list(range(n_layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    Pn = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = [p["surface_family"] for p in prompts]
    ufam = sorted(set(fams)); rng.shuffle(ufam)
    tr = np.array([f in set(ufam[:int(round(len(ufam) * args.train_frac))]) for f in fams])

    by = {0: {}, 1: {}}
    for i, p in enumerate(prompts):
        by[int(y[i])].setdefault(p["surface_family"], []).append(i)
    def exemplar(cls, avoid):
        fl = [f for f in by[cls] if f != avoid] or list(by[cls]); f = fl[rng.integers(len(fl))]
        j = by[cls][f][rng.integers(len(by[cls][f]))]
        return f"{prompts[j]['prompt']}\nAnswer (alpha or beta):{args.beta_answer if cls==1 else args.alpha_answer}"
    def forced(i):
        fam = prompts[i]["surface_family"]; ea, eb = exemplar(0, fam), exemplar(1, fam)
        s = [ea, eb] if i % 2 == 0 else [eb, ea]
        return s[0] + "\n\n" + s[1] + "\n\n" + prompts[i]["prompt"] + "\nAnswer (alpha or beta):"
    def tap(L): return blocks[L + 1] if L < last else bm.norm

    H = {L: np.zeros((Pn, d), np.float32) for L in layers}
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    OIN = {L: np.zeros((Pn, nh * hd), np.float32) for L in layers}      # o_proj input at answer pos
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing residual+grad+oproj_in over %d prompts; %d layers; nh=%d hd=%d ...", Pn, len(layers), nh, hd)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, koin, hs = {}, {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            def mko(L=L):
                def pre(m, a): koin[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(mko(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                OIN[L][i] = koin[L]
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)
    for p_ in model.parameters(): p_.requires_grad_(False)

    rows = []
    for L in layers:
        Hl = H[L].astype(np.float64)
        w_res = fisher_axis(Hl[tr], y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        delta = unit_raw(Hl[y == 1].mean(0) - Hl[y == 0].mean(0))
        W_O = blocks[L].self_attn.o_proj.weight.detach().float().cpu().numpy()    # (d, nh*hd)
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            head_vec = OIN[L][:, sl].astype(np.float64) @ W_O[:, sl].T            # (Pn, d)
            hn = float(np.mean(np.linalg.norm(head_vec, axis=1)))
            fw, pw = frac_along(head_vec, w_res); fu, pu = frac_along(head_vec, u)
            fd, pd = frac_along(head_vec, delta); fg, pg = frac_along(head_vec, gamma)
            rows.append(dict(layer=int(L), head=int(h), head_norm=hn,
                             frac_wres=fw, frac_u=fu, frac_delta=fd, frac_gamma=fg,
                             proj_wres=pw, proj_u=pu, proj_delta=pd, proj_gamma=pg))
        # per-layer: identify answer-writers (gamma) and axis-writers (wres) by norm-weighted projection
        lr = [r for r in rows if r["layer"] == L]
        gscore = np.array([abs(r["proj_gamma"]) * r["head_norm"] for r in lr])
        wscore = np.array([abs(r["proj_wres"]) * r["head_norm"] for r in lr])
        aw = set(np.argsort(gscore)[::-1][:args.topk_heads].tolist())
        xw = set(np.argsort(wscore)[::-1][:args.topk_heads].tolist())
        logger.info("L%02d | answer-writers(γ)=%s axis-writers(w)=%s overlap=%d | top-γ head frac: γ=%.2f u=%.2f w=%.2f δ=%.2f",
                    L, sorted(aw), sorted(xw), len(aw & xw),
                    lr[max(range(len(lr)), key=lambda k: gscore[k])]["frac_gamma"],
                    lr[max(range(len(lr)), key=lambda k: gscore[k])]["frac_u"],
                    lr[max(range(len(lr)), key=lambda k: gscore[k])]["frac_wres"],
                    lr[max(range(len(lr)), key=lambda k: gscore[k])]["frac_delta"])

    with open(out / "head_triad.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 104)
    print("HEAD -> TRIAD — which channel does each head write into (read/use/write/readout)?")
    print("=" * 104)
    print("Per layer: mean fraction of head output-norm in each channel, over the TOP answer-writer & axis-writer head.")
    print("layer | answer-writer head: fracs (γ/u/w/δ) | axis-writer head: fracs (γ/u/w/δ) | aw∩xw")
    for L in layers:
        lr = [r for r in rows if r["layer"] == L]
        gscore = [abs(r["proj_gamma"]) * r["head_norm"] for r in lr]
        wscore = [abs(r["proj_wres"]) * r["head_norm"] for r in lr]
        a = lr[int(np.argmax(gscore))]; x = lr[int(np.argmax(wscore))]
        aw = set(np.argsort(gscore)[::-1][:args.topk_heads]); xw = set(np.argsort(wscore)[::-1][:args.topk_heads])
        print(f"  L{L:02d}  | h{a['head']:02d}: {a['frac_gamma']:.2f}/{a['frac_u']:.2f}/{a['frac_wres']:.2f}/{a['frac_delta']:.2f}   "
              f"| h{x['head']:02d}: {x['frac_gamma']:.2f}/{x['frac_u']:.2f}/{x['frac_wres']:.2f}/{x['frac_delta']:.2f}   | {len(aw & xw)}")
    print("\nReading: if answer-writer heads put most of their norm in γ/u (readout/used) and axis-writer heads put")
    print("most in w_res (read), and the two sets barely overlap, then attention writes the READ axis and the USED")
    print("axis with DIFFERENT heads -- the head-level decoded != used. δ fraction shows who writes the class-mean")
    print("shift. Per-(layer,head) fractions saved to head_triad.csv.")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/head_triad")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[-1])
    p.add_argument("--topk_heads", type=int, default=5)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test: self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
