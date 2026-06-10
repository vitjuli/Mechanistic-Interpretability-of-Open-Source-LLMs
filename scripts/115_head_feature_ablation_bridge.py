"""
115_head_feature_ablation_bridge.py   [do answer-writer heads feed USED feats, axis-writers feed DECODED-not-used?]
========================================================================================================
Causal bridge between attention heads and the feature quadrants (112b). For chosen writer layers we
classify heads into answer-writers (norm-weighted projection onto gamma) and axis-writers (onto w_res),
exactly as in 114. Then for each selected head (+ random-head controls) we ABLATE it -- zero its slice of
the o_proj input at the answer position -- run a forward, capture downstream transcoder feature activations,
and measure the mean activation DROP for:
    Q1 features (concept-relevant & attributing  = USED)            [from 112b feature_metrics_full.csv]
    Q2 features (concept-relevant & NOT attributing = DECODED-not-used)
    other active features
relative to the clean run and to the RANDOM-head null. The hypothesis: ablating an answer-writer head drops
USED (Q1) feature activation more than DECODED-not-used (Q2); ablating an axis-writer head drops Q2 more.
If so, attention routes the read channel and the used channel into different feature populations -- the
causal head<->feature face of decoded != used.

Reads 112b output: feature_metrics_full.csv (cols: layer, feature, relevant_exp, is_attr, ...).
Scoped: a few writer layers, top-K heads each + random controls, prompt subsample, downstream feature layers.
SELF-TEST (no torch):  python 115_head_feature_ablation_bridge.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("head_feat")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def group_drop(clean_acts, abl_acts, idx):
    """mean over features in idx of mean_prompt |clean - abl| (absolute activation change)."""
    if len(idx) == 0:
        return float("nan")
    diff = np.abs(clean_acts[:, idx] - abl_acts[:, idx])      # (P, |idx|)
    return float(diff.mean())


def self_test():
    rng = np.random.default_rng(0); P, F = 30, 200
    clean = np.abs(rng.standard_normal((P, F)))
    q1 = np.arange(0, 20); q2 = np.arange(20, 40)
    abl = clean.copy()
    abl[:, q1] *= 0.2       # ablation strongly drops Q1 features
    abl[:, q2] *= 0.95      # barely drops Q2
    dq1 = group_drop(clean, abl, q1); dq2 = group_drop(clean, abl, q2)
    assert dq1 > 3 * dq2, f"ablation hits Q1 >> Q2 ({dq1:.2f} vs {dq2:.2f})"
    print(f"[self_test] OK — group drop detects Q1>>Q2 asymmetry ({dq1:.2f} vs {dq2:.2f}).")


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
    nh = model.config.num_attention_heads; hd = getattr(model.config, "head_dim", d // nh)
    W_U = model.lm_head.weight.detach()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    gamma = unit_raw((W_U[b_id] - W_U[a_id]).float().cpu().numpy().astype(np.float64))
    writer_layers = sorted({L for L in args.writer_layers if 0 <= L < n_layers})
    feat_layers = sorted({L for L in args.feature_layers if 0 <= L < n_layers})

    # ---- load 112b feature classification ----
    q1_by, q2_by, act_by = {}, {}, {}
    with open(args.feature_csv) as f:
        for r in _csv.DictReader(f):
            L = int(r["layer"]); fid = int(r["feature"])
            rel = int(r.get("relevant_exp", r.get("relevant_cons", 0))); at = int(r["is_attr"])
            act_by.setdefault(L, []).append(fid)
            if rel and at: q1_by.setdefault(L, []).append(fid)
            elif rel and not at: q2_by.setdefault(L, []).append(fid)
    logger.info("loaded 112b features: layers=%s | example L%d: Q1=%d Q2=%d active=%d",
                sorted(act_by)[:5], feat_layers[0] if feat_layers else -1,
                len(q1_by.get(feat_layers[0], [])) if feat_layers else 0,
                len(q2_by.get(feat_layers[0], [])) if feat_layers else 0,
                len(act_by.get(feat_layers[0], [])) if feat_layers else 0)

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

    # ---- pass 1: classify heads at writer layers (need w_res, gamma proj of each head's write) ----
    H = {L: np.zeros((Pn, d), np.float32) for L in writer_layers}
    OIN = {L: np.zeros((Pn, nh * hd), np.float32) for L in writer_layers}
    logger.info("pass1: classifying heads at writer layers %s ...", writer_layers)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, koin, hs = {}, {}, []
        for L in writer_layers:
            def mk(L=L):
                def pre(m, a): keep[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            def mko(L=L):
                def pre(m, a): koin[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(mko(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**enc, use_cache=False)
            for L in writer_layers:
                H[L][i] = keep[L]; OIN[L][i] = koin[L]
        finally:
            for h in hs: h.remove()
        if (i + 1) % 200 == 0: logger.info("  %d/%d", i + 1, Pn)

    selected = {}    # layer -> dict(answer=[heads], axis=[heads], random=[heads])
    for L in writer_layers:
        w_res = fisher_axis(H[L][tr].astype(np.float64), y[tr], args.shrink)
        W_O = blocks[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
        gsc, wsc = [], []
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            hv = OIN[L][:, sl].astype(np.float64) @ W_O[:, sl].T
            hn = float(np.mean(np.linalg.norm(hv, axis=1)))
            gsc.append(abs(float(np.mean(hv @ gamma))) * hn); wsc.append(abs(float(np.mean(hv @ w_res))) * hn)
        gsc = np.array(gsc); wsc = np.array(wsc)
        aw = np.argsort(gsc)[::-1][:args.topk_heads].tolist()
        xw = np.argsort(wsc)[::-1][:args.topk_heads].tolist()
        pool = [h for h in range(nh) if h not in set(aw) | set(xw)]
        rnd = list(rng.choice(pool, size=min(args.n_random, len(pool)), replace=False))
        selected[L] = dict(answer=aw, axis=xw, random=rnd)
        logger.info("L%02d writers: answer=%s axis=%s (overlap=%d) random=%s", L, aw, xw, len(set(aw) & set(xw)), rnd)

    # ---- transcoders for feature layers ----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcoder"))
    from transcoder import load_transcoder_set
    ts = load_transcoder_set(args.transcoder_set, device=args.device, dtype=torch.bfloat16, lazy_load=True)

    # eval subsample (balanced)
    ai = np.where(y == 0)[0]; bi = np.where(y == 1)[0]; rng.shuffle(ai); rng.shuffle(bi)
    eval_idx = list(map(int, np.r_[ai[: args.n_eval // 2], bi[: args.n_eval // 2]]))

    def feats_for(idx_list, ablate=None):
        """run forwards over idx_list, return {m: (len, Factive_at_m? no -> full F) acts} at answer pos.
        ablate=(layer, head) zeros that head's o_proj-input slice at the last position."""
        accmin = {m: np.zeros((len(idx_list), d), np.float32) for m in feat_layers}
        for k, i in enumerate(idx_list):
            enc = tok([forced(i)], return_tensors="pt").to(args.device); kmin, hs = {}, []
            for m in feat_layers:
                def mkm(m=m):
                    def pre(mm, a): kmin[m] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                    return pre
                hs.append(blocks[m].mlp.register_forward_pre_hook(mkm(), with_kwargs=False))
            if ablate is not None:
                L0, h0 = ablate
                def abl_hook(mm, a, h0=h0):
                    a[0][:, -1, h0 * hd:(h0 + 1) * hd] = 0.0; return None
                hs.append(blocks[L0].self_attn.o_proj.register_forward_pre_hook(abl_hook, with_kwargs=False))
            try:
                with torch.no_grad():
                    model(**enc, use_cache=False)
                for m in feat_layers:
                    accmin[m][k] = kmin[m]
            finally:
                for h in hs: h.remove()
        # encode to feature acts
        out = {}
        for m in feat_layers:
            with torch.no_grad():
                out[m] = ts[m].encode(torch.tensor(accmin[m], dtype=torch.bfloat16, device=args.device)).float().cpu().numpy()
        return out

    logger.info("computing CLEAN feature acts on %d prompts ...", len(eval_idx))
    clean = feats_for(eval_idx, ablate=None)

    rows = []
    for L in writer_layers:
        for kind in ("answer", "axis", "random"):
            for h in selected[L][kind]:
                abl = feats_for(eval_idx, ablate=(L, h))
                for m in feat_layers:
                    if m <= L:        # only downstream features are causally affected
                        continue
                    q1 = q1_by.get(m, []); q2 = q2_by.get(m, []); other = list(set(act_by.get(m, [])) - set(q1) - set(q2))
                    dq1 = group_drop(clean[m], abl[m], q1); dq2 = group_drop(clean[m], abl[m], q2)
                    doth = group_drop(clean[m], abl[m], other)
                    rows.append(dict(writer_layer=int(L), head=int(h), kind=kind, feat_layer=int(m),
                                     n_Q1=len(q1), n_Q2=len(q2), drop_Q1=dq1, drop_Q2=dq2, drop_other=doth,
                                     ratio_Q1_Q2=float(dq1 / (dq2 + 1e-9)) if (dq1 == dq1 and dq2 == dq2) else float("nan")))
            logger.info("L%02d %-6s heads done", L, kind)

    with open(out / "head_feature_ablation.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    # ---- summary: average over heads of each kind & downstream feat layers ----
    print("\n" + "=" * 104)
    print("HEAD -> FEATURE ablation — do answer-writers feed USED(Q1) feats, axis-writers feed DECODED-not-used(Q2)?")
    print("=" * 104)
    print("Mean activation drop when ablating a head, averaged over heads of each kind & downstream feature layers.")
    print("writer L | kind   | drop Q1(used) | drop Q2(dec-not-used) | drop other | Q1/Q2 ratio")
    for L in writer_layers:
        for kind in ("answer", "axis", "random"):
            sub = [r for r in rows if r["writer_layer"] == L and r["kind"] == kind and r["feat_layer"] > L]
            if not sub: continue
            mq1 = np.nanmean([r["drop_Q1"] for r in sub]); mq2 = np.nanmean([r["drop_Q2"] for r in sub])
            moth = np.nanmean([r["drop_other"] for r in sub])
            print(f"   L{L:02d}    | {kind:6s} |   {mq1:.4f}     |     {mq2:.4f}        |  {moth:.4f}   |  {mq1/(mq2+1e-9):.2f}")
    print("\nReading: compare answer-writer vs axis-writer vs random. If answer-writers drop Q1(used) more (high Q1/Q2)")
    print("and axis-writers drop Q2(decoded-not-used) more (low Q1/Q2), each above the random-head baseline, then")
    print("attention routes read and used channels into different feature populations -- causal head<->feature")
    print("decoded != used. If all kinds look like random, the head->feature routing is not channel-specific.")
    print("Saved head_feature_ablation.csv (per head x downstream layer).")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--feature_csv", default="data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/head_feat_ablation")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--writer_layers", type=int, nargs="*", default=[21, 24])
    p.add_argument("--feature_layers", type=int, nargs="*", default=[24, 28, 32, 35])
    p.add_argument("--topk_heads", type=int, default=5)
    p.add_argument("--n_random", type=int, default=8)
    p.add_argument("--n_eval", type=int, default=24)
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
