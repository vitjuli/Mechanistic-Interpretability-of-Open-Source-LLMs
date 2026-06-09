"""
109_feature_alignment_u_vs_wres.py   [which dictionary features load u (used) vs w_res (read)?]
========================================================================================================
We know u (used) and w_res (read) are geometrically near-orthogonal. This asks what they MEAN in dictionary
terms: which transcoder features align with each? For chosen layers, compute, over all transcoder features
f with decoder row d_f = W_dec[f]:
    cos(d_f, u_hat)   and   cos(d_f, w_res_hat)
and report (i) the top-k features most aligned with u and with w_res, (ii) the OVERLAP of the two top sets,
(iii) alignment distribution (max |cos|, how many features exceed a threshold for each direction), (iv) the
correlation between a feature's u-alignment and its w_res-alignment across the dictionary. If the top-u and
top-w_res feature sets are largely DISJOINT and per-feature alignments are uncorrelated, then 'used' and
'read' are carried by different features -- decoded != used gets a content interpretation, not just an angle.
(Whether those features actually fire is exp 91's question: aligned carrier features there did NOT fire.)

Loads transcoders lazily per layer (matches exp 91). Heavier than 106-108 (GPU matmul over the dictionary).
SELF-TEST (no torch):  python 109_feature_alignment_u_vs_wres.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("feat_align")


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
    rng = np.random.default_rng(0); d, F = 64, 2000
    u = unit_raw(rng.standard_normal(d)); rw = rng.standard_normal(d); w = unit_raw(rw - (rw @ u) * u)  # w _|_ u
    Wdec = rng.standard_normal((F, d))
    # plant: features 0-9 aligned with u, features 10-19 aligned with w, disjoint
    Wdec[:10] += u * 6.0; Wdec[10:20] += w * 6.0
    Wn = Wdec / (np.linalg.norm(Wdec, axis=1, keepdims=True) + 1e-9)
    cu = np.abs(Wn @ u); cw = np.abs(Wn @ w)
    top_u = set(np.argsort(cu)[::-1][:10]); top_w = set(np.argsort(cw)[::-1][:10])
    assert len(top_u & top_w) == 0, "u-aligned and w-aligned top features are disjoint"
    assert abs(np.corrcoef(cu, cw)[0, 1]) < 0.2, "per-feature u/w alignment uncorrelated"
    print("[self_test] OK — disjoint top sets + uncorrelated per-feature alignment recovered.")


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
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})

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

    # capture u, w_res at chosen layers
    H = {L: np.zeros((Pn, d), np.float32) for L in layers}
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing residual+grad over %d prompts; layers=%s ...", Pn, layers)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, hs = {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)

    # free model grad memory before loading transcoders
    for p_ in model.parameters(): p_.requires_grad_(False)

    # ---- load transcoders (lazy) ----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcoder"))
    from transcoder import load_transcoder_set
    ts = load_transcoder_set(args.transcoder_set, device=args.device, dtype=torch.bfloat16, lazy_load=True)

    rows, top_rows = [], []
    K = args.topk
    for L in layers:
        u = torch.tensor(unit_raw(G[L][tr].astype(np.float64).mean(0)), dtype=torch.float32, device=args.device)
        w = torch.tensor(fisher_axis(H[L][tr].astype(np.float64), y[tr], args.shrink), dtype=torch.float32, device=args.device)
        Wdec = ts[L].W_dec.to(args.device).float()                # (F, d)
        Wn = Wdec / (Wdec.norm(dim=1, keepdim=True) + 1e-9)
        cu = (Wn @ u).abs(); cw = (Wn @ w).abs()
        F = Wn.shape[0]
        top_u = torch.topk(cu, K).indices.cpu().numpy(); top_w = torch.topk(cw, K).indices.cpu().numpy()
        overlap = len(set(top_u.tolist()) & set(top_w.tolist()))
        # per-feature correlation between u-alignment and w-alignment (subsample for speed)
        idx = torch.randperm(F, device=args.device)[: min(F, 20000)]
        cuc = cu[idx].cpu().numpy(); cwc = cw[idx].cpu().numpy()
        corr = float(np.corrcoef(cuc, cwc)[0, 1])
        thr = args.cos_thr
        rows.append(dict(layer=int(L), n_features=int(F),
                         max_cos_u=float(cu.max().item()), max_cos_w=float(cw.max().item()),
                         n_feat_cos_u_gt=int((cu > thr).sum().item()), n_feat_cos_w_gt=int((cw > thr).sum().item()),
                         top_overlap=overlap, topk=K, per_feature_corr_u_w=corr))
        for rank, (fid, cv) in enumerate(zip(top_u.tolist(), torch.topk(cu, K).values.cpu().numpy().tolist())):
            top_rows.append(dict(layer=int(L), direction="u", rank=rank, feature=int(fid), cos=float(cv)))
        for rank, (fid, cv) in enumerate(zip(top_w.tolist(), torch.topk(cw, K).values.cpu().numpy().tolist())):
            top_rows.append(dict(layer=int(L), direction="w_res", rank=rank, feature=int(fid), cos=float(cv)))
        logger.info("L%02d | max cos(d_f,u)=%.3f max cos(d_f,w)=%.3f | #|cos>%.2f|: u=%d w=%d | top-%d overlap=%d | corr(u,w)=%+.3f",
                    L, rows[-1]["max_cos_u"], rows[-1]["max_cos_w"], thr,
                    rows[-1]["n_feat_cos_u_gt"], rows[-1]["n_feat_cos_w_gt"], K, overlap, corr)
        del Wdec, Wn, cu, cw; torch.cuda.empty_cache()

    with open(out / "feature_alignment.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    with open(out / "feature_alignment_top.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(top_rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in top_rows]

    print("\n" + "=" * 100)
    print("FEATURE ALIGNMENT — which dictionary features load u (used) vs w_res (read)?")
    print("=" * 100)
    print("layer | max cos(d_f,u) | max cos(d_f,w_res) | top-%d overlap | per-feature corr(u-align, w-align)" % K)
    for r in rows:
        print(f"  L{r['layer']:02d}  |     {r['max_cos_u']:.3f}      |       {r['max_cos_w']:.3f}        "
              f"|       {r['top_overlap']:>2d}        |          {r['per_feature_corr_u_w']:+.3f}")
    print("\nReading: small top-overlap + near-zero per-feature corr => the features aligned with the USED direction")
    print("are different from those aligned with the READ axis: used and read are carried by different dictionary")
    print("features (content meaning of decoded != used). Cross-reference the top feature IDs (feature_alignment_top.csv)")
    print("to their semantics via the feature-analysis pipeline. NB: alignment is geometric; whether these features")
    print("actually fire is exp 91 (there: aligned carrier features did not fire). Saved feature_alignment*.csv")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/feat_align")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[13, 17, 21, 24])
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--cos_thr", type=float, default=0.10)
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
