"""
111_feature_attribution_map.py   [activation (present) vs attribution (used) at the feature level + flow]
========================================================================================================
Geometry told us how features LIE; attribution tells us what they DO for the answer -- the functional axis
that separates 'used' from 'decoded/written'. Linearized attribution (attribution-graph style, valid here:
exp 89 margin R^2>=0.98) of feature f to the answer margin:
        attr_f(prompt) = a_f(prompt) * < W_dec[f], g(prompt) >,   g = grad(logit_beta - logit_alpha) at layer
The feature's contribution to the residual is a_f * W_dec[f]; its first-order effect on the margin is its
projection onto the used gradient g. Per layer we build two per-feature axes over ACTIVE features:
   activation  = mean_prompt a_f            (presence  -> decoded/written side)
   attribution = mean_prompt attr_f         (contribution to the answer -> used side)
and report: (i) correlation between activation and |attribution| across features (low => present-but-not-
used), (ii) overlap of top-activation vs top-attribution feature sets, (iii) total |attribution| per layer
= attribution FLOW across depth (does it ramp where u ignites, L21-24, cf commitment onset L17?), with a
label-shuffled null for attribution (shuffle g's class structure). Functional complement to the geometry.

Needs transcoders. SELF-TEST (no torch): python 111_feature_attribution_map.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("attr_map")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def self_test():
    rng = np.random.default_rng(0); d, F, P = 256, 300, 120   # d>P so G has a null space
    Wdec = rng.standard_normal((F, d))
    g = rng.standard_normal((P, d)); gbar = unit_raw(g.mean(0))
    a = np.abs(rng.standard_normal((P, F)))
    # null space of G (rows): decoder rows here have <W,g_i>=0 for ALL prompts -> exactly zero attribution
    _, _, Vt = np.linalg.svd(g, full_matrices=True); null_basis = Vt[P:].T      # d x (d-P)
    Wdec[:30] = (null_basis @ rng.standard_normal((d - P, 30))).T               # present-but-not-used
    a[:, :30] *= 5.0                                                            # ...and very active
    Wdec[30:60] = gbar[None, :] * 3.0 + 0.2 * rng.standard_normal((30, d))      # used (aligned with g)
    attr = a * (Wdec @ g.T).T
    act = a.mean(0); at = np.abs(attr.mean(0))
    top_act = set(np.argsort(act)[::-1][:30]); top_at = set(np.argsort(at)[::-1][:30])
    corr = float(np.corrcoef(act, at)[0, 1])
    assert at[:30].max() < 1e-6, "null-space features have ~zero attribution despite high activation"
    assert len(top_act & top_at) < 8, "top-activation and top-attribution features largely differ"
    print(f"[self_test] OK — activation/attribution decouple (corr={corr:+.2f}, top-overlap={len(top_act&top_at)}/30).")


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
    def mlp_in_module(L): return blocks[L].mlp

    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    MIN = {L: np.zeros((Pn, d), np.float32) for L in layers}
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing grad+mlp_input over %d prompts; layers=%s ...", Pn, layers)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, kmin, hs = {}, {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            def mkm(L=L):
                def pre(m, a): kmin[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            hs.append(mlp_in_module(L).register_forward_pre_hook(mkm(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                MIN[L][i] = kmin[L]
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)
    for p_ in model.parameters(): p_.requires_grad_(False)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcoder"))
    from transcoder import load_transcoder_set
    ts = load_transcoder_set(args.transcoder_set, device=args.device, dtype=torch.bfloat16, lazy_load=True)

    rows, top_rows = [], []
    for L in layers:
        with torch.no_grad():
            a = ts[L].encode(torch.tensor(MIN[L], dtype=torch.bfloat16, device=args.device)).float().cpu().numpy()  # (Pn,F)
        fire_rate = (a > 0).mean(0); active = np.where(fire_rate >= args.tau)[0]
        if len(active) < 3:
            logger.info("L%02d: %d active -> skip", L, len(active)); continue
        Wdec = ts[L].W_dec.detach().float().cpu().numpy()           # (F,d)
        Da = Wdec[active]                                            # (|A|,d)
        aA = a[:, active]                                            # (Pn,|A|)
        Gl = G[L].astype(np.float64)                                # (Pn,d)
        dotg = (Da @ Gl.T).T                                        # (Pn,|A|) = <W_dec[f], g_prompt>
        attr = aA * dotg                                            # (Pn,|A|) linearized attribution
        act_f = aA.mean(0)                                          # activation per feature
        attr_f = attr.mean(0)                                       # signed attribution per feature
        abs_attr_f = np.abs(attr_f)
        # correlation activation vs |attribution| across features
        corr = float(np.corrcoef(act_f, abs_attr_f)[0, 1]) if len(active) > 2 else float("nan")
        # top sets
        K = min(args.topk, len(active))
        top_act = set(np.argsort(act_f)[::-1][:K].tolist())
        top_attr = set(np.argsort(abs_attr_f)[::-1][:K].tolist())
        overlap = len(top_act & top_attr)
        # attribution flow: total |attribution| this layer + null (shuffle prompt-class of g via row shuffle)
        total_abs_attr = float(np.sum(abs_attr_f))
        nulls = []
        for _ in range(args.n_null):
            perm = rng.permutation(Pn)
            attr_n = aA * (Da @ Gl[perm].T).T
            nulls.append(float(np.sum(np.abs(attr_n.mean(0)))))
        total_abs_attr_null = float(np.mean(nulls))
        rows.append(dict(layer=int(L), n_active=int(len(active)),
                         corr_act_absattr=corr, top_overlap=overlap, topk=K,
                         total_abs_attr=total_abs_attr, total_abs_attr_null=total_abs_attr_null,
                         attr_over_null=total_abs_attr / (total_abs_attr_null + 1e-12)))
        # save top features (by attribution and by activation) with both metrics
        order_at = np.argsort(abs_attr_f)[::-1][:K]
        for rk, j in enumerate(order_at.tolist()):
            top_rows.append(dict(layer=int(L), rank=rk, feature=int(active[j]),
                                 activation=float(act_f[j]), attribution=float(attr_f[j]),
                                 fire_rate=float(fire_rate[active[j]])))
        logger.info("L%02d |A|=%4d | corr(act,|attr|)=%+.3f | top-%d overlap(act,attr)=%d | flow |attr|=%.1f (null %.1f, x%.1f)",
                    L, len(active), corr, K, overlap, total_abs_attr, total_abs_attr_null,
                    total_abs_attr / (total_abs_attr_null + 1e-12))
        del Wdec, Da, aA, dotg, attr; torch.cuda.empty_cache()

    with open(out / "attribution_map.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    with open(out / "attribution_top_features.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(top_rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in top_rows]

    print("\n" + "=" * 100)
    print("FEATURE ATTRIBUTION MAP — activation (present) vs attribution (used), and attribution flow by layer")
    print("=" * 100)
    print("layer | #active | corr(act,|attr|) | top overlap(act,attr) | attribution flow |attr| (x null)")
    for r in rows:
        print(f"  L{r['layer']:02d}  |  {r['n_active']:4d}  |     {r['corr_act_absattr']:+.3f}      |          "
              f"{r['top_overlap']:>2d}           |   {r['total_abs_attr']:.1f}  (x{r['attr_over_null']:.1f})")
    print("\nReading: low corr(act,|attr|) + small top-overlap => the features that are PRESENT (high activation)")
    print("are not the ones that are USED (high attribution): present-but-not-used at the feature level (functional")
    print("decoded/written != used). Attribution flow rising sharply late (and x-null growing) locates WHERE features")
    print("start to drive the answer; compare to u-ignition (L21-24) and commitment onset (L17). Saved attribution_map.csv,")
    print("attribution_top_features.csv (top features by attribution with their activation -- cross-ref to semantics).")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/attr_map")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[-1])
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n_null", type=int, default=5)
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
