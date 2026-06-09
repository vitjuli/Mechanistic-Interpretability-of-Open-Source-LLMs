"""
112_feature_attribution_quadrants.py   [concept-selectivity x attribution: the 2x2 feature map]
========================================================================================================
The feature-level version of the whole thesis. For each active transcoder feature, two independent axes:
  selectivity = AUC( activation a_f , label alpha/beta )   -- does the feature's FIRING track the concept?
  attribution = mean_prompt a_f * < W_dec[f], g >          -- does the feature DRIVE the answer? (used channel)
Cross them into four quadrants (thresholds sel_thr on |AUC-0.5|*2, attr_thr on |attribution| percentile):
  Q1  selective & attributing      -> concept features that actually drive the answer
  Q2  selective & NOT attributing  -> DECODED-BUT-NOT-USED features (concept present in firing, no answer effect)
  Q3  NOT selective & attributing  -> USED-BUT-NOT-CONCEPT features (drive the answer w/o tracking physics = surface/format route)
  Q4  neither
Per layer: quadrant counts, example feature IDs for Q2 and Q3 (the informative off-diagonals), correlation
between selectivity and |attribution|, and nulls (shuffle labels for selectivity; shuffle prompts for attr).
If Q1 is sparse while Q2 (selective-not-used) and Q3 (used-not-concept) are populated, the concept is decoded
by features that don't drive the answer, and the answer is driven by features that don't encode the concept.

Needs transcoders. SELF-TEST (no torch): python 112_feature_attribution_quadrants.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("attr_quad")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def auc_cols(A, y):
    """AUC of each column of A (P x F) vs binary y (P,). Returns (F,) in [0,1]."""
    y = np.asarray(y, int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.full(A.shape[1], np.nan)
    out = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        s = A[:, j]; order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        # (cheap tie handling: group equal values)
        out[j] = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def self_test():
    rng = np.random.default_rng(0); d, F, P = 64, 400, 240
    y = (rng.random(P) > 0.5).astype(int)
    Wdec = rng.standard_normal((F, d)); g = rng.standard_normal((P, d)); gbar = unit_raw(g.mean(0))
    a = np.abs(rng.standard_normal((P, F)))
    # Q1 (sel & attr): 0-19 fire more for beta AND decoder ~ gbar
    a[y == 1, :20] += 3.0; Wdec[:20] = gbar[None] * 3 + 0.2 * rng.standard_normal((20, d))
    # Q2 (sel, no attr): 20-39 fire more for beta but decoder _|_ gbar
    a[y == 1, 20:40] += 3.0; Wdec[20:40] -= (Wdec[20:40] @ gbar)[:, None] * gbar
    # Q3 (attr, not sel): 40-59 fire same for both but decoder ~ gbar
    Wdec[40:60] = gbar[None] * 3 + 0.2 * rng.standard_normal((20, d))
    sel = np.abs(auc_cols(a, y) - 0.5) * 2
    attr = np.abs((a * (Wdec @ g.T).T).mean(0))
    sthr = 0.3; athr = np.percentile(attr, 80)
    q1 = np.mean((sel[:20] > sthr) & (attr[:20] > athr))
    q2 = np.mean((sel[20:40] > sthr) & (attr[20:40] <= athr))
    q3 = np.mean((sel[40:60] <= sthr) & (attr[40:60] > athr))
    assert q1 > 0.5 and q2 > 0.5 and q3 > 0.4, f"quadrants recovered q1={q1:.2f} q2={q2:.2f} q3={q3:.2f}"
    print(f"[self_test] OK — 2x2 quadrants recovered (q1={q1:.2f} q2={q2:.2f} q3={q3:.2f}).")


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

    rows, ex_rows = [], []
    for L in layers:
        with torch.no_grad():
            a = ts[L].encode(torch.tensor(MIN[L], dtype=torch.bfloat16, device=args.device)).float().cpu().numpy()
        fire_rate = (a > 0).mean(0); active = np.where(fire_rate >= args.tau)[0]
        if len(active) < 5:
            logger.info("L%02d: %d active -> skip", L, len(active)); continue
        Wdec = ts[L].W_dec.detach().float().cpu().numpy()
        aA = a[:, active]; Da = Wdec[active]; Gl = G[L].astype(np.float64)
        attr_f = (aA * (Da @ Gl.T).T).mean(0)              # signed attribution per feature
        abs_attr = np.abs(attr_f)
        sel_auc = auc_cols(aA, y); sel = np.abs(sel_auc - 0.5) * 2     # concept-selectivity in [0,1]
        sthr = args.sel_thr; athr = np.percentile(abs_attr, args.attr_pct)
        is_sel = sel > sthr; is_attr = abs_attr > athr
        q1 = int(np.sum(is_sel & is_attr)); q2 = int(np.sum(is_sel & ~is_attr))
        q3 = int(np.sum(~is_sel & is_attr)); q4 = int(np.sum(~is_sel & ~is_attr))
        corr = float(np.corrcoef(sel, abs_attr)[0, 1]) if len(active) > 2 else float("nan")
        # nulls
        sel_null = np.abs(auc_cols(aA, rng.permutation(y)) - 0.5) * 2
        perm = rng.permutation(Pn); attr_null = np.abs((aA * (Da @ Gl[perm].T).T).mean(0))
        rows.append(dict(layer=int(L), n_active=int(len(active)),
                         Q1_sel_attr=q1, Q2_sel_NOTattr=q2, Q3_NOTsel_attr=q3, Q4_neither=q4,
                         corr_sel_absattr=corr,
                         frac_sel=float(np.mean(is_sel)), frac_attr=float(np.mean(is_attr)),
                         mean_sel_null=float(np.mean(sel_null > sthr)),
                         attr_thr=float(athr), sel_thr=float(sthr)))
        # example features for the informative off-diagonals (Q2 decoded-not-used, Q3 used-not-concept)
        for tag, mask in (("Q2_sel_NOTattr", is_sel & ~is_attr), ("Q3_NOTsel_attr", ~is_sel & is_attr)):
            idx = np.where(mask)[0]
            order = idx[np.argsort(sel[idx])[::-1]] if tag.startswith("Q2") else idx[np.argsort(abs_attr[idx])[::-1]]
            for j in order[: args.n_examples]:
                ex_rows.append(dict(layer=int(L), quadrant=tag, feature=int(active[j]),
                                    selectivity=float(sel[j]), sel_auc=float(sel_auc[j]),
                                    attribution=float(attr_f[j]), fire_rate=float(fire_rate[active[j]])))
        logger.info("L%02d |A|=%4d | Q1(sel&attr)=%d Q2(sel,¬attr)=%d Q3(¬sel,attr)=%d Q4=%d | corr(sel,|attr|)=%+.3f | sel-null=%.0f%%",
                    L, len(active), q1, q2, q3, q4, corr, 100 * float(np.mean(sel_null > sthr)))
        del Wdec, aA, Da; torch.cuda.empty_cache()

    with open(out / "attribution_quadrants.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    with open(out / "attribution_quadrant_examples.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(ex_rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in ex_rows]

    print("\n" + "=" * 100)
    print("FEATURE QUADRANTS — concept-selectivity x attribution (the 2x2 feature map of the thesis)")
    print("=" * 100)
    print("layer | #active | Q1 sel&attr | Q2 sel&¬attr (decoded-not-used) | Q3 ¬sel&attr (used-not-concept) | Q4 | corr")
    for r in rows:
        print(f"  L{r['layer']:02d}  |  {r['n_active']:4d}  |     {r['Q1_sel_attr']:3d}     |             "
              f"{r['Q2_sel_NOTattr']:4d}                |              {r['Q3_NOTsel_attr']:4d}               "
              f"| {r['Q4_neither']:4d} | {r['corr_sel_absattr']:+.2f}")
    print("\nReading: sparse Q1 + populated Q2 (concept-selective features that DON'T drive the answer) + populated Q3")
    print("(features that DRIVE the answer without tracking the concept = surface/format route) is the feature-level")
    print("decoded/written != used. corr(sel,|attr|) near 0 confirms selectivity and usage are independent across")
    print("features. sel-null% is the false-positive selectivity rate under shuffled labels (subtract it mentally from")
    print("frac_sel). Example Q2/Q3 feature IDs in attribution_quadrant_examples.csv -> cross-ref to semantics.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/attr_quad")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[13, 17, 21, 24, 28, 32, 35])
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--sel_thr", type=float, default=0.3)
    p.add_argument("--attr_pct", type=float, default=90.0)
    p.add_argument("--n_examples", type=int, default=15)
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
