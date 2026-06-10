"""
112b_feature_quadrants_multimetric.py   [concept-relevance x attribution, 5 metrics, 2 defs, all 36 layers]
========================================================================================================
Fixes the AUC blind spot raised in review: a feature can be concept-relevant yet have global AUC ~ 0.5 if
it (a) fires strongly on BOTH classes, or (b) fires only on a NARROW SUBGROUP of one class. AUC sees only a
monotonic shift of means. So we measure concept-relevance with FIVE complementary faces, then build the 2x2
quadrants under TWO definitions (conservative = AUC only; expanded = relevant by ANY face) to show robustness.

Per active feature (fires on >= tau of prompts):
  - auc_global    : AUC(activation, label)              -- magnitude discrimination (the original axis)
  - fire_sel      : |fire_rate_beta - fire_rate_alpha|  -- PRESENCE discrimination (catches narrow subgroups)
  - mi_label      : MI(binned activation, label)        -- ANY dependence (nonlinear/threshold/bimodal); AUC is its monotonic case
  - fire_rate, mean_act_active                          -- ENGAGEMENT (separates 'fires on both' from 'dead')
  - mi_family     : MI(binned activation, surface_family) -- surface-structure diagnostic (not used in the concept gate)
  - attribution   : mean a_f * <W_dec[f], g>            -- USED axis (linearized; exp 89 R^2>=0.98)

Thresholds are NULL-CALIBRATED: selectivity faces vs shuffled labels (95th pct), attribution vs shuffled
prompts (95th pct). Concept-relevant_conservative = auc-face exceeds null. Concept-relevant_expanded = ANY
of {auc, fire_sel, mi_label} exceeds null. Quadrants:
  Q1 relevant & attributing | Q2 relevant & NOT attributing (decoded-not-used) | Q3 NOT relevant & attributing
  (used-not-concept / surface route) | Q4 neither -- computed for BOTH definitions.

Needs transcoders. All 36 layers by default. SELF-TEST (no torch): python 112b_feature_quadrants_multimetric.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("quad2")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def auc_cols(A, y):
    y = np.asarray(y, int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.full(A.shape[1], np.nan)
    out = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        s = A[:, j]; order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        out[j] = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def bin_activation(col, n_pos=3):
    """0 = inactive; 1..n_pos = terciles of the positive activations."""
    b = np.zeros(len(col), int); pos = col > 0
    if pos.sum() >= n_pos:
        qs = np.quantile(col[pos], np.linspace(0, 1, n_pos + 1)[1:-1])
        b[pos] = 1 + np.digitize(col[pos], qs)
    elif pos.sum() > 0:
        b[pos] = 1
    return b


def discrete_mi(x, z):
    """mutual information (nats) between two integer-labelled vectors."""
    x = np.asarray(x); z = np.asarray(z); n = len(x)
    xs = np.unique(x); zs = np.unique(z)
    if len(xs) < 2 or len(zs) < 2: return 0.0
    mi = 0.0
    px = {v: np.mean(x == v) for v in xs}; pz = {v: np.mean(z == v) for v in zs}
    for v in xs:
        xm = (x == v)
        for w in zs:
            pxy = np.mean(xm & (z == w))
            if pxy > 0:
                mi += pxy * np.log(pxy / (px[v] * pz[w] + 1e-30))
    return float(mi)


def self_test():
    rng = np.random.default_rng(0); P = 400
    y = (rng.random(P) > 0.5).astype(int)
    # feature A: classic magnitude discrimination -> high AUC
    fa = np.abs(rng.standard_normal(P)) + 1.5 * y
    # feature B: NARROW SUBGROUP -> fires only on ~8% of beta, zero elsewhere; AUC~0.5 but fire_sel/mi high
    fb = np.zeros(P); bsub = np.where(y == 1)[0]; fire = bsub[: max(1, len(bsub) // 12)]; fb[fire] = 2.0 + rng.random(len(fire))
    # feature C: fires strongly on BOTH classes equally -> AUC~0.5, mi_label~0, high engagement
    fc = np.abs(rng.standard_normal(P)) + 2.0
    def faces(col):
        a = bin_activation(col)
        auc = auc_cols(col[:, None], y)[0]; sel = abs(2 * auc - 1)
        fr_a = np.mean(col[y == 0] > 0); fr_b = np.mean(col[y == 1] > 0); fsel = abs(fr_b - fr_a)
        mil = discrete_mi(a, y)
        return sel, fsel, mil, np.mean(col > 0)
    sA, fA, mA, eA = faces(fa); sB, fB, mB, eB = faces(fb); sC, fC, mC, eC = faces(fc)
    assert sA > 0.4, "A discriminates by magnitude (AUC)"
    assert sB < 0.25 and (fB > 0.05 or mB > 0.01), "B: narrow subgroup hidden from AUC but caught by fire_sel/MI"
    assert sC < 0.2 and mC < 0.02 and eC > 0.9, "C: fires on both, AUC~0.5 & MI~0 but high engagement"
    print(f"[self_test] OK — A(auc={sA:.2f}) B(auc={sB:.2f},fire={fB:.2f},mi={mB:.3f}) C(auc={sC:.2f},mi={mC:.3f},eng={eC:.2f}); narrow & both-class cases recovered.")


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
    fam_ids = {f: i for i, f in enumerate(sorted(set(fams)))}
    fam_arr = np.array([fam_ids[f] for f in fams])
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
    logger.info("capturing grad+mlp_input over %d prompts; %d layers ...", Pn, len(layers))
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

    rows, feat_rows = [], []
    for L in layers:
        with torch.no_grad():
            a = ts[L].encode(torch.tensor(MIN[L], dtype=torch.bfloat16, device=args.device)).float().cpu().numpy()
        fire_rate = (a > 0).mean(0); active = np.where(fire_rate >= args.tau)[0]
        if len(active) < 5:
            logger.info("L%02d: %d active -> skip", L, len(active)); continue
        Wdec = ts[L].W_dec.detach().float().cpu().numpy()
        aA = a[:, active]; Da = Wdec[active]; Gl = G[L].astype(np.float64)
        nA = len(active)

        # ---- attribution ----
        attr_f = (aA * (Da @ Gl.T).T).mean(0); abs_attr = np.abs(attr_f)

        # ---- concept-relevance faces ----
        auc_g = auc_cols(aA, y); sel_auc = np.abs(2 * auc_g - 1)
        fr_a = (aA[y == 0] > 0).mean(0); fr_b = (aA[y == 1] > 0).mean(0); fire_sel = np.abs(fr_b - fr_a)
        binned = np.stack([bin_activation(aA[:, j]) for j in range(nA)], axis=1)   # (Pn, nA)
        mi_label = np.array([discrete_mi(binned[:, j], y) for j in range(nA)])
        mi_family = np.array([discrete_mi(binned[:, j], fam_arr) for j in range(nA)])
        mean_act_active = np.array([aA[aA[:, j] > 0, j].mean() if (aA[:, j] > 0).any() else 0.0 for j in range(nA)])

        # ---- null calibration (shuffle labels for selectivity faces) ----
        Kn = args.n_null_sel
        null_auc, null_fire, null_mi = [], [], []
        for _ in range(Kn):
            ys = rng.permutation(y)
            null_auc.append(np.abs(2 * auc_cols(aA, ys) - 1))
            null_fire.append(np.abs((aA[ys == 1] > 0).mean(0) - (aA[ys == 0] > 0).mean(0)))
            null_mi.append(np.array([discrete_mi(binned[:, j], ys) for j in range(nA)]))
        thr_auc = float(np.percentile(np.concatenate(null_auc), 95))
        thr_fire = float(np.percentile(np.concatenate(null_fire), 95))
        thr_mi = float(np.percentile(np.concatenate(null_mi), 95))
        # attribution null: shuffle prompts of g
        Ka = args.n_null_attr; null_attr = []
        for _ in range(Ka):
            perm = rng.permutation(Pn); null_attr.append(np.abs((aA * (Da @ Gl[perm].T).T).mean(0)))
        thr_attr = float(np.percentile(np.concatenate(null_attr), 95))

        is_attr = abs_attr > thr_attr
        relevant_cons = sel_auc > thr_auc
        relevant_exp = (sel_auc > thr_auc) | (fire_sel > thr_fire) | (mi_label > thr_mi)

        def quad(rel):
            return (int(np.sum(rel & is_attr)), int(np.sum(rel & ~is_attr)),
                    int(np.sum(~rel & is_attr)), int(np.sum(~rel & ~is_attr)))
        c1, c2, c3, c4 = quad(relevant_cons)
        e1, e2, e3, e4 = quad(relevant_exp)
        # how many extra features the expanded def catches that AUC missed, and via which face
        extra = relevant_exp & ~relevant_cons
        extra_fire = int(np.sum(extra & (fire_sel > thr_fire)))
        extra_mi = int(np.sum(extra & (mi_label > thr_mi)))

        rows.append(dict(layer=int(L), n_active=int(nA),
                         cons_Q1=c1, cons_Q2=c2, cons_Q3=c3, cons_Q4=c4,
                         exp_Q1=e1, exp_Q2=e2, exp_Q3=e3, exp_Q4=e4,
                         n_relevant_cons=int(relevant_cons.sum()), n_relevant_exp=int(relevant_exp.sum()),
                         extra_caught=int(extra.sum()), extra_by_fire=extra_fire, extra_by_mi=extra_mi,
                         thr_auc=thr_auc, thr_fire=thr_fire, thr_mi=thr_mi, thr_attr=thr_attr,
                         corr_selauc_absattr=float(np.corrcoef(sel_auc, abs_attr)[0, 1]) if nA > 2 else float("nan")))
        # per-feature rows (all metrics) for cross-ref
        for j in range(nA):
            feat_rows.append(dict(layer=int(L), feature=int(active[j]), fire_rate=float(fire_rate[active[j]]),
                                  auc_global=float(auc_g[j]), sel_auc=float(sel_auc[j]),
                                  fire_rate_alpha=float(fr_a[j]), fire_rate_beta=float(fr_b[j]), fire_sel=float(fire_sel[j]),
                                  mi_label=float(mi_label[j]), mi_family=float(mi_family[j]),
                                  mean_act_active=float(mean_act_active[j]),
                                  attribution=float(attr_f[j]), abs_attr=float(abs_attr[j]),
                                  is_attr=int(is_attr[j]), relevant_cons=int(relevant_cons[j]), relevant_exp=int(relevant_exp[j])))
        logger.info("L%02d |A|=%4d | CONS Q1/Q2/Q3=%d/%d/%d | EXP Q1/Q2/Q3=%d/%d/%d | extra caught=%d (fire %d, mi %d)",
                    L, nA, c1, c2, c3, e1, e2, e3, int(extra.sum()), extra_fire, extra_mi)
        del Wdec, aA, Da, binned; torch.cuda.empty_cache()

    with open(out / "quadrants_multimetric.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    with open(out / "feature_metrics_full.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(feat_rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in feat_rows]

    print("\n" + "=" * 116)
    print("MULTI-METRIC FEATURE QUADRANTS — concept-relevance (5 faces, 2 defs) x attribution, all layers, null-calibrated")
    print("=" * 116)
    print("       |      | CONSERVATIVE (AUC only)     | EXPANDED (auc OR fire OR mi)  | extra features AUC missed")
    print("layer  | |A|  | Q1   Q2(dec¬use) Q3(use¬con)| Q1   Q2(dec¬use) Q3(use¬con) | total (via fire / via mi)")
    for r in rows:
        print(f"  L{r['layer']:02d}  | {r['n_active']:4d} | {r['cons_Q1']:3d}    {r['cons_Q2']:4d}      {r['cons_Q3']:4d}   "
              f"| {r['exp_Q1']:3d}    {r['exp_Q2']:4d}      {r['exp_Q3']:4d}   |   {r['extra_caught']:3d}  ({r['extra_by_fire']:3d} / {r['extra_by_mi']:3d})")
    print("\nReading: compare CONSERVATIVE vs EXPANDED. 'extra caught' = features that are concept-relevant by PRESENCE")
    print("(narrow subgroup, fire_sel) or by general dependence (mi_label) but invisible to AUC. If many, the AUC-only")
    print("picture undercounts concept features (your concern) -- Q2 (decoded-not-used) grows and Q3 (used-not-concept)")
    print("shrinks under the fairer definition. If few, the AUC picture was already robust. Per-feature metrics (incl.")
    print("mi_family surface diagnostic) in feature_metrics_full.csv. Thresholds null-calibrated (shuffled labels/prompts).")
    print("=" * 116 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/quad_multimetric")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[-1])
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--n_null_sel", type=int, default=20)
    p.add_argument("--n_null_attr", type=int, default=10)
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
