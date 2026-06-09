"""
107_alpha_beta_asymmetry.py   [is the behavioural alpha-default visible in the per-class geometry?]
========================================================================================================
Behaviour is asymmetric: alpha-recall 0.996 vs beta-recall 0.446, all errors are beta->alpha (alpha-default).
So far we worked with the CONTRAST (beta-alpha). Here we look at each class SEPARATELY, to see whether the
asymmetry has a geometric signature. Per layer, along the used direction u and the readable axis w_res:
  - per-class mean / std of the projection (where each cloud sits, how spread)
  - class overlap (how separable) and which class sits closer to the decision threshold along u
  - the fraction of each class on the 'beta side' of the u-threshold (signed margin proxy)
  - distance of each class centroid to the midpoint (is beta squeezed toward alpha?)
If beta prompts sit closer to / on the alpha side of the u-threshold while alpha sits firmly on its side,
that is the geometric face of the alpha-default. Light (residuals+grad only).

SELF-TEST (no torch):  python 107_alpha_beta_asymmetry.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ab_asym")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def overlap_coef(x0, x1):
    """fraction misclassified by the optimal 1-D threshold midway -> rough overlap (0=separated,.5=identical)."""
    m0, m1 = x0.mean(), x1.mean(); thr = 0.5 * (m0 + m1)
    if m1 >= m0:
        err = (np.mean(x0 > thr) + np.mean(x1 <= thr)) / 2
    else:
        err = (np.mean(x0 <= thr) + np.mean(x1 > thr)) / 2
    return float(err)


def self_test():
    rng = np.random.default_rng(0)
    a = rng.normal(-1.0, 1.0, 400); b = rng.normal(+0.3, 1.5, 400)   # beta squeezed toward alpha + wider
    thr = 0.5 * (a.mean() + b.mean())
    frac_b_on_alpha = float(np.mean(b < thr))
    assert frac_b_on_alpha > 0.3, "asymmetry: a chunk of beta sits on the alpha side"
    assert overlap_coef(a, b) > 0.1, "non-trivial overlap"
    print("[self_test] OK — per-class overlap + side-fraction computed.")


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

    H = {L: np.zeros((Pn, d), np.float32) for L in range(n_layers)}
    G = {L: np.zeros((Pn, d), np.float32) for L in range(n_layers)}
    clean_pred = np.zeros(Pn, int)
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing residual+grad over %d prompts...", Pn)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, hs = {}, []
        for L in range(n_layers):
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.detach().argmax().item())
            (lo[b_id] - lo[a_id]).backward()
            for L in range(n_layers):
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)

    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean acc=%.3f (n_alpha=%d n_beta=%d)", acc, int((y == 0).sum()), int((y == 1).sum()))

    rows = []
    for L in range(n_layers):
        Hl = H[L].astype(np.float64)
        w = fisher_axis(Hl[tr], y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        for name, axis in (("u", u), ("w_res", w)):
            pa = Hl[y == 0] @ axis; pb = Hl[y == 1] @ axis
            thr = 0.5 * (pa.mean() + pb.mean())
            beta_side = +1 if pb.mean() >= pa.mean() else -1     # which way is 'beta'
            # fraction of each class on the BETA side of the threshold
            fa_beta = float(np.mean((pa - thr) * beta_side > 0))
            fb_beta = float(np.mean((pb - thr) * beta_side > 0))
            rows.append(dict(layer=int(L), axis=name,
                             mean_alpha=float(pa.mean()), mean_beta=float(pb.mean()),
                             std_alpha=float(pa.std()), std_beta=float(pb.std()),
                             overlap=overlap_coef(pa, pb),
                             frac_alpha_on_beta_side=fa_beta, frac_beta_on_beta_side=fb_beta,
                             sep_in_std=float(abs(pb.mean() - pa.mean()) / (0.5 * (pa.std() + pb.std()) + 1e-9))))
        ru = next(r for r in rows if r["layer"] == L and r["axis"] == "u")
        logger.info("L%02d [u] α@%.2f(σ%.2f) β@%.2f(σ%.2f) | β-on-βside=%.2f α-on-βside=%.2f sep=%.2fσ overlap=%.2f",
                    L, ru["mean_alpha"], ru["std_alpha"], ru["mean_beta"], ru["std_beta"],
                    ru["frac_beta_on_beta_side"], ru["frac_alpha_on_beta_side"], ru["sep_in_std"], ru["overlap"])

    with open(out / "alpha_beta_asymmetry.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("ALPHA/BETA ASYMMETRY along the USED direction u (does beta sit closer to the alpha side?)")
    print("=" * 100)
    print("layer | frac BETA on beta-side | frac ALPHA on beta-side | separation(σ) | overlap | std_α | std_β")
    for r in [r for r in rows if r["axis"] == "u"]:
        print(f"  L{r['layer']:02d}  |        {r['frac_beta_on_beta_side']:.2f}          |        "
              f"{r['frac_alpha_on_beta_side']:.2f}           |     {r['sep_in_std']:.2f}      |  {r['overlap']:.2f}  "
              f"|  {r['std_alpha']:.2f} | {r['std_beta']:.2f}")
    print("\nReading: if 'frac BETA on beta-side' is LOW (beta leaking onto the alpha side) while 'frac ALPHA on")
    print("beta-side' is ~0 (alpha firmly on its own side), that is the geometric alpha-default. Compare std_alpha")
    print("vs std_beta (is one class more diffuse) and overlap across depth. Saved alpha_beta_asymmetry.csv")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/ab_asym")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
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
