r"""
106_writing_decomposition.py   [how much of what the model WRITES lands in the used vs the quiet channel?]
========================================================================================================
Three directions per layer: w_res (how to READ alpha/beta, Fisher), delta=mu_beta-mu_alpha (what the model
WRITES -- the separation it actually creates between the class means), u=grad(margin) (what the output USES).
We know pairwise they are near-orthogonal early and partly align late. This script formalises the thesis
quantitatively by DECOMPOSING the writing-direction delta (and w_res) into:
    delta = (delta . u_hat) u_hat   +   delta_perp
            \_____used part_____/      \_quiet part_/
and reporting, per layer, the fraction of delta's energy in the used channel vs the orthogonal complement
of u (the readable-but-unused 'quiet' channel). If most of delta is in delta_perp, the model writes the
alpha/beta separation into a channel it does NOT use for the answer -- decoded/written != used, made
numerical. Also decomposes delta against w_res, and w_res against u, for a full triad picture.

Uses only captured clean residuals + gradients (forced format). Light (no transcoders).
SELF-TEST (no torch):  python 106_writing_decomposition.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("writing_decomp")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def energy_along(vec, uhat):
    """fraction of vec's squared norm along unit uhat, and the orthogonal remainder fraction."""
    v = np.asarray(vec, float); n2 = float(v @ v) + 1e-30
    par = float(v @ uhat) ** 2
    return par / n2, 1.0 - par / n2


def self_test():
    rng = np.random.default_rng(0); d = 64
    u = unit_raw(rng.standard_normal(d))
    # delta mostly orthogonal to u (quiet writing): 0.2 along u, 1.0 orthogonal
    rp = rng.standard_normal(d); perp = unit_raw(rp - (rp @ u) * u)
    delta = 0.2 * u + 1.0 * perp
    f_used, f_quiet = energy_along(delta, u)
    assert f_quiet > 0.9 and abs(f_used + f_quiet - 1.0) < 1e-9, "mostly quiet writing detected"
    print("[self_test] OK — energy decomposition recovers a mostly-quiet writing direction.")


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
    W_U = model.lm_head.weight.detach()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    gamma = unit_raw((W_U[b_id] - W_U[a_id]).float().cpu().numpy().astype(np.float64))

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
            (lo[b_id] - lo[a_id]).backward()
            for L in range(n_layers):
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)

    rows = []
    for L in range(n_layers):
        Hl = H[L].astype(np.float64)
        w = fisher_axis(Hl[tr], y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        delta = Hl[y == 1].mean(0) - Hl[y == 0].mean(0)        # writing direction (raw, not unit)
        dnorm = float(np.linalg.norm(delta))
        # decompose delta along u and along w_res
        d_used_u, d_quiet_u = energy_along(delta, u)
        d_along_w, d_off_w = energy_along(delta, w)
        # decompose w_res along u (how much of the readable axis is in the used channel)
        w_used_u, w_quiet_u = energy_along(w, u)
        rows.append(dict(layer=int(L), delta_norm=dnorm,
                         delta_frac_in_used=d_used_u, delta_frac_in_quiet=d_quiet_u,
                         delta_frac_along_wres=d_along_w,
                         wres_frac_in_used=w_used_u, wres_frac_in_quiet=w_quiet_u,
                         cos_delta_u=float(unit_raw(delta) @ u), cos_delta_wres=float(unit_raw(delta) @ w),
                         cos_w_u=float(w @ u), cos_u_gamma=float(u @ gamma), cos_w_gamma=float(w @ gamma)))
        logger.info("L%02d | δ→used=%.3f δ→quiet=%.3f | δ‖wres=%.3f | wres→used=%.3f | cos(δ,u)=%+.3f cos(w,u)=%+.3f",
                    L, d_used_u, d_quiet_u, d_along_w, w_used_u, float(unit_raw(delta) @ u), float(w @ u))

    with open(out / "writing_decomposition.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("WRITING DECOMPOSITION — what fraction of delta(written) lands in the USED vs QUIET channel?")
    print("=" * 100)
    print("layer | delta-in-USED | delta-in-QUIET | delta‖w_res | w_res-in-USED")
    for r in rows:
        print(f"  L{r['layer']:02d}  |    {r['delta_frac_in_used']:.3f}     |     {r['delta_frac_in_quiet']:.3f}     "
              f"|    {r['delta_frac_along_wres']:.3f}   |    {r['wres_frac_in_used']:.3f}")
    early = [r for r in rows if r["layer"] < 18]; late = [r for r in rows if r["layer"] >= 24]
    print(f"\nmean delta-in-USED: early(L<18)={np.mean([r['delta_frac_in_used'] for r in early]):.3f}  "
          f"late(L>=24)={np.mean([r['delta_frac_in_used'] for r in late]):.3f}")
    print("Reading: delta-in-QUIET near 1.0 => the model writes the alpha/beta separation into a channel it does")
    print("NOT use for the answer; delta-in-USED rising late = the written separation only enters the used channel")
    print("near the readout. w_res-in-USED ~0 confirms the readable axis is almost entirely in the quiet channel.")
    print("Saved writing_decomposition.csv"); print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/writing_decomp")
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
