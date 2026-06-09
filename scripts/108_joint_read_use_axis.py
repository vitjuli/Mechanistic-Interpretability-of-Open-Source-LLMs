"""
108_joint_read_use_axis.py   [is there ANY direction that is BOTH readable AND used?]
========================================================================================================
We showed w_res (readable, AUC~0.99) is ~orthogonal to u (used). But maybe a DIFFERENT direction is both
decodable and aligned with the output's used direction. This searches for one. For a trade-off weight
lambda in [0,1], find the unit direction v maximizing
    J(v) = (1-lambda) * decodability(v)  +  lambda * |cos(v, u)|
and trace the Pareto frontier (decodability vs alignment-with-u) as lambda sweeps 0->1. decodability(v) is
measured as the held-out AUC of the projection (and as a Fisher-ratio surrogate for the optimisation).
If the frontier shows the two objectives TRADE OFF hard -- every high-AUC direction has |cos(v,u)|~0 and
every high-cos(v,u) direction has AUC~0.5 -- then on this task being readable and being used are
structurally incompatible (a strong form of decoded != used). Anchors: AUC(u) itself, AUC(w_res), cos(w_res,u).

Optimisation: in the captured residual space, the decodability-optimal direction is the Fisher axis; the
u-optimal direction is u. We parametrise v on the geodesic / weighted blend and also do a few gradient
steps on the smooth surrogate. Light (residuals+grad only).

SELF-TEST (no torch):  python 108_joint_read_use_axis.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("joint_axis")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def auc(scores, yy):
    s = np.asarray(scores, float); y = np.asarray(yy, int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0)), Sw, (mu1 - mu0)


def fisher_ratio(v, mu_diff, Sw):
    num = float(v @ mu_diff) ** 2; den = float(v @ (Sw @ v)) + 1e-12
    return num / den


def self_test():
    rng = np.random.default_rng(0); d, n = 64, 300
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d)); ru = rng.standard_normal(d); u = unit_raw(ru - (ru @ w) * w)  # u _|_ w
    H = 0.3 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w)        # class signal ONLY along w
    # blend v(lambda)=unit((1-l)w + l u); AUC should fall, |cos(v,u)| should rise -> hard trade-off
    aucs, coss = [], []
    for l in (0.0, 0.5, 1.0):
        v = unit_raw((1 - l) * w + l * u); aucs.append(auc(H @ v, y)); coss.append(abs(v @ u))
    assert aucs[0] > 0.9 and aucs[-1] < 0.65 and coss[-1] > coss[0], "trade-off: AUC down as cos(v,u) up"
    print("[self_test] OK — Pareto blend shows AUC vs cos(v,u) trade-off when u _|_ class signal.")


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

    lambdas = np.linspace(0, 1, args.n_lambda)
    rows, frontier = [], {}
    layers = sorted({L for L in args.layers if 0 <= L < n_layers}) if args.layers != [-1] else list(range(n_layers))
    for L in layers:
        Htr = H[L][tr].astype(np.float64); Hte = H[L][~tr].astype(np.float64)
        w, Sw, mu_diff = fisher_axis(Htr, y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        auc_w = auc(Hte @ w, y[~tr]); auc_u = auc(Hte @ u, y[~tr]); cos_wu = float(abs(w @ u))
        for lam in lambdas:
            # blend on the sphere then a few gradient steps on the smooth surrogate
            v = unit_raw((1 - lam) * w + lam * u)
            for _ in range(args.gd_steps):
                # grad of (1-lam)*fisher_ratio + lam*|cos(v,u)| wrt v, projected to sphere
                fr_num = float(v @ mu_diff); Sv = Sw @ v; fr_den = float(v @ Sv) + 1e-12
                g_fisher = (2 * fr_num * mu_diff) / fr_den - (fr_num ** 2) * (2 * Sv) / (fr_den ** 2)
                g_cos = np.sign(v @ u) * u
                g = (1 - lam) * g_fisher / (abs(fisher_ratio(w, mu_diff, Sw)) + 1e-9) + lam * g_cos
                g = g - (g @ v) * v                       # tangent
                v = unit_raw(v + args.gd_lr * g)
            a_v = auc(Hte @ v, y[~tr]); c_v = float(abs(v @ u))
            rows.append(dict(layer=int(L), lam=float(lam), auc=float(a_v), cos_v_u=c_v,
                             auc_wres=float(auc_w), auc_u=float(auc_u), cos_wres_u=cos_wu))
        # frontier summary: best AUC achievable at high alignment, and vice versa
        Lrows = [r for r in rows if r["layer"] == L]
        hi_align = [r for r in Lrows if r["cos_v_u"] >= 0.5]
        best_auc_at_align = max((r["auc"] for r in hi_align), default=float("nan"))
        hi_auc = [r for r in Lrows if r["auc"] >= 0.9]
        best_align_at_auc = max((r["cos_v_u"] for r in hi_auc), default=float("nan"))
        frontier[L] = (best_auc_at_align, best_align_at_auc, auc_w, auc_u, cos_wu)
        logger.info("L%02d | AUC(w)=%.2f AUC(u)=%.2f cos(w,u)=%.3f | best AUC@cos>=.5: %.2f | best cos@AUC>=.9: %.2f",
                    L, auc_w, auc_u, cos_wu, best_auc_at_align, best_align_at_auc)

    with open(out / "joint_read_use.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("JOINT READ-AND-USE AXIS — can any direction be both decodable (high AUC) AND aligned with u?")
    print("=" * 100)
    print("layer | AUC(w_res) | AUC(u) | cos(w_res,u) | best AUC at cos(v,u)>=0.5 | best cos(v,u) at AUC>=0.9")
    for L in layers:
        ba, bc, aw, au, cw = frontier[L]
        print(f"  L{L:02d}  |    {aw:.2f}    |  {au:.2f}  |    {cw:.3f}     |          {ba:.2f}            |        {bc:.2f}")
    print("\nReading: if 'best AUC at cos>=0.5' stays LOW (you must give up decodability to align with u) AND")
    print("'best cos at AUC>=0.9' stays LOW (decodable directions are all ~orthogonal to u), then readability and")
    print("usage are structurally incompatible on this task -- no single axis is both read and used. Where AUC(u)")
    print("rises late, the two objectives start to become compatible (u becomes ~the readout). Saved joint_read_use.csv")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/joint_axis")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[8, 13, 17, 21, 24, 28, 32, 35])
    p.add_argument("--n_lambda", type=int, default=11)
    p.add_argument("--gd_steps", type=int, default=40)
    p.add_argument("--gd_lr", type=float, default=0.05)
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
