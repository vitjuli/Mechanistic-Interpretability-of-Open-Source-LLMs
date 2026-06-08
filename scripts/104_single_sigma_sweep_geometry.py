"""
104_single_sigma_sweep_geometry.py   [c=4 steering on every layer + clean geometry for plotting]
========================================================================================================
One pass, forced regime, c=4. For EACH of the 36 layers ell:
  steer h^(ell) <- h^(ell) +- c*sigma_ell*v_hat   (alpha->beta with +, beta->alpha with -),
  for v in {w_res, u, random}, full forward, and record:
    - answer (top-1), intact-flip (top-1 becomes the target alpha/beta)
    - margin (logit_beta-logit_alpha), dmargin vs clean
    - AUC-A : decodability ON the steered layer ell, AFTER steering (analytic: proj shifts by +-c*sigma).
              TAUTOLOGICAL (we move along the probe axis) -- saved because requested, flagged in output.
    - AUC-B : decodability at READOUT layers {24, 35}, AFTER steering ell, by capturing the readout
              residual during the perturbed forward and applying the clean-trained probe. Meaningful:
              does the perturbation PROPAGATE to the readout? (only readouts strictly after ell.)

Clean geometry per layer (no intervention), for plotting:
  signed cos(u,gamma), cos(w_res,gamma), cos(delta,gamma), cos(u,w_res), cos(delta,w_res), cos(u,delta)
  sigma_wres, sigma_u, ||mu_alpha-mu_beta||, clean AUC at ell/24/35
  2D projections onto TWO planes -- P1={w_res, gamma}, P2={u, w_res} -- of: every prompt (cloud, centred
  on mu_all), mu_alpha, mu_beta, and the unit direction vectors {gamma, u, w_res, delta}.

gamma = W_U[beta]-W_U[alpha] (readout contrast); w_res=Fisher(train); u=mean grad(train); delta=mu_beta-mu_alpha.

Outputs (data/.../sweep104/): geometry_by_layer.csv, clouds_2d.csv, vectors_2d.csv,
  steering_by_layer.csv (aggregated), steering_detail.csv (per prompt).

SELF-TEST (no torch):  python 104_single_sigma_sweep_geometry.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sweep104")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def auc(scores, yy):
    s = np.asarray(scores, float); y = np.asarray(yy, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def plane_basis(v1, v2):
    """orthonormal basis (e1,e2): e1 along v1, e2 = component of v2 orthogonal to e1."""
    e1 = unit_raw(v1); e2 = unit_raw(v2 - (v2 @ e1) * e1); return e1, e2


def self_test():
    rng = np.random.default_rng(0); d, n = 64, 200
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))
    H = 0.3 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w)
    assert auc(H @ w, y) > 0.9
    e1, e2 = plane_basis(w, rng.standard_normal(d))
    assert abs(e1 @ e2) < 1e-9 and abs(np.linalg.norm(e1) - 1) < 1e-9, "orthonormal plane"
    # analytic AUC-A collapse: shift alpha(+) beta(-) by c*sigma along w
    sig = float(np.std(H @ w)); c = 4.0
    p = H @ w; sign = np.where(y == 0, +1.0, -1.0); ps = p + sign * c * sig
    assert auc(ps, y) < auc(p, y), "steering toward wrong class lowers on-axis AUC (tautological)"
    print("[self_test] OK — auc, orthonormal plane, analytic on-axis collapse verified.")


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
    W_U = model.lm_head.weight.detach()
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    gamma = unit_raw((W_U[b_id] - W_U[a_id]).float().cpu().numpy().astype(np.float64))
    layers = list(range(n_layers))
    readouts = [r for r in args.readouts if 0 <= r < n_layers]

    prompts = [json.loads(l) for l in open(args.prompts)]
    Pn = len(prompts)
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

    # ---- capture clean residual + grad + margin/pred over all prompts ----
    H = {L: np.zeros((Pn, d), np.float32) for L in layers}
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    margin_clean = np.zeros(Pn); clean_pred = np.zeros(Pn, int)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    logger.info("capturing clean residual+grad over %d prompts, %d layers...", Pn, n_layers)
    for i in range(Pn):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keep, hs = {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.detach().argmax().item())
            margin_clean[i] = float(lo[b_id].detach() - lo[a_id].detach())
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, Pn)

    intact_rate = float(np.mean(np.isin(clean_pred, [a_id, b_id])))
    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f acc=%.3f", intact_rate, acc)

    tr = is_train
    # ---- per-layer clean directions + geometry ----
    dirs = {}
    geo_rows, cloud_rows, vec_rows = [], [], []
    wres_R = {R: fisher_axis(H[R][tr].astype(np.float64), y[tr], args.shrink) for R in readouts}
    for L in layers:
        Hl = H[L].astype(np.float64)
        w_res = fisher_axis(Hl[tr], y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        mu_all = Hl.mean(0); mu_a = Hl[y == 0].mean(0); mu_b = Hl[y == 1].mean(0)
        delta = mu_b - mu_a; delta_h = unit_raw(delta)
        rdir = unit_raw(rng.standard_normal(d))
        dirs[L] = {"w_res": w_res, "u": u, "random": rdir}
        sig = {nm: float(np.std(Hl[tr] @ v)) for nm, v in dirs[L].items()}
        # signed cosines
        def c_(a, b): return float(a @ b)
        geo = dict(layer=int(L),
                   cos_u_gamma=c_(u, gamma), cos_wres_gamma=c_(w_res, gamma), cos_delta_gamma=c_(delta_h, gamma),
                   cos_u_wres=c_(u, w_res), cos_delta_wres=c_(delta_h, w_res), cos_u_delta=c_(u, delta_h),
                   sigma_wres=sig["w_res"], sigma_u=sig["u"], sigma_random=sig["random"],
                   mu_gap=float(np.linalg.norm(delta)),
                   auc_clean_layer=float(auc(Hl[~tr] @ w_res, y[~tr])))
        # 2D planes
        planes = {"P1_wres_gamma": plane_basis(w_res, gamma), "P2_u_wres": plane_basis(u, w_res)}
        for pname, (e1, e2) in planes.items():
            Xc = (Hl - mu_all)
            px = Xc @ e1; py = Xc @ e2
            geo[f"{pname}_std_x"] = float(np.std(px)); geo[f"{pname}_std_y"] = float(np.std(py))
            for i in range(Pn):
                cloud_rows.append(dict(layer=int(L), plane=pname, prompt_idx=int(i),
                                       label=int(y[i]), x=float(px[i]), y=float(py[i])))
            for nm, vv in {"gamma": gamma, "u": u, "w_res": w_res, "delta": delta_h}.items():
                vec_rows.append(dict(layer=int(L), plane=pname, object=nm,
                                     x=float(vv @ e1), y=float(vv @ e2)))
            for nm, mm in {"mu_alpha": mu_a, "mu_beta": mu_b}.items():
                vec_rows.append(dict(layer=int(L), plane=pname, object=nm,
                                     x=float((mm - mu_all) @ e1), y=float((mm - mu_all) @ e2)))
        geo_rows.append(geo)
        logger.info("L%02d geom: cos(u,γ)=%+.3f cos(wres,γ)=%+.3f cos(δ,γ)=%+.3f | σ_wres=%.2f σ_u=%.2f μgap=%.1f",
                    L, geo["cos_u_gamma"], geo["cos_wres_gamma"], geo["cos_delta_gamma"],
                    geo["sigma_wres"], geo["sigma_u"], geo["mu_gap"])

    # ---- steering eval set ----
    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)
    a_idx, b_idx = a_idx[: args.n_eval], b_idx[: args.n_eval]
    eval_idx = np.r_[a_idx, b_idx]; eval_y = y[eval_idx]
    sign_of = {int(i): (+1.0 if y[i] == 0 else -1.0) for i in eval_idx}   # alpha->+, beta->-
    target_of = {int(i): (b_id if y[i] == 0 else a_id) for i in eval_idx}
    logger.info("steering: %d alpha->beta + %d beta->alpha; readouts=%s; c=%.1f", len(a_idx), len(b_idx), readouts, args.c)

    def steer_and_capture(i, ell, vec, caps):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        def steer(m, a):
            a[0][:, -1, :] = a[0][:, -1, :] + v; return (a[0],) + tuple(a[1:])
        keep = {}
        hooks = [tap_module(ell).register_forward_pre_hook(steer, with_kwargs=False)]
        for R in caps:
            def mk(R=R):
                def pre(m, a): keep[R] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            hooks.append(tap_module(R).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
        finally:
            for h in hooks: h.remove()
        return int(lo.argmax().item()), float(lo[b_id] - lo[a_id]), keep

    steer_rows, detail_rows = [], []
    for L in layers:
        caps = [R for R in readouts if R > L]                  # readouts strictly downstream
        for nm, vhat in dirs[L].items():
            sigma = float(np.std(H[L][tr].astype(np.float64) @ vhat))
            csig = args.c * sigma
            # collect
            steered_proj_A = {}                                # analytic on-axis (w_res only meaningful, but compute for all)
            readoutR_proj = {R: {} for R in caps}
            tops, dms, intact = {}, [], 0
            for i in eval_idx:
                vec = sign_of[int(i)] * csig * vhat
                top, m, keep = steer_and_capture(int(i), L, vec, caps)
                tops[int(i)] = top
                dms.append(m - margin_clean[i])
                intact += int(top == target_of[int(i)])
                # AUC-A analytic: clean proj on THIS dir + sign*csig
                steered_proj_A[int(i)] = float(H[L][i].astype(np.float64) @ vhat) + sign_of[int(i)] * csig
                for R in caps:
                    readoutR_proj[R][int(i)] = float(keep[R].astype(np.float64) @ wres_R[R])
                detail_rows.append(dict(layer=int(L), direction=nm, prompt_idx=int(i), label=int(y[i]),
                                        arm=("a2b" if y[i] == 0 else "b2a"), c=args.c, sigma=sigma, c_sigma=csig,
                                        margin_clean=float(margin_clean[i]), margin_steered=float(m),
                                        dmargin=float(m - margin_clean[i]), answer_id=int(top),
                                        intact=int(top == target_of[int(i)])))
            # AUCs over eval set (true labels)
            aucA = float(auc([steered_proj_A[int(i)] for i in eval_idx], eval_y))   # tautological for w_res
            aucB = {R: float(auc([readoutR_proj[R][int(i)] for i in eval_idx], eval_y)) for R in caps}
            row = dict(layer=int(L), direction=nm, c=args.c, sigma=sigma, c_sigma=csig,
                       intact_flip=intact / len(eval_idx), mean_dmargin=float(np.mean(dms)),
                       auc_A_steered=aucA,
                       auc_B24_steered=aucB.get(24, float("nan")), auc_B35_steered=aucB.get(35, float("nan")),
                       auc_clean_layer=next(g["auc_clean_layer"] for g in geo_rows if g["layer"] == L),
                       auc_clean_24=float(auc(H[24][~tr].astype(np.float64) @ wres_R[24], y[~tr])) if 24 in readouts else float("nan"),
                       auc_clean_35=float(auc(H[35][~tr].astype(np.float64) @ wres_R[35], y[~tr])) if 35 in readouts else float("nan"))
            steer_rows.append(row)
            logger.info("  L%02d %-7s c=4 cσ=%.1f | intact=%.2f Δm=%+.2f | AUC-A=%.2f AUC-B24=%s AUC-B35=%s",
                        L, nm, csig, row["intact_flip"], row["mean_dmargin"], aucA,
                        f"{aucB.get(24, float('nan')):.2f}" if 24 in caps else "--",
                        f"{aucB.get(35, float('nan')):.2f}" if 35 in caps else "--")

    # ---- write everything ----
    def dump(name, rows):
        with open(out / name, "w", newline="") as f:
            w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    dump("geometry_by_layer.csv", geo_rows)
    dump("clouds_2d.csv", cloud_rows)
    dump("vectors_2d.csv", vec_rows)
    dump("steering_by_layer.csv", steer_rows)
    dump("steering_detail.csv", detail_rows)

    print("\n" + "=" * 104)
    print("SINGLE-SIGMA (c=4) PER-LAYER STEERING + GEOMETRY")
    print("=" * 104)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}; readouts={readouts}\n")
    print("layer | dir     | intact | Δmargin | AUC-A(taut) | AUC-B@24 | AUC-B@35   (clean AUC@24=%.2f @35=%.2f)" %
          (auc(H[24][~tr].astype(np.float64) @ wres_R[24], y[~tr]) if 24 in readouts else float('nan'),
           auc(H[35][~tr].astype(np.float64) @ wres_R[35], y[~tr]) if 35 in readouts else float('nan')))
    for r in steer_rows:
        if r["direction"] == "random":
            continue
        print("  L%02d  | %-7s | %.2f   | %+6.2f  |   %.2f      |  %s   |  %s" %
              (r["layer"], r["direction"], r["intact_flip"], r["mean_dmargin"], r["auc_A_steered"],
               (f"{r['auc_B24_steered']:.2f}" if r["auc_B24_steered"] == r["auc_B24_steered"] else " -- "),
               (f"{r['auc_B35_steered']:.2f}" if r["auc_B35_steered"] == r["auc_B35_steered"] else " -- ")))
    print("\nReading: AUC-A is tautological (we shift along the probe axis). AUC-B is the real one: does the early")
    print("perturbation reach the readout? AUC-B near the clean value => perturbation washed out (readout re-decodes");
    print("the true label); AUC-B collapsing => perturbation propagated. Pair with intact-flip: w_res is expected to")
    print("leave both answer (intact~0) and readout decodability (AUC-B~clean) intact; u flips the answer late.")
    print("All values saved to geometry_by_layer.csv / clouds_2d.csv / vectors_2d.csv / steering_*.csv")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/sweep104")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--c", type=float, default=4.0)
    p.add_argument("--readouts", type=int, nargs="*", default=[24, 35])
    p.add_argument("--n_eval", type=int, default=24)
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
