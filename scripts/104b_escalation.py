"""
104b_escalation.py   [why no w_res flip even pushing harder? escalate c in {4,8,16,32}]
========================================================================================================
The exp-89 calculus says Delta(margin) ~ c*sigma*||g||*cos(g, v_hat). For w_res, cos(g,w_res)~0.03 at every
layer, so the readout-relevant component of the push is ~0 no matter how large c is. Raising c only grows
the TOTAL-norm disruption, at which point a RANDOM direction also flips the forced answer (norm artifact,
cf exp 75/96). So the only honest question is: does w_res ever SEPARATE from the random null as c grows?

This escalates c in {4,8,16,32} along {w_res, u, random} on late layers (default L18-35), forced regime,
and per (layer, c, direction) reports:
  intact-flip (both arms), mean dmargin, cos(g,dir), calculus prediction c*sigma*||g||*cos(g,dir), AUC-B@35.
Then prints the GAP (w_res - random) and (u - random) at each c, flagging random>0.2 as the norm-artifact
zone. If w_res never beats random in the valid (random~0) window, there is no flip on w_res at ANY c, and
the reason is geometric (g _|_ w_res), not scale.

SELF-TEST (no torch):  python 104b_escalation.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("escal")


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


def self_test():
    rng = np.random.default_rng(0); d, n = 2560, 200
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))
    ru = rng.standard_normal(d); u = unit_raw(ru - (ru @ w) * w)
    H = 30.0 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w) * 5.0
    g = u * 4.0
    for c in (4, 8, 16, 32):
        pred_w = abs(float(np.dot(g, c * float(np.std(H @ w)) * w)))
        pred_u = abs(float(np.dot(g, c * float(np.std(H @ u)) * u)))
        assert pred_u > 10 * (pred_w + 1e-9), f"at c={c} prediction along u >> along w"
    print("[self_test] OK — calculus prediction along w stays ~0 at all c (cos(g,w)~0); along u grows.")


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
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})
    R35 = n_layers - 1

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

    cap_layers = sorted(set(layers) | {R35})
    H = {L: np.zeros((Pn, d), np.float32) for L in cap_layers}
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    margin_clean = np.zeros(Pn); clean_pred = np.zeros(Pn, int)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    logger.info("capturing residual(+grad) over %d prompts; layers=%s ...", Pn, layers)
    for i in range(Pn):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keep, hs = {}, []
        for L in cap_layers:
            def mk(L=L):
                def pre(m, a):
                    if L in layers:
                        a[0].retain_grad()
                    keep[L] = a[0]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.detach().argmax().item())
            margin_clean[i] = float(lo[b_id].detach() - lo[a_id].detach())
            (lo[b_id] - lo[a_id]).backward()
            for L in cap_layers:
                H[L][i] = keep[L].detach()[0, -1, :].float().cpu().numpy()
            for L in layers:
                g = keep[L].grad
                G[L][i] = g[0, -1, :].float().cpu().numpy() if g is not None else 0.0
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, Pn)

    intact_rate = float(np.mean(np.isin(clean_pred, [a_id, b_id])))
    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f acc=%.3f", intact_rate, acc)

    tr = is_train
    wres_R35 = fisher_axis(H[R35][tr].astype(np.float64), y[tr], args.shrink)
    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)
    a_idx, b_idx = a_idx[: args.n_eval], b_idx[: args.n_eval]
    eval_idx = np.r_[a_idx, b_idx]; eval_y = y[eval_idx]
    sign_of = {int(i): (+1.0 if y[i] == 0 else -1.0) for i in eval_idx}
    target_of = {int(i): (b_id if y[i] == 0 else a_id) for i in eval_idx}
    logger.info("escalation: %d+%d prompts; c=%s; readout=L%d", len(a_idx), len(b_idx), args.c_list, R35)

    def steer_capture(i, ell, vec, capR):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        def steer(m, a):
            a[0][:, -1, :] = a[0][:, -1, :] + v; return (a[0],) + tuple(a[1:])
        keep = {}
        hooks = [tap_module(ell).register_forward_pre_hook(steer, with_kwargs=False)]
        if capR is not None:
            def capt(m, a): keep["R"] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
            hooks.append(tap_module(capR).register_forward_pre_hook(capt, with_kwargs=False))
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
        finally:
            for h in hooks: h.remove()
        return int(lo.argmax().item()), float(lo[b_id] - lo[a_id]), keep.get("R")

    rows = []
    for L in layers:
        w_res = fisher_axis(H[L][tr].astype(np.float64), y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        rdir = unit_raw(rng.standard_normal(d))
        gmean_norm = float(np.mean([np.linalg.norm(G[L][i]) for i in eval_idx]))
        dirs = {"w_res": w_res, "u": u, "random": rdir}
        capR = R35 if L < R35 else None
        for c in args.c_list:
            res_c = {}
            for nm, vhat in dirs.items():
                sigma = float(np.std(H[L][tr].astype(np.float64) @ vhat))
                csig = c * sigma
                cos_gv = float(np.mean([abs(unit_raw(G[L][i].astype(np.float64)) @ vhat) for i in eval_idx]))
                pred = csig * gmean_norm * cos_gv
                dms, intact, proj = [], 0, {}
                for i in eval_idx:
                    vec = sign_of[int(i)] * csig * vhat
                    top, m, hR = steer_capture(int(i), L, vec, capR)
                    dms.append(m - margin_clean[i]); intact += int(top == target_of[int(i)])
                    if hR is not None:
                        proj[int(i)] = float(hR.astype(np.float64) @ wres_R35)
                aucB = float(auc([proj[int(i)] for i in eval_idx], eval_y)) if proj else float("nan")
                res_c[nm] = dict(intact=intact / len(eval_idx), dm=float(np.mean(dms)),
                                 cos=cos_gv, pred=pred, csig=csig, aucB35=aucB)
            gap_w = res_c["w_res"]["intact"] - res_c["random"]["intact"]
            gap_u = res_c["u"]["intact"] - res_c["random"]["intact"]
            artifact = res_c["random"]["intact"] > 0.2
            for nm in ("w_res", "u", "random"):
                r = res_c[nm]
                rows.append(dict(layer=int(L), c=float(c), direction=nm, csig=r["csig"], cos_g_dir=r["cos"],
                                 pred_dmargin=r["pred"], dmargin=r["dm"], intact=r["intact"], aucB35=r["aucB35"],
                                 gap_wres_minus_random=gap_w, gap_u_minus_random=gap_u, norm_artifact_zone=int(artifact)))
            logger.info("  L%02d c=%2.0f | w_res: intact=%.2f cs=%.0f cos=%.3f | u: intact=%.2f cs=%.0f | random: intact=%.2f "
                        "| GAP(w-r)=%+.2f GAP(u-r)=%+.2f %s",
                        L, c, res_c["w_res"]["intact"], res_c["w_res"]["csig"], res_c["w_res"]["cos"],
                        res_c["u"]["intact"], res_c["u"]["csig"], res_c["random"]["intact"],
                        gap_w, gap_u, "[ARTIFACT ZONE: random>0.2]" if artifact else "")

    with open(out / "escalation.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 104)
    print("STEERING ESCALATION — does w_res EVER separate from random as c grows?")
    print("=" * 104)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}\n")
    print("Per c: max over layers of GAP(w_res - random) in the VALID window (random<=0.2), and where u sits:")
    for c in args.c_list:
        cr = [r for r in rows if r["c"] == c and r["direction"] == "w_res"]
        valid = [r for r in cr if not r["norm_artifact_zone"]]
        best_gap_w = max((r["gap_wres_minus_random"] for r in valid), default=float("nan"))
        ur = [r for r in rows if r["c"] == c and r["direction"] == "u"]
        max_intact_u = max((r["intact"] for r in ur), default=float("nan"))
        n_artifact = sum(1 for r in cr if r["norm_artifact_zone"])
        print(f"  c={c:>2.0f}: max GAP(w_res-random) in valid window = {best_gap_w:+.2f}   |   max intact(u) = {max_intact_u:.2f}   "
              f"|   layers in artifact zone: {n_artifact}/{len(cr)}")
    print("\nReading: if max GAP(w_res-random) stays ~0 (or negative) at every c -> w_res NEVER beats random ->")
    print("no flip on w_res at any strength; cause is geometric (cos(g,w_res)~0.03), not scale. u flips (intact->1).")
    print("Rows where random>0.2 are norm-artifact: any w_res intact there is non-specific, not the concept.")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/escalation")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--c_list", type=float, nargs="*", default=[4.0, 8.0, 16.0, 32.0])
    p.add_argument("--layers", type=int, nargs="*", default=list(range(18, 36)))
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
