"""
154_dump_realized_writes.py   [CSD3, v3]  raw W_dec + per-class acts + TRUE MLP output + pooled cov
==================================================================================================
Wired to the 110/119 machinery. Captures per prompt at the answer position: residual (value + grad),
MLP INPUT (pre-hook, for transcoder encode) and MLP OUTPUT (forward-hook, the TRUE MLP contribution),
then saves per layer (contiguous range so the analyzer can take consecutive δ differences):

  res_alpha_L / res_beta_L     residual class means      -> delta, (with Sw) genuine Fisher w_res
  grad_alpha_L / grad_beta_L   residual-gradient class means -> u (mean grad)
  mlp_true_alpha_L / mlp_true_beta_L   TRUE MLP-output class means -> exact MLP write into delta
  Sw_L          (d,d)          POOLED WITHIN-CLASS COVARIANCE -> w_res = solve((1-s)Sw+s·tr/d·I, μβ-μα)
  W_dec_active_L, act_alpha_L, act_beta_L, feat_ids_L   transcoder decoders + per-class acts

Decomposition (in analyzer 155): for each axis Δ = axis_L − axis_{L-1}, split into rotation/length,
and EACH into three exact sources: MLP_true (real MLP output diff), attention = Δ − MLP_true (exact
remainder, attention + LN/other), and transcoder error = MLP_true − Σw_f (recon error, explicit).
Invariant MLP+attn(+recon split) reconstructs Δ exactly.

THE NEW HOOK vs the previous dump: register_forward_hook on blocks[L].mlp for the MLP OUTPUT.

USAGE (CSD3, per concept — mirror j110 answers):
  python scripts/154_dump_realized_writes.py --prompts data/prompts/B1_alpha_beta.jsonl \
    --layers 12 13 ... 35 --out_dir data/analysis/runD_v2/B1_alpha_beta/realized_writes \
    --pos_label beta --alpha_answer " alpha" --beta_answer " beta" --family_key surface_family --min_prompts 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dump154")


def pooled_within_cov(res, y):
    """pooled within-class covariance Sw (d,d) from per-sample residual res (n,d) and labels y."""
    d = res.shape[1]; S = np.zeros((d, d), dtype=np.float64); dof = 0
    for c in (0, 1):
        Xc = res[y == c]
        if len(Xc) < 2:
            continue
        Xc = Xc - Xc.mean(0); S += Xc.T @ Xc; dof += len(Xc) - 1
    return S / max(dof, 1)


def run(args):
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
    y = np.array([1 if p["correct_answer"].strip() == args.pos_label else 0 for p in prompts])
    logger.info("prompts: %d (class0=%d, class1=%d) | layers %s", Pn, int((y == 0).sum()), int((y == 1).sum()), layers)

    by = {0: {}, 1: {}}
    for i, p in enumerate(prompts):
        by[int(y[i])].setdefault(p.get(args.family_key, "NA"), []).append(i)
    def exemplar(cls, avoid):
        fl = [f for f in by[cls] if f != avoid] or list(by[cls]); f = fl[rng.integers(len(fl))]
        j = by[cls][f][rng.integers(len(by[cls][f]))]
        return f"{prompts[j]['prompt']}{args.answer_suffix}{args.beta_answer if cls==1 else args.alpha_answer}"
    def forced(i):
        if args.use_prompt_asis:
            return prompts[i]["prompt"] + args.answer_suffix
        fam = prompts[i].get(args.family_key, "NA"); ea, eb = exemplar(0, fam), exemplar(1, fam)
        s = [ea, eb] if i % 2 == 0 else [eb, ea]
        return s[0] + "\n\n" + s[1] + "\n\n" + prompts[i]["prompt"] + args.answer_suffix
    def tap(L): return blocks[L + 1] if L < last else bm.norm     # residual at answer pos

    # ---- capture residual(value+grad), mlp_input, mlp_output at answer pos ----
    H = {L: np.zeros((Pn, d), np.float32) for L in layers}      # residual value
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}      # residual gradient (for u)
    MIN = {L: np.zeros((Pn, d), np.float32) for L in layers}    # mlp input (encode)
    MOUT = {L: np.zeros((Pn, d), np.float32) for L in layers}   # mlp output (true write)
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing residual+grad + mlp_in + mlp_out over %d prompts ...", Pn)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device)
        keep, kmin, kmout, hs = {}, {}, {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            def mkm(L=L):
                def pre(m, a): kmin[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            def mko(L=L):
                def post(m, a, o):
                    t = o[0] if isinstance(o, (tuple, list)) else o
                    kmout[L] = t.detach()[0, -1, :].float().cpu().numpy(); return None
                return post
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            hs.append(blocks[L].mlp.register_forward_pre_hook(mkm(), with_kwargs=False))
            hs.append(blocks[L].mlp.register_forward_hook(mko(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                MIN[L][i] = kmin[L]; MOUT[L][i] = kmout[L]
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)
    for p_ in model.parameters(): p_.requires_grad_(False)

    # ---- transcoders: encode, save W_dec + per-class acts; save class means + Sw ----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcoder"))
    from transcoder import load_transcoder_set
    ts = load_transcoder_set(args.transcoder_set, device=args.device, dtype=torch.bfloat16, lazy_load=True)

    for L in layers:
        for nm, arr in [("res", H[L]), ("grad", G[L]), ("mlp_true", MOUT[L])]:
            a64 = arr.astype(np.float64)
            np.save(out / f"{nm}_alpha_L{L:02d}.npy", a64[y == 0].mean(0).astype(np.float32))
            np.save(out / f"{nm}_beta_L{L:02d}.npy", a64[y == 1].mean(0).astype(np.float32))
        Sw = pooled_within_cov(H[L].astype(np.float64), y)
        np.save(out / f"Sw_L{L:02d}.npy", Sw.astype(np.float32))

        mlp_in = torch.tensor(MIN[L], dtype=torch.bfloat16, device=args.device)
        with torch.no_grad():
            acts = ts[L].encode(mlp_in).float().cpu().numpy()
        active = np.where((acts > 0).sum(0) >= args.min_prompts)[0]
        if len(active) == 0:
            logger.warning("L%02d: no active features >= %d", L, args.min_prompts); continue
        Wdec = ts[L].W_dec.detach().float().cpu().numpy()[active].astype(np.float32)
        act_alpha = acts[y == 0][:, active].mean(0).astype(np.float32)
        act_beta = acts[y == 1][:, active].mean(0).astype(np.float32)
        np.save(out / f"W_dec_active_L{L:02d}.npy", Wdec)
        np.save(out / f"act_alpha_L{L:02d}.npy", act_alpha)
        np.save(out / f"act_beta_L{L:02d}.npy", act_beta)
        np.save(out / f"feat_ids_L{L:02d}.npy", active.astype(np.int64))

        mlp_true_diff = MOUT[L][y == 1].astype(np.float64).mean(0) - MOUT[L][y == 0].astype(np.float64).mean(0)
        recon = ((act_beta - act_alpha).astype(np.float64)[:, None] * Wdec.astype(np.float64)).sum(0)
        tc_err = float(np.linalg.norm(mlp_true_diff - recon) / (np.linalg.norm(mlp_true_diff) + 1e-12))
        logger.info("L%02d: active=%d | tc rel-error=%.3f | tr(Sw)=%.1f", L, len(active), tc_err, float(np.trace(Sw)))

    json.dump({"layers": layers, "min_prompts": args.min_prompts, "n_prompts": int(Pn), "shrink": args.shrink},
              open(out / "dump_meta.json", "w"), indent=2)
    logger.info("done -> %s", out)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_prompts", type=int, default=10)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--pos_label", default="beta")
    p.add_argument("--answer_suffix", default="\nAnswer (alpha or beta):")
    p.add_argument("--use_prompt_asis", action="store_true")
    p.add_argument("--family_key", default="surface_family")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
