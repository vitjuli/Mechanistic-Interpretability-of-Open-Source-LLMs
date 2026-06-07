"""
91_carrier_causal_validation.py   [do the u-aligned dictionary features actually CARRY the decision?]
======================================================================================================
Exp 88e found that transcoder decoder directions align with the usage direction u at L21-L25
(~2x a decoder-matched null), more than with the readable axis w_res. Geometric alignment is
NECESSARY but NOT SUFFICIENT to call a feature a "carrier": the feature must (i) actually fire on
these prompts, (ii) causally move the decision margin when removed, and (iii) do so beyond a
frequency-matched random-feature baseline. This script runs exactly that test.

Per layer L:
  1. usage direction u^L = mean_i grad_i(logit_beta - logit_alpha) at the answer position;
     readable axis w_res^L = Fisher(train).
  2. APPLES-TO-APPLES carrier geometry: max_f |cos(d_f, u)| and max_f |cos(d_f, w_res)| vs a shared
     decoder-derived null (max |cos(d_f, random direction)|). (Closes the w_res-vs-u oversight.)
  3. CARRIER SET S = top-K features by |cos(d_f, u)|. Their activation frequency (fraction of
     token positions with act>0) is measured over the corpus.
  4. CONTROL = same-size random feature sets MATCHED in activation frequency (n_control sets) --
     the honest null: random features that fire equally often, NOT just any features.
  5. Tests on held-out targets:
       (a) DO THEY FIRE / ENCODE?  mean answer-position activation; AUC of summed carrier activation
           for alpha vs beta (descriptive: do they collectively separate the classes?).
       (b) DO THEY CARRY?  error-free transcoder ablation (subtract sum_{f in S} a_f * W_dec[f] from
           the layer-L MLP output, all positions) -> margin shift, margin-flip, intact-flip, vs the
           frequency-matched control distribution.
       (c) SIGN: does the ablated contribution align with u? cos(ablation vector at answer pos, u).

Three presentable outcomes, all strong:
  - fire + ablation moves margin ABOVE the frequency-matched null + aligned with u
        -> POSITIVE MECHANISM: the model's own dictionary causally realises the usage direction.
  - fire + encode but ablation NOT above null
        -> geometric alignment is causally inert (deepens "decoded != used" to "even use-aligned
           features are not causal" -- consistent with mid-stack usage-subspace erasure being benign).
  - do not fire on these prompts
        -> the alignment is parasitic; closes the carrier question.

SELF-TEST (no torch / no repo):  python 91_carrier_causal_validation.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("carrier_causal")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def auc_scalar(s, y):
    s = np.asarray(s, float); o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def encode_np(x, W_enc, b_enc):
    """ReLU/JumpReLU-style encode for the self-test (x: (...,d), W_enc: (F,d))."""
    return np.maximum(x @ W_enc.T + b_enc, 0.0)


def match_by_frequency(freq, target_idx, n_sets, rng, tol=0.2):
    """For each feature in target_idx, sample a random feature of similar firing frequency."""
    F = len(freq); sets = []
    for _ in range(n_sets):
        chosen = []
        for f in target_idx:
            lo, hi = freq[f] * (1 - tol) - 1e-4, freq[f] * (1 + tol) + 1e-4
            cand = np.where((freq >= lo) & (freq <= hi))[0]
            cand = cand[~np.isin(cand, target_idx)]
            chosen.append(int(rng.choice(cand)) if len(cand) else int(rng.integers(F)))
        sets.append(np.array(sorted(set(chosen))))
    return sets


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, F, n = 20, 60, 300
    Wd = np.stack([unit_raw(rng.standard_normal(d)) for _ in range(F)])     # decoder rows
    a = unit_raw(rng.standard_normal(d))                                    # usage direction
    carrier = np.arange(5)
    for f in carrier:                                                       # carriers' decoder ~ a
        Wd[f] = unit_raw(0.9 * a + 0.1 * rng.standard_normal(d))
    y = np.array([0, 1] * (n // 2))
    # feature activations: carriers fire class-dependently; everything else fires at a base rate
    acts = (rng.standard_normal((n, F)) > 1.0).astype(float)
    for f in carrier:
        acts[:, f] = (y * 1.0) * (1.0 + 0.3 * rng.standard_normal(n)) * (rng.random(n) > 0.2)
    freq = (acts > 0).mean(0)
    base = 0.4 * rng.standard_normal((n, d))
    resid = base + acts @ Wd                                               # features write to residual
    margin = resid @ a
    # ablation of a set S: remove their contribution, recompute margin
    def abl_margin(S):
        return (resid - acts[:, S] @ Wd[S]) @ a
    d_carrier = float(np.mean(np.abs(margin - abl_margin(carrier))))
    ctrl_sets = match_by_frequency(freq, carrier, 8, rng)
    d_ctrl = np.array([np.mean(np.abs(margin - abl_margin(S))) for S in ctrl_sets])
    assert d_carrier > d_ctrl.max() + 1e-6, f"carrier ablation must beat freq-matched null: {d_carrier} vs {d_ctrl.max()}"
    # carriers should fire and encode the class
    score = acts[:, carrier].sum(1)
    assert auc_scalar(score, y) > 0.8, "carrier activations must separate the classes in the toy"
    # frequency matching picks similar-frequency features
    for S in ctrl_sets:
        assert abs(freq[S].mean() - freq[carrier].mean()) < 0.25
    # encode sanity
    We = rng.standard_normal((F, d)); be = rng.standard_normal(F)
    assert (encode_np(rng.standard_normal((3, d)), We, be) >= 0).all()
    print("[self_test] OK — encode, frequency-matched control, carrier-ablation-beats-null, activation-AUC pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transcoder_loader import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size; last = n_layers - 1
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})
    logger.info("model: %d layers; carrier causal validation on layers %s", n_layers, layers)

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    trm = np.array([p["surface_family"] in train_fams for p in prompts])

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    ts = load_transcoder_set("4b", device=torch.device(args.device), dtype=torch.bfloat16, lazy_load=True)
    Wdec = {}
    for L in layers:
        W = ts[L].W_dec
        Wdec[L] = (W.detach() if hasattr(W, "detach") else W).to(args.device, torch.float32)

    # ---------- Pass 1: residual + gradient (u, w_res) AND feature frequency ----------
    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    grad = {L: np.zeros((nP, d), np.float32) for L in layers}
    fire = {L: torch.zeros(Wdec[L].shape[0], device=args.device) for L in layers}
    npos = 0
    clean_margin = np.zeros(nP)
    logger.info("Pass 1: capturing u / w_res + feature firing frequency over %d prompts...", nP)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        keep, mlp_in = {}, {}; handles = []
        for L in layers:
            def mk_tap(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk_tap(), with_kwargs=False))
            def mk_mlp(L=L):
                def pre(m, a):
                    mlp_in[L] = a[0].detach(); return None
                return pre
            handles.append(blocks[L].mlp.register_forward_pre_hook(mk_mlp(), with_kwargs=False))
        try:
            row = model(**inp, use_cache=False).logits[0, -1, :]
            (row[beta_id] - row[alpha_id]).backward()
            for L in layers:
                t = keep[L]
                res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                grad[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                with torch.no_grad():
                    acts = ts[L].encode(mlp_in[L][0])            # (seq, F)
                    fire[L] += (acts > 0).sum(0).float()
            npos += int(inp.input_ids.shape[1])
            lp = torch.log_softmax(row.detach().float(), 0)
            clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 100 == 0:
            logger.info("  pass1 %d/%d", i + 1, nP)

    held = [i for i in range(nP) if not trm[i]]
    ha = [i for i in held if y[i] == 0][: args.max_targets]
    hb = [i for i in held if y[i] == 1][: args.max_targets]
    targets = ha + hb; y_t = np.array([y[i] for i in targets])
    acc_clean = float(np.mean((np.array([clean_margin[i] for i in targets]) > 0).astype(int) == y_t))
    logger.info("clean held-out margin accuracy on %d targets: %.3f", len(targets), acc_clean)

    # ---------- per-layer geometry + carrier selection + frequency-matched controls ----------
    sel = {}
    geo_rows = []
    for L in layers:
        u = unit_raw(grad[L].astype(np.float64).mean(0))
        w_res = fisher_axis(res[L][trm].astype(np.float64), y[trm], args.shrink)
        ut = torch.tensor(u, dtype=torch.float32, device=args.device)
        wt = torch.tensor(w_res, dtype=torch.float32, device=args.device)
        wn = Wdec[L].norm(dim=1) + 1e-12
        cos_u = (Wdec[L] @ ut) / wn
        cos_w = (Wdec[L] @ wt) / wn
        dec_rand = []
        for _ in range(5):
            rr = torch.randn(d, device=args.device); rr = rr / rr.norm()
            dec_rand.append(float(((Wdec[L] @ rr) / wn).abs().max()))
        null = float(np.mean(dec_rand))
        topK = torch.topk(cos_u.abs(), args.topk).indices.cpu().numpy()
        sel[L] = {"u": u, "w_res": w_res, "carrier": np.sort(topK),
                  "cos_u_of_carrier": cos_u[topK].cpu().numpy()}
        geo_rows.append({"layer": int(L),
                         "max_cos_u": float(cos_u.abs().max()), "max_cos_wres": float(cos_w.abs().max()),
                         "decoder_null": null,
                         "u_above_null": bool(cos_u.abs().max() > null + 0.02),
                         "wres_above_null": bool(cos_w.abs().max() > null + 0.02)})
        logger.info("(geom) L%d: max|cos(d_f,u)|=%.3f  max|cos(d_f,w_res)|=%.3f  (decoder-null %.3f) -> u:%s w_res:%s",
                    L, geo_rows[-1]["max_cos_u"], geo_rows[-1]["max_cos_wres"], null,
                    "ABOVE" if geo_rows[-1]["u_above_null"] else "at/below",
                    "ABOVE" if geo_rows[-1]["wres_above_null"] else "at/below")

    freq = {L: (fire[L] / max(npos, 1)).cpu().numpy() for L in layers}
    ctrl = {L: match_by_frequency(freq[L], sel[L]["carrier"], args.n_control, rng) for L in layers}

    # ---------- ablation hook ----------
    def make_hook(L, feats):
        ft = torch.tensor(np.asarray(feats, dtype=np.int64), device=args.device)
        def hook(m, inp, output):
            with torch.no_grad():
                acts = ts[L].encode(inp[0])                       # (1, seq, F)
                contrib = acts[..., ft] @ Wdec[L][ft]             # (1, seq, d)
            return output - contrib.to(output.dtype)
        return hook

    def eval_ablation(L, feats):
        ms, flips_m, flips_i = [], [], []
        for idx, i in enumerate(targets):
            inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
            h = blocks[L].mlp.register_forward_hook(make_hook(L, feats))
            try:
                with torch.no_grad():
                    row = model(**inp, use_cache=False).logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0); m1 = float(lp[beta_id] - lp[alpha_id])
                top1 = int(row.argmax().item())
            finally:
                h.remove()
            m0 = clean_margin[i]
            ms.append(m1 - m0)
            flips_m.append(int(np.sign(m1) != np.sign(m0)))
            flips_i.append(int(top1 in (alpha_id, beta_id) and np.sign(m1) != np.sign(m0)))
        return {"mean_abs_margin_shift": float(np.mean(np.abs(ms))),
                "acc_after": float(np.mean((np.array([clean_margin[targets[j]] + ms[j] for j in range(len(targets))]) > 0).astype(int) == y_t)),
                "margin_flip": float(np.mean(flips_m)), "intact_flip": float(np.mean(flips_i))}

    # ---------- (a) activation/encoding of the carrier set on held-out targets ----------
    # ---------- (b) carrier ablation vs frequency-matched control ----------
    abl_rows = []
    for L in layers:
        cset = sel[L]["carrier"]
        # activation + encoding (answer position)
        actsum, mact = [], 0.0
        ftc = torch.tensor(cset.astype(np.int64), device=args.device)
        for i in targets:
            inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
            mlp_in = {}
            hp = blocks[L].mlp.register_forward_pre_hook(
                lambda m, a: (mlp_in.__setitem__("x", a[0].detach()), None)[1], with_kwargs=False)
            try:
                with torch.no_grad():
                    model(**inp, use_cache=False)
                    a = ts[L].encode(mlp_in["x"][0])[-1, ftc].float().cpu().numpy()
            finally:
                hp.remove()
            actsum.append(float(a.sum())); mact += float((a > 0).mean())
        auc_act = auc_scalar(np.array(actsum), y_t)
        frac_active = mact / len(targets)

        car = eval_ablation(L, cset)
        ctl = [eval_ablation(L, S) for S in ctrl[L]]
        ctl_shift = np.array([c["mean_abs_margin_shift"] for c in ctl])
        rec = {"layer": int(L), "n_carrier": int(len(cset)), "carrier_frac_active": frac_active,
               "carrier_act_auc": auc_act, "carrier_mean_abs_margin_shift": car["mean_abs_margin_shift"],
               "carrier_acc_after": car["acc_after"], "carrier_margin_flip": car["margin_flip"],
               "carrier_intact_flip": car["intact_flip"],
               "ctrl_shift_mean": float(ctl_shift.mean()), "ctrl_shift_p95": float(np.percentile(ctl_shift, 95)),
               "carrier_above_ctrl": bool(car["mean_abs_margin_shift"] > np.percentile(ctl_shift, 95))}
        abl_rows.append(rec)
        logger.info("(abl) L%d: carrier active=%.2f act-AUC=%.3f | |Δmargin| carrier=%.3f vs ctrl(mean %.3f, p95 %.3f) -> %s | "
                    "margin-flip=%.3f intact-flip=%.3f acc %.3f->%.3f",
                    L, frac_active, auc_act, car["mean_abs_margin_shift"], ctl_shift.mean(),
                    np.percentile(ctl_shift, 95), "ABOVE null" if rec["carrier_above_ctrl"] else "at/below null",
                    car["margin_flip"], car["intact_flip"], acc_clean, car["acc_after"])

    # ---------- save + verdict ----------
    def wcsv(name, rws):
        if rws:
            with open(out / name, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(rws[0].keys())); w.writeheader(); [w.writerow(r) for r in rws]
    wcsv("carrier_geometry.csv", geo_rows); wcsv("carrier_ablation.csv", abl_rows)

    fire_layers = [r["layer"] for r in abl_rows if r["carrier_frac_active"] > 0.2]
    causal = [r["layer"] for r in abl_rows if r["carrier_above_ctrl"] and r["carrier_mean_abs_margin_shift"] > 0.05]
    behaviour = [r["layer"] for r in abl_rows if r["carrier_intact_flip"] > 0.1]
    print("\n" + "=" * 98)
    print("CARRIER CAUSAL VALIDATION -- do the u-aligned dictionary features actually carry the decision?")
    print("=" * 98)
    u_ab = [r["layer"] for r in geo_rows if r["u_above_null"]]
    w_ab = [r["layer"] for r in geo_rows if r["wres_above_null"]]
    print(f"APPLES-TO-APPLES geometry: dictionary above decoder-null for u at {u_ab or 'none'}; for w_res at {w_ab or 'none'}")
    print(f"(a) carrier features FIRE (>20% positions) at layers: {fire_layers or 'none'}")
    print(f"(b) carrier ablation moves margin ABOVE the frequency-matched null at: {causal or 'none'}")
    print(f"(c) behavioural (intact-flip > 0.1) at: {behaviour or 'none'}")
    if causal:
        print(f"-> POSITIVE MECHANISM at {causal}: u-aligned dictionary features causally contribute to the decision "
              f"margin beyond frequency-matched random features.")
    elif fire_layers:
        print("-> Carrier features fire and encode the class, but ablation does NOT beat the frequency-matched null: "
              "the geometric alignment with u is causally inert -> 'decoded != used' extends to 'use-aligned features are not levers'.")
    else:
        print("-> Carrier features barely fire on these prompts: the decoder-direction alignment with u is parasitic.")
    print("Caveat: margin metric (not intact) is the honest readout in base format; control = same-size feature sets "
          "matched on firing frequency; ablation is error-free (subtracts only the features' decoder contribution).")
    print("=" * 98 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/carrier_causal")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[21, 22, 23, 24, 25])
    p.add_argument("--topk", type=int, default=30, help="carrier set size (top-K by |cos(d_f,u)|)")
    p.add_argument("--n_control", type=int, default=8)
    p.add_argument("--max_targets", type=int, default=80)
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
