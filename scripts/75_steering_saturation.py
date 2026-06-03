"""
75_steering_saturation.py   [CSD3 / GPU, mirrors 65's machinery]
===================================================================
Follow-up to 65: "what if we push HARDER -- is there a magnitude where the answer
flips, and does it ever exist?" 65 went to c=2 sigma and saw no flip. This pushes
to large c (default up to 32 sigma) and answers the question HONESTLY by measuring
two things at every magnitude, because flip-rate alone is misleading:

  (1) SPECIFICITY: does w_res flip MORE than shuffled-label / random directions of
      the SAME magnitude? If w_res and random rise together, any flips are "a big
      push perturbs the model", NOT "the concept axis is causal".

  (2) ANSWER INTACTNESS: after steering, is the top output token still a real answer
      (alpha or beta), or has the model broken into garbage / collapsed to one token?
      A flip that happens only AFTER the answer breaks is not concept-steering -- it
      is the model falling apart and landing on a different token by accident.

The decisive quantity is the INTACT FLIP RATE: flips that occur WHILE the answer is
still alpha/beta, AND exceed the shuffled band. If that is ~0 at every magnitude,
then "no magnitude steers the concept" is confirmed with the full curve in hand.

Also reports the relative push size ||step|| / ||h_clean|| so "how hard are we
really pushing" is explicit: when this approaches/exceeds 1 we are adding a vector
as large as the entire state -- definitionally the breakage regime.

THREE OUTCOMES (decided on the curve):
  A) w_res intact-flip stays ~0 and tracks shuffled at all c -> NOT causal (strongest);
     flips, if any, coincide with answer breakage and are non-specific.
  B) w_res intact-flip RISES ABOVE shuffled at some c while answers remain intact
     -> causal at high amplitude after all. Would change the conclusion. (Unlikely
        given 65's logit-lens garbage, but this is the test that could overturn it.)
  C) nothing flips even at extreme c -> even stronger negative.

I/O mirrors 65 exactly (concept_npz with gbar/Sigma_inv; prompts jsonl; same hooks).
SELF-TEST: python 75_steering_saturation.py --self_test
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("steer_sat")


def whitener(Si):
    w, V = np.linalg.eigh(Si); return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v


def fisher_axis(H, y, shrink=0.1):
    m0, m1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - m0, H[y == 1] - m1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(len(y) - 2, 1); Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, m1 - m0))


def directional_flip(dl_clean, dl_steered, toward):
    if toward == "beta":
        return int(dl_clean < 0 and dl_steered > 0)
    return int(dl_clean > 0 and dl_steered < 0)


# =====================================================================
# Self-test: a causal model (flips while intact) vs a non-causal one
# (flips only when "broken"); confirm the intact-flip logic separates them.
# =====================================================================
def self_test():
    rng = np.random.default_rng(75)
    d, n = 64, 200
    y = rng.integers(0, 2, n)
    w = rng.standard_normal(d); w /= np.linalg.norm(w)
    H = rng.standard_normal((n, d)) * 0.5 + np.outer((y * 2 - 1.) * 1.5, w)

    # CAUSAL toy readout: margin = <h, w>; pushing along w flips it, answer "intact"
    # as long as |margin| not absurd. "broken" if push norm > break_thr (collapse).
    break_thr = 6.0
    def causal_eval(h, step):
        hs = h + step
        margin = float(hs @ w)
        broken = np.linalg.norm(step) > break_thr      # too big a push => garbage
        top_is_answer = not broken
        return margin, top_is_answer
    # NON-CAUSAL: margin ignores w (depends on orthogonal coord); only breakage flips
    w_perp = rng.standard_normal(d); w_perp -= (w_perp @ w) * w; w_perp /= np.linalg.norm(w_perp)
    def noncausal_eval(h, step):
        hs = h + step
        margin = float(hs @ w_perp)
        broken = np.linalg.norm(step) > break_thr
        return (rng.standard_normal() if broken else margin), (not broken)

    cgrid = [1, 2, 4, 8, 16]
    sig = 1.0
    def intact_flip_rate(evalfn, direction):
        alpha_idx = [i for i in range(n) if (H[i] @ w) < 0][:60]
        out = {}
        for c in cgrid:
            flips_intact = []
            for i in alpha_idx:
                m0, _ = evalfn(H[i], np.zeros(d))
                m1, intact = evalfn(H[i], (c * sig) * direction)
                f = int(m0 < 0 and m1 > 0)
                flips_intact.append(f * int(intact))      # only count if intact
            out[c] = float(np.mean(flips_intact)) if flips_intact else 0.0
        return out

    causal_curve = intact_flip_rate(causal_eval, w)
    noncausal_curve = intact_flip_rate(noncausal_eval, w)
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  causal model    intact-flip by c: {causal_curve}")
    print(f"  non-causal model intact-flip by c: {noncausal_curve}")
    assert max(causal_curve.values()) > 0.5, "causal model must show intact flips"
    assert max(noncausal_curve.values()) < 0.2, "non-causal must NOT flip while intact"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(intact-flip metric counts a flip only while the answer is still a real")
    print(" answer -- separating true steering from flips-via-breakage)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run (model forward passes with additive hook at a tap) -- mirrors 65
# =====================================================================
def _chain(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cd = np.load(args.concept_npz)
    Sigma_inv = cd["Sigma_inv"].astype(np.float64); d = Sigma_inv.shape[0]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks)
    norm_mod = _chain(model, "model.norm")
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    n_tr = int(round(len(fams) * args.train_frac)); train_fams = set(fams[:n_tr])
    steer_layers = args.steer_layers or [24]            # default: the strongest layer from 65
    taps = [f"postL{L}" for L in steer_layers] + (["final"] if args.include_final else [])

    def module_for(tap):
        return norm_mod if tap == "final" else blocks[int(tap.replace("postL", "")) + 1]

    # capture clean residuals at taps (for fisher + sigma + ||h||)
    def capture(ptext):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in steer_layers:
            if L + 1 < n_layers:
                def mk(L=L):
                    def pre(m, a): g[f"postL{L}"] = a[0][0, -1, :].detach().float().cpu().numpy()
                    return pre
                handles.append(blocks[L + 1].register_forward_pre_hook(mk(), with_kwargs=False))
        if args.include_final:
            def mkf():
                def pre(m, a): g["final"] = a[0][0, -1, :].detach().float().cpu().numpy()
                return pre
            handles.append(norm_mod.register_forward_pre_hook(mkf(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return g

    logger.info("capturing clean residuals for %d prompts...", len(prompts))
    H = {t: [] for t in taps}; y = []; trm = []; hn = {t: [] for t in taps}
    for i, p in enumerate(prompts):
        g = capture(p["prompt"])
        for t in taps:
            H[t].append(g[t]); hn[t].append(float(np.linalg.norm(g[t])))
        y.append(1 if p["correct_answer"].strip() == "beta" else 0)
        trm.append(p["surface_family"] in train_fams)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))
    for t in taps:
        H[t] = np.array(H[t], np.float64)
    y = np.array(y); trm = np.array(trm)

    geom = {}
    for t in taps:
        Htr, ytr = H[t][trm], y[trm]
        w_res = fisher_axis(Htr, ytr, args.shrink)
        shuf = [fisher_axis(Htr, rng.permutation(ytr), args.shrink) for _ in range(args.n_shuffle)]
        sigma = float(np.std(Htr @ w_res))
        geom[t] = {"w_res": w_res, "shuf": shuf, "sigma": sigma,
                   "hnorm_mean": float(np.mean(hn[t]))}
        logger.info("tap %s: sigma=%.4f  mean||h||=%.1f", t, sigma, geom[t]["hnorm_mean"])

    def dl_and_top(ptext, tap, delta):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        if delta is None:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
        else:
            dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
            def pre(m, a):
                hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
            h = module_for(tap).register_forward_pre_hook(pre, with_kwargs=False)
            try:
                with torch.no_grad():
                    o = model(**inp, use_cache=False)
            finally:
                h.remove()
        logits = o.logits[0, -1, :].float()
        lp = torch.log_softmax(logits, 0)
        top = int(torch.argmax(logits).item())
        return float(lp[beta_id] - lp[alpha_id]), top

    held = [p for p in prompts if p["surface_family"] not in train_fams]
    ha = [p for p in held if p["correct_answer"].strip() == "alpha"][:args.max_targets]
    hb = [p for p in held if p["correct_answer"].strip() == "beta"][:args.max_targets]
    cgrid = sorted(args.c_grid)
    answer_ids = {alpha_id, beta_id}

    rows = []
    def run(tap, targets, toward, signed_cs):
        sig = geom[tap]["sigma"]
        dirs = {"w_res": unit_raw(geom[tap]["w_res"])}
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        for j, sd in enumerate(geom[tap]["shuf"]):
            dirs[f"shuffled{j}"] = unit_raw(sd)
        for tgt in targets:
            dlc, topc = dl_and_top(tgt["prompt"], tap, None)
            if toward == "beta" and not (dlc < 0):
                continue
            if toward == "alpha" and not (dlc > 0):
                continue
            for c in signed_cs:
                for name, vec in dirs.items():
                    step = (c * sig) * vec
                    dls, tops = dl_and_top(tgt["prompt"], tap, step)
                    rows.append({"tap": tap, "toward": toward, "dir": name, "c": abs(c),
                                 "flip": directional_flip(dlc, dls, toward),
                                 "intact": int(tops in answer_ids),
                                 "rel_push": float(np.linalg.norm(step) / (np.linalg.norm(H[tap][0]) + 1e-9))})

    for tap in taps:
        logger.info("saturation steering at %s (alpha->beta)...", tap)
        run(tap, ha, "beta", cgrid)
        logger.info("saturation steering at %s (beta->alpha)...", tap)
        run(tap, hb, "alpha", [-c for c in cgrid])

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "steering_saturation_curve.csv", index=False)
    df["dg"] = df["dir"].map(lambda k: "random" if k.startswith("random")
                             else "shuffled" if k.startswith("shuffled") else k)

    # build curves: per tap, per c -> w_res flip, w_res INTACT-flip, shuffled p95 flip,
    # intact-rate, mean relative push
    curve = []
    for tap in taps:
        for c in cgrid:
            sub = df[(df.tap == tap) & (df.c == c)]
            w = sub[sub.dg == "w_res"]
            shuf = sub[sub.dg == "shuffled"].groupby("dir")["flip"].mean()
            rand = sub[sub.dg == "random"].groupby("dir")["flip"].mean()
            wres_flip = float(w["flip"].mean()) if len(w) else float("nan")
            wres_intact_flip = float((w["flip"] * w["intact"]).mean()) if len(w) else float("nan")
            curve.append({
                "tap": tap, "c": c,
                "wres_flip": round(wres_flip, 3),
                "wres_intact_flip": round(wres_intact_flip, 3),
                "shuffled_flip_p95": round(float(np.percentile(shuf, 95)), 3) if len(shuf) else None,
                "random_flip_mean": round(float(rand.mean()), 3) if len(rand) else None,
                "intact_rate": round(float(w["intact"].mean()), 3) if len(w) else None,
                "rel_push_mean": round(float(w["rel_push"].mean()), 3) if len(w) else None,
            })
            logger.info("%s c=%5.1f: wres_flip=%.2f intact_flip=%.2f shuf_p95=%.2f rand=%.2f "
                        "intact=%.2f relpush=%.2f", tap, c, wres_flip, wres_intact_flip,
                        np.percentile(shuf, 95) if len(shuf) else float('nan'),
                        rand.mean() if len(rand) else float('nan'),
                        w["intact"].mean() if len(w) else float('nan'),
                        w["rel_push"].mean() if len(w) else float('nan'))

    # verdict: is there ANY c where wres_intact_flip beats shuffled p95 AND answers mostly intact?
    causal_c = [r for r in curve
                if r["shuffled_flip_p95"] is not None
                and r["wres_intact_flip"] > r["shuffled_flip_p95"]
                and r["wres_intact_flip"] >= args.tau_flip
                and (r["intact_rate"] or 0) >= args.intact_min]
    max_intact_flip = max((r["wres_intact_flip"] for r in curve), default=float("nan"))
    verdict = (
        f"OUTCOME B (CAUSAL AT HIGH AMPLITUDE): at {causal_c[0]['tap']} c={causal_c[0]['c']}, w_res "
        f"intact-flip {causal_c[0]['wres_intact_flip']:.2f} exceeds shuffled p95 "
        f"{causal_c[0]['shuffled_flip_p95']:.2f} while answers remain intact "
        f"({causal_c[0]['intact_rate']:.2f}). A causal magnitude EXISTS -- revisit the conclusion."
        if causal_c else
        f"OUTCOME A/C (NO CAUSAL MAGNITUDE): across c up to {max(cgrid)} sigma, w_res intact-flip "
        f"never exceeds the shuffled band at/above tau={args.tau_flip} while answers stay intact "
        f"(max intact-flip {max_intact_flip:.2f}). Any flips coincide with answer breakage / are "
        "non-specific. 'No magnitude steers the concept' is confirmed with the full curve.")

    summary = {"params": {"taps": taps, "c_grid": cgrid, "tau_flip": args.tau_flip,
                          "intact_min": args.intact_min, "n_shuffle": args.n_shuffle,
                          "n_random": args.n_random},
               "sigma": {t: geom[t]["sigma"] for t in taps},
               "mean_h_norm": {t: geom[t]["hnorm_mean"] for t in taps},
               "curve": curve, "verdict": verdict}
    with open(out / "steering_saturation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n" + "=" * 92)
    print("STEERING SATURATION  --  does ANY push magnitude flip the concept? (intact + specific)")
    print("=" * 92)
    print(f"{'tap':>8} {'c':>6} {'wres_flip':>10} {'intact_flip':>12} {'shuf_p95':>9} "
          f"{'rand':>6} {'intact%':>8} {'rel_push':>9}")
    for r in curve:
        print(f"{r['tap']:>8} {r['c']:>6.1f} {r['wres_flip']:>10.2f} {r['wres_intact_flip']:>12.2f} "
              f"{(r['shuffled_flip_p95'] or 0):>9.2f} {(r['random_flip_mean'] or 0):>6.2f} "
              f"{(r['intact_rate'] or 0):>8.2f} {(r['rel_push_mean'] or 0):>9.2f}")
    print("\nVERDICT: " + verdict)
    print(f"\nwrote: {out}/steering_saturation_summary.json + steering_saturation_curve.csv")
    print("=" * 92)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--alpha_answer", default="alpha")
    p.add_argument("--beta_answer", default="beta")
    p.add_argument("--steer_layers", type=int, nargs="*", default=[24])
    p.add_argument("--include_final", action="store_true", help="also steer at final pre-norm")
    p.add_argument("--c_grid", type=float, nargs="*", default=[0.5, 1, 2, 4, 8, 16, 32])
    p.add_argument("--max_targets", type=int, default=60)
    p.add_argument("--n_shuffle", type=int, default=10)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--tau_flip", type=float, default=0.5, help="intact-flip rate to call it causal")
    p.add_argument("--intact_min", type=float, default=0.5, help="min intact-rate for a flip to count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="data/analysis/runD_v2/steering_saturation")
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
