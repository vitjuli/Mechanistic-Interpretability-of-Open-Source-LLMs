"""
102_multilayer_wres_steering.py   [does a SUSTAINED push along w_res across all layers flip behaviour?]
========================================================================================================
Exp 65/75/90 steered along w_res at a FEW layers -> intact-flip 0 (concept axis not a lever). But a
single layer may be weak because downstream blocks rewrite the residual. The exp-89 calculus predicts
Delta(margin) ~ c * sum_L sigma_L <g^L, w_res_hat^L>: 36 small terms (each ~cos 0.027) might accumulate.
This script tests the readable axis under a SUSTAINED push:

  (a) SINGLE-LAYER sweep: +-c*sigma_L*w_res_hat^L at each of the 36 layers individually (alpha->beta with
      +, beta->alpha with -), escalating c. vs random-direction null. -> per-layer profile (completeness).
  (b) COMPOSED: push along w_res_hat^L at ALL layers at once, small per-layer c_each. intact-flip both
      arms. Controls (matched #layers): multi-layer random-direction null + multi-layer shuffled-label
      Fisher null (the strong null: a per-layer axis that separates classes by chance).

A composed intact-flip ABOVE both nulls => the concept is usable when continuously reinforced (a finding).
A composed intact-flip AT null even at large c_each (where random also fires = norm artifact) => the
readable axis is inert even when reinforced across the whole stack (the strongest negative). Norm artifact
guard: only c_each where the RANDOM/shuffled null stays ~0 are valid.

SELF-TEST (no torch):  python 102_multilayer_wres_steering.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ml_wres")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def self_test():
    rng = np.random.default_rng(0); d, n = 16, 200
    y = (rng.random(n) > 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))
    H = 0.3 * rng.standard_normal((n, d)) + np.outer(2 * y - 1, w)
    wr = fisher_axis(H, y)
    assert abs(wr @ w) > 0.6, "fisher should recover separating dir"
    ws = fisher_axis(H, rng.permutation(y))         # shuffled-label axis
    assert abs(ws @ w) < abs(wr @ w), "shuffled-label axis should align with true sep dir less"
    # composition accumulation: sum of small per-layer projections
    gs = [unit_raw(rng.standard_normal(d)) for _ in range(36)]
    total = sum(abs(g @ w) for g in gs)
    assert total > 1.0, "36 small terms accumulate"
    print("[self_test] OK — fisher recovery, shuffled-null weaker, composition accumulates.")


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
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = list(range(n_layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    P = len(prompts)
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

    # ---- capture residuals (answer pos) for w_res + sigma ----
    H = np.zeros((P, n_layers, d), np.float32)
    clean_pred = np.zeros(P, int)
    logger.info("capturing forced residuals over %d prompts...", P)
    for i in range(P):
        enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
        keep, hs = {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): keep[L] = a[0].detach()[0, -1, :]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                lo = model(**enc, use_cache=False).logits[0, -1, :]
            clean_pred[i] = int(lo.argmax().item())
            for L in layers: H[i, L] = keep[L].float().cpu().numpy()
        finally:
            for h in hs: h.remove()
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, P)

    intact_rate = float(np.mean(np.isin(clean_pred, [a_id, b_id])))
    acc = float(np.mean((clean_pred == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f acc=%.3f", intact_rate, acc)

    tr = is_train
    w_res = {L: fisher_axis(H[tr, L].astype(np.float64), y[tr], args.shrink) for L in layers}
    sig = {L: float(np.std(H[tr, L].astype(np.float64) @ w_res[L])) for L in layers}
    # shuffled-label axes (a few draws) + per-layer sigma for them
    yshuf = [rng.permutation(y[tr]) for _ in range(args.n_shuffle)]
    w_shuf = [{L: fisher_axis(H[tr, L].astype(np.float64), ys, args.shrink) for L in layers} for ys in yshuf]

    a_idx = np.where((y == 0) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    b_idx = np.where((y == 1) & np.isin(clean_pred, [a_id, b_id]) & (~tr))[0]
    rng.shuffle(a_idx); rng.shuffle(b_idx)

    def steer_hook(vec):
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        def pre(m, a):
            a[0][:, -1, :] = a[0][:, -1, :] + v; return (a[0],) + tuple(a[1:])
        return pre

    def intact_single(idxs, L, vhat, sigma, c, target_id, sign):
        if len(idxs) == 0: return float("nan")
        vec = sign * c * sigma * vhat; k = 0
        for i in idxs:
            enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
            h = tap_module(L).register_forward_pre_hook(steer_hook(vec), with_kwargs=False)
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                h.remove()
            k += int(int(lo.argmax().item()) == target_id)
        return k / len(idxs)

    def intact_composed(idxs, dir_per_layer, c, target_id, sign):
        if len(idxs) == 0: return float("nan")
        vecs = {L: sign * c * sig_of[L] * dir_per_layer[L] for L in layers}
        k = 0
        for i in idxs:
            enc = tok([forced_text(i)], return_tensors="pt").to(args.device)
            hs = [tap_module(L).register_forward_pre_hook(steer_hook(vecs[L]), with_kwargs=False) for L in layers]
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                for h in hs: h.remove()
            k += int(int(lo.argmax().item()) == target_id)
        return k / len(idxs)

    # ---------- (a) single-layer sweep ----------
    ai, bi = a_idx[: args.n_single], b_idx[: args.n_single]
    logger.info("=== single-layer sweep: %d+%d prompts, c=%s ===", len(ai), len(bi), args.c_single)
    rows_s = []
    for L in layers:
        rdir = unit_raw(rng.standard_normal(d)); sr = float(np.std(H[tr, L].astype(np.float64) @ rdir))
        for c in args.c_single:
            iw = 0.5 * (intact_single(ai, L, w_res[L], sig[L], c, b_id, +1) +
                        intact_single(bi, L, w_res[L], sig[L], c, a_id, -1))
            ir = 0.5 * (intact_single(ai, L, rdir, sr, c, b_id, +1) +
                        intact_single(bi, L, rdir, sr, c, a_id, -1))
            rows_s.append(dict(layer=int(L), c=float(c), intact_wres=iw, intact_random=ir))
            logger.info("  [single L%02d c=%.1f] w_res=%.2f random=%.2f", L, c, iw, ir)

    # ---------- (b) composed across all layers ----------
    ai2, bi2 = a_idx[: args.n_comp], b_idx[: args.n_comp]
    logger.info("=== composed all-layer: %d+%d prompts, c_each=%s ===", len(ai2), len(bi2), args.c_comp)
    rows_c = []
    rand_dirs = {L: unit_raw(rng.standard_normal(d)) for L in layers}
    for c in args.c_comp:
        # w_res sigma
        sig_of = sig
        iw = 0.5 * (intact_composed(ai2, w_res, c, b_id, +1) + intact_composed(bi2, w_res, c, a_id, -1))
        # random null (matched #layers), sigma along each random dir
        sig_of = {L: float(np.std(H[tr, L].astype(np.float64) @ rand_dirs[L])) for L in layers}
        ir = 0.5 * (intact_composed(ai2, rand_dirs, c, b_id, +1) + intact_composed(bi2, rand_dirs, c, a_id, -1))
        # shuffled-label null (avg over draws), sigma along each shuffled axis
        ishs = []
        for ws in w_shuf:
            sig_of = {L: float(np.std(H[tr, L].astype(np.float64) @ ws[L])) for L in layers}
            ishs.append(0.5 * (intact_composed(ai2, ws, c, b_id, +1) + intact_composed(bi2, ws, c, a_id, -1)))
        ish = float(np.mean(ishs))
        rows_c.append(dict(c_each=float(c), intact_wres=iw, intact_random=ir, intact_shuffled=ish))
        logger.info("  [composed c_each=%.2f] w_res=%.2f | random=%.2f shuffled=%.2f", c, iw, ir, ish)

    with open(out / "single_layer_sweep.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows_s[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows_s]
    with open(out / "composed_sweep.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows_c[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows_c]

    print("\n" + "=" * 100)
    print("MULTI-LAYER w_res STEERING (forced, intact-flip) — does a sustained push flip behaviour?")
    print("=" * 100)
    print(f"forced clean intact-rate {intact_rate:.2f}, acc {acc:.2f}\n")
    bestL = max(layers, key=lambda L: max((r["intact_wres"] for r in rows_s if r["layer"] == L), default=0))
    bestLv = max(r["intact_wres"] for r in rows_s if r["layer"] == bestL)
    print(f"(a) single-layer: best w_res-layer L{bestL} intact={bestLv:.2f} "
          f"(random there {max(r['intact_random'] for r in rows_s if r['layer']==bestL):.2f}); "
          f"layers above random: "
          f"{sorted({r['layer'] for r in rows_s if r['intact_wres'] > r['intact_random'] + 0.1})}")
    print("\n(b) composed across ALL layers (valid only where random/shuffled ~0):")
    for r in rows_c:
        valid = max(r["intact_random"], r["intact_shuffled"]) < 0.1
        flag = "VALID" if valid else "norm-artifact (null fires)"
        beats = r["intact_wres"] > max(r["intact_random"], r["intact_shuffled"]) + 0.1
        note = " <= w_res beats nulls" if (valid and beats) else (" <= w_res at null" if valid else "")
        print(f"  c_each={r['c_each']:.2f}: w_res {r['intact_wres']:.2f} | random {r['intact_random']:.2f} "
              f"shuffled {r['intact_shuffled']:.2f}  [{flag}]{note}")
    print("\nReading: a composed w_res intact-flip ABOVE both nulls in a VALID window => concept usable when "
          "reinforced across the stack (finding). At null in valid window even composed => readable axis inert "
          "even when continuously reinforced (strongest negative). Large c_each where random also fires is a norm "
          "artifact, not evidence.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/ml_wres")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--c_single", type=float, nargs="*", default=[2.0, 8.0])
    p.add_argument("--c_comp", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--n_single", type=int, default=16)
    p.add_argument("--n_comp", type=int, default=32)
    p.add_argument("--n_shuffle", type=int, default=2)
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
