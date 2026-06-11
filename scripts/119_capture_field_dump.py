"""
119_capture_field_dump.py   [one-pass capture: per-prompt fields for ALL layers, FULL corpus]
==============================================================================================
Architectural fix. Scripts 86/89 capture residuals + margin-gradients in memory, use them,
and DISCARD them; their CSVs are aggregates over layer/target subsets. Every downstream
question (flip-law calibration, null calibration, reliability bootstraps, future concepts)
re-pays the GPU cost. This script captures ONCE and dumps EVERYTHING:

  per layer L in 0..n_layers-1 (residual tap = input of block L+1; final = input of norm):
      res_L{L:02d}.npy    (n_prompts, d)  float32   residual at answer position
      grad_L{L:02d}.npy   (n_prompts, d)  float32   d(logit_beta - logit_alpha)/d h^(L)
  meta.npz:
      y                   (n,)   int      0=alpha 1=beta (correct answer)
      clean_margin        (n,)   float    log-softmax margin lp[beta]-lp[alpha]
                                          (== raw logit difference exactly)
      baseline_top1       (n,)   int      argmax token id of the clean run
      baseline_intact     (n,)   int      top1 in {alpha_id, beta_id}
      topk_ids/topk_logits(n,K)           top-K clean logits (for later intact analyses)
      rms_final           (n,)   float    final-residual rms (frozen-rms DLA convention)
      alpha_id, beta_id, d, n_layers, model_name, prompts_sha
      wU_diff             (d,)            unembedding contrast gamma(beta)-gamma(alpha)
  families.json           surface_family per prompt (split reconstruction downstream)

Cost: ONE forward+backward per prompt (the backward populates gradients at every tap
simultaneously). 538 prompts => minutes-to-an-hour on a single A100. Disk: ~0.4 GB.

Downstream consumers: 116 (flip-law retro), 118 (null calibration + reliability),
and any future per-prompt geometry without touching the GPU.

Conventions copied verbatim from 86: tap(L) = forward_pre_hook on blocks[L+1] (norm for the
last), Fisher/split logic NOT done here (done downstream so the dump stays split-agnostic).

SELF-TEST (no torch / no repo):  python 119_capture_field_dump.py --self_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("capture119")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def finite_diff_grad(f, h, eps=1e-5):
    g = np.zeros_like(h)
    for j in range(len(h)):
        e = np.zeros_like(h); e[j] = eps
        g[j] = (f(h + e) - f(h - e)) / (2 * eps)
    return g


def topk_row(row, k):
    idx = np.argsort(-row)[:k]
    return idx.astype(np.int64), row[idx].astype(np.float32)


def sha16(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()[:16]


# =====================================================================
# Self-test: the dump round-trips and gradients mean what we say they mean
# =====================================================================
def self_test():
    import tempfile
    rng = np.random.default_rng(0)
    d, n, n_layers, K = 12, 24, 3, 5
    a = unit_raw(rng.standard_normal(d))                      # toy usage direction
    H = {L: rng.standard_normal((n, d)) for L in range(n_layers)}
    margin = lambda h: float(h @ a)
    G = {L: np.stack([finite_diff_grad(margin, H[L][i]) for i in range(n)]) for L in range(n_layers)}
    for L in range(n_layers):
        assert all(abs(unit_raw(G[L][i]) @ a) > 0.999 for i in range(n)), "gradient must equal usage dir in toy"

    vocab = 30
    logits = rng.standard_normal((n, vocab))
    ids = np.zeros((n, K), np.int64); vals = np.zeros((n, K), np.float32)
    for i in range(n):
        ids[i], vals[i] = topk_row(logits[i], K)
    assert np.all(vals[:, 0] >= vals[:, -1]), "topk sorted descending"
    assert np.all(ids[:, 0] == logits.argmax(1)), "top1 consistent"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for L in range(n_layers):
            np.save(out / f"res_L{L:02d}.npy", H[L].astype(np.float32))
            np.save(out / f"grad_L{L:02d}.npy", G[L].astype(np.float32))
        np.savez(out / "meta.npz", y=np.arange(n) % 2, clean_margin=(np.stack([H[0][i] for i in range(n)]) @ a),
                 d=d, n_layers=n_layers)
        H0 = np.load(out / "res_L00.npy"); m = np.load(out / "meta.npz")
        assert H0.shape == (n, d) and int(m["n_layers"]) == n_layers
        # round-trip preserves the calculus: pred = g . delta equals measured margin change
        delta = 0.3 * rng.standard_normal(d)
        i = 3
        pred = float(np.load(out / "grad_L01.npy")[i] @ delta)
        meas = margin(H[1][i] + delta) - margin(H[1][i])
        assert abs(pred - meas) < 1e-4, "dumped gradient supports first-order prediction"
    print("[self_test] OK — gradient semantics, topk, dump round-trip, first-order prediction pass.")


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

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    W_U = _chain(model, "lm_head").weight.detach().float().cpu().numpy()
    wU_diff = (W_U[beta_id] - W_U[alpha_id]).astype(np.float32)
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
    last = n_layers - 1
    layers = list(range(n_layers))                      # ALL layers, no subsetting — by design
    logger.info("model: %d layers, d=%d | capturing ALL taps, FULL corpus", n_layers, d)

    prompt_files = [args.prompts] + (args.extra_prompts or [])
    prompts = []
    for pf in prompt_files:
        prompts += [json.loads(l) for l in open(pf)]
    nP = len(prompts)
    per_prompt_ids = all(("tok_id_class0" in p and "tok_id_class1" in p) for p in prompts)
    logger.info("prompts: %d total from %s | per-prompt class token ids: %s",
                nP, prompt_files, per_prompt_ids)
    if per_prompt_ids:
        id0 = np.array([int(p["tok_id_class0"]) for p in prompts])
        id1 = np.array([int(p["tok_id_class1"]) for p in prompts])
    else:
        id0 = np.full(nP, alpha_id); id1 = np.full(nP, beta_id)

    for p_ in model.parameters():
        p_.requires_grad_(True)

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    grad = {L: np.zeros((nP, d), np.float32) for L in layers}
    y = np.zeros(nP, np.int64)
    clean_margin = np.zeros(nP, np.float64)
    raw_margin = np.zeros(nP, np.float64)
    baseline_top1 = np.zeros(nP, np.int64)
    baseline_intact = np.zeros(nP, np.int64)
    rms_final = np.zeros(nP, np.float32)
    topk_ids = np.zeros((nP, args.topk), np.int64)
    topk_logits = np.zeros((nP, args.topk), np.float32)
    families = []

    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        keep = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            o = model(**inp, use_cache=False)
            row = o.logits[0, -1, :]
            rm = row[int(id1[i])] - row[int(id0[i])]
            rm.backward()
            rowf = row.detach().float()
            lp = torch.log_softmax(rowf, 0)
            clean_margin[i] = float(lp[int(id1[i])] - lp[int(id0[i])])
            raw_margin[i] = float(rm.item())
            baseline_top1[i] = int(rowf.argmax().item())
            baseline_intact[i] = int(baseline_top1[i] in (int(id0[i]), int(id1[i])))
            ids_np, vals_np = topk_row(rowf.cpu().numpy(), args.topk)
            topk_ids[i], topk_logits[i] = ids_np, vals_np
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            t = keep[L]
            res[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
            grad[L][i] = (t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0)
        hf = res[last][i].astype(np.float64)
        rms_final[i] = float(np.sqrt(np.mean(hf ** 2) + rms_eps))
        model.zero_grad(set_to_none=True)
        y[i] = int(prompts[i].get("y_canonical",
                                  1 if p["correct_answer"].strip() == "beta" else 0))
        families.append(p.get("surface_family", f"__nofam_{i}"))
        if (i + 1) % 50 == 0:
            logger.info("  capture %d/%d", i + 1, nP)

    # ---------- consistency checks before writing ----------
    # raw margin must equal log-softmax margin exactly (the 89 identity)
    gap = float(np.max(np.abs(raw_margin - clean_margin)))
    assert gap < 1e-3, f"margin identity violated: max gap {gap}"
    # final-tap gradient must align with the unembedding contrast modulo final rmsnorm —
    # we check the cheap necessary condition: nonzero and stable norm
    gn = np.linalg.norm(grad[last], axis=1)
    assert float(gn.min()) > 0, "zero gradient at final tap — hook wiring broken"
    if per_prompt_ids:
        # per-prompt contrasts: validate against the mean unembedding contrast over prompts
        wU = W_U
        contr = np.stack([wU[int(id1[i])] - wU[int(id0[i])] for i in range(nP)]).mean(0)
        cf = float(unit_raw(grad[last].mean(0).astype(np.float64)) @ unit_raw(contr.astype(np.float64)))
    else:
        cf = float(unit_raw(grad[last].mean(0).astype(np.float64)) @ unit_raw(wU_diff.astype(np.float64)))
    logger.info("sanity: cos(u_final, gamma_bar) = %+.3f (high => machinery valid)", cf)

    # ---------- dump ----------
    for L in layers:
        np.save(out / f"res_L{L:02d}.npy", res[L])
        np.save(out / f"grad_L{L:02d}.npy", grad[L])
    np.savez(out / "meta.npz",
             y=y, clean_margin=clean_margin, baseline_top1=baseline_top1,
             baseline_intact=baseline_intact, topk_ids=topk_ids, topk_logits=topk_logits,
             rms_final=rms_final, wU_diff=wU_diff,
             alpha_id=alpha_id, beta_id=beta_id, d=d, n_layers=n_layers,
             per_prompt_ids=per_prompt_ids, id_class0=id0, id_class1=id1,
             model_name=args.model_name,
             cos_u_final_gamma_bar=cf,
             prompts_sha=";".join(f"{Path(pf).name}:{sha16(pf)}" for pf in prompt_files))
    json.dump(families, open(out / "families.json", "w"))
    json.dump([{"idx": i, "prompt": p["prompt"], "correct_answer": p["correct_answer"],
                "surface_family": families[i], "cue_type": p.get("cue_type")}
               for i, p in enumerate(prompts)],
              open(out / "prompt_index.json", "w"))

    size_mb = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e6
    print("\n" + "=" * 80)
    print("CAPTURE FIELD DUMP COMPLETE")
    print("=" * 80)
    print(f"prompts: {nP} | layers: {n_layers} | d: {d} | npy size: {size_mb:.0f} MB")
    print(f"baseline intact rate: {baseline_intact.mean():.3f} | margin identity gap: {gap:.2e}")
    print(f"cos(u_final, gamma_bar) = {cf:+.3f}")
    print(f"dump dir: {out}")
    print("consumers: 116_flip_law_retro.py, 118_null_calibration.py")
    print("=" * 80 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--extra_prompts", nargs="*", default=None,
                   help="optional further jsonl files (e.g. the supplement) — FULL corpus by default")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--topk", type=int, default=50)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
