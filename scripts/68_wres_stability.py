"""
68_wres_stability.py   [JOB 4 / 4]
===================================================================
Robustness for the central D1 finding (64): alpha/beta is linearly decodable in
the residual stream (held-out AUC ~0.99) along an axis w_res that is ~orthogonal
to gbar. A reviewer will ask: is w_res a REAL, stable direction, or an artefact
of one train/held-out split (a probe overfitting in d=2560)?

This resamples the family-level split many times and measures:
  * held-out AUC distribution of w_res (is decodability robust, not split-luck?)
  * w_res STABILITY: causal-metric cosine between w_res fitted on independent
    resamples (a stable direction => high pairwise cos_C; an overfit artefact =>
    low, near-null cos). With a random-direction null for calibration.
  * cos_C(w_res, gbar) distribution (is the orthogonality-to-gbar finding stable?)
  * (optional) shuffled-label AUC -- should be ~0.5 (sanity: real signal, not leakage).

No transcoders needed: only the model + residual capture + Sigma_inv (60_ npz).
Captures residuals once at the chosen depth, then resamples splits in-memory.

SELF-TEST: python 68_wres_stability.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wres_stability")


def whitener(Si):
    w, V = np.linalg.eigh(Si); w = np.clip(w, 0, None); return (V * np.sqrt(w)) @ V.T

def causal_cos(a, b, Si):
    num = float(a @ Si @ b); na = float(np.sqrt(max(a @ Si @ a, 1e-30))); nb = float(np.sqrt(max(b @ Si @ b, 1e-30)))
    return num / (na * nb)

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1); Sw = .5 * (Sw + Sw.T)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit(np.linalg.solve(Sw, mu1 - mu0))

def auc(H, y, w):
    s = H @ w; o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def self_test():
    rng = np.random.default_rng(68); d = 150
    ev = np.concatenate([np.linspace(12, 3, 12), np.linspace(2, .05, d - 12)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); S = (Q * ev) @ Q.T; S = .5 * (S + S.T)
    Si = np.linalg.inv(S + 1e-3 * np.mean(np.diag(S)) * np.eye(d))
    w_true = unit(rng.standard_normal(d)); n = 400; y = rng.integers(0, 2, n)
    H = rng.standard_normal((n, d)) @ np.linalg.cholesky(S).T * 0.5 + np.outer((y * 2 - 1.) * 2., w_true)
    # resample splits
    cos_pairs, aucs, cos_true = [], [], []
    ws = []
    for s in range(8):
        idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]
        w = fisher_axis(H[tr], y[tr]); ws.append(w)
        aucs.append(auc(H[te], y[te], w)); cos_true.append(abs(causal_cos(w, w_true, Si)))
    for a, b in combinations(ws, 2):
        cos_pairs.append(abs(causal_cos(a, b, Si)))
    # null: cos between random dirs
    null = [abs(causal_cos(unit(rng.standard_normal(d)), unit(rng.standard_normal(d)), Si)) for _ in range(200)]
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  held-out AUC: mean {np.mean(aucs):.3f} min {np.min(aucs):.3f}")
    print(f"  pairwise stability cos_C(w_i,w_j): mean {np.mean(cos_pairs):.3f}  null p95 {np.percentile(null,95):.3f}")
    print(f"  cos_C(w, w_true): mean {np.mean(cos_true):.3f}")
    assert np.mean(aucs) > 0.9, "decodability must be robust"
    assert np.mean(cos_pairs) > 3 * np.percentile(null, 95), "w_res must be stable across splits (>> null)"
    print("\nALL SELF-TEST ASSERTIONS PASSED \n--------------------------------------------\n")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cd = np.load(args.concept_npz); Si = cd["Sigma_inv"].astype(np.float64)
    gbar = cd["gbar"].astype(np.float64); d = Si.shape[0]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = model.model.layers; n_layers = len(blocks)
    norm_mod = model.model.norm

    prompts = [json.loads(l) for l in open(args.prompts)]
    L = args.tap_layer
    use_final = (L < 0)

    def capture(ptext):
        inp = tok([ptext], return_tensors="pt").to(args.device); g = {}; hs = []
        if use_final:
            def pre(m, a): g["h"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
            hs.append(norm_mod.register_forward_pre_hook(pre, with_kwargs=False))
        else:
            def pre(m, a): g["h"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
            hs.append(blocks[L + 1].register_forward_pre_hook(pre, with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inp, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return g["h"]

    logger.info("Capturing residuals at %s for %d prompts...", "final" if use_final else f"postL{L}", len(prompts))
    H, y, fam = [], [], []
    for p in prompts:
        H.append(capture(p["prompt"]))
        y.append(1 if p["correct_answer"].strip() == "beta" else 0)
        fam.append(p["surface_family"])
    H = np.array(H, np.float64); y = np.array(y); fam = np.array(fam)
    families = sorted(set(fam))

    aucs, cos_g, ws, shuf_auc = [], [], [], []
    for s in range(args.n_resample):
        fl = list(families); rng.shuffle(fl)
        ntr = int(round(len(fl) * args.train_frac)); trf = set(fl[:ntr])
        trm = np.array([f in trf for f in fam])
        w = fisher_axis(H[trm], y[trm], args.shrink)
        ws.append(w); aucs.append(auc(H[~trm], y[~trm], w)); cos_g.append(causal_cos(w, gbar, Si))
        if args.shuffle_check:
            ysh = rng.permutation(y[trm])
            wsh = fisher_axis(H[trm], ysh, args.shrink)
            shuf_auc.append(auc(H[~trm], y[~trm], wsh))

    cos_pairs = [abs(causal_cos(a, b, Si)) for a, b in combinations(ws, 2)]
    A = whitener(Si)
    def _ccos(u, v):
        uu = A @ u; uu /= (np.linalg.norm(uu) + 1e-30)
        vv = A @ v; vv /= (np.linalg.norm(vv) + 1e-30)
        return abs(float(uu @ vv))
    null = [_ccos(rng.standard_normal(d), rng.standard_normal(d)) for _ in range(500)]

    res = {
        "tap": "final" if use_final else f"postL{L}",
        "heldout_auc": {"mean": float(np.mean(aucs)), "min": float(np.min(aucs)),
                        "max": float(np.max(aucs)), "std": float(np.std(aucs))},
        "wres_stability_cos": {"mean": float(np.mean(cos_pairs)), "p05": float(np.percentile(cos_pairs, 5)),
                               "p95": float(np.percentile(cos_pairs, 95))},
        "random_cos_null": {"mean": float(np.mean(null)), "p95": float(np.percentile(null, 95))},
        "cos_C_wres_gbar": {"mean": float(np.mean(cos_g)), "std": float(np.std(cos_g)),
                            "min": float(np.min(cos_g)), "max": float(np.max(cos_g))},
        "shuffled_label_auc": ({"mean": float(np.mean(shuf_auc)), "max": float(np.max(shuf_auc))}
                               if shuf_auc else None),
        "n_resample": args.n_resample,
        "stable": bool(np.mean(cos_pairs) > 3 * np.percentile(null, 95)),
    }
    with open(out / f"wres_stability_{res['tap']}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\n" + "=" * 76)
    print(f"w_res STABILITY at {res['tap']}  ({args.n_resample} family resamples)")
    print("=" * 76)
    print(f"  held-out AUC:        mean {res['heldout_auc']['mean']:.3f}  "
          f"min {res['heldout_auc']['min']:.3f}  (decodability robust?)")
    print(f"  w_res stability cos: mean {res['wres_stability_cos']['mean']:.3f}  "
          f"[{res['wres_stability_cos']['p05']:.3f}, {res['wres_stability_cos']['p95']:.3f}]")
    print(f"  random-cos null:     mean {res['random_cos_null']['mean']:.3f}  p95 {res['random_cos_null']['p95']:.3f}")
    print(f"  cos_C(w_res, gbar):  mean {res['cos_C_wres_gbar']['mean']:+.3f}  "
          f"[{res['cos_C_wres_gbar']['min']:+.3f}, {res['cos_C_wres_gbar']['max']:+.3f}]  (orthogonality stable?)")
    if res["shuffled_label_auc"]:
        print(f"  shuffled-label AUC:  mean {res['shuffled_label_auc']['mean']:.3f} (sanity, expect ~0.5)")
    print(f"\n  => w_res {'IS a stable real direction' if res['stable'] else 'is NOT stable (overfit artefact)'} "
          f"(stability {res['wres_stability_cos']['mean']:.2f} vs 3x null p95 {3*res['random_cos_null']['p95']:.2f})")
    print(f"\nwrote: {out}/wres_stability_{res['tap']}.json")
    print("=" * 76)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concept_npz", default="data/analysis/runD_v2/geometry_stage1/concept_directions.npz")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--tap_layer", type=int, default=-1, help="post-block layer; -1 = final pre-norm residual")
    p.add_argument("--n_resample", type=int, default=20)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--shuffle_check", action="store_true", help="also fit shuffled-label probe (expect AUC~0.5)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
