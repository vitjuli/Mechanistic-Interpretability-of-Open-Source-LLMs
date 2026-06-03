"""
67_carrier_ablation_full.py   [JOB 3 / 4]
===================================================================
Full version of 64's D2. SFR=1.000 already proved the carrier is causally
NECESSARY on 24 hand-picked beta-mechanism pairs. The open question is BREADTH:
is the carrier broadly causal for alpha/beta, or causal only on that narrow
subset? 64's D2 (capped at 100 prompts, single aggregate) put the carrier at the
68th percentile of a random-37-feature null -- i.e. NOT clearly above random.
This script does it properly: ALL prompts, PER-LAYER breakdown, bootstrap CI on
both the carrier effect and the random-null band.

ABLATION METHOD (stated for the thesis): decoder-direction removal. For each
chosen feature we project its decoder direction d_f out of the residual stream at
the feature's layer (orthogonal projection-out at the decision token), then read
the change in Delta logit = logit(beta) - logit(alpha). Carrier and random sets
use the IDENTICAL method and the SAME per-layer feature counts, so the comparison
is method-internally valid regardless of ablation flavour. (This is a removal of
the WRITE direction, consistent with 62/63/65; it is not the full transcoder
reconstruction-swap, which is a different, larger intervention.)

READS OUT (per layer-set and pooled):
  * mean |Delta logit| under carrier ablation, with bootstrap CI
  * random-37 null band (n_random sets), with percentiles
  * carrier percentile within the null  (>=95% => broadly causal; ~50-70% => not)
  * fraction of prompts whose answer sign flips under carrier ablation (necessity)

VERDICT:
  BROADLY CAUSAL   carrier effect above null p95  -> carrier is a general mechanism
  NARROWLY CAUSAL  carrier effect within null band -> causal only on SFR subset;
                   co-importance selected correlates, not the general mechanism.

SELF-TEST: python 67_carrier_ablation_full.py --self_test
"""

from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("carrier_ablation")
FID = re.compile(r"^[Ll](\d+)[_:\-][Ff]?(\d+)$")


def bootstrap_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def parse_fid(f):
    m = FID.match(str(f).strip())
    if not m:
        raise ValueError(f"bad feature_id {f!r}")
    return int(m.group(1)), int(m.group(2))


def self_test():
    # Logic test for the ablation/comparison bookkeeping (no torch): a "causal"
    # carrier removes more logit-margin than random sets; a "correlate" carrier ~ random.
    rng = np.random.default_rng(67)
    n = 120
    # simulate |Δlogit| effects
    carrier_causal = np.abs(rng.normal(0.8, 0.2, n))     # large effect
    carrier_corr = np.abs(rng.normal(0.25, 0.1, n))      # ~ random
    def null_means(center, k=50):
        return np.array([np.abs(rng.normal(center, 0.1, n)).mean() for _ in range(k)])
    null = null_means(0.25)
    pct_causal = float((null < carrier_causal.mean()).mean())
    pct_corr = float((null < carrier_corr.mean()).mean())
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  causal carrier:   mean|Δ|={carrier_causal.mean():.3f}  percentile vs null={100*pct_causal:.0f}%")
    print(f"  correlate carrier:mean|Δ|={carrier_corr.mean():.3f}  percentile vs null={100*pct_corr:.0f}%")
    m, lo, hi = bootstrap_ci(carrier_causal, seed=1)
    print(f"  bootstrap CI (causal): {m:.3f} [{lo:.3f}, {hi:.3f}]")
    assert pct_causal > 0.95, "causal carrier must exceed null p95"
    assert pct_corr < 0.9, "correlate carrier must sit within null band"
    assert lo < m < hi, "bootstrap CI must bracket the mean"
    print("\nALL SELF-TEST ASSERTIONS PASSED \n--------------------------------------------\n")


def resolve(args):
    feats = {}
    def add(L, fi): feats.setdefault(L, []).append(fi)
    if args.features:
        for t in args.features.split(","):
            L, fi = parse_fid(t); add(L, fi)
    elif args.feature_file:
        for ln in open(args.feature_file):
            ln = ln.strip()
            if ln:
                L, fi = parse_fid(ln); add(L, fi)
    elif args.cluster_labels:
        import pandas as pd
        cl = pd.read_csv(args.cluster_labels)
        if "feature_id" not in cl.columns:
            cl = cl.rename(columns={cl.columns[0]: "feature_id"})
        def nid(c):
            s = str(c).strip().lstrip("Cc"); return s[:-2] if s.endswith(".0") else s
        want = {nid(c) for c in args.clusters.split(",")} if args.clusters else None
        for fid, c in zip(cl["feature_id"].astype(str), cl[args.cluster_col].apply(nid)):
            if want is None or c in want:
                try:
                    L, fi = parse_fid(fid)
                except ValueError:
                    continue
                if args.layers and L not in args.layers:
                    continue
                add(L, fi)
    else:
        raise SystemExit("need --features/--feature_file/--cluster_labels")
    return feats


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = model.model.layers
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    feats = resolve(args); layers = sorted(feats)
    from src.transcoder import load_transcoder_set
    tset = load_transcoder_set(model_size=args.model_size, device=args.device, lazy_load=True, layers=layers)

    # W_dec cache (eliminates slow _get_decoder_vectors per-call)
    W_dec_cache_t: Dict[int, "torch.Tensor"] = {}
    def _ensure_wdec_t(L):
        if L not in W_dec_cache_t:
            tc = tset[L]
            if not hasattr(tc, "W_dec"):
                raise RuntimeError(f"transcoder L{L} has no W_dec")
            logger.info("Caching W_dec for layer %d (shape=%s)...", L, tc.W_dec.shape)
            W_dec_cache_t[L] = tc.W_dec.detach().float().to(args.device)
        return W_dec_cache_t[L]

    def rows(L, idx):
        W = _ensure_wdec_t(L)
        return W[torch.as_tensor(idx, dtype=torch.long, device=args.device)]

    def dl_with_ablation(ptext, abl: Dict[int, List[int]]):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = model(**inp, use_cache=False)
        lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
        dlc = float(lp[beta_id] - lp[alpha_id])
        handles = []
        for L, idx in abl.items():
            R = rows(L, idx)
            def mk(R=R, L=L):
                def pre(m, a):
                    hs = a[0].clone(); h = hs[0, -1, :]
                    for r in R:
                        h = h - (torch.dot(h, r) / (torch.dot(r, r) + 1e-8)) * r
                    hs[0, -1, :] = h; return (hs,)
                return pre
            handles.append(blocks[L].register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                o2 = model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), 0)
        dla = float(lp2[beta_id] - lp2[alpha_id])
        return dlc, dla

    # layer-sets to evaluate: each carrier layer alone + all carrier layers together
    layer_sets = {f"L{L}": {L: feats[L]} for L in layers}
    layer_sets["ALL"] = {L: feats[L] for L in layers}

    def rand_set(template: Dict[int, List[int]]):
        a = {}
        for L, idx in template.items():
            nf = tset[L].W_dec.shape[0] if hasattr(tset[L], "W_dec") else args.d_tc
            a[L] = [int(rng.integers(nf)) for _ in idx]
        return a

    results = {"ablation": "decoder-direction projection-out", "sets": {}}
    for sname, abl in layer_sets.items():
        c_eff, c_flip = [], []
        for p in prompts:
            dlc, dla = dl_with_ablation(p["prompt"], abl)
            c_eff.append(abs(dlc - dla))
            tgt = 1 if p["correct_answer"].strip() == "beta" else 0
            sign_clean = 1 if dlc > 0 else 0
            sign_abl = 1 if dla > 0 else 0
            c_flip.append(int(sign_clean != sign_abl))
        cm, clo, chi = bootstrap_ci(c_eff, seed=args.seed)
        null_means = []
        logger.info("  null: %d random sets × %d prompts...", args.n_random, len(prompts))
        for s in range(args.n_random):
            if s > 0 and s % 10 == 0:
                logger.info("    random set %d/%d", s, args.n_random)
            a = rand_set(abl)
            # FIX: single call per prompt (was 2 calls — double cost)
            eff = []
            for p in prompts:
                dlc_r, dla_r = dl_with_ablation(p["prompt"], a)
                eff.append(abs(dlc_r - dla_r))
            null_means.append(float(np.mean(eff)))
        null = np.array(null_means)
        pct = float((null < cm).mean())
        results["sets"][sname] = {
            "carrier_mean_abs_dlogit": cm, "carrier_CI": [clo, chi],
            "carrier_flip_frac": float(np.mean(c_flip)),
            "null_band": {"mean": float(null.mean()), "p05": float(np.percentile(null, 5)),
                          "p95": float(np.percentile(null, 95))},
            "carrier_percentile_vs_null": pct,
            "broadly_causal": bool(cm > np.percentile(null, 95)),
        }
        logger.info("%4s: carrier |Δ|=%.4f [%.4f,%.4f] flip=%.2f  null mean=%.4f p95=%.4f  pct=%.0f%%  %s",
                    sname, cm, clo, chi, np.mean(c_flip), null.mean(), np.percentile(null, 95),
                    100 * pct, "BROAD" if cm > np.percentile(null, 95) else "narrow")

    all_set = results["sets"]["ALL"]
    verdict = ("BROADLY CAUSAL: carrier ablation exceeds the random-37 null p95 -> the carrier is a "
               "general mechanism for alpha/beta." if all_set["broadly_causal"]
               else f"NARROWLY CAUSAL: carrier ablation sits within the random null "
                    f"(percentile {100*all_set['carrier_percentile_vs_null']:.0f}%) -> causal only on the "
                    "SFR subset; co-importance selected correlates, not the general mechanism.")
    results["verdict"] = verdict

    with open(out / "carrier_ablation_full.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\n" + "=" * 80)
    print("CARRIER ABLATION (full) -- causal breadth carrier vs random")
    print("=" * 80)
    print(f"{'set':>6} {'carrier|Δ|':>11} {'CI':>20} {'flip':>6} {'null_p95':>9} {'pct':>6}")
    for s, r in results["sets"].items():
        ci = f"[{r['carrier_CI'][0]:.3f},{r['carrier_CI'][1]:.3f}]"
        print(f"{s:>6} {r['carrier_mean_abs_dlogit']:>11.4f} {ci:>20} {r['carrier_flip_frac']:>6.2f} "
              f"{r['null_band']['p95']:>9.4f} {100*r['carrier_percentile_vs_null']:>5.0f}%")
    print("\nVERDICT: " + verdict)
    print(f"\nwrote: {out}/carrier_ablation_full.json")
    print("=" * 80)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/iia_failure_diagnosis")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--model_size", default="4b", choices=["0.6b", "1.7b", "4b", "8b", "14b"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--features", default=None)
    p.add_argument("--feature_file", default=None)
    p.add_argument("--cluster_labels", default=None)
    p.add_argument("--cluster_col", default="coimp_louvain")
    p.add_argument("--clusters", default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--n_random", type=int, default=50)
    p.add_argument("--d_tc", type=int, default=163840)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
