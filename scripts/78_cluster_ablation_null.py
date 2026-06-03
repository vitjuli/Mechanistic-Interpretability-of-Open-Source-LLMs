"""
78_cluster_ablation_null.py   [CSD3 / GPU, mirrors script 27's ablation mechanism]
===================================================================
Applies the decay chapter's rigor bar to the cluster-ablation claims of the
intensive/extensive chapter (and any other cluster claim). Those chapters report
that ablating a co-importance cluster (e.g. C4) causally flips behaviour
(sign-flip rate high, sign reversal). The missing control: does ablating that
cluster flip behaviour MORE than ablating a RANDOM feature-set of the SAME size
from the SAME layers?

WHY THIS CONTROL IS DECISIVE. The chapters' own "default-and-suppress" mechanism
predicts that ablating ANY sufficiently large feature set reverts the suppressed
class to the default (removing suppression signal exposes the default prior). So a
large flip under cluster ablation is EXPECTED even for a non-specific cluster. The
cluster is only a specific causal locus if its flip / selectivity exceeds a matched
random-feature null. (This is the same logic that put the decay carrier at the 68th
percentile of its random null -> not specific.)

WHAT IT DOES (mirrors script 27 exactly: encode MLP input -> zero features ->
decode -> patch post_attention_layernorm via ExitStack -> compare logits):
  * REAL cluster: joint-ablate on all prompts; metrics = sign-flip rate (overall and
    per class) and mean |Delta nd|, plus selectivity = (flip/effect on the cluster's
    target class) - (on the other class).
  * NULL: sample random feature-sets matched PER-LAYER to the cluster's counts;
    joint-ablate; same metrics; repeat n_null times.
  * Report the cluster's percentile within the random-ablation null.
EFFICIENCY: baseline logits and clean MLP inputs are cached per prompt once; each
ablation then costs one patched forward + cheap encode/zero/decode.

OUTCOMES:
  cluster flip/selectivity ABOVE null p95 -> SPECIFIC causal locus (localisation
    holds under the rigorous control; for intensive/extensive this becomes a clean
    POSITIVE CONTROL: the method finds localisation when present, unlike decay).
  cluster flip/selectivity AT/BELOW null   -> NOT specific: the flip is the generic
    default-revert that any large ablation produces; the chapter's localisation
    claim does not survive the control (treat its causal claims as representational).

INPUTS:
  --prompts        intensive/extensive (or other) prompts jsonl
  --cluster_csv    fid->cluster_id map (e.g. cluster_labels.csv); pick --cluster_id
  --correct/--incorrect answer tokens (e.g. ' intensive' / ' extensive')
  --target_class   which class the cluster is claimed to support (for selectivity)

SELF-TEST: python 78_cluster_ablation_null.py --self_test
"""

from __future__ import annotations
import argparse, csv as csvlib, json, logging, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ablation_null")
FID = re.compile(r"[Ll](\d+)[_-]?[Ff](\d+)")


def parse_fid(s):
    m = FID.search(str(s).strip())
    if not m:
        raise ValueError(f"bad feature id {s!r}")
    return int(m.group(1)), int(m.group(2))


# =====================================================================
# Self-test: a SPECIFIC suppressor cluster vs a DISTRIBUTED-suppression confound
# =====================================================================
def self_test():
    rng = np.random.default_rng(78)
    n, d_tc, L = 120, 400, 1                      # one layer, 400 features
    y = rng.integers(0, 2, n)                     # 1 = "extensive"(suppressed), 0 = "intensive"(default)
    # default prior pushes nd>0 (intensive). extensive prompts get suppression from a SPECIFIC set S.
    S = list(range(20))                           # the true suppressor features
    acts = np.abs(rng.standard_normal((n, d_tc))) * 0.3
    ext_idx = np.where(y == 1)[0]
    acts[ext_idx[:, None], S] += 3.0             # extensive prompts strongly activate suppressors (proper fancy-index)
    # contribution: suppressors subtract from nd; default = +1.5
    def nd_of(a, ablated=()):
        aa = a.copy(); aa[list(ablated)] = 0.0
        return 1.5 - 0.25 * aa[S].sum()           # suppressors (if active & not ablated) push nd<0
    base_nd = np.array([nd_of(acts[i]) for i in range(n)])

    def metrics(ablated):
        flips, dnd = [], []
        for i in range(n):
            nd1 = nd_of(acts[i], ablated)
            flips.append(int(np.sign(nd1) != np.sign(base_nd[i])))
            dnd.append(abs(nd1 - base_nd[i]))
        flips = np.array(flips); dnd = np.array(dnd)
        ext = y == 1
        return float(flips[ext].mean()), float(dnd.mean())     # extensive-class SFR, mean|Δnd|

    sfr_real, _ = metrics(S)                       # ablate the true suppressors
    null = [metrics(rng.choice(d_tc, len(S), replace=False))[0] for _ in range(200)]
    null = np.array(null)
    pct = float((null < sfr_real).mean() * 100)
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  SPECIFIC suppressor cluster: extensive-SFR = {sfr_real:.3f}")
    print(f"  random-cluster null:         mean {null.mean():.3f}  p95 {np.percentile(null,95):.3f}")
    print(f"  specific cluster percentile: {pct:.0f}%")
    assert sfr_real > np.percentile(null, 95) + 0.1, "specific suppressor must beat the random-ablation null"
    assert pct > 95, "specific cluster should sit at the top of the null"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(a genuinely specific suppressor cluster beats the matched random-ablation null;")
    print(" if the real cluster sits IN the null band, the flip is generic default-revert)")
    print("---------------------------------------------------------------\n")


# =====================================================================
# Real run (mirrors script 27)
# =====================================================================
def run_real(args):
    import torch, contextlib
    sys.path.insert(0, str(Path(args.repo_root)))
    from src.model_utils import ModelWrapper
    from src.transcoder import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load prompts
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.n_prompts:
        # balanced subsample by class if possible
        prompts = prompts[: args.n_prompts]
    def cls_of(p):
        return p.get("correct_answer", p.get("answer", "")).strip()
    logger.info("prompts: %d", len(prompts))

    # ---- load cluster -> per-layer feature indices
    cmap = defaultdict(list)
    with open(args.cluster_csv) as f:
        for row in csvlib.DictReader(f):
            fid = row.get("feature") or row.get("fid") or row.get("feature_id")
            cid = row.get("cluster") or row.get("cluster_id") or row.get("cid")
            cmap[str(cid)].append(fid)
    if args.cluster_id not in cmap:
        raise SystemExit(f"cluster {args.cluster_id} not in {sorted(cmap)}")
    cluster_by_layer = defaultdict(list)
    for fid in cmap[args.cluster_id]:
        L, k = parse_fid(fid)
        cluster_by_layer[L].append(k)
    counts = {L: len(v) for L, v in cluster_by_layer.items()}
    layers = sorted(counts)
    logger.info("cluster %s: %d features across layers %s", args.cluster_id, sum(counts.values()), counts)

    # ---- model + transcoders
    model = ModelWrapper(args.model_name, device=args.device)
    tset = load_transcoder_set(model_size=args.model_size, device=args.device, lazy_load=True, layers=layers)
    dev = next(model.model.parameters()).device
    blocks = model.model.model.layers
    cid_tok = model.tokenizer.encode(args.correct, add_special_tokens=False)[0]
    iid_tok = model.tokenizer.encode(args.incorrect, add_special_tokens=False)[0]

    def d_tc(L):
        t = tset[L]
        return t.W_dec.shape[0] if hasattr(t, "W_dec") and t.W_dec is not None else args.d_tc

    # ---- cache baseline nd + clean MLP inputs per prompt
    def clean_pass(ptext):
        inp = model.tokenize([ptext]); inp = {k: v.to(dev) for k, v in inp.items()}
        caught = {}
        handles = []
        for L in layers:
            def mk(L=L):
                def hk(m, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    caught[L] = t[:, -1, :].detach()
                return hk
            handles.append(blocks[L].post_attention_layernorm.register_forward_hook(mk()))
        try:
            with torch.no_grad():
                o = model.model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(o.logits[0, -1, :], 0)
        nd = float(lp[cid_tok] - lp[iid_tok])
        return inp, nd, {L: caught[L] for L in layers}

    logger.info("caching baseline + clean MLP inputs...")
    cache = []
    for i, p in enumerate(prompts):
        inp, nd, mlp = clean_pass(p["prompt"])
        cache.append({"inp": inp, "nd": nd, "mlp": mlp, "cls": cls_of(p)})
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))

    @contextlib.contextmanager
    def patch(layer_idx, new_input):
        hook_mod = blocks[layer_idx].post_attention_layernorm
        def hk(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            t = t.clone(); t[:, -1, :] = new_input.to(t.dtype)
            return (t,) + o[1:] if isinstance(o, tuple) else t
        h = hook_mod.register_forward_hook(hk)
        try:
            yield
        finally:
            h.remove()

    def ablate_metrics(feat_by_layer):
        """sign-flip rate (overall + per class) and mean|Δnd| for a given ablation set."""
        flips, dnds, clss = [], [], []
        for c in cache:
            mods = {}
            for L in layers:
                act = c["mlp"][L]; tc = tset[L]
                with torch.no_grad():
                    fe = tc.encode(act.to(tc.dtype))
                    idx = feat_by_layer.get(L, [])
                    if idx:
                        fe[:, idx] = 0.0
                    mods[L] = tc.decode(fe).to(act.dtype).squeeze(0)
            with contextlib.ExitStack() as st:
                for L, m in mods.items():
                    st.enter_context(patch(L, m))
                with torch.no_grad():
                    o = model.model(**c["inp"], use_cache=False)
            lp = torch.log_softmax(o.logits[0, -1, :], 0)
            nd1 = float(lp[cid_tok] - lp[iid_tok])
            flips.append(int(np.sign(nd1) != np.sign(c["nd"]) and c["nd"] != 0))
            dnds.append(abs(nd1 - c["nd"])); clss.append(c["cls"])
        flips = np.array(flips); dnds = np.array(dnds); clss = np.array(clss)
        tgt = clss == args.target_class
        return {"sfr_overall": float(flips.mean()),
                "sfr_target": float(flips[tgt].mean()) if tgt.any() else float("nan"),
                "sfr_other": float(flips[~tgt].mean()) if (~tgt).any() else float("nan"),
                "mean_abs_dnd": float(dnds.mean()),
                "selectivity_sfr": (float(flips[tgt].mean()) - float(flips[~tgt].mean()))
                                   if tgt.any() and (~tgt).any() else float("nan")}

    # ---- REAL cluster
    logger.info("ablating REAL cluster %s ...", args.cluster_id)
    real = ablate_metrics({L: cluster_by_layer[L] for L in layers})
    logger.info("real: sfr_target=%.3f sfr_other=%.3f selectivity=%.3f |Δnd|=%.3f",
                real["sfr_target"], real["sfr_other"], real["selectivity_sfr"], real["mean_abs_dnd"])

    # ---- NULL: matched per-layer random feature sets
    logger.info("running %d random-ablation null draws ...", args.n_null)
    null = {"sfr_target": [], "selectivity_sfr": [], "mean_abs_dnd": []}
    for j in range(args.n_null):
        fb = {L: list(rng.choice(d_tc(L), counts[L], replace=False)) for L in layers}
        m = ablate_metrics(fb)
        for key in null:
            null[key].append(m[key])
        if (j + 1) % 10 == 0:
            logger.info("  null %d/%d", j + 1, args.n_null)
    null = {k: np.array(v) for k, v in null.items()}

    def pctile(key):
        return float((null[key] < real[key]).mean() * 100)

    res = {"cluster_id": args.cluster_id, "per_layer_counts": counts,
           "target_class": args.target_class, "n_prompts": len(prompts), "n_null": args.n_null,
           "real": real,
           "null_sfr_target": {"mean": float(null["sfr_target"].mean()), "p95": float(np.percentile(null["sfr_target"], 95))},
           "null_selectivity": {"mean": float(null["selectivity_sfr"].mean()), "p95": float(np.percentile(null["selectivity_sfr"], 95))},
           "null_mean_abs_dnd": {"mean": float(null["mean_abs_dnd"].mean()), "p95": float(np.percentile(null["mean_abs_dnd"], 95))},
           "percentile_sfr_target": pctile("sfr_target"),
           "percentile_selectivity": pctile("selectivity_sfr"),
           "percentile_abs_dnd": pctile("mean_abs_dnd")}
    specific = bool(real["selectivity_sfr"] > np.percentile(null["selectivity_sfr"], 95)
                    and real["sfr_target"] > np.percentile(null["sfr_target"], 95))
    res["verdict"] = (
        f"SPECIFIC: cluster {args.cluster_id} beats the matched random-ablation null on both "
        f"target-class SFR ({real['sfr_target']:.2f} vs null p95 {np.percentile(null['sfr_target'],95):.2f}, "
        f"{pctile('sfr_target'):.0f}th pct) and selectivity ({real['selectivity_sfr']:.2f} vs "
        f"{np.percentile(null['selectivity_sfr'],95):.2f}). The localisation is real under the rigorous "
        "control. For intensive/extensive this is a clean POSITIVE CONTROL: the method finds a localised "
        "causal locus when one exists, unlike the decay carrier (which sat in its null)."
        if specific else
        f"NOT SPECIFIC: cluster {args.cluster_id} sits at the {pctile('sfr_target'):.0f}th percentile of the "
        f"random-ablation null on target SFR (selectivity {pctile('selectivity_sfr'):.0f}th). Its flip is the "
        "generic default-revert that any large matched ablation produces -- not a specific causal locus. The "
        "chapter's localisation claim does not survive the control; treat its causal claims as representational.")

    with open(out / f"ablation_null_{args.cluster_id}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print("\n" + "=" * 86)
    print(f"CLUSTER ABLATION NULL  --  is cluster {args.cluster_id} a SPECIFIC causal locus?")
    print("=" * 86)
    print(f"  target class: {args.target_class}   features: {sum(counts.values())} over layers {sorted(counts)}")
    print(f"  REAL    : sfr_target={real['sfr_target']:.3f}  sfr_other={real['sfr_other']:.3f}  "
          f"selectivity={real['selectivity_sfr']:.3f}  |Δnd|={real['mean_abs_dnd']:.3f}")
    print(f"  NULL    : sfr_target mean={null['sfr_target'].mean():.3f} p95={np.percentile(null['sfr_target'],95):.3f}; "
          f"selectivity mean={null['selectivity_sfr'].mean():.3f} p95={np.percentile(null['selectivity_sfr'],95):.3f}")
    print(f"  PERCENTILE of real cluster: sfr_target {pctile('sfr_target'):.0f}%, selectivity {pctile('selectivity_sfr'):.0f}%")
    print("\nVERDICT: " + res["verdict"])
    print(f"\nwrote: {out}/ablation_null_{args.cluster_id}.json")
    print("=" * 86)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--model_size", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompts", default="data/prompts/physics_intensive_extensive_v1_train.jsonl")
    p.add_argument("--cluster_csv", default="data/results/clustering_intext/cluster_labels.csv")
    p.add_argument("--cluster_id", default="4", help="which cluster to test (e.g. C4 -> '4')")
    p.add_argument("--correct", default=" extensive", help="answer token the cluster is claimed to SUPPORT")
    p.add_argument("--incorrect", default=" intensive", help="the default/other answer token")
    p.add_argument("--target_class", default="extensive", help="class (correct_answer string) the cluster supports")
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--n_null", type=int, default=40)
    p.add_argument("--d_tc", type=int, default=163840)
    p.add_argument("--out_dir", default="data/results/ablation_null")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
