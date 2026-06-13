"""
128_attribution_carrier_ablation.py   [does the dictionary realize u through a rare subset?]
=============================================================================================
Methodological note from exp 91: functional carriers must be selected by ATTRIBUTION,
not decoder cosine. Script 111 already defines attribution correctly:
    attr_f = a_f * <W_dec[f], g>,   g = grad(logit_beta - logit_alpha)   (= our <u,.> functional)
and feature_metrics_full.csv flags the attribution-selected features (is_attr=1, "Q1":
selective AND attributing), concentrated in the ignition band L19-24.

This script runs the decisive ablation. At each probe layer, it zero-ablates the Q1
attribution-carrier features (clamp their transcoder activations to 0, add the resulting
decoder-output delta back into the residual) on baseline-correct held prompts, and
measures the behavioral effect:
    d_margin   change in logit(beta)-logit(alpha)
    flip_rate  sign flips (margin / intact)
Compared against a NULL: random sets of features MATCHED on activation frequency and
count (so "ablating any N comparably-active features" is the control, isolating the
attribution selection itself).

Outcome (settles the localization question):
  - Q1 ablation moves the output >> freq-matched null  -> the dictionary REALIZES u
    through a rare attribution-selected subset (partial localization of the used axis).
  - Q1 ablation ~ null (both ~0)                        -> the solution is NOT localizable
    even by attribution: the used axis is distributed beyond any sparse feature set.

Needs transcoders (mwhanna/qwen3-4b-transcoders) + the model. GPU.

SELF-TEST (no torch / no repo):  python 128_attribution_carrier_ablation.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ablate128")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
# =====================================================================
def freq_matched_null_sets(all_feats, fire_rates, target_feats, n_sets, rng, tol=0.05):
    """Sample n_sets random feature sets of the same SIZE as target_feats, each member
    matched in activation frequency (within tol) to a distinct target member.
    Falls back to nearest-frequency sampling if the tolerance band is sparse."""
    fr = dict(zip(all_feats, fire_rates))
    pool = [f for f in all_feats if f not in set(target_feats)]
    pool_fr = np.array([fr[f] for f in pool])
    sets = []
    for _ in range(n_sets):
        chosen = []
        used = set()
        for tf in target_feats:
            tfr = fr[tf]
            cand = [i for i, f in enumerate(pool)
                    if f not in used and abs(pool_fr[i] - tfr) <= tol]
            if not cand:
                order = np.argsort(np.abs(pool_fr - tfr))
                cand = [i for i in order if pool[i] not in used][:50]
            pick = pool[int(rng.choice(cand))]
            chosen.append(pick); used.add(pick)
        sets.append(chosen)
    return sets


def summarize_ablation(d_margins, m0, intact_after):
    """d_margins: per-prompt change in margin under ablation; m0 baseline margins;
    intact_after: per-prompt top-1 still in label set. Returns effect metrics."""
    d_margins = np.asarray(d_margins, float); m0 = np.asarray(m0, float)
    m1 = m0 + d_margins
    flip = ((np.sign(m1) != np.sign(m0))).mean()
    intact_flip = (((np.sign(m1) != np.sign(m0))) & np.asarray(intact_after, bool)).mean()
    return {"mean_abs_dmargin": float(np.mean(np.abs(d_margins))),
            "mean_dmargin": float(np.mean(d_margins)),
            "flip_rate": float(flip),
            "intact_flip_rate": float(intact_flip)}


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    all_feats = list(range(200))
    fire = rng.uniform(0, 1, 200)
    target = [5, 50, 120]
    sets = freq_matched_null_sets(all_feats, fire, target, n_sets=20, rng=rng, tol=0.05)
    assert all(len(s) == 3 for s in sets), "null sets keep the target size"
    assert all(not (set(s) & set(target)) for s in sets), "null excludes the targets"
    # frequency matching: each null member close to a target member's frequency
    fr = dict(zip(all_feats, fire))
    for s in sets[:5]:
        diffs = sorted(abs(fr[a] - fr[b]) for a in s for b in target)
        assert diffs[0] < 0.12, "at least one close frequency match per set"

    # ablation summary: a strong negative dmargin on correct-beta prompts -> flips
    m0 = np.array([1.0, 1.5, -1.0, -2.0, 0.5])           # mixed baseline margins
    d = np.array([-2.0, -0.2, -0.1, 0.1, -1.0])          # ablation pushes margin down
    intact = [1, 1, 1, 0, 1]
    s = summarize_ablation(d, m0, intact)
    # prompts 0 (1.0->-1.0) and 4 (0.5->-0.5) cross zero -> 2/5 flip
    assert abs(s["flip_rate"] - 2 / 5) < 1e-9, f"flip rate wrong: {s['flip_rate']}"
    assert abs(s["intact_flip_rate"] - 2 / 5) < 1e-9, "both flips intact here"
    assert s["mean_abs_dmargin"] > 0
    print("[self_test] OK — freq-matched null sets, ablation summary (flip/intact) pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(args.repo_root)))
    from transcoder_loader import load_transcoder_set  # repo helper

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- features: Q1 attribution carriers per layer ----
    feat_rows = list(_csv.DictReader(open(args.feature_metrics)))
    q1_by_layer = defaultdict(list)
    fire_by_layer = defaultdict(dict)
    for r in feat_rows:
        L = int(r["layer"]); f = int(r["feature"]); fr = float(r["fire_rate"])
        fire_by_layer[L][f] = fr
        if r.get("is_attr") == "1":
            q1_by_layer[L].append(f)
    layers = [L for L in args.layers if q1_by_layer.get(L)]
    logger.info("Q1 carriers per probe layer: %s", {L: len(q1_by_layer[L]) for L in layers})

    # ---- data ----
    prompts = [json.loads(l) for l in open(args.corpus)]
    fams = [p["surface_family"] for p in prompts]
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)
    meta = np.load(Path(args.dump_dir) / "meta.npz", allow_pickle=True)
    y = meta["y"].astype(int); m0_all = meta["clean_margin"].astype(np.float64)
    id0 = meta["id_class0"].astype(int) if "id_class0" in meta else np.full(len(y), int(meta["alpha_id"]))
    id1 = meta["id_class1"].astype(int) if "id_class1" in meta else np.full(len(y), int(meta["beta_id"]))
    correct = ((y == 1) & (m0_all > 0)) | ((y == 0) & (m0_all < 0))
    held = np.where(~trm)[0]
    targets = [int(i) for i in held if correct[i]][: args.max_targets or None]
    logger.info("targets: %d baseline-correct held prompts", len(targets))

    # ---- model + transcoders ----
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers")
    mlp_of = lambda L: blocks[L].mlp

    def ablate_eval(i, L, feats):
        """Run with the transcoder at layer L; zero the given features; add the decoder
        delta (recon with features zeroed - recon full) into the MLP output at the
        decision position. Returns margin and intact. The transcoder reconstructs the
        MLP output from the MLP input, so decode takes input_acts (skip connection)."""
        tc = TCSET[L]
        inp = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        feats_t = torch.tensor(sorted(feats), dtype=torch.long, device=args.device)

        def hook(mod, mlp_in, mlp_out):
            x = mlp_in[0]                                      # (B, T, d) MLP input
            acts = tc.encode(x)                                # (B, T, d_tc)
            recon_full = tc.decode(acts, x)
            if len(feats_t) > 0:
                acts_ab = acts.clone()
                acts_ab[..., feats_t] = 0.0
                recon_ab = tc.decode(acts_ab, x)
                delta = recon_ab - recon_full
            else:
                delta = torch.zeros_like(recon_full)
            return mlp_out + delta

        h = mlp_of(L).register_forward_hook(hook)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
            margin = float(row[int(id1[i])] - row[int(id0[i])])
            intact = int(int(row.argmax()) in (int(id0[i]), int(id1[i])))
        finally:
            h.remove()
        return margin, intact

    TCSET = load_transcoder_set("4b", repo_id=args.transcoder_repo, device=args.device,
                                dtype=torch.float32, layers=layers)
    rows = []
    for L in layers:
        q1 = q1_by_layer[L]
        # baseline margins under the transcoder (ablate empty set) to isolate ablation delta
        base_m, _ = zip(*[ablate_eval(i, L, []) for i in targets])
        base_m = np.array(base_m)

        # Q1 ablation
        q1_m, q1_it = zip(*[ablate_eval(i, L, q1) for i in targets])
        d_q1 = np.array(q1_m) - base_m
        s_q1 = summarize_ablation(d_q1, base_m, q1_it)

        # frequency-matched null
        all_f = list(fire_by_layer[L].keys())
        all_fr = [fire_by_layer[L][f] for f in all_f]
        null_sets = freq_matched_null_sets(all_f, all_fr, q1, args.n_null, rng, args.freq_tol)
        null_stats = []
        for ns in null_sets:
            nm, nit = zip(*[ablate_eval(i, L, ns) for i in targets])
            null_stats.append(summarize_ablation(np.array(nm) - base_m, base_m, nit))
        def nq(key): return float(np.quantile([s[key] for s in null_stats], 0.95))
        def nm_(key): return float(np.mean([s[key] for s in null_stats]))

        row = {"layer": L, "n_q1": len(q1), "n_targets": len(targets),
               "q1_abs_dmargin": s_q1["mean_abs_dmargin"], "q1_flip": s_q1["flip_rate"],
               "q1_intact_flip": s_q1["intact_flip_rate"],
               "null_abs_dmargin_mean": nm_("mean_abs_dmargin"), "null_abs_dmargin_p95": nq("mean_abs_dmargin"),
               "null_flip_mean": nm_("flip_rate"), "null_flip_p95": nq("flip_rate"),
               "q1_above_null_dmargin": int(s_q1["mean_abs_dmargin"] > nq("mean_abs_dmargin")),
               "q1_above_null_flip": int(s_q1["flip_rate"] > nq("flip_rate"))}
        rows.append(row)
        logger.info("L%02d: Q1 |dm|=%.4f flip=%.3f | null |dm| mean/p95=%.4f/%.4f flip p95=%.3f | above: dm=%s flip=%s",
                    L, row["q1_abs_dmargin"], row["q1_flip"], row["null_abs_dmargin_mean"],
                    row["null_abs_dmargin_p95"], row["null_flip_p95"],
                    bool(row["q1_above_null_dmargin"]), bool(row["q1_above_null_flip"]))

    with open(out / "attribution_carrier_ablation.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    any_above = [r["layer"] for r in rows if r["q1_above_null_dmargin"]]
    print("\n" + "=" * 92)
    print("ATTRIBUTION-CARRIER ABLATION — is the used axis localizable in a sparse feature set?")
    print("=" * 92)
    for r in rows:
        print(f"  L{r['layer']:02d}: Q1({r['n_q1']}) |dmargin|={r['q1_abs_dmargin']:.4f} flip={r['q1_flip']:.3f} "
              f"| null p95 |dmargin|={r['null_abs_dmargin_p95']:.4f} flip={r['null_flip_p95']:.3f} "
              f"{'ABOVE NULL' if r['q1_above_null_dmargin'] else 'at null'}")
    print(f"\nlayers where Q1 ablation exceeds the freq-matched null (dmargin): {any_above if any_above else 'NONE'}")
    print("INTERPRETATION:")
    print("  - ABOVE NULL somewhere -> the dictionary realizes the used axis through a rare")
    print("    attribution-selected subset: PARTIAL localization of u in feature space.")
    print("  - NONE (Q1 ~ null, both near 0) -> the used axis is NOT localizable even by")
    print("    attribution: distributed beyond any sparse feature set. Final negative point.")
    print(f"saved: {out/'attribution_carrier_ablation.csv'}")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/B1_alpha_beta.jsonl")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--feature_metrics", default="data/analysis/feature_metrics_full.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/carrier_ablation")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--transcoder_repo", default="mwhanna/qwen3-4b-transcoders")
    p.add_argument("--layers", type=int, nargs="*", default=[19, 20, 21, 22, 23, 24])
    p.add_argument("--n_null", type=int, default=20)
    p.add_argument("--freq_tol", type=float, default=0.05)
    p.add_argument("--max_targets", type=int, default=0)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
