"""
116b_verify_dump_anomalies.py   [post-GO checks on the 119 dump: 4 quick verifications]
=========================================================================================
Pure CPU, seconds. Verifies the two anomalies and two side-checks from the 116/118 runs:

(1) INTACT=0 ANOMALY: baseline_intact = 0.000 although margin-accuracy ~0.72.
    Decode the actual top-1 / top-5 tokens from topk_ids and report their frequency
    table. Expected culprit: tokenization (top-1 is "alpha"/"beta" WITHOUT the leading
    space, or a newline/format token), in which case margins and the flip law are
    unaffected but every intact-style metric needs the corrected answer-token ids.
    Also reports: margin-accuracy among prompts whose top-1 is a no-space variant.

(2) SATURATION AT 0.72: predicted flip-rates plateau at 0.72, not 1.0. Hypothesis:
    the 86 flip definition only counts prompts STARTING on the correct side of the
    margin, so the ceiling = baseline margin accuracy. Verify: fraction of prompts
    with sign-correct clean margin should equal the plateau. Also prints the
    per-class breakdown and the alternative normalized metric (flip among
    baseline-correct) recommended for the cross-concept battery.

(3) d_eff ROBUSTNESS: is d_eff ~ 17 real structure or 1-2 massive-activation
    channels? Recompute d_eff per layer after removing the top-k variance
    dimensions (k = 0, 1, 2, 5, 10) and report the profile.

(4) SPAN-RANK CONSISTENCY: within-span p95 ~ 0.10 should match 1.96/sqrt(rank);
    print rank and the analytic value next to the measured p95 (read from 118's
    CSV if present).

Usage:  python 116b_verify_dump_anomalies.py --dump_dir data/analysis/runD_v2/field_dump \
            [--model_name Qwen/Qwen3-4B] [--null_csv data/analysis/runD_v2/null_calibration/null_calibration_per_layer.csv]
Tokenizer load needs no GPU; if transformers is unavailable, pass --no_decode to skip (1)'s decoding.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("verify116b")


def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def d_eff_from_centered(Hc):
    n = Hc.shape[0]
    G = (Hc @ Hc.T) / n
    tr1 = float(np.trace(G)); tr2 = float((G * G).sum())
    return tr1 * tr1 / (tr2 + 1e-30)


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--null_csv", default="data/analysis/runD_v2/null_calibration/null_calibration_per_layer.csv")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--no_decode", action="store_true")
    p.add_argument("--layers_deff", type=int, nargs="*", default=[0, 8, 16, 24, 35])
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    args = p.parse_args()

    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    y = meta["y"].astype(int)
    m0 = meta["clean_margin"].astype(np.float64)
    topk_ids = meta["topk_ids"]
    alpha_id, beta_id = int(meta["alpha_id"]), int(meta["beta_id"])
    nP = len(y)
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)

    print("\n" + "=" * 92)
    print("(1) INTACT=0 ANOMALY — what is the model actually outputting?")
    print("=" * 92)
    top1 = topk_ids[:, 0]
    print(f"expected answer ids: alpha_id={alpha_id}  beta_id={beta_id}")
    print(f"top-1 == alpha_id/beta_id on {int(np.isin(top1, [alpha_id, beta_id]).sum())}/{nP} prompts")
    if not args.no_decode:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            cnt = Counter(int(t) for t in top1)
            print("top-1 token frequency (top 10):")
            for tid, c in cnt.most_common(10):
                print(f"   id={tid:>7}  {repr(tok.decode([tid])):>14}  x{c}")
            # candidate no-space variants
            cands = {}
            for nm, s in (("alpha_nospace", "alpha"), ("beta_nospace", "beta"),
                          ("Alpha", " Alpha"), ("Beta", " Beta"),
                          ("alpha_caps", "Alpha"), ("beta_caps", "Beta")):
                ids = tok.encode(s, add_special_tokens=False)
                cands[nm] = ids[0] if ids else None
            print("candidate variant ids:", cands)
            var_ids = [v for v in cands.values() if v is not None]
            in_var = np.isin(top1, var_ids)
            print(f"top-1 in no-space/caps variants: {int(in_var.sum())}/{nP}")
            # does the top-1 variant AGREE with the margin sign? (semantic intactness)
            a_like = [cands.get("alpha_nospace"), cands.get("alpha_caps"), cands.get("Alpha")]
            b_like = [cands.get("beta_nospace"), cands.get("beta_caps"), cands.get("Beta")]
            sem = 0; agree = 0
            for i in range(nP):
                t = int(top1[i])
                if t in a_like or t in b_like:
                    sem += 1
                    pred_beta = t in b_like
                    agree += int(pred_beta == (m0[i] > 0))
            if sem:
                print(f"semantic top-1 (any alpha/beta variant): {sem}/{nP}; "
                      f"agrees with margin sign on {agree}/{sem} = {agree/sem:.3f}")
            print("=> if variants dominate: margins/flip-law UNAFFECTED; fix intact metrics by "
                  "passing the variant ids as answer ids (and align with 85/86 conventions).")
        except Exception as e:
            print(f"(tokenizer unavailable: {e}) — rerun with transformers or on the cluster")

    print("\n" + "=" * 92)
    print("(2) SATURATION AT 0.72 — is the ceiling just baseline margin accuracy?")
    print("=" * 92)
    correct = ((y == 1) & (m0 > 0)) | ((y == 0) & (m0 < 0))
    acc = float(correct.mean())
    print(f"baseline margin accuracy (full corpus): {acc:.4f}   <- compare to the 0.72 plateau")
    print(f"  per class: alpha-correct {float(correct[y==0].mean()):.3f} (n={int((y==0).sum())}) | "
          f"beta-correct {float(correct[y==1].mean()):.3f} (n={int((y==1).sum())})")
    print(f"held-out only: {float(correct[~trm].mean()):.4f}")
    print("=> if plateau == accuracy: NOT a bug — the 86 flip definition only counts prompts that")
    print("   start on the correct side. For the B1 battery ADD the normalized metric:")
    print("   flip-rate among baseline-correct prompts (ceiling 1.0, comparable across concepts).")

    print("\n" + "=" * 92)
    print("(3) d_eff ROBUSTNESS — massive-activation channels?")
    print("=" * 92)
    for L in args.layers_deff:
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        Hc = H[trm] - H[trm].mean(0)
        var = Hc.var(0)
        order = np.argsort(-var)
        row = []
        for k in (0, 1, 2, 5, 10):
            keep = np.ones(Hc.shape[1], bool); keep[order[:k]] = False
            row.append(f"k={k}: {d_eff_from_centered(Hc[:, keep]):.0f}")
        top_share = float(var[order[:2]].sum() / var.sum())
        print(f"  L{L:02d}: d_eff after removing top-k variance dims  {' | '.join(row)}   "
              f"(top-2 dims hold {top_share:.1%} of variance)")
    print("=> if d_eff jumps strongly at k=1..2: report both raw and outlier-trimmed d_eff;")
    print("   the within-span and cov nulls remain the operative calibration either way.")

    print("\n" + "=" * 92)
    print("(4) SPAN-RANK CONSISTENCY — does p95 ~ 1.96/sqrt(rank)?")
    print("=" * 92)
    n_train = int(trm.sum())
    print(f"train prompts (span upper bound): {n_train} -> analytic p95 ~ {1.96/np.sqrt(n_train):.4f}")
    if Path(args.null_csv).exists():
        with open(args.null_csv) as f:
            rows = list(_csv.DictReader(f))
        med_p95 = float(np.median([float(r["span_wres_p95"]) for r in rows]))
        ranks = [int(float(r.get("span_rank_wres", n_train))) for r in rows]
        print(f"measured: median span p95 = {med_p95:.4f} | span rank median = {int(np.median(ranks))}")
        print("=> match within ~10% confirms the span null is behaving exactly as theory says.")
    else:
        print(f"({args.null_csv} not found — run 118 first or pass --null_csv)")
    print()


if __name__ == "__main__":
    main()
