"""
Script 37b: IIA validation extensions — saves per-pair flip data AND supports
combined multi-cluster patching.

Two modes:
  (A) --mode per_pair: Runs IIA on specified target_clusters, saves PER-PAIR
      flip flags so we can compute overlap between sub-detectors (e.g. do all
      4 L24 sub-clusters flip the same pairs?).
  (B) --mode combined: Runs IIA where the "cluster" is the UNION of features
      from multiple target_clusters (e.g. L24 sub-clusters + L18 together).
      Tests whether combined patching is more potent than individual.

Output (per_pair mode):
  iia_per_pair_flips.csv — rows: (cluster, pair_id, direction, baseline_margin,
                                  patched_margin, flipped, pa_idx, pb_idx)
  iia_overlap_summary.csv — pairwise Jaccard overlap of flipped-pair sets
                            between clusters

Output (combined mode):
  iia_combined_patches.csv — rows: (combo_name, n_features, n_flips, iia)

Reuses the IIA machinery from script 37 (imports its helpers).

Usage:
  # Mode A: L24 sub-cluster overlap
  python3 scripts/37b_iia_validation.py \\
      --mode per_pair \\
      --target_clusters "16,18,19,20,35" \\
      --cluster_col agglo_coimp_subgroup_k30 \\
      --prompt_metadata data/analysis/runD_v2/grouping/prompt_metadata.csv \\
      --clustering_dir data/analysis/runD_v2/clustering_full \\
      --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp

  # Mode B: combined L24+L18 patching
  python3 scripts/37b_iia_validation.py \\
      --mode combined \\
      --combinations "16+35,16+18+19+20,16+18+19+20+35" \\
      --cluster_col agglo_coimp_subgroup_k30 \\
      ...
"""
import argparse, json, logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

# Reuse helpers from script 37
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
m37 = import_module("37_iia_probe_clusters")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["per_pair","combined"], required=True)
    parser.add_argument("--target_clusters", type=str, default="",
                        help="Mode A: comma-separated cluster IDs (e.g. '16,18,19,20')")
    parser.add_argument("--combinations", type=str, default="",
                        help="Mode B: semicolon-separated combos, each '+' joined "
                             "(e.g. '16+35;16+18+19+20')")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cluster_col", type=str, default="agglo_coimp_subgroup_k30")
    parser.add_argument("--prompt_metadata", type=str, required=True)
    parser.add_argument("--clustering_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    clustering_dir = Path(args.clustering_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    rng = np.random.default_rng(args.seed)

    # ── Load clustering ──────────────────────────────────────────────────────
    cl = pd.read_csv(clustering_dir / "cluster_labels.csv")[["feature_id", args.cluster_col]]
    cl = cl.rename(columns={args.cluster_col: "cluster"})
    cl = cl.dropna(subset=["cluster"]).copy()
    cl["cluster"] = cl["cluster"].astype(int)

    def parse_feat(fid):
        layer = int(fid.split("_F")[0][1:])
        feat = int(fid.split("_F")[1])
        return layer, feat
    cl[["layer","feat_idx"]] = pd.DataFrame(cl["feature_id"].map(parse_feat).tolist(), index=cl.index)
    all_clusters = sorted(cl["cluster"].unique())
    all_layers = sorted(cl["layer"].unique())

    cluster_feat_by_layer: Dict[int, Dict[int, List[int]]] = {}
    for cid in all_clusters:
        sub = cl[cl["cluster"] == cid]
        by_layer: Dict[int, List[int]] = {}
        for layer, grp in sub.groupby("layer"):
            by_layer[int(layer)] = grp["feat_idx"].tolist()
        cluster_feat_by_layer[cid] = by_layer

    # ── Build evaluation set: which clusters/combos to test ──────────────────
    if args.mode == "per_pair":
        targets = [int(x) for x in args.target_clusters.split(",") if x.strip()]
        eval_targets = [(str(t), cluster_feat_by_layer[t]) for t in targets]
    else:  # combined
        combos = [c.strip() for c in args.combinations.split(";") if c.strip()]
        eval_targets = []
        for combo in combos:
            ids = [int(x) for x in combo.split("+")]
            merged: Dict[int, List[int]] = {}
            for cid in ids:
                for layer, fids in cluster_feat_by_layer[cid].items():
                    merged.setdefault(layer, []).extend(fids)
            eval_targets.append((combo, merged))

    log.info(f"Mode: {args.mode}  |  Evaluating {len(eval_targets)} target(s)")
    for name, by_layer in eval_targets:
        n = sum(len(v) for v in by_layer.values())
        log.info(f"  {name}: {n} features across layers {sorted(by_layer.keys())}")

    # ── Load prompts and pairs (same matching as script 37) ──────────────────
    pm = pd.read_csv(args.prompt_metadata)
    alpha_rows = pm[pm["correct_answer"] == "alpha"].reset_index(drop=True)
    beta_rows = pm[pm["correct_answer"] == "beta"].reset_index(drop=True)
    n_pairs = min(len(alpha_rows), len(beta_rows))
    if args.max_pairs:
        n_pairs = min(n_pairs, args.max_pairs)
    alpha_idx = rng.choice(len(alpha_rows), size=n_pairs, replace=False)
    beta_idx = rng.choice(len(beta_rows), size=n_pairs, replace=False)
    pairs = [(alpha_rows.iloc[i], beta_rows.iloc[j]) for i, j in zip(alpha_idx, beta_idx)]
    log.info(f"Pairs: {n_pairs}  |  Total trials per target: 2 x {n_pairs} = {2*n_pairs}")

    # ── Load model + transcoders ────────────────────────────────────────────
    model, model_size = m37.load_model(device)
    model.model.eval()
    try:
        device = str(next(model.model.parameters()).device)
    except StopIteration:
        pass
    from src.transcoder.transcoder_loader import load_transcoder_set
    tc_set = load_transcoder_set(model_size=model_size, device=device,
                                  dtype=torch.bfloat16, lazy_load=True,
                                  layers=all_layers)
    transcoders = {layer: tc_set[layer] for layer in all_layers}

    # Precompute baseline margins + activations for all unique prompts
    log.info("Collecting baseline activations...")
    prompt_cache = {}
    unique_idxs = set()
    for pa, pb in pairs:
        unique_idxs.add(int(pa["prompt_idx"]))
        unique_idxs.add(int(pb["prompt_idx"]))
    pm_by_idx = pm.set_index("prompt_idx")
    for pidx in sorted(unique_idxs):
        row = pm_by_idx.loc[pidx]
        correct = " " + row["correct_answer"]
        incorrect = " " + row["incorrect_answer"]
        margin, feats = m37.get_features_and_margin(
            model, transcoders, row["prompt_text"], correct, incorrect, all_layers, device
        )
        prompt_cache[pidx] = {
            "text": row["prompt_text"], "correct": correct, "incorrect": incorrect,
            "margin": margin, "feats": feats,
        }
    log.info(f"Cached {len(prompt_cache)} prompts")

    # ── Run IIA per target, saving per-pair data ────────────────────────────
    per_pair_rows = []
    summary_rows = []
    for target_name, by_layer in eval_targets:
        n_feats = sum(len(v) for v in by_layer.values())
        flip_count = 0
        n_tested = 0
        for pair_id, (pa_row, pb_row) in enumerate(pairs):
            pa_idx = int(pa_row["prompt_idx"])
            pb_idx = int(pb_row["prompt_idx"])
            if pa_idx not in prompt_cache or pb_idx not in prompt_cache:
                continue
            pa = prompt_cache[pa_idx]; pb = prompt_cache[pb_idx]

            # Direction 1: patch alpha with beta features
            m_a = m37.get_patched_margin(
                model, transcoders, pa["text"], pa["correct"], pa["incorrect"],
                pa["feats"], pb["feats"], by_layer, device,
            )
            if pa["margin"] != 0 and m_a != 0:
                flipped_a = int(np.sign(m_a) != np.sign(pa["margin"]))
                flip_count += flipped_a
                n_tested += 1
                per_pair_rows.append(dict(
                    target=target_name, pair_id=pair_id, direction="a_to_b",
                    pa_idx=pa_idx, pb_idx=pb_idx,
                    baseline_margin=float(pa["margin"]), patched_margin=float(m_a),
                    flipped=flipped_a,
                ))

            # Direction 2: patch beta with alpha features
            m_b = m37.get_patched_margin(
                model, transcoders, pb["text"], pb["correct"], pb["incorrect"],
                pb["feats"], pa["feats"], by_layer, device,
            )
            if pb["margin"] != 0 and m_b != 0:
                flipped_b = int(np.sign(m_b) != np.sign(pb["margin"]))
                flip_count += flipped_b
                n_tested += 1
                per_pair_rows.append(dict(
                    target=target_name, pair_id=pair_id, direction="b_to_a",
                    pa_idx=pa_idx, pb_idx=pb_idx,
                    baseline_margin=float(pb["margin"]), patched_margin=float(m_b),
                    flipped=flipped_b,
                ))

        iia = flip_count / n_tested if n_tested > 0 else None
        log.info(f"  {target_name} (n_feat={n_feats}): IIA={iia:.4f} ({flip_count}/{n_tested})")
        summary_rows.append(dict(
            target=target_name, n_features=n_feats, n_pairs=n_pairs,
            n_tested=n_tested, n_flips=flip_count, iia=iia,
        ))

    # Save
    pp_df = pd.DataFrame(per_pair_rows)
    sm_df = pd.DataFrame(summary_rows)

    if args.mode == "per_pair":
        pp_out = out_dir / "iia_per_pair_flips.csv"
        sm_out = out_dir / "iia_per_pair_summary.csv"
        pp_df.to_csv(pp_out, index=False)
        sm_df.to_csv(sm_out, index=False)
        log.info(f"Saved: {pp_out}, {sm_out}")

        # Compute pairwise Jaccard overlap of flipped (pair_id, direction) sets
        log.info("Computing pairwise Jaccard overlap of flipped pairs...")
        targets = pp_df["target"].unique()
        overlap = {}
        flipped_sets = {}
        for t in targets:
            flipped_sets[t] = set(
                pp_df[(pp_df["target"]==t) & (pp_df["flipped"]==1)]
                .apply(lambda r: (r["pair_id"], r["direction"]), axis=1)
            )
        ovl_rows = []
        for t1 in targets:
            for t2 in targets:
                a, b = flipped_sets[t1], flipped_sets[t2]
                j = len(a & b) / len(a | b) if (a or b) else 1.0
                ovl_rows.append({"target1": t1, "target2": t2,
                                 "n_flips_a": len(a), "n_flips_b": len(b),
                                 "shared": len(a & b), "union": len(a | b),
                                 "jaccard": round(j, 4)})
        ovl_df = pd.DataFrame(ovl_rows)
        ovl_df.to_csv(out_dir / "iia_overlap_summary.csv", index=False)
        log.info(f"Saved: {out_dir/'iia_overlap_summary.csv'}")

        # Pivot view
        piv = ovl_df.pivot(index="target1", columns="target2", values="jaccard")
        print("\n── Pairwise Jaccard overlap of flipped pairs ──")
        print(piv.to_string())

    else:  # combined mode
        out = out_dir / "iia_combined_patches.csv"
        sm_df.to_csv(out, index=False)
        log.info(f"Saved: {out}")
        print("\n── Combined patches IIA ──")
        print(sm_df.to_string(index=False))


if __name__ == "__main__":
    main()
