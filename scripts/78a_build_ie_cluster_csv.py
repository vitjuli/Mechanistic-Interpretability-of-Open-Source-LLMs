"""
Helper for j78: rebuild the feature_id -> cluster_id CSV for intensive/extensive
clusters (k=6).

`71_ie_cluster_analysis.py` was supposed to write `ie_feature_meta.json` but it
isn't on disk. We reuse the EXACT `load_feature_matrix` logic from script 71
to recover the feature order, pair with `ie_cluster_labels_k6.npy`, and emit a
CSV that `78_cluster_ablation_null.py` consumes.

Run on CSD3 (needs the transcoder feature .npy files for the IE behaviour).
Outputs:
  data/results/abstraction_ie/physics_intensive_extensive_v1/ie_cluster_csv_k6.csv
"""
from __future__ import annotations
import argparse, csv as csvlib, sys
from pathlib import Path
import numpy as np


def load_feature_matrix(behaviour, split, tc_base="data/results/transcoder_features"):
    """Verbatim from 71_ie_cluster_analysis.py:load_feature_matrix (lines 69-118)."""
    tc_base = Path(tc_base)
    layer_dirs = sorted(tc_base.glob("layer_*"), key=lambda p: int(p.name.split("_")[1]))

    rows, meta = [], []
    for ld in layer_dirs:
        idx_path = ld / f"{behaviour}_{split}_top_k_indices.npy"
        val_path = ld / f"{behaviour}_{split}_top_k_values.npy"
        if not idx_path.exists() or not val_path.exists():
            continue
        layer_idx = int(ld.name.split("_")[1])
        indices = np.load(idx_path)
        values  = np.load(val_path)
        unique_feats = np.unique(indices)
        for feat_idx in unique_feats:
            match = (indices == feat_idx)
            act_row = (values * match).sum(axis=1).astype(np.float32)
            rows.append(act_row)
            meta.append({"feature_id": f"L{layer_idx}_F{feat_idx}",
                         "layer": layer_idx, "feature_idx": int(feat_idx)})

    if not rows:
        raise FileNotFoundError(
            f"No transcoder feature files under {tc_base} for {behaviour}/{split}"
        )

    act_matrix = np.stack(rows, axis=0).astype(np.float32)
    min_freq = max(5, int(0.02 * act_matrix.shape[1]))
    freq = (act_matrix > 0).sum(axis=1)
    keep = freq >= min_freq
    meta = [m for m, k in zip(meta, keep) if k]
    n_kept = int(keep.sum())
    print(f"  Loaded {n_kept} features (filter freq≥{min_freq}; from {len(keep)} raw)")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="physics_intensive_extensive_v1")
    p.add_argument("--split", default="train")
    p.add_argument("--labels_npy",
                   default="data/results/abstraction_ie/physics_intensive_extensive_v1/ie_cluster_labels_k6.npy")
    p.add_argument("--out_csv",
                   default="data/results/abstraction_ie/physics_intensive_extensive_v1/ie_cluster_csv_k6.csv")
    args = p.parse_args()

    print(f"Rebuilding feature order for {args.behaviour}/{args.split}...")
    meta = load_feature_matrix(args.behaviour, args.split)
    labels = np.load(args.labels_npy)

    if len(meta) != len(labels):
        raise SystemExit(
            f"MISMATCH: rebuilt meta has {len(meta)} features but labels has {len(labels)}. "
            "The transcoder feature files must match exactly the version used in script 71. "
            "Check that 04_extract_transcoder_features was rerun with the same params."
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csvlib.writer(fh)
        w.writerow(["feature_id", "cluster"])
        for m, c in zip(meta, labels):
            w.writerow([m["feature_id"], int(c)])

    from collections import Counter
    counts = Counter(int(c) for c in labels)
    print(f"wrote {len(meta)} rows to {out_path}")
    print(f"cluster counts: {dict(sorted(counts.items()))}")
    print(f"  (expected for k=6 IE: C0=2114 C1=1 C2=2 C3=11 C4=115 C5=4)")


if __name__ == "__main__":
    main()
