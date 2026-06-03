"""
Helper for j79: build feature_acts.npz from top_k_indices files saved by script 04.

Script 04 saves per-layer: data/results/transcoder_features/layer_{L}/
  {behaviour}_{split}_top_k_indices.npy   (n_prompts, top_k)  int32
  {behaviour}_{split}_top_k_values.npy    (n_prompts, top_k)  float32

Script 79 wants a single npz with keys L{idx} mapping to (n_prompts, d_tc) dense
or sparse feature-activation matrices. We build those by zeroing a (n, d_tc)
matrix and setting top-k entries to the corresponding values.

Compressed npz keeps the size manageable (sparse boolean-ish patterns).

Usage:
  python -u scripts/79a_build_feature_acts_npz.py \
    --behaviour physics_decay_type_probe_v2 --split train \
    --d_tc 163840 \
    --out  data/analysis/active_set_overlap/feature_acts_v2.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--base", default="data/results/transcoder_features")
    p.add_argument("--d_tc", type=int, default=163840, help="transcoder dictionary size")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="restrict to these layers (default: all available)")
    p.add_argument("--out",
                   default="data/analysis/active_set_overlap/feature_acts.npz")
    args = p.parse_args()

    base = Path(args.base)
    layer_dirs = sorted(base.glob("layer_*"), key=lambda p: int(p.name.split("_")[1]))

    feat = {}
    for ld in layer_dirs:
        L = int(ld.name.split("_")[1])
        if args.layers and L not in args.layers:
            continue
        idx_path = ld / f"{args.behaviour}_{args.split}_top_k_indices.npy"
        val_path = ld / f"{args.behaviour}_{args.split}_top_k_values.npy"
        if not idx_path.exists() or not val_path.exists():
            continue
        indices = np.load(idx_path)   # (n, top_k) int
        values  = np.load(val_path)   # (n, top_k) float
        n, top_k = indices.shape
        # Build a (n, d_tc) sparse-as-dense in float16 to save space
        A = np.zeros((n, args.d_tc), dtype=np.float16)
        rows = np.repeat(np.arange(n), top_k)
        cols = indices.flatten()
        vals = values.flatten().astype(np.float16)
        # filter out zero-valued slots (top-k may include zeros if real <k features fire)
        nz = vals > 0
        A[rows[nz], cols[nz]] = vals[nz]
        feat[f"L{L}"] = A
        print(f"  L{L:>2}: {n} prompts, mean active ≈ {(A > 0).sum(axis=1).mean():.1f}/{top_k}")

    if not feat:
        raise SystemExit(
            f"No top_k_indices files found under {base} for "
            f"{args.behaviour}/{args.split}. Run 04_extract_transcoder_features first."
        )

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **feat)
    print(f"\nwrote {len(feat)} layers to {out} "
          f"(size: {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
