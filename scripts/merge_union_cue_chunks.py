"""
merge_union_cue_chunks.py — мердж всех 5 чанков (и v2 если есть) в один CSV.

Schema-agnostic: автоматически находит ключ для дедупа (pair_idx | pair_id | (cluster, pair_idx)).
Выдаёт saniти-репорт: сколько уникальных pairs, какие кластеры покрыты, есть ли дыры.

USAGE:
    python3 scripts/merge_union_cue_chunks.py
    python3 scripts/merge_union_cue_chunks.py --include_v2   # если нужны и v2 чанки
"""
import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/analysis/runD_v2/union_cue")
    ap.add_argument("--out", default="data/analysis/runD_v2/union_cue/union_cue_pairs_all.csv")
    ap.add_argument("--include_v2", action="store_true",
                    help="include chunk_v2_* dirs in addition to chunk_0_87 etc")
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(glob.glob(str(root / "chunk_*/union_cue_pairs.csv")))
    if not args.include_v2:
        paths = [p for p in paths if "/chunk_v2_" not in p]

    if not paths:
        raise SystemExit(f"no chunk CSVs found under {root}")

    print(f"found {len(paths)} chunk CSVs:")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        print(f"  {p}: {len(df):4d} rows, cols={list(df.columns)[:6]}{'...' if len(df.columns)>6 else ''}")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    print(f"\nbefore dedup: {len(out)} rows")
    print(f"columns ({len(out.columns)}): {list(out.columns)}")

    # auto-detect dedup key
    candidates = ["pair_idx", "pair_id", "idx", "pair"]
    key_single = next((k for k in candidates if k in out.columns), None)

    if key_single and "cluster" in out.columns:
        # per-pair × cluster schema
        key = ["cluster", key_single]
        print(f"detected per-(pair, cluster) schema -> dedup on {key}")
    elif key_single:
        key = [key_single]
        print(f"detected per-pair schema -> dedup on {key}")
    else:
        # heuristic: assume last column is metric, dedup on all the rest
        key = [c for c in out.columns if out[c].dtype == "object" or c.endswith("_idx")]
        print(f"no obvious pair key -> dedup on {key}")

    if key:
        out = out.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)

    print(f"after dedup:  {len(out)} rows")

    # quick coverage report
    if key_single in out.columns:
        unique_pairs = out[key_single].nunique()
        rng = (out[key_single].min(), out[key_single].max())
        print(f"unique {key_single} values: {unique_pairs} (range {rng})")
        if "cluster" in out.columns:
            print(f"unique clusters: {out['cluster'].nunique()}, "
                  f"avg rows per pair: {len(out)/unique_pairs:.1f}")
            cov = out.groupby(key_single).size()
            if cov.min() < cov.max():
                print(f"  WARNING: pair coverage uneven — min={cov.min()}, max={cov.max()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
