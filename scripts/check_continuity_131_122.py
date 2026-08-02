#!/usr/bin/env python
"""
check_continuity_131_122.py  [flip-law gate: does 131 reproduce 122 on the shared direction?]

131 re-runs the tier-2 steering protocol to add the writing direction (delta) that 122 never
swept. Both scripts build their directions from the same field dump with the same split, sigma
and hook, so the direction they SHARE (usage) must produce the same flips. If it does not, the
two cell files were not produced under the same conventions and must not be merged by 132.

This gate exists because exactly that failure happened once: 131 defaulted to Qwen3-4B-Base
while the dump and 122 used Qwen3-4B, and the delta cells silently entered the assembled law.

Criterion: for every (layer, c) cell, |flip_131 - flip_122| <= tol (default 0.03), computed on
the prompt indices the two files share. Exit code 1 on any violation.

USAGE
  python scripts/check_continuity_131_122.py \
      --cells_122 data/analysis/runD_v2/B1_alpha_beta/cells_tier2.csv \
      --cells_131 data/analysis/runD_v2/B1_alpha_beta/cells_tier2_delta.csv \
      --tol 0.03 --concept B1_alpha_beta

SELF-TEST (no repo data):  python scripts/check_continuity_131_122.py --self_test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# =====================================================================
# Pure core (exercised by --self_test)
# =====================================================================
def is_flip(y: int, m0: float, m1: float) -> int:
    """Sign change of the margin toward the opposite class — the definition used by 122
    (aggregate_cell), 131 (inline) and 132 (load_cells)."""
    return int((y == 0 and m0 < 0 and m1 > 0) or (y == 1 and m0 > 0 and m1 < 0))


def load_cells(path: Path, want_dir: str):
    """-> {(layer, c): {idx: (y, m0, m1)}} for one direction."""
    out: dict[tuple[str, float], dict[str, tuple[int, float, float]]] = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["dir"] != want_dir:
                continue
            out[(r["layer"], float(r["c"]))][r["idx"]] = (
                int(float(r["y"])), float(r["m0"]), float(r["m1"]))
    return out


def compare(cells_a, cells_b, tol):
    """Per-cell flip rates on the shared prompt indices. Returns (rows, n_m0_mismatch)."""
    rows, n_m0_bad = [], 0
    for key in sorted(set(cells_a) & set(cells_b), key=lambda k: (int(k[0]), k[1])):
        A, B = cells_a[key], cells_b[key]
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        fa = sum(is_flip(*A[i]) for i in shared) / len(shared)
        fb = sum(is_flip(*B[i]) for i in shared) / len(shared)
        n_m0_bad += sum(1 for i in shared if abs(A[i][1] - B[i][1]) > 1e-6)
        rows.append({"layer": int(key[0]), "c": key[1], "n": len(shared),
                     "flip_122": fa, "flip_131": fb, "delta": fb - fa,
                     "ok": abs(fb - fa) <= tol})
    return rows, n_m0_bad


def self_test():
    # one cell that agrees, one that does not
    a = {("16", 1.0): {"0": (0, -1.0, +0.5), "1": (1, +1.0, -0.5), "2": (0, -1.0, -0.5)},
         ("22", 1.0): {"0": (0, -1.0, +0.5), "1": (1, +1.0, -0.5)}}
    b = {("16", 1.0): {"0": (0, -1.0, +0.5), "1": (1, +1.0, -0.5), "2": (0, -1.0, -0.4)},
         ("22", 1.0): {"0": (0, -1.0, -0.9), "1": (1, +1.0, +0.9)}}
    rows, n_bad = compare(a, b, tol=0.03)
    assert n_bad == 0, n_bad
    r16 = next(r for r in rows if r["layer"] == 16)
    r22 = next(r for r in rows if r["layer"] == 22)
    assert abs(r16["flip_122"] - 2 / 3) < 1e-12 and abs(r16["delta"]) < 1e-12 and r16["ok"]
    assert abs(r22["flip_122"] - 1.0) < 1e-12 and abs(r22["flip_131"]) < 1e-12
    assert abs(r22["delta"] + 1.0) < 1e-12 and not r22["ok"]
    # m0 disagreement is detected (different baseline -> different dump/model)
    c = {("16", 1.0): {"0": (0, -2.0, +0.5)}}
    d = {("16", 1.0): {"0": (0, -1.0, +0.5)}}
    _, n_bad2 = compare(c, d, tol=0.03)
    assert n_bad2 == 1, n_bad2
    print("[self_test] OK — flip identity, per-cell comparison, m0 mismatch detection pass.")


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--cells_122", help="cells_tier2.csv written by 122 (--dump_cells)")
    ap.add_argument("--cells_131", help="cells_tier2_delta.csv written by 131")
    ap.add_argument("--dir", default="usage", help="direction present in BOTH files")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--concept", default="")
    ap.add_argument("--out_json", default=None, help="optional path for the machine-readable verdict")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    assert a.cells_122 and a.cells_131, "--cells_122/--cells_131 required"

    A = load_cells(Path(a.cells_122), a.dir)
    B = load_cells(Path(a.cells_131), a.dir)
    if not A or not B:
        print(f"GATE FAIL [{a.concept}]: direction {a.dir!r} missing "
              f"(122 cells={len(A)}, 131 cells={len(B)})")
        sys.exit(1)

    rows, n_m0_bad = compare(A, B, a.tol)
    bad = [r for r in rows if not r["ok"]]
    worst = max((abs(r["delta"]) for r in rows), default=float("nan"))
    mean = sum(abs(r["delta"]) for r in rows) / len(rows) if rows else float("nan")

    print(f"continuity gate [{a.concept}] dir={a.dir} tol=±{a.tol}: "
          f"{len(rows)} shared cells, max|Δ|={worst:.4f}, mean|Δ|={mean:.4f}, "
          f"over tolerance: {len(bad)}")
    if n_m0_bad:
        print(f"  ⚠ {n_m0_bad} prompts disagree on the BASELINE margin m0 — the two files do not "
              f"come from the same dump. Fix that first; the flip comparison below is meaningless.")
    for r in sorted(bad, key=lambda r: -abs(r["delta"]))[:10]:
        print(f"   L{r['layer']:<3d} c={r['c']:<6g} n={r['n']:<4d} "
              f"122={r['flip_122']:.3f} 131={r['flip_131']:.3f} Δ={r['delta']:+.3f}")

    verdict = {"concept": a.concept, "dir": a.dir, "tol": a.tol, "n_cells": len(rows),
               "max_abs_delta": worst, "mean_abs_delta": mean, "n_over_tol": len(bad),
               "n_m0_mismatch": n_m0_bad, "pass": not bad and not n_m0_bad}
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(verdict, open(a.out_json, "w"), indent=2)

    if verdict["pass"]:
        print("GATE PASS — 131 reproduces 122 on the shared direction; delta cells may be assembled.")
        return
    print("GATE FAIL — do NOT run 132 on these cells and do NOT move delta numbers into the text. "
          "Check --model_name against meta.npz, then --split_seed/--train_frac/--shrink.")
    sys.exit(1)


if __name__ == "__main__":
    main()
