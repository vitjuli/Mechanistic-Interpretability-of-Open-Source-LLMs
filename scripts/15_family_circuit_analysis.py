#!/usr/bin/env python3
"""
Per-family and per-feature circuit analysis for physics_decay_type.

Answers:
  1. Per-family disruption: which surface families are most/least affected when circuit
     features are ablated? (sign_flip_rate, mean_abs_effect by family × layer)
  2. Feature interpretation: for each top feature, which families does it affect most?
     Show actual prompt text for highest-effect examples.
  3. Family-specific vs universal features: does the circuit split by family?

Usage:
    python scripts/15_family_circuit_analysis.py [--ui_run <run_id>] [--split train|test] [--top_n 15]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BEHAVIOUR = "physics_decay_type"
PROJECT_ROOT = Path(__file__).parent.parent

# Approximate circuit features from top ablation results (CSD3 circuit on remote).
# These are the top 11 features by disruption signal that match the known circuit.
CIRCUIT_FEATURES = {
    "L22_F110496", "L23_F83556", "L23_F71067",
    "L24_F60777", "L24_F52031", "L24_F18943", "L24_F88968", "L24_F249",
    "L25_F71226", "L25_F126439", "L25_F110282",
}

ANSI = {
    "bold": "\033[1m", "reset": "\033[0m",
    "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "cyan": "\033[36m",
}


def bold(s): return f"{ANSI['bold']}{s}{ANSI['reset']}"
def cyan(s): return f"{ANSI['cyan']}{s}{ANSI['reset']}"


def load_data(ui_run_dir: Path, prompts_file: Path):
    prompts = {
        i: json.loads(l)
        for i, l in enumerate(prompts_file.read_text().splitlines())
        if l.strip()
    }
    abl = pd.read_csv(ui_run_dir / "raw_sources" / f"intervention_ablation_{BEHAVIOUR}.csv")
    # feature_id is already formatted as "L{layer}_F{idx}" in the CSV
    # Attach prompt metadata
    for field in ["surface_family", "difficulty", "physics_concept", "keyword_free"]:
        abl[field] = abl["prompt_idx"].map(lambda i: prompts.get(i, {}).get(field, "?"))
    abl["prompt_text"] = abl["prompt_idx"].map(lambda i: prompts.get(i, {}).get("prompt", ""))
    return abl, prompts


# ─── 1. Per-family overview ───────────────────────────────────────────────────

def analysis_family_overview(abl: pd.DataFrame) -> None:
    print(bold("\n═══ 1. Per-family overview (all features) ═══"))

    # Baseline accuracy per family (deduplicate by prompt_idx per feature)
    per_prompt = abl.drop_duplicates("prompt_idx")[["prompt_idx", "surface_family",
                                                     "difficulty", "baseline_logit_diff",
                                                     "physics_concept"]].copy()
    per_prompt["correct"] = per_prompt["baseline_logit_diff"] > 0

    fam_base = (
        per_prompt.groupby("surface_family")["correct"]
        .agg(["sum", "count"])
        .assign(acc=lambda d: d["sum"] / d["count"])
        .rename(columns={"sum": "n_correct", "count": "n_prompts"})
    )

    print(f"\n{'Family':<8} {'n':<5} {'Baseline acc':>13}  {'Prompts correct'}")
    for fam, row in fam_base.sort_index().iterrows():
        bar = "█" * int(row["acc"] * 20)
        flag = "" if row["acc"] >= 0.80 else "  ⚠ hard"
        print(f"  {fam:<6} {int(row['n_prompts']):<5} {row['acc']:>12.1%}  {bar}{flag}")

    # Disruption per family
    print(f"\n{'Family':<8} {'sign_flip_rate':>14} {'mean_abs_eff':>12} {'n_flips':>8}")
    fam_abl = (
        abl.groupby("surface_family")
        .agg(
            sign_flip_rate=("sign_flipped", "mean"),
            mean_abs_effect=("abs_effect_size", "mean"),
            n_flips=("sign_flipped", "sum"),
            n_rows=("sign_flipped", "count"),
        )
    )
    for fam, row in fam_abl.sort_index().iterrows():
        print(f"  {fam:<6} {row['sign_flip_rate']:>14.3f} {row['mean_abs_effect']:>12.3f} {int(row['n_flips']):>8}")

    # Difficulty breakdown
    print(bold("\n  By difficulty:"))
    diff_abl = (
        abl.groupby("difficulty")
        .agg(
            sign_flip_rate=("sign_flipped", "mean"),
            mean_abs_effect=("abs_effect_size", "mean"),
            n_rows=("sign_flipped", "count"),
        )
    )
    for diff, row in diff_abl.iterrows():
        print(f"    {diff:<12} sfr={row['sign_flip_rate']:.3f}  abs_eff={row['mean_abs_effect']:.3f}")


# ─── 2. Per-family disruption by layer ────────────────────────────────────────

def analysis_family_by_layer(abl: pd.DataFrame) -> None:
    print(bold("\n═══ 2. Per-family disruption by layer ═══"))
    pivot = (
        abl.groupby(["surface_family", "layer"])["sign_flipped"]
        .mean()
        .unstack("layer")
        .fillna(0)
    )
    layers = sorted(pivot.columns)
    # Print heat table
    header = f"  {'Fam':<5}" + "".join(f" L{l:>2}" for l in layers)
    print(header)
    for fam, row in pivot.iterrows():
        vals = "".join(
            f" {row[l]:>4.0%}" if row[l] >= 0.10
            else (f" {row[l]:>4.0%}" if row[l] >= 0.05 else "     .")
            for l in layers
        )
        print(f"  {fam:<5}{vals}")
    print("  (shown: sign_flip_rate per family×layer; '.' = <5%)")


# ─── 3. Circuit feature interpretation ───────────────────────────────────────

def analysis_circuit_features(abl: pd.DataFrame, top_n: int = 15) -> None:
    print(bold(f"\n═══ 3. Circuit feature interpretation (top {top_n} by disruption) ═══"))

    feat_agg = (
        abl.groupby("feature_id")
        .agg(
            n_flips=("sign_flipped", "sum"),
            sfr=("sign_flipped", "mean"),
            mean_abs=("abs_effect_size", "mean"),
            layer=("layer", "first"),
        )
        .sort_values(["n_flips", "mean_abs"], ascending=False)
        .head(top_n)
    )

    for fid, row in feat_agg.iterrows():
        in_circuit = "★ circuit" if fid in CIRCUIT_FEATURES else ""
        print(f"\n  {bold(fid)} (L{int(row['layer'])}) {in_circuit}")
        print(f"    n_flips={int(row['n_flips'])}  sfr={row['sfr']:.3f}  mean_abs={row['mean_abs']:.3f}")

        feat_rows = abl[abl["feature_id"] == fid].copy()

        # Per-family breakdown
        fam_dist = (
            feat_rows.groupby("surface_family")["sign_flipped"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "flips", "count": "n", "mean": "sfr"})
            .sort_values("sfr", ascending=False)
        )
        fam_parts = [f"{fam}: {r['sfr']:.0%}({int(r['flips'])}/{int(r['n'])})"
                     for fam, r in fam_dist.iterrows()]
        print(f"    By family: {', '.join(fam_parts)}")

        # Per-concept breakdown (alpha vs beta)
        concept_dist = (
            feat_rows.groupby("physics_concept")["sign_flipped"]
            .agg(["sum", "mean"])
            .rename(columns={"sum": "flips", "mean": "sfr"})
        )
        concept_parts = [f"{c.replace('_decay','')}: sfr={r['sfr']:.0%}({int(r['flips'])})"
                         for c, r in concept_dist.iterrows()]
        print(f"    By concept: {', '.join(concept_parts)}")

        # Top prompts by abs effect
        top_prompts = feat_rows.nlargest(3, "abs_effect_size")[
            ["prompt_idx", "abs_effect_size", "effect_size", "sign_flipped",
             "surface_family", "difficulty", "physics_concept", "prompt_text"]
        ]
        print("    Top prompts by abs_effect:")
        for _, pr in top_prompts.iterrows():
            flip_mark = " [FLIP]" if pr["sign_flipped"] else ""
            print(f"      [{pr['surface_family']} {pr['difficulty'][:4]} {pr['physics_concept'][:5]}]"
                  f" eff={pr['effect_size']:+.3f}{flip_mark}")
            print(f"        {pr['prompt_text'][:100]}")


# ─── 4. Family-specificity of features ───────────────────────────────────────

def analysis_feature_specificity(abl: pd.DataFrame) -> None:
    print(bold("\n═══ 4. Feature family-specificity ═══"))
    print("  Are circuit features universal (affect all families) or family-specific?\n")

    circuit_rows = abl[abl["feature_id"].isin(CIRCUIT_FEATURES)].copy()
    if circuit_rows.empty:
        print("  (No circuit features found in ablation data)")
        return

    families = sorted(abl["surface_family"].unique())
    feat_fam = (
        circuit_rows.groupby(["feature_id", "surface_family"])["sign_flipped"]
        .mean()
        .unstack("surface_family")
        .fillna(0)
    )

    header = f"  {'Feature':<18}" + "".join(f" {f:>6}" for f in families) + "  range"
    print(header)
    for fid, row in feat_fam.sort_index().iterrows():
        vals = "".join(f" {row.get(f, 0):>6.0%}" for f in families)
        rng = row.max() - row.min()
        flag = "  ← family-specific" if rng > 0.15 else ""
        print(f"  {fid:<18}{vals}  {rng:.2f}{flag}")

    print("\n  (range > 0.15 → sign_flip_rate varies substantially by family)")


# ─── 5. Failure analysis ─────────────────────────────────────────────────────

def analysis_failures(abl: pd.DataFrame, prompts: dict) -> None:
    print(bold("\n═══ 5. Failure analysis — hardest prompts ═══"))

    per_prompt = (
        abl.groupby("prompt_idx")
        .agg(
            baseline=("baseline_logit_diff", "first"),
            sfr=("sign_flipped", "mean"),
            n_flips=("sign_flipped", "sum"),
            surface_family=("surface_family", "first"),
            difficulty=("difficulty", "first"),
            concept=("physics_concept", "first"),
        )
        .sort_values("sfr", ascending=False)
    )

    wrong = per_prompt[per_prompt["baseline"] <= 0]
    print(f"\n  {len(wrong)} prompts where model is already wrong at baseline:")
    for pid, row in wrong.head(8).iterrows():
        p = prompts.get(pid, {})
        print(f"    [{row['surface_family']} {row['difficulty'][:4]}] "
              f"{p.get('prompt', '')[:90]}")

    fragile = per_prompt[per_prompt["baseline"] > 0].nlargest(5, "sfr")
    print(f"\n  Top 5 fragile (correct at baseline but most flipped by ablation):")
    for pid, row in fragile.iterrows():
        p = prompts.get(pid, {})
        print(f"    [{row['surface_family']} {row['difficulty'][:4]}] sfr={row['sfr']:.0%}  "
              f"{p.get('prompt', '')[:80]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui_run", default=None)
    parser.add_argument("--split", default=None, choices=["train", "test"],
                        help="Which split's prompt file to use. Auto-detected from run name if omitted.")
    parser.add_argument("--prompts_file", default=None,
                        help="Explicit path to the prompts JSONL (overrides --split).")
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    ui_offline = PROJECT_ROOT / "data" / "ui_offline"
    if args.ui_run:
        ui_run_dir = ui_offline / args.ui_run
    else:
        candidates = sorted(
            d for d in ui_offline.iterdir()
            if d.is_dir() and f"_{BEHAVIOUR}_" in d.name
        )
        if not candidates:
            print(f"No UI run found for {BEHAVIOUR}")
            return
        ui_run_dir = candidates[-1]

    # Resolve prompts file
    if args.prompts_file:
        prompts_file = Path(args.prompts_file)
    elif args.split:
        prompts_file = PROJECT_ROOT / "data" / "prompts" / f"{BEHAVIOUR}_{args.split}.jsonl"
    else:
        # Auto-detect from run name: contains "_test_" or "_train_"
        split = "test" if "_test_" in ui_run_dir.name else "train"
        prompts_file = PROJECT_ROOT / "data" / "prompts" / f"{BEHAVIOUR}_{split}.jsonl"

    print(cyan(f"Run: {ui_run_dir.name}"))
    print(cyan(f"Prompts: {prompts_file.name}"))

    abl, prompts = load_data(ui_run_dir, prompts_file)
    analysis_family_overview(abl)
    analysis_family_by_layer(abl)
    analysis_circuit_features(abl, args.top_n)
    analysis_feature_specificity(abl)
    analysis_failures(abl, prompts)

    print()


if __name__ == "__main__":
    main()
