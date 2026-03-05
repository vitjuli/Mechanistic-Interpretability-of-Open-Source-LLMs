#!/usr/bin/env python3
"""
Analysis script for multilingual_circuits behaviour.

Computes the four Anthropic "Multilingual Circuits" claims:
  Claim 1: Language-specific features exist (low per-layer IoU at early/late layers)
  Claim 2: Shared cross-language features exist (high IoU at middle layers)
  Claim 3: Shared features concentrated in middle layers
  Claim 4: Bridge features whose ablation degrades BOTH EN and FR

Also:
  - Baseline gate check: EN sign_acc >= 0.90, FR sign_acc >= 0.75, mean_norm_diff >= 1.0
  - C3 patching: disruption_rate, flip_rate, mean_effect_size ± SEM

Usage (on CSD3 after running the full pipeline):
  python scripts/a_analyze_multilingual_circuits.py \
      --behaviour multilingual_circuits --split train

Outputs:
  data/analysis/multilingual_circuits/
    gate_check.txt           — baseline gate pass/fail
    iou_per_layer.csv        — per-layer IoU of EN vs FR feature sets
    bridge_features.csv      — features with negative effect in both EN and FR
    c3_patching_stats.txt    — disruption_rate, flip_rate, mean_effect_size
    REPORT.md                — human-readable summary of all four claims
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def get_paths(behaviour: str, split: str) -> dict:
    base = Path("data")
    interv_dir = base / "results" / "interventions" / behaviour
    return {
        "train_jsonl":  base / "prompts" / f"{behaviour}_{split}.jsonl",
        "baseline_csv": base / "results" / f"baseline_{behaviour}_{split}.csv",
        "graph_json":   base / "results" / "attribution_graphs" / behaviour
                              / f"attribution_graph_{split}_n48.json",
        "features_dir": base / "results" / "transcoder_features",
        "ablation_csv": interv_dir / f"intervention_ablation_{behaviour}.csv",
        "c3_csv":       interv_dir / f"intervention_patching_C3_{behaviour}.csv",
        "out_dir":      base / "analysis" / behaviour,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(jsonl_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(jsonl_path)]
    df = pd.DataFrame(rows)
    df.index.name = "prompt_idx"
    df = df.reset_index()
    return df


def sem(arr):
    """Standard error of the mean."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(n))


def bootstrap_ci(arr, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap 95% CI for the mean. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline gate check
# ─────────────────────────────────────────────────────────────────────────────

def check_baseline_gate(baseline_csv: Path, out_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("1. BASELINE GATE CHECK")
    print("=" * 60)

    if not baseline_csv.exists():
        print(f"  ERROR: {baseline_csv} not found.")
        return {}

    df = pd.read_csv(baseline_csv)

    # sign correct = logprob_diff > 0
    df["sign_correct"] = df["logprob_diff"] > 0

    overall_sign_acc = df["sign_correct"].mean()
    overall_mean_norm = df["logprob_diff_normalized"].mean()

    if "language" not in df.columns:
        print("  WARNING: 'language' column missing from baseline CSV. Cannot check per-language.")
        lang_stats = {}
    else:
        lang_stats = {}
        for lang, grp in df.groupby("language"):
            sign_acc = grp["sign_correct"].mean()
            mean_norm = grp["logprob_diff_normalized"].mean()
            lang_stats[lang] = {"sign_accuracy": sign_acc, "mean_norm_diff": mean_norm,
                                 "n": len(grp), "n_pass": int(grp["sign_correct"].sum())}

    # Gate thresholds
    gates = {
        "en_sign_acc":   {"value": lang_stats.get("en", {}).get("sign_accuracy", float("nan")),
                           "threshold": 0.90, "op": ">="},
        "fr_sign_acc":   {"value": lang_stats.get("fr", {}).get("sign_accuracy", float("nan")),
                           "threshold": 0.75, "op": ">="},
        "mean_norm_diff": {"value": overall_mean_norm, "threshold": 1.0, "op": ">="},
    }

    lines = []
    all_pass = True
    for name, g in gates.items():
        v = g["value"]
        t = g["threshold"]
        passed = (v >= t) if not math.isnan(v) else False
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        line = f"  [{status}] {name}: {v:.4f} (threshold {g['op']} {t})"
        print(line)
        lines.append(line)

    # Per-language breakdown
    print()
    for lang, s in lang_stats.items():
        print(f"  {lang.upper()}: sign_acc={s['sign_accuracy']:.3f}  "
              f"mean_norm_diff={s['mean_norm_diff']:.3f}  "
              f"n={s['n']}  n_pass={s['n_pass']}")
    print(f"\n  Overall sign_acc: {overall_sign_acc:.4f}  "
          f"mean_norm_diff: {overall_mean_norm:.4f}")

    gate_status = "PASS" if all_pass else "FAIL"
    print(f"\n  ► Baseline gate: {gate_status}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gate_check.txt", "w") as f:
        f.write(f"Baseline gate: {gate_status}\n\n")
        for l in lines:
            f.write(l.strip() + "\n")
        f.write("\nPer-language:\n")
        for lang, s in lang_stats.items():
            f.write(f"  {lang.upper()}: sign_acc={s['sign_accuracy']:.4f}  "
                    f"mean_norm_diff={s['mean_norm_diff']:.4f}  "
                    f"n={s['n']}  n_pass={s['n_pass']}\n")
        f.write(f"\nOverall: sign_acc={overall_sign_acc:.4f}  "
                f"mean_norm_diff={overall_mean_norm:.4f}\n")

    return {"gate_status": gate_status, "gates": gates, "lang_stats": lang_stats,
            "overall_sign_acc": overall_sign_acc, "overall_mean_norm": overall_mean_norm}


# ─────────────────────────────────────────────────────────────────────────────
# 2 + 3. Per-layer IoU of EN vs FR feature sets
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(features_dir: Path, behaviour: str, split: str,
                n_en: int, n_fr: int, layers: list, out_dir: Path) -> pd.DataFrame:
    """
    Compute per-layer IoU between EN and FR feature sets.

    Uses top_k_indices.npy from script 04 (shape: n_prompts × top_k).
    EN prompts are assumed to be indices 0..(n_en-1),
    FR prompts are indices n_en..(n_en+n_fr-1).
    """
    print("\n" + "=" * 60)
    print("2+3. PER-LAYER IOU (EN vs FR feature activation sets)")
    print("=" * 60)

    rows = []
    for layer in layers:
        npy_path = (features_dir / f"layer_{layer}"
                    / f"{behaviour}_{split}_top_k_indices.npy")
        if not npy_path.exists():
            print(f"  WARNING: {npy_path} not found, skipping layer {layer}")
            continue

        idx = np.load(npy_path)  # (n_prompts, ...) possibly (n_prompts, 1, top_k) or (n_prompts, top_k)
        if idx.ndim == 3:
            idx = idx[:, 0, :]    # take decision token position → (n_prompts, top_k)
        elif idx.ndim == 1:
            print(f"  WARNING: unexpected shape {idx.shape} for layer {layer}")
            continue

        n_total = idx.shape[0]
        en_end = n_en
        fr_end = n_en + n_fr
        assert fr_end <= n_total, (
            f"Expected {n_en} EN + {n_fr} FR = {fr_end} prompts, "
            f"but .npy has {n_total} rows"
        )

        en_feats = set(idx[:en_end, :].flatten().tolist())
        fr_feats = set(idx[en_end:fr_end, :].flatten().tolist())

        intersection = en_feats & fr_feats
        union = en_feats | fr_feats
        iou = len(intersection) / len(union) if union else 0.0

        rows.append({
            "layer": layer,
            "n_en_features": len(en_feats),
            "n_fr_features": len(fr_feats),
            "n_intersection": len(intersection),
            "n_union": len(union),
            "iou": round(iou, 4),
        })

        print(f"  Layer {layer:2d}: EN={len(en_feats):4d}  FR={len(fr_feats):4d}  "
              f"∩={len(intersection):3d}  ∪={len(union):4d}  IoU={iou:.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No IoU data computed — check feature extraction paths.")
        return df

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "iou_per_layer.csv", index=False)

    max_iou_layer = df.loc[df["iou"].idxmax(), "layer"]
    mean_iou = df["iou"].mean()
    print(f"\n  Mean IoU across layers: {mean_iou:.4f}")
    print(f"  Layer with max IoU:     {max_iou_layer} ({df['iou'].max():.4f})")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bridge features (consistent negative ablation effect in BOTH EN and FR)
# ─────────────────────────────────────────────────────────────────────────────

def find_bridge_features(ablation_csv: Path, train_jsonl: Path,
                         n_en: int, out_dir: Path) -> pd.DataFrame:
    """
    A "bridge" feature has mean_effect < 0 for BOTH EN and FR prompts.

    effect_size = intervened_logit_diff - baseline_logit_diff
    A negative value means ablating the feature HURTS the model (feature supports
    the correct answer) — relevant for both languages → bridge.

    Works with both per-feature rows (feature_id column present and non-empty)
    and legacy bundled rows (feature_id absent/empty → derived from feature_indices[0]).
    """
    print("\n" + "=" * 60)
    print("4. BRIDGE FEATURES")
    print("=" * 60)

    if not ablation_csv.exists():
        print(f"  ERROR: {ablation_csv} not found.")
        return pd.DataFrame()

    df = pd.read_csv(ablation_csv)

    # Resolve feature_id: prefer explicit column; fall back to parsing feature_indices[0]
    per_feature_mode = (
        "feature_id" in df.columns
        and df["feature_id"].notna().any()
        and (df["feature_id"].astype(str) != "").any()
    )
    if per_feature_mode:
        print(f"  Using explicit feature_id column (per-feature mode, {len(df)} rows)")
    else:
        print(f"  Deriving feature_id from feature_indices[0] (bundled mode, {len(df)} rows)")
        def parse_feature_idx(val):
            try:
                lst = ast.literal_eval(str(val))
                return lst[0] if isinstance(lst, list) and lst else None
            except Exception:
                return None
        fidx = df["feature_indices"].apply(parse_feature_idx)
        df["feature_id"] = "L" + df["layer"].astype(str) + "_F" + fidx.astype(str)

    # Always extract numeric feature index from feature_id for display/pivot
    df["feature_idx_0"] = (
        df["feature_id"].str.extract(r"_F(\d+)$")[0].astype(float).astype("Int64")
    )

    # Assign language by prompt_idx (EN: 0..n_en-1, FR: n_en..)
    df["language"] = df["prompt_idx"].apply(lambda i: "en" if i < n_en else "fr")

    # Compute per-feature, per-language mean_effect_size
    grp = df.groupby(["feature_id", "layer", "feature_idx_0", "language"])["effect_size"]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["feature_id", "layer", "feature_idx_0", "language", "mean_effect", "std_effect", "n"]
    stats["sem_effect"] = stats["std_effect"] / np.sqrt(stats["n"].clip(lower=1))

    # Pivot to wide: one row per feature
    wide = stats.pivot_table(
        index=["feature_id", "layer", "feature_idx_0"],
        columns="language",
        values=["mean_effect", "sem_effect", "n"],
    )
    wide.columns = [f"{v}_{lang}" for v, lang in wide.columns]
    wide = wide.reset_index()

    # Bridge criterion: mean_effect < 0 in BOTH languages
    wide["is_bridge"] = (wide.get("mean_effect_en", pd.Series(dtype=float)).fillna(0) < 0) & \
                        (wide.get("mean_effect_fr", pd.Series(dtype=float)).fillna(0) < 0)

    # Bridge score = min(|mean_effect_en|, |mean_effect_fr|) — only for bridge features
    wide["bridge_score"] = np.where(
        wide["is_bridge"],
        wide[["mean_effect_en", "mean_effect_fr"]].abs().min(axis=1),
        0.0,
    )

    bridges = wide[wide["is_bridge"]].sort_values("bridge_score", ascending=False)
    all_features = wide

    print(f"  Total graph features:  {len(all_features)}")
    print(f"  Bridge features:       {len(bridges)}")
    if len(all_features) > 0:
        print(f"  Bridge fraction:       {len(bridges)/len(all_features):.2%}")

    if not bridges.empty:
        print(f"\n  Top bridge features (sorted by min |effect| in both languages):")
        for _, row in bridges.head(10).iterrows():
            me_en = row.get("mean_effect_en", float("nan"))
            me_fr = row.get("mean_effect_fr", float("nan"))
            sem_en = row.get("sem_effect_en", float("nan"))
            sem_fr = row.get("sem_effect_fr", float("nan"))
            print(f"    {row['feature_id']:20s}  "
                  f"EN: {me_en:+.3f}±{sem_en:.3f}  "
                  f"FR: {me_fr:+.3f}±{sem_fr:.3f}  "
                  f"score={row['bridge_score']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / "bridge_features.csv", index=False)
    bridges.to_csv(out_dir / "bridge_features_only.csv", index=False)
    print(f"\n  Saved: bridge_features.csv ({len(wide)} features), "
          f"bridge_features_only.csv ({len(bridges)} bridge features)")

    return bridges


# ─────────────────────────────────────────────────────────────────────────────
# C3 patching stats
# ─────────────────────────────────────────────────────────────────────────────

def analyze_c3_patching(c3_csv: Path, train_jsonl: Path, out_dir: Path) -> dict:
    """
    C3 patching: EN antonym features patched into FR context.

    effect_size < 0 → disruption (patching EN features hurts FR model's confidence
    in the correct FR answer).

    Per-feature mode: each row = one feature × one pair × one layer.
    Bundled mode: each row = all features × one pair × one layer.

    Reported:
      disruption_rate           = fraction of rows with effect_size < 0
      flip_rate                 = fraction of rows with sign_flipped = True
      mean_effect_size ± SEM    = overall and per-layer
      lang_swap_strength        = fraction of FEATURES with mean_effect < 0 across all pairs
                                  (per-feature mode only; analogous to bridge criterion)
    """
    print("\n" + "=" * 60)
    print("C3 PATCHING — LANGUAGE SWAP (EN→FR)")
    print("=" * 60)

    if not c3_csv.exists():
        print(f"  ERROR: {c3_csv} not found.")
        return {}

    df = pd.read_csv(c3_csv)

    # Load train prompts to attach concept_index
    prompts = load_prompts(train_jsonl)
    prompts_en = prompts[prompts["language"] == "en"].set_index("prompt_idx")

    if "concept_index" in df.columns:
        # Already in the CSV as a direct column
        pass
    elif "metadata" in df.columns:
        # Parse concept_index (and template_idx) from the metadata dict column.
        # This is the preferred path after the per-feature conversion which stores
        # concept_index and template_idx in the metadata dict for each row.
        def _extract_meta(meta_str, field):
            try:
                d = ast.literal_eval(str(meta_str))
                return d.get(field) if isinstance(d, dict) else None
            except Exception:
                return None
        df["concept_index"] = df["metadata"].apply(lambda m: _extract_meta(m, "concept_index"))
        df["template_idx"]  = df["metadata"].apply(lambda m: _extract_meta(m, "template_idx"))

    # Overall stats
    effect = df["effect_size"].values
    flipped = df["sign_flipped"].astype(bool).values

    disruption_rate = float((effect < 0).mean())
    flip_rate        = float(flipped.mean())
    mean_eff         = float(np.mean(effect))
    sem_eff          = sem(effect)
    ci_lo, ci_hi     = bootstrap_ci(effect)

    print(f"\n  Rows: {len(df)}")
    print(f"  disruption_rate (effect_size < 0): {disruption_rate:.3f}  "
          f"[target ≥ 0.40]  {'✓ PASS' if disruption_rate >= 0.40 else '✗ FAIL'}")
    print(f"  flip_rate (sign_flipped = True):   {flip_rate:.3f}")
    print(f"  mean_effect_size:                  {mean_eff:+.4f} ± {sem_eff:.4f}  "
          f"(95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")

    # Per-layer breakdown
    print(f"\n  Per-layer disruption_rate:")
    layer_stats = []
    for layer, grp in df.groupby("layer"):
        dr = (grp["effect_size"] < 0).mean()
        me = grp["effect_size"].mean()
        layer_stats.append({"layer": layer, "disruption_rate": round(dr, 4),
                             "mean_effect": round(me, 4), "n": len(grp)})
        print(f"    Layer {int(layer):2d}: disrupt={dr:.3f}  mean_eff={me:+.4f}  n={len(grp)}")

    # Per-concept breakdown (if concept_index available)
    if "concept_index" in df.columns and not df["concept_index"].isna().all():
        print(f"\n  Per-concept disruption_rate:")
        concept_stats = []
        for cidx, grp in df.groupby("concept_index"):
            dr = (grp["effect_size"] < 0).mean()
            me = grp["effect_size"].mean()
            concept_stats.append({"concept_index": cidx, "disruption_rate": round(dr, 4),
                                   "mean_effect": round(me, 4), "n": len(grp)})
            print(f"    Concept {int(cidx):2d}: disrupt={dr:.3f}  mean_eff={me:+.4f}  n={len(grp)}")
    else:
        concept_stats = []
        print("  (concept_index not available in patching CSV)")

    # Per-feature language swap strength (per-feature mode only)
    # = fraction of FEATURES whose mean patching effect across all pairs is < 0
    # This measures which features reliably contribute to cross-language routing disruption.
    lang_swap_strength = float("nan")
    lang_swap_features = []
    if "feature_id" in df.columns and df["feature_id"].notna().any() and (df["feature_id"].astype(str) != "").any():
        feat_means = df.groupby("feature_id")["effect_size"].mean()
        n_disrupting = int((feat_means < 0).sum())
        n_total_feats = len(feat_means)
        lang_swap_strength = n_disrupting / n_total_feats if n_total_feats > 0 else float("nan")
        lang_swap_features = feat_means[feat_means < 0].sort_values().index.tolist()
        print(f"\n  Lang-swap disrupting features (mean_effect < 0): "
              f"{n_disrupting}/{n_total_feats} = {lang_swap_strength:.3f}")
        print(f"  Top disrupting: {lang_swap_features[:5]}")
        # Save per-feature lang-swap stats
        feat_df = feat_means.reset_index()
        feat_df.columns = ["feature_id", "mean_effect_size"]
        feat_df["disrupts"] = feat_df["mean_effect_size"] < 0
        feat_df = feat_df.sort_values("mean_effect_size")
        out_dir.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(out_dir / "c3_patching_per_feature.csv", index=False)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_text = (
        f"C3 Patching Summary\n"
        f"===================\n"
        f"Rows: {len(df)}\n"
        f"disruption_rate: {disruption_rate:.4f}  (target >= 0.40)\n"
        f"flip_rate:        {flip_rate:.4f}\n"
        f"mean_effect_size: {mean_eff:+.4f} ± {sem_eff:.4f}  "
        f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])\n"
        f"lang_swap_strength (frac features with mean_effect < 0): "
        f"{lang_swap_strength:.4f}\n"
    )
    with open(out_dir / "c3_patching_stats.txt", "w") as f:
        f.write(summary_text)
        f.write("\nPer-layer:\n")
        for s in layer_stats:
            f.write(f"  layer={s['layer']:2d}  disrupt={s['disruption_rate']:.4f}  "
                    f"mean_eff={s['mean_effect']:+.4f}  n={s['n']}\n")
        if concept_stats:
            f.write("\nPer-concept:\n")
            for s in concept_stats:
                f.write(f"  concept={s['concept_index']:2d}  disrupt={s['disruption_rate']:.4f}  "
                        f"mean_eff={s['mean_effect']:+.4f}  n={s['n']}\n")

    return {
        "disruption_rate": disruption_rate,
        "flip_rate": flip_rate,
        "mean_effect_size": mean_eff,
        "sem_effect_size": sem_eff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "lang_swap_strength": lang_swap_strength,
        "n_lang_swap_features": len(lang_swap_features),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(gate: dict, iou_df: pd.DataFrame, bridge_df: pd.DataFrame,
                 c3_stats: dict, behaviour: str, out_dir: Path):
    print("\n" + "=" * 60)
    print("WRITING REPORT.md")
    print("=" * 60)

    n_bridges = len(bridge_df) if isinstance(bridge_df, pd.DataFrame) else 0

    # IoU summary
    if isinstance(iou_df, pd.DataFrame) and not iou_df.empty:
        max_iou_row = iou_df.loc[iou_df["iou"].idxmax()]
        min_iou_row = iou_df.loc[iou_df["iou"].idxmin()]
        mean_iou = iou_df["iou"].mean()
        # Middle layers (12–20) vs early+late
        mid = iou_df[iou_df["layer"].between(12, 20)]
        early_late = iou_df[~iou_df["layer"].between(12, 20)]
        mid_mean_iou = mid["iou"].mean() if not mid.empty else float("nan")
        el_mean_iou  = early_late["iou"].mean() if not early_late.empty else float("nan")
    else:
        max_iou_row = {"layer": "N/A", "iou": float("nan")}
        min_iou_row = {"layer": "N/A", "iou": float("nan")}
        mean_iou = mid_mean_iou = el_mean_iou = float("nan")

    gate_ok = gate.get("gate_status", "UNKNOWN")
    lang_en = gate.get("lang_stats", {}).get("en", {})
    lang_fr = gate.get("lang_stats", {}).get("fr", {})
    dr = c3_stats.get("disruption_rate", float("nan"))

    lss = c3_stats.get("lang_swap_strength", float("nan"))
    n_lsf = c3_stats.get("n_lang_swap_features", "N/A")

    # Anthropic mapping table
    anthropic_map = f"""
## Anthropic → Ours: Match vs Mismatch

### Matches (after per-feature conversion)
| Aspect | Anthropic | Ours |
|---|---|---|
| Intervention type | Per-feature causal (SAE feature ablation/patching) | Per-feature causal (transcoder feature ablation/patching) ✓ |
| Language pairs | EN + FR (antonym task) | EN + FR (antonym task) ✓ |
| Intervention target | C3: patch EN features into FR context | C3: patch EN features into FR context ✓ |
| Bridge features | Consistent negative effect in both languages | Consistent negative mean_effect in EN + FR ✓ |

### Mismatches (documented; not changed)
| Aspect | Anthropic | Ours | Impact |
|---|---|---|---|
| Token positions | All positions in paragraph | Decision token only (last) | IoU less discriminative |
| Feature type | Sparse Autoencoder (SAE) features | Transcoder features | Different feature geometry |
| Graph topology | Full circuit (feature–feature edges) | Star (input→feature→output only) | Community detection trivial |
| Languages | EN + FR (+ possibly others) | EN + FR only | Narrower reproduction |
| N prompts | ~thousands (pre-trained circuit) | 48 (24 EN + 24 FR) | Smaller sample |

### Claim-level Results

| Anthropic Claim | Metric | Our Value | Status |
|---|---|---|---|
| (1) Language-specific features exist | Min per-layer IoU | {min_iou_row.get('iou', float('nan')):.4f} | PROXY — partial support |
| (2) Shared cross-lang features exist | Max per-layer IoU | {max_iou_row.get('iou', float('nan')):.4f} | PROXY — partial support |
| (3) Shared features in middle layers | IoU middle(12–20) vs early/late | {mid_mean_iou:.4f} vs {el_mean_iou:.4f} | PROXY — weak (decision token limits contrast) |
| (4) Bridge features degrade both langs | n bridge features / C3 lang-swap strength | {n_bridges} bridges; {lss:.3f} C3 disrupt frac | PARTIAL ✓ |
"""

    lines = [
        f"# Multilingual Circuits Analysis — {behaviour}",
        f"",
        f"Behaviour: `{behaviour}` | Split: train | n_prompts: 48 (24 EN + 24 FR)",
        f"",
        f"## Baseline Gate",
        f"",
        f"| Metric | Value | Threshold | Status |",
        f"|---|---|---|---|",
        f"| EN sign_accuracy | {lang_en.get('sign_accuracy', float('nan')):.4f} | ≥ 0.90 | "
        f"{'PASS' if lang_en.get('sign_accuracy', 0) >= 0.90 else 'FAIL'} |",
        f"| FR sign_accuracy | {lang_fr.get('sign_accuracy', float('nan')):.4f} | ≥ 0.75 | "
        f"{'PASS' if lang_fr.get('sign_accuracy', 0) >= 0.75 else 'FAIL'} |",
        f"| mean_norm_logprob_diff | {gate.get('overall_mean_norm', float('nan')):.4f} | ≥ 1.0 | "
        f"{'PASS' if gate.get('overall_mean_norm', 0) >= 1.0 else 'FAIL'} |",
        f"",
        f"**Overall gate: {gate_ok}**",
        f"",
        f"## C3 Patching (Language Swap EN→FR)",
        f"",
        f"| Metric | Value | Target |",
        f"|---|---|---|",
        f"| disruption_rate (effect < 0) | {dr:.4f} | ≥ 0.40 |",
        f"| flip_rate (sign_flipped) | {c3_stats.get('flip_rate', float('nan')):.4f} | report only |",
        f"| mean_effect_size ± SEM | {c3_stats.get('mean_effect_size', float('nan')):+.4f} ± "
        f"{c3_stats.get('sem_effect_size', float('nan')):.4f} | report only |",
        f"| 95% bootstrap CI | [{c3_stats.get('ci_lo', float('nan')):+.4f}, "
        f"{c3_stats.get('ci_hi', float('nan')):+.4f}] | — |",
        f"",
        f"**C3 target met: {'YES' if dr >= 0.40 else 'NO'}**",
        f"",
        f"## Per-Layer IoU (EN vs FR feature activation sets)",
        f"",
        f"Mean IoU: {mean_iou:.4f}",
        f"Max IoU layer: {max_iou_row.get('layer', 'N/A')} (IoU = {max_iou_row.get('iou', float('nan')):.4f})",
        f"Min IoU layer: {min_iou_row.get('layer', 'N/A')} (IoU = {min_iou_row.get('iou', float('nan')):.4f})",
        f"Middle layers (12–20) mean IoU: {mid_mean_iou:.4f}",
        f"Early/late layers mean IoU:     {el_mean_iou:.4f}",
        f"",
        f"See `iou_per_layer.csv` for full per-layer breakdown.",
        f"",
        f"## Bridge Features (Claim 4)",
        f"",
        f"Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.",
        f"",
        f"Total graph features: N/A (see bridge_features.csv)",
        f"Bridge features:      {n_bridges}",
        f"",
        f"See `bridge_features_only.csv` for details.",
        f"",
        anthropic_map,
        f"",
        f"## Notes",
        f"",
        f"- IoU uses top-50 transcoder features at last (decision) token position per prompt.",
        f"- Bridge features require consistent negative mean_effect in BOTH languages;",
        f"  score = min(|mean_effect_en|, |mean_effect_fr|).",
        f"- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).",
        f"  A per-PAIR disruption_rate (any layer) would be higher.",
        f"- (1) and (2) are PROXY measures relative to Anthropic (who use token-level",
        f"  activation sets across full paragraphs; we use attribution graph features).",
    ]

    report_path = out_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multilingual circuits analysis")
    parser.add_argument("--behaviour", default="multilingual_circuits")
    parser.add_argument("--split",     default="train")
    parser.add_argument("--n_en",      type=int, default=24,
                        help="Number of EN prompts in train split (default: 24)")
    parser.add_argument("--n_fr",      type=int, default=24,
                        help="Number of FR prompts in train split (default: 24)")
    parser.add_argument("--layers",    nargs="+", type=int,
                        default=list(range(10, 26)),
                        help="Layers to compute IoU for (default: 10-25)")
    args = parser.parse_args()

    P = get_paths(args.behaviour, args.split)
    out_dir = P["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multilingual Circuits Analysis")
    print(f"  Behaviour: {args.behaviour}  Split: {args.split}")
    print(f"  Output:    {out_dir}")

    # 1. Baseline gate
    gate = check_baseline_gate(P["baseline_csv"], out_dir)

    # 2+3. Per-layer IoU
    iou_df = compute_iou(
        P["features_dir"], args.behaviour, args.split,
        n_en=args.n_en, n_fr=args.n_fr,
        layers=args.layers, out_dir=out_dir,
    )

    # 4. Bridge features
    bridge_df = find_bridge_features(P["ablation_csv"], P["train_jsonl"],
                                     n_en=args.n_en, out_dir=out_dir)

    # C3 patching stats
    c3_stats = analyze_c3_patching(P["c3_csv"], P["train_jsonl"], out_dir)

    # Final report
    write_report(gate, iou_df, bridge_df, c3_stats, args.behaviour, out_dir)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Results in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
