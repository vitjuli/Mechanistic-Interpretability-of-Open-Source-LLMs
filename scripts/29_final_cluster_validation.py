"""
29_final_cluster_validation.py

Computes:
  A. Cross-form reuse metrics for all 11 co-importance Louvain clusters
  B. Final cluster validation table (acceptance criteria)
  C. Causal-module classification table

Outputs:
  data/results/cluster_semantics/cluster_reuse_metrics.csv
  data/results/cluster_semantics/final_cluster_validation_table.csv
  (figures written to docs/ by the main document writer)
"""

import json
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
G       = ROOT / "data/results/grouping"
CS      = ROOT / "data/results/cluster_semantics"
CJ      = ROOT / "data/results/cluster_joint_ablation"
CLUST   = ROOT / "data/results/clustering"

# ── Load data ──────────────────────────────────────────────────────────────────
fp   = pd.read_csv(G / "feature_prompt_contributions.csv")
pm   = pd.read_csv(G / "prompt_metadata.csv")
cps  = pd.read_csv(CS / "cluster_prompt_scores.csv")    # cluster × prompt scores
coh  = pd.read_csv(CS / "cluster_pairwise_coherence.csv")
cfs  = pd.read_csv(CS / "cluster_feature_summary.csv")
syn  = pd.read_csv(CJ / "synergy_summary.csv")
iby  = pd.read_csv(CJ / "interaction_by_answer.csv")
ppi  = pd.read_csv(CJ / "per_prompt_interaction.csv")
ab   = pd.read_csv(CJ / "analytic_bounds.csv")
stab = pd.read_csv(CS / "cluster_enrichment_stability_summary.csv")
cl   = pd.read_csv(CLUST / "cluster_labels.csv")

# Merge cluster_id into fp
cl_col = "coimp_louvain"
cl["feature_id"] = cl["feature_id"].astype(str)
fp["feature_id"] = fp["feature_id"].astype(str)
fp = fp.merge(cl[["feature_id", cl_col]].rename(columns={cl_col: "cluster_id"}),
              on="feature_id", how="left")

N_PROMPTS = 470
CLUSTER_IDS = sorted(cps["cluster_id"].unique())

print("Loaded all data.")
print(f"  {len(fp)} contribution rows | {fp['feature_id'].nunique()} features | "
      f"{fp['prompt_idx'].nunique()} prompts | {fp['cluster_id'].nunique()} clusters")

# ─────────────────────────────────────────────────────────────────────────────
# A. Cross-form reuse metrics
# ─────────────────────────────────────────────────────────────────────────────

def gini_simpson(weights: np.ndarray) -> float:
    """Gini-Simpson diversity index: 1 - sum(p_i^2). Range [0,1]."""
    if weights.sum() == 0:
        return 0.0
    p = weights / weights.sum()
    return float(1.0 - (p**2).sum())

def weighted_entropy(weights: np.ndarray) -> float:
    """Entropy of a weight distribution. Range [0, log(N)]."""
    if weights.sum() == 0:
        return 0.0
    p = weights / weights.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# For each cluster, compute reuse metrics
reuse_rows = []

for cid in CLUSTER_IDS:
    sub_fp = fp[fp["cluster_id"] == cid].copy()
    # cps already has level/correct_answer/cue_label; merge extra pm cols
    extra_pm_cols = [c for c in ["cue_type","latent_state_target","is_kw_variant",
                                  "is_anchor","is_auxiliary","keyword_free","has_alpha_keyword",
                                  "has_beta_keyword","contrastive_pair_id","wording_variant",
                                  "semantic_equiv_group"]
                     if c in pm.columns]
    sub_pm = cps[cps["cluster_id"] == cid].merge(
        pm[["prompt_idx"] + extra_pm_cols], on="prompt_idx", how="left")

    n_feats = cfs[cfs["cluster_id"] == cid].shape[0]
    total_abs_weight = sub_pm["cluster_mean_abs_effect"].sum()

    # ── R1: Level coverage ───────────────────────────────────────────────────
    # Does the cluster fire above median across all 4 levels?
    level_weights = sub_pm.groupby("level_x")["cluster_mean_abs_effect"].sum() \
        if "level_x" in sub_pm.columns \
        else sub_pm.groupby("level")["cluster_mean_abs_effect"].sum()
    level_weights = level_weights.reindex(["1","2","3","AUX"], fill_value=0)
    # fraction of levels covered (weight > 10% of max)
    threshold = level_weights.max() * 0.10
    n_levels_covered = (level_weights > threshold).sum()
    level_diversity = gini_simpson(level_weights.values)

    # ── R2: Group diversity within dominant latent target ───────────────────
    # For the dominant answer type, how many distinct groups appear in decisive prompts?
    dom_ans = sub_pm.groupby("correct_answer")["cluster_mean_abs_effect"].sum().idxmax()
    dom_sub = sub_pm[sub_pm["correct_answer"] == dom_ans]
    group_weights = dom_sub.groupby("group_id")["cluster_mean_abs_effect"].sum()
    n_groups_dom = len(group_weights)
    n_total_groups_dom = pm[pm["correct_answer"] == dom_ans]["group_id"].nunique()
    group_coverage_frac = n_groups_dom / n_total_groups_dom if n_total_groups_dom > 0 else 0
    group_diversity_dom = gini_simpson(group_weights.values)

    # ── R3: CUE-type diversity ────────────────────────────────────────────────
    cue_weights = sub_pm.groupby("cue_label")["cluster_mean_abs_effect"].sum()
    n_cues = len(cue_weights)
    cue_diversity = gini_simpson(cue_weights.values)

    # ── R4: KW / keyword-free reuse ─────────────────────────────────────────
    # Do keyword-free variants activate the cluster at all?
    kw_free  = sub_pm[sub_pm["keyword_free"] == True]["cluster_mean_abs_effect"].mean()
    kw_pres  = sub_pm[sub_pm["keyword_free"] == False]["cluster_mean_abs_effect"].mean()
    kw_ratio = kw_free / kw_pres if (kw_pres and kw_pres > 0) else None
    # Ratio < 0.5: keyword-free is much weaker → surface-cue sensitive
    # Ratio ≥ 0.5: keyword-free maintains activation → concept-level reuse

    # ── R5: Cross-level ANSWER TARGET coverage ─────────────────────────────
    # For each level, does the cluster fire on prompts with the dominant answer type?
    # Metric: for the dominant answer, fraction of (level × has_top50_prompt) that are present.
    top50 = sub_pm.nlargest(50, "cluster_mean_abs_effect")
    dom_levels_with_prompts = top50[top50["correct_answer"] == dom_ans]["level"].nunique()
    total_levels_dom = sub_pm[sub_pm["correct_answer"] == dom_ans]["level"].nunique()
    cross_level_coverage = dom_levels_with_prompts / total_levels_dom if total_levels_dom > 0 else 0.0

    # Also: effective kw-free ratio at cluster level
    # How consistent are effects across keyword-free vs keyword-present variants of the same concept?
    # Use the top-50 decisive prompts: kw-free count / total count ratio
    top50_kw_free  = top50[top50.get("keyword_free", pd.Series(dtype=bool)) == True].shape[0] if "keyword_free" in top50.columns else 0
    top50_kw_pres  = top50[top50.get("keyword_free", pd.Series(dtype=bool)) == False].shape[0] if "keyword_free" in top50.columns else 0
    # Expected: ~30% of prompts are keyword-free
    kw_free_frac   = top50_kw_free / 50 if 50 > 0 else 0
    kw_base_frac   = sub_pm["keyword_free"].mean() if "keyword_free" in sub_pm.columns else 0.3
    cross_level_cue_jaccard = cross_level_coverage  # repurpose this column

    # ── R6: Cross-form reuse per top-50 decisive prompts ────────────────────
    # Among the top-50, how many distinct wording_variants appear for the same semantic_equiv_group?
    if "wording_variant" in sub_pm.columns and "semantic_equiv_group" in sub_pm.columns:
        top50_pm = sub_pm.nlargest(50, "cluster_mean_abs_effect")
        # For each semantic group, count distinct variants that appear
        variant_counts = top50_pm.groupby("semantic_equiv_group")["wording_variant"].nunique()
        n_multi_variant = (variant_counts >= 2).sum()
        n_sem_groups = len(variant_counts)
        multi_variant_frac = n_multi_variant / n_sem_groups if n_sem_groups > 0 else 0.0
    else:
        multi_variant_frac = None

    # ── R7: Answer-conditioned coverage (% of α-groups and β-groups covered) ─
    for ans in ["alpha", "beta"]:
        ans_sub = sub_pm[sub_pm["correct_answer"] == ans]
        groups_in_top50 = ans_sub.nlargest(50, "cluster_mean_abs_effect")["group_id"].nunique()
        total_groups_ans = pm[pm["correct_answer"] == ans]["group_id"].nunique()

    alpha_sub = sub_pm[sub_pm["correct_answer"] == "alpha"]
    beta_sub  = sub_pm[sub_pm["correct_answer"] == "beta"]
    alpha_group_coverage = alpha_sub.nlargest(50, "cluster_mean_abs_effect")["group_id"].nunique() / \
        max(pm[pm["correct_answer"] == "alpha"]["group_id"].nunique(), 1)
    beta_group_coverage  = beta_sub.nlargest(50, "cluster_mean_abs_effect")["group_id"].nunique() / \
        max(pm[pm["correct_answer"] == "beta"]["group_id"].nunique(), 1)

    # ── Composite reuse score ────────────────────────────────────────────────
    # Weighted combination:
    # 35% level diversity + 25% group diversity + 20% cue diversity
    # 10% cross-level coverage + 10% kw reuse
    kw_score = min(kw_ratio, 1.5) / 1.5 if kw_ratio is not None else 0.5
    composite_reuse = (
        0.35 * level_diversity +
        0.25 * group_diversity_dom +
        0.20 * cue_diversity +
        0.10 * cross_level_cue_jaccard +
        0.10 * kw_score
    )

    reuse_rows.append({
        "cluster_id":              cid,
        "n_features":              n_feats,
        "dominant_answer":         dom_ans,
        "n_levels_covered":        int(n_levels_covered),
        "level_diversity_gini":    round(level_diversity, 4),
        "group_coverage_frac":     round(group_coverage_frac, 4),
        "group_diversity_gini":    round(group_diversity_dom, 4),
        "n_cue_types":             int(n_cues),
        "cue_diversity_gini":      round(cue_diversity, 4),
        "kw_free_mean_abs":        round(float(kw_free), 4) if kw_free is not None else None,
        "kw_present_mean_abs":     round(float(kw_pres), 4) if kw_pres is not None else None,
        "kw_ratio":                round(float(kw_ratio), 4) if kw_ratio is not None else None,
        "cross_level_cue_jaccard": round(cross_level_cue_jaccard, 4),
        "multi_variant_frac":      round(multi_variant_frac, 4) if multi_variant_frac is not None else None,
        "alpha_group_coverage":    round(alpha_group_coverage, 4),
        "beta_group_coverage":     round(beta_group_coverage, 4),
        "composite_reuse_score":   round(composite_reuse, 4),
    })

df_reuse = pd.DataFrame(reuse_rows)
df_reuse.to_csv(CS / "cluster_reuse_metrics.csv", index=False)
print(f"\nSaved cluster_reuse_metrics.csv ({len(df_reuse)} rows)")
print(df_reuse[["cluster_id","n_features","n_levels_covered","level_diversity_gini",
                "group_diversity_gini","kw_ratio","cross_level_cue_jaccard",
                "composite_reuse_score"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# B. Final cluster validation table
# ─────────────────────────────────────────────────────────────────────────────

# Collect all evidence per cluster
def get_stab_verdict(cid, field, value):
    """Return enrichment_type for a (cluster, field, value) combination."""
    row = stab[(stab["cluster_id"] == cid) &
               (stab["metadata_field"] == field) &
               (stab["metadata_value"] == str(value))]
    return row["enrichment_type"].iloc[0] if len(row) > 0 else "absent"

def best_stable_motif(cid):
    """Return the highest-lift stable enrichment for a cluster."""
    rows = stab[(stab["cluster_id"] == cid) & (stab["enrichment_type"] == "stable")]
    if len(rows) == 0:
        return "none", None
    best = rows.nlargest(1, "best_lift").iloc[0]
    return f"{best['metadata_value']}({best['metadata_field']})", float(best["best_lift"])

# Thresholds for acceptance criteria
T = {
    "abs_cosine_min":     0.85,   # within-cluster abs cosine coherence
    "signed_cosine_min":  0.50,   # within-cluster signed coherence (directional)
    "coimp_min":          0.25,   # co-importance Jaccard
    "interaction_ratio_max": 0.50,# ratio below this = functionally redundant (causal module)
    "reuse_min":          0.40,   # composite reuse score minimum
    "n_stable_motifs_min": 2,     # minimum stable enrichment motifs
    "sign_flip_meaningful": 0.05, # actual sign flip rate meaningful (>5%)
}

# Criteria definitions:
# C1 (directional consistency): signed_cosine >= T["signed_cosine_min"] AND n_features >= 2
# C2 (coherence): abs_cosine >= T["abs_cosine_min"]
# C3 (semantic specificity): at least T["n_stable_motifs_min"] stable enrichments
# C4 (cross-form reuse): composite_reuse_score >= T["reuse_min"]
# C5 (causal impact): interaction_ratio <= T["interaction_ratio_max"] OR sign_flip_rate_joint > T["sign_flip_meaningful"]
# C6 (level stability): n_levels_covered >= 3

val_rows = []

for cid in CLUSTER_IDS:
    coh_row  = coh[coh["cluster_id"] == cid].iloc[0]
    syn_row  = syn[syn["cluster_id"] == cid].iloc[0]
    ab_row   = ab[ab["cluster_id"]   == cid].iloc[0]
    reuse_row = df_reuse[df_reuse["cluster_id"] == cid].iloc[0]
    cfs_rows = cfs[cfs["cluster_id"] == cid]
    n_feats  = len(cfs_rows)

    # Coherence
    abs_cos   = coh_row.get("mean_abs_cosine_within", float("nan"))
    sign_cos  = coh_row.get("mean_signed_cosine_within", float("nan"))
    coimp     = coh_row.get("mean_coimportance_within", float("nan"))
    layer_span= int(coh_row.get("max_layer", 0) - coh_row.get("min_layer", 0))

    # Interaction (causal)
    ratio     = float(syn_row["mean_interaction_ratio"])
    sfr_joint = float(syn_row["sign_flip_rate_joint"])
    frac_red  = float(syn_row["frac_redundant"])
    frac_syn  = float(syn_row["frac_synergistic"])
    ttest_p   = float(syn_row["ttest_vs_additive_p"])
    is_additive = n_feats == 1  # singletons are trivially additive

    # Synergy / causal type
    if n_feats == 1:
        causal_type = "singleton"
    elif ratio < 0.20 and frac_red > 0.95:
        causal_type = "redundant_bank"
    elif frac_syn > 0.10:
        causal_type = "mixed_push_pull"
    elif ratio < 0.50 and frac_red > 0.85:
        causal_type = "redundant_module"
    elif ratio > 0.80:
        causal_type = "additive_independent"
    else:
        causal_type = "partial_module"

    # Enrichment stability
    n_stable = len(stab[(stab["cluster_id"] == cid) & (stab["enrichment_type"] == "stable")])
    n_narrow = len(stab[(stab["cluster_id"] == cid) & (stab["enrichment_type"] == "narrow_decisive")])
    best_motif, best_lift = best_stable_motif(cid)

    # Orient delta from cps
    a = cps[(cps["cluster_id"]==cid) & (cps["correct_answer"]=="alpha")]["cluster_mean_effect"].mean()
    b = cps[(cps["cluster_id"]==cid) & (cps["correct_answer"]=="beta")]["cluster_mean_effect"].mean()
    orient_delta = float(a - b) if (not math.isnan(a) and not math.isnan(b)) else float("nan")

    # Reuse
    reuse_score = float(reuse_row["composite_reuse_score"])
    n_lev = int(reuse_row["n_levels_covered"])
    kw_ratio = reuse_row.get("kw_ratio")

    # Circuit / discriminator membership
    n_circuit = int(cfs_rows["is_circuit_feature"].sum()) if "is_circuit_feature" in cfs_rows.columns else 0
    n_alpha_d = int(cfs_rows["is_global_alpha_discrim"].sum()) if "is_global_alpha_discrim" in cfs_rows.columns else 0
    n_beta_d  = int(cfs_rows["is_global_beta_discrim"].sum()) if "is_global_beta_discrim" in cfs_rows.columns else 0

    # ── Criteria evaluation ──────────────────────────────────────────────────
    # NaN check helpers
    def ok(val, threshold, op):
        if math.isnan(val): return False
        return (val >= threshold) if op=="ge" else (val <= threshold)

    # C1: directional consistency
    c1_direct = ok(sign_cos, T["signed_cosine_min"], "ge") if n_feats >= 2 else True  # singletons trivially pass

    # C2: coherence (abs cosine)
    c2_cohere = ok(abs_cos, T["abs_cosine_min"], "ge") if n_feats >= 2 else True

    # C3: semantic specificity
    c3_semantic = n_stable >= T["n_stable_motifs_min"]

    # C4: cross-form reuse
    c4_reuse = reuse_score >= T["reuse_min"]

    # C5: causal impact — ratio ≤ 0.5, OR meaningful sign flips, OR significant ttest
    c5_causal = (not is_additive) and (
        ok(ratio, T["interaction_ratio_max"], "le") or
        sfr_joint > T["sign_flip_meaningful"] or
        ttest_p < 0.001
    )

    # C6: level stability (fires across multiple levels)
    c6_level = n_lev >= 3

    n_criteria_met = sum([c1_direct, c2_cohere, c3_semantic, c4_reuse, c5_causal, c6_level])

    # ── Final status assignment ──────────────────────────────────────────────
    # For accepted_candidate: must pass BOTH coherence criteria (c1+c2) AND causal (c5)
    # plus at least 2 of (c3, c4, c6).
    coherent = c1_direct and c2_cohere
    if n_feats == 1:
        if c3_semantic and c6_level and sfr_joint > T["sign_flip_meaningful"]:
            status = "partial_candidate"
        else:
            status = "descriptive_only"
    elif coherent and c5_causal and n_criteria_met >= 5:
        status = "accepted_candidate"
    elif coherent and c5_causal and c3_semantic:
        status = "accepted_candidate"
    elif c5_causal and c3_semantic and n_criteria_met >= 4:
        # Has causal + semantic but lacks full coherence (heterogeneous cluster)
        status = "partial_candidate"
    elif c3_semantic and n_criteria_met >= 3:
        status = "partial_candidate"
    elif n_criteria_met >= 2 and c3_semantic:
        status = "descriptive_only"
    else:
        status = "not_interpretable"

    # Short reason
    passing = [name for name, passed in [
        ("directional_consistency", c1_direct),
        ("abs_coherence", c2_cohere),
        ("semantic_specificity", c3_semantic),
        ("cross_form_reuse", c4_reuse),
        ("causal_impact", c5_causal),
        ("level_coverage", c6_level),
    ] if passed]
    failing = [name for name, passed in [
        ("directional_consistency", c1_direct),
        ("abs_coherence", c2_cohere),
        ("semantic_specificity", c3_semantic),
        ("cross_form_reuse", c4_reuse),
        ("causal_impact", c5_causal),
        ("level_coverage", c6_level),
    ] if not passed]

    if status == "accepted_candidate":
        reason = f"Passes {n_criteria_met}/6 criteria. Key: {', '.join(passing[:3])}"
    elif status == "partial_candidate":
        reason = f"Passes {n_criteria_met}/6. Fails: {', '.join(failing[:2])}"
    elif status == "descriptive_only":
        reason = f"Only {n_criteria_met}/6 criteria met; lacks causal evidence or reuse"
    else:
        reason = f"Only {n_criteria_met}/6; insufficient evidence for interpretation"

    # Semantic label
    LABELS = {
        0:  "Early β-charge detection (L10)",
        1:  "L11 lepton-class pair",
        2:  "Singleton L12 (β-antineutrino cue)",
        3:  "L13 β-attribution pair (role-label reversal)",
        4:  "L20 β-process attribution pair",
        5:  "Singleton L15 (baryon-4 / nuclear object)",
        6:  "L16 α-particle composition module",
        7:  "Multi-layer convergence hub (L14–L23)",
        8:  "Singleton L18 β⁻-emission discriminator",
        9:  "L19–L21 transitional bidirectional cluster",
        10: "L24–L25 output decision module",
    }

    val_rows.append({
        "cluster_id":                cid,
        "n_features":                n_feats,
        "layer_span":                layer_span,
        "semantic_label":            LABELS.get(cid, f"C{cid}"),
        "coherence_abs_cosine":      round(abs_cos, 4) if not math.isnan(abs_cos) else None,
        "coherence_signed_cosine":   round(sign_cos, 4) if not math.isnan(sign_cos) else None,
        "coherence_coimportance":    round(coimp, 4) if not math.isnan(coimp) else None,
        "orient_delta":              round(orient_delta, 4) if not math.isnan(orient_delta) else None,
        "semantic_enrichment_best":  best_motif,
        "semantic_enrichment_lift":  round(best_lift, 2) if best_lift else None,
        "n_stable_motifs":           n_stable,
        "n_narrow_motifs":           n_narrow,
        "cross_form_reuse_score":    round(reuse_score, 4),
        "n_levels_covered":          n_lev,
        "kw_ratio":                  round(float(kw_ratio), 3) if kw_ratio is not None and not math.isnan(float(kw_ratio)) else None,
        "contains_circuit_feature":  n_circuit > 0,
        "n_circuit_features":        n_circuit,
        "n_alpha_discrim":           n_alpha_d,
        "n_beta_discrim":            n_beta_d,
        "interaction_ratio":         round(ratio, 4),
        "frac_redundant":            round(frac_red, 4),
        "frac_synergistic":          round(frac_syn, 4),
        "sign_flip_rate_joint":      round(sfr_joint, 4),
        "ttest_p_vs_additive":       round(ttest_p, 5),
        "causal_type":               causal_type,
        "c1_directional":            c1_direct,
        "c2_coherence":              c2_cohere,
        "c3_semantic":               c3_semantic,
        "c4_reuse":                  c4_reuse,
        "c5_causal":                 c5_causal,
        "c6_level":                  c6_level,
        "n_criteria_met":            n_criteria_met,
        "final_status":              status,
        "short_reason":              reason,
    })

df_val = pd.DataFrame(val_rows)
df_val.to_csv(CS / "final_cluster_validation_table.csv", index=False)
print(f"\nSaved final_cluster_validation_table.csv ({len(df_val)} rows)")

print("\n=== FINAL VALIDATION TABLE ===")
display_cols = ["cluster_id","n_features","layer_span","interaction_ratio","frac_redundant",
                "n_stable_motifs","cross_form_reuse_score","n_levels_covered",
                "n_criteria_met","final_status"]
print(df_val[display_cols].to_string(index=False))

print("\n=== STATUS SUMMARY ===")
print(df_val["final_status"].value_counts().to_string())

print("\n=== CRITERIA BREAKDOWN ===")
criteria = ["c1_directional","c2_coherence","c3_semantic","c4_reuse","c5_causal","c6_level"]
for c in criteria:
    n_pass = df_val[c].sum()
    print(f"  {c}: {n_pass}/11 clusters pass")

print("\n=== CAUSAL TYPE ===")
print(df_val["causal_type"].value_counts().to_string())
