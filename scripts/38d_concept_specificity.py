"""
Script 38d: Concept-Specificity test (Variant B for k=30).

Implements bottom-up concept discovery for each subcluster:

  Step 1: For each cluster G, compute group-level mean activation μ_g
          for all 148 semantic-equivalence-groups.
  Step 2: Identify two extremes by cluster's own activation pattern:
            TOP    = top-K groups by μ_g   (cluster fires positively)
            BOTTOM = bot-K groups by μ_g   (cluster fires negatively / weakly)
          These extremes define what the cluster "is about" data-driven.
  Step 3: Test that the activation difference between TOP and BOTTOM
          is robust to paraphrase noise via ICC-style ratio:
            σ²_concept = Var(μ_TOP - μ_BOTTOM)               between-set
            σ²_within  = mean within-cue paraphrase variance for TOP ∪ BOTTOM
            CS_ICC     = σ²_concept / (σ²_concept + σ²_within)

  Cross-validation (de-circularize):
            5-fold split of paraphrases inside each cue group.
            TOP/BOTTOM identified on TRAIN halves;
            CS_ICC evaluated on TEST halves.
            If TEST_CS_ICC ≈ TRAIN_CS_ICC → real concept (not overfit to noise).

  Comparison with predefined V_h:
            For each cluster, report V_h_overlap = fraction of TOP groups
            that are α-correct (vs β-correct). If overlap ≈ 1 (or ≈ 0),
            the cluster's concept ≈ V_h. If overlap ≈ 0.5, the cluster
            encodes a sub-concept orthogonal to V_h.

Inputs (k=30):
  data/analysis/runD_v2/activations/activation_matrix.npy
  data/analysis/runD_v2/activations/feature_ids.txt
  data/analysis/runD_v2/activations/prompt_idxs.txt
  data/analysis/runD_v2/carrier_stability/subgroup_decomp/feature_subgroup_assignments.csv
  data/analysis/runD_v2/grouping/prompt_metadata.csv

Outputs:
  data/analysis/runD_v2/carrier_stability/subgroup_decomp/concept_specificity.csv
  data/analysis/runD_v2/carrier_stability/subgroup_decomp/concept_specificity_summary.json
  data/analysis/runD_v2/carrier_stability/subgroup_decomp/cluster_top_bottom_groups.csv
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--act_dir",   type=Path, default=ROOT / "data/analysis/runD_v2/activations")
ap.add_argument("--group_dir", type=Path, default=ROOT / "data/analysis/runD_v2/grouping")
ap.add_argument("--cluster_csv", type=Path,
                default=ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp/feature_subgroup_assignments.csv")
ap.add_argument("--out_dir",   type=Path,
                default=ROOT / "data/analysis/runD_v2/carrier_stability/subgroup_decomp")
ap.add_argument("--top_k",     type=int, default=10,
                help="number of groups in TOP/BOTTOM extremes (default 10 of 148)")
ap.add_argument("--n_folds",   type=int, default=5)
ap.add_argument("--seed",      type=int, default=42)
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ── Load ─────────────────────────────────────────────────────────────────────
act = np.load(args.act_dir / "activation_matrix.npy")  # (n_feat, n_prompts)
feat_ids = (args.act_dir / "feature_ids.txt").read_text().strip().split("\n")
prompt_idxs = [int(x) for x in (args.act_dir / "prompt_idxs.txt").read_text().strip().split("\n")]
print(f"Activations: {act.shape}")

cl = pd.read_csv(args.cluster_csv)
fid_to_cid = dict(zip(cl["feature_id"], cl["subgroup_cluster"]))
clusters = defaultdict(list)
for f, c in fid_to_cid.items():
    clusters[c].append(f)
print(f"Subclusters: {len(clusters)} (k=30 subgroup decomp)")

pm = pd.read_csv(args.group_dir / "prompt_metadata.csv").set_index("prompt_idx")
feat_to_row = {f: i for i, f in enumerate(feat_ids)}

# For each prompt: (group_id, V_h, paraphrase_variant_idx)
groups   = np.array([str(pm.loc[p, "semantic_equiv_group"]) for p in prompt_idxs])
vh       = np.array([str(pm.loc[p, "correct_answer"])       for p in prompt_idxs])
variants = np.array([int(pm.loc[p, "wording_variant"])      for p in prompt_idxs])

unique_groups = sorted(set(groups))
print(f"Cue groups: {len(unique_groups)}  α-prompts: {(vh=='alpha').sum()}  β: {(vh=='beta').sum()}")


def split_paraphrases_kfold(k: int = 5):
    """Return list of (train_mask, test_mask) splits over PROMPTS, where each
    cue group's paraphrases are stratified across folds."""
    folds = []
    fold_idx = np.zeros(len(prompt_idxs), dtype=int)
    for g in unique_groups:
        mask = groups == g
        idxs = np.where(mask)[0]
        rng_local = np.random.default_rng(args.seed + hash(g) % 1000)
        rng_local.shuffle(idxs)
        for i, p_idx in enumerate(idxs):
            fold_idx[p_idx] = i % k
    for f in range(k):
        test_mask = fold_idx == f
        train_mask = ~test_mask
        folds.append((train_mask, test_mask))
    return folds


folds = split_paraphrases_kfold(args.n_folds)


def group_means(A_G: np.ndarray, mask: np.ndarray = None):
    """Return dict {group_id: mean activation across prompts in group ∩ mask}."""
    if mask is None:
        mask = np.ones_like(A_G, dtype=bool)
    out = {}
    for g in unique_groups:
        sel = (groups == g) & mask
        if sel.sum() > 0:
            out[g] = float(A_G[sel].mean())
    return out


def within_group_var(A_G: np.ndarray, group_set, mask: np.ndarray = None):
    """Mean within-cue paraphrase variance across groups in group_set."""
    if mask is None:
        mask = np.ones_like(A_G, dtype=bool)
    vars_ = []
    for g in group_set:
        sel = (groups == g) & mask
        if sel.sum() > 1:
            vars_.append(float(np.var(A_G[sel], ddof=1)))
    return float(np.mean(vars_)) if vars_ else float("nan")


def compute_cs_icc(A_G, top_set, bot_set, mask=None):
    """Compute CS-ICC given pre-identified TOP/BOTTOM group sets."""
    mu_top  = [m for g, m in group_means(A_G, mask).items() if g in top_set]
    mu_bot  = [m for g, m in group_means(A_G, mask).items() if g in bot_set]
    if not mu_top or not mu_bot:
        return float("nan"), float("nan"), float("nan")
    sig = float(np.mean(mu_top) - np.mean(mu_bot))
    sigma2_between = sig**2 / 2.0   # variance of 2 means relative to their grand mean
    within_top = within_group_var(A_G, top_set, mask)
    within_bot = within_group_var(A_G, bot_set, mask)
    within_avg = np.nanmean([within_top, within_bot])
    if sigma2_between + within_avg < 1e-12:
        return float("nan"), sig, within_avg
    cs_icc = sigma2_between / (sigma2_between + within_avg)
    return float(cs_icc), float(sig), float(within_avg)


# ── Per cluster analysis ─────────────────────────────────────────────────────
results = []
top_bottom_dump = []

for cid in sorted(clusters.keys()):
    fids = clusters[cid]
    rows = [feat_to_row[f] for f in fids if f in feat_to_row]
    if not rows:
        continue
    A_G = act[rows, :].mean(axis=0)   # (n_prompts,)

    # ── identify TOP/BOTTOM on FULL data (for reporting + diagnostic) ───────
    mu_full = group_means(A_G)
    sorted_g = sorted(mu_full.items(), key=lambda x: x[1])
    bottom_g = set([g for g, _ in sorted_g[:args.top_k]])
    top_g    = set([g for g, _ in sorted_g[-args.top_k:]])

    cs_icc_full, sig_full, within_full = compute_cs_icc(A_G, top_g, bottom_g)

    # ── V_h composition of TOP and BOTTOM groups ────────────────────────────
    def vh_frac_alpha(group_set):
        # fraction of α-correct cue groups in set
        alphas = sum(1 for g in group_set
                     if (vh[groups == g] == "alpha").mean() > 0.5)
        return alphas / len(group_set) if group_set else 0.0
    top_alpha_frac = vh_frac_alpha(top_g)
    bot_alpha_frac = vh_frac_alpha(bottom_g)

    # ── Cross-validation: identify TOP/BOTTOM on TRAIN, test on TEST ────────
    cs_icc_test = []
    for train_mask, test_mask in folds:
        mu_train = group_means(A_G, train_mask)
        sorted_t = sorted(mu_train.items(), key=lambda x: x[1])
        bot_train = set([g for g, _ in sorted_t[:args.top_k]])
        top_train = set([g for g, _ in sorted_t[-args.top_k:]])
        cs_test, _, _ = compute_cs_icc(A_G, top_train, bot_train, test_mask)
        if not np.isnan(cs_test):
            cs_icc_test.append(cs_test)
    cs_icc_test_mean = float(np.mean(cs_icc_test)) if cs_icc_test else float("nan")

    # ── Random-baseline: pick random partition of groups ────────────────────
    cs_icc_rand = []
    for _ in range(20):
        shuf = list(unique_groups)
        rng.shuffle(shuf)
        rand_bot = set(shuf[:args.top_k])
        rand_top = set(shuf[-args.top_k:])
        cs_r, _, _ = compute_cs_icc(A_G, rand_top, rand_bot)
        if not np.isnan(cs_r):
            cs_icc_rand.append(cs_r)
    cs_icc_rand_mean = float(np.mean(cs_icc_rand)) if cs_icc_rand else float("nan")

    # ── Compare with V_h-ICC (formula 4.1) ──────────────────────────────────
    mu_alpha = float(A_G[vh == "alpha"].mean())
    mu_beta  = float(A_G[vh == "beta"].mean())
    sigma2_Vh = (mu_alpha - mu_beta) ** 2 / 2.0
    sigma2_para = within_group_var(A_G, unique_groups)
    vh_icc = sigma2_Vh / (sigma2_Vh + sigma2_para) if (sigma2_Vh + sigma2_para) > 1e-12 else float("nan")

    # ── Concept interpretation ──────────────────────────────────────────────
    # If TOP and BOTTOM are pure-α/pure-β (or close) → cluster ≈ V_h detector
    # If mixed → cluster encodes sub-concept
    if abs(top_alpha_frac - 0.5) > 0.4 and abs(bot_alpha_frac - 0.5) > 0.4 and \
       np.sign(top_alpha_frac - 0.5) == -np.sign(bot_alpha_frac - 0.5):
        concept_type = "V_h-detector"
    elif abs(top_alpha_frac - 0.5) < 0.3 and abs(bot_alpha_frac - 0.5) < 0.3:
        concept_type = "Sub-concept (mixed V_h)"
    else:
        concept_type = "Partial V_h-correlation"

    # ── Save diagnostic ──────────────────────────────────────────────────────
    for g in top_g:
        cue_label = pm[pm["semantic_equiv_group"] == g]["cue_label"].iloc[0] \
                    if not pm[pm["semantic_equiv_group"] == g].empty else "—"
        top_bottom_dump.append(dict(cluster_id=cid, set="TOP", group_id=g,
                                    cue_label=cue_label, mu_g=mu_full[g],
                                    correct_answer=pm[pm["semantic_equiv_group"] == g]["correct_answer"].iloc[0]))
    for g in bottom_g:
        cue_label = pm[pm["semantic_equiv_group"] == g]["cue_label"].iloc[0] \
                    if not pm[pm["semantic_equiv_group"] == g].empty else "—"
        top_bottom_dump.append(dict(cluster_id=cid, set="BOTTOM", group_id=g,
                                    cue_label=cue_label, mu_g=mu_full[g],
                                    correct_answer=pm[pm["semantic_equiv_group"] == g]["correct_answer"].iloc[0]))

    results.append(dict(
        cluster_id=cid, n_features=len(rows),
        cs_icc_full=round(cs_icc_full, 4),
        cs_icc_test_cv=round(cs_icc_test_mean, 4),
        cs_icc_random_baseline=round(cs_icc_rand_mean, 4),
        cs_icc_minus_random=round(cs_icc_full - cs_icc_rand_mean, 4),
        signal_top_minus_bot=round(sig_full, 4),
        within_paraphrase_var=round(within_full, 6),
        vh_icc=round(vh_icc, 4),
        mu_alpha=round(mu_alpha, 4),
        mu_beta=round(mu_beta, 4),
        top_alpha_frac=round(top_alpha_frac, 3),
        bot_alpha_frac=round(bot_alpha_frac, 3),
        concept_type=concept_type,
        passes_cs_icc=(cs_icc_full >= 0.5),
        cv_consistent=(abs(cs_icc_full - cs_icc_test_mean) < 0.15),
    ))

df = pd.DataFrame(results).sort_values("cs_icc_full", ascending=False)
df.to_csv(args.out_dir / "concept_specificity.csv", index=False)

dump = pd.DataFrame(top_bottom_dump)
dump.to_csv(args.out_dir / "cluster_top_bottom_groups.csv", index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"CONCEPT-SPECIFICITY per subcluster (k=30, top_k={args.top_k})")
print(f"{'='*100}\n")
print(df[["cluster_id", "n_features", "cs_icc_full", "cs_icc_test_cv",
         "cs_icc_random_baseline", "vh_icc",
         "top_alpha_frac", "bot_alpha_frac", "concept_type",
         "passes_cs_icc", "cv_consistent"]].to_string(index=False))

n_pass = int(df["passes_cs_icc"].sum())
n_cv   = int(df["cv_consistent"].sum())
n_vh   = int((df["concept_type"] == "V_h-detector").sum())
n_sub  = int((df["concept_type"] == "Sub-concept (mixed V_h)").sum())
n_part = int((df["concept_type"] == "Partial V_h-correlation").sum())
print(f"\nPASS CS-ICC ≥ 0.5:  {n_pass}/{len(df)}")
print(f"CV-consistent:      {n_cv}/{len(df)}")
print(f"Concept type:  V_h-detector={n_vh}  sub-concept={n_sub}  partial={n_part}")
print(f"Mean CS-ICC (full): {df['cs_icc_full'].mean():.3f}")
print(f"Mean CS-ICC (CV):   {df['cs_icc_test_cv'].mean():.3f}")
print(f"Mean random baseline: {df['cs_icc_random_baseline'].mean():.3f}")
print(f"Mean V_h-ICC (4.1):   {df['vh_icc'].mean():.3f}")

summary = dict(
    k=30, top_k=args.top_k, n_folds=args.n_folds,
    n_clusters=len(df), n_pass=n_pass, n_cv_consistent=n_cv,
    n_Vh_detector=n_vh, n_subconcept=n_sub, n_partial=n_part,
    mean_cs_icc_full=round(float(df["cs_icc_full"].mean()), 4),
    mean_cs_icc_cv=round(float(df["cs_icc_test_cv"].mean()), 4),
    mean_cs_icc_random=round(float(df["cs_icc_random_baseline"].mean()), 4),
    mean_vh_icc=round(float(df["vh_icc"].mean()), 4),
)
with open(args.out_dir / "concept_specificity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {args.out_dir}/concept_specificity.{{csv,json}} + cluster_top_bottom_groups.csv")
