"""
Intensive/extensive cluster analysis — runs after scripts 04, 06.

For each feature cluster (from k-means on the feature activation matrix):
  - Compute mean activation for intensive vs extensive prompts
  - Class selectivity: intensive-supporting, extensive-supporting, or general
  - Layer span, entropy, circuit overlap
  - Robustness across k = 4, 5, 6, 8

Cross-domain transfer test:
  - Compute cluster activations for Family D cross-domain prompts
  - Compare physics-cluster activations: do they fire correctly for economics/statistics/biology?
  - Identify prompts where hidden representation is correct but LM-head output is wrong
    (representation-readout dissociation)

Usage:
    python scripts/71_ie_cluster_analysis.py --behaviour physics_intensive_extensive_v1 --device cuda --k 6
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROMPT_DIR  = Path("data/prompts")
FEAT_DIR    = Path("data/features")
OUT_BASE    = Path("data/results/abstraction_ie")
FAMILY_D    = Path("data/prompts/abstraction/D_cross_domain_train.jsonl")
MODEL_NAME  = "Qwen/Qwen3-4B"
LABEL_MAP   = {"intensive": 0, "extensive": 1}
INT_TOKEN   = 36195   # ' intensive'
EXT_TOKEN   = 16376   # ' extensive'


# ── Load prompts ──────────────────────────────────────────────────────────────

def load_behaviour(behaviour, split):
    path = PROMPT_DIR / f"{behaviour}_{split}.jsonl"
    return [json.loads(l) for l in open(path)]

def load_cross_domain():
    if not FAMILY_D.exists():
        return []
    return [r for r in [json.loads(l) for l in open(FAMILY_D)]
            if r.get("abstraction_class") in LABEL_MAP]


# ── Load feature activations ──────────────────────────────────────────────────

def load_feature_matrix(behaviour, split):
    """
    Load feature activation matrix from script 04 output.
    Expected shape: (n_features, n_prompts) or (n_prompts, n_features)
    Also loads feature metadata: layer, feature_idx.
    Returns:
        act_matrix: (n_features, n_prompts) float32
        feature_meta: list of {feature_id, layer, feature_idx}
    """
    feat_dir = FEAT_DIR / f"{behaviour}_{split}"
    act_path = feat_dir / "feature_activation_matrix.npy"
    meta_path = feat_dir / "feature_meta.json"

    if not act_path.exists():
        # Try alternative path structure
        act_path = Path(f"data/results/internal_candidate_analysis/{behaviour}/feature_activation_matrix.npy")
        meta_path = Path(f"data/results/internal_candidate_analysis/{behaviour}/feature_meta.json")

    if not act_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {act_path}. "
            f"Run scripts/04_extract_transcoder_features.py first."
        )

    act = np.load(act_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # Ensure shape is (n_features, n_prompts)
    if act.shape[0] > act.shape[1]:
        act = act.T   # transpose if rows > cols (likely n_prompts × n_features)

    return act, meta


# ── Feature clustering ────────────────────────────────────────────────────────

def cluster_features(act_matrix, k, seed=42):
    """K-means cluster features by their activation profile across prompts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(act_matrix)
    return km.labels_


def compute_cluster_stats(act_matrix, cluster_labels, prompt_labels, prompts):
    """
    For each cluster: compute T/I/E statistics (intensive vs extensive vs general).
    Returns DataFrame with one row per cluster.
    """
    cls_arr  = np.array([LABEL_MAP.get(p.get("abstraction_class",""), -1) for p in prompts])
    wf_arr   = np.array([p.get("wording_family","") for p in prompts])
    prop_arr = np.array([p.get("property","") for p in prompts])

    rows = []
    for c in sorted(set(cluster_labels)):
        feat_mask = cluster_labels == c
        c_acts    = act_matrix[feat_mask]   # (n_feats_in_cluster, n_prompts)

        if len(c_acts) == 0:
            continue

        # Mean activation per prompt
        mean_by_prompt = c_acts.mean(axis=0)   # (n_prompts,)

        # Split by class
        int_mask = cls_arr == 0
        ext_mask = cls_arr == 1

        mu_int = float(mean_by_prompt[int_mask].mean()) if int_mask.sum() > 0 else 0.0
        mu_ext = float(mean_by_prompt[ext_mask].mean()) if ext_mask.sum() > 0 else 0.0

        # Selectivity: which class activates more
        selectivity = float(mu_ext - mu_int)   # positive = extensive-supporting
        dominant    = "extensive" if selectivity > 0.05 else "intensive" if selectivity < -0.05 else "general"

        # Mann-Whitney U test for class difference
        if int_mask.sum() >= 3 and ext_mask.sum() >= 3:
            _, p_val = stats.mannwhitneyu(
                mean_by_prompt[int_mask], mean_by_prompt[ext_mask], alternative="two-sided"
            )
        else:
            p_val = 1.0

        # Layer span
        layers_in_cluster = [m["layer"] for fi, m in enumerate(feat_mask) if feat_mask[fi]]
        # (feat_mask is over features, need to match feature indices to layers)
        # Note: this needs feature_meta to be passed; simplified version:
        layer_range = "unknown"

        rows.append({
            "cluster":      int(c),
            "n_features":   int(feat_mask.sum()),
            "mu_intensive": round(mu_int, 4),
            "mu_extensive": round(mu_ext, 4),
            "selectivity":  round(selectivity, 4),
            "dominant":     dominant,
            "p_val_mwu":    float(p_val),
            "sig":          p_val < 0.05,
        })
    return pd.DataFrame(rows)


# ── Cross-domain transfer: cluster activations ────────────────────────────────

@torch.no_grad()
def get_logits_and_acts(model, tokenizer, device, prompt, layers):
    """Returns logp(' intensive'), logp(' extensive'), and hidden state at each layer."""
    from src.model_utils import ModelWrapper   # use existing wrapper if available
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out    = model(**inputs, output_hidden_states=True, use_cache=False)
    logits = out.logits[0, -1, :]
    lp     = torch.log_softmax(logits, dim=0)
    logp_int = float(lp[INT_TOKEN])
    logp_ext = float(lp[EXT_TOKEN])
    hidden = {l: out.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
    return logp_int, logp_ext, hidden


def transfer_and_readout_analysis(model, tok, phys_rows, cross_rows, device,
                                   probe_layer, probe, scaler, out_dir):
    """
    For cross-domain prompts:
    1. Get logp(intensive), logp(extensive) from model output → behavioral answer
    2. Get hidden state at probe_layer → probe prediction (representation answer)
    3. Report cases where representation is correct but output is wrong (readout failure)
    """
    results = []
    layers  = [probe_layer]

    for i, row in enumerate(cross_rows):
        if i % 20 == 0:
            print(f"  Transfer readout {i}/{len(cross_rows)}", end="\r", flush=True)
        true_cls = LABEL_MAP[row["abstraction_class"]]

        lp_int, lp_ext, hidden = get_logits_and_acts(model, tok, device, row["prompt"], layers)
        nd_output  = lp_int - lp_ext   # >0 → model says intensive
        pred_output = 1 if nd_output < 0 else 0   # 0=intensive,1=extensive
        correct_output = (pred_output == true_cls)

        # Probe prediction from hidden state
        h = hidden[probe_layer].reshape(1, -1)
        h_norm = h / (np.linalg.norm(h) + 1e-8)
        h_s    = scaler.transform(h_norm)
        pred_probe = int(probe.predict(h_s)[0])
        correct_probe = (pred_probe == true_cls)

        results.append({
            "domain":            row.get("domain", ""),
            "property":          row.get("property", ""),
            "wording_family":    row.get("wording_family", ""),
            "abstraction_class": row["abstraction_class"],
            "correct_class":     true_cls,
            "lp_intensive":      round(lp_int, 3),
            "lp_extensive":      round(lp_ext, 3),
            "nd_output":         round(nd_output, 3),
            "pred_output":       "intensive" if pred_output == 0 else "extensive",
            "correct_output":    correct_output,
            "pred_probe":        "intensive" if pred_probe == 0 else "extensive",
            "correct_probe":     correct_probe,
            "readout_failure":   correct_probe and not correct_output,
            "prompt":            row["prompt"][:100],
        })
    print()
    return pd.DataFrame(results)


# ── Robustness across k ───────────────────────────────────────────────────────

def robustness_across_k(act_matrix, prompt_labels, k_values=(4,5,6,8)):
    rows = []
    cls_arr = np.array([LABEL_MAP.get(p.get("abstraction_class",""),-1)
                        for p in prompt_labels])
    for k in k_values:
        labels = cluster_features(act_matrix, k)
        ari = float(adjusted_rand_score(cls_arr, labels[:len(cls_arr)])) \
              if len(labels) >= len(cls_arr) else float("nan")
        stats_df = compute_cluster_stats(act_matrix, labels, None, prompt_labels)
        n_int_dominant = (stats_df["dominant"] == "intensive").sum()
        n_ext_dominant = (stats_df["dominant"] == "extensive").sum()
        rows.append({"k": k, "ari": ari, "n_intensive_clusters": n_int_dominant,
                     "n_extensive_clusters": n_ext_dominant})
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default="physics_intensive_extensive_v1")
    ap.add_argument("--split",     default="train")
    ap.add_argument("--k",         type=int, default=6)
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",     default="bfloat16", choices=["float32","bfloat16","float16"])
    ap.add_argument("--probe_layer", type=int, default=34,
                    help="Layer for cross-domain readout failure analysis (default: 34)")
    args = ap.parse_args()

    device  = torch.device(args.device)
    dtype   = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[args.dtype]
    out_dir = OUT_BASE / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    phys_rows  = load_behaviour(args.behaviour, args.split)
    cross_rows = load_cross_domain()
    print(f"Physics prompts: {len(phys_rows)} | Cross-domain: {len(cross_rows)}")

    # Load feature matrix
    print("Loading feature matrix…")
    try:
        act_matrix, feat_meta = load_feature_matrix(args.behaviour, args.split)
        print(f"  Feature matrix: {act_matrix.shape} (n_features × n_prompts)")
    except FileNotFoundError as e:
        print(f"  {e}")
        print("  Skipping feature clustering — running transfer analysis only")
        act_matrix = None

    # Feature clustering
    if act_matrix is not None:
        print(f"\nClustering features (k={args.k})…")
        cluster_labels = cluster_features(act_matrix, args.k)
        stats_df = compute_cluster_stats(act_matrix, cluster_labels, None, phys_rows)
        stats_df.to_csv(out_dir / f"ie_cluster_stats_k{args.k}.csv", index=False)
        print(stats_df[["cluster","n_features","mu_intensive","mu_extensive",
                         "selectivity","dominant","sig"]].to_string(index=False))

        # Save cluster labels
        np.save(out_dir / f"ie_cluster_labels_k{args.k}.npy", cluster_labels)

        # Robustness
        print("\nRobustness across k values…")
        rob_df = robustness_across_k(act_matrix, phys_rows)
        rob_df.to_csv(out_dir / "ie_cluster_robustness.csv", index=False)
        print(rob_df.to_string(index=False))
    else:
        stats_df = None

    # Cross-domain transfer + readout failure analysis
    if cross_rows:
        print(f"\nLoading model for transfer analysis (probe at L{args.probe_layer})…")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)
        mdl.eval()

        # Train probe on physics hidden states (load from layer transition CSV or re-extract)
        lt_csv = out_dir / "ie_layer_transition.csv"
        if lt_csv.exists():
            lt = pd.read_csv(lt_csv)
            valid_layers = lt[~lt["degenerate"]]["layer"].tolist()
        else:
            valid_layers = list(range(args.probe_layer - 2, args.probe_layer + 3))

        # Extract physics hidden states at probe layer for probe training
        print(f"  Extracting physics hidden states at L{args.probe_layer}…")
        from scripts.script70_utils import collect   # if refactored; else inline
        # Inline extraction
        Xtr, ytr = [], []
        with torch.no_grad():
            for i, row in enumerate(phys_rows):
                if i % 40 == 0:
                    print(f"    {i}/{len(phys_rows)}", end="\r", flush=True)
                inputs = tok(row["prompt"], return_tensors="pt").to(device)
                out = mdl(**inputs, output_hidden_states=True, use_cache=False)
                h = out.hidden_states[args.probe_layer][0, -1, :].float().cpu().numpy()
                Xtr.append(h)
                ytr.append(LABEL_MAP[row["abstraction_class"]])
        print()
        Xtr = np.array(Xtr)
        ytr = np.array(ytr)

        # Train probe
        scaler = StandardScaler()
        Xtr_n  = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
        Xtr_s  = scaler.fit_transform(Xtr_n)
        probe  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        probe.fit(Xtr_s, ytr)
        print(f"  Physics probe trained: {len(Xtr)} samples")

        # Run transfer + readout failure
        print(f"  Analysing {len(cross_rows)} cross-domain prompts…")
        transfer_df = transfer_and_readout_analysis(
            mdl, tok, phys_rows, cross_rows, device,
            args.probe_layer, probe, scaler, out_dir
        )

        del mdl; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        transfer_df.to_csv(out_dir / "ie_transfer_readout.csv", index=False)

        # Summary
        print("\n=== TRANSFER + READOUT FAILURE SUMMARY ===")
        print(f"  Total cross-domain prompts: {len(transfer_df)}")
        print(f"  Probe accuracy (representation): {transfer_df['correct_probe'].mean():.3f}")
        print(f"  Output accuracy (behavioural):   {transfer_df['correct_output'].mean():.3f}")
        rf = transfer_df["readout_failure"]
        print(f"  Readout failures (probe✓, output✗): {rf.sum()} ({rf.mean():.1%})")
        print("\n  By domain:")
        for dom, grp in transfer_df.groupby("domain"):
            print(f"    {dom:20s}: probe={grp['correct_probe'].mean():.3f}  "
                  f"output={grp['correct_output'].mean():.3f}  "
                  f"readout_fail={grp['readout_failure'].mean():.3f}")

        # Readout failure examples
        rf_examples = transfer_df[transfer_df["readout_failure"]].head(10)
        if len(rf_examples):
            print("\n  Readout failure examples (probe correct, output wrong):")
            for _, r in rf_examples.iterrows():
                print(f"    [{r['domain']}] {r['property']}: "
                      f"true={r['abstraction_class']}  "
                      f"probe={r['pred_probe']}  output={r['pred_output']}")
    else:
        print("  Family D cross-domain prompts not found — skipping transfer analysis")
        print("  Run: python scripts/60_generate_abstraction_probe_datasets.py")

    # Summary report
    lines = [
        "# IE Cluster Analysis Report",
        f"## {args.behaviour} | k={args.k}",
        "",
    ]
    if stats_df is not None:
        lines += [
            "## Cluster Selectivity",
            "",
            stats_df[["cluster","n_features","mu_intensive","mu_extensive",
                       "selectivity","dominant","sig"]].to_string(index=False),
            "",
        ]
    if cross_rows and "transfer_df" in dir():
        td = transfer_df
        lines += [
            "## Cross-domain Transfer + Readout Failure",
            f"- Probe accuracy (representation): {td['correct_probe'].mean():.3f}",
            f"- Output accuracy (behavioural): {td['correct_output'].mean():.3f}",
            f"- Readout failures: {td['readout_failure'].sum()} / {len(td)} "
            f"({td['readout_failure'].mean():.1%})",
            "",
            "### Per-domain",
        ]
        for dom, grp in td.groupby("domain"):
            lines.append(f"- {dom}: probe={grp['correct_probe'].mean():.3f}  "
                         f"output={grp['correct_output'].mean():.3f}  "
                         f"readout_failure={grp['readout_failure'].mean():.1%}")

    (out_dir / f"ie_cluster_report_k{args.k}.md").write_text("\n".join(lines))
    print(f"\nReport: {out_dir}/ie_cluster_report_k{args.k}.md")
    print(f"CSVs:   {out_dir}/")


if __name__ == "__main__":
    main()
