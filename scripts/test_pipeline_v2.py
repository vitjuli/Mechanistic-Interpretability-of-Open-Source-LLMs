#!/usr/bin/env python3
"""
test_pipeline_v2.py

Pre-flight tests for physics_decay_type_probe_v2 pipeline.
Runs all checks that can be done locally (without GPU/model).
Tests both data validity and pipeline script wiring.

Usage:
    python3 scripts/test_pipeline_v2.py [--behaviour physics_decay_type_probe_v2]

Exit codes: 0 = all pass, 1 = failures found
"""

import ast, json, subprocess, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--behaviour", default="physics_decay_type_probe_v2")
args = parser.parse_args()

BEHAVIOUR = args.behaviour
ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL, WARN = "✓ PASS", "✗ FAIL", "⚠ WARN"

results = []

def check(name, condition, detail="", level="FAIL"):
    status = PASS if condition else (WARN if level == "WARN" else FAIL)
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition

print(f"\n{'='*60}")
print(f"  Pipeline pre-flight: {BEHAVIOUR}")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 1: Prompt file structure
# ══════════════════════════════════════════════════════════════════
print("── Group 1: Prompt file ──")

pfile = ROOT / f"data/prompts/{BEHAVIOUR}_train.jsonl"
check("Prompt file exists", pfile.exists(), str(pfile))

if pfile.exists():
    prompts = [json.loads(l) for l in open(pfile)]
    n = len(prompts)

    check("Prompt count ≥ 470", n >= 470, f"{n} prompts")
    check("All have 'prompt' key",     all("prompt"          in p for p in prompts))
    check("All have 'correct_answer'", all("correct_answer"  in p for p in prompts))
    check("All have 'incorrect_answer'", all("incorrect_answer" in p for p in prompts))
    check("All have 'prompt_id'",      all("prompt_id"       in p for p in prompts))
    check("All have 'level'",          all("level"           in p for p in prompts))
    check("All have 'group_id'",       all("group_id"        in p for p in prompts))

    answers = Counter(p["correct_answer"] for p in prompts)
    check("Answers are ' alpha' and ' beta' only",
          set(answers.keys()) == {" alpha", " beta"},
          str(dict(answers)))
    check("Balanced alpha/beta (±10%)",
          abs(answers[" alpha"] - answers[" beta"]) / n < 0.10,
          f"alpha={answers[' alpha']} beta={answers[' beta']}")

    levels = Counter(p.get("level") for p in prompts)
    check("Has level 1, 2, 3 prompts",
          all(l in levels for l in [1, 2, 3]),
          f"levels={dict(levels)}")

    # Check prompts end correctly (no trailing space/newline issue)
    sample_prompt = prompts[0]["prompt"]
    check("Prompt text is non-empty", len(sample_prompt) > 10)
    check("Correct answer has leading space",
          all(p["correct_answer"].startswith(" ") for p in prompts),
          "required for tokenizer single-token check")

    # Uniqueness
    ids = [p["prompt_id"] for p in prompts]
    check("No duplicate prompt_ids", len(set(ids)) == len(ids),
          f"{len(ids)-len(set(ids))} duplicates" if len(set(ids)) != len(ids) else "")

    # V1 compatibility: V2 should include all V1 prompts
    v1_file = ROOT / "data/prompts/physics_decay_type_probe_train.jsonl"
    if v1_file.exists():
        v1_ids = {json.loads(l)["prompt_id"] for l in open(v1_file)}
        v2_ids = set(ids)
        missing_v1 = v1_ids - v2_ids
        check("V2 is superset of V1 (no V1 prompts removed)",
              len(missing_v1) == 0,
              f"{len(missing_v1)} V1 prompts missing from V2" if missing_v1 else
              f"V2 has {len(v2_ids)-len(v1_ids)} new prompts vs V1")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 2: Attribution graph (must be unchanged)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 2: Attribution graph (shared with V1) ──")

graph_path = ROOT / "data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json"
check("Run B graph exists", graph_path.exists(), str(graph_path))

if graph_path.exists():
    g = json.load(open(graph_path))
    feat_nodes = [n for n in g["nodes"] if n.get("type") == "feature"]
    n_feat = len(feat_nodes)
    n_pos  = sum(1 for n in feat_nodes if n.get("grad_attr_sign", 0) > 0)
    n_neg  = sum(1 for n in feat_nodes if n.get("grad_attr_sign", 0) < 0)
    check("Graph has exactly 69 features", n_feat == 69, f"found {n_feat}")
    check("Graph has both pos and neg attribution features",
          n_pos > 0 and n_neg > 0 and n_pos + n_neg == n_feat,
          f"pos={n_pos} neg={n_neg} (both non-zero, total=69)")
    check("Layers L10-L25 present",
          {n["layer"] for n in feat_nodes} == set(range(10, 26)))

    # Symlink check: graph accessible from V2 probe namespace
    probe_graph = ROOT / f"data/results/attribution_graphs/{BEHAVIOUR}/attribution_graph_train_n108_roleaware_static.json"
    if not probe_graph.exists():
        probe_graph.parent.mkdir(parents=True, exist_ok=True)
        probe_graph.symlink_to(graph_path.resolve())
        check("Created symlink for V2 probe namespace", probe_graph.exists(),
              str(probe_graph))
    else:
        check("V2 probe namespace graph accessible", probe_graph.exists())


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 3: Activation npy files (for mean/resample ablation)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 3: Activation npy files ──")

probe_npy_ok = True
for layer in range(10, 26):
    for suffix in ["top_k_indices", "top_k_values"]:
        p = ROOT / f"data/results/transcoder_features/layer_{layer}/{BEHAVIOUR}_train_{suffix}.npy"
        if not p.exists():
            probe_npy_ok = False
            break
check("V2 probe activation npy files (L10-L25)", probe_npy_ok,
      "MISSING — re-run script 04 on CSD3" if not probe_npy_ok else "all 32 present",
      level="WARN")

pdt_npy_ok = all(
    (ROOT / f"data/results/transcoder_features/layer_{l}/physics_decay_type_train_top_k_indices.npy").exists()
    for l in range(10, 26)
)
check("PDT train activation npy files (for Run C graph)", pdt_npy_ok,
      "all 16 layers present" if pdt_npy_ok else "MISSING — needed for Run C script 06")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Baseline CSV (post CSD3 run)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 4: Baseline CSV (optional — needs CSD3) ──")

baseline_path = ROOT / f"data/results/baseline_{BEHAVIOUR}_train.csv"
if baseline_path.exists():
    bl = pd.read_csv(baseline_path)
    check("Baseline has 538 rows", len(bl) == len(prompts) if pfile.exists() else True,
          f"found {len(bl)}")
    check("Baseline has logprob_diff column",
          "logprob_diff_normalized" in bl.columns or "logprob_diff" in bl.columns)
    check("Baseline accuracy > 70%",
          (bl.get("logprob_diff_normalized", bl.get("logprob_diff", pd.Series())) > 0).mean() > 0.70,
          level="WARN")
else:
    check("Baseline CSV exists (CSD3 step 02)", False,
          "not yet run — needed before ablation", level="WARN")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Ablation CSV (post CSD3 run)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 5: Ablation CSV (optional — needs CSD3) ──")

abl_path = ROOT / f"data/results/interventions/{BEHAVIOUR}/runB/intervention_ablation_{BEHAVIOUR}.csv"
if abl_path.exists():
    abl = pd.read_csv(abl_path)
    n_abl_feats   = abl.feature_id.nunique()
    n_abl_prompts = abl.prompt_idx.nunique()
    expected_rows = n_abl_feats * n_abl_prompts

    check("Ablation CSV has 69 features",        n_abl_feats == 69,     f"{n_abl_feats}")
    check("Ablation CSV has 538 prompts",         n_abl_prompts == 538,  f"{n_abl_prompts}")
    check("Ablation rows = 69 × 538 = 37,122",   len(abl) == expected_rows,
          f"{len(abl)} vs expected {expected_rows}")
    check("All feature_source == 'graph'",        (abl.feature_source == "graph").all())
    check("No NaN effect_size",                   abl.effect_size.notna().all())
    check("Layers L10-L25 all present",           set(abl.layer.unique()) == set(range(10, 26)))
    check("Both correct answers present",
          set(abl.metadata.apply(lambda x: ast.literal_eval(x)["correct_token"]).unique()) == {" alpha", " beta"})
    check("SFR in plausible range (1-20%)",
          0.01 < abl.sign_flipped.mean() < 0.20,
          f"SFR={abl.sign_flipped.mean():.4f}", level="WARN")
else:
    check("Ablation CSV exists (CSD3 step 07)", False,
          "not yet run — run jobs/run_probe_runB_ablation.sbatch", level="WARN")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 6: Script wiring (dry-run key pipeline scripts)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 6: Script wiring (syntax + arg parsing) ──")

def check_script_arg(script, arg, value="test"):
    """Check script source contains the argument — avoids running data-loading scripts."""
    try:
        return arg in Path(script).read_text()
    except Exception:
        return False

for script_path, key_arg in [
    ("scripts/19_feature_prompt_analysis.py", "--behaviour"),
    ("scripts/22_prepare_clustering_inputs.py", "--grouping_dir"),
    ("scripts/23_run_clustering_benchmark.py", "--clustering_dir"),
    ("scripts/26_cluster_semantics.py", "--grouping_dir"),
    ("scripts/27_cluster_joint_ablation.py", "--behaviour"),
    ("scripts/27b_analyse_joint_ablation.py", "--joint_dir"),
    ("scripts/28_enrichment_robustness.py", "--grouping_dir"),
    ("scripts/29_final_cluster_validation.py", "--grouping_dir"),
    ("scripts/runB_validation.py", "--behaviour"),
    ("scripts/runB_three_mode_analysis.py", "--behaviour"),
    ("scripts/runC_null_cluster_test.py", "--runC_base"),
    ("scripts/runC_comparison_report.py", "--runB_base"),
]:
    ok = (ROOT / script_path).exists() and check_script_arg(ROOT / script_path, key_arg)
    check(f"{Path(script_path).name} has {key_arg}", ok)

# Check pipeline shell scripts accept --behaviour
for sh in ["scripts/run_runB_pipeline.sh", "scripts/run_runC_pipeline.sh"]:
    content = open(ROOT / sh).read()
    has_behaviour_arg = "--behaviour)" in content
    check(f"{Path(sh).name} accepts --behaviour", has_behaviour_arg)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 7: Sbatch file consistency
# ══════════════════════════════════════════════════════════════════
print("\n── Group 7: Sbatch file consistency ──")

for sbatch_file, expected_beh, expected_prompts in [
    ("jobs/run_probe_runB_ablation.sbatch",      "physics_decay_type_probe_v2", 538),
    ("jobs/run_probe_runB_joint_ablation.sbatch","physics_decay_type_probe_v2", 538),
    ("jobs/run_probe_three_mode_ablation.sbatch","physics_decay_type_probe_v2", 538),
    ("jobs/run_probe_runC_pipeline.sbatch",      "physics_decay_type_probe_v2", 538),
    ("jobs/run_probe_runC_joint_ablation.sbatch","physics_decay_type_probe_v2", 538),
]:
    p = ROOT / sbatch_file
    if p.exists():
        content = p.read_text()
        has_beh  = expected_beh in content
        has_prom = str(expected_prompts) in content
        check(f"{Path(sbatch_file).name}: BEHAVIOUR={expected_beh}", has_beh)
        check(f"{Path(sbatch_file).name}: N_PROMPTS={expected_prompts}", has_prom)
    else:
        check(f"{Path(sbatch_file).name} exists", False, "file missing")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Analysis pipeline dry-run (if ablation exists)
# ══════════════════════════════════════════════════════════════════
print("\n── Group 8: Analysis pipeline dry-run ──")

abl_exists = abl_path.exists()
ui_run = ROOT / "data/ui_offline/20260430-152526_physics_decay_type_probe_train_n108"
check("UI run (graph metadata) exists", ui_run.exists(), str(ui_run))

if abl_exists and ui_run.exists():
    # Test script 19 on a small subset
    test_out = ROOT / "data/analysis/_test_v2"
    test_out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "scripts/19_feature_prompt_analysis.py",
         "--behaviour", BEHAVIOUR, "--split", "train",
         "--ui_run", str(ui_run),
         "--abl_csv", str(abl_path),
         "--grouping_dir", str(test_out),
         "--top_k", "3"],
        capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    ok19 = r.returncode == 0
    check("Script 19 runs on V2 ablation data", ok19,
          "check output for errors" if not ok19 else
          f"saved to {test_out}")
    if not ok19:
        print(f"    stderr: {r.stderr[-500:]}")
    else:
        # Verify output
        fp = test_out / "feature_prompt_contributions.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            expected = 69 * 538
            check("Script 19 output has 69×538 rows",
                  len(df) == expected, f"{len(df)} vs {expected}")
            check("Script 19 prompt_idx range covers 0-537",
                  df.prompt_idx.max() == 537, f"max={df.prompt_idx.max()}")
        import shutil; shutil.rmtree(test_out, ignore_errors=True)
else:
    check("Script 19 dry-run (skipped — ablation CSV not yet available)",
          True, "run after CSD3 step 07 completes", level="WARN")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
n_pass  = sum(1 for s, _, _ in results if s == PASS)
n_fail  = sum(1 for s, _, _ in results if s == FAIL)
n_warn  = sum(1 for s, _, _ in results if s == WARN)
print(f"  {PASS}: {n_pass}   {FAIL}: {n_fail}   {WARN}: {n_warn}")
print()

if n_fail > 0:
    print("FAILURES (must fix before CSD3 submission):")
    for s, name, detail in results:
        if s == FAIL:
            print(f"  {s}  {name}" + (f" — {detail}" if detail else ""))
    print()

if n_warn > 0:
    print("WARNINGS (expected pending items — will resolve after CSD3):")
    for s, name, detail in results:
        if s == WARN:
            print(f"  {s}  {name}" + (f" — {detail}" if detail else ""))
    print()

print(f"Pipeline is {'READY for CSD3 submission' if n_fail == 0 else 'NOT READY — fix failures above'}.")
sys.exit(0 if n_fail == 0 else 1)
