# Multilingual Circuits — Pipeline Audit Report

**Date:** 2026-03-06
**Script audited:** `scripts/a_analyze_multilingual_circuits.py`
**Reference:** `scripts/02_run_baseline.py`, `data/prompts/multilingual_circuits_train.jsonl`

---

## Executive Summary

Four suspected issues were investigated. Two are real bugs with direct impact on IoU and bridge-feature computations (Issues 1 and 2 partial). Two are not bugs (Issues 3 and 4 are data/design artefacts, not code errors). The most severe bug causes IoU and bridge features to measure **concept-group similarity** instead of **language similarity** — a completely wrong comparison.

---

## Issue 1 — EN/FR Assignment Bug (CONFIRMED CRITICAL)

### What the code did

`compute_iou()` (lines 206–214 pre-fix):
```python
en_end = n_en          # = 24
fr_end = n_en + n_fr   # = 48
en_feats = set(idx[:en_end, :].flatten())   # rows 0–23
fr_feats = set(idx[en_end:fr_end, :].flatten())  # rows 24–47
```

`find_bridge_features()` (line 299 pre-fix):
```python
df["language"] = df["prompt_idx"].apply(lambda i: "en" if i < n_en else "fr")
```

### Why it is wrong

The train JSONL is **interleaved by concept**, not blocked by language:

```
idx=0  EN concept 0    idx=3  FR concept 0
idx=6  EN concept 2    idx=9  FR concept 2
idx=12 EN concept 3    idx=15 FR concept 3
...
idx=24 EN concept 5    idx=27 FR concept 5
...
```

Consequence of `prompt_idx < 24 → "en"`:

| Group | True-EN prompts | True-FR prompts | Concepts covered |
|---|---|---|---|
| Buggy "EN" (idx 0–23) | 12 | 12 | {0, 2, 3, 4} |
| Buggy "FR" (idx 24–47) | 12 | 12 | {5, 6, 7, 8} |

**The IoU computation was measuring feature-set similarity between concept-group {0,2,3,4} vs concept-group {5,6,7,8}, not between English prompts vs French prompts.** Claims 1–3 (language-specific features, shared cross-lang features, middle-layer concentration) were all based on this wrong comparison.

### Scientific impact

- **IoU values (0.55–0.73, mean 0.647) are invalid** for Claims 1–3.
- After the fix, the IoU measures true EN vs FR. Since EN and FR prompts for the same concepts activate similar features, the corrected IoU is expected to be **higher** (more feature sharing across languages than across concept groups).
- Bridge features (30/53) were computed by grouping ablation effects into the same wrong concept-half groups. These numbers are not interpretable as cross-lingual bridge features.
- **C3 patching and the baseline gate are NOT affected** — they use `concept_index` metadata and `language` column from the CSV respectively.

### Fix applied

`compute_iou()`: replaced index-based slicing with explicit index lists derived from the JSONL `language` column:
```python
prompts = load_prompts(train_jsonl)
en_indices = sorted(prompts.loc[prompts["language"] == "en", "prompt_idx"].tolist())
fr_indices = sorted(prompts.loc[prompts["language"] == "fr", "prompt_idx"].tolist())
en_feats = set(idx[en_indices, :].flatten())
fr_feats = set(idx[fr_indices, :].flatten())
```

`find_bridge_features()`: replaced lambda with dict join on `prompt_idx`:
```python
idx_to_lang = dict(zip(prompts["prompt_idx"], prompts["language"]))
df["language"] = df["prompt_idx"].map(idx_to_lang)
```

Removed `n_en` / `n_fr` parameters from both functions and from `main()` argparse.

---

## Issue 2 — Baseline Metric Inconsistency (CONFIRMED MINOR / NUMERICALLY IRRELEVANT)

### What was found

- `02_run_baseline.py` line 416: `acc_sign = (df_valid["logprob_diff_normalized"] > 0).mean()`
- `a_analyze_multilingual_circuits.py` line 105 (pre-fix): `df["sign_correct"] = df["logprob_diff"] > 0`

These use different columns (`logprob_diff` vs `logprob_diff_normalized`).

### Why it does not change results

The baseline output confirmed:
```
Mean correct answer length:  1.00 tokens
Mean incorrect answer length: 1.00 tokens
Length mismatch rate: 0.0%
```

With `correct_token_len = incorrect_token_len = 1`:
```
logprob_diff_normalized = (correct_log_prob / 1) - (incorrect_log_prob / 1) = logprob_diff
```

The two expressions are **numerically identical** for this dataset. The sign of `logprob_diff_normalized > 0` is always the same as `logprob_diff > 0`.

### Fix applied (for correctness and consistency)

Changed to `logprob_diff_normalized > 0` with a comment explaining the equivalence and the reason for the choice (consistency with script 02 and robustness to future multi-token answers).

---

## Issue 3 — `success` Column Inconsistency (NOT A BUG)

### What was suspected

Rows where `logprob_diff_normalized > 0` but `success = False`.

### What is actually happening

`02_run_baseline.py` line 136:
```python
success = logprob_diff_normalized > min_score_diff
```

`min_score_diff` is a **positive threshold** (configured per-behaviour, default around 0.5). So a prompt with `0 < logprob_diff_normalized < min_score_diff` will have:
- `recomputed_sign_success = True` (logprob_diff_normalized > 0)
- `success = False` (logprob_diff_normalized ≤ min_score_diff)

This is **correct and intentional**: `success` is a stricter criterion (minimum signal strength), while sign accuracy only checks direction.

`a_analyze_multilingual_circuits.py` already ignores the `success` column for its gate check and recomputes from `logprob_diff_normalized > 0`. This is the right design.

### Action

Added `generate_fr_failure_audit()` which explicitly documents this distinction in `fr_failure_audit.csv` (columns: `recomputed_sign_success`, `original_success`, `mismatch_flag`), with a console note explaining the mismatch is expected.

---

## Issue 4 — FR Weakness: Prompt Design vs Model Weakness (DATA QUALITY, NOT CODE)

### Diagnosis

From `multilingual_circuits_train.jsonl`, the 8 FR failures concentrate on two concepts:

**Concept 2: `"vite" → "lent"` (word-class mismatch)**
- `"vite"` is a French **adverb** (= "quickly").
- `"lent"` is a French **adjective** (= "slow").
- The natural French adverb antonym of "vite" is `"lentement"` (= "slowly"). The model correctly doubts `"lent"` as a completion.
- The EN equivalent `"fast" → "slow"` is adjective → adjective, consistent.
- Fix: replace `"vite"` with `"rapide"` (French adjective meaning "fast") for a clean adj→adj pair.

**Concept 7: `"propre" → "sale"` (polysemy)**
- `"propre"` means both "clean" (antonym "sale") and "own/proper" (as in *ma propre maison*).
- The model is uncertain which sense is intended; both forward templates fail.
- Fix: replace `"propre"` with an unambiguous French adjective pair, or accept 66.7% FR accuracy.

**Template effect (concept 4: `"vide" → "plein"`)**
- The forward template (t0) passes; reversed templates (t2, t3) fail.
- This is a minor template-order sensitivity, not a fundamental concept problem.
- The model knows "plein" is the antonym of "vide" but is less confident in reversed-order prompts.

### EN = 100%, 5/8 FR concepts perfect

The model handles French correctly when the word pairs are linguistically clean. The failure is in the **dataset vocabulary design**, not in the model's French capability.

### Template coverage note

Prompts use template indices from `{0, 1, 2, 3}` but **not all four per concept** — each concept has exactly 3 train templates (the 4th is held out in test). The exact held-out index varies by concept (different `rng.shuffle` outcome per concept after Fix 1). Per-template analysis must account for this uneven coverage.

---

## What Changed

| File | Change | Reason |
|---|---|---|
| `scripts/a_analyze_multilingual_circuits.py` | `compute_iou()`: replaced index-based slicing with JSONL-derived language indices | Bug fix (Issue 1) |
| `scripts/a_analyze_multilingual_circuits.py` | `find_bridge_features()`: replaced `prompt_idx < n_en` with dict join from JSONL | Bug fix (Issue 1) |
| `scripts/a_analyze_multilingual_circuits.py` | `check_baseline_gate()`: `logprob_diff` → `logprob_diff_normalized` | Consistency (Issue 2) |
| `scripts/a_analyze_multilingual_circuits.py` | Added `generate_fr_failure_audit()` | Diagnostic output (Issue 3) |
| `scripts/a_analyze_multilingual_circuits.py` | Removed `--n_en`/`--n_fr` args from `main()` | No longer needed after fix |

---

## What Was NOT Changed

- `scripts/02_run_baseline.py`: no bugs found; gate logic is correct.
- `scripts/07_run_interventions.py`: C3 patching uses `concept_index` metadata for pairing; not affected.
- Prompt files: no changes.
- Config files: no changes.

---

## What Remains Uncertain

1. **Corrected IoU values**: The `.npy` feature files are on CSD3; the true EN vs FR IoU has not yet been recomputed with the fixed code. Expected direction: higher than 0.647 (more feature sharing when comparing EN vs FR across all concepts vs. comparing two concept groups).

2. **Corrected bridge features**: Similarly, the corrected bridge feature count and identity are unknown until the analysis is re-run on CSD3.

3. **Whether Claim 3 (middle-layer concentration) is reproducible**: With the corrected IoU, if EN and FR are highly similar across all layers (because both cover all concepts), the middle vs early/late contrast may be even weaker than before.

4. **FR vocabulary fix impact**: Replacing `"vite"` with `"rapide"` and possibly `"propre"` with an unambiguous word would require regenerating prompts and re-running the full pipeline. This is a separate decision from the code bugs above.

---

## Confirmed Bugs Summary

| # | Location | Severity | Impact |
|---|---|---|---|
| 1a | `compute_iou()` lines 206–214 | **CRITICAL** | IoU measures concept-group not language similarity |
| 1b | `find_bridge_features()` line 299 | **CRITICAL** | Bridge features grouped by concept-half not language |
| 2 | `check_baseline_gate()` line 105 | Minor (numerically 0 impact) | Inconsistent column name; fixed for robustness |

**Non-bugs confirmed:** Issue 3 (success column), Issue 4 (FR weakness is vocabulary design).
