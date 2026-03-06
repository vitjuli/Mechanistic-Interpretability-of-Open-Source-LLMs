# FR Vocabulary Fix — Controlled Experiment

**Experiment date:** 2026-03-06
**Scope:** Concept 2 only (`"vite" → "lent"` replaced by `"rapide" → "lent"`)
**Status:** Code and prompts updated locally. CSD3 baseline rerun PENDING.

---

## 1. What Was Changed

**File:** `scripts/01_generate_prompts.py`, line 586

```diff
- (2, "fast",   "slow",    "vite",     "lent",        True),
+ (2, "fast",   "slow",    "rapide",   "lent",        True),  # was "vite" (adverb)
```

**Prompt-level effect (concept 2, FR, train set — idx 9, 10, 11):**

| idx | template | Before prompt | After prompt |
|---|---|---|---|
| 9  | t0 | Le contraire de "vite" est | Le contraire de "rapide" est |
| 10 | t1 | L'antonyme de "vite" est | L'antonyme de "rapide" est |
| 11 | t2 | "vite" est le contraire de | "rapide" est le contraire de |

Correct answer: `' lent'` (unchanged).
Incorrect answer: `' vite'` → `' rapide'` (the new word to reject).

**Test set concept 2 FR (idx 3):**

| Before | After |
|---|---|
| `"vite" est l'antonyme de` | `"rapide" est l'antonyme de` |

All other 45 prompts: **unchanged**.
Template matching (EN and FR hold out same `template_idx` per concept): **preserved** — verified.

---

## 2. Linguistic Justification

| | EN concept 2 | Old FR concept 2 | New FR concept 2 |
|---|---|---|---|
| word | fast (adj) | vite (adverb) | rapide (adj) |
| antonym | slow (adj) | lent (adj) | lent (adj) |
| word-class pair | adj → adj ✓ | adverb → adj ✗ | adj → adj ✓ |

**Why `"vite"` is wrong:** In French, `"vite"` is an adverb meaning "quickly/rapidly". Its natural adverbial antonym is `"lentement"` (slowly), not `"lent"` (slow, adjective). Completing `'"vite" est le contraire de'` with `" lent"` requires a cross-category leap that is linguistically awkward. The model is not wrong to be uncertain here.

**Why `"rapide"` is correct:** `"rapide"` is the French adjective for "fast", directly parallel to EN `"fast"`. The pair `rapide / lent` is a canonical French adjective antonym pair, equivalent to `fast / slow`. This is what the Anthropic dataset likely used.

No threshold was changed. No other concepts were touched.

---

## 3. Before/After Baseline Metrics

### BEFORE (confirmed — SLURM job 24321571, analysis run 2026-03-06)

| Metric | Value | Status |
|---|---|---|
| EN sign accuracy | 1.000 (24/24) | PASS ≥ 0.90 |
| FR sign accuracy | 0.667 (16/24) | PASS ≥ 0.65 |
| Overall sign accuracy | 0.833 (40/48) | — |
| Mean norm logprob diff | 3.383 | PASS ≥ 1.00 |
| FR mean norm diff | 0.621 | — |

### AFTER (PENDING — requires CSD3 baseline rerun)

Run on CSD3:
```bash
git pull
python -u scripts/02_run_baseline.py --behaviour multilingual_circuits --split train
python scripts/a_analyze_multilingual_circuits.py
```

Expected direction (hypothesis, not yet confirmed):
- EN sign accuracy: 1.000 (no change — EN prompts unchanged)
- FR sign accuracy: ~0.792 (19/24) if all 3 concept 2 failures were due to word-class mismatch
- Overall: ~0.896 (43/48)
- FR mean norm diff: increase expected (model more confident on `rapide/lent`)

These are **hypotheses**. Report confirmed values here after rerun.

| Metric | Value | Status |
|---|---|---|
| EN sign accuracy | PENDING | — |
| FR sign accuracy | PENDING | — |
| Overall sign accuracy | PENDING | — |
| Mean norm logprob diff | PENDING | — |

---

## 4. Per-Concept FR Results Before vs After

### BEFORE (source of truth: analysis script output — partially inferred)

| Concept | FR word | FR antonym | Pass/Fail | Notes |
|---|---|---|---|---|
| 0 | petit | grand | 3/3 PASS | confirmed (no failures in output) |
| 2 | vite | lent | **0/3 FAIL** | confirmed: t2 failure shown; inferred t0,t1 also fail (systematic word-class) |
| 3 | nouveau | vieux | 3/3 PASS | confirmed |
| 4 | vide | plein | **1/3 PASS** | confirmed: t2 (-3.76), t3 (-3.32) fail; t0 inferred pass |
| 5 | haut | bas | 3/3 PASS | confirmed |
| 6 | long | court | 3/3 PASS | confirmed |
| 7 | propre | sale | **0/3 FAIL** | confirmed: t0 (-1.13), t1 (-2.81) fail; t3 inferred fail |
| 8 | facile | difficile | 3/3 PASS | confirmed |

Note: "inferred" means consistent with the 8-failure total and the known worst-5 list, but not directly read from per-row CSV (which is on CSD3). The exact per-row breakdown will be confirmed by `fr_failure_audit.csv`.

### AFTER (PENDING — fill in after CSD3 rerun)

Expected: concept 2 changes from 0/3 to 3/3 PASS. All other concepts unchanged.

| Concept | FR word | Pass/Fail before | Pass/Fail after |
|---|---|---|---|
| 2 | rapide (was vite) | 0/3 | PENDING |
| 4 | vide | 1/3 | PENDING (should be unchanged) |
| 7 | propre | 0/3 | PENDING (should be unchanged) |
| others | — | 3/3 | PENDING (should be unchanged) |

---

## 5. Per-Template FR Results Before vs After

Templates used in the multilingual_circuits behaviour:
- t0: `Le contraire de "{word}" est` (forward)
- t1: `L'antonyme de "{word}" est` (forward)
- t2: `"{word}" est le contraire de` (reversed)
- t3: `"{word}" est l'antonyme de` (reversed)

Note: not every template appears for every concept — each concept has exactly 3 of the 4 templates in train (one held out for test).

### BEFORE (inferred from known failures — requires CSV confirmation)

| Template | FR train prompts | Estimated pass | Notes |
|---|---|---|---|
| t0 | 8 | ~6/8 | propre-t0 fails, vide-t0 passes, vite-t0 inferred fail |
| t1 | 5 | ~3/5 | propre-t1 fails, vite-t1 inferred fail |
| t2 | 8 | ~5/8 | vide-t2 fails, vite-t2 fails |
| t3 | 8 | ~5/8 | vide-t3 fails, propre-t3 inferred fails |

Template coverage is uneven because each concept holds out a different template. Raw template accuracy is confounded with concept difficulty.

### AFTER (PENDING)

If concept 2 is fully fixed (3 new passes: t0, t1, t2), the per-template counts improve by 1 each for t0, t1, t2.

---

## 6. Evidence Assessment: Dataset Design vs Model Weakness for Concept 2

### Evidence that concept 2 was a dataset design issue

1. **EN passes 100%:** The model correctly completes `"fast" is the opposite of` → `slow`. The semantic concept is understood.
2. **Cross-category mismatch is linguistically principled:** A French speaker would not naturally complete `"vite" est le contraire de` → `lent` (mixing adverb and adjective). The natural completion is `lentement` (adverb). The model's hesitation is not a capability failure.
3. **All other adj→adj FR pairs pass:** Concepts 0, 3, 5, 6, 8 (all clean adjective pairs) pass with 100% FR accuracy.
4. **No other language shows this pattern:** EN concept 2 ("fast"→"slow") is adjective→adjective and passes 100%.

### What the rerun must show to confirm

- All 3 FR concept 2 prompts pass with `rapide → lent` (sign_correct = True, norm_diff > 0).
- EN is unchanged at 100%.
- The norm_diff for concept 2 FR rows is clearly positive (not marginal).

If any of the 3 new `rapide/lent` prompts still fail, that would require further investigation of the `rapide/lent` pair specifically.

---

## 7. Analysis Outputs After Fix (PENDING — requires CSD3 rerun)

After CSD3 baseline rerun, also regenerate:
```bash
python scripts/a_analyze_multilingual_circuits.py
```

Expected outputs to update:
- `data/analysis/multilingual_circuits/REPORT.md`
- `data/analysis/multilingual_circuits/fr_failure_audit.csv`
- `data/analysis/multilingual_circuits/gate_check.txt`

Note: The corrected `a_analyze_multilingual_circuits.py` (IoU language-assignment bug fixed) will also produce corrected IoU and bridge feature results for the first time.

---

## 8. Concept 7 Conservative Evaluation

**Current state:** `"propre" → "sale"` (EN: `"clean" → "dirty"`)

`"propre"` is polysemous in French:
- Meaning A: "clean" (antonym: "sale") — e.g., *les mains propres*
- Meaning B: "own/proper" (adjectival pronoun sense) — e.g., *ma propre maison* ("my own house")

Both forward templates (t0, t1) failed, which rules out template-order as the cause. The model does not confidently associate `"propre"` with `"sale"`.

### How to evaluate concept 7 after concept 2 fix

Only evaluate concept 7 replacement if, after the concept 2 fix, FR accuracy is still failing the gate (< 0.65). From prior analysis, 5 remaining failures are spread across concepts 4 and 7. If concept 7 is still 0/3, replacement is justified.

### Candidate replacements for FR concept 7 (propre/sale)

| Candidate | FR word | FR antonym | Word class | Notes |
|---|---|---|---|---|
| A | **net** | **sale** | adj → adj | "net" = clean/tidy/clear in French. Less polysemous than "propre" in antonym context. "sale" stays. |
| B | **pur** | **impur** | adj → adj | "pur"=pure, "impur"=impure. Preserves clean semantics. Risk: "impur" may be 2 tokens. Must audit. |
| C | **propre** | **sale** | keep current | Adding more prompts won't help polysemy. Not recommended. |
| D | Replace concept entirely | e.g., "doux"/"dur" | adj → adj | Changes concept from clean/dirty to soft/hard. Loses semantic parallelism with EN. Not recommended. |

**Recommendation: Option A (`net → sale`).**

Rationale:
- `"net"` is a clear French adjective meaning "clean, tidy, precise" (e.g., *un résultat net* = a clean result).
- In the context of `Le contraire de "net" est` the natural completion is `"sale"`, with little ambiguity.
- `"sale"` is already in the dataset (as the antonym) and is unambiguous (dirty/nasty).
- Both are common single-token words in modern tokenizers.
- The semantic field (cleanliness/dirtiness) is preserved.

Risk: `"net"` has a secondary meaning in French as the "internet" (le net) and as "net" in finance (valeur nette). In the explicit antonym-completion template format, this context is unlikely to confuse the model, but it cannot be ruled out without testing.

**Option B (`pur/impur`)** should be checked for tokenization before use — `"impur"` may tokenize as 2 tokens.

**Do NOT replace concept 7 until concept 2 fix results are confirmed** and concept 7 is verified to still be the dominant remaining failure.

---

## Files Changed

| File | Change |
|---|---|
| `scripts/01_generate_prompts.py` | Line 586: `"vite"` → `"rapide"` (fr_word for concept 2) |
| `data/prompts/multilingual_circuits_train.jsonl` | Regenerated (48 lines, concept 2 FR rows updated) |
| `data/prompts/multilingual_circuits_test.jsonl` | Regenerated (16 lines, concept 2 FR test row updated) |

## Commands to Run on CSD3

```bash
# 1. Pull changes
git pull

# 2. Re-run baseline only (step 02)
python -u scripts/02_run_baseline.py --behaviour multilingual_circuits --split train

# 3. Re-run analysis (with IoU bug fixed + FR failure audit)
python scripts/a_analyze_multilingual_circuits.py

# 4. Check outputs
cat data/analysis/multilingual_circuits/gate_check.txt
cat data/analysis/multilingual_circuits/REPORT.md
cat data/analysis/multilingual_circuits/fr_failure_audit.csv
```

## Confirmed vs Hypothesis

| Claim | Status |
|---|---|
| `"vite"` is an adverb, not an adjective | **Confirmed** (linguistic fact) |
| `"rapide"` and `"lent"` are the correct adj→adj parallel | **Confirmed** (linguistic fact) |
| Concept 2 failures were all 3 train templates | **Inferred** from 8-total failures + worst-5 list; confirmed by `fr_failure_audit.csv` after rerun |
| Fixing concept 2 will bring FR to ~79% | **Hypothesis** — not yet confirmed |
| Concept 7 is still 0/3 after concept 2 fix | **Hypothesis** — confirmed by rerun |
