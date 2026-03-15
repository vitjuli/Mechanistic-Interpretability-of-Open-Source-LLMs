# Decision Memo: multilingual_circuits After rapide Fix

**Date:** 2026-03-06
**Source data:** `data/results/baseline_multilingual_circuits_train.csv` (post-rapide fix, SLURM 24321571 re-run)
**Status:** FINAL — based on exact CSV values, no inference

---

## 1. Executive Summary

FR sign accuracy is **0.667 (16/24)** — unchanged from before the rapide fix. The gate passes (threshold ≥ 0.65). 8 failures remain, concentrated in two concepts with distinct root causes: concept 4 (vide/plein) is a reversed-template sensitivity issue; concept 7 (propre/sale) is polysemy. Two additional marginal failures (concepts 5 and 6, reversed template t3) likely reflect the same template-order effect. One residual concept 2 failure (rapide t2, norm −0.096) is a near-miss at the boundary.

**Recommendation: KEEP — the behaviour is scientifically valid as-is.**

---

## 2. Exact FR Failure Breakdown

### 2a. Per-prompt failures (sign_correct = False), sorted by norm_diff

| concept | word | antonym | template | norm_diff | sign | threshold |
|---|---|---|---|---|---|---|
| 4 | vide | plein | t3 (reversed) | **−3.763** | FAIL | FAIL |
| 4 | vide | plein | t2 (reversed) | **−3.324** | FAIL | FAIL |
| 7 | propre | sale | t1 (forward) | **−2.810** | FAIL | FAIL |
| 7 | propre | sale | t0 (forward) | **−1.126** | FAIL | FAIL |
| 4 | vide | plein | t0 (forward) | **−1.000** | FAIL | FAIL |
| 6 | long | court | t3 (reversed) | **−0.274** | FAIL | pass |
| 5 | haut | bas | t3 (reversed) | **−0.135** | FAIL | pass |
| 2 | rapide | lent | t2 (reversed) | **−0.096** | FAIL | pass |

### 2b. Per-concept FR results

| concept | word | antonym | train pass | norm_diffs |
|---|---|---|---|---|
| 0 | petit | grand | **3/3** | 2.07, 1.31, 2.39 |
| 2 | rapide | lent | **2/3** | 1.99, 3.31, −0.10 (t2 fail) |
| 3 | nouveau | vieux | **3/3** | 1.31, 3.24, 3.36 |
| 4 | vide | plein | **0/3** | −1.00, −3.32, −3.76 |
| 5 | haut | bas | **2/3** | 5.32, 1.01, −0.14 (t3 fail) |
| 6 | long | court | **2/3** | 0.45, 1.60, −0.27 (t3 fail) |
| 7 | propre | sale | **1/3** | −1.13, −2.81, 0.07 |
| 8 | facile | difficile | **3/3** | 3.19, 1.80, 1.15 |

### 2c. Per-template FR results

| template | direction | FR pass |
|---|---|---|
| t0 `Le contraire de "{word}" est` | forward | 6/8 |
| t1 `L'antonyme de "{word}" est` | forward | 2/3 |
| t2 `"{word}" est le contraire de` | reversed | 4/6 |
| t3 `"{word}" est l'antonyme de` | reversed | 4/7 |

Coverage is uneven because each concept holds out a different template. The 3 t3-reversed failures (vide, haut, long) and 1 t2-reversed failure (rapide) suggest **reversed-template prompts are harder for FR**. Among t3 slots, all 3 failures are reversed-order.

---

## 3. Root Cause Analysis by Concept

### Concept 4: vide → plein (0/3) — TEMPLATE-ORDER EFFECT

All 3 train templates fail, including the forward template (t0, norm −1.00). The model gives higher logprob to `"vide"` than to `"plein"` in all prompt forms. This is **not a word-class issue** — `vide` and `plein` are both adjectives. The model appears to weakly associate `vide`/`plein` as an antonym pair in French, or the reversed templates override this.

Evidence for template-order as primary driver:
- t2 (−3.32) and t3 (−3.76) are the worst two prompts in the entire dataset.
- t0 (−1.00) fails more narrowly — close to the sign boundary.
- EN concept 4 (empty/full) passes 3/3 with comfortable margins.
- This is the only concept where even the forward template fails.

**Assessment: likely model weakness for this specific FR word pair, not dataset design error.** `vide/plein` is a legitimate adjective antonym pair (adj→adj, clean word class). The model simply does not strongly associate them. Changing the vocabulary would be the only fix, but this risks gaming the metric.

### Concept 7: propre → sale (1/3) — POLYSEMY

Both forward templates fail (t0: −1.13, t1: −2.81). The reversed template barely passes (t3: +0.07 — near-miss). The model does not confidently associate `propre` with `sale` in antonym context, consistent with `propre`'s dual meaning (clean / own).

EN concept 7 (clean/dirty) passes 3/3 — the model understands the concept; the failure is FR-specific and attributable to polysemy.

**Assessment: dataset vocabulary design weakness.** Replacing `propre` with an unambiguous French adjective (e.g., `net → sale`) would likely fix this concept. However, since the gate already passes without this fix, no change is required for the baseline.

### Marginal failures (concepts 2, 5, 6 — one reversed template each)

- Concept 2 (rapide t2: −0.096): near-miss after the rapide fix; the model almost passes
- Concept 5 (haut t3: −0.135): very close to boundary
- Concept 6 (long t3: −0.274): moderate failure; `long` is the same word in FR and EN, which may suppress activation

All three are reversed-template (t2 or t3) marginal failures. None indicate fundamental model incapacity — the same concepts pass on other templates.

---

## 4. Stale Artifact Assessment

### Downstream data computed from the old pipeline run (before rapide fix)

| Artifact | Status | Impact |
|---|---|---|
| Baseline CSV (`baseline_multilingual_circuits_train.csv`) | **CURRENT** — re-run with rapide prompts | Ground truth for all metrics |
| Graph (`.pt` files, `multilingual_circuits_train_n48_*`) | **STALE** — built from vite prompts | IoU, bridge features, C3 all derived from old graph |
| Per-feature activation (`top_feats_*.npy`) | **STALE** | IoU values invalid for Claims 1–3 |
| IoU per-layer values | **STALE** (double stale: vite prompts + language-assignment bug) | Both the raw numbers and EN/FR grouping were wrong |
| Bridge features (30/53) | **STALE** (double stale) | Count and identity unreliable |
| C3 patching results | **STALE** — patching CSVs generated from vite prompts; C3 pairs used vite FR prompts | disruption_rate (0.560) computed from old prompt set |
| `fr_failure_audit.csv` | **CURRENT** — generated from post-rapide baseline | Per-row FR breakdown is accurate |
| `gate_check.txt` | **CURRENT** | PASS confirmed |

**Critical:** IoU values are stale for two independent reasons: (1) language assignment bug (fixed in code but not yet re-run), (2) prompts have changed (rapide vs vite). The corrected IoU can only be known after a full pipeline re-run.

**C3 patching:** The 0.560 disruption rate was computed from the old prompt set. It may differ when re-run with rapide prompts because concept 2 FR prompts changed. The effect is likely small (concept 2 is 1/8 concepts) but is not confirmed.

---

## 5. Decision

### Gate status

| Metric | Value | Threshold | Status |
|---|---|---|---|
| EN sign accuracy | 1.000 (24/24) | ≥ 0.90 | **PASS** |
| FR sign accuracy | 0.667 (16/24) | ≥ 0.65 | **PASS** |
| Mean norm logprob diff | 3.511 | ≥ 1.00 | **PASS** |

The behaviour passes all gates. The 8 FR failures are explicable by known linguistic factors (reversed-template sensitivity, polysemy) and do not indicate random or contradictory model behaviour.

### Recommendation: KEEP — with full pipeline re-run before final claims

The multilingual_circuits behaviour is scientifically sound:
- EN 100% confirms the model has the antonym concept
- FR failures are linguistically principled — a researcher can report and explain them
- The gate-PASS at FR ≥ 0.65 is appropriate for the stated research goal (cross-lingual circuit analysis)

**Required before making Claims 1–3 (IoU, bridge features):**

1. Re-run the full pipeline with rapide prompts (steps 04→06→07→09 on CSD3)
2. Re-run `a_analyze_multilingual_circuits.py` with the IoU bug fix active

The corrected IoU values (true EN vs FR, not concept-group vs concept-group) are the primary unknown. Expected direction: higher than the stale 0.35–0.48 range, because EN and FR prompts for the same concepts share more features than two concept-group halves.

### If a vocabulary fix is desired before the final pipeline re-run

Replace concept 7 (`propre → sale`) with an unambiguous adjective pair. **Option A: `net → sale`** (recommended — `net` is an unambiguous French adjective meaning "clean/clear"; `sale` stays; both are common single-token words; adj→adj pair preserved).

This is optional. The gate already passes. The decision should be based on whether 1/3 concept-7 FR accuracy is acceptable for the thesis narrative.

**Do NOT touch concept 4 (vide/plein).** It is a genuine model weakness for this word pair, not a vocabulary design error. Replacing it would obscure a real finding.

---

## 6. Next Minimal Step

```bash
# On CSD3 — full pipeline re-run from step 04 (prompts already regenerated)
python scripts/04_extract_features.py --behaviour multilingual_circuits --split train
python scripts/06_build_graph.py --behaviour multilingual_circuits --split train
python scripts/07_run_interventions.py --behaviour multilingual_circuits --split train
python scripts/09_prep_ui.py --behaviour multilingual_circuits --split train
python scripts/a_analyze_multilingual_circuits.py
```

After re-run: check corrected IoU per-layer, corrected bridge feature count, updated C3 disruption rate.
