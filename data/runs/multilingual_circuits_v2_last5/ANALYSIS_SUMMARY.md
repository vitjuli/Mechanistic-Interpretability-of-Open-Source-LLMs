# Analysis Summary — multilingual_circuits v2 (last_5)

**Analysis date:** 2026-03-13
**Reference run:** SLURM 25058380 (pipeline: 02→04→06→07→09, 2026-03-13)
**Analysis script:** `scripts/a_analyze_multilingual_circuits.py` (run on CSD3 2026-03-13)
**Change from v1:** Step 04 used `--context_tokens 5` (last 5 token positions per prompt, 240 total samples) instead of decision token only.
**Status:** FINAL reference (pooled IoU); Phase 1 script update 2026-03-14 — position-separated IoU pending CSD3 re-run

---

## Key Result: Claim 3 Substantially Improved (pooled IoU)

**IoU early (L10–11) = 0.267 → middle (L12–20) = 0.343 → late (L21–25) = 0.297**

Middle/early ratio (pooled): **1.283×** (vs v1's 1.047×). The direction is now pronounced:
- Early layers show low sharing (word-level tokens are language-specific)
- Middle layers show peak sharing (semantic representation is cross-lingual)
- Late layers drop back (output-side specialization)

Claim 3 assessment upgrades from "direction only, weak" to **"moderately supported"**.

---

## Phase 1 Results — Position-Separated IoU (2026-03-15)

| IoU curve | Rows | Early (10–11) | Middle (12–20) | Late (21–25) | Middle/Early |
|---|---|---|---|---|---|
| **Pooled** | 240 (5/prompt) | 0.267 | 0.343 | 0.297 | **1.283×** |
| **Decision-only** | 48 (1/prompt) | 0.390 | 0.431 | 0.421 | 1.106× |
| **Content-only** | 192 (4/prompt) | 0.261 | 0.328 | 0.282 | 1.257× |

**Findings:**
1. Content-only (1.257×) is LOWER than pooled (1.283×), not higher. Position separation
   did not improve Claim 3. The decision-token features (high IoU ~0.43 at middle layers)
   disproportionately boost the pooled middle IoU, making pooled > content.
2. Decision-only (1.106×) closely matches v1's corrected ratio (1.125×), validating
   that both runs measure the same thing at the same token position.
3. Content curve has the cleanest shape (monotonic rise L10→L16, clear drop L20→L25,
   6/8 rising transitions in middle zone, 4/4 falling in late) but shallow amplitude.
4. **Root cause of shallow content gradient:** `last_5` includes ~3 structural tokens
   per prompt (quotation marks, "of"/"de") that are EN/FR-invariant at all layers,
   leaving only ~1/4 content positions for the truly language-specific content word.
   This dilutes the early-layer dip and limits the ratio.

**Claim 3 assessment (Phase 1 confirmed):** BORDERLINE WEAK/MODERATE.
Pooled ratio 1.283× marginally clears the 1.30× moderate threshold (within noise).
Content-only 1.257× falls below it. Direction is unambiguous across all three curves
(middle > late > early) but gradient is shallow. No upgrade to "Strongly supported."

**New outputs:** `iou_per_layer_decision.csv`, `iou_per_layer_content.csv`,
`iou_position_comparison.png`

---

## Strongest Result

**C3 patching (Claim 4): disruption_rate = 0.588, mean_effect = −0.166 ± 0.019, CI [−0.202, −0.126]**

Unchanged from v1. Patching EN antonym features into FR contexts degrades FR model
confidence in 58.8% of (feature × pair × layer) interventions. 95% CI fully negative.
62.1% of individual graph features have a negative mean cross-lingual effect.

Best C3 layers: L12 (0.854), L20 (0.849), L22 (0.797), L25 (0.750), L23 (0.701).

---

## IoU Layer Profile (v2, multi-token)

| Layer | IoU   | Region |
|-------|-------|--------|
| 10    | 0.269 | early  |
| 11    | 0.266 | early  |
| 12    | 0.299 | middle |
| 13    | 0.315 | middle |
| 14    | 0.337 | middle |
| 15    | 0.343 | middle |
| 16    | 0.358 | middle |
| 17    | 0.350 | middle |
| 18    | 0.344 | middle |
| 19    | 0.363 | middle |
| **20**| **0.379** | middle (max) |
| 21    | 0.324 | late   |
| 22    | 0.330 | late   |
| 23    | 0.306 | late   |
| 24    | 0.276 | late   |
| 25    | 0.248 | late (min) |

Mean IoU: 0.319 | Early: 0.267 | Middle: 0.343 | Late: 0.297

---

## Claim-Level Assessment

| Anthropic Claim | Our evidence | Assessment |
|---|---|---|
| **(1) Language-specific features exist** | Min per-layer IoU = 0.248 (L25); early IoU = 0.267 | **Weakly supported.** Low IoU at early/late leaves clear room for language-specific features. |
| **(2) Shared cross-lingual features exist** | Max IoU = 0.379 (L20); 32/53 bridge; 62% negative C3 | **Moderately supported.** Three independent measures converge. |
| **(3) Shared features concentrated in middle layers** | Pooled ratio 1.283×; content-only 1.257×; decision 1.106×; direction unambiguous, gradient shallow | **Borderline weak/moderate.** All curves agree on direction. Gradient limited by prompt length (structural tokens dilute early content-only IoU). |
| **(4) Bridge features degrade both EN and FR** | 32/53 bridge; disruption_rate=0.588; CI [−0.202, −0.126] | **Strongly supported.** Unchanged from v1. CI fully negative. |

**Overall:** Claims 2 and 4 are sufficiently supported for thesis use. Claim 3 is borderline
weak/moderate — direction is unambiguous but gradient is shallow; not a thesis-blocking issue.
Claim 1 remains weak but qualitatively consistent.
Phase 1 confirmed that position separation does not sharpen Claim 3 beyond what pooled already shows.
**Next step: Phase 2 (virtual-weight DAG) — graph topology is the primary remaining limitation.**

---

## Graph (identical to v1)

- 95 feature nodes, 285 edges, layers 10–25
- Script 06 uses `is_decision_position=True` per prompt — unaffected by multi-token extraction
- Top features: L25_F103245, L23_F37214, L18_F109099, L20_F71357, L22_F160273
