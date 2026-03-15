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

## Phase 1 Update — Position-Separated IoU (script updated 2026-03-14)

`scripts/a_analyze_multilingual_circuits.py` now computes three IoU curves when run
in multi-token mode (v2 data, `position_map.json` required):

| Curve | Rows used | Expected behavior | Claim 3 signal? |
|---|---|---|---|
| **Pooled** | All 240 rows (5 per prompt) | Already computed above | Moderate (1.283×) |
| **Decision** | 48 rows (`is_decision_position=True`) | Flat layer profile (~1.05×); semantic token, no language contrast | Low |
| **Content** | 192 rows (`is_decision_position=False`) | Steeper gradient; lexical token, language-specific early layers | **Primary signal** |

**To obtain position-separated results:** re-run analysis script on CSD3:
```bash
python scripts/a_analyze_multilingual_circuits.py --behaviour multilingual_circuits --split train
```
This requires `data/results/transcoder_features/multilingual_circuits_train_position_map.json`
and layer-wise `top_k_indices.npy` files (on CSD3, not synced locally).

**New outputs after CSD3 run:**
- `iou_per_layer_decision.csv` — decision-token IoU per layer
- `iou_per_layer_content.csv` — content-position IoU per layer
- `iou_position_comparison.png` — figure with all three curves

**Claim 3 upgrade criteria (content-position ratio):**
- ≥ 1.50× → Strongly supported
- 1.30–1.50× → Moderately supported (no change from current)
- < 1.30× → No upgrade warranted; pooled ratio remains the reference

**Do not update the claim assessment below until actual numbers are available.**

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
| **(3) Shared features concentrated in middle layers** | Middle 0.343 > early 0.267 > late 0.297; ratio 1.283× | **Moderately supported.** Clear, consistent gradient. Middle layer peak is now unambiguous. |
| **(4) Bridge features degrade both EN and FR** | 32/53 bridge; disruption_rate=0.588; CI [−0.202, −0.126] | **Strongly supported.** Unchanged from v1. CI fully negative. |

**Overall:** Claims 2, 3, and 4 are sufficiently supported for thesis use. Claim 1 remains weak but
qualitatively consistent. The multi-token measurement resolves the main limitation of v1.

---

## Graph (identical to v1)

- 95 feature nodes, 285 edges, layers 10–25
- Script 06 uses `is_decision_position=True` per prompt — unaffected by multi-token extraction
- Top features: L25_F103245, L23_F37214, L18_F109099, L20_F71357, L22_F160273
