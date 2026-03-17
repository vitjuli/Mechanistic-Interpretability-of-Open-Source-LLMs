# Run Index — multilingual_circuits

All versioned snapshots of analysis runs. Each folder is a self-contained
reference: prompts + baseline + interventions + analysis outputs, frozen at
the time the analysis script was run.

---

## v1 — multilingual_circuits_v1

**Analysis date:** 2026-03-12
**SLURM job:** 24386104 (pipeline: 02→04→06→07→09, 2026-03-06)
**Analysis script:** `scripts/a_analyze_multilingual_circuits.py` (run on CSD3 2026-03-12)

### Prompt set (pinned)
| File | Lines | Key fix |
|---|---|---|
| `multilingual_circuits_train.jsonl` | 48 | FR concept 2: `rapide→lent` (was `vite→lent`, adverb/adj mismatch) |
| `multilingual_circuits_test.jsonl` | 16 | Same fix |

### Pipeline artifacts
| File | Description |
|---|---|
| `baseline_multilingual_circuits_train.csv` | Qwen3-4B-Instruct, 48 prompts, all 1-token answers |
| `intervention_ablation_multilingual_circuits.csv` | 53 features × 48 prompts × layers (2544 rows) |
| `intervention_patching_C3_multilingual_circuits.csv` | 24 EN→FR pairs × layers × features (2280 rows) |

### Analysis outputs
| File | Description |
|---|---|
| `gate_check.txt` | Baseline gate: PASS |
| `iou_per_layer.csv` | 16 layers, corrected EN/FR language assignment |
| `bridge_features.csv` | All 53 ablated features with per-language effects |
| `bridge_features_only.csv` | 32 bridge features (mean_effect < 0 in BOTH EN and FR) |
| `c3_patching_stats.txt` | C3 disruption summary + per-layer + per-concept |
| `c3_patching_per_feature.csv` | Per-feature lang-swap strength |
| `REPORT.md` | Human-readable claim-level summary |

### Key metrics
| Metric | Value | Target | Status |
|---|---|---|---|
| EN sign_accuracy | 1.000 (24/24) | ≥ 0.90 | PASS |
| FR sign_accuracy | 0.667 (16/24) | ≥ 0.65 | PASS |
| mean_norm_logprob_diff | 3.511 | ≥ 1.00 | PASS |
| IoU mean (all layers) | 0.423 | — | — |
| IoU early (10–11) | 0.390 | — | — |
| IoU middle (12–20) | 0.439 | — | — |
| IoU late (21–25) | 0.422 | — | — |
| **Middle/early ratio** | **1.125×** | > 1 | WEAK ✓ |
| Bridge features | 32/53 (60.4%) | — | — |
| C3 disruption_rate | 0.588 | ≥ 0.40 | PASS |
| C3 mean_effect_size | −0.166 ± 0.019 | CI fully negative | ✓ |

**Note (2026-03-15):** Middle/early ratio was previously recorded as 1.047× using a
non-standard formula (mid / mean(early∪late)). Corrected to 1.125× (mid/early).
See `COMPARISON_v1_v2.md` for details.

### Fixes active in this run
1. FR concept 2 vocabulary: `vite` → `rapide` (adj/adj match)
2. Template-matched split: Fix 1 (same held-out template per concept across EN and FR)
3. IoU language assignment: JSONL-derived EN/FR indices (not index-based slicing)
4. Bridge language label: JSONL dict join (not `prompt_idx < n_en` lambda)
5. Gate metric: `logprob_diff_normalized > 0` (was `logprob_diff`)

### Graph
- 95 feature nodes, 285 edges, layers 10–25
- Top features by specific_score: L25_F103245, L23_F37214, L18_F109099, L20_F71357, L22_F160273

---

## v2 — multilingual_circuits_v2_last5

**Analysis date:** 2026-03-13
**SLURM job:** 25058380 (pipeline: 02→04→06→07→09, 2026-03-13)
**SBATCH:** `jobs/multilingual_circuits_v2_last5_02_09.sbatch`
**Change vs v1:** Step 04 uses `--context_tokens 5` → `token_positions="last_5"` (240 samples, 5 per prompt) instead of decision token only (48 samples). All other steps identical.

### Purpose
Improve Claim 3. Decision-token IoU (v1) loses the layer-wise language-specificity gradient because the residual stream at the last token has already finished lexical/language processing. With `last_5`, content-word positions (e.g. `"rapide"`) are included; early-layer features there differ by language, creating a steeper early-vs-middle IoU contrast.

### Analysis script change
`compute_iou()` in `scripts/a_analyze_multilingual_circuits.py` updated (backward-compatible): detects multi-token mode via `idx.shape[0] != n_prompts`, loads `position_map.json` once to map sample rows → prompt_idx, groups EN/FR samples correctly.

### Key metrics
| Metric | Value | vs v1 (corrected) |
|---|---|---|
| EN sign_accuracy | 1.000 | same |
| FR sign_accuracy | 0.667 | same |
| IoU mean (all layers) | 0.319 | 0.423 (lower absolute, expected) |
| IoU early (10–11) — pooled | 0.267 | 0.390 |
| IoU middle (12–20) — pooled | 0.343 | 0.439 |
| IoU late (21–25) — pooled | 0.297 | 0.422 |
| **Middle/early ratio — pooled** | **1.283×** | **1.125×** (corrected from 1.047×) |
| IoU early — decision-only | 0.390 | 0.390 (matches v1 ✓) |
| IoU middle — decision-only | 0.431 | 0.439 |
| Middle/early — decision-only | 1.106× | 1.125× (validates v1) |
| Middle/early — content-only | 1.257× | — |
| IoU max | L20 = 0.379 (pooled) | L16 = 0.493 |
| IoU min | L25 = 0.248 (pooled) | L25 = 0.360 |
| Bridge features | 32/53 (60.4%) | same |
| C3 disruption_rate | 0.588 | same |
| C3 mean_effect_size | −0.166 ± 0.019 | same |

### Claim 3 assessment (Phase 1, 2026-03-15)
- v1 (corrected): **1.125×** (decision-only; previously 1.047× due to formula error)
- v2 decision-only: **1.106×** (matches v1 — same measurement, validates consistency)
- v2 content-only: **1.257×** (content positions diluted by structural tokens)
- v2 pooled: **1.283×** (best available signal; borderline moderate/weak)
- All three curves: middle > late > early — direction unambiguous, gradient shallow
- **Status: borderline weak/moderate — direction confirmed, amplitude modest**

### Phase 1 new outputs (in `data/analysis/multilingual_circuits/`)
- `iou_per_layer_decision.csv` — decision-token IoU per layer
- `iou_per_layer_content.csv` — content-position IoU per layer
- `iou_position_comparison.png` — comparison figure

See `COMPARISON_v1_v2.md` for full comparison.

---

## v3 — multilingual_circuits_v3_vw

**Analysis date:** 2026-03-16
**SLURM job:** 25236913 (step 06 only; features from SLURM 25058380)
**SBATCH:** `jobs/multilingual_circuits_v3_vw_06.sbatch`
**UI run:** `data/ui_offline/20260316-203723_multilingual_circuits_train_n48/`
**Change vs v2:** Step 06 uses `--vw_threshold 0.01` → adds virtual-weight inter-feature edges between adjacent layer pairs. All other steps identical to v2.

### Graph metrics
| Metric | v2 (star) | v3 (DAG) |
|---|---|---|
| Feature nodes | 95 | 95 (unchanged) |
| Star edges | 285 | 285 |
| VW inter-feature edges | 0 | **539** |
| Total edges | 285 | **824** |
| Louvain communities | 1–2 (flat) | **4** (layer-structured) |
| VW threshold | — | 0.01 |
| VW |weight| range | — | 0.0102 – 2.1356 |
| VW |weight| mean/median | — | 0.117 / 0.071 |

### VW edge density by layer pair
| Pair | VW edges | | Pair | VW edges |
|---|---|-|---|---|
| L10→L11 | 51 | | L19→L20 | 23 |
| L11→L12 | 32 | | L20→L21 | 45 |
| L12→L13 | 33 | | L21→L22 | 43 |
| L13→L14 | 6  | | L22→L23 | 79 |
| L14→L15 | 4  | | L23→L24 | 110 |
| L15→L16 | 4  | | L24→L25 | 97 |
| L16→L17 | 2  | | — | — |
| L17→L18 | 4  | | — | — |
| L18→L19 | 6  | | — | — |

**Pattern:** Early (L10–L13) and late (L20–L25) zones are densely connected; middle (L13–L19) is sparse bottleneck (2–6 edges per pair).

### Louvain communities (4)
| Community | Features | Layer range | I/O node | Interpretation |
|---|---|---|---|---|
| C0 | 3 | L24–L25 | — | Late outlier cluster |
| C1 | 28 | L10–L13 | output_correct | Early features → correct output |
| C2 | 29 | L14–L22 | output_incorrect | Middle features → incorrect output |
| C3 | 35 | L22–L25 | input | Late features ← input |

L22 is split between C2 and C3, marking the transition zone.

### Interpretation
The 4 communities reveal functional decomposition absent in the star graph:
- C1 (early): processes input tokens, feeds correct-answer direction
- C2 (middle): executes antonym transformation, connected to output_incorrect direction
- C3 (late): strongly input-driven late features; peak cross-lingual IoU (L20) sits at the C2/C3 boundary
- The sparse middle zone (L13–L19) aligns with the IoU rise from L10→L20 in v2 analysis

---
