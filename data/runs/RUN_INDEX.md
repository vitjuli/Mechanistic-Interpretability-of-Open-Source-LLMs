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
| IoU middle (12–20) | 0.431 | > early/late | WEAK ✓ |
| IoU early/late | 0.412 | — | — |
| Bridge features | 32/53 (60.4%) | — | — |
| C3 disruption_rate | 0.588 | ≥ 0.40 | PASS |
| C3 mean_effect_size | −0.166 ± 0.019 | CI fully negative | ✓ |

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
| Metric | Value | vs v1 |
|---|---|---|
| EN sign_accuracy | 1.000 | same |
| FR sign_accuracy | 0.667 | same |
| IoU mean (all layers) | 0.319 | 0.423 (lower absolute, expected) |
| IoU early (10–11) | 0.267 | 0.390 |
| IoU middle (12–20) | 0.343 | 0.431 |
| IoU late (21–25) | 0.297 | 0.421 |
| **Middle/early ratio** | **1.283×** | 1.047× |
| IoU max | L20 = 0.379 | L16 = 0.493 |
| IoU min | L25 = 0.248 | L25 = 0.360 |
| Bridge features | 32/53 (60.4%) | same |
| C3 disruption_rate | 0.588 | same |
| C3 mean_effect_size | −0.166 ± 0.019 | same |

### Claim 3 upgrade
- v1: 1.047× (direction only, weak)
- v2: **1.283×, middle > late > early** — moderately supported

See `COMPARISON_v1_v2.md` for full comparison.

---
