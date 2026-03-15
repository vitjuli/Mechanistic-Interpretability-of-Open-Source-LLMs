# Next Pipeline Improvement — Claim 3 (Middle-Layer Concentration)

**Decision: multi-token-position feature extraction**

---

## The problem, precisely

IoU is computed by comparing the set of top-50 transcoder features active at the
**decision token** (last token position) for EN prompts vs FR prompts.

At the decision token (e.g., the word "est" / "is" at the end of the prompt), the model
has already completed its language-specific lexical processing. The residual stream at this
position carries a mixture of cross-lingual semantic content — the same antonym circuit
fires regardless of input language. There is no strong reason for early-layer features
to be more language-specific at this single position.

Anthropic's result relies on the full token sequence: early-layer features at
*word-level tokens* (e.g., "rapide", "lent") are strongly tied to the surface form of
those words, while middle-layer features at later tokens are more semantic. When you
measure IoU across all token positions, early layers show high language-specificity
(because word-level representations are language-specific) and middle layers show
high sharing (because semantic representations are more language-agnostic).

Our current approach collapses to one position and loses this gradient.

---

## Why NOT the alternative (improved FR prompt set)

Improving the FR prompt set (e.g., replacing `vide/plein` or `propre/sale`) would:
- Raise FR sign_accuracy from 0.667 to perhaps 0.75
- Slightly change which concepts are in the bridge/C3 analysis
- Have **zero effect on the IoU layer profile** — the same single decision token would
  be measured, producing the same ~1.05x middle/early ratio

The FR accuracy is already above the gate threshold. Fixing more vocabulary pairs is
a cleanup task, not a measurement improvement. It does not address Claim 3 at all.

---

## The fix: `token_positions="last_N"` in script 04

Script 04 (`04_extract_transcoder_features.py`) already supports the `--token_positions`
argument with modes `"decision"`, `"all"`, `"last_N"` (where N is an integer).

The change is a single flag addition to the SBATCH step 04 command and the analysis script's IoU function.

**What changes:**

| | Current | Proposed |
|---|---|---|
| `--token_positions` | `decision` (1 token) | `last_5` (last 5 tokens per prompt) |
| Samples per prompt | 1 | 5 |
| Total samples in `.npy` | 48 | 240 |
| IoU computation | top-50 features at position T | union of top-50 features across last-5 positions |
| Expected effect | flat IoU across layers (0.39–0.49) | lower early-layer IoU (language-specific tokens) → steeper middle peak |

**Why `last_5` specifically:**
- The last 5 tokens of a prompt like `Le contraire de "rapide" est` cover: `"rapide"`, `"`, ` est` (and padding if shorter). These include the content word in quotes, which is the most language-specific position. The final token (`est`) is already semantic. Using 5 tokens captures both the language-specific (early layers at the word token) and semantic (later layers at the last token) gradient.
- `last_3` would work but may miss the content word for longer prompts.
- `all` is expensive (variable-length, up to 15+ tokens) and mixes many padding/function-word positions that add noise.

---

## Concrete implementation (minimal changes)

### 1. Script 04 — add `--token_positions last_5` flag

In `jobs/multilingual_circuits_02_09.sbatch`, change the step 04 command:

```diff
 python -u scripts/04_extract_transcoder_features.py \
     --behaviour "$BEHAVIOUR" \
     --split "$SPLIT" \
     --model_size 4b \
     --layers $LAYERS \
-    --top_k 50
+    --top_k 50 \
+    --token_positions last_5
```

### 2. Analysis script — IoU aggregation over multiple positions

`compute_iou()` currently reads `idx.shape = (n_prompts, top_k)` (decision mode).
With `last_5`, the `.npy` shape is `(n_prompts × 5, top_k)`. The analysis script already
handles this: when `idx.ndim == 3`, it slices `idx[:, 0, :]` (decision token only).
For `last_5`, the shape is `(n_samples, top_k)` where `n_samples = n_prompts × 5`.

The correct approach: for each prompt, take the **union** of top-50 features across all
5 positions, then compute IoU on those per-prompt feature sets. This makes EN and FR
per-prompt feature sets larger and more informative for layer-level comparison.

The change to `compute_iou()` is straightforward: instead of indexing by prompt, group
samples by `prompt_idx` from the position map and take the union per prompt. The position
map (`multilingual_circuits_train_position_map.json`) already exists and links each sample
to a `(prompt_idx, token_pos, is_decision_position)`.

### 3. No change to graph, interventions, or baseline

Steps 02, 06, 07, and 09 are unaffected. Only step 04 and the IoU computation in the
analysis script change. The bridge features and C3 patching continue to use the decision
token (which is correct — interventions target the last-token logit difference).

---

## Expected outcome

If the multi-token IoU reproduces the Anthropic direction strongly:
- Early layers (10–11): IoU drops noticeably (word-level tokens are language-specific)
- Middle layers (12–20): IoU peak becomes more pronounced
- Late layers (21–25): IoU intermediate or slightly lower than middle

This would change Claim 3 from "direction only, weak" to "moderately supported".

If the effect is still small:
- The limitation is not measurement but genuinely architectural: Qwen3-4B's
  transcoders may not show the same layer-wise language/semantic gradient as
  Anthropic's SAE on a larger model with longer prompts. This is a valid
  thesis finding in itself — it documents where the replication breaks down.

---

## Cost

- One additional SBATCH run of step 04 only (~15–20 min at most, 5× more samples)
- Then re-run the analysis script with updated IoU function (~1 min, CPU only)
- No model weights needed for analysis; only the `.npy` files need regenerating
- Steps 06, 07, 09 do NOT need to be re-run

**This is the lowest-cost, highest-information-value improvement available.**
