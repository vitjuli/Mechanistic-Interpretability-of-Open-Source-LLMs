# multilingual_circuits_v2_last5 — PENDING

This directory will hold the v2 run snapshot after CSD3 completion.

## Change from v1

Step 04 uses `--token_positions last_5` (5 token positions per prompt, 240 total samples) instead of `decision` (1 token, 48 samples).

All other steps (02, 06, 07, 09) are identical to v1.

## Purpose

Improve Claim 3 (middle-layer concentration of shared features). The decision-token IoU in v1 showed only a 1.047× middle/early ratio — too weak for a strong claim. With `last_5`, content-word tokens (e.g. `"rapide"`) are included; early-layer features there are language-specific, creating the layer-wise gradient that Anthropic observed.

## Pipeline

SBATCH: `jobs/multilingual_circuits_v2_last5_02_09.sbatch`

## Prompts

Same as v1 (unchanged):
- `multilingual_circuits_train.jsonl` — 48 prompts (24 EN + 24 FR)
- `multilingual_circuits_test.jsonl` — 16 prompts (8 EN + 8 FR)

## Status

- [ ] CSD3 run submitted
- [ ] Pipeline complete
- [ ] Analysis run
- [ ] Snapshot populated
