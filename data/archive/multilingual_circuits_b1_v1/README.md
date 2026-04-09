# multilingual_circuits_b1 — ARCHIVED 2026-04-08

**Status**: Complete. Archived to free working memory. All scripts remain in the main codebase.

---

## What This Is

Mechanistic interpretability analysis of Qwen3-4B-Base on a multilingual antonym task.
The model is asked to complete "The opposite of X is" in English and French.
The goal: find which transcoder features and circuits underlie this behaviour,
and whether those circuits are shared across languages (replicating Anthropic 2025).

**Three canonical runs** (all results preserved here):

| Run | SLURM | Date | Graph | Necessity | Key change |
|---|---|---|---|---|---|
| B1-v2 | 25679695 | 2026-03-21 | 86 nodes / 633 edges | 10.4% | OLS beta proxy (baseline) |
| B1-gradient | 26929559 | 2026-04-02 | 137 nodes / 1447 edges | 35.4% | Gradient × activation attribution |
| **B1-sparsified ★** | **27031474** | **2026-04-04** | **58 nodes / 84 edges** | **43.75%** | **k=3/layer + FIX 1 activation-weighted VW** |

---

## Key Results (B1-sparsified, canonical)

| Metric | Value |
|---|---|
| EN baseline accuracy | 1.000 |
| FR baseline accuracy | 0.792 |
| IoU middle/early ratio (pooled) | 1.090× (WEAK) |
| Bridge feature rate | 67.35% |
| Necessity (disruption rate) | **43.75%** |
| S1.5 sign preservation | 71.9% |
| S2 transfer rate | 12.5% (NEGATIVE) |
| Trajectory accuracy | 75.0% |
| Path propagation consistency (backbone) | **0.820** |
| Top causal path | input → L23_F6889 → L24_F35447 → L25_F43384 → output |
| Semantic hub | L22_F41906 (bridge score 0.677) |
| FR competitor community | 100% FR-leaning, inhibitory (confirmed script 13) |

---

## What's Here

```
prompts/
  multilingual_circuits_b1_train.jsonl    96 prompts (48 EN + 48 FR)
  multilingual_circuits_b1_test.jsonl     32 prompts (16 EN + 16 FR)

results/
  baseline_multilingual_circuits_b1_train.csv
  attribution_graphs/
    attribution_graph_train_n96.json              (star graph / beta proxy)
    attribution_graph_train_n96_roleaware.json    (gradient role-aware, B1-gradient)
    ── NOTE: B1-sparsified graph is in ui_offline/raw_sources/
  causal_edges/
    causal_edges_multilingual_circuits_b1_train.json
    circuits_multilingual_circuits_b1_train.json                  (B1-gradient circuit)
    circuits_multilingual_circuits_b1_train_roleaware_k3_t05.json (B1-sparsified circuit ★)
    manifest_multilingual_circuits_b1_train.json
    presentation_graph_multilingual_circuits_b1_train.json
  interventions/
    intervention_ablation_multilingual_circuits_b1.csv
    intervention_patching_C3_multilingual_circuits_b1.csv
  reasoning_traces/
    reasoning_traces_train.jsonl
    error_cases_train.json     (18 incorrect FR prompts)
    ablation_supplement_train.csv
    prompt_features_train.csv
    prompt_paths_train.csv
    layerwise_decision_trace_train.csv
  ui_offline/
    20260404-032832_multilingual_circuits_b1_train_n96/   ← B1-sparsified canonical UI run
      raw_sources/   ← all source JSONs including sparsified graph
      *.json / *.csv ← UI-ready files

dashboard_data/
  ← 24 JSON/CSV files from dashboard_b1/public/data/ (B1-v2 dashboard)
  ← To restore dashboard: copy these to dashboard_b1/public/data/ and npm run dev

docs/
  report_multilingual.md    ← Full 1270-line analysis report (all sections)

jobs/
  multilingual_circuits_b1_*.sbatch    ← All SLURM job scripts (scripts 08–14 + full pipeline)
```

**NOT archived here** (live in main codebase, unchanged):
- `scripts/01_generate_prompts.py` through `scripts/14_path_validation.py`
- `src/` library code (transcoder, model_utils, ui_offline)
- `data/results/transcoder_features/` — large (one npy per layer per split), stays in main repo
- `data/ui_offline/` older runs (2026-03-17 through 2026-04-03) — can be deleted if space needed

---

## How to Restore

### Option A — Dashboard only (no GPU)
```bash
cp data/archive/multilingual_circuits_b1_v1/dashboard_data/* dashboard_b1/public/data/
cd dashboard_b1 && npm run dev
# → http://localhost:5173
```

### Option B — Rerun from circuits (no feature extraction)
```bash
# Circuits are archived; skip scripts 01–07 and go straight to 08+
# Sparsified graph is at:
#   results/ui_offline/20260404-.../raw_sources/attribution_graph_train_n96_roleaware_k3_t05.json
# Copy back to live results dir first:
cp "data/archive/multilingual_circuits_b1_v1/results/ui_offline/20260404-032832_multilingual_circuits_b1_train_n96/raw_sources/attribution_graph_train_n96_roleaware_k3_t05.json" \
   data/results/attribution_graphs/multilingual_circuits_b1/

# Then rerun from step 08:
sbatch jobs/multilingual_circuits_b1_full_pipeline.sbatch  # or individual steps
```

### Option C — Full re-run from scratch (GPU, ~4h on A100)
```bash
git pull
python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1
sbatch jobs/multilingual_circuits_b1_full_pipeline.sbatch
# Step 06 canonical params:
#   --graph_node_mode role_aware --top_k_per_layer 3 --vw_threshold 0.05
#   --activation_weighted --output_suffix _roleaware_k3_t05
```

---

## Methodology Summary

**Pipeline**: 01 (prompts) → 02 (baseline) → 04 (features) → 06 (graph) → 07 (interventions) → 08 (causal edges) → 09 (UI prep) → 10 (traces) → 11 (ablation supplement) → 12 (competition) → 13 (L22 intervention) → 14 (path validation)

**Attribution formula**:
```
α_k^ℓ(p) = a_k^ℓ(p) × (∂Δlogit(p)/∂MLP_output^ℓ) · W_dec^ℓ[k,:]
```

**VW edge formula (FIX 1, activation-weighted)**:
```
edge(i→j) = mean_act_i × W_enc_{ℓ+1}[j,:] · W_dec_ℓ[:,i]
edge_type = "attribution_approx_v1"
```

**Path validation (FIX 3)**: Ablate feature A at L_A → measure Δact_B at L_B + Δlogit at output.
3 metrics: mean_delta_act_B, mean_delta_logit, propagation_consistency.

---

## Known Limitations

1. **IoU gradient weak (1.090×)**: Dataset too small (3 prompts/concept/language) to cleanly separate structural from semantic overlap. No random baseline computed — significance unknown.
2. **S2 transfer fails**: EN→FR language-component patching shifts logit by −1.125 (negative). Circuit not cleanly decomposable along language lines.
3. **Community fragmentation at sparsification boundary**: B1-sparsified density=1.45 edges/node causes 17-community fragmentation. Use B1-v2 communities for topology narrative.
4. **No attention decomposition**: Hook at post_attention_layernorm misses all QK-mediated interactions.
5. **No test-set validation**: Circuit validated on train split only.

---

## Archived: 2026-04-08
## Behaviour type: Type 2 — Candidate Set (EN/FR antonym prediction)
## Model: Qwen3-4B-Base + per-layer transcoders (mwhanna/qwen3-4b-transcoders)
