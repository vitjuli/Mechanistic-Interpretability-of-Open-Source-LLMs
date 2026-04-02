# Multilingual Circuits B1 — Final Archive
**Archived:** 2026-03-25 | **Status:** COMPLETE — superseded by `physics_conservation` behavior

---

## 1. What This Was

Reproduction + extension of Anthropic's multilingual circuits paper using Qwen3-4B + transcoders.
Core question: Do EN and FR antonym prompts share circuit features? Do shared features concentrate in middle layers (Claim 3)?

Behavior: `multilingual_circuits_b1`
- 8 cross-language antonym concepts (hot/cold, big/small, ...)
- 8 prompt templates per concept per language (T0–T7), FR templates rewritten for B1-v2
- 96 train (48 EN + 48 FR), 32 test (16 EN + 16 FR)
- Binary: predict antonym word (correct vs incorrect token)

---

## 2. Canonical Version: B1-v2 (SLURM 25679695, 2026-03-21)

**Why v2, not v1:** B1-v1 had a quote-completion artifact (`"{word}"` pattern in all FR templates caused model to predict closing quote). B1-v2 removed the pattern from all 8 FR templates.

**Baseline gate (B1-v2):**
- EN accuracy: 1.000 (48/48 correct)
- FR accuracy: 0.792 (38/48 correct)
- mean_norm_logprob_diff: 3.511
- Status: PASS

**Graph:**
- Star graph: 84 features, 252 edges
- Role-aware graph: 86 features, 633 edges (+ 4 pure content + 6 "both" nodes)
- Layers: 10–25

---

## 3. Authoritative Thesis Numbers

### Claim 3 — Shared features concentrate in middle layers
| Mode     | Early | Middle | Late | Ratio (mid/early) | Status |
|----------|-------|--------|------|--------------------|--------|
| Pooled   | 0.2375 | 0.2589 | 0.2262 | **1.090×** | WEAK |
| Decision | 0.2182 | 0.2262 | 0.2363 | 1.037×     | FLAT  |
| Content  | 0.2217 | 0.2352 | 0.2012 | 1.061×     | WEAK  |

**Use 1.090× (pooled) as the thesis number. Direction unambiguous; gradient shallow.**

### Claim — Bridge features
- 33/49 circuit features are bridge (EN+FR active): **67.35%**
- C3 disruption: 0.645, CI [−0.403, −0.343]

### Language profiles (86 nodes, role-aware graph)
- balanced: 67 (77.9%)
- fr_leaning: 15 (17.4%)
- en_leaning: 4 (4.7%)
- insufficient: 0

**KEY:** fr_leaning dropped from 44 (v1) to 15 (v2) — most v1 fr_leaning features were quote-completion artifacts.

### Communities (7, VW-subgraph 379 edges)
- C0 (L10–16, balanced, early input processing)
- C1 (L17–20, 100% balanced — cross-lingual semantic core)
- C3 (L20–23, 93% balanced — L22_F41906 hub, genuine semantic transformer)
- C4 (L23–25, balanced — output preparation)
- **C2 (L21–25, 100% fr_leaning):** L22_F108295 + L23_F64429 — FR-specific COMPETITOR circuit

### Causal edges (Script 08, SLURM 25743140)
- Circuit: 21 feature nodes + 3 I/O = 24 total, 55 edges, 50 paths
- Top path: input → L21_F27974 → L22_F41906 → output_correct
- Necessity: 0.1042 (+0.932) — DISTRIBUTED (no single bottleneck)
- S2 transfer: 0.125, shift: −1.309 — WEAK (Claim 5 downgraded)

### Reasoning traces (Script 10, SLURM 25765101)
- 78/96 correct, 18 incorrect (all FR)
- 16/18 incorrect: dominant trajectory = correct (late-layer reversal untraced)
- L22_F41906: top discriminator (Δ = −0.727 correct minus incorrect)
- L13_F70603: preserved cross-language feature (Δ = −0.635)
- L23_F64429: HURTS incorrect prompts (fr_leaning competitor)

### Competition analysis (Script 12)
- argmax_is_quote_rate (residual): 0.278 (5/18 — word-specific, not template)
- mean margin: −4.288; median rank: 78

---

## 4. Key Data Paths

| Item | Path |
|------|------|
| Prompts | `data/prompts/multilingual_circuits_b1_{train,test}.jsonl` |
| B1-v2 analysis | `data/analysis/multilingual_circuits/` |
| B1-v2 snapshot | `data/runs/multilingual_circuits_v2_last5/` |
| UI data | `data/ui_offline/20260306-104332_multilingual_circuits_train_n48/` |
| Attribution graph | `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json` |
| Intervention CSVs | `data/results/interventions/multilingual_circuits_b1/` |
| Reasoning traces | `data/results/reasoning_traces/multilingual_circuits_b1/` |
| Causal edges | `data/results/causal_edges/multilingual_circuits_b1/` |
| Dashboard data | `dashboard_b1/public/data/` |

---

## 5. Dashboard

`dashboard_b1/` — React 19 + Vite + Plotly + D3 + Zustand
- Launch: `cd dashboard_b1 && npm run dev` → http://localhost:5173
- Data prep: `python scripts/prepare_b1_dashboard.py`
- Tabs: Prompt, Trace, Timeline, Compare, Interventions, Overlap (prompt-level) + IoU, Communities, Failures, Heatmap, Scatter, Experiment, Features (global)

---

## 6. Claim Status Summary

| Claim | Metric | Value | Status |
|-------|--------|-------|--------|
| C1: model does the behavior | EN acc | 100% | STRONG |
| C2: shared circuit features | bridge ratio | 67.35% | MODERATE |
| C3: middle-layer IoU peak | pooled 1.090× | direction clear | WEAK/BORDERLINE |
| C5: circuit transfer | S2 shift −1.309 | | WEAK |
| Community split (C2 vs C3/C4) | late-layer FR competitor | confirmed | NOVEL FINDING |

---

## 7. Known Limitations

- Claim 3 gradient is shallow (1.090×); limited by 3 prompts/concept × concept-paired IoU = worse (1.162×)
- 18/96 incorrect prompts are all FR; late-layer correction mechanism untraced
- quote-completion residual: 5/18 still fail for word-specific reasons (chaud/propre)
- Fundamental constraint: only 3 prompts per (concept, language) in training split

---

*See also: `runs/RUN_INDEX.md` for full SLURM job history. All authoritative numbers also in `MEMORY.md`.*
