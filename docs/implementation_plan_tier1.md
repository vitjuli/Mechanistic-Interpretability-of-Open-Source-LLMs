# Tier 1 Implementation Plan
## Features: T1-A (Physics Suite) · T1-B (Taxonomy Tags) · T1-C (Circuit Generalization) · T1-D (Auto NL Explanation)

**Status**: PLAN ONLY — not yet implemented. Review before coding.

---

## 1. Technical Assessment

### 1.1 Existing Data Structures and What Must Change

**Prompt JSONL** (`data/prompts/{behaviour}_{split}.jsonl`)

Current fields (from `multilingual_circuits_b1_train.jsonl`):
```json
{"prompt": "...", "correct_answer": " large", "incorrect_answer": " small",
 "concept_index": 0, "template_idx": 0, "language": "en", "cross_lang_valid": true}
```

New fields required for physics behaviors (T1-A):
```json
{"prompt": "...", "correct_answer": " alpha", "incorrect_answer": " beta",
 "concept_index": 0, "template_idx": 0,
 "behaviour": "physics_decay_type",
 "behaviour_type": "latent_state",
 "physics_concept": "alpha_decay",
 "difficulty": "standard",
 "numeric_variant": null}
```

**No schema migration needed for existing behaviors** — script 01 output is JSONL and the pipeline only reads the `prompt`, `correct_answer`, `incorrect_answer` fields for everything past script 02. The extra fields are passed through transparently as metadata.

**`run_manifest.json`** — needs `behaviour_type` field added:
```json
{"behaviour": "physics_decay_type", "behaviour_type": "latent_state", ...}
```
This is written in `prepare_b1_dashboard.py` / `scripts/09_prepare_offline_ui.py`. One-line change.

**`community_summary.json`** — needs optional `behaviour_type` and `label` fields per community:
```json
{"community_id": 0, "label": "Early cross-lingual", "behaviour_type": "abstraction", ...}
```
Written by `a_analyze_multilingual_circuits.py` / `prepare_b1_dashboard.py`. Community labels are currently hardcoded in the prepare script — this is fine for manual assignment.

**`circuit.json`** — needs `behaviour_type` field at top level:
```json
{"n_features": 16, "behaviour_type": "abstraction", "behaviour_type_label": "Abstraction", ...}
```
Written by `scripts/08_causal_edges.py`. One-line addition.

**`prompt_traces.json`** — no schema change needed. The `behaviour_type` comes from the `run_manifest.json` which is already loaded by the dashboard.

**UI store** (`src/store/useStore.js`) — no new state needed for T1-B tags (they are display-only, derived from loaded data). T1-C (generalization) and T1-D (explanation) need new state fields (see per-feature plans below).

**`loader.js` / `indexes.js`** — no changes needed for T1-B. T1-C adds a computed `stabilityIndex` to indexes. T1-D adds a `generateExplanation()` utility (pure function, no fetch).

---

### 1.2 Riskiest Assumptions for Physics Behaviors (T1-A)

**Risk 1 — Single-token answers may not hold for all physics behaviors.**
The pipeline requires `correct_answer` and `incorrect_answer` to be single tokens (one tokenizer ID). For antonym behaviors this was audited carefully (B1-v1 had `froid` = 2 tokens). For physics:
- "alpha" → probably single token ` alpha` ✓
- "beta" → probably single token ` beta` ✓
- "classical" → probably single token ` classical` ✓
- "relativistic" → likely 2+ tokens ⚠️
- "allowed" / "forbidden" → need to audit ⚠️
- "yes" / "no" → single tokens ✓
- "intensive" / "extensive" → need to audit ⚠️

**Mitigation**: run a tokenization audit (like `scripts/b_tokenize_audit_multilingual.py`) for all physics answer tokens before running any pipeline step. Exclude any answer pair where either token is multi-token. Redesign the prompt format if needed (e.g., " Yes"/" No" instead of " classical"/" relativistic").

**Risk 2 — The baseline gate may fail for some physics behaviors.**
Script 02 gate requires EN sign_accuracy ≥ 0.90. If Qwen3-4B does not reliably answer physics questions correctly (plausible for harder behaviors like B2 decay chains), the gate will block all downstream analysis. This is not a bug — it is the correct behavior — but it means some physics behaviors may need different prompt engineering before they can run.

**Mitigation**: Run script 02 alone first on each behavior's 5-prompt mini-set before committing to full pipeline runs. Use `--no_gate` flag if available, or lower the threshold for initial exploration.

**Risk 3 — Attribution graphs may be empty or trivial for some physics behaviors.**
If the model answers with very high confidence (logit diff >> 5), gradient × activation attribution collapses all features to similar scores (saturated softmax flattens gradients). If it answers with very low confidence (logit diff ≈ 0), the graph will be full of noise. Physics B3/B4/B5/B6 (selection rules, gauge equivalence) may be in the low-confidence zone for Qwen3-4B-Base.

**Mitigation**: Before building UI, check logit margin distribution across all 40 Phase 0 prompts. Flag behaviors where >50% of prompts have logit diff < 1.0.

**Risk 4 — The `GENERATORS` pattern assumes the generator controls train/test split sizes.**
Current behavior generators (grammar_agreement, physics_scalar_vector_operator) take `n_train` and `n_test` as arguments and produce exactly those many prompts. The physics B1–B8 generators will need to be designed the same way — but the factorial design (concept × template × difficulty) may produce a fixed total that doesn't divide cleanly into configurable sizes. 

**Mitigation**: Design each generator to produce a fixed full set and then split it. Use `seed`-based shuffle to make train/test split deterministic.

**Risk 5 — Multi-behavior dashboard: the current dashboard is hard-coded for one behavior.**
`run_manifest.json` has a single `behaviour` field. `prepare_b1_dashboard.py` has `BEHAVIOUR = "multilingual_circuits_b1"` hardcoded. If we want a dashboard showing multiple physics behaviors simultaneously, the data model needs a multi-behavior design. If we want separate dashboards per behavior (simpler), no change is needed.

**Decision needed**: single-behavior dashboard per physics behavior (run `prepare_b1_dashboard.py --behaviour physics_decay_type`)? Or one unified dashboard? **Recommendation: separate per-behavior runs for Phase 1, unified view deferred to Phase 2.**

---

### 1.3 Dependencies Between Features

```
T1-A (Physics Suite)
  └─ Phase 0 (5 prompts for B1 + B6 through full pipeline)
  └─ Phase 1 (all 8 × 5 prompts)

T1-B (Taxonomy Tags)
  ├─ Depends on: behaviour_type field in prompt JSONL and manifest (T1-A creates these)
  ├─ Can be implemented in UI immediately using existing multilingual behavior
  └─ Can run in parallel with T1-A Phase 1

T1-C (Circuit Generalization)
  ├─ Depends on: existing 96-prompt multilingual dataset (no new data needed for initial test)
  ├─ Requires: prompt_traces.json has template_idx field (it does — via reasoning traces)
  └─ Can be implemented as a pure frontend computation once T1-B is done

T1-D (Auto NL Explanation)
  ├─ Depends on: existing prompt_traces.json, circuit.json, community_summary.json (all available)
  ├─ No new backend computation needed
  └─ Fully independent — can be done at any point
```

---

## 2. Phase 0 Prompt Sets

### 2.1 B1 — Decay Type (Latent-State)

**Hypothesis**: prompt encodes nucleus parameters → model infers emitted particle type → model classifies decay. The circuit should have intermediate features encoding charge (+2e → alpha) and mass number (4 → He-4).

**Answer tokens**: ` alpha` / ` beta` — audit required but likely single-token.

**Important note on token choice**: ` gamma` is also a valid decay type and should NOT appear as a distractor (the model may output it for some prompts, making the binary logit diff undefined). Keep prompts strictly in contexts where the answer is unambiguously alpha vs beta.

```jsonl
{"prompt": "A nucleus emits a particle with charge +2e and mass number 4. The decay type is:", "correct_answer": " alpha", "incorrect_answer": " beta", "behaviour": "physics_decay_type", "behaviour_type": "latent_state", "physics_concept": "alpha_decay", "template_idx": 0, "concept_index": 0, "difficulty": "standard", "numeric_variant": "Z2_A4"}
{"prompt": "An unstable nucleus releases a helium-4 nucleus during decay. This process is called:", "correct_answer": " alpha", "incorrect_answer": " beta", "behaviour": "physics_decay_type", "behaviour_type": "latent_state", "physics_concept": "alpha_decay", "template_idx": 1, "concept_index": 0, "difficulty": "standard", "numeric_variant": "he4_direct"}
{"prompt": "A radioactive nucleus emits a particle consisting of 2 protons and 2 neutrons. The decay is classified as:", "correct_answer": " alpha", "incorrect_answer": " beta", "behaviour": "physics_decay_type", "behaviour_type": "latent_state", "physics_concept": "alpha_decay", "template_idx": 2, "concept_index": 0, "difficulty": "standard", "numeric_variant": "proton_neutron_count"}
{"prompt": "During radioactive decay, a nucleus ejects a particle with atomic number 2 and mass number 4. The decay type is:", "correct_answer": " alpha", "incorrect_answer": " beta", "behaviour": "physics_decay_type", "behaviour_type": "latent_state", "physics_concept": "alpha_decay", "template_idx": 3, "concept_index": 0, "difficulty": "indirect", "numeric_variant": "Z2_A4_indirect"}
{"prompt": "A nucleus spontaneously emits a ⁴He nucleus. This radioactive process is known as:", "correct_answer": " alpha", "incorrect_answer": " beta", "behaviour": "physics_decay_type", "behaviour_type": "latent_state", "physics_concept": "alpha_decay", "template_idx": 4, "concept_index": 0, "difficulty": "notation", "numeric_variant": "isotope_notation"}
```

**Note on concept_index=0 for all 5**: these are all surface variants of the same underlying computation (α-decay identification). In Phase 1, we would add concept_index=1 with β-decay prompts (where correct is ` beta`, incorrect is ` alpha`) to create a proper bidirectional concept set.

---

### 2.2 B6 — Intensive vs Extensive (Abstraction)

**Hypothesis**: surface quantity → abstract category {intensive, extensive} → output prediction. This is the same Type 3 (Abstraction) structure as the multilingual antonym task, but in a different domain. If the same mid-layer abstraction features fire for both, that is strong evidence for cross-domain abstract circuits.

**Answer tokens**: Need audit. Options:
- ` yes`/` no` — single-token guaranteed ✓ (recommended)
- ` intensive`/` extensive` — likely multi-token ⚠️

**Recommendation**: Use yes/no format for B6 to guarantee single-token answers.

```jsonl
{"prompt": "If you combine two identical samples of water at the same temperature, does the temperature of the combined sample change?", "correct_answer": " No", "incorrect_answer": " Yes", "behaviour": "physics_intensive_extensive", "behaviour_type": "abstraction", "physics_concept": "intensive_property", "template_idx": 0, "concept_index": 0, "difficulty": "standard", "quantity": "temperature", "property_type": "intensive"}
{"prompt": "Doubling the amount of a gas at constant pressure — does its temperature change?", "correct_answer": " No", "incorrect_answer": " Yes", "behaviour": "physics_intensive_extensive", "behaviour_type": "abstraction", "physics_concept": "intensive_property", "template_idx": 1, "concept_index": 0, "difficulty": "standard", "quantity": "temperature", "property_type": "intensive"}
{"prompt": "Two identical iron blocks at 300K are placed in thermal contact and allowed to equilibrate. Is the final temperature different from 300K?", "correct_answer": " No", "incorrect_answer": " Yes", "behaviour": "physics_intensive_extensive", "behaviour_type": "abstraction", "physics_concept": "intensive_property", "template_idx": 2, "concept_index": 0, "difficulty": "indirect", "quantity": "temperature", "property_type": "intensive"}
{"prompt": "You merge two water samples of equal volume and equal temperature. Does the resulting temperature differ from the original?", "correct_answer": " No", "incorrect_answer": " Yes", "behaviour": "physics_intensive_extensive", "behaviour_type": "abstraction", "physics_concept": "intensive_property", "template_idx": 3, "concept_index": 0, "difficulty": "standard", "quantity": "temperature", "property_type": "intensive"}
{"prompt": "Temperature is an intensive property of matter. If you combine two samples at the same temperature, is the resulting temperature higher than the original?", "correct_answer": " No", "incorrect_answer": " Yes", "behaviour": "physics_intensive_extensive", "behaviour_type": "abstraction", "physics_concept": "intensive_property", "template_idx": 4, "concept_index": 0, "difficulty": "explicit_label", "quantity": "temperature", "property_type": "intensive"}
```

**Note on concept_index=0 for all 5**: same quantity (temperature) across all 5. In Phase 1, add concept_index=1 for an extensive property (e.g., mass: "does combining two samples change total mass?" → correct: " Yes") to create the contrastive set that tests whether the model's circuit discriminates intensive vs extensive.

**Critical check for Phase 0**: Run the tokenization audit on ` No` and ` Yes` first. These should be single-token for Qwen3's tokenizer (likely token IDs for " No" and " Yes" with leading space). Verify before running the pipeline.

---

### 2.3 Full B1–B8 Prompt Inventory (Phase 1 reference)

For the plan: here is the complete design for all 8 behaviors including answer-token strategy and concept axes.

| Behavior | Type | Answer tokens | Concept axis (concept_index) | Numeric variants? |
|---|---|---|---|---|
| B1 — Decay type | Latent-state | ` alpha` / ` beta` | 0=α-decay, 1=β-decay | Z, A values |
| B2 — Decay chain | Latent-state | Element symbol (e.g., ` Pb`) | 0=(U-238 standard), 1=(Th-232) | # of α/β decays |
| B3 — Selection rule | Candidate-set | ` Yes` / ` No` | 0=forbidden (s→s), 1=allowed (s→p) | different orbital pairs |
| B4 — Approx regime | Candidate-set | ` classical` / ` relativistic` | 0=classical (v≪c), 1=relativistic (v≈c) | v/c ratio values |
| B5 — Gauge equiv | Abstraction | ` Yes` / ` No` | 0=gauge equiv (same B), 1=non-equiv | different gauge transforms |
| B6 — Intensive/ext | Abstraction | ` No` / ` Yes` | 0=intensive (temp), 1=extensive (mass) | different quantities |
| B7 — Entropy gate | Gating | ` No` / ` Yes` | 0=forbidden (ΔS<0), 1=allowed | different processes |
| B8 — Compression work | Gating | ` on` / ` by` | 0=compression (W on), 1=expansion (W by) | different ΔV scenarios |

**B2 warning**: "Pb" may tokenize as ` Pb` or `Pb` or `P` + `b` depending on context. Run tokenization audit before committing. If multi-token, redesign as "The final atomic number is:" → " 82" / " 92" (numbers are reliably single-token for 2-digit values).

**B4 warning**: ` classical` and ` relativistic` — high risk of multi-token. Alternative: " Yes" / " No" format ("Should classical mechanics be used to calculate the kinetic energy?").

**Recommendation for all behaviors**: default to Yes/No answer format unless the specific answer token (` alpha`, ` beta`, ` on`, ` by`) can be confirmed single-token by tokenization audit. The yes/no format trades some naturalness for guaranteed pipeline compatibility.

---

## 3. Implementation Plan Per Feature

### T1-A: Physics Behavior Suite

**Files that change:**
- `scripts/01_generate_prompts.py` — add 8 new generator functions + register in `GENERATORS`
- `configs/experiment_config.yaml` — add 8 new behavior entries with train_size/test_size/thresholds
- `scripts/prepare_b1_dashboard.py` — parameterise `BEHAVIOUR` (currently hardcoded)
- `scripts/prepare_physics_dashboard.py` — exists for physics_scalar_vector_operator, can be copied

**New files:**
- `data/prompts/physics_decay_type_{train,test}.jsonl` (generated)
- `jobs/physics_decay_type_full_pipeline.sbatch` (SLURM)
- × 8 behaviors

**Data flow:**
```
01_generate_prompts.py --behaviour physics_decay_type
  → data/prompts/physics_decay_type_train.jsonl (40 prompts)
  → data/prompts/physics_decay_type_test.jsonl (10 prompts)

02_run_baseline.py --behaviour physics_decay_type
  → gate check (EN accuracy ≥ 0.90)

04_extract_transcoder_features.py --behaviour physics_decay_type
  → data/results/transcoder_features/ (layers 10–25)

06_build_attribution_graph.py --behaviour physics_decay_type --graph_node_mode role_aware
  → data/results/attribution_graphs/physics_decay_type/

07_run_interventions.py --behaviour physics_decay_type
  → data/results/interventions/physics_decay_type/

08_causal_edges.py --behaviour physics_decay_type
  → data/results/causal_edges/physics_decay_type/circuits_*.json

scripts/prepare_physics_dashboard.py --behaviour physics_decay_type
  → dashboard_physics_decay_type/public/data/
```

**`behaviour_type` threading**: The `behaviour_type` field in the JSONL is carried through as a pass-through metadata field in scripts 04/06/07/08/09 (none of them filter on it). The `prepare_*_dashboard.py` script reads `run_manifest.json` and writes `behaviour_type` there if the input JSONL contains it. The dashboard reads it from `run_manifest.json`.

**Complexity**: Large. Estimated 2–3 days for all 8 generators + tokenization audits + pipeline validation. Phase 0 (B1 + B6 only, 10 prompts) is ~4 hours.

**Key design decision needed before starting:**
- One shared dashboard codebase with a behavior selector (complex, long-term), or separate per-behavior dashboard directories (simpler, works now)?
- **Recommendation**: separate directories for Phase 1 (each behavior gets its own `dashboard_physics_decay_type/`), share a common data layer in Phase 2 if needed.

---

### T1-B: Behavior Taxonomy Tags

**Files that change:**

*Backend (1 file):*
- `scripts/prepare_b1_dashboard.py` (and equivalent per-behavior prepare scripts) — add `behaviour_type` to `run_manifest.json` output

*Frontend (5 files):*
- `src/utils/colors.js` — add `BEHAVIOUR_TYPE_COLORS` map
- `src/App.jsx` — add type tag to Causal Circuit box
- `src/components/tabs/CommunityCards.jsx` — add pill badge to each card
- `src/data/loader.js` — pass `behaviour_type` from `run_manifest` into indexes
- `src/data/indexes.js` — expose `behaviourType` in computed indexes

**New component:**
- `src/components/shared/BehaviourTypePill.jsx` — reusable pill badge (12 lines)

**Data flow:**
```
run_manifest.json
  → { behaviour: "multilingual_circuits_b1", behaviour_type: "abstraction" }

loader.js
  → data.runManifest.behaviour_type

App.jsx
  → <BehaviourTypePill type={data.runManifest.behaviour_type} />
  → displayed in Causal Circuit box

CommunityCards.jsx
  → community.behaviour_type (from community_summary.json if present, else from manifest)
  → <BehaviourTypePill type={...} />
```

**`community_summary.json` community labels:** The current structure has no `label` or `behaviour_type` per community. Options:
1. Add `label` and `community_type` fields to `community_summary.json` (written by `prepare_*.py`)
2. Keep community labels as a frontend-only constant map (e.g., `COMMUNITY_LABELS[behaviourType][communityId]`)

**Recommendation**: Option 2 for Phase 1 — a constant in `CommunityCards.jsx` or a config JSON file at `dashboard_b1/public/data/community_labels.json`. This separates the research-authored narrative from the auto-computed metrics, which is the right architecture. The prepare script produces metrics; the researcher adds narrative labels manually.

**Color coding:**
```js
// src/utils/colors.js
export const BEHAVIOUR_TYPE_COLORS = {
  latent_state:   '#2dd4bf',  // teal
  candidate_set:  '#f59e0b',  // amber
  abstraction:    '#a78bfa',  // purple
  gating:         '#fb7185',  // coral
};

export const BEHAVIOUR_TYPE_LABELS = {
  latent_state:   'Latent-state',
  candidate_set:  'Candidate-set',
  abstraction:    'Abstraction',
  gating:         'Gating',
};
```

**Complexity**: Small. ~3 hours total. Can be done immediately on the existing multilingual dashboard.

---

### T1-C: Circuit Generalization Test Panel

**What it computes:**

For each set of prompts sharing the same `concept_index` (same underlying computation, different surface forms), compute:
- For each circuit feature: `support_fraction` = (number of templates where feature is active) / (total templates)
- `CSI` = |features with support_fraction = 1.0| / |all features in union|
- Per-feature stability = support_fraction × mean |contribution| where active

**Files that change:**

*No backend changes needed for initial implementation using existing data.* The `prompt_traces.json` already contains `template_idx`, `concept_index`, `feature_contributions` per prompt. The generalization computation is pure frontend.

*Frontend (3 files):*
- `src/components/tabs/GeneralizationPanel.jsx` — new component (~150 lines)
- `src/components/tabs/TabContainer.jsx` — add new tab entry
- `src/utils/filterData.js` — add `computeStabilityIndex()` utility

**Computation in `filterData.js`:**
```js
export function computeStabilityIndex(promptTraces, conceptIndex, contribThreshold = 0.05) {
  // 1. Filter traces to this concept
  const traces = promptTraces.filter(t => t.concept_index === conceptIndex);
  const templates = [...new Set(traces.map(t => t.template_idx))];

  // 2. For each template, collect active features (|contrib| > threshold)
  const templateFeatureSets = templates.map(tmpl => {
    const trace = traces.find(t => t.template_idx === tmpl);
    return new Set(
      (trace?.feature_contributions || [])
        .filter(f => Math.abs(f.contribution_to_correct) > contribThreshold)
        .map(f => f.feature_id)
    );
  });

  // 3. Intersection and union
  const union = new Set(templateFeatureSets.flatMap(s => [...s]));
  const intersection = [...union].filter(f => templateFeatureSets.every(s => s.has(f)));
  const csi = union.size > 0 ? intersection.size / union.size : 0;

  // 4. Per-feature stability
  const stability = {};
  for (const featureId of union) {
    const activeFracs = templateFeatureSets.filter(s => s.has(featureId)).length / templates.length;
    const contribs = traces
      .filter(t => (t.feature_contributions||[]).some(f => f.feature_id === featureId))
      .map(t => Math.abs(t.feature_contributions.find(f => f.feature_id === featureId)?.contribution_to_correct || 0));
    const meanContrib = contribs.length > 0 ? contribs.reduce((a,b) => a+b, 0)/contribs.length : 0;
    stability[featureId] = { support_fraction: activeFracs, mean_contrib: meanContrib,
                              stability_score: activeFracs * meanContrib,
                              category: activeFracs === 1.0 ? 'shared' : activeFracs > 0.5 ? 'variable' : 'unique' };
  }

  return { csi, n_shared: intersection.length, n_union: union.size,
           n_templates: templates.length, stability, templates };
}
```

**UI layout for `GeneralizationPanel.jsx`:**
- Top: concept selector (dropdown filtered to available concept_index values)
- Metric bar: CSI = X.XX | Shared = N | Variable = M | Unique = K | Templates = T
- Matrix: rows = templates (T0, T1, ...), columns = circuit features (sorted: shared first → variable → unique)
  - Cell color: shared=green, variable=amber, unique=red (opacity = stability_score)
  - Cell value: |contribution| as a small number
- Bottom: per-feature stability bar chart (horizontal, sorted by stability_score)

**S2 Hacking detection**: Flag features where support_fraction < 0.3 AND they only appear on the "hardest" templates (highest template_idx). Add a `potential_backup` boolean to the stability output.

**Complexity**: Medium. ~1 day. Can be validated immediately on existing multilingual data (6 EN templates, concept 0 = small/large).

**Important constraint**: this panel works correctly only when `prompt_traces.json` contains `feature_contributions` per prompt. Looking at the actual data, it does (verified above — `top_correct_features` and `top_incorrect_features` are present). However, the full `feature_contributions` array (all features, not just top-3) may need to come from `prompt_features.csv` instead. The plan: use `prompt_features.csv` which has all features per prompt with their effect sizes.

---

### T1-D: Auto NL Explanation Per Circuit

**Files that change:**

*No backend changes needed.* All required data is in existing loaded files.

*Frontend (3 files):*
- `src/utils/generateExplanation.js` — new utility, pure function (~80 lines)
- `src/components/prompt/PromptInspector.jsx` — add collapsible explanation section
- `src/components/shared/ExplanationBlock.jsx` — new component (copyable text box, ~30 lines)

**Template implementation in `generateExplanation.js`:**

```js
export function generateCircuitExplanation(trace, circuit, communityData, behaviourType) {
  if (!trace || !circuit) return null;

  const margin = trace.baseline_logit_diff;
  const correct = trace.correct_answer?.trim();
  const incorrect = trace.incorrect_answer?.trim();

  // 1. Dominant zone
  const zones = trace.zone_summary || {};
  const dominantZone = Object.entries(zones)
    .filter(([,z]) => z.measured_contribution)
    .sort(([,a],[,b]) => Math.abs(b.measured_contribution) - Math.abs(a.measured_contribution))[0];
  const zoneName = dominantZone?.[0] || 'mid';
  const zoneContrib = dominantZone?.[1]?.measured_contribution?.toFixed(3) || '?';
  const zoneNFeats = dominantZone?.[1]?.n_features || '?';

  // 2. Top supporting feature
  const topFeat = trace.top_correct_features?.[0];
  const topFeatId = topFeat?.feature_id || '?';
  const topFeatLayer = topFeat?.layer || '?';
  const topFeatContrib = topFeat?.contribution_to_correct?.toFixed(3) || '?';

  // 3. Top competing feature
  const topCompeting = trace.top_incorrect_features?.[0];
  const compFeatId = topCompeting?.feature_id || null;
  const compFeatContrib = topCompeting?.contribution_to_correct?.toFixed(3) || '?';

  // 4. Necessity from circuit validation
  const necessity = circuit.validation?.disruption_rate;
  const necessityPct = necessity != null ? `${(necessity * 100).toFixed(1)}%` : '?';

  // 5. Type label
  const typeLabels = {
    abstraction: 'an abstract representation',
    latent_state: 'a latent intermediate state',
    candidate_set: 'a candidate selection process',
    gating: 'a gating decision',
  };
  const typeDesc = typeLabels[behaviourType] || 'an internal computation';

  let text = `For the prompt "${trace.prompt}", the model predicts "${correct}" ` +
    `over "${incorrect}" with a logit margin of ${margin?.toFixed(2) ?? '?'}.\n\n`;

  text += `The circuit operates primarily through ${typeDesc}. ` +
    `The dominant contribution comes from ${zoneName}-layer features (${zoneNFeats} features, ` +
    `Δlogit = ${zoneContrib}).\n\n`;

  text += `The key feature is [${topFeatId}] at layer ${topFeatLayer}, which contributes ` +
    `${topFeatContrib > 0 ? '+' : ''}${topFeatContrib} toward the correct answer.`;

  if (compFeatId) {
    text += ` A competing pathway through [${compFeatId}] simultaneously pushes ` +
      `toward "${incorrect}" (contribution = ${compFeatContrib}).`;
  }

  text += `\n\nCircuit-level ablation disrupts the model's answer on ${necessityPct} of prompts, ` +
    `confirming that these features are causally responsible (not merely correlated).`;

  // 6. Top path if available
  const topPath = trace.top_paths?.[0];
  if (topPath) {
    text += `\n\nThe dominant computational path is: ${topPath.path_str}`;
  }

  return text;
}
```

**UI integration in `PromptInspector.jsx`:**
- Add a collapsible `<details>` element at the bottom of the prompt inspector
- Summary line: "Circuit explanation (auto-generated)"
- Content: rendered paragraph text + copy-to-clipboard button
- The copy action pastes plain text (no markdown formatting)

**Constraint enforcement**: The explanation generator is a pure function from existing loaded data. It cannot and does not call any LLM or external service. The only "intelligence" is template-filling from pre-computed metrics. This is enforced architecturally: `generateExplanation.js` has no async operations, no fetch calls, no imports except utility formatters.

**Complexity**: Small. ~2 hours. No backend work. Can be done as the final step since it's fully independent.

---

## 4. Risks and Open Questions

### Open Question 1 — Single-behavior vs multi-behavior dashboard

**Question**: Should each physics behavior get its own separate dashboard directory (like `dashboard_physics/` and `dashboard_physics_decay_type/`), or should we build a unified dashboard that loads multiple behaviors?

**Impact**: High. Affects the architecture of T1-A, T1-B, and T1-C.

**Recommendation**: Separate dashboards for Phase 1. They share the same codebase (copy of `dashboard_b1/` with behaviour-specific `prepare_*.py` script). A unified multi-behavior view can be added in Phase 2 by adding a behavior-selector dropdown to the sidebar — the data model change is one new field in the loader.

**Decide before**: starting T1-A implementation.

---

### Open Question 2 — Community label authorship

**Question**: Who assigns the community label (e.g., "Early cross-lingual pathway") and the community behaviour_type? The analyst (manually, in a config file) or the pipeline (automatically)?

**Impact**: Medium. Affects T1-B implementation.

**Recommendation**: Manual authorship in a separate `community_labels.json` file per behaviour run, stored in `dashboard_{name}/public/data/`. The prepare script generates metrics only; the researcher edits the labels file. This is scientifically honest — the label is an interpretation, not a measurement.

**Format:**
```json
{
  "0": {"label": "Early cross-lingual", "description": "L10–L15 features, 97% balanced", "behaviour_type": "abstraction"},
  "1": {"label": "Semantic hub", "description": "L20–L23, L22_F41906 hub", "behaviour_type": "abstraction"},
  "2": {"label": "FR competitor", "description": "L22, 100% fr_leaning, inhibitory", "behaviour_type": null}
}
```

The `behaviour_type: null` for community C2 is intentional — not every community fits the taxonomy cleanly (C2 is an artifact/competitor, not a computation type).

**Decide before**: implementing T1-B community badges.

---

### Open Question 3 — CSI threshold and its effect on the narrative

**Question**: What contribution threshold should define a "feature is active" for the CSI computation? Too low (threshold → 0) and every feature appears everywhere (CSI → 1.0 artificially). Too high and the circuit looks fragile.

**Impact**: Medium. Affects T1-C narrative.

**Recommendation**: Default `contribThreshold = 0.05` (approximately 1% of the max observed contribution). Show the threshold as a user-adjustable slider in the Generalization panel so researchers can explore sensitivity.

**Decide before**: finalising the CSI formula.

---

### Open Question 4 — Path validation results in the report / dashboard

**Question**: The path validation results from script 14 (SLURM 27224134) are not yet surfaced in the dashboard. Should they be added as a new display in T1-D explanations or T1-C generalization?

**Impact**: Low for the plan. High for the final paper.

**Recommendation**: Add path validation results to the circuit explanation template (T1-D). Add a new field to `circuit.json` or `run_manifest.json` for `path_validation_summary`. This requires running `prepare_b1_dashboard.py` to copy path_validation results into the dashboard data directory.

**Decide before**: starting T1-D implementation.

---

### Open Question 5 — Do we need a tokenization audit script before Phase 0?

**Question**: The existing `scripts/b_tokenize_audit_multilingual.py` audits multilingual answer tokens. We need the same for physics answer tokens.

**Recommendation**: Yes. Write a 30-line audit script as Phase 0 step 0 — before writing any prompts or running any pipeline. Pass every planned `correct_answer` and `incorrect_answer` token through `tokenizer.encode(token, add_special_tokens=False)` and assert `len(ids) == 1`. This takes 1 hour and prevents failures at step 04.

---

## 5. Implementation Order

```
Day 1 (half-day) — Phase 0 prep
  ├─ Tokenization audit for B1 and B6 answer tokens
  ├─ Write 10 prompts (5×B1, 5×B6) to JSONL by hand
  └─ Run scripts 01–08 on B1+B6 mini-set; check attribution graph

Day 1–2 — T1-B (Taxonomy Tags)
  ├─ Add BehaviourTypePill component (colors.js + new component)
  ├─ Add to App.jsx Causal Circuit box
  ├─ Add to CommunityCards.jsx
  └─ Create community_labels.json for multilingual_circuits_b1
      (uses existing dashboard, no new data needed)

Day 2–3 — T1-A Phase 1 (all 8 behaviors, 5 prompts each)
  ├─ Write 8 generators in script 01 (or write JSONL by hand for speed)
  ├─ Add 8 behaviors to experiment_config.yaml
  ├─ Run full pipeline for all 8 behaviors on CSD3
  └─ Identify which 3–4 behaviors produce clean circuits → expand those to 25+ prompts

Day 3 — T1-D (Auto NL Explanation)
  ├─ generateExplanation.js utility
  ├─ Collapsible section in PromptInspector.jsx
  └─ Test on existing multilingual data (all data available)

Day 4 — T1-C (Circuit Generalization)
  ├─ computeStabilityIndex() in filterData.js
  ├─ GeneralizationPanel.jsx
  ├─ Add tab to TabContainer.jsx
  └─ Validate on existing 6-template EN multilingual data

Day 4–5 — T1-A Phase 2 (expand clean behaviors to 25–30 prompts)
  ├─ Add template variants and concept contrastors
  └─ Re-run pipeline; update dashboard for each clean behavior
```

---

## 6. Summary Table

| Feature | Files changed (backend) | Files changed (frontend) | New files | Complexity | Blocker |
|---|---|---|---|---|---|
| T1-A Phase 0 | script 01, config | — | 10 JSONL prompts | 4h | Tokenization audit |
| T1-A Phase 1 | script 01 ×8, config ×8 | — | 8 generators, 8 JSONL pairs | 2–3 days | Phase 0 GO |
| T1-B Tags | prepare_*.py (1 line) | App.jsx, CommunityCards.jsx, colors.js, loader.js, indexes.js | BehaviourTypePill.jsx, community_labels.json | 3h | None |
| T1-C Generalization | None | TabContainer.jsx, filterData.js | GeneralizationPanel.jsx | 1 day | prompt_features.csv shape check |
| T1-D Explanation | None | PromptInspector.jsx | generateExplanation.js, ExplanationBlock.jsx | 2h | None |
