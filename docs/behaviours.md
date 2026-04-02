# Behaviour Catalogue

This document describes the behaviours implemented for mechanistic interpretability
analysis, following the 4-type framework from `behavior_analysis.pdf`.

## Framework: 4 Types of Behaviour

| Type | Name | Structure | Example |
|------|------|-----------|---------|
| 1 | **Latent states** | Input -> latent state -> latent state -> output | Dallas -> Texas -> Austin |
| 2 | **Candidate set** | Input -> candidate set -> constraint/filter -> output | Selection rules, regime choice |
| 3 | **Abstraction** | Surface input -> abstract meaning -> surface output | Multilingual, gauge invariance |
| 4 | **Gating** | Input -> internal classifier -> action | Refusal, thermodynamic constraints |

Each behaviour is tested via three properties:
- **Causal directionality** (ablation): intervening on internal variable Z changes output y
- **Persistency** (surface variation): same mechanism under rephrased inputs
- **Substitutability** (semantic swap): replacing Z with Z' produces predictable output change

---

## 1. `grammar_agreement` (Type 2: Candidate set)

**Task**: Subject-verb number agreement. Given "The dogs", predict " sleep" (not " sleeps").

**Latent variable**: Grammatical number (singular/plural).

**Mapping to framework**:
- *Candidate set*: verb forms {singular, plural} are candidates
- *Constraint*: grammatical number of the subject selects the correct verb form
- Structure: `subject -> {number: sing/plur} -> verb_form`

**Answer tokens**: ` is`/` are`, ` was`/` were`, ` sleep`/` sleeps`, etc.

**Pairing for patching**: singular targets <- plural sources, and vice-versa.

**Effect sign interpretation**:
- Positive effect_size = intervention increased margin (model became more correct)
- Negative effect_size = intervention decreased margin (model became less correct)
- For ablation of positive-beta features: expect negative effect (margin decreases)
- Sign flip = intervention changed which answer the model prefers

---

## 2. `physics_scalar_vector_operator` (Type 3: Abstraction)

**Task**: Classify a physical operator or quantity as scalar-valued or vector-valued.

**Prompt structure**:
```
"The divergence of a vector field produces a"  -> " scalar"
"The gradient of a scalar field produces a"    -> " vector"
"The dot product of two vectors is a"          -> " scalar"
"The cross product of two vectors is a"        -> " vector"
```

**Answer tokens**: ` scalar` / ` vector` (single-token, balanced classes).

**Latent variable**: Abstract category Z in {scalar, vector}.

**Mapping to framework** (Type 3: Abstraction):
- *Surface form*: diverse physics language (divergence, curl, gradient, dot product,
  electric potential, velocity, etc.)
- *Abstract meaning*: scalar vs vector classification
- *Surface output*: the token " scalar" or " vector"
- Structure: `operator_description -> z_category in {scalar, vector} -> answer`

This mirrors the "gauge equivalence" behaviour from `behavior_analysis.pdf`:
- Surface form (specific operator names, notation variants) maps to an invariant
  abstract category (scalar vs vector)
- The abstract category determines the output
- Surface variations (context sentences, notation, field names) should not change
  the underlying mechanism

**Physics concepts covered**:

| Concept | Class | Examples |
|---------|-------|----------|
| divergence | scalar | div of vector field |
| Laplacian (of scalar) | scalar | Laplacian of temperature field |
| dot product | scalar | inner product of two vectors |
| electric potential | scalar | voltage, scalar potential |
| energy quantities | scalar | kinetic energy, work, power |
| thermodynamic scalars | scalar | temperature, pressure, mass |
| curl | vector | curl of vector field |
| gradient | vector | gradient of scalar field |
| cross product | vector | vector product of two vectors |
| field vectors | vector | electric field, magnetic field |
| mechanics vectors | vector | velocity, momentum, force |
| vector potential | vector | magnetic vector potential |

**Surface-form variants** (persistency testing):
- Context prefixes: "In classical electromagnetism, ...", "Consider a smooth field...", etc.
- Name variants: "divergence"/"div", "gradient"/"grad", "curl"/"rot"
- Field descriptions: "vector field", "velocity field", "force field", etc.
- Template variations: 4 different phrasings per concept

**Semantic swaps** (substitutability testing):
- Swapping scalar<->vector class flips the correct answer
- For patching: scalar targets paired with vector sources, and vice-versa
- Swapping "divergence" features with "curl" features should flip scalar->vector

**Effect sign interpretation**:
- Positive effect_size = intervention moved margin toward correct answer
- Negative effect_size = intervention moved margin away from correct answer
- For ablation: expect negative effect on features that drive the correct classification
- Sign flip = model switches from "scalar" to "vector" (or vice-versa) after intervention
- Key hypothesis: features encoding the abstract scalar/vector distinction should show
  large |effect_size| under ablation and consistent sign flips under cross-class patching

**Extension potential**:
- This behaviour could extend to Type 1 (latent states) by chaining operations:
  "The gradient of the divergence of a vector field is a ___"
- Could extend to Type 4 (gating) by asking whether operations are valid:
  "Can you take the curl of a scalar field?" -> gate decision
- The abstract scalar/vector features may transfer to other physics tasks,
  testing the "transferrable abstract features" hypothesis from Type 3

---

## Adding New Behaviours

To add a new behaviour:

1. **`scripts/01_generate_prompts.py`**: Add a `generate_{name}_prompts()` function
   and register it in the `GENERATORS` dict.

2. **`configs/experiment_config.yaml`**: Add entry under `behaviours:` with
   `train_size`, `test_size`, `min_logit_diff`, `success_threshold`.

3. **`scripts/02_run_baseline.py`**: Add to `--behaviour` choices.

4. **`scripts/04_extract_transcoder_features.py`**: Add to `--behaviour` choices.

5. **`scripts/06_build_attribution_graph.py`**: Add to `--behaviour` choices.

6. **`scripts/07_run_interventions.py`**: Add to `--behaviour` choices.
   If the behaviour has a non-generic class structure (like singular/plural or
   scalar/vector), add pairing logic in `create_prompt_pairs()`.

7. **`scripts/09_prepare_offline_ui.py`**: No changes needed (accepts any string).

8. **`jobs/`**: Create an sbatch file following `physics_op_02_09.sbatch` pattern.
