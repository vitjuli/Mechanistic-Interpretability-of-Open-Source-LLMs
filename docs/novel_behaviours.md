# Novel Behaviours for Mechanistic Interpretability Analysis

## Overview

This document defines 20 novel behaviours suitable for circuit-level analysis using pre-trained transcoders. Each behaviour has a clear latent variable, clean evaluation signal, systematic prompt generation, and concrete intervention plans.

**Distribution:**
- Type A (Latent-state propagation): 6 behaviours
- Type B (Candidate set selection): 5 behaviours
- Type C (Abstraction vs surface form): 5 behaviours
- Type D (Gating / policy circuits): 4 behaviours
- Physics/science-oriented: 14 behaviours
- Internal checking behaviours: 6 behaviours

---

## Behaviour 1: Dimensional Consistency Verification

**Type:** D (Gating / policy circuits)

**Core latent variable(s):**
- Dimensional signature vector [M^a, L^b, T^c, I^d, ...] for each term
- Binary consistency flag (match/mismatch)

**Behavioural hypothesis:**
The model internally computes dimensional signatures for each side of an equation and compares them via a gating circuit. When signatures mismatch, a "violation detector" feature activates and gates the output toward "inconsistent."

**Prompt template:**
```
Equation: F = m * v^2 / r
Dimensionally consistent? Answer:
```
Target: ` Yes` (this is centripetal force, dimensionally correct)

**Prompt-family generator recipe:**
1. **Equation type:** kinematic, electromagnetic, thermodynamic, quantum (10+ categories)
2. **Consistency status:** 50% valid, 50% invalid (controlled)
3. **Error type (for invalid):** missing power, wrong variable, extra factor, unit confusion
4. **Variable naming:** standard (F, m, v) vs non-standard (X, Y, Z)
5. **Complexity:** 2-term, 3-term, 4-term equations
6. **Notation:** symbolic (F=ma) vs verbal ("force equals mass times acceleration")

**Target / score:**
- Single token: ` Yes` or ` No`
- Score: accuracy on balanced valid/invalid set

**Negative controls:**
1. Equations with only numerical constants (no dimensions to check): "3 = 2 + 1" → should not activate dimensional checking
2. Purely symbolic expressions with undefined quantities: "x = y * z" → should not confidently answer without dimensional context

**Intervention plan:**
1. **Ablate dimensional-signature features:** Ablate features in middle layers that encode [M, L, T] signatures. Prediction: model loses ability to distinguish valid from invalid; accuracy drops to ~50%.
2. **Swap dimensional features between valid/invalid:** Patch features from a dimensionally-valid equation into a dimensionally-invalid one. Prediction: model flips answer from No to Yes.

**Why it is novel:**
Unlike standard unit conversion tasks, this requires computing and comparing implicit dimensional signatures without explicit unit labels—a meta-level consistency check rather than a direct calculation.

**Risks / confounds:**
The model may use surface-level heuristics (e.g., "v^2/r looks like centripetal" pattern matching) instead of genuine dimensional analysis. Control by using unfamiliar variable names.

---

## Behaviour 2: Reference Frame Velocity Composition

**Type:** A (Latent-state propagation)

**Core latent variable(s):**
- Velocity in intermediate reference frame (v_AB)
- Cumulative velocity in final frame (v_AC)

**Behavioural hypothesis:**
The model propagates velocity through a chain of reference frames by computing v_AC = v_AB + v_BC (Galilean) or relativistic composition. The intermediate velocity v_AB is a latent state that must be computed and then combined with v_BC.

**Prompt template:**
```
Train A moves at 30 m/s relative to ground. Person B walks at 2 m/s relative to train A in the same direction. Ball C moves at 5 m/s relative to B in the same direction. Speed of C relative to ground:
```
Target: ` 37` (30 + 2 + 5)

**Prompt-family generator recipe:**
1. **Number of frames:** 2, 3, or 4 intermediate frames
2. **Direction:** same direction, opposite direction, mixed
3. **Velocity magnitudes:** small integers (1-50 m/s) to avoid tokenization issues
4. **Context:** train/person, car/passenger, boat/river, spaceship (non-relativistic)
5. **Question format:** "speed of X relative to Y" with varying X, Y
6. **Units:** m/s, km/h (requires unit handling as secondary latent)

**Target / score:**
- Numerical answer (1-3 tokens)
- Score: exact match or |predicted - correct| < 1

**Negative controls:**
1. Single frame (no propagation needed): "Car moves at 30 m/s relative to ground. Speed of car relative to ground:" → trivial, should not require intermediate computation
2. Unrelated quantities: "Train A is 100m long. Person B weighs 70kg. Length of B:" → should recognize non-composable quantities

**Intervention plan:**
1. **Ablate intermediate velocity features:** Target features encoding v_AB in middle layers. Prediction: model fails to correctly propagate; answers with single velocity or random composition.
2. **Patch intermediate velocity:** Replace v_AB features with those from a different problem where v_AB = 10 instead of 32. Prediction: final answer shifts by the difference.

**Why it is novel:**
Unlike simple arithmetic, this requires tracking a physically meaningful intermediate state (relative velocity) through a causal chain. The latent variable has physical semantics, not just numerical.

**Risks / confounds:**
Model might memorize common velocity combinations. Mitigate by using unusual magnitudes and varying frame orderings.

---

## Behaviour 3: Oxidation State Tracking Through Reaction Chain

**Type:** A (Latent-state propagation)

**Core latent variable(s):**
- Oxidation state of tracked atom at each reaction step
- Final oxidation state

**Behavioural hypothesis:**
The model tracks the oxidation state of a specific atom (e.g., Fe) through a sequence of redox reactions. Each reaction modifies the oxidation state, which must be propagated as a latent variable.

**Prompt template:**
```
Fe starts at oxidation state 0. Reaction 1: Fe loses 2 electrons. Reaction 2: Fe loses 1 electron. Final oxidation state of Fe:
```
Target: ` +3`

**Prompt-family generator recipe:**
1. **Element:** Fe, Cu, Mn, S, N, Cr (elements with multiple oxidation states)
2. **Number of steps:** 2, 3, or 4 reactions
3. **Reaction type:** "loses N electrons", "gains N electrons", "oxidized by X", "reduced by Y"
4. **Starting state:** 0, +2, -1, etc.
5. **Magnitude of change:** 1, 2, or 3 electrons per step
6. **Direction consistency:** all oxidation, all reduction, or mixed

**Target / score:**
- Oxidation state: single token like ` +3`, ` -2`, ` 0`
- Score: exact match

**Negative controls:**
1. No redox (acid-base reaction): "Fe at +3 reacts with OH-. Oxidation state of Fe:" → should remain +3
2. Spectator species: "Fe at +2. Cu loses 2 electrons. Oxidation state of Fe:" → Fe unchanged, tests if model correctly ignores irrelevant changes

**Intervention plan:**
1. **Ablate oxidation-state features:** Remove features encoding intermediate oxidation state after Reaction 1. Prediction: model cannot correctly accumulate changes; defaults to starting state or last change only.
2. **Swap oxidation trajectory:** Patch features from "gains electrons" into "loses electrons" context. Prediction: sign of final state flips.

**Why it is novel:**
Unlike simple arithmetic, this requires domain-specific knowledge that electron loss = oxidation = positive state change. Tests whether model maintains chemically-meaningful latent state.

**Risks / confounds:**
Model may learn surface patterns ("loses" → add, "gains" → subtract) without understanding chemistry. Control by using varied phrasings ("oxidized to", "reduced from").

---

## Behaviour 4: Symmetry-Allowed Quantum Transitions

**Type:** B (Candidate set selection)

**Core latent variable(s):**
- Symmetry labels of initial and final states (e.g., parity, angular momentum)
- Selection rule constraint (Δl = ±1, Δm = 0, ±1, etc.)
- Binary: transition allowed/forbidden

**Behavioural hypothesis:**
The model generates a candidate set of possible transitions, then filters by selection rules. Features encoding symmetry labels gate which transitions survive the filter.

**Prompt template:**
```
Hydrogen atom. Electric dipole transition. Initial state: 2s. Which final state is allowed? Options: 1s, 2p, 3s, 3d
Answer:
```
Target: ` 2p` (Δl = +1 allowed; s→s and s→d forbidden by Δl = ±1)

**Prompt-family generator recipe:**
1. **System:** hydrogen, helium, generic atom, diatomic molecule
2. **Transition type:** electric dipole (Δl=±1), magnetic dipole (Δl=0), electric quadrupole (Δl=0,±2)
3. **Initial state:** 1s, 2s, 2p, 3s, 3p, 3d
4. **Distractor states:** symmetry-forbidden options
5. **Number of options:** 3, 4, or 5
6. **Notation:** spectroscopic (2p) vs quantum numbers (n=2, l=1)

**Target / score:**
- Single option token: ` 2p`, ` 3p`, etc.
- Score: accuracy on selection-rule-valid choice

**Negative controls:**
1. All options allowed: "Initial: 2p. Options: 1s, 3s, 3d" → all satisfy Δl=±1 for p; should recognize ambiguity
2. No valid option: "Initial: 2s. Electric dipole. Options: 1s, 3s, 4s" → all forbidden; should indicate "none"

**Intervention plan:**
1. **Ablate selection rule features:** Remove features encoding Δl constraint. Prediction: model selects randomly among options or defaults to most common.
2. **Swap transition type:** Patch "electric quadrupole" features into "electric dipole" context. Prediction: different transition becomes allowed (e.g., s→d now valid).

**Why it is novel:**
Unlike factual recall of specific transitions, this requires applying abstract selection rules to novel initial/final state combinations—a constraint satisfaction problem with physics semantics.

**Risks / confounds:**
Model may memorize specific allowed transitions (2s→2p) rather than learning selection rules. Control by using less common states and varying transition types.

---

## Behaviour 5: Approximation Regime Selection

**Type:** B (Candidate set selection)

**Core latent variable(s):**
- Relevant dimensionless parameter (e.g., v/c, λ/L, kT/E)
- Regime label: classical/quantum, relativistic/non-relativistic, etc.

**Behavioural hypothesis:**
Given physical parameters, the model identifies which approximation regime applies and selects the appropriate simplified formula from candidates.

**Prompt template:**
```
Electron with mass 9.1e-31 kg, speed 1e4 m/s, c = 3e8 m/s.
Which formula for kinetic energy? Options: KE = (1/2)mv^2, KE = (γ-1)mc^2, KE = p^2/2m
Answer:
```
Target: ` KE = (1/2)mv^2` (v << c, so non-relativistic)

**Prompt-family generator recipe:**
1. **Regime boundary:** v/c ratio (0.001 to 0.9), λ/L ratio, kT/ℏω ratio
2. **Physical context:** mechanics, quantum, thermal, electromagnetic
3. **Parameter presentation:** explicit numbers, symbolic ratios, order-of-magnitude
4. **Number of candidate formulas:** 2, 3, or 4
5. **Boundary cases:** near the transition (v/c ~ 0.1) to test regime identification
6. **Units:** SI, CGS, natural units

**Target / score:**
- Selected formula (token sequence)
- Score: correct regime identification

**Negative controls:**
1. Exactly at boundary: v/c = 0.1 → both approximations have similar error; should indicate "either" or "exact needed"
2. Missing parameter: "Electron at high speed. Which formula?" → cannot determine regime without v/c

**Intervention plan:**
1. **Ablate regime-classifier features:** Remove features encoding v/c comparison. Prediction: model defaults to one formula regardless of parameters.
2. **Swap velocity features:** Replace v=1e4 features with v=2e8 features. Prediction: model switches to relativistic formula.

**Why it is novel:**
This tests whether the model performs genuine regime identification (comparing dimensionless ratios to thresholds) rather than pattern-matching on keywords like "electron" or "relativistic."

**Risks / confounds:**
Model may use keyword heuristics ("electron" → quantum, "speed of light" → relativistic). Control by using unconventional contexts.

---

## Behaviour 6: Unit Canonicalization

**Type:** C (Abstraction vs surface form)

**Core latent variable(s):**
- Canonical SI representation of quantity
- Conversion factor pathway

**Behavioural hypothesis:**
The model maps diverse surface unit representations (miles, feet, eV, calories) to a canonical abstract form (SI units), then maps back to the requested output format.

**Prompt template:**
```
Energy: 2.5 eV. Express in joules.
Answer:
```
Target: ` 4e-19` or ` 4.0e-19` (2.5 × 1.6e-19)

**Prompt-family generator recipe:**
1. **Quantity type:** energy, length, time, mass, temperature, pressure
2. **Input unit:** SI, CGS, imperial, natural units, domain-specific (eV, amu, Å)
3. **Output unit:** SI, CGS, or different domain-specific
4. **Magnitude:** varying orders of magnitude
5. **Precision:** integer, 1 decimal, scientific notation
6. **Context:** with/without physical context ("electron energy" vs just "energy")

**Target / score:**
- Numerical answer with unit (2-4 tokens)
- Score: |log10(predicted/correct)| < 0.1 (order of magnitude tolerance)

**Negative controls:**
1. Same units in/out: "5 meters in meters" → should return ` 5` without conversion
2. Incompatible units: "5 kg in meters" → should indicate error/impossible

**Intervention plan:**
1. **Ablate conversion-factor features:** Remove features encoding eV→J conversion. Prediction: model outputs wrong magnitude or copies input number.
2. **Swap unit-identity features:** Patch "eV" encoding with "keV" encoding. Prediction: answer shifts by factor of 1000.

**Why it is novel:**
Unlike simple "convert X to Y" drills, this tests whether the model maintains an abstract canonical representation. The latent is the SI-equivalent value, invariant under surface unit changes.

**Risks / confounds:**
Model may memorize specific conversions without abstract representation. Test with unusual unit combinations (e.g., erg to BTU).

---

## Behaviour 7: Equation Form Equivalence Detection

**Type:** C (Abstraction vs surface form)

**Core latent variable(s):**
- Algebraic normal form (canonical polynomial representation)
- Boolean equivalence flag

**Behavioural hypothesis:**
The model transforms two algebraically different surface forms to a common abstract representation, then compares them. Equivalent equations should activate "same" features regardless of surface differences.

**Prompt template:**
```
Equation A: PV = nRT
Equation B: P = nRT/V
Equivalent? Answer:
```
Target: ` Yes`

**Prompt-family generator recipe:**
1. **Transformation type:** rearrangement, factoring, expansion, substitution
2. **Equation domain:** ideal gas, kinematics, circuits, thermodynamics
3. **Complexity:** 2-variable, 3-variable, 4-variable
4. **Equivalence status:** 50% equivalent, 50% non-equivalent
5. **Non-equivalence type:** sign error, missing term, wrong power, different constant
6. **Notation:** symbolic, with numerical coefficients, mixed

**Target / score:**
- Single token: ` Yes` or ` No`
- Score: accuracy on balanced equivalent/non-equivalent set

**Negative controls:**
1. Trivially identical: "A: F=ma, B: F=ma" → surface match, not testing abstraction
2. Semantically different domain: "A: F=ma, B: PV=nRT" → not comparable, should indicate N/A

**Intervention plan:**
1. **Ablate canonical-form features:** Remove features encoding normalized algebraic structure. Prediction: model fails on non-obvious equivalences (e.g., (a+b)^2 vs a^2+2ab+b^2).
2. **Patch equivalence decision:** Swap features from an equivalent pair into a non-equivalent context. Prediction: model incorrectly says "Yes."

**Why it is novel:**
This tests genuine algebraic understanding rather than string matching. The model must compute an abstract representation that's invariant under allowed transformations.

**Risks / confounds:**
Model may use shallow heuristics (same variables = equivalent). Control with careful variable-matched non-equivalents.

---

## Behaviour 8: Conservation Law Violation Detection

**Type:** D (Gating / policy circuits) + Internal checking

**Core latent variable(s):**
- Conserved quantity totals (energy, momentum, charge) before and after
- Violation flag for each conservation law

**Behavioural hypothesis:**
The model computes totals of conserved quantities on each side of a process description and activates a "violation detector" if they don't match. This gating circuit blocks physically impossible scenarios.

**Prompt template:**
```
Process: A 2kg ball at rest is hit by a 1kg ball moving at 6 m/s. After collision, the 2kg ball moves at 4 m/s and the 1kg ball moves at 3 m/s. Both move in the original direction.
Conservation violated? Answer:
```
Target: ` No` (momentum: 1×6 = 2×4 + 1×3 → 6 = 11 ✗, but wait let me recalculate... Initial: 1×6 = 6. Final: 2×4 + 1×3 = 11. So ` Yes` momentum is violated)

Let me redo: Initial p = 1×6 = 6 kg⋅m/s. Final p = 2×4 + 1×3 = 11 kg⋅m/s. Violated!
Target: ` Yes`

**Prompt-family generator recipe:**
1. **Conservation law:** momentum, energy, charge, mass, angular momentum
2. **Violation status:** 50% valid, 50% violated
3. **Process type:** collision, decay, reaction, emission, absorption
4. **Number of objects:** 2, 3, or 4 bodies
5. **Violation magnitude:** small (10%) vs large (50%) discrepancy
6. **Presentation:** numerical values vs symbolic ratios

**Target / score:**
- Single token: ` Yes` (violated) or ` No` (conserved)
- Score: accuracy on balanced violated/conserved set

**Negative controls:**
1. Open system: "Ball falls under gravity. Momentum conserved?" → No, but due to external force, not violation
2. Non-physical statement: "Ball's color changes from red to blue. Conservation violated?" → N/A for color

**Intervention plan:**
1. **Ablate total-computation features:** Remove features encoding summed momentum. Prediction: model cannot detect violations; answers randomly.
2. **Swap violation-detector features:** Patch features from a violated scenario into a valid one. Prediction: model incorrectly flags valid process.

**Why it is novel:**
Unlike textbook conservation problems asking for final velocity, this requires internal consistency checking—computing both sides and comparing rather than solving for unknowns.

**Risks / confounds:**
Numerical precision issues in multi-step calculations. Use clean integer values.

---

## Behaviour 9: Limiting Case Extrapolation

**Type:** A (Latent-state propagation)

**Core latent variable(s):**
- Parameter value (approaching 0, ∞, or special point)
- Asymptotic behavior of expression

**Behavioural hypothesis:**
The model propagates a parameter toward a limit and determines the asymptotic behavior of a physical expression. The latent state is the dominant term as the parameter approaches the limit.

**Prompt template:**
```
Expression: E = mc^2 / sqrt(1 - v^2/c^2)
As v → 0, E approaches:
```
Target: ` mc^2` (rest mass energy)

**Prompt-family generator recipe:**
1. **Limit type:** x→0, x→∞, x→1, x→special value
2. **Expression domain:** relativistic, quantum, thermodynamic, classical
3. **Leading behavior:** constant, linear, quadratic, divergent, zero
4. **Complexity:** single-variable, multi-variable
5. **Notation:** algebraic, with numerical constants
6. **Physical interpretation required:** yes/no

**Target / score:**
- Simplified expression (2-5 tokens)
- Score: exact match or algebraic equivalence

**Negative controls:**
1. No limiting behavior: "E = mc^2 as v → 0" where E doesn't depend on v → should return E = mc^2 unchanged
2. Undefined limit: "1/x as x → 0" → should indicate divergence/undefined, not a finite answer

**Intervention plan:**
1. **Ablate limit-propagation features:** Remove features tracking parameter approaching limit. Prediction: model returns full expression or wrong dominant term.
2. **Swap limit direction:** Patch v→0 features with v→c features. Prediction: model gives divergent behavior instead of rest mass.

**Why it is novel:**
Unlike direct evaluation, this requires identifying which terms dominate as a parameter varies—a qualitative analysis rather than numerical computation.

**Risks / confounds:**
Model may memorize specific limiting cases without general procedure. Vary expressions beyond standard textbook forms.

---

## Behaviour 10: Error Propagation Direction

**Type:** B (Candidate set selection) — selects dominant error source from candidate inputs

**Core latent variable(s):**
- Relative uncertainty at each stage (Δx/x)
- Dominant error contributor

**Behavioural hypothesis:**
The model tracks how measurement uncertainties propagate through a calculation chain, identifying which input error dominates the output uncertainty.

**Prompt template:**
```
Calculation: density = mass / volume
mass = 5.0 ± 0.5 kg (10% error)
volume = 2.0 ± 0.1 m^3 (5% error)
Dominant error source:
```
Target: ` mass` (10% > 5%)

**Prompt-family generator recipe:**
1. **Operation type:** division, multiplication, addition, power, mixed
2. **Number of inputs:** 2, 3, or 4 quantities
3. **Error magnitudes:** varied to create clear/ambiguous dominance
4. **Correlation:** uncorrelated, correlated errors
5. **Presentation:** percentage, absolute, relative
6. **Formula complexity:** single operation, chained operations

**Target / score:**
- Single variable name token: ` mass`, ` volume`, etc.
- Score: accuracy on identifying dominant source

**Negative controls:**
1. Equal errors: "mass 10% error, volume 10% error" → should indicate "both" or "equal"
2. Additive combination: "total = m1 + m2" with small m1 → dominant depends on absolute, not relative error

**Intervention plan:**
1. **Ablate error-comparison features:** Remove features encoding relative error magnitudes. Prediction: model defaults to first-listed variable or random.
2. **Swap error magnitudes:** Patch error values so volume has larger error. Prediction: model flips answer to "volume."

**Why it is novel:**
Unlike standard error propagation calculating final uncertainty, this asks which source dominates—requiring comparative reasoning about propagated uncertainties.

**Risks / confounds:**
Model may use positional heuristics (first variable mentioned). Randomize variable ordering.

---

## Behaviour 11: Intensive vs Extensive Property Classification

**Type:** C (Abstraction vs surface form)

**Core latent variable(s):**
- Scaling behavior under system doubling (intensive: unchanged, extensive: doubles)
- Property category label

**Behavioural hypothesis:**
The model maps surface property names to their abstract thermodynamic classification based on scaling behavior. The latent variable is the scaling exponent (0 for intensive, 1 for extensive).

**Prompt template:**
```
Property: Temperature
Intensive or Extensive?
Answer:
```
Target: ` Intensive`

**Prompt-family generator recipe:**
1. **Property type:** temperature, pressure, density, mass, volume, entropy, energy, concentration
2. **Context:** thermodynamic, chemical, mechanical
3. **Presentation:** property name, property definition, property in context
4. **Ambiguous cases:** specific heat (extensive per mole, intensive per kg)
5. **System description:** with/without explicit system size mentioned
6. **Domain:** classical thermo, statistical mechanics, chemistry

**Target / score:**
- Single token: ` Intensive` or ` Extensive`
- Score: accuracy across property types

**Negative controls:**
1. Not a physical property: "Color: Intensive or Extensive?" → N/A category
2. Defined intensive quantity: "pressure = force/area" → definition reveals answer, not testing classification

**Intervention plan:**
1. **Ablate scaling-behavior features:** Remove features encoding size-dependence. Prediction: model defaults to more common category or guesses.
2. **Swap property-identity features:** Patch "temperature" encoding with "energy" encoding. Prediction: model flips to Extensive.

**Why it is novel:**
Unlike factual recall of classifications, this tests whether the model understands the underlying scaling principle that defines the categories.

**Risks / confounds:**
Model may memorize property→category mapping. Test with less common properties or novel definitions.

---

## Behaviour 12: Stoichiometric Balance Verification

**Type:** D (Gating / policy circuits) + Internal checking

**Core latent variable(s):**
- Atom count per element on each side
- Balance flag (all elements match / some don't)

**Behavioural hypothesis:**
The model computes atom counts for each element on reactant and product sides, then a comparison circuit gates output to "balanced" or "unbalanced."

**Prompt template:**
```
Reaction: 2H2 + O2 → 2H2O
Balanced? Answer:
```
Target: ` Yes` (H: 4=4, O: 2=2)

**Prompt-family generator recipe:**
1. **Reaction type:** combustion, synthesis, decomposition, redox, acid-base
2. **Balance status:** 50% balanced, 50% unbalanced
3. **Imbalance type:** missing coefficient, wrong subscript, missing product
4. **Complexity:** 2, 3, or 4 species
5. **Coefficient magnitude:** small (1-4), larger (5-10)
6. **Notation:** molecular formulas, with state symbols, with charges (ionic)

**Target / score:**
- Single token: ` Yes` or ` No`
- Score: accuracy on balanced set

**Negative controls:**
1. Nuclear reaction: "U-235 → Ba + Kr + 3n" → mass number balance, not atom count
2. Incomplete formula: "H2 + O → H2O" → O should be O2; tests formula knowledge vs balance

**Intervention plan:**
1. **Ablate atom-counting features:** Remove features encoding element totals. Prediction: model cannot verify balance; guesses.
2. **Patch element-count features:** Insert incorrect H count features. Prediction: model reports unbalanced for balanced reaction.

**Why it is novel:**
Unlike balancing equations (finding coefficients), this is a verification task requiring systematic checking—testing internal consistency computation.

**Risks / confounds:**
Model may use heuristics (equal coefficients on both sides). Use reactions where balance requires different coefficients.

---

## Behaviour 13: Causal Ordering Constraint

**Type:** D (Gating / policy circuits)

**Core latent variable(s):**
- Temporal/causal order of events
- Causality violation flag

**Behavioural hypothesis:**
The model checks whether a described sequence of events respects causal ordering (effect cannot precede cause). A gating circuit activates on causal violations.

**Prompt template:**
```
Event sequence: Light from star reaches Earth (t=0). Star explodes (t=1 year before).
Causally possible? Answer:
```
Target: ` No` (light arrival cannot be observed before emission)

Let me reconsider: If the star explodes at t=-1 year and light reaches at t=0, that's causally valid (light travels for 1 year). Let me fix:

**Prompt template:**
```
Event sequence: Observer sees star explode at t=0. Light leaves star at t=+1 year.
Causally possible? Answer:
```
Target: ` No` (effect precedes cause)

**Prompt-family generator recipe:**
1. **Domain:** relativistic, everyday causation, thermodynamic (entropy), quantum
2. **Violation type:** effect before cause, faster-than-light signal, entropy decrease
3. **Temporal specification:** explicit times, relative ordering, implied sequence
4. **Validity status:** 50% valid, 50% invalid
5. **Complexity:** 2-event, 3-event chains
6. **Misdirection:** valid sequences that sound strange, invalid ones that sound plausible

**Target / score:**
- Single token: ` Yes` (possible) or ` No` (impossible)
- Score: accuracy on causal validity

**Negative controls:**
1. No temporal claim: "Star is bright. Earth is far." → no causal claim to evaluate
2. Correlation without causation: "Rooster crows, sun rises" → correlation, not causal claim

**Intervention plan:**
1. **Ablate temporal-ordering features:** Remove features encoding event sequence. Prediction: model cannot detect ordering violations.
2. **Swap event-order features:** Reverse temporal features in context. Prediction: model flips causality judgment.

**Why it is novel:**
Unlike timeline comprehension, this tests whether the model enforces physical causality constraints—a meta-level validity check.

**Risks / confounds:**
Model may use linguistic cues ("before"/"after") without understanding physical causality. Use implicit causal structures.

---

## Behaviour 14: Tensor Contraction Validity

**Type:** B (Candidate set selection)

**Core latent variable(s):**
- Index structure of each tensor (contravariant/covariant, dimension)
- Contraction compatibility (matching index pairs)

**Behavioural hypothesis:**
Given tensors with specified index structures, the model filters candidate contractions to those with properly matching indices—a constraint satisfaction over index labels.

**Prompt template:**
```
Tensors: A^{ij}, B_{jk}
Valid contraction? Options: A^{ij}B_{jk}, A^{ij}B_{ik}, A^{ij}B_{lm}
Answer:
```
Target: ` A^{ij}B_{jk}` (j index contracts: upper and lower)

**Prompt-family generator recipe:**
1. **Tensor rank:** (1,0), (0,1), (2,0), (1,1), (0,2), etc.
2. **Number of tensors:** 2, 3, or 4
3. **Index dimension:** 2D, 3D, 4D (implicit)
4. **Contraction pattern:** single, double, trace
5. **Distractor contractions:** non-matching indices, repeated index on same level
6. **Notation:** index notation, matrix notation, component notation

**Target / score:**
- Contraction expression (3-8 tokens)
- Score: accuracy on valid contraction selection

**Negative controls:**
1. All valid contractions: multiple correct options → should indicate ambiguity
2. No valid contraction: incompatible index structures → should indicate "none"

**Intervention plan:**
1. **Ablate index-matching features:** Remove features encoding up/down index distinction. Prediction: model selects invalid contractions.
2. **Swap index features:** Replace j^upper with j_lower encoding. Prediction: different contraction becomes valid.

**Why it is novel:**
Unlike matrix multiplication, this requires abstract index algebra—tracking symbolic indices rather than numerical dimensions.

**Risks / confounds:**
Model may use positional heuristics (adjacent indices contract). Vary index positions systematically.

---

## Behaviour 15: Order of Magnitude Estimation

**Type:** B (Candidate set selection)

**Core latent variable(s):**
- Logarithmic scale representation of quantity
- Order of magnitude (power of 10)

**Behavioural hypothesis:**
The model estimates order of magnitude by mapping quantities to logarithmic scale, then selects from candidates based on proximity to estimated value.

**Prompt template:**
```
Number of atoms in a human body. Order of magnitude?
Options: 10^10, 10^18, 10^28, 10^40
Answer:
```
Target: ` 10^28` (approximately 7×10^27)

**Prompt-family generator recipe:**
1. **Quantity type:** counting (atoms, stars), physical (distances, energies), everyday (grains of sand)
2. **Scale:** microscopic, human-scale, astronomical, cosmological
3. **Candidate spread:** 2 orders of magnitude apart, varying number of options
4. **Distractors:** common misconceptions (e.g., atoms in body = 10^10)
5. **Hint level:** with/without reference values provided
6. **Precision required:** nearest OoM, within 2 OoM

**Target / score:**
- Order of magnitude (3-5 tokens like ` 10^28`)
- Score: |log10(predicted) - log10(correct)| ≤ 1

**Negative controls:**
1. Exact value given: "10^23 atoms. Order of magnitude:" → trivial extraction
2. Impossible quantity: "Number of atoms in a photon" → should indicate N/A

**Intervention plan:**
1. **Ablate scale-estimation features:** Remove features encoding logarithmic magnitude. Prediction: model selects based on surface pattern or random.
2. **Patch reference scale:** Insert "nanometer" scale features into "astronomical" context. Prediction: model underestimates by many orders.

**Why it is novel:**
Unlike precise calculation, this tests Fermi estimation—a coarse-grained reasoning mode that may use distinct circuitry from exact computation.

**Risks / confounds:**
Model may memorize common Fermi estimates. Use novel estimation targets.

---

## Behaviour 16: Logical Quantifier Scope Resolution

**Type:** A (Latent-state propagation)

**Core latent variable(s):**
- Quantifier binding structure (∀ vs ∃, nesting order)
- Scope assignment for each variable

**Behavioural hypothesis:**
The model parses nested quantifiers and propagates scope assignments through the logical structure. Different scope orderings yield different truth values.

**Prompt template:**
```
Statement: "Every student passed some exam."
Reading: Each student passed at least one exam (possibly different exams).
Quantifier order:
```
Target: ` ∀x∃y` (universal over students scopes over existential over exams)

**Prompt-family generator recipe:**
1. **Quantifier types:** universal (every, all, each), existential (some, a, exists)
2. **Nesting depth:** 2, 3 quantifiers
3. **Domain:** students/exams, people/places, numbers
4. **Ambiguity:** genuinely ambiguous, unambiguous
5. **Reading specification:** surface form, logical form requested
6. **Scope order:** ∀∃, ∃∀, mixed

**Target / score:**
- Logical form (3-6 tokens): ` ∀x∃y`, ` ∃y∀x`
- Score: exact match on quantifier order

**Negative controls:**
1. Single quantifier: "Every student passed" → no scope ambiguity
2. Different readings equivalent: "Some student exists" → scope doesn't matter

**Intervention plan:**
1. **Ablate scope-assignment features:** Remove features encoding quantifier nesting. Prediction: model defaults to surface order or random.
2. **Swap quantifier features:** Replace "every" encoding with "some" encoding. Prediction: scope structure changes.

**Why it is novel:**
Unlike semantic parsing of simple sentences, this targets the specific computation of quantifier scope—a well-studied linguistic phenomenon with clear latent structure.

**Risks / confounds:**
Model may use surface word order as proxy for scope. Use constructions where scope differs from word order.

---

## Behaviour 17: Negation Scope Propagation

**Type:** A (Latent-state propagation)

**Core latent variable(s):**
- Negation operator position in logical form
- Scope of negation (which predicates are negated)

**Behavioural hypothesis:**
The model tracks negation scope through compositional structure, determining which elements fall under the negation. Intervention on scope features should flip entailment judgments.

**Prompt template:**
```
Sentence: "The detector did not find all particles."
Logically equivalent to: "The detector found no particles."
True or False?
```
Target: ` False` (¬∀x.P(x) ≢ ∀x.¬P(x); "not all" ≠ "none")

**Prompt-family generator recipe:**
1. **Negation type:** sentential, constituent, implicit (few, barely)
2. **Quantifier interaction:** all/some, every/any, most/few
3. **Embedding depth:** simple, embedded clause, double negation
4. **Equivalence status:** 50% equivalent, 50% non-equivalent
5. **Domain:** physical, everyday, abstract
6. **Misdirection:** surface-similar non-equivalents

**Target / score:**
- Single token: ` True` or ` False`
- Score: accuracy on equivalence judgment

**Negative controls:**
1. No negation: "The detector found all particles = The detector found all particles" → trivially true
2. Obvious non-equivalence: "Found all" vs "Found none" → surface difference makes it easy

**Intervention plan:**
1. **Ablate negation-scope features:** Remove features encoding negation position. Prediction: model fails on negation-quantifier interactions.
2. **Swap negation features:** Move negation from wide to narrow scope. Prediction: equivalence judgment flips.

**Why it is novel:**
Unlike sentiment negation, this tests formal scope computation—a structural property independent of lexical content.

**Risks / confounds:**
Model may learn specific patterns ("not all" ≠ "none") without general scope understanding. Vary quantifiers and predicates.

---

## Behaviour 18: Coordination Structure Resolution

**Type:** C (Abstraction vs surface form)

**Core latent variable(s):**
- Coordination level (NP-coordination vs VP-coordination vs S-coordination)
- Distributed reading vs collective reading

**Behavioural hypothesis:**
The model parses ambiguous coordination structures to a canonical abstract representation, then generates appropriate entailments based on the parse.

**Prompt template:**
```
Sentence: "Old men and women attended."
Ambiguous reading? Answer:
```
Target: ` Yes` (old modifies "men" only vs "men and women")

**Prompt-family generator recipe:**
1. **Ambiguity type:** modifier attachment, coordination scope, collective/distributive
2. **Syntactic category:** NP, VP, PP, AP coordination
3. **Modifier type:** adjective, PP, relative clause
4. **Ambiguity status:** genuinely ambiguous, unambiguous
5. **Resolution cue:** context that disambiguates, no context
6. **Domain:** everyday, technical, physics

**Target / score:**
- Single token: ` Yes` (ambiguous) or ` No` (unambiguous)
- Score: accuracy on ambiguity detection

**Negative controls:**
1. Unambiguous coordination: "Red apples and oranges" → adjective clearly modifies only "apples" (different fruit)
2. Repeated modifier: "Old men and old women" → explicitly unambiguous

**Intervention plan:**
1. **Ablate attachment features:** Remove features encoding modifier scope. Prediction: model cannot detect ambiguity.
2. **Patch coordination features:** Insert low-attachment features into high-attachment context. Prediction: ambiguity judgment changes.

**Why it is novel:**
Unlike standard parsing, this tests meta-linguistic awareness of structural ambiguity—the model must recognize multiple valid parses exist.

**Risks / confounds:**
Model may use world knowledge (old men are common) rather than structural analysis. Use nonsense words to isolate syntax.

---

## Behaviour 19: Physical Plausibility Gating

**Type:** D (Gating / policy circuits) + Internal checking

**Core latent variable(s):**
- Physical constraint satisfaction (energy bounds, speed limits, scale consistency)
- Plausibility flag

**Behavioural hypothesis:**
The model checks described scenarios against physical constraints (c limit, Planck scale, energy bounds) and gates responses to "implausible" when constraints are violated.

**Prompt template:**
```
Scenario: A car accelerates to 500,000 km/s in 10 seconds.
Physically plausible? Answer:
```
Target: ` No` (exceeds speed of light)

**Prompt-family generator recipe:**
1. **Constraint type:** speed of light, absolute zero, conservation laws, scale constraints
2. **Violation type:** exceeds bound, violates conservation, wrong scale
3. **Plausibility status:** 50% plausible, 50% implausible
4. **Violation magnitude:** small (technical), large (obvious)
5. **Domain:** mechanics, thermodynamics, relativity, quantum
6. **Misdirection:** implausible-sounding but valid scenarios

**Target / score:**
- Single token: ` Yes` (plausible) or ` No` (implausible)
- Score: accuracy on physical validity

**Negative controls:**
1. Fictional context: "In the story, the ship travels at warp 9" → should not apply physics constraints to fiction
2. Approximate values: "Car travels at about the speed of sound" → plausible, shouldn't require exact bound

**Intervention plan:**
1. **Ablate constraint-checking features:** Remove features encoding physical bounds. Prediction: model accepts implausible scenarios.
2. **Swap constraint features:** Insert "c = 3×10^8 m/s" bound into context with different bound. Prediction: threshold for plausibility shifts.

**Why it is novel:**
Unlike physics problem solving, this tests constraint satisfaction checking—a gating computation that should activate for any violation, not domain-specific.

**Risks / confounds:**
Model may use keyword heuristics ("faster than light" → implausible). Use implicit violations without keywords.

---

## Behaviour 20: Coordinate Invariant Recognition

**Type:** C (Abstraction vs surface form)

**Core latent variable(s):**
- Coordinate-independent geometric quantity (dot product, proper length, scalar curvature)
- Invariant vs variant classification

**Behavioural hypothesis:**
The model recognizes that certain quantities are unchanged under coordinate transformations (invariants) and maps surface expressions to their invariance status.

**Prompt template:**
```
Quantity: Length contraction factor γ
Coordinate system changed from rest frame to moving frame.
Value changes? Answer:
```
Target: ` No` (γ is invariant—it depends on relative velocity, not coordinate choice)

Wait, this is tricky. γ = 1/√(1-v²/c²) where v is relative velocity. If we change coordinates, v stays the same (it's the relative velocity). So γ is invariant. Let me reconsider the prompt:

**Prompt template (revised):**
```
Quantity: Electric field magnitude at a point
Observer changes from rest frame to moving frame.
Value changes? Answer:
```
Target: ` Yes` (E field transforms under Lorentz transformation)

**Prompt-family generator recipe:**
1. **Quantity type:** scalar invariant (proper time, rest mass), vector component (E, B), tensor component
2. **Transformation type:** Lorentz boost, rotation, Galilean, general coordinate
3. **Invariance status:** 50% invariant (scalars), 50% variant (components)
4. **Domain:** special relativity, general relativity, electromagnetism, classical mechanics
5. **Presentation:** explicit formula, named quantity, described quantity
6. **Frame specification:** rest to moving, lab to CM, explicit velocity

**Target / score:**
- Single token: ` Yes` (changes) or ` No` (invariant)
- Score: accuracy on invariance classification

**Negative controls:**
1. Trivially invariant: "Speed of light" under any transformation → built-in constraint
2. No transformation specified: "Electric field value:" → cannot answer without transformation

**Intervention plan:**
1. **Ablate invariance features:** Remove features encoding transformation properties. Prediction: model guesses based on surface features.
2. **Swap quantity features:** Replace E-field encoding with proper-time encoding. Prediction: answer flips from Yes to No.

**Why it is novel:**
Unlike calculating transformed values, this tests whether the model has learned the abstract distinction between invariants and components—a fundamental physics concept.

**Risks / confounds:**
Model may memorize "proper time is invariant" without understanding why. Test with less common invariants and varied transformations.

---

## Summary Table

| # | Title | Type | Physics | Checking | Latent Variable |
|---|-------|------|---------|----------|-----------------|
| 1 | Dimensional Consistency Verification | D | ✓ | ✓ | Dimensional signature |
| 2 | Reference Frame Velocity Composition | A | ✓ | | Intermediate velocity |
| 3 | Oxidation State Tracking | A | ✓ | | Oxidation state |
| 4 | Symmetry-Allowed Quantum Transitions | B | ✓ | | Symmetry labels |
| 5 | Approximation Regime Selection | B | ✓ | | Dimensionless ratio |
| 6 | Unit Canonicalization | C | ✓ | | SI representation |
| 7 | Equation Form Equivalence | C | ✓ | ✓ | Algebraic normal form |
| 8 | Conservation Law Violation Detection | D | ✓ | ✓ | Conserved quantity totals |
| 9 | Limiting Case Extrapolation | A | ✓ | | Dominant term |
| 10 | Error Propagation Direction | B | ✓ | | Relative uncertainties |
| 11 | Intensive vs Extensive Classification | C | ✓ | | Scaling exponent |
| 12 | Stoichiometric Balance Verification | D | ✓ | ✓ | Atom counts |
| 13 | Causal Ordering Constraint | D | ✓ | ✓ | Temporal order |
| 14 | Tensor Contraction Validity | B | ✓ | | Index structure |
| 15 | Order of Magnitude Estimation | B | ✓ | | Logarithmic scale |
| 16 | Logical Quantifier Scope | A | | | Scope assignment |
| 17 | Negation Scope Propagation | A | | | Negation position |
| 18 | Coordination Structure Resolution | C | | | Coordination level |
| 19 | Physical Plausibility Gating | D | ✓ | | Constraint satisfaction |
| 20 | Coordinate Invariant Recognition | C | ✓ | | Invariance class |

**Type Distribution:** A: 5, B: 5, C: 5, D: 5
**Physics-oriented:** 14/20
**Internal checking:** 6/20

---

## Implementation Priority

**Tier 1 (Clearest latent structure, easiest to implement):**
1. Dimensional Consistency Verification
2. Stoichiometric Balance Verification
3. Unit Canonicalization
6. Oxidation State Tracking

**Tier 2 (Strong physics content, moderate complexity):**
4. Conservation Law Violation Detection
5. Approximation Regime Selection
7. Symmetry-Allowed Transitions
8. Reference Frame Velocity Composition

**Tier 3 (Novel but may have attribution challenges):**
9-20 (linguistic behaviors, abstract physics invariants)

---

*Document version: 1.0*
*Created: 2025-02*
*For: Mechanistic Interpretability Thesis Project*
