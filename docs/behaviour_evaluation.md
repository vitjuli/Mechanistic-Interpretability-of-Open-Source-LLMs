# Behaviour Evaluation for Mechanistic Interpretability

## Evaluation Criteria
For each behaviour:
1. **Feasibility** (low/medium/high) for our transcoder pipeline
2. **Latent variables** and likely feature representations
3. **Prompt-family variations** (3–6 controlled factors)
4. **Intervention tests** (ablation + swap-in with directional predictions)
5. **Confounds/heuristics** and mitigations
6. **Recommendation** (keep/modify/drop)

---

# TYPE A: Latent-State Propagation

---

## A1. Decay-Type Inference

**Prompt:** "A nucleus emits a particle with charge +2e and mass number 4. What type of decay occurred?"
**Expected answer:** `alpha`

### 1. Feasibility: HIGH

Clean single-token output. Clear computational pathway: particle properties → decay classification. Well-suited for feature-level intervention.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Emitted particle charge | Charge-magnitude features in early-mid layers |
| Emitted particle mass number | Mass-number encoding features |
| Particle identity (He-4 nucleus) | Composite "helium nucleus" concept |
| Decay type label | Classification features in late layers |

**Computational pathway:**
```
Input properties → Particle signature (Z=2, A=4) → Match to known particles → Decay label
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Property specification** | Both (charge+mass), charge only, mass only, descriptive ("helium nucleus") | Test if model needs both or can infer |
| **Numerical format** | "+2e", "charge 2", "doubly positive", "Z=2" | Avoid surface pattern matching |
| **Decay types** | Alpha, beta⁻, beta⁺, gamma, neutron emission, positron | Cover full classification space |
| **Context framing** | "A nucleus emits...", "During radioactive decay...", "A heavy atom releases..." | Test context independence |
| **Distractor properties** | Unusual combinations (charge +2, mass 3) to test genuine computation | Catch memorization |

**Example variants:**
- "A radioactive atom releases a doubly-charged particle with four nucleons. The decay type is:"
- "An unstable nucleus ejects a helium-4 nucleus. This is called:"
- "In a nuclear process, the emitted particle has A=4 and Z=2. The decay is:"

### 4. Intervention Tests

**Intervention 1: Ablate charge-encoding features**
- **Target:** Features in layers 12-18 that encode charge magnitude
- **Method:** Zero ablation of top-attributed charge features
- **Prediction:** Model loses ability to distinguish alpha (Z=2) from beta (Z=±1); accuracy drops to ~25% (random among decay types)
- **Control:** Ablate unrelated features → no effect

**Intervention 2: Swap particle-property features**
- **Target:** Features encoding "+2e" charge
- **Method:** Patch features from a beta-decay prompt (charge -1e) into alpha-decay context
- **Prediction:** Model outputs "beta" instead of "alpha"
- **Magnitude:** Effect should be near-complete flip if features are causal

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| Keyword lookup | Model memorizes "+2e AND mass 4 = alpha" | Use varied phrasings; test partial information |
| Training data bias | Alpha decay most common in training | Balance decay types in prompt set |
| Two-step reasoning collapse | Model might have single lookup feature | Test partial property prompts |

### 6. Recommendation: **KEEP**

Strong behaviour with clear latent structure, clean output, and meaningful physics content. Good candidate for Tier 1 implementation.

---

## A2. Reference-Frame Sign Propagation

**Prompt:** "A person walks forward inside a train that is moving backward relative to the ground. Is the person moving forward or backward relative to the ground?"
**Expected answer:** Depends on magnitudes (AMBIGUOUS as stated)

### 1. Feasibility: MEDIUM (requires modification)

Current form is **ambiguous** — if person walks at 2 m/s forward and train moves 10 m/s backward, person moves backward relative to ground. If person walks 15 m/s forward, person moves forward.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Train velocity relative to ground (sign) | Direction features (±) |
| Person velocity relative to train (sign) | Direction features (±) |
| Magnitude comparison (implicit) | Magnitude-dominance features |
| Composed velocity sign | Final direction classification |

### 3. Prompt-Family Variations (for MODIFIED version)

**Modified prompt (unambiguous):**
"A person walks slowly forward inside a train that is moving quickly backward relative to the ground. Is the person moving forward or backward relative to the ground?"
**Expected:** `backward` (fast train dominates)

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Magnitude qualifiers** | "slowly/quickly", "at 2 m/s / 10 m/s", "much faster/slower" | Disambiguate |
| **Direction combinations** | Same direction, opposite direction | Test sign composition |
| **Reference frame asked** | "relative to ground", "relative to train", "relative to person" | Test frame handling |
| **Context objects** | Person/train, boat/river, plane/wind, car/road | Generalization |
| **Qualifier position** | "Person walks slowly, train moves quickly" vs reversed | Order independence |

### 4. Intervention Tests

**Intervention 1: Ablate magnitude-comparison features**
- **Target:** Features encoding "quickly > slowly" comparison
- **Method:** Zero ablation
- **Prediction:** Model defaults to one direction regardless of qualifiers; accuracy → 50%

**Intervention 2: Swap direction features**
- **Target:** Features encoding "backward" for train
- **Method:** Patch "forward" train features
- **Prediction:** Answer flips (now both move same direction, person moves forward relative to ground)

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| Linguistic "forward" ambiguity | Person's forward vs train's forward | Use explicit reference frames |
| Frequency bias | Most training examples might have person moving forward | Balance directions |
| Magnitude heuristics | "Quickly" always wins | Test edge cases where slow wins |

### 6. Recommendation: **MODIFY then KEEP**

Requires explicit magnitude information to be unambiguous. With modification, tests genuine velocity composition with clear latent structure.

**Modified canonical prompt:**
"A person walks slowly forward inside a fast train moving backward relative to the ground. Relative to the ground, the person moves:"
**Target:** `backward`

---

## A3. Spin/Angular Momentum Conservation in Decay

**Prompt:** "A particle with zero spin decays into two identical photons. What is the total spin of the final state?"
**Expected answer:** `0`

### 1. Feasibility: LOW

This is **trivially true by conservation** — the answer is just "initial spin = final spin = 0". No genuine multi-step computation; the model just needs to echo the conservation constraint.

### 2. Latent Variables

| Latent Variable | Problem |
|-----------------|---------|
| Initial spin | Given directly (0) |
| Conservation rule | Just applies identity |
| Final spin | Equals initial by conservation |

**The computation is degenerate:** 0 → conservation → 0

### 3. Why This Fails as a Behaviour

- No intermediate state to track (initial = final directly)
- No candidate selection (only one valid answer)
- Tests recall of conservation law, not its application
- A stronger version would require combining angular momenta: "Spin-1/2 particle combines with spin-1/2 particle. What are the possible total spins?"

### 4. Confounds

- Model may output "0" simply because it appears in the prompt
- Conservation is a direct constraint, not a computation

### 5. Recommendation: **DROP**

Replace with a behaviour requiring non-trivial angular momentum coupling or multi-step propagation.

**Suggested replacement (see Additional Behaviours section):**
"A spin-1/2 electron and a spin-1/2 positron annihilate. Can the total angular momentum of the photon pair be 1?"
**Target:** `no` (two photons can have total spin 0 or 2, not 1)

---

# TYPE B: Candidate Set Selection

---

## B1. Selection-Rule Allow/Forbid

**Prompt:** "Is an electric dipole transition from an s-state to another s-state allowed?"
**Expected answer:** `no`

### 1. Feasibility: HIGH

Clean binary output. Clear latent: s-state has l=0, E1 requires Δl=±1, so s→s (Δl=0) is forbidden.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Initial orbital angular momentum | l-value encoding (l=0 for s) |
| Final orbital angular momentum | l-value encoding |
| Selection rule constraint | Δl=±1 rule features |
| Transition type | E1/M1/E2 classification |
| Gate output | Allowed/forbidden binary |

**Computational pathway:**
```
Initial state → l_initial → Δl computation → Compare to Δl=±1 → Gate
Final state → l_final →
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Initial state** | s (l=0), p (l=1), d (l=2), f (l=3) | Test all orbitals |
| **Final state** | s, p, d, f | Full matrix of transitions |
| **Transition type** | Electric dipole (E1), magnetic dipole (M1), electric quadrupole (E2) | Different selection rules |
| **State notation** | "s-state", "l=0 state", "spherically symmetric orbital" | Surface form independence |
| **Phrasing** | "allowed", "forbidden", "possible", "permitted by selection rules" | Linguistic variation |
| **Atom context** | Hydrogen, generic, "in an atom", none | Context independence |

**Key test matrix:**
| Initial | Final | E1 (Δl=±1) | Expected |
|---------|-------|------------|----------|
| s | s | Δl=0 | no |
| s | p | Δl=1 | yes |
| s | d | Δl=2 | no |
| p | s | Δl=1 | yes |
| p | p | Δl=0 | no |
| p | d | Δl=1 | yes |
| d | f | Δl=1 | yes |
| d | s | Δl=2 | no |

### 4. Intervention Tests

**Intervention 1: Ablate Δl-computation features**
- **Target:** Features encoding the difference between l-values
- **Method:** Zero ablation of top-attributed features
- **Prediction:** Model loses selection rule application; accuracy → 50%
- **Specificity test:** Should not affect magnetic dipole transitions (Δl=0 allowed)

**Intervention 2: Swap orbital-identity features**
- **Target:** Features encoding "s-state" (l=0)
- **Method:** Patch "p-state" features into s-state context
- **Prediction:** s→s becomes "p→s" in computation → Δl=1 → answer flips to "yes"

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| Memorized lookup table | "s→s forbidden" as pattern | Test all 16 combinations; use varied notation |
| Transition type confusion | Only testing E1 | Include M1 (Δl=0 allowed) and E2 (Δl=0,±2) |
| Keyword heuristics | "s-state" triggers "forbidden" | Use "l=0 state" notation variants |

### 6. Recommendation: **KEEP**

Excellent behaviour with clear latent structure, binary output, and rich prompt-family space. Tests genuine rule application vs lookup.

---

## B2. Approximation Regime Choice

**Prompt:** "An electron moves at 1% of the speed of light. Should its kinetic energy be treated classically or relativistically?"
**Expected answer:** `classically`

### 1. Feasibility: HIGH

Binary output. Clear latent: v/c ratio thresholding. Well-defined computation.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Velocity magnitude | Numerical encoding |
| Speed of light reference | c-value features |
| v/c ratio | Ratio computation features |
| Threshold comparison | v/c ≪ 1 vs v/c ~ 1 |
| Regime label | Classical/relativistic classification |

**Computational pathway:**
```
v → v/c computation → Threshold comparison (v/c << 1?) → Regime selection
c →
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Speed specification** | "1% of c", "0.01c", "3×10⁶ m/s", "very slowly", "near light speed" | Format independence |
| **Threshold values** | 0.1%, 1%, 5%, 10%, 50%, 90%, 99% of c | Map regime boundary |
| **Particle type** | Electron, proton, "a particle", "an object" | Avoid particle-type heuristics |
| **Property asked** | Kinetic energy, momentum, mass | Different relativistic effects |
| **Framing** | "treated classically", "use Newtonian mechanics", "apply special relativity" | Phrasing independence |

**Critical test points:**
| v/c | Expected | Notes |
|-----|----------|-------|
| 0.001 (0.1%) | classical | Clearly non-relativistic |
| 0.01 (1%) | classical | Standard boundary |
| 0.1 (10%) | classical (borderline) | γ ≈ 1.005 |
| 0.5 (50%) | relativistic | γ ≈ 1.15 |
| 0.9 (90%) | relativistic | γ ≈ 2.3 |

### 4. Intervention Tests

**Intervention 1: Ablate v/c comparison features**
- **Target:** Features computing or encoding the v/c ratio
- **Method:** Zero ablation
- **Prediction:** Model defaults to one regime regardless of speed; accuracy → 50%

**Intervention 2: Swap velocity-magnitude features**
- **Target:** Features encoding "1%" in velocity context
- **Method:** Patch "90%" features into "1%" context
- **Prediction:** Answer flips from "classically" to "relativistically"

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Electron" → quantum heuristic | Keyword association | Use varied particle types |
| "Speed of light" → relativistic | Keyword in prompt triggers answer | Use numerical speeds without mentioning c |
| Percentage parsing errors | 1% vs 100% confusion | Test with multiple formats |

### 6. Recommendation: **KEEP**

Strong regime-selection behaviour with clear threshold latent. Good for testing v/c circuit.

---

## B3. Low-Temperature Dominance in Partition Function

**Prompt:** "At extremely low temperature, which energy state dominates the partition function of a system?"
**Expected answer:** `ground state`

### 1. Feasibility: MEDIUM-HIGH

Tests statistical mechanics understanding. Output is 2 tokens ("ground state") but unambiguous.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Temperature regime | T → 0 limit encoding |
| Boltzmann factor structure | e^{-E/kT} suppression understanding |
| Energy level ordering | Ground state = lowest energy |
| Dominant contribution | Probability concentration features |

**Computational pathway:**
```
T → 0 limit → Boltzmann factors e^{-Eᵢ/kT} → Lowest E dominates → Ground state
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Temperature description** | "extremely low", "near absolute zero", "T → 0", "cryogenic", "approaching 0 K" | Linguistic independence |
| **System type** | "a system", "an atom", "a quantum harmonic oscillator", "a gas molecule" | Generalization |
| **State terminology** | "energy state", "energy level", "quantum state", "configuration" | Surface form |
| **Dominance phrasing** | "dominates", "has highest probability", "contributes most", "is most occupied" | Semantic variation |
| **Opposite regime** | "extremely high temperature" → "all states equally likely" | Bidirectional test |

### 4. Intervention Tests

**Intervention 1: Ablate temperature-regime features**
- **Target:** Features encoding the T → 0 limit
- **Method:** Zero ablation
- **Prediction:** Model cannot identify low-T limit; may output "all states" or random

**Intervention 2: Swap temperature features**
- **Target:** Features encoding "extremely low"
- **Method:** Patch "extremely high" temperature features
- **Prediction:** Answer changes to "all states contribute equally" or "no single state dominates"

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Low temperature" → "ground state" keyword | Direct association without mechanism | Include intermediate temperatures |
| Missing high-T contrast | Only testing one direction | Add high-T prompts to family |
| "Partition function" as trigger | Technical term might cue answer | Use "probability distribution" variant |

### 6. Recommendation: **KEEP**

Good statistical mechanics behaviour. Tests understanding of Boltzmann distribution limiting behaviour.

---

# TYPE C: Abstraction vs Surface Form

---

## C1. Gauge Equivalence

**Prompt:** "Do two vector potentials that differ by the gradient of a scalar field produce the same magnetic field?"
**Expected answer:** `yes`

### 1. Feasibility: HIGH

Binary output. Tests deep physics understanding (gauge invariance). Clear latent: B = ∇×A and ∇×(∇f) = 0.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Vector potential A | Field representation features |
| Gauge transformation | A → A + ∇f structure |
| Curl operation | ∇× encoding |
| Identity ∇×∇f = 0 | Mathematical identity features |
| Invariance conclusion | Same/different B classification |

**Computational pathway:**
```
A₁ = A₂ + ∇f → B₁ = ∇×A₁ = ∇×(A₂ + ∇f) = ∇×A₂ + ∇×∇f = ∇×A₂ = B₂
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Transformation type** | "gradient of scalar", "∇f", "derivative of a function", "added a curl-free field" | Mathematical phrasing |
| **Field asked about** | Magnetic field B, electric field E (more complex), physical observables | Different gauge behavior |
| **Potential type** | Vector potential A, scalar potential φ, four-potential | Generalization |
| **Phrasing** | "same field", "identical B", "equivalent physically", "indistinguishable" | Semantic variation |
| **Negative test** | "differ by a constant vector" (not gauge transform) | Test specificity |

### 4. Intervention Tests

**Intervention 1: Ablate gauge-invariance features**
- **Target:** Features encoding ∇×∇ = 0 or gauge transformation structure
- **Method:** Zero ablation
- **Prediction:** Model loses understanding that A + ∇f gives same B; answer becomes uncertain

**Intervention 2: Swap field-type features**
- **Target:** Features encoding "magnetic field"
- **Method:** Patch "electric field" features (E depends on both A and φ, more complex gauge behavior)
- **Prediction:** Answer may change or become less confident (E transformation involves time derivatives of A)

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Gauge invariance" keyword recall | Memorized fact without understanding | Avoid the word "gauge" in prompts |
| Vector calculus identity lookup | ∇×∇=0 as pattern | Use varied mathematical phrasings |
| Training data skew | Gauge invariance commonly discussed | Test with less common transformations |

### 6. Recommendation: **KEEP**

Excellent abstraction behaviour. Tests whether model understands gauge invariance mechanism, not just the keyword.

---

## C2. Intensive vs Extensive via Operational Test

**Prompt:** "If you double the amount of the same substance at fixed temperature and pressure, does the density change?"
**Expected answer:** `no`

### 1. Feasibility: HIGH

Binary output. Clear operational test of scaling behaviour. Tests intensive/extensive classification without using those terms.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Property identity | Density = mass/volume encoding |
| Scaling operation | "Double amount" → 2× |
| Mass scaling | m → 2m |
| Volume scaling | V → 2V (at fixed T, P) |
| Ratio invariance | (2m)/(2V) = m/V |
| Classification output | Intensive → unchanged |

**Computational pathway:**
```
Density = m/V → Double amount: (m, V) → (2m, 2V) → ρ' = 2m/2V = m/V = ρ → No change
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Property tested** | Density, temperature, pressure (intensive); mass, volume, energy, entropy (extensive) | Full classification matrix |
| **Operation** | "Double", "halve", "triple", "combine two identical samples" | Different scaling factors |
| **Substance** | "the same substance", "water", "an ideal gas", "a metal" | Generalization |
| **Conditions** | "Fixed T and P", "constant temperature", "same conditions", none | Condition specification |
| **Phrasing** | "does X change", "is X different", "what happens to X" | Question framing |

**Key test matrix:**
| Property | Type | Double amount | Expected |
|----------|------|---------------|----------|
| Density | Intensive | → | no change |
| Temperature | Intensive | → | no change |
| Pressure | Intensive | → | no change |
| Mass | Extensive | → | doubles |
| Volume | Extensive | → | doubles |
| Energy | Extensive | → | doubles |

### 4. Intervention Tests

**Intervention 1: Ablate scaling-invariance features**
- **Target:** Features encoding ratio invariance or intensive property structure
- **Method:** Zero ablation
- **Prediction:** Model treats density like extensive property; answers "yes" (changes)

**Intervention 2: Swap property-identity features**
- **Target:** Features encoding "density"
- **Method:** Patch "mass" features into density context
- **Prediction:** Answer flips to "yes" (mass doubles when amount doubles)

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Density" → "intensive" lookup | Memorized classification | Test with operational definition only |
| Substance-specific knowledge | "Water density is 1 g/cm³" might trigger | Use generic "substance" |
| "Same substance" interpretation | Might confuse with same sample | Use "identical samples combined" variant |

### 6. Recommendation: **KEEP**

Excellent operational test of a fundamental physics concept. Tests understanding, not terminology recall.

---

## C3. Dimensional Abstraction (Velocity)

**Prompt:** "Does the expression distance divided by time have the dimensions of velocity?"
**Expected answer:** `yes`

### 1. Feasibility: HIGH but TOO TRIVIAL

This is essentially the **definition** of velocity. The model likely has "velocity = distance/time" as a strong association, making this more of a recall task than abstraction.

### 2. Why This is Weak

- **Definitional:** v = d/t is taught explicitly
- **No computation:** Just pattern matching
- **No abstraction test:** Surface form IS the canonical form

### 3. Recommended Modification

Test dimensional analysis with **non-obvious** expressions:

**Modified prompt:**
"Does the expression (mass × length²) / time² have the dimensions of energy?"
**Target:** `yes` (E = [M][L]²[T]⁻² ≡ kg⋅m²/s² = Joule)

Or use a **mismatch** test:
"Does the expression force × time have the dimensions of energy?"
**Target:** `no` (F×t = [M][L][T]⁻¹ = momentum, not energy)

### 4. Intervention Tests (for modified version)

**Intervention 1: Ablate dimensional-signature features**
- **Target:** Features encoding [M], [L], [T] signatures
- **Method:** Zero ablation
- **Prediction:** Model cannot compute composite dimensions; accuracy drops

**Intervention 2: Swap target-dimension features**
- **Target:** Features encoding "energy" dimensional signature
- **Method:** Patch "momentum" features
- **Prediction:** Answer to "mass×length²/time² = energy?" flips to "no"

### 5. Recommendation: **MODIFY or DROP**

Current form is too trivial. Replace with non-obvious dimensional analysis.

**Replacement prompt:**
"Does the expression (force × distance) / time have the dimensions of power?"
**Target:** `yes` (P = W/t = [M][L]²[T]⁻³)

---

# TYPE D: Gating / Physical Consistency Checking

---

## D1. Thermodynamic Directionality

**Prompt:** "Can heat spontaneously flow from a colder object to a hotter one in an isolated system?"
**Expected answer:** `no`

### 1. Feasibility: HIGH

Binary output. Clear gating structure (2nd law violation detector). Well-defined latent.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Temperature comparison | T_cold < T_hot encoding |
| Heat flow direction | Cold → hot vs hot → cold |
| Spontaneity condition | No external work |
| 2nd law constraint | Entropy must increase |
| Gate output | Allowed/forbidden |

**Computational pathway:**
```
Heat: cold → hot? → Spontaneous? → 2nd Law check → Entropy would decrease → FORBIDDEN
```

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Direction** | Cold → hot (no), hot → cold (yes) | Bidirectional test |
| **System type** | Isolated, closed, open (with work input) | Different constraints |
| **Spontaneity qualifier** | "Spontaneously", "naturally", "without work input", "by itself" | Semantic variation |
| **Phrasing** | "Can heat flow...", "Is it possible for heat to...", "Does heat ever..." | Question framing |
| **Energy type** | Heat, thermal energy, internal energy | Terminology variation |
| **Exception cases** | With refrigerator/heat pump (requires work) | Test boundary understanding |

### 4. Intervention Tests

**Intervention 1: Ablate 2nd-law gate features**
- **Target:** Features encoding entropy/spontaneity constraint
- **Method:** Zero ablation
- **Prediction:** Model loses ability to distinguish allowed/forbidden; accuracy → 50%

**Intervention 2: Swap temperature-direction features**
- **Target:** Features encoding "colder to hotter"
- **Method:** Patch "hotter to colder" features
- **Prediction:** Answer flips from "no" to "yes"

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Heat flows hot to cold" phrase | Memorized statement | Use varied phrasings |
| Refrigerator knowledge | "Heat pumps exist" → confusion | Emphasize "spontaneous" and "isolated" |
| Statistical mechanics edge | Fluctuations allow brief violations | Focus on macroscopic systems |

### 6. Recommendation: **KEEP**

Strong gating behaviour with clear physics content. Tests fundamental thermodynamic understanding.

---

## D2. Relativistic Causality

**Prompt:** "Can a signal travel faster than light in vacuum?"
**Expected answer:** `no`

### 1. Feasibility: HIGH

Binary output. Clear c-bound gating. Well-known physics constraint.

### 2. Latent Variables and Features

| Latent Variable | Expected Feature Representation |
|-----------------|--------------------------------|
| Speed of light value | c = 3×10⁸ m/s encoding |
| Speed comparison | v > c? |
| Object type | Signal/information vs phase velocity |
| Causality constraint | Faster than light → causality violation |
| Gate output | Possible/impossible |

### 3. Prompt-Family Variations

| Factor | Variations | Purpose |
|--------|-----------|---------|
| **Object type** | Signal, information, particle, "anything", energy (careful—phase velocity CAN exceed c) | Test specificity |
| **Speed reference** | "Faster than light", "superluminal", "above c", "exceeding 3×10⁸ m/s" | Phrasing variation |
| **Medium** | "In vacuum", "through space", "in free space", no qualifier | Context specification |
| **Phrasing** | "Can", "Is it possible", "Does physics allow", "Is there anything that can" | Question framing |
| **Edge cases** | Phase velocity (yes), group velocity (no for information), Alcubierre (speculative) | Boundary tests |

### 4. Intervention Tests

**Intervention 1: Ablate c-bound features**
- **Target:** Features encoding the speed-of-light limit
- **Method:** Zero ablation
- **Prediction:** Model loses causality constraint; may answer "yes"

**Intervention 2: Swap object-type features**
- **Target:** Features encoding "signal" (information carrier)
- **Method:** Patch "phase velocity" features
- **Prediction:** Answer changes to "yes" (phase velocity CAN exceed c)

### 5. Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|------------|
| "Nothing faster than light" memorized | Phrase recall | Use varied phrasings |
| Phase velocity confusion | Phase velocity > c is allowed | Specify "information" or "signal" |
| Quantum entanglement | "Spooky action" misconception | Include control prompts about entanglement (no information transfer) |

### 6. Recommendation: **KEEP**

Strong gating behaviour, but needs careful negative controls for phase velocity and entanglement edge cases.

---

## D3. Conservation-Law Gate (Charge)

**Prompt:** "Is electric charge conserved in particle interactions?"
**Expected answer:** `yes`

### 1. Feasibility: LOW

This is essentially **factual recall**, not a gating computation. The answer is always "yes" regardless of context—there's no gate being exercised.

### 2. Why This Fails as a Behaviour

- **No computation:** Just recall that charge is conserved
- **No candidate filtering:** No scenarios to evaluate
- **No gating:** Would need a specific interaction to check
- **Trivially true:** Like asking "Is 2+2=4?"

### 3. What Would Make It Work

A **proper conservation-law gate** would present a **specific interaction** and ask if charge is conserved:

**Better prompt:**
"In the process e⁻ + e⁺ → γ + γ, is electric charge conserved?"
**Target:** `yes` (-1 + 1 = 0 + 0)

Or a violation test:
"In the hypothetical process p → e⁺ + π⁰, would electric charge be conserved?"
**Target:** `no` (+1 → +1 + 0, actually yes; let me fix: p → e⁺ + γ would be +1 → +1 + 0, yes; p → e⁻ + π⁺ would be +1 → -1 + 1 = 0, no)

Actually: "In the hypothetical process p⁺ → n + e⁺, would electric charge be conserved?"
**Target:** `no` (+1 → 0 + 1 = 1, wait that's +1 → +1, yes it's conserved)

Better: "In the hypothetical process n → p⁺ + p⁺ + e⁻, would electric charge be conserved?"
**Target:** `no` (0 → +1 + 1 - 1 = +1 ≠ 0)

### 4. Recommendation: **DROP or HEAVILY MODIFY**

Current form is factual recall, not gating. Replace with specific interaction verification.

---

# Additional Stronger Behaviours

Based on the evaluation, here are 3 additional physics-native behaviours stronger than the weakest items (A3, C3-original, D3):

---

## Additional Behaviour 1: Radioactive Decay Series Endpoint (Type A)

**Replaces:** A3 (Spin conservation—too trivial)

**Prompt:** "Uranium-238 undergoes a series of 8 alpha decays and 6 beta-minus decays. What element does it become?"
**Expected answer:** `lead` (or `Pb`)

### Core Latent Variables
- Initial (Z, A) state: (92, 238)
- Alpha decay effect: ΔZ = -2, ΔA = -4
- Beta⁻ decay effect: ΔZ = +1, ΔA = 0
- Accumulated changes: multi-step propagation
- Final Z → element lookup

### Computational Pathway
```
U-238 (Z=92, A=238)
After 8 alpha: Z = 92 - 16 = 76, A = 238 - 32 = 206
After 6 beta⁻: Z = 76 + 6 = 82, A = 206
Z = 82 → Lead (Pb)
```

### Why Stronger Than A3
- Requires genuine multi-step computation
- Latent state (Z, A) propagates through chain
- Non-trivial arithmetic with physical meaning
- Cannot be answered by simple conservation statement

### Prompt Variations
| Factor | Variations |
|--------|-----------|
| Starting isotope | U-238, Th-232, U-235 |
| Number of alpha decays | 4, 6, 8, 10 |
| Number of beta decays | 2, 4, 6 |
| Question format | "What element", "What is the atomic number", "What isotope" |

### Intervention Tests
1. **Ablate alpha-effect features:** Model loses ΔZ = -2 per alpha; final Z wrong
2. **Swap decay-count features:** Replace "8 alpha" with "6 alpha"; endpoint changes

---

## Additional Behaviour 2: Quantum Number Constraint (Type B)

**Replaces:** C3-original (Dimensional velocity—too trivial)

**Prompt:** "For an electron in a hydrogen atom with principal quantum number n=2, is the orbital angular momentum quantum number l=2 allowed?"
**Expected answer:** `no`

### Core Latent Variables
- Principal quantum number n
- Orbital quantum number l
- Constraint: l < n (l ∈ {0, 1, ..., n-1})
- Gate: constraint satisfied?

### Computational Pathway
```
n = 2 → l ∈ {0, 1}
Query: l = 2
Check: 2 < 2? → No
Output: not allowed
```

### Why Stronger
- Tests genuine constraint checking (l < n)
- Binary output
- Clear latent: quantum number validity
- Cannot be answered without applying the rule

### Prompt Variations
| Factor | Variations |
|--------|-----------|
| Principal quantum number | n = 1, 2, 3, 4 |
| Query l value | l = 0, 1, 2, 3 |
| Valid/invalid balance | 50% valid, 50% invalid |
| System | Hydrogen, "an atom", "a one-electron system" |

### Intervention Tests
1. **Ablate l<n constraint features:** Model loses ability to check validity
2. **Swap n-value features:** Replace n=2 with n=3 → l=2 becomes allowed

---

## Additional Behaviour 3: Work Sign in Thermodynamic Process (Type D)

**Replaces:** D3 (Charge conservation—factual recall)

**Prompt:** "During an isothermal compression of an ideal gas, is work done on or by the gas?"
**Expected answer:** `on` (or "on the gas")

### Core Latent Variables
- Process type: compression vs expansion
- Volume change sign: ΔV < 0 for compression
- Work definition: W = ∫P dV
- Sign determination: ΔV < 0 → W < 0 → work done ON gas

### Computational Pathway
```
Isothermal compression → ΔV < 0
W = ∫P dV → W < 0 (physics convention: W is work BY gas)
W < 0 → Work done ON gas
```

### Why Stronger
- Tests sign reasoning with physical meaning
- Clear latent: volume change sign → work sign
- Not just factual recall
- Gating/classification based on process type

### Prompt Variations
| Factor | Variations |
|--------|-----------|
| Process type | Compression (on), expansion (by), constant volume (zero) |
| Thermal condition | Isothermal, adiabatic, isobaric |
| System | Ideal gas, "a gas", "the system" |
| Question format | "On or by", "positive or negative", "increases or decreases system energy" |

### Intervention Tests
1. **Ablate process-type features:** Model loses compression→work-on mapping
2. **Swap process features:** Replace "compression" with "expansion" → answer flips to "by"

---

# Summary Recommendations

| Behaviour | Recommendation | Reason |
|-----------|---------------|--------|
| **A1** Decay-type inference | **KEEP** | Clear latent, clean output, good interventions |
| **A2** Reference-frame propagation | **MODIFY** | Needs magnitude disambiguation |
| **A3** Spin conservation | **DROP** | Too trivial; replace with decay series |
| **B1** Selection rules | **KEEP** | Excellent constraint checking |
| **B2** Approximation regime | **KEEP** | Clean threshold latent |
| **B3** Partition function | **KEEP** | Good statistical mechanics test |
| **C1** Gauge equivalence | **KEEP** | Deep physics abstraction |
| **C2** Intensive/extensive | **KEEP** | Operational test, not terminology |
| **C3** Dimensional velocity | **MODIFY** | Too trivial; use complex expressions |
| **D1** Thermodynamic direction | **KEEP** | Clear gating structure |
| **D2** Relativistic causality | **KEEP** | With edge case controls |
| **D3** Charge conservation | **DROP** | Factual recall; replace with work sign |

## Final Behaviour Set (12 total)

**Type A (Latent propagation):**
1. A1: Decay-type inference ✓
2. A2-mod: Reference-frame with magnitudes ✓
3. A-new: Decay series endpoint ✓

**Type B (Constraint filtering):**
4. B1: Selection rules ✓
5. B2: Approximation regime ✓
6. B3: Partition function dominance ✓
7. B-new: Quantum number constraint ✓

**Type C (Abstraction):**
8. C1: Gauge equivalence ✓
9. C2: Intensive/extensive operational ✓
10. C3-mod: Complex dimensional analysis ✓

**Type D (Gating):**
11. D1: Thermodynamic directionality ✓
12. D2: Relativistic causality ✓
13. D-new: Work sign in compression ✓

---

*Document version: 1.0*
*Created: 2025-02*
