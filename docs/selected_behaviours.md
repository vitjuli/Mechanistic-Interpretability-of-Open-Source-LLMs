# Selected Behaviours for Circuit Analysis

8 behaviours (2 per type) selected for implementation based on feasibility, clear latent structure, and physics interest.

---

# Type A: Latent-State Propagation

---

## A1. Decay-Type Inference

**Prompt:** "A nucleus emits a particle with charge +2e and mass number 4. The decay type is:"
**Target:** `alpha`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Emitted charge | +2e → Z=2 |
| Emitted mass number | A=4 |
| Particle signature | (Z=2, A=4) → He-4 nucleus |
| Decay classification | Alpha decay |

### Prompt-Family Generator (100 prompts)

```python
DECAY_DATA = {
    "alpha": {"charge": "+2e", "mass": 4, "alt": ["helium nucleus", "doubly charged particle with 4 nucleons"]},
    "beta_minus": {"charge": "-1e", "mass": 0, "alt": ["electron", "negatively charged particle with no mass number"]},
    "beta_plus": {"charge": "+1e", "mass": 0, "alt": ["positron", "positively charged particle with no mass number"]},
    "gamma": {"charge": "0", "mass": 0, "alt": ["photon", "electromagnetic radiation"]},
}

TEMPLATES = [
    "A nucleus emits a particle with charge {charge} and mass number {mass}. The decay type is:",
    "A radioactive atom releases a {alt}. This is called:",
    "An unstable nucleus ejects a particle with Z={z} and A={mass}. The decay is:",
    "During radioactive decay, a particle with charge {charge} is emitted. The decay type is:",
]
```

**Controlled factors:**
1. Property specification: both, charge only, mass only, descriptive
2. Numerical format: "+2e" vs "charge 2" vs "doubly positive"
3. Decay type: alpha, beta⁻, beta⁺, gamma (balanced)
4. Template variation: 4 templates

### Evaluation
- **Target tokens:** ` alpha`, ` beta`, ` gamma`
- **Metric:** Exact match accuracy
- **Baseline threshold:** 80%

### Interventions

**1. Ablate charge-encoding features**
- Target: Features in layers 12-20 encoding charge magnitude
- Method: Zero top-5 attributed features for charge
- Expected: Accuracy drops to ~25% (random among 4 types)

**2. Swap particle-property features**
- Target: Features encoding "+2e"
- Method: Patch from beta-minus prompt (charge -1e)
- Expected: Output flips from "alpha" to "beta"

### Negative Controls
1. "A nucleus emits light. The decay type is:" → gamma (no charge/mass needed)
2. "A nucleus with 92 protons exists. The decay type is:" → Nonsensical (no emission described)

---

## A2. Radioactive Decay Series Endpoint

**Prompt:** "Uranium-238 undergoes a series of 8 alpha decays and 6 beta-minus decays. The final element is:"
**Target:** `lead` (or `Pb`)

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Initial state | (Z=92, A=238) |
| Alpha effect | ΔZ=-2, ΔA=-4 per decay |
| Beta⁻ effect | ΔZ=+1, ΔA=0 per decay |
| Accumulated Z | 92 - 8×2 + 6×1 = 82 |
| Accumulated A | 238 - 8×4 = 206 |
| Element lookup | Z=82 → Lead |

### Computational Chain
```
U-238 (92, 238) → [8α] → (76, 206) → [6β⁻] → (82, 206) → Pb-206
```

### Prompt-Family Generator (80 prompts)

```python
SERIES_DATA = [
    {"start": "Uranium-238", "Z": 92, "A": 238, "alpha": 8, "beta": 6, "end": "lead", "end_Z": 82},
    {"start": "Thorium-232", "Z": 90, "A": 232, "alpha": 6, "beta": 4, "end": "lead", "end_Z": 82},
    {"start": "Uranium-235", "Z": 92, "A": 235, "alpha": 7, "beta": 4, "end": "lead", "end_Z": 82},
]

TEMPLATES = [
    "{start} undergoes {alpha} alpha decays and {beta} beta-minus decays. The final element is:",
    "After {alpha} alpha and {beta} beta⁻ decays, {start} becomes:",
    "A {start} nucleus emits {alpha} alpha particles and {beta} electrons. It transforms into:",
]
```

**Controlled factors:**
1. Starting isotope: U-238, Th-232, U-235
2. Number of alpha decays: 6, 7, 8
3. Number of beta decays: 4, 5, 6
4. Question format: element name vs atomic number vs isotope

### Evaluation
- **Target tokens:** ` lead`, ` Pb`, ` 82`
- **Metric:** Exact match (element name or symbol)
- **Baseline threshold:** 75%

### Interventions

**1. Ablate alpha-effect features**
- Target: Features encoding ΔZ=-2 per alpha
- Method: Zero ablation
- Expected: Final Z calculation wrong; element incorrect

**2. Swap decay-count features**
- Target: Features encoding "8 alpha"
- Method: Patch "6 alpha" features
- Expected: Final Z changes (92-12+6=86 vs 92-16+6=82); output changes to radon

### Negative Controls
1. "Uranium-238 exists. The element is:" → uranium (no decay, trivial)
2. "A nucleus undergoes decay. The final element is:" → Incomplete (no numbers)

---

# Type B: Candidate Set Selection

---

## B1. Selection Rule Allow/Forbid

**Prompt:** "Is an electric dipole transition from an s-state to another s-state allowed?"
**Target:** `no`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Initial l-value | s-state → l=0 |
| Final l-value | s-state → l=0 |
| Δl computation | 0 - 0 = 0 |
| E1 selection rule | Δl = ±1 required |
| Gate output | Δl=0 ≠ ±1 → forbidden |

### Prompt-Family Generator (80 prompts)

```python
ORBITALS = {"s": 0, "p": 1, "d": 2, "f": 3}
TRANSITIONS = {"E1": {"rule": "delta_l_pm1", "allowed_delta": [1, -1]},
               "M1": {"rule": "delta_l_0", "allowed_delta": [0]},
               "E2": {"rule": "delta_l_0_pm2", "allowed_delta": [0, 2, -2]}}

TEMPLATES = [
    "Is an {trans_type} transition from an {init}-state to a {final}-state allowed?",
    "For {trans_type} radiation, can an electron transition from l={l_init} to l={l_final}?",
    "Is the {init} → {final} transition allowed for electric dipole radiation?",
]

def is_allowed(init, final, trans_type):
    delta_l = ORBITALS[final] - ORBITALS[init]
    return delta_l in TRANSITIONS[trans_type]["allowed_delta"]
```

**Controlled factors:**
1. Initial state: s, p, d, f
2. Final state: s, p, d, f
3. Transition type: E1 (Δl=±1), M1 (Δl=0), E2 (Δl=0,±2)
4. Notation: "s-state" vs "l=0" vs "spherically symmetric"
5. Balance: 50% allowed, 50% forbidden

### Key Test Matrix (E1 transitions)
| Initial | Final | Δl | E1 Allowed? |
|---------|-------|-----|-------------|
| s | s | 0 | no |
| s | p | 1 | yes |
| s | d | 2 | no |
| p | s | -1 | yes |
| p | p | 0 | no |
| p | d | 1 | yes |
| d | p | -1 | yes |
| d | f | 1 | yes |

### Evaluation
- **Target tokens:** ` yes`, ` no`
- **Metric:** Accuracy on balanced set
- **Baseline threshold:** 85%

### Interventions

**1. Ablate Δl-computation features**
- Target: Features encoding l-value difference
- Method: Zero ablation
- Expected: Accuracy → 50% (random)

**2. Swap orbital features**
- Target: Features encoding "s-state" (l=0)
- Method: Patch "p-state" (l=1) features
- Expected: s→s becomes effectively p→s; answer flips to "yes"

### Negative Controls
1. "Is a transition from 2s to 2s allowed?" → Same as main but with principal quantum number (should still be "no")
2. "Is a transition from ground state to excited state allowed?" → Underspecified (no orbital info)

---

## B2. Approximation Regime Selection

**Prompt:** "An electron moves at 1% of the speed of light. Should its kinetic energy be computed classically or relativistically?"
**Target:** `classically`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Velocity value | v = 0.01c |
| Speed of light | c reference |
| v/c ratio | 0.01 |
| Threshold comparison | 0.01 ≪ 1 |
| Regime label | Classical |

### Prompt-Family Generator (100 prompts)

```python
SPEEDS = [
    {"v_c": 0.001, "desc": "0.1%", "regime": "classical"},
    {"v_c": 0.01, "desc": "1%", "regime": "classical"},
    {"v_c": 0.05, "desc": "5%", "regime": "classical"},
    {"v_c": 0.1, "desc": "10%", "regime": "classical"},  # borderline
    {"v_c": 0.3, "desc": "30%", "regime": "relativistic"},
    {"v_c": 0.5, "desc": "50%", "regime": "relativistic"},
    {"v_c": 0.9, "desc": "90%", "regime": "relativistic"},
    {"v_c": 0.99, "desc": "99%", "regime": "relativistic"},
]

PARTICLES = ["An electron", "A proton", "A particle", "An object"]

TEMPLATES = [
    "{particle} moves at {desc} of the speed of light. Should its kinetic energy be computed classically or relativistically?",
    "{particle} has velocity {v_c}c. The appropriate treatment for its momentum is:",
    "At {desc} of c, should we use Newtonian or relativistic mechanics for {particle}?",
]
```

**Controlled factors:**
1. v/c ratio: 0.1% to 99% (8 values spanning boundary)
2. Particle type: electron, proton, generic
3. Speed format: percentage, decimal, absolute value
4. Property asked: kinetic energy, momentum, mass
5. Phrasing: "classical/relativistic" vs "Newtonian/Einstein" vs "non-relativistic/relativistic"

### Evaluation
- **Target tokens:** ` classically`, ` relativistically`
- **Metric:** Accuracy (expect sharp transition around v/c ~ 0.1-0.3)
- **Baseline threshold:** 85%

### Interventions

**1. Ablate v/c comparison features**
- Target: Features encoding velocity-to-c ratio
- Method: Zero ablation
- Expected: Model defaults to one regime; accuracy → 50%

**2. Swap velocity features**
- Target: Features encoding "1%"
- Method: Patch "90%" features
- Expected: Answer flips from "classically" to "relativistically"

### Negative Controls
1. "An electron exists. Classical or relativistic?" → No velocity given
2. "Light travels at c. Classical or relativistic?" → Light itself (always relativistic, but different sense)

---

# Type C: Abstraction vs Surface Form

---

## C1. Gauge Equivalence

**Prompt:** "Do two vector potentials that differ by the gradient of a scalar function produce the same magnetic field?"
**Target:** `yes`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Vector potential A | Field representation |
| Gauge transformation | A → A + ∇f |
| Curl operation | B = ∇ × A |
| Identity | ∇ × (∇f) = 0 |
| Invariance | B₁ = B₂ |

### Computational Chain
```
A₂ = A₁ + ∇f
B₂ = ∇ × A₂ = ∇ × (A₁ + ∇f) = ∇ × A₁ + ∇ × ∇f = B₁ + 0 = B₁
```

### Prompt-Family Generator (60 prompts)

```python
TRANSFORMATIONS = [
    {"desc": "the gradient of a scalar function", "invariant": True},
    {"desc": "the gradient of a scalar field", "invariant": True},
    {"desc": "∇φ for some scalar φ", "invariant": True},
    {"desc": "a curl-free vector field", "invariant": True},
    {"desc": "a constant vector", "invariant": True},  # Also gauge-like
    {"desc": "the curl of another vector field", "invariant": False},  # Changes B
]

FIELDS = [
    {"name": "magnetic field", "symbol": "B", "depends_on": "curl of A"},
    {"name": "electric field", "symbol": "E", "depends_on": "A and φ"},  # More complex
]

TEMPLATES = [
    "Do two vector potentials that differ by {trans} produce the same {field}?",
    "If A₂ = A₁ + {trans}, is the {field} unchanged?",
    "Two potentials differ by {trans}. Same {field}?",
]
```

**Controlled factors:**
1. Transformation type: gradient (yes), curl (no), constant (yes)
2. Field asked: magnetic (simpler), electric (more complex)
3. Notation: words vs symbols vs mixed
4. Phrasing: "same field" vs "unchanged" vs "equivalent"

### Evaluation
- **Target tokens:** ` yes`, ` no`
- **Metric:** Accuracy on transformation types
- **Baseline threshold:** 80%

### Interventions

**1. Ablate gauge-invariance features**
- Target: Features encoding ∇×∇=0 identity
- Method: Zero ablation
- Expected: Model uncertain; may say "no" for gradient transformations

**2. Swap transformation features**
- Target: Features encoding "gradient"
- Method: Patch "curl" features
- Expected: Answer flips from "yes" to "no" (curl of vector changes B)

### Negative Controls
1. "Do two identical vector potentials produce the same magnetic field?" → Trivially yes
2. "Does the magnetic field depend on the vector potential?" → Different question (yes, but via curl)

---

## C2. Intensive vs Extensive (Operational)

**Prompt:** "If you combine two identical samples of water at the same temperature, does the temperature of the combined sample change?"
**Target:** `no`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Property type | Temperature |
| Operation | Combine (double amount) |
| Scaling behavior | Intensive: unchanged under scaling |
| Classification | T is intensive → no change |

### Prompt-Family Generator (80 prompts)

```python
PROPERTIES = {
    # Intensive (no change)
    "temperature": {"type": "intensive", "answer": "no"},
    "pressure": {"type": "intensive", "answer": "no"},
    "density": {"type": "intensive", "answer": "no"},
    "concentration": {"type": "intensive", "answer": "no"},
    # Extensive (changes/doubles)
    "mass": {"type": "extensive", "answer": "yes"},
    "volume": {"type": "extensive", "answer": "yes"},
    "energy": {"type": "extensive", "answer": "yes"},
    "entropy": {"type": "extensive", "answer": "yes"},
}

OPERATIONS = [
    "combine two identical samples",
    "double the amount",
    "merge two equal portions",
    "put together two identical quantities",
]

TEMPLATES = [
    "If you {op} of {substance} at the same {prop}, does the {prop} of the result change?",
    "Two identical samples of {substance} are combined. Does the total {prop} equal twice the original?",
    "After {op} of {substance}, is the {prop} different?",
]
```

**Controlled factors:**
1. Property: 4 intensive, 4 extensive (balanced)
2. Operation: combine, double, merge
3. Substance: water, gas, metal, generic
4. Question phrasing: "change" vs "double" vs "different"

### Evaluation
- **Target tokens:** ` yes`, ` no`
- **Metric:** Accuracy on intensive/extensive classification
- **Baseline threshold:** 85%

### Interventions

**1. Ablate intensive/extensive features**
- Target: Features encoding scaling behavior
- Method: Zero ablation
- Expected: Model guesses; accuracy → 50%

**2. Swap property features**
- Target: Features encoding "temperature"
- Method: Patch "mass" features
- Expected: Answer flips from "no" to "yes"

### Negative Controls
1. "Does water have temperature?" → Not a scaling question
2. "If you heat water, does the temperature change?" → Different operation (not doubling)

---

# Type D: Gating / Physical Consistency

---

## D1. Thermodynamic Directionality

**Prompt:** "Can heat spontaneously flow from a colder body to a hotter body without external work?"
**Target:** `no`

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Temperature comparison | T_cold < T_hot |
| Flow direction | Cold → hot |
| Spontaneity check | No work input |
| 2nd Law constraint | ΔS_universe ≥ 0 |
| Gate output | Violates 2nd Law → forbidden |

### Prompt-Family Generator (80 prompts)

```python
DIRECTIONS = [
    {"from": "colder", "to": "hotter", "spontaneous": False},
    {"from": "hotter", "to": "colder", "spontaneous": True},
    {"from": "cold", "to": "hot", "spontaneous": False},
    {"from": "hot", "to": "cold", "spontaneous": True},
]

CONDITIONS = [
    "without external work",
    "spontaneously",
    "naturally",
    "in an isolated system",
    "without any energy input",
]

TEMPLATES = [
    "Can heat {cond} flow from a {from_t} body to a {to_t} body?",
    "Does heat ever {cond} move from {from_t} to {to_t} objects?",
    "Is it possible for thermal energy to transfer {cond} from a {from_t} region to a {to_t} region?",
]
```

**Controlled factors:**
1. Direction: cold→hot (no), hot→cold (yes)
2. Spontaneity qualifier: 5 variations
3. System specification: isolated, closed, none
4. Energy term: heat, thermal energy, internal energy
5. Object term: body, object, region, system

### Evaluation
- **Target tokens:** ` yes`, ` no`
- **Metric:** Accuracy (balanced directions)
- **Baseline threshold:** 90%

### Interventions

**1. Ablate 2nd-law gate features**
- Target: Features encoding spontaneity/entropy constraint
- Method: Zero ablation
- Expected: Model loses directionality; accuracy → 50%

**2. Swap direction features**
- Target: Features encoding "colder to hotter"
- Method: Patch "hotter to colder" features
- Expected: Answer flips from "no" to "yes"

### Negative Controls
1. "Can a refrigerator move heat from cold to hot?" → Yes (with work input)
2. "Does heat exist?" → Not a directionality question

---

## D2. Work Sign in Thermodynamic Process

**Prompt:** "During an isothermal compression of an ideal gas, is work done on the gas or by the gas?"
**Target:** `on` (or "on the gas")

### Latent Variables
| Variable | Representation |
|----------|---------------|
| Process type | Compression |
| Volume change | ΔV < 0 |
| Work integral | W = ∫P dV |
| Sign determination | ΔV < 0 → W_by_gas < 0 |
| Classification | Work done ON gas |

### Computational Chain
```
Compression → Volume decreases → ΔV < 0
W_by_gas = ∫P dV < 0 (negative work by gas)
Negative work BY gas = Positive work ON gas
```

### Prompt-Family Generator (80 prompts)

```python
PROCESSES = [
    {"name": "compression", "dV": "negative", "work_on": True},
    {"name": "expansion", "dV": "positive", "work_on": False},
    {"name": "compressed", "dV": "negative", "work_on": True},
    {"name": "expanded", "dV": "positive", "work_on": False},
]

THERMAL_CONDITIONS = [
    "isothermal",
    "adiabatic",
    "isobaric",
    "",  # unspecified
]

SYSTEMS = ["an ideal gas", "a gas", "the system", "a piston-cylinder system"]

TEMPLATES = [
    "During {thermal} {process} of {system}, is work done on the gas or by the gas?",
    "When {system} undergoes {thermal} {process}, work is done:",
    "In {thermal} {process}, the gas has work done:",
]
```

**Controlled factors:**
1. Process: compression (on), expansion (by)
2. Thermal condition: isothermal, adiabatic, isobaric, unspecified
3. System: ideal gas, generic gas, piston system
4. Question format: "on or by" vs "on the gas / by the gas"

### Evaluation
- **Target tokens:** ` on`, ` by`
- **Metric:** Accuracy (balanced processes)
- **Baseline threshold:** 85%

### Interventions

**1. Ablate process-type features**
- Target: Features encoding compression vs expansion
- Method: Zero ablation
- Expected: Model guesses; accuracy → 50%

**2. Swap process features**
- Target: Features encoding "compression"
- Method: Patch "expansion" features
- Expected: Answer flips from "on" to "by"

### Negative Controls
1. "Is work done during compression?" → Yes, but doesn't specify direction
2. "What is the work in an isochoric process?" → Zero (no volume change)

---

# Implementation Summary

| ID | Behaviour | Type | Target | Complexity |
|----|-----------|------|--------|------------|
| A1 | Decay-type inference | A | `alpha`/`beta`/`gamma` | Low |
| A2 | Decay series endpoint | A | `lead`/element name | Medium |
| B1 | Selection rule allow/forbid | B | `yes`/`no` | Low |
| B2 | Approximation regime | B | `classically`/`relativistically` | Low |
| C1 | Gauge equivalence | C | `yes`/`no` | Medium |
| C2 | Intensive vs extensive | C | `yes`/`no` | Low |
| D1 | Thermodynamic direction | D | `yes`/`no` | Low |
| D2 | Work sign | D | `on`/`by` | Low |

## Recommended Implementation Order

**Phase 1 (Simplest, binary outputs):**
1. D1 - Thermodynamic directionality
2. B1 - Selection rules
3. C2 - Intensive/extensive

**Phase 2 (Binary with physics depth):**
4. B2 - Approximation regime
5. D2 - Work sign
6. C1 - Gauge equivalence

**Phase 3 (Multi-token outputs):**
7. A1 - Decay-type inference
8. A2 - Decay series endpoint

---

*Selected: 2025-02*
*Total: 8 behaviours (2 per type)*
