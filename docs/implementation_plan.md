# Implementation Plan: Complete Pipeline Documentation

## Overview

This document provides a comprehensive step-by-step description of the entire pipeline, from prompt generation to attribution graphs and interventions. Each step includes mathematical formulas, function signatures, algorithmic details, and data flow.

**Reference:** This work reproduces methods from Anthropic's "On the Biology of a Large Language Model" (Lindsey, Gurnee, et al., 2025) on Qwen3 models using pre-trained transcoders from the [circuit-tracer](https://github.com/safety-research/circuit-tracer) project.

---

## Step 1: Prompt Generation

**Script:** `scripts/01_generate_prompts.py`

### Purpose
Generate synthetic prompts for 4 behaviours where circuits are theoretically likely to exist.

### Behaviours

#### 1.1 Grammar Agreement (`generate_grammar_agreement_prompts`)

**Task:** Subject-verb number agreement

**Prompt Structure:**
```
prompt = "The {subject}"
correct_answer = " {verb_singular}" if subject is singular else " {verb_plural}"
incorrect_answer = " {verb_plural}" if subject is singular else " {verb_singular}"
```

**Data:**
- Subjects: 20 singular nouns ("cat", "dog", ...) + 20 plural forms ("cats", "dogs", ...)
- Verb pairs: 15 pairs (("is", "are"), ("was", "were"), ("sits", "sit"), ...)
- Continuations: 10 phrases ("in the room", "near the door", ...)

**Combinatorics:**
```
Total combinations = 20 subjects × 2 numbers × 15 verb_pairs × 2 continuations = 1200
```

**Output:** 80 train + 20 test prompts (randomly sampled)

#### 1.2 Factual Recall (`generate_factual_recall_prompts`)

**Task:** Country-capital knowledge retrieval

**Prompt Templates:**
```python
templates = [
    ("The capital of {country} is", " {answer}"),
    ("The capital city of {country} is", " {answer}"),
    ("{country}'s capital is", " {answer}"),
    ("What is the capital of {country}? The answer is", " {answer}"),
]
```

**Token Length Matching:**
For each (country, correct_city, wrong_cities) triple, select incorrect answers prioritizing matched token lengths:
```python
def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

# Sort wrong cities by |len(correct) - len(wrong)|, prefer length_diff == 0
```

**Data:** 15 country-capital facts with 3 distractor cities each

#### 1.3 Sentiment Continuation (`generate_sentiment_continuation_prompts`)

**Task:** Sentiment-consistent word completion

**Prompt Structure:**
```python
# Positive context
prompt = "The movie was absolutely"
correct_answer = " wonderful"  # positive word
incorrect_answer = " terrible"  # negative word

# Negative context
prompt = "The movie was completely"
correct_answer = " terrible"  # negative word
incorrect_answer = " wonderful"  # positive word
```

**Data:** 15 positive contexts + 15 negative contexts + 5 additional variants each

#### 1.4 Arithmetic (`generate_arithmetic_prompts`)

**Task:** Two-digit addition

**Prompt Templates:**
```python
templates = [
    ("{a} + {b} =", " {answer}"),
    ("What is {a} + {b}? The answer is", " {answer}"),
    ("Calculate: {a} + {b} =", " {answer}"),
    ("The sum of {a} and {b} is", " {answer}"),
]
```

**Error Types for Distractors:**
```python
wrong_answers = []
wrong_answers.append(correct_sum + 10)  # Off-by-10 (carry mistake)
wrong_answers.append(correct_sum - 10)
wrong_answers.append(correct_sum + 1)   # Off-by-1
wrong_answers.append(correct_sum - 1)
wrong_answers.append(correct_sum + 9)   # Digit swap type
wrong_answers.append(correct_sum - 9)
```

**Operand Ranges:** (10-50), (20-80), (30-99)

### Output Format

JSONL files saved to `data/prompts/{behaviour}_{split}.jsonl`:
```json
{
    "prompt": "The cat",
    "correct_answer": " is",
    "incorrect_answer": " are",
    "subject": "cat",
    "number": "singular",
    ...
}
```

### Reproducibility
```python
random.seed(42)  # Fixed seed for prompt_generation
```

---

## Step 2: Baseline Evaluation

**Script:** `scripts/02_run_baseline.py`

### Purpose
Validate that the model can perform each behaviour with sufficient clarity before circuit analysis.

### Core Function: `evaluate_behaviour`

```python
def evaluate_behaviour(
    model: ModelWrapper,
    prompts: List[Dict],
    behaviour_name: str,
    min_score_diff: float = 2.0,  # Normalized logprob diff threshold
) -> pd.DataFrame
```

### Mathematical Formulation

#### Log Probability Computation

For a prompt $p$ and target sequence $t = (t_1, t_2, \ldots, t_n)$:

$$\log P(t \mid p) = \sum_{i=1}^{n} \log P(t_i \mid p, t_1, \ldots, t_{i-1})$$

**Implementation (teacher forcing):**
```python
# In model_utils.py: get_sequence_log_probs()
log_softmax = F.log_softmax(logits, dim=-1)
for j, target_token_id in enumerate(target_ids):
    token_position = prompt_len + j
    logit_position = token_position - 1  # Causal shift
    token_log_prob = log_softmax[logit_position, target_token_id]
    total_log_prob += token_log_prob
```

#### Normalized Log Probability Difference

To avoid length bias, we normalize by token count:

$$\Delta_{\text{norm}} = \frac{\log P(t_{\text{correct}} \mid p)}{|t_{\text{correct}}|} - \frac{\log P(t_{\text{incorrect}} \mid p)}{|t_{\text{incorrect}}|}$$

**Success Criterion:**
$$\text{success} = \Delta_{\text{norm}} > \tau$$

where $\tau$ is the per-behaviour threshold (typically 1.5-2.0).

#### Accuracy Computation
$$\text{accuracy} = \frac{\sum_{i=1}^{N} \mathbb{1}[\Delta_{\text{norm}}^{(i)} > \tau]}{N}$$

### Decision Threshold

| Behaviour | min_score_diff ($\tau$) | success_threshold |
|-----------|------------------------|-------------------|
| Grammar Agreement | 2.0 | 80% |
| Factual Recall | 2.0 | 80% |
| Sentiment Continuation | 1.5 | 75% |
| Arithmetic | 2.0 | 80% |

### Algorithm

```
1. For each prompt p in prompts:
   a. Tokenize p, correct_answer, incorrect_answer
   b. Concatenate: full_seq = p + answer (for both correct/incorrect)
   c. Forward pass through model
   d. Extract log_probs at teacher-forced positions
   e. Sum log_probs for each answer
   f. Normalize by token length
   g. Compute normalized difference
   h. Mark success if diff > threshold

2. Compute aggregate statistics:
   - Overall accuracy
   - Sign accuracy (correct > incorrect)
   - Mean/median/std of normalized diffs
   - Token length statistics

3. Decision: PASS if accuracy >= success_threshold
```

### Output
- `data/results/baseline_{behaviour}_{split}.csv`: Per-prompt results
- `data/results/baseline_metrics_{split}.json`: Aggregate metrics
- `data/results/figures/baseline_*.png`: Visualizations

---

## Step 3: Feature Extraction with Transcoders

**Script:** `scripts/04_extract_transcoder_features.py`

### Purpose
Extract interpretable features from MLP computations using pre-trained per-layer transcoders (PLTs).

### Transcoder Architecture

A per-layer transcoder approximates the MLP computation:

$$\text{MLP}(x) \approx W_{\text{dec}} \cdot \sigma(W_{\text{enc}} \cdot x + b_{\text{enc}}) + b_{\text{dec}}$$

where:
- $x \in \mathbb{R}^{d_{\text{model}}}$: MLP input (residual stream)
- $W_{\text{enc}} \in \mathbb{R}^{d_{\text{transcoder}} \times d_{\text{model}}}$: Encoder weights
- $W_{\text{dec}} \in \mathbb{R}^{d_{\text{transcoder}} \times d_{\text{model}}}$: Decoder weights
- $\sigma$: Activation function (JumpReLU)
- $d_{\text{transcoder}} \gg d_{\text{model}}$ (sparse overcomplete)

### JumpReLU Activation

$$\text{JumpReLU}(x; \theta) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{otherwise} \end{cases}$$

**Properties:**
- Creates hard gap between zero and smallest non-zero activation
- Threshold $\theta$ is learnable per feature
- Produces cleaner sparsity than ReLU ($\theta = 0$)

**Backward Pass (Straight-Through Estimator):**
```python
# Gradient w.r.t. x
x_grad = (x > threshold) * grad_output

# Gradient w.r.t. threshold (rectangle approximation)
threshold_grad = -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
```

### Feature Extraction Pipeline

#### Step 3.1: Capture MLP Inputs

```python
def capture_mlp_inputs(
    model: ModelWrapper,
    prompts: List[Dict],
    layer_indices: List[int],
    batch_size: int = 8,
    token_positions: str = "last_5",
) -> Dict[int, torch.Tensor]
```

**Process:**
1. Forward pass with `output_hidden_states=True`
2. Extract `hidden_states[layer_idx]` (pre-MLP residual stream)
3. Select token positions (last 5 or all)
4. Concatenate across batches

**Output:** `{layer_idx: tensor(n_samples, d_model)}`

#### Step 3.2: Encode Through Transcoders

```python
def extract_features(
    transcoder_set: TranscoderSet,
    mlp_inputs: Dict[int, torch.Tensor],
    top_k: int = 50,
) -> Dict[int, Dict]
```

**Encoding:**
```python
# In single_layer_transcoder.py
def encode(self, input_acts, apply_activation_function=True):
    pre_acts = F.linear(input_acts, self.W_enc, self.b_enc)
    if apply_activation_function:
        return self.activation_function(pre_acts)  # JumpReLU
    return pre_acts
```

**Feature Statistics:**
```python
# Active features per sample
active_mask = features > 0
active_features = set(torch.where(active_mask.any(dim=0))[0])

# Feature frequencies
feature_frequencies = active_mask.float().mean(dim=0)

# Top-k features per sample
top_k_values, top_k_indices = torch.topk(features, k=top_k, dim=1)
```

### Available Transcoders

| Model Size | Repository | Layers | Hidden Size |
|------------|------------|--------|-------------|
| 0.6B | mwhanna/qwen3-0.6b-transcoders-lowl0 | 28 | 1024 |
| 1.7B | mwhanna/qwen3-1.7b-transcoders-lowl0 | 28 | 1536 |
| 4B | mwhanna/qwen3-4b-transcoders | 36 | 2560 |
| 8B | mwhanna/qwen3-8b-transcoders | 36 | 4096 |
| 14B | mwhanna/qwen3-14b-transcoders-lowl0 | 40 | 5120 |

### Output Files

```
data/results/transcoder_features/
├── layer_10/
│   ├── {behaviour}_{split}_top_k_indices.npy
│   ├── {behaviour}_{split}_top_k_values.npy
│   ├── {behaviour}_{split}_feature_frequencies.npy
│   ├── {behaviour}_{split}_full_activations.npy  # Optional
│   └── {behaviour}_{split}_layer_meta.json
├── layer_11/
│   └── ...
└── {behaviour}_{split}_summary.json
```

---

## Step 4: Attribution Graph Construction

**Script:** `scripts/06_build_attribution_graph.py`

### Purpose
Build causal dependency graphs showing how transcoder features contribute to model outputs.

### Attribution Method

#### Gradient-Based Attribution

For a target token with logit $\ell_{\text{target}}$, the attribution of feature $f_i$ at layer $l$ is:

$$A_{l,i} = f_i^{(l)} \cdot \left| \frac{\partial \ell_{\text{target}}}{\partial f_i^{(l)}} \right|$$

This is the **activation × gradient** formula, measuring how much each feature activation contributes to the output.

**Implementation:**
```python
def compute_output_attribution(self, prompt, target_token):
    # Forward pass
    outputs = model(**inputs, output_hidden_states=True)
    target_logit = logits[0, -1, target_id]

    for layer in layers:
        layer_act = hidden_states[layer][:, -1, :].clone()
        layer_act.requires_grad_(True)

        # Encode to features
        features = transcoder.encode(layer_act)

        # Compute gradient
        grad = torch.autograd.grad(target_logit, layer_act)[0]

        # Project gradient into feature space
        grad_features = transcoder.encode(grad, apply_activation_function=False)

        # Attribution = activation × |gradient|
        attribution = features * grad_features.abs()
```

#### Differential Attribution

For binary classification (correct vs incorrect), we compute:

$$A_{l,i}^{\text{diff}} = A_{l,i}^{\text{correct}} - A_{l,i}^{\text{incorrect}}$$

This highlights features that distinguish the correct answer.

### Virtual Weights for Cross-Layer Attribution

Since PLTs operate on single layers, we approximate cross-layer effects:

$$W_{\text{virtual}}^{(l \to l')} = W_{\text{enc}}^{(l')} \cdot (W_{\text{dec}}^{(l)})^T$$

This matrix describes how features at layer $l$ influence features at layer $l'$ through the linear pathway.

```python
def compute_virtual_weights(self, source_layer, target_layer):
    source_tc = self.transcoder_set[source_layer]
    target_tc = self.transcoder_set[target_layer]
    return target_tc.W_enc @ source_tc.W_dec.T
```

### Graph Construction

#### Per-Prompt Graph

```python
def build_graph_for_prompt(self, prompt, correct_token, incorrect_token) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add input/output nodes
    G.add_node("input", type="input")
    G.add_node(f"output_{correct_token}", type="output")
    G.add_node(f"output_{incorrect_token}", type="output")

    # Add feature nodes with top-k attribution
    for layer in layers:
        diff_attr = correct_attr[layer] - incorrect_attr[layer]
        top_k_values, top_k_indices = torch.topk(diff_attr.abs(), k=top_k_edges)

        for feat_idx, attr_val in zip(top_k_indices, top_k_values):
            if attr_val > attribution_threshold:
                feat_id = f"L{layer}_F{feat_idx}"
                G.add_node(feat_id, type="feature", layer=layer, ...)
                G.add_edge("input", feat_id, weight=attr_val)
                G.add_edge(feat_id, f"output_{correct_token}", weight=...)
```

#### Aggregated Graph

Aggregate across multiple prompts, keeping features that appear frequently:

```python
def aggregate_graphs(self, prompts, n_prompts=20, min_frequency=0.2):
    feature_stats = defaultdict(lambda: {
        "count": 0,
        "total_attr_correct": 0.0,
        "total_attr_incorrect": 0.0,
    })

    for prompt in prompts[:n_prompts]:
        G = self.build_graph_for_prompt(prompt, ...)
        for node, data in G.nodes(data=True):
            if data.get("type") == "feature":
                feat_key = (data["layer"], data["feature_idx"])
                feature_stats[feat_key]["count"] += 1
                ...

    # Keep features appearing in >= 20% of prompts
    min_count = int(n_prompts * min_frequency)
    G_agg = nx.DiGraph()
    for (layer, feat_idx), stats in feature_stats.items():
        if stats["count"] >= min_count:
            G_agg.add_node(f"L{layer}_F{feat_idx}", ...)
```

### Output Files

```
data/results/attribution_graphs/{behaviour}/
├── attribution_graph_{split}.graphml
└── attribution_graph_{split}.json
```

**JSON Format:**
```json
{
    "nodes": [
        {"id": "L15_F1234", "type": "feature", "layer": 15, "feature_idx": 1234, ...}
    ],
    "edges": [
        {"source": "input", "target": "L15_F1234", "weight": 0.05}
    ],
    "metadata": {...}
}
```

---

## Step 5: Intervention Experiments

**Script:** `scripts/07_run_interventions.py`

### Purpose
Validate circuit hypotheses through causal perturbations.

### Intervention Types

#### 5.1 Feature Ablation

**Zero Ablation:** Set feature activations to zero
$$f_i' = 0$$

**Inhibition:** Negate feature activations
$$f_i' = -\alpha \cdot f_i$$

where $\alpha$ is the inhibition factor (default: 1.0).

```python
def run_ablation_experiment(self, prompt, correct_token, incorrect_token,
                           layer, feature_indices, mode="zero"):
    # Get baseline
    baseline_diff = self.compute_logit_diff(prompt, correct_token, incorrect_token)

    # Encode to features
    features = transcoder.encode(layer_act)

    # Apply ablation
    if mode == "zero":
        features[:, feature_indices] = 0.0
    elif mode == "inhibit":
        features[:, feature_indices] = -inhibition_factor * features[:, feature_indices]

    # Decode back
    modified_act = transcoder.decode(features)

    # Measure effect
    effect_size = baseline_diff - intervened_diff
    relative_effect = effect_size / |baseline_diff|
```

**Effect Measurement:**
$$\Delta_{\text{effect}} = \Delta_{\text{baseline}} - \Delta_{\text{intervened}}$$

#### 5.2 Activation Patching

Swap features from source prompt into target prompt computation:

```python
def run_patching_experiment(self, source_prompt, target_prompt,
                           layer, feature_indices=None):
    # Get features from both prompts
    source_features = transcoder.encode(source_act)
    target_features = transcoder.encode(target_act)

    # Patch
    if feature_indices is None:
        patched_features = source_features  # Full layer
    else:
        patched_features = target_features.clone()
        patched_features[:, feature_indices] = source_features[:, feature_indices]
```

**Pairing Strategy:**
- Grammar: Singular ↔ Plural
- Sentiment: Positive ↔ Negative
- Arithmetic/Factual: Consecutive prompts

#### 5.3 Feature Importance Sweep

Correlate feature activations with logit differences across prompts:

```python
def run_feature_importance_sweep(self, prompts, layer, n_prompts=20):
    # Collect features and logit diffs
    feature_matrix = []  # (n_prompts, d_transcoder)
    logit_diffs = []

    for prompt in prompts[:n_prompts]:
        features = transcoder.encode(layer_act)
        feature_matrix.append(features)
        logit_diffs.append(compute_logit_diff(prompt))

    # Correlation per feature
    for feat_idx in range(d_transcoder):
        feat_acts = feature_matrix[:, feat_idx]
        correlation = np.corrcoef(feat_acts, logit_diffs)[0, 1]
```

### InterventionResult Dataclass

```python
@dataclass
class InterventionResult:
    prompt_idx: int
    prompt: str
    baseline_logit_diff: float
    intervened_logit_diff: float
    effect_size: float          # Δ_baseline - Δ_intervened
    relative_effect: float      # effect / |baseline|
    intervention_type: str      # "ablation_zero", "patching", etc.
    layer: int
    features_intervened: List[int]
    correct_token: str
    incorrect_token: str
```

### Output Files

```
data/results/interventions/{behaviour}/
├── intervention_ablation_{behaviour}.csv
├── intervention_ablation_{behaviour}_summary.json
├── intervention_patching_{behaviour}.csv
├── intervention_patching_{behaviour}_summary.json
└── feature_importance_layer_{layer}.csv
```

---

## Step 6: Logit Difference Metric

The logit difference is the primary metric throughout the pipeline:

$$\Delta \ell = \ell_{\text{correct}} - \ell_{\text{incorrect}}$$

where $\ell_{\text{token}} = \text{logits}[-1, \text{token\_id}]$ (logit at last position).

**Computation:**
```python
def compute_logit_diff(self, prompt, correct_token, incorrect_token):
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Last position

    correct_id = tokenizer.encode(correct_token, add_special_tokens=False)[0]
    incorrect_id = tokenizer.encode(incorrect_token, add_special_tokens=False)[0]

    return logits[correct_id].item() - logits[incorrect_id].item()
```

---

## Summary: Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Prompt Generation                                        │
│   Input: Config (behaviours, seeds)                              │
│   Output: data/prompts/{behaviour}_{train,test}.jsonl           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Baseline Evaluation                                      │
│   Input: Prompts + Model                                         │
│   Compute: Normalized log-prob differences                       │
│   Output: data/results/baseline_*.csv, metrics.json             │
│   Decision: PASS if accuracy ≥ threshold                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Feature Extraction                                       │
│   Input: Prompts + Model + Pre-trained Transcoders              │
│   Compute: MLP inputs → Transcoder encode → Features            │
│   Output: data/results/transcoder_features/                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Attribution Graphs                                       │
│   Input: Prompts + Model + Transcoders                          │
│   Compute: Gradient-based attribution (activation × gradient)   │
│   Output: data/results/attribution_graphs/                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Interventions                                            │
│   Input: Prompts + Model + Transcoders + Attribution Graphs     │
│   Compute: Ablation, Patching, Feature Importance               │
│   Output: data/results/interventions/                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Mathematical Summary

| Concept | Formula |
|---------|---------|
| Log probability | $\log P(t \mid p) = \sum_i \log P(t_i \mid p, t_{<i})$ |
| Normalized log-prob diff | $\Delta_{\text{norm}} = \frac{\log P(t_c \mid p)}{|t_c|} - \frac{\log P(t_w \mid p)}{|t_w|}$ |
| JumpReLU | $\sigma(x; \theta) = x \cdot \mathbb{1}[x > \theta]$ |
| Transcoder encode | $f = \sigma(W_{\text{enc}} \cdot x + b_{\text{enc}})$ |
| Transcoder decode | $\hat{y} = W_{\text{dec}} \cdot f + b_{\text{dec}}$ |
| Attribution | $A_i = f_i \cdot \|\nabla_{f_i} \ell\|$ |
| Differential attribution | $A_i^{\text{diff}} = A_i^{\text{correct}} - A_i^{\text{incorrect}}$ |
| Virtual weights | $W_{\text{virtual}}^{(l \to l')} = W_{\text{enc}}^{(l')} \cdot (W_{\text{dec}}^{(l)})^T$ |
| Logit difference | $\Delta \ell = \ell_{\text{correct}} - \ell_{\text{incorrect}}$ |
| Ablation effect | $\Delta_{\text{effect}} = \Delta_{\text{baseline}} - \Delta_{\text{intervened}}$ |

---

## Configuration Reference

### Seeds (Reproducibility)
- Prompt generation: 42
- Intervention sampling: 456
- PyTorch: 789

### Analysis Layers (4B model)
- Early: [0-4]
- Middle: [15-20]
- Late: [30-35]
- Default: [10-25]

### Thresholds
- Attribution edge threshold: 0.01
- Top-k edges per node: 10
- Feature frequency threshold: 20%
- Activation threshold: 0.1

---

*Last updated: 2025-02*
*Pipeline version: Transcoder-based (circuit-tracer integration)*
