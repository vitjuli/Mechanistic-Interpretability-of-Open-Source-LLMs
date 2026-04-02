
  ---
  Implementation Plan — Two Methodological Upgrades

  ---
  A. Current State

  A.1 Attribution (Script 06 — aggregate_graphs_per_prompt_union)

  The current pipeline does not compute gradients. Instead it uses a two-stage correlation proxy:

  Stage 1 — compute_beta() (scripts/06_build_attribution_graph.py:441):
  For each feature (layer, feat), computes a dataset-level OLS regression coefficient:
  beta = Cov(a_k, Δlogit) / Var(a_k)
  where Δlogit = logprob_correct − logprob_incorrect, computed over all N train prompts with a missing=0 model (features not in top-k treated as activation=0). This is a single scalar per feature,
  constant across all prompts.

  Stage 2 — per-prompt score (scripts/06_build_attribution_graph.py:395):
  For each prompt p, score each feature as:
  score_k^p = activation_k^p × beta_k
  This is called activation × beta in code and comments. The top-k features per prompt are selected by |score_k^p|. Union is taken across all prompts, with features kept if they appear in ≥ min_prompts
  prompts.

  Edge weights and node attributes:
  - Edge weight = mean_score_conditional = mean of (activation × beta) over prompts where feature is in top-k
  - Node stores: beta, beta_sign, specific_score = mean_abs × (1 − frequency), mean_abs_score_conditional

  What this is not: beta is a population-level proxy for the gradient. The actual per-prompt gradient ∂Δlogit / ∂a_k^p would depend on the specific forward pass for prompt p, not on a dataset average.
  The two are proportional only if the gradient is approximately constant across prompts (which it is not, in general).

  A.2 Multilingual Overlap / IoU (Script a_analyze_multilingual_circuits.py:245)

  Active-set definition: Feature k is in the active set for prompt p at layer L if it is among the top-50 features saved by script 04 (i.e., it has the 50th-largest or higher activation value). There is
   no explicit activation threshold — presence in the top-k list defines activity.

  Pooling: All EN prompts are pooled into one big union, all FR prompts into another:
  S_EN^L = ⋃_{p ∈ EN} {top-50 features of p at layer L}
  S_FR^L = ⋃_{p ∈ FR} {top-50 features of p at layer L}
  IoU_L = |S_EN^L ∩ S_FR^L| / |S_EN^L ∪ S_FR^L|

  No random baseline: There is no comparison to unrelated-prompt pairs. The 0.22–0.26 IoU values cannot currently be interpreted as meaningful because we don't know whether IoU_unrelated ≈ 0.22 (the
  signal is noise) or IoU_unrelated ≈ 0.10 (the signal is real).

  Top-k sensitivity: Because S_EN is the union over 48 EN prompts of top-50-each, |S_EN| ≈ 800–1100 features per layer (confirmed in data). This is well below d_tc = 163,840, so IoU is not trivially
  saturated.

  ---
  B. Target Anthropic-style Behavior

  B.1 Attribution — What Anthropic Does vs What Is Feasible

  Anthropic's actual procedure (from "Attribution Graphs: Methods"):
  1. A single Cross-Layer Transcoder (CLT) replaces all MLPs simultaneously. The CLT is a directed multi-layer transcoder — feature k at layer ℓ can have direct weights to feature j at layer ℓ' for any
  ℓ' > ℓ.
  2. For a given prompt, they run the replacement model (CLT-replaced) with full gradient tracking.
  3. Attribution of feature k at layer ℓ to the output logit difference: α_k^ℓ = a_k^ℓ × (∂Δlogit / ∂a_k^ℓ) — the standard gradient × activation formula.
  4. The gradient flows through: subsequent attention layers, subsequent CLT features (via direct CLT edges), and the final unembed projection.
  5. A pruning threshold removes features where |α_k^ℓ| < ε, giving the attribution graph.

  What we can do faithfully:
  - We can compute exact activation × gradient attribution, but through our per-layer transcoders running simultaneously in the forward pass (the "replacement model" using PLTs)
  - JumpReLU has a custom autograd function with a straight-through estimator (x_grad = (x > threshold) × grad_output), so gradients flow through it
  - We can hook all 16 layer transcoders into the forward pass at once, run with torch.enable_grad(), backpropagate from Δlogit, and extract a_k^ℓ × grad_k^ℓ for each active feature

  What is NOT feasible exactly, and why:
  - No CLT: Our transcoders are per-layer; there are no direct weight matrices from feature k at layer ℓ to feature j at layer ℓ'. The gradient through our replacement model flows through attention
  layers between transcoder layers, but the transcoder-to-transcoder gradient signal is mediated by attention and LayerNorm, not by explicit CLT edges. This means our gradient is computed through a
  structurally different replacement model.
  - No error nodes: Anthropic's replacement model includes residual "error nodes" at each layer to account for CLT approximation error. Our replacement model (PLTs in parallel) produces some
  reconstruction error that we cannot represent.
  - Memory: Running 16 transcoder forward passes simultaneously with gradient tracking requires holding activations for all 16 layers in memory. For Qwen3-4B with batch size 1, this is feasible on GPU
  (estimated ~8GB activation memory per prompt). Batch processing is not feasible for gradient mode — must process one prompt at a time.

  B.2 IoU — What Anthropic Does vs What Is Feasible

  Anthropic's procedure:
  1. Takes natural paragraphs (several hundred words), generates translations in FR and ZH
  2. For each (paragraph, language pair, layer): collects ALL feature activations, applies threshold t > 0
  3. Active set: {k : a_k > t} — a threshold-based set, not top-k
  4. IoU on active sets
  5. Critical: unrelated baseline: computes the same IoU for randomly paired paragraphs (EN paragraph X with FR paragraph Y, X≠Y). This establishes what IoU looks like when there is no semantic
  relationship.
  6. Reports: related IoU and unrelated IoU on same plot; middle layers show related >> unrelated

  What is feasible in our setup:
  - Our prompts are short (5–12 tokens), not paragraphs. Full-activation thresholding is approximate with top-k=50 saved data: we can apply a threshold on the saved top_k_values to exclude weakly active
   features.
  - We can compute pair-level IoU (per concept pair) rather than pooled-union IoU, then average. This is structurally closer to "per-paragraph" IoU.
  - We can define an unrelated baseline as: pair EN prompt from concept i with FR prompt from concept j (i ≠ j), compute IoU, average across pairs.

  What is NOT feasible exactly:
  - We cannot use a true absolute threshold on full activations without re-running script 04 with --save_full_acts (large files, ~50MB per layer per behaviour). We will use a threshold on saved top-k
  values instead (if top_k_value > t, feature is "active").
  - Our prompts are highly templated and short; the "content" of each prompt is one concept word. This means concept-level IoU is inherently noisier than paragraph-level IoU. We cannot fully replicate
  the statistical power of Anthropic's setup.

  ---
  C. Proposed Implementation

  C.1 Upgrade 1 — Gradient × Activation Attribution

  C.1.1 New function in src/model_utils.py

  Add method compute_gradient_attribution_per_prompt() to ModelWrapper. This runs the replacement model (all layer transcoders patched in) with gradient tracking and returns per-feature gradient
  attributions.

  Signature:
  def compute_gradient_attribution_per_prompt(
      self,
      prompt: str,
      correct_token_id: int,
      incorrect_token_id: int,
      transcoder_set: TranscoderSet,
      layer_indices: List[int],
      top_k: int = 50,
  ) -> Dict[int, Dict]:  # layer -> {feat_idx: attribution_score}

  Logic:
  1. Tokenize prompt, run model with torch.enable_grad() (override @torch.no_grad())
  2. At each post_attention_layernorm in layer_indices, hook to:
    - Capture x^ℓ (MLP input), create a^ℓ = transcoder.encode(x^ℓ) with requires_grad=True and retain_grad
    - Compute tc_out^ℓ = a^ℓ @ transcoder.W_dec + transcoder.b_dec
    - Replace the MLP output with tc_out^ℓ via an output hook (hook on model.layers[L].mlp)
  3. Forward pass produces logits; compute Δlogit = logit[correct_token_id] − logit[incorrect_token_id] at the decision token position
  4. Δlogit.backward()
  5. For each layer, read a^ℓ (saved activations) and a^ℓ.grad (from .backward())
  6. Compute α_k^ℓ = a_k^ℓ × a_k^ℓ.grad[k] for each k in top-k by |a_k|
  7. Return dict of {layer: {feat_idx: α_k}}

  Note on hooks: Two hooks per layer are needed:
  - Input hook on post_attention_layernorm: capture x^ℓ, compute a^ℓ = transcoder.encode(x^ℓ), register gradient
  - Output hook on mlp: replace MLP output with transcoder decode of a^ℓ

  The MLP output hook must be careful with residual stream: it replaces mlp(x^ℓ) with W_dec @ a^ℓ + b_dec, which becomes the new MLP contribution to the residual stream.

  C.1.2 New flag + function in scripts/04_extract_transcoder_features.py

  Add CLI argument --compute_gradients (bool flag).

  When set, after extracting top-k features for each prompt, call model.compute_gradient_attribution_per_prompt() and save:
  data/results/transcoder_features/layer_{L}/
    {behaviour}_{split}_gradient_attribution.npy   # shape (n_samples_decision, top_k), dtype float32
    {behaviour}_{split}_gradient_attribution_meta.json  # attribution_method, top_k, threshold

  The _gradient_attribution.npy stores α_k^ℓ for each of the top-k features at the decision token position per prompt, aligned with top_k_indices.npy (same ordering, same shape).

  Update {behaviour}_{split}_layer_meta.json to add:
  {
    "has_gradient_attribution": true,
    "attribution_method": "gradient_x_activation_replacement_model"
  }

  C.1.3 New method in scripts/06_build_attribution_graph.py

  Add aggregate_graphs_gradient_attribution() to TranscoderAttributionBuilder.

  Key differences from aggregate_graphs_per_prompt_union():
  - Reads gradient_attribution.npy instead of using top_k_values × beta
  - Feature score per prompt = α_k^p (from gradient attribution file)
  - No compute_beta() call needed
  - Edge weights = mean(α_k^p) averaged over prompts where feature is top-k by |α_k^p|
  - Node attributes: replace beta, beta_sign, specific_score with mean_gradient_attr, mean_abs_gradient_attr, gradient_attr_sign
  - max_frequency filter still applies (drop always-on features)

  CLI flag: --attribution_method {beta_proxy, gradient} — default remains beta_proxy for backward compatibility; gradient requires --compute_gradients to have been run in script 04.

  Metadata: Graph JSON gets a new top-level field:
  {
    "attribution_method": "gradient_x_activation_replacement_model",
    "attribution_version": "v2_gradient"
  }

  C.1.4 Deprecation of old path

  - Keep aggregate_graphs_per_prompt_union() intact but mark it as # DEPRECATED: correlation proxy; use aggregate_graphs_gradient_attribution() in docstring
  - beta, beta_sign, specific_score node attributes remain in old graph; new graph has gradient_attr equivalents
  - The role-aware method aggregate_graphs_role_aware() will call the gradient version when --attribution_method gradient is passed

  C.1.5 Diagnostics

  New script scripts/a_compare_attribution_methods.py:
  - Loads both the old beta-proxy graph and new gradient graph for the same behaviour/split
  - Reports: Jaccard similarity of node sets, rank correlation of edge weights, distribution of (attribution_old / attribution_new) ratio
  - Saves data/analysis/{behaviour}/attribution_method_comparison.csv

  ---
  C.2 Upgrade 2 — Anthropic-style IoU with Baseline

  C.2.1 Activation threshold on top-k values

  Current: active set = all features in the top-k list (no value filter).
  New: active set = features in top-k list with top_k_value > t.

  Add parameter activation_threshold (default 0.0 for backward compatibility). To select a meaningful threshold: inspect the distribution of saved top_k_values across layers and set t to, e.g., the 10th
   percentile of nonzero values. This is equivalent to what Anthropic does with a global threshold on CLT feature activations.

  C.2.2 New function compute_iou_with_baseline() in a_analyze_multilingual_circuits.py

  Replaces the inner logic of compute_iou(). Two modes, controlled by new parameter --iou_mode {pooled, concept_pair} (default pooled for backward compatibility):

  concept_pair mode (new, Anthropic-closer):
  related_pairs: all (EN prompt from concept c, FR prompt from concept c) pairs
  unrelated_pairs: (EN from concept c_i, FR from concept c_j, c_i ≠ c_j) — N_related pairs randomly sampled

  For each pair (EN_p, FR_p) and each layer L:
    S_EN = {k : top_k_value_EN_p_k > t}
    S_FR = {k : top_k_value_FR_p_k > t}
    IoU_pair = |S_EN ∩ S_FR| / |S_EN ∪ S_FR|

  IoU_related[L] = mean over all related pairs
  IoU_unrelated[L] = mean over all unrelated pairs
  delta[L] = IoU_related[L] - IoU_unrelated[L]
  signal_ratio[L] = IoU_related[L] / IoU_unrelated[L]

  For B1-v2 (8 concepts × 6 EN templates × 6 FR templates):
  - Related pairs: 8 × 36 = 288
  - Unrelated pairs: same N, randomly sampled with different concept_index

  Pair-level IoU requires top_k_values.npy to be loaded per row — they are already saved and loaded in the position map.

  C.2.3 New outputs

  data/analysis/{behaviour}/
    iou_related_vs_baseline_concept_pair.csv    — columns: layer, IoU_related, IoU_unrelated, delta, signal_ratio
    iou_baseline_report.md                     — structured summary table with verdict per zone

  iou_baseline_report.md contains:
  - Table of IoU_related, IoU_unrelated, delta per layer
  - Zone means (early/mid/late) for both related and unrelated
  - Statement: "Middle layer related IoU is X× above early layer, and Y× above unrelated baseline in middle layers"
  - Verdict: SIGNIFICANT if IoU_related_mid - IoU_unrelated_mid > 0.03, WEAK if 0.01–0.03, NOISE if < 0.01

  C.2.4 New flag in a_analyze_multilingual_circuits.py

  --iou_mode pooled           # current behavior (unchanged)
  --iou_mode concept_pair     # new Anthropic-closer mode
  --activation_threshold 0.0  # default; set > 0 to require minimum activation value
  --n_unrelated_pairs 288     # how many unrelated pairs to sample (default = n_related)
  --random_seed 42

  Backward compatibility: the existing compute_iou() function is not removed. The new function is added alongside it and called only when --iou_mode concept_pair is specified.

  C.2.5 Export to IoU data for dashboard

  prepare_b1_dashboard.py gets a new optional input: if iou_related_vs_baseline_concept_pair.csv exists, include it in iou_data.json under a new key "baseline". The dashboard IoUChart.jsx can then
  render the baseline curve alongside the existing three curves.

  ---
  D. Validation Plan

  D.1 Gradient Attribution Validation

  1. Sanity check: For a single prompt with a known correct answer, verify that features with positive α_k have decoder directions that project positively onto (unembed[correct] − unembed[incorrect]).
  This is a local faithfulness check.
  2. Rank correlation: For the same graph, compare ranking of features by |activation × beta| vs |activation × gradient|. Expected: r > 0.6 (positively correlated, since both measure influence on
  margin). If r < 0.3, something is wrong.
  3. Graph size comparison: New graph should have similar total node count (±20%) to old graph, since both use the same top-k union structure. Large divergence suggests a bug.
  4. Sign agreement: For features in both graphs, check that sign(beta) = sign(mean_gradient_attr) for the majority (>75%). Disagreements should be investigated.
  5. Script 07 compatibility check: Run script 07 on the new graph (with its new node attributes). Verify that intervention results are loadable and consistent (ablation still moves logit in the
  expected direction for top-ranked features).

  D.2 IoU Baseline Validation

  1. Null hypothesis check: Compute IoU_unrelated for same-language same-concept pairs (EN-c_i vs EN-c_i, different templates). This "within-language, same-concept" IoU should be higher than IoU_related
   (EN vs FR, same concept). If not, our active sets are dominated by non-semantic features.
  2. Permutation test: Repeat IoU_unrelated computation with 10 different random seeds. Report mean ± std of unrelated IoU per layer to confirm it is stable.
  3. Threshold sensitivity: Run with t = 0.0, t = 0.5, t = 1.0. Report how the related/unrelated gap changes. If the gap only appears at high thresholds, the signal is driven by weakly active features.
  4. Comparison to current pooled: Report both pooled and concept_pair curves on the same plot. If concept_pair related IoU is similar to pooled IoU, the two methods are consistent. If concept_pair is
  much lower, it indicates that pooling was inflating the signal.
  5. Key verdict check: If IoU_related_mid - IoU_unrelated_mid < 0.01 for all tested thresholds, Claim 3 should be downgraded to UNCONFIRMED in the thesis. This is the most important single number the
  validation needs to produce.

  ---
  E. Risk Assessment

  E.1 Gradient Attribution

  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────┬──────────────────────────────────────────────────────────────────────────────┐
  │                                                  Risk                                                   │ Severity │                                  Mitigation                                  │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Memory: running 16 transcoders + grad tracking simultaneously may OOM on 16GB GPU                       │ High     │ Process one prompt at a time (batch_size=1); checkpoint gradients if needed  │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ JumpReLU straight-through estimator gives grad=0 for inactive features                                  │ Low      │ This is correct behavior — inactive features have zero attribution           │
  │                                                                                                         │          │ regardless                                                                   │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ MLP output hook interferes with attention patterns (causal mask, KV cache)                              │ Medium   │ Use use_cache=False and ensure hooks are properly removed after each prompt  │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Lazy-loaded transcoder weights (lazy_decoder=True) not differentiable                                   │ Medium   │ Force eager load of W_dec before gradient pass:                              │
  │                                                                                                         │          │ tc.W_dec.requires_grad_(False)                                               │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Script 07 uses beta_sign from graph node for ablation direction; new graph has gradient_attr_sign       │ Medium   │ Keep beta_sign in new graph as alias for gradient_attr_sign; script 07 is    │
  │ instead                                                                                                 │          │ unchanged                                                                    │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ aggregate_graphs_role_aware() internally calls the union method; gradient version needs a parallel      │ Low      │ Add use_gradient=True flag to role-aware method; it delegates to gradient    │
  │ role-aware gradient variant                                                                             │          │ union internally                                                             │
  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────┘

  E.2 IoU Baseline

  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────┬─────────────────────────────────────────────────────────────────────┐
  │                                                       Risk                                                       │ Severity │                             Mitigation                              │
  ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────┤
  │ Concept_pair IoU has high variance (per-pair set sizes ~50 features → noisy intersection)                        │ High     │ Report per-layer SEM bars; use all 288 pairs, not a subsample       │
  ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────┤
  │ top_k_values.npy needed per sample row; currently loaded as a 2D array (n_samples, top_k) — row indexing by      │ Medium   │ Already done via position_map which maps sample_idx → prompt_idx;   │
  │ concept_index requires merging with JSONL metadata                                                               │          │ JSONL has concept_index                                             │
  ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────┤
  │ Concept-mismatched unrelated pairs may still share language-specific tokens (e.g., quotation marks, punctuation) │ Low      │ Inspect top shared unrelated features manually; they should be      │
  │  — giving artificially high IoU_unrelated                                                                        │          │ positional/structural                                               │
  ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────┤
  │ Dashboard iou_data.json schema change breaks IoUChart.jsx if new key baseline is missing                         │ Low      │ Use optional chaining in JSX; new key is additive, not breaking     │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────┴─────────────────────────────────────────────────────────────────────┘

  ---
  Expected Scientific Impact

  Upgrade 1 (Gradient attribution):

  The beta-proxy and gradient attribution are expected to produce largely similar graphs with meaningful but not dramatic differences. The two methods rank features by their influence on Δlogit, and
  both are ultimately constrained by the same transcoder feature extraction (top-k=50 activation values from the same runs). Gradient attribution is more local and per-prompt; it may surface features
  that have high influence on specific prompts but low dataset-average correlation, especially for FR error cases. The qualitative community structure and bridge features are unlikely to change, because
   those are dominated by large-activation features that also have large gradient attribution. The most likely meaningful change: the fraction of fr_leaning vs balanced features in the graph may shift
  slightly, as features that activate strongly but with inconsistent margin correlation could be demoted or promoted. Verdict: improves methodological rigor substantially; small to moderate effect on
  quantitative results; unlikely to change qualitative conclusions about community structure.

  Upgrade 2 (IoU baseline):

  This is the higher-stakes change. The current 1.090× ratio is already weak. If IoU_unrelated in the concept_pair mode is close to IoU_related (say within 0.02–0.03), then Claim 3 as currently framed —

⏺ Now I have everything. Here is the full plan.

  ---
  Implementation Plan: Two Methodological Upgrades

  ---
  A. Current State

  A.1 Attribution (Scripts 04 + 06)

  Script 04 captures MLP inputs at post_attention_layernorm and runs the transcoder encoder OFFLINE, saving only top_k_indices.npy and top_k_values.npy (top-50 features by raw activation). No gradient
  is ever computed. The model runs entirely under @torch.no_grad().

  Script 06, TranscoderAttributionBuilder, derives the graph through a Pearson correlation proxy:

  beta[(layer, feat)] = Cov(activation, Δlogit) / Var(activation)
                      [across all N prompts, missing=0 model]

  score_per_prompt = activation_k × beta[(layer, feat)]

  node selected if: score appears in top-k per prompt AND
                    prompt-count >= min_prompts AND
                    frequency < max_frequency

  Edge weight in graph = mean(activation × beta) across prompts where feature is in top-k.

  Concrete objects saved in graph JSON: corr, abs_corr, mean_activation_given_present, mean_activation_all, mean_score_conditional, mean_abs_score_conditional, specific_score, beta, beta_sign.

  What this is not: beta is a dataset-level regression coefficient. activation × beta is an approximation to per-prompt attribution. The gradient of Δlogit with respect to activation_k is never
  computed. No backpropagation is ever called anywhere in the pipeline.

  A.2 IoU Analysis (Script a_analyze_multilingual_circuits.py, compute_iou())

  S_EN^ℓ = ⋃_{p ∈ EN} top_k_indices[p, layer]   # union of top-50 index sets
  S_FR^ℓ = ⋃_{p ∈ FR} top_k_indices[p, layer]

  IoU_ℓ = |S_EN^ℓ ∩ S_FR^ℓ| / |S_EN^ℓ ∪ S_FR^ℓ|

  Problems with this:
  1. No activation threshold: a feature is "active" if it appears anywhere in the top-50 ranked by raw activation value, not if it exceeds a meaningful threshold. A feature ranked 50th with activation
  0.001 counts the same as one with activation 5.
  2. Pooled union, not pair-level: all 48 EN prompts are merged into one set, then compared to all 48 FR prompts. This is not per-paragraph comparison as in Anthropic.
  3. No unrelated-pair baseline: we have no way to say whether our measured IoU values (0.22–0.26) are above chance. Anthropic's key finding is IoU_related >> IoU_unrelated_baseline.

  ---
  B. Target Anthropic-style Behaviour

  B.1 Attribution — What Anthropic Does

  From "Attribution Graphs: Methods": features are scored per prompt as α_k^ℓ = activation_k^ℓ × (∂Δlogit / ∂activation_k^ℓ), computed through the CLT replacement model. Because the CLT replaces all
  MLPs simultaneously, the gradient ∂Δlogit/∂a_k^ℓ flows through all subsequent attention layers AND subsequent CLT features. The graph is pruned per-prompt and then aggregated.

  What is feasible: We can compute activation × gradient per prompt using the replacement-model approach: hook each MLP to replace its output with the transcoder output during a differentiable forward
  pass, let the transcoder features accumulate gradients, then backpropagate from Δlogit. This gives the correct gradient for our per-layer transcoder architecture.

  What is not feasible exactly: Anthropic's CLT is a single module with features at all layers simultaneously. Gradients flow between features at different layers through the CLT weight structure. Our
  per-layer transcoders are independent; cross-layer gradient flow goes only through the unmodified attention mechanism. This means our ∂Δlogit/∂a_k^ℓ does NOT capture the indirect path a_k^ℓ →
  transcoder_output^ℓ → residual → a_j^{ℓ+1} → ... → logit. It only captures the direct path from a_k^ℓ through the remaining layers after ℓ. This is an inherent architecture gap.

  The result is still dramatically better than the current correlation proxy, because:
  - It is per-prompt, not dataset-average
  - It actually measures causal influence on the specific output, not population correlation
  - Features that are active but causally irrelevant for this prompt get near-zero attribution
  - The sign is correct per-prompt (not an average sign)

  B.2 IoU — What Anthropic Does

  For each paragraph i, they have a translation pair (p_i^EN, p_i^FR). For each layer ℓ:

  S_i^{EN,ℓ} = {k : activation_k^ℓ(p_i^EN) > 0}   (threshold at 0, i.e., positive JumpReLU output)
  S_i^{FR,ℓ} = {k : activation_k^ℓ(p_i^FR) > 0}

  IoU_related_ℓ = mean_i( |S_i^EN ∩ S_i^FR| / |S_i^EN ∪ S_i^FR| )

  Baseline: same formula for random (non-matching) i-j pairs

  What is feasible in our setup: We have 8 concept groups × 6 train templates per (concept, language). For related pairs: same concept, different surface form, EN vs FR. For unrelated baseline:
  different concept, EN vs FR. We can compute pair-level IoU instead of pooled-union IoU.

  Activation threshold: our top-k=50 values are saved. We can apply a minimum activation value threshold to the saved top-k values (e.g., top_k_value > t) to approximate "active features". The JumpReLU
  threshold is the natural choice — but it's per-feature, not a global scalar. A practical approximation: use a single global threshold applied to the saved activation values (e.g., 0.1, or the minimum
  nonzero top-k value — which is already approximately the JumpReLU threshold because JumpReLU zeros features below threshold and our top-k captures the survivors).

  What is not feasible exactly: Anthropic runs full paragraphs (multi-sentence contexts). We have short synthetic prompts (1–2 sentences). Their feature activation sets are richer. We also only have
  top-50 features per prompt per layer, not all active features. Using top-50 means our S_i^ℓ has at most 50 elements; Anthropic's sets have variable size. This is a meaningful approximation gap.

  ---
  C. Proposed Implementation

  C.1 Upgrade 1 — Gradient × Activation Attribution

  C.1.1 New method in scripts/06_build_attribution_graph.py

  Add to TranscoderAttributionBuilder:

  New method compute_per_prompt_gradient_attribution(prompt_idx):

  For prompt p:
  1. tokenize prompt, get input_ids
  2. for each layer ℓ in self.layers:
     - register a forward-hook on model.model.layers[ℓ].mlp
     - hook replaces MLP forward output with transcoder output:
         x = input[0]  # MLP input (post-layernorm), shape (1, T, d_model)
         x_decision = x[0, decision_pos, :]   # single decision token
         pre_act = x_decision @ W_enc.T + b_enc
         a = JumpReLU(pre_act)                # shape (d_tc,)
         a.register_hook(capture_grad(ℓ))     # registers ∂Δlogit/∂a
         tc_out = a @ W_dec + b_dec
         return tc_out expanded back to (1, T, d_model) [only decision pos modified]
  3. run forward pass (WITH gradient tracking: no no_grad)
  4. compute Δlogit = logit[correct_token_id] - logit[incorrect_token_id]
  5. Δlogit.backward()
  6. for each layer ℓ:
     attribution_k = a_k × grad_k   # element-wise, shape (d_tc,)
     keep only top_k_indices from script 04 (sparse: only recompute for saved features)
  7. remove hooks, zero gradients

  Key implementation notes:
  - Only the decision token position's transcoder is run (not all sequence positions) — matches script 04 behavior
  - Transcoder weights need to be in float32 for gradient computation (bfloat16 gradients accumulate poorly)
  - Hooks must be removed after each prompt to avoid accumulation
  - Δlogit.backward(retain_graph=False) — no need to retain

  New method aggregate_graphs_gradient_attribution(n_prompts, k_per_prompt, min_prompts, max_frequency, vw_threshold):

  Same outer loop as aggregate_graphs_per_prompt_union(), but:
  - per-prompt scores = gradient_attribution_k (not activation × beta)
  - feature_scores[(layer, feat)] = [gradient_attribution_k^p for each prompt p where feature in top-k]
  - edge weight = mean(gradient_attribution_k) across prompts
  - node attributes:
      mean_grad_attr_conditional   (conditional on being in top-k)
      mean_abs_grad_attr           (absolute value mean)
      mean_grad_attr_missing0      (missing=0 mean over all N prompts)
      n_prompts, frequency         (same as before)

  New CLI flag in script 06: --attribution_method {correlation,gradient} (default: correlation for backward compatibility). When gradient is specified, calls aggregate_graphs_gradient_attribution(). Old
   method is kept, not removed.

  New graph JSON field: graph["attribution_metadata"] = {"method": "gradient_x_activation", "n_prompts": N, "k_per_prompt": K, "top_k_features": 50}.

  Retained for backward compat: beta, beta_sign, specific_score are NOT removed from the correlation method. They simply won't be present when the gradient method is used.

  C.1.2 Output file naming

  Graph with gradient attribution: attribution_graph_{split}_n{N}_gradattr.json (new suffix _gradattr). Existing _roleaware.json files untouched.

  C.1.3 No changes to script 04

  Script 04's outputs (top_k_indices.npy, top_k_values.npy) are still needed and used as the sparse support for gradient computation (we only compute gradients for features in the saved top-50, not all
  163,840).

  C.2 Upgrade 2 — Anthropic-style IoU with Baseline

  C.2.1 New function in scripts/a_analyze_multilingual_circuits.py

  New function compute_iou_with_baseline(features_dir, behaviour, split, train_jsonl, layers, out_dir, activation_threshold, n_unrelated_pairs):

  1. Load prompts → get (concept_index, language, template_idx, prompt_idx) for each
  2. Build related pairs:
     For each (concept_index c, EN template t_EN, FR template t_FR):
         pair = (prompt_idx_EN[c,t_EN], prompt_idx_FR[c,t_FR])
     → ~48 EN × 48 FR → use only cross-template pairs (t_EN ≠ t_FR) to avoid same-template bias
     In practice: for each concept, pair each EN template with one FR template (6 EN × 6 FR = 36 per concept, 288 total)
     Or simpler: 6 train templates → 6×6 all pairs minus diagonal = 30 pairs per concept × 8 concepts = 240 pairs

  3. Build unrelated pairs:
     Pair EN prompts from concept c_i with FR prompts from concept c_j, c_i ≠ c_j
     Sample n_unrelated_pairs randomly (default: same count as related pairs for balance)

  4. For each layer ℓ:
     Load top_k_indices.npy, top_k_values.npy

     For each pair (p_EN, p_FR):
         S_EN = {k : top_k_values[p_EN, :][rank_of_k] > activation_threshold}
         S_FR = {k : top_k_values[p_FR, :][rank_of_k] > activation_threshold}
         iou_pair = |S_EN ∩ S_FR| / |S_EN ∪ S_FR|

     IoU_related[ℓ]   = mean(iou_pair for related pairs)
     IoU_unrelated[ℓ] = mean(iou_pair for unrelated pairs)
     delta[ℓ]         = IoU_related[ℓ] - IoU_unrelated[ℓ]
     ratio[ℓ]         = IoU_related[ℓ] / IoU_unrelated[ℓ]

  5. Save:
     iou_related_vs_baseline.csv   (layer, iou_related, iou_unrelated, delta, ratio, n_related_pairs, n_unrelated_pairs)
     iou_related_vs_baseline.png   (two curves + delta shading)

  6. Compute zone stats for BOTH curves and report:
     Claim 3 test: delta_mid > delta_early? Does the relative elevation peak at middle layers?

  Activation threshold choices: run with three values and report all:
  - t=0.0 (same as current — any positive JumpReLU output; this is the natural threshold since JumpReLU zeros features below its learnable per-feature threshold)
  - t=0.5 (modest threshold filtering out very weakly active features)
  - t=1.0 (stricter)

  Primary metric uses t=0.0 (faithful to JumpReLU semantics).

  C.2.2 Updated outputs in prepare_b1_dashboard.py

  Add iou_data.json keys "baseline" and "related_vs_baseline" so the IoU chart tab can show both curves. The existing "pooled", "decision", "content" keys remain.

  C.2.3 New CLI flag in a_analyze_multilingual_circuits.py

  --iou_with_baseline flag triggers the new function alongside the existing compute_iou(). Both are run; outputs are written to separate files. Existing files untouched.

  ---
  D. Validation Plan

  D.1 Gradient Attribution (Upgrade 1)

  1. Sanity check — sign consistency: For a prompt where the model is very confident (Δlogit >> 0), the top-ranked gradient-attribution features should have positive attribution (supporting the correct
  answer). Verify sign(gradient_attribution) ≈ sign(beta) for the top-20 features in >80% of cases.
  2. Overlap check: Compute Jaccard similarity between the feature sets selected by the old correlation method and the new gradient method, per layer. Expected: 50–80% overlap (they should select mostly
   the same important features, with gradient being more precise). If overlap is <30%, something is wrong with the gradient computation.
  3. Necessity check: Apply the same circuit validation from script 08 to the gradient-attributed graph. The disruption rate should be ≥ the correlation graph's 10.4% (gradient is more causally precise
  so should identify more necessary features, not fewer).
  4. Computational sanity: Print the mean |activation × gradient| vs mean |activation × beta| per layer. They should be in the same order of magnitude. Large discrepancies suggest a numerical issue.
  5. Comparison run: Run both methods (--attribution_method correlation and --attribution_method gradient) on the same 96 prompts, save both graphs. Produce a diff report: which features appear in
  gradient-only, correlation-only, both.

  D.2 IoU Baseline (Upgrade 2)

  1. Baseline sanity: IoU_unrelated should be lower than IoU_related at every layer (if not, the methodology is wrong or the behaviour has no cross-lingual structure at all).
  2. Expected magnitude: Random pair IoU expected ~0.10–0.18 (rough estimate based on sparsity: if each prompt activates ~50 features of 163,840, random overlap ~50²/163,840 ≈ 0.015 without any
  structure, but sharing across templates inflates this). If IoU_unrelated > 0.20, the top-k truncation is creating artificial overlap (features active on almost all prompts).
  3. Zone test: Compute delta_early, delta_mid, delta_late. The Anthropic claim is delta_mid > delta_early. If this is reversed, Claim 3 is not supported even at the pair level.
  4. Activation threshold sensitivity: Compare results at t=0.0, t=0.5, t=1.0. If the delta pattern is robust across thresholds, the result is stable. If the sign flips with threshold, there is a
  threshold-sensitivity problem.
  5. New ratio: Report IoU_related_mid / IoU_unrelated_mid as a new metric (cross-lingual enrichment at mid layers). This is more interpretable than the current same-layer ratio.

  ---
  E. Risk Assessment

  E.1 Script 07 (Interventions) Compatibility

  Script 07 reads the attribution graph JSON and selects features by beta_sign and node attributes for the strict mode. If using the gradient graph, beta_sign will not be present. Fix: gradient method
  nodes must include a grad_attr_sign field (+1/-1/0) as a drop-in replacement. Script 07 already has --from_circuit mode that bypasses graph attributes entirely, so the canonical circuit (script 08) is
   unaffected. The graph-based strict mode needs to accept either beta_sign or grad_attr_sign.

  E.2 Script 08 (Causal Edges) Compatibility

  Script 08 reads the graph JSON for candidate feature pairs. It only uses node existence and layer/feature_idx attributes, not beta/specific_score. No change needed.

  E.3 prepare_b1_dashboard.py and Dashboard

  The dashboard reads graph.json and inspects node attributes. New gradient-attribution nodes have different attribute names (mean_grad_attr_conditional vs mean_score_conditional). The dashboard's
  FeatureInspector component shows arbitrary node metadata, so it will display whatever attributes exist. The graph coloring and filtering are based on lang_profile (from node_labels.json), not on score
   attributes. No dashboard code change required — new attributes will simply appear as additional metadata in the inspector.

  E.4 IoU CSV Changes

  prepare_b1_dashboard.py reads iou_per_layer.csv (old) and writes iou_data.json. Adding new keys to iou_data.json is backward-compatible (the dashboard IoU chart reads specific keys it knows about).
  The new "baseline" key will simply be ignored by the existing chart until the chart is updated to display it.

  E.5 Memory — Gradient Computation

  Running a differentiable forward pass with 16 transcoder replacements per prompt requires holding the computation graph in memory. For one prompt on a 4B-parameter model with bfloat16 weights:
  attention activations ≈ 200MB, transcoder features ≈ 16 × 163,840 × 4B float32 ≈ ~1GB intermediate. Processing 96 prompts sequentially (not batched), this is fine for a GPU with ≥16GB. On CPU (CSD3),
  memory is fine but speed is 50–100× slower. Add a note that gradient attribution requires GPU.

  E.6 Fallback if Hook Replacement Fails

  If the MLP replacement hook causes issues (e.g., gradient graph doesn't flow through, or shape mismatch), fallback is: compute ∂Δlogit/∂mlp_output^ℓ through the ORIGINAL model (no transcoder
  replacement), then project as W_dec[k, :] · grad_wrt_mlp_output^ℓ. Label this as "linear_decoder_projection" in metadata. This is still a major improvement over the current correlation proxy.

  ---
  Expected Scientific Impact

  Upgrade 1 — Gradient Attribution

  Faithfulness improvement: High. We go from a dataset-level statistical proxy to a per-prompt causal quantity. This is the right thing to do.

  Result change: Moderate. The identified feature sets will be similar for the top features (features with high correlation are usually also causally active), but the ranking will change for mid-range
  features. The graph topology will shift slightly. Node count likely stays within ±20% of current 86.

  Conclusion change: Unlikely for the main claims. The bridge rate, community structure, and IoU are all computed from the intervention data (scripts 07/08/10), not from the graph attribution scores.
  Since the circuit (script 08) is validated by causal patching regardless of how it was found, the gradient attribution improves how we FIND the circuit but doesn't change the circuit validation
  numbers.

  Methodological rigor: Large improvement. Removes the biggest methodological gap from the Anthropic comparison.

  Upgrade 2 — IoU Baseline

  Faithfulness improvement: Very high. This adds the single most important missing experiment (the one that determines whether Claim 3 is meaningful or noise).

  Result change: Potentially large and uncertain. Two scenarios:
  - Pessimistic: IoU_unrelated ≈ 0.20–0.22, IoU_related ≈ 0.22–0.26, delta ≈ 0.02–0.04. Middle layer elevation is real but tiny. Claim 3 holds as WEAK.
  - Optimistic: IoU_unrelated ≈ 0.10–0.15, IoU_related ≈ 0.22–0.26, delta ≈ 0.08–0.14, and delta_mid > delta_early clearly. Claim 3 upgrades from WEAK to MODERATE/STRONG.

  Conclusion change: Potentially significant. If the baseline turns out to be close to the related-pair IoU, the thesis must acknowledge that the IoU signal cannot distinguish genuine cross-lingual
  sharing from incidental co-activation. If the baseline is clearly lower, the thesis gets its strongest quantitative argument for Claim 3.

  Methodological rigor: Essential. Without this, IoU values are uninterpretable in isolation. This is the upgrade with the highest scientific value per implementation effort.

  Recommendation: Implement Upgrade 2 first — it is lower-risk (analysis-only, no model re-run), faster to implement, and has higher potential to change conclusions. Upgrade 1 is methodologically
  correct but computationally expensive and unlikely to change conclusions materially.

  ---
  Waiting for your approval before writing any code.

