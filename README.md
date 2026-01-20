# Mechanistic Interpretability of an Open-Source LLM (Qwen3-4B-Instruct)
Small-scale reproduction of *On the Biology of a Large Language Model* (Transformer Circuits, 2025)

Authour: Iuliia Vitiugova
DAMTP, University of Cambridge 

This repository investigates whether circuits described in *On the Biology of a Large Language Model* can be reproduced in an open model (**Qwen3-4B-Instruct**).  
The focus is on **mechanistic interpretability**: extracting **interpretable features** from internal activations using **Sparse Autoencoders (SAEs)**, building **pruned attribution graphs**, and validating causal claims via **feature-level interventions** (ablation / swap-in).

---

## 1) Notation

We work with an autoregressive transformer language model that predicts the next token given a prompt.

- `T` = sequence length (number of tokens)
- `V` = vocabulary size
- `d` = model dimension (`d_model`)
- `L` = number of transformer layers
- `H` = number of attention heads
- `d_h = d / H` = per-head dimension

A prompt is tokenized into token ids:

`prompt -> (t_1, t_2, ..., t_T)` with `t_i in {1, ..., V}`

The model state is the **residual stream matrix**:

`X^(0) in R^(T x d)`

where row `X^(0)_i` is the vector at token position `i`.

---

## 2) Input embeddings

The initial residual stream is computed from token + positional embeddings:

`X^(0) = TokEmb(t_1:T) + PosEmb(1:T)`

where:
- `TokEmb(t_i) in R^d`
- `PosEmb(i) in R^d`

---

## 3) One transformer layer (Pre-LN)

Each layer `l in {1, ..., L}` has two sequential blocks:

1) Self-Attention (moves information across token positions)  
2) MLP / FFN (processes information within each token position)

Both use LayerNorm and residual additions.

We write:
- `LN(.)` = LayerNorm
- `Attn_l(.)` = attention block at layer `l`
- `MLP_l(.)` = MLP block at layer `l`

Layer update (high level):

`X -> X + Attn_l(LN(X)) -> X + MLP_l(LN(X))`

---

## 4) Self-attention in detail (Q, K, V, softmax)

Let the attention input be:

`X_tilde^(l) = LN(X^(l))`  in `R^(T x d)`

### 4.1) Q, K, V projections (per head)

For head `h`, we have projection matrices:

`W_Q^(l,h), W_K^(l,h), W_V^(l,h) in R^(d x d_h)`

Compute:

`Q^(l,h) = X_tilde^(l) W_Q^(l,h)` in `R^(T x d_h)`  
`K^(l,h) = X_tilde^(l) W_K^(l,h)` in `R^(T x d_h)`  
`V^(l,h) = X_tilde^(l) W_V^(l,h)` in `R^(T x d_h)`

Interpretation:
- `Q_i`: what token position `i` is looking for
- `K_j`: what token position `j` contains
- `V_j`: information token `j` can provide

### 4.2) Scaled dot-product attention scores

For positions `(i, j)`:

`S_ij^(l,h) = (Q_i^(l,h) · K_j^(l,h)) / sqrt(d_h)`

So:

`S^(l,h) in R^(T x T)`

### 4.3) Causal mask (autoregressive)

Token `i` cannot attend to future tokens `j > i`.

Define mask:

`M_ij = 0   if j <= i`  
`M_ij = -inf if j > i`

Apply:

`S_tilde^(l,h) = S^(l,h) + M`

### 4.4) Softmax attention weights

Row-wise softmax:

`A_i:^(l,h) = softmax(S_tilde_i:^(l,h))`

So:

`A^(l,h) in R^(T x T)` and `sum_j A_ij^(l,h) = 1`

### 4.5) Weighted sum of values

`Y^(l,h) = A^(l,h) V^(l,h)`  in `R^(T x d_h)`

### 4.6) Multi-head combine + output projection

Concatenate heads:

`Y^(l) = Concat(Y^(l,1), ..., Y^(l,H))` in `R^(T x d)`

Project:

`Attn_l(X_tilde^(l)) = Y^(l) W_O^(l)` in `R^(T x d)`

with `W_O^(l) in R^(d x d)`.

### 4.7) Residual add after attention

`X^(l+1/2) = X^(l) + Attn_l(LN(X^(l)))`

---

## 5) MLP / FFN in detail

The MLP is applied independently at each token position.

Let:

`U^(l) = LN(X^(l+1/2))` in `R^(T x d)`

Define MLP parameters:
- `W_1^(l) in R^(d x m)` (expansion)
- `W_2^(l) in R^(m x d)` (projection back)
- `b_1^(l) in R^m`, `b_2^(l) in R^d`
- `phi(.)` = nonlinearity (e.g. GELU/SiLU)

### 5.1) MLP hidden activations (what we collect)

`h^(l) = phi(U^(l) W_1^(l) + b_1^(l))` in `R^(T x m)`

This `h^(l)` is the **MLP activations** used for SAE training.

### 5.2) MLP output

`MLP_l(U^(l)) = h^(l) W_2^(l) + b_2^(l)` in `R^(T x d)`

### 5.3) Residual add after MLP

`X^(l+1) = X^(l+1/2) + MLP_l(LN(X^(l+1/2)))`

---

## 6) Baseline forward pass (full transformer)

Initialize:

`X^(0) = TokEmb(t_1:T) + PosEmb(1:T)`

For `l = 1..L`:

`X^(l+1/2) = X^(l) + Attn_l(LN(X^(l)))`  
`X^(l+1)   = X^(l+1/2) + MLP_l(LN(X^(l+1/2)))`

Final residual stream:

`X^(L) in R^(T x d)`

---

## 7) Logits and next-token prediction

Let `x_last = X^(L)_T` be the final position vector (`x_last in R^d`).

Unembedding matrix:

`W_U in R^(d x V)`

Logits:

`logits = x_last W_U` in `R^V`

Probabilities:

`p = softmax(logits)`

Next token (deterministic decoding):

`t_(T+1) = argmax_v p_v`

Generation repeats by appending `t_(T+1)` and running the forward pass again.

---

## 8) Sparse autoencoders (SAEs) on MLP activations

For selected layers `l`, we train an SAE on MLP activations `h^(l)`.

Each SAE maps:

`h in R^m -> z in R^k -> h_hat in R^m`

where:
- `k` = SAE bottleneck size (number of learned features)
- `z` is sparse (most coordinates near 0)

### 8.1) Encoder (features)

`z = E_l(h)` in `R^k`

### 8.2) Decoder (reconstruction)

`h_hat = D_l(z)` in `R^m`

### 8.3) SAE training objective

We minimize:

`L_SAE = ||h - h_hat||_2^2 + lambda * ||z||_1`

where:
- reconstruction term enforces faithfulness
- L1 penalty enforces sparsity

---

## 9) SAE-based replacement model

To enable feature-level circuit tracing and interventions, we optionally replace the original MLP hidden activations `h^(l)` with their SAE reconstruction `h_hat^(l)` during the transformer forward pass.

For a chosen layer `l`:

1) compute original MLP hidden activations:

`h^(l) = phi(U^(l) W_1^(l) + b_1^(l))`

2) encode into sparse features:

`z^(l) = E_l(h^(l))`

3) decode back:

`h_hat^(l) = D_l(z^(l))`

4) compute MLP output using reconstructed activations:

`MLP_hat_l(U^(l)) = h_hat^(l) W_2^(l) + b_2^(l)`

5) residual update becomes:

`X^(l+1) = X^(l+1/2) + MLP_hat_l(LN(X^(l+1/2)))`

This defines an SAE-based **replacement representation** of the MLP computation.

---

## 10) Feature-level interventions (validation)

We validate candidate circuits by intervening on SAE features `z`.

### 10.1) Inhibition / ablation

Choose a set of feature indices `S subset {1, ..., k}` and set them to zero:

`z'_i = 0   if i in S`  
`z'_i = z_i if i not in S`

Then decode and continue forward:

`h_hat = D_l(z')`  
`X^(l+1) = X^(l+1/2) + (h_hat W_2^(l) + b_2^(l))`

We quantify causal effects via logit changes:

`Delta logit(v) = logit_after(v) - logit_before(v)`

for a decisive token `v` (e.g. the correct answer token).

### 10.2) Swap-in

Let `z_clean` be features from a clean run and `z_corr` from a corrupted run.

Swap a subset `S`:

`z'_i = z_clean_i if i in S`  
`z'_i = z_corr_i  if i not in S`

Decode and continue forward as above, and measure whether the output/logits shift as expected.

---

## 11) Full forward pass with optional SAE replacement and interventions

Let `L_SAE subset {1, ..., L}` be the set of layers where we apply SAE replacement.

Initialize:

`X = TokEmb(t_1:T) + PosEmb(1:T)`

For each layer `l = 1..L`:

1) attention + residual:

`X = X + Attn_l(LN(X))`

2) MLP input:

`U = LN(X)`

3) MLP hidden activations:

`h = phi(U W_1^(l) + b_1^(l))`

4) optional SAE replacement + intervention:

if `l in L_SAE`:

`z = E_l(h)`  
`z = Intervene(z)`   (optional ablation/swap)  
`h = D_l(z)`

5) MLP output + residual:

`X = X + (h W_2^(l) + b_2^(l))`

Finally:

`logits = X_last W_U`

---

## 12) Planned pipeline in this repository

This repo implements the following steps:

1. Baselines: run Qwen3-4B-Instruct on small prompt sets with fixed seeds/decoding.
2. Activation capture: collect MLP activations `h^(l)` for selected layers.
3. Train SAEs: fit `(E_l, D_l)` per layer with train/val split and reported bottleneck sizes.
4. Feature interpretation: map features to tokens/behaviours via top-activating examples.
5. Pruned attribution graphs: build small graphs from input tokens -> SAE features -> decisive logits.
6. Validation: perform ablation and swap-in interventions, quantify `Delta logits` and behaviour changes.
7. Reproducibility pack: scripts, configs, seeds, and graphs for full reproducibility.

---

## References

- Lindsey, J., Gurnee, W., et al. (2025). *On the Biology of a Large Language Model*. Anthropic Transformer Circuits.
