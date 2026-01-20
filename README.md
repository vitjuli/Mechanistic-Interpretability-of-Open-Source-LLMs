# Mechanistic-Interpretability-of-Open-Source-LLMs
Mechanistic Interpretability of Open Source LLMs: reproduction of On the Biology of a Large Language Model Investigation of mechanisms can be found in an open model, Qwen3-4B-Instruct. DAMPT, University of Cambridge.

# Mechanistic Interpretability of an Open-Source LLM (Qwen3-4B-Instruct)  
## Transformer Forward Pass → Activations → Sparse Autoencoders (SAEs) → Replacement Model → Attribution Graphs → Interventions

This repository reproduces (at small scale) key steps from *On the Biology of a Large Language Model* using an open model (**Qwen3-4B-Instruct**).  
The goal is to identify **interpretable internal features** and **causal circuits** for selected behaviours, using **Sparse Autoencoders (SAEs)** trained on **MLP activations**, then validated via **feature-level interventions** (ablation / swap-in).

---

## 1) Problem setup and notation

We consider an autoregressive language model that predicts the next token given a prompt.

### Tokens
Given a prompt string, we tokenize it into a sequence:
\[
\text{prompt} \;\to\; (t_1, t_2, \dots, t_T)
\]
where each \(t_i\) is a token id in a vocabulary of size \(V\).

### Embedding dimension and residual stream
Let:
- \(T\) = sequence length
- \(d\) = model dimension (`d_model`)
- \(L\) = number of transformer layers

The model maintains a **residual stream matrix**:
\[
X^{(0)} \in \mathbb{R}^{T \times d}
\]
where row \(X^{(0)}_i\) is the vector state at token position \(i\).

---

## 2) Input embeddings (start of the forward pass)

### Token embeddings + positional embeddings
The initial residual stream is:
\[
X^{(0)} = \mathrm{TokEmb}(t_{1:T}) + \mathrm{PosEmb}(1:T)
\]
where:
- \(\mathrm{TokEmb}: \{1,\dots,V\}\to\mathbb{R}^{d}\)
- \(\mathrm{PosEmb}(i)\in\mathbb{R}^{d}\)

---

## 3) Transformer layer structure (Pre-LN)

Each transformer layer \(\ell \in \{1,\dots,L\}\) contains two blocks:

1. **Self-Attention** (moves information across token positions)
2. **MLP / FFN** (processes information within each token position)

Each block is applied with:
- LayerNorm
- Residual connection (addition)

We denote:
- \(\mathrm{LN}(\cdot)\) = LayerNorm
- \(\mathrm{Attn}_\ell(\cdot)\) = attention block in layer \(\ell\)
- \(\mathrm{MLP}_\ell(\cdot)\) = MLP block in layer \(\ell\)

---

# 4) Self-Attention in detail (Q, K, V, softmax)

Let the input to attention at layer \(\ell\) be:
\[
\tilde{X}^{(\ell)} = \mathrm{LN}(X^{(\ell)}) \in \mathbb{R}^{T \times d}
\]

## 4.1) Linear projections to queries, keys, values

For a single attention head \(h\), define matrices:
\[
W_Q^{(\ell,h)},W_K^{(\ell,h)},W_V^{(\ell,h)} \in \mathbb{R}^{d \times d_h}
\]
where \(d_h = d/H\) and \(H\) is the number of heads.

Compute:
\[
Q^{(\ell,h)} = \tilde{X}^{(\ell)} W_Q^{(\ell,h)} \in \mathbb{R}^{T \times d_h}
\]
\[
K^{(\ell,h)} = \tilde{X}^{(\ell)} W_K^{(\ell,h)} \in \mathbb{R}^{T \times d_h}
\]
\[
V^{(\ell,h)} = \tilde{X}^{(\ell)} W_V^{(\ell,h)} \in \mathbb{R}^{T \times d_h}
\]

Interpretation:
- **Query** \(Q_i\): what token \(i\) is looking for
- **Key** \(K_j\): what token \(j\) contains
- **Value** \(V_j\): what information token \(j\) provides if attended to

## 4.2) Attention scores (scaled dot-product)

For each pair of token positions \((i,j)\):
\[
S^{(\ell,h)}_{ij} = \frac{\langle Q^{(\ell,h)}_i,\;K^{(\ell,h)}_j\rangle}{\sqrt{d_h}}
\]
This yields:
\[
S^{(\ell,h)} \in \mathbb{R}^{T \times T}
\]

## 4.3) Causal mask (autoregressive constraint)

Because the model is autoregressive, token \(i\) cannot attend to future tokens \(j>i\).
We apply a causal mask \(M\in\mathbb{R}^{T\times T}\):
\[
M_{ij} =
\begin{cases}
0 & j\le i\\
-\infty & j>i
\end{cases}
\]
and compute:
\[
\tilde{S}^{(\ell,h)} = S^{(\ell,h)} + M
\]

## 4.4) Softmax to obtain attention weights

Row-wise softmax:
\[
A^{(\ell,h)}_{i:} = \mathrm{softmax}\left(\tilde{S}^{(\ell,h)}_{i:}\right)
\]
so:
\[
A^{(\ell,h)} \in \mathbb{R}^{T \times T}
\quad\text{and}\quad
\sum_{j=1}^T A^{(\ell,h)}_{ij} = 1
\]

## 4.5) Weighted sum of values

Each output token representation for head \(h\) is:
\[
Y^{(\ell,h)} = A^{(\ell,h)}V^{(\ell,h)} \in \mathbb{R}^{T \times d_h}
\]

## 4.6) Multi-head concatenation + output projection

Concatenate all heads:
\[
Y^{(\ell)} = \mathrm{Concat}\left(Y^{(\ell,1)},\dots,Y^{(\ell,H)}\right) \in \mathbb{R}^{T \times d}
\]

Then project back to \(d\):
\[
\mathrm{Attn}_\ell(\tilde{X}^{(\ell)}) = Y^{(\ell)} W_O^{(\ell)} \in \mathbb{R}^{T \times d}
\]
with \(W_O^{(\ell)}\in\mathbb{R}^{d\times d}\).

## 4.7) Residual addition after attention

\[
X^{(\ell+\frac{1}{2})} = X^{(\ell)} + \mathrm{Attn}_\ell(\mathrm{LN}(X^{(\ell)}))
\]

---

# 5) MLP / Feed-Forward block in detail

The MLP is applied **independently at each token position**, i.e. no mixing across tokens.

Let:
\[
U^{(\ell)} = \mathrm{LN}\left(X^{(\ell+\frac{1}{2})}\right) \in \mathbb{R}^{T\times d}
\]

For each token position \(i\), the MLP is:
\[
\mathrm{MLP}_\ell(U^{(\ell)}_i)
=
W^{(\ell)}_2 \,\phi\!\left(W^{(\ell)}_1 U^{(\ell)}_i + b^{(\ell)}_1\right) + b^{(\ell)}_2
\]

Where:
- \(W^{(\ell)}_1 \in \mathbb{R}^{d \times m}\) expands dimension
- \(W^{(\ell)}_2 \in \mathbb{R}^{m \times d}\) projects back
- \(m\) = MLP hidden width (often \(\gg d\))
- \(\phi(\cdot)\) = nonlinearity (e.g. GELU/SiLU)

### MLP activations (the object we collect for SAE training)
Define the **pre-output hidden activations**:
\[
h^{(\ell)}_i = \phi\!\left(W^{(\ell)}_1 U^{(\ell)}_i + b^{(\ell)}_1\right) \in \mathbb{R}^{m}
\]
Stacked over all tokens:
\[
h^{(\ell)} \in \mathbb{R}^{T\times m}
\]

Then:
\[
\mathrm{MLP}_\ell(U^{(\ell)}) = W^{(\ell)}_2 h^{(\ell)} + b^{(\ell)}_2
\]

## 5.1) Residual addition after MLP
\[
X^{(\ell+1)} = X^{(\ell+\frac{1}{2})} + \mathrm{MLP}_\ell(\mathrm{LN}(X^{(\ell+\frac{1}{2})}))
\]

---

# 6) Full transformer forward pass (baseline)

Initialize:
\[
X^{(0)} = \mathrm{TokEmb}(t_{1:T}) + \mathrm{PosEmb}(1:T)
\]

For \(\ell = 1,\dots,L\):
\[
X^{(\ell+\frac{1}{2})} = X^{(\ell)} + \mathrm{Attn}_\ell(\mathrm{LN}(X^{(\ell)}))
\]
\[
X^{(\ell+1)} = X^{(\ell+\frac{1}{2})} + \mathrm{MLP}_\ell(\mathrm{LN}(X^{(\ell+\frac{1}{2})}))
\]

After the final layer, we obtain:
\[
X^{(L)} \in \mathbb{R}^{T\times d}
\]

---

# 7) Output logits and next-token prediction

Let \(x_{\text{last}} = X^{(L)}_{T}\in\mathbb{R}^{d}\) be the residual stream at the final token position.

The unembedding matrix is:
\[
W_U \in \mathbb{R}^{d\times V}
\]

Logits over the vocabulary:
\[
\text{logits} = x_{\text{last}} W_U \in \mathbb{R}^{V}
\]

Probabilities:
\[
p = \mathrm{softmax}(\text{logits})
\]

Next token:
\[
t_{T+1} = \arg\max_{v} \; p_v \quad (\text{for deterministic decoding})
\]

Generation repeats by appending the new token and running the forward pass again.

---

# 8) Sparse Autoencoders (SAEs) on MLP activations

To obtain interpretable internal features, we train a **Sparse Autoencoder** per selected layer \(\ell\), using collected MLP activations \(h^{(\ell)}\).

For a given layer \(\ell\), the SAE operates on vectors:
\[
h \in \mathbb{R}^{m}
\]

## 8.1) Encoder → sparse features
\[
z = E_\ell(h) \in \mathbb{R}^{k}
\]
where:
- \(k\) = SAE bottleneck size (number of features)
- \(z\) is sparse (most entries near 0)

## 8.2) Decoder → reconstructed activations
\[
\hat{h} = D_\ell(z) \in \mathbb{R}^{m}
\]

## 8.3) SAE training objective
A typical objective is:
\[
\mathcal{L}_{\text{SAE}}
=
\|h - \hat{h}\|_2^2
+
\lambda \|z\|_1
\]
where:
- reconstruction loss ensures faithfulness
- \(L_1\) penalty encourages sparsity

---

# 9) SAE-based replacement model (feature-level representation inside the transformer)

To enable circuit tracing and interventions, we can **replace** the original MLP hidden activations \(h^{(\ell)}\) by their SAE reconstruction.

For a chosen layer \(\ell\):

1) compute MLP hidden activations:
\[
h^{(\ell)} = \phi(W^{(\ell)}_1 U^{(\ell)} + b^{(\ell)}_1)
\]

2) encode into sparse features:
\[
z^{(\ell)} = E_\ell(h^{(\ell)})
\]

3) decode back:
\[
\hat{h}^{(\ell)} = D_\ell(z^{(\ell)})
\]

4) compute MLP output using reconstructed activations:
\[
\widehat{\mathrm{MLP}}_\ell(U^{(\ell)}) = W^{(\ell)}_2 \hat{h}^{(\ell)} + b^{(\ell)}_2
\]

5) residual update becomes:
\[
X^{(\ell+1)} = X^{(\ell+\frac{1}{2})} + \widehat{\mathrm{MLP}}_\ell(\mathrm{LN}(X^{(\ell+\frac{1}{2})}))
\]

This defines an **SAE-based replacement representation** of the MLP computation.

---

# 10) Feature-level interventions (validation)

We validate candidate circuits by intervening on SAE features \(z\).

## 10.1) Inhibition / ablation
Given a set of feature indices \(S\subseteq\{1,\dots,k\}\):
\[
z'_i =
\begin{cases}
0 & i\in S\\
z_i & i\notin S
\end{cases}
\]

Then:
\[
\hat{h} = D_\ell(z')
\]
\[
X^{(\ell+1)} = X^{(\ell+\frac{1}{2})} + W^{(\ell)}_2 \hat{h} + b^{(\ell)}_2
\]

We quantify the causal effect by measuring:
\[
\Delta \text{logit}(v) = \text{logit}_{\text{after}}(v) - \text{logit}_{\text{before}}(v)
\]
for a decisive token \(v\) (e.g. the correct answer token).

## 10.2) Swap-in intervention
Let \(z^{\text{clean}}\) be features from a clean run and \(z^{\text{corr}}\) from a corrupted run.
Swap features in subset \(S\):
\[
z'_i =
\begin{cases}
z^{\text{clean}}_i & i\in S\\
z^{\text{corr}}_i & i\notin S
\end{cases}
\]
Then decode and continue the forward pass as above.

---

# 11) Full forward pass with optional SAE replacement and interventions

Let \(\mathcal{L}_{\text{SAE}}\subseteq\{1,\dots,L\}\) be the set of layers where we apply SAE replacement.

Initialize:
\[
X = \mathrm{TokEmb}(t_{1:T}) + \mathrm{PosEmb}(1:T)
\]

For each layer \(\ell = 1,\dots,L\):

### Attention + residual
\[
X \leftarrow X + \mathrm{Attn}_\ell(\mathrm{LN}(X))
\]

### MLP input
\[
U \leftarrow \mathrm{LN}(X)
\]

### MLP hidden activations
\[
h \leftarrow \phi(W^{(\ell)}_1 U + b^{(\ell)}_1)
\]

### Optional SAE replacement / intervention
If \(\ell \in \mathcal{L}_{\text{SAE}}\):
\[
z \leftarrow E_\ell(h)
\]
\[
z \leftarrow \mathrm{Intervene}(z) \quad \text{(optional ablation/swap)}
\]
\[
h \leftarrow D_\ell(z)
\]

### MLP output + residual
\[
X \leftarrow X + \left(W^{(\ell)}_2 h + b^{(\ell)}_2\right)
\]

Finally:
\[
\text{logits} = X_{\text{last}} W_U
\]

---

# 12) Planned deliverables in this repository

This repo implements the following pipeline:

1. **Baselines**: run Qwen3-4B-Instruct on prompt sets with fixed seeds and decoding settings.
2. **Activation capture**: collect MLP activations \(h^{(\ell)}\) for selected layers.
3. **Train SAEs**: fit \(E_\ell, D_\ell\) per layer with train/val split and reported bottleneck sizes.
4. **Feature interpretation**: map features to tokens/behaviours via top-activating examples.
5. **Attribution graphs (pruned)**: build small graphs from input tokens → SAE features → decisive logits.
6. **Validation**: perform feature ablation and swap-in interventions, quantify \(\Delta\)logits and behaviour changes.
7. **Repro pack**: release scripts, configs, seeds, and graphs for reproducibility.

---

## References
- Lindsey, J., Gurnee, W., et al. (2025). *On the Biology of a Large Language Model*. Anthropic Transformer Circuits.
