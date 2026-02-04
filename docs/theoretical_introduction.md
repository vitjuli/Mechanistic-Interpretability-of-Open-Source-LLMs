# Theoretical Introduction

## From Tokenization to Attribution Graphs: A Complete Mathematical Framework

This chapter provides a self-contained mathematical treatment of every component in our mechanistic interpretability pipeline, from the raw text input to the causal circuit analysis. We ground each formula in the specific architecture of Qwen3-4B-Instruct-2507 and relate our methodology to Anthropic's cross-layer transcoder (CLT) approach (Lindsey et al., 2025).

---

## 1. Tokenization

### 1.1 Vocabulary and Byte-Pair Encoding

Let $\Sigma$ be the set of all Unicode characters. A **tokenizer** is a function

$$\text{Tokenize}: \Sigma^* \to V^*$$

that maps an arbitrary string to a sequence of tokens from a finite vocabulary $V = \{v_1, v_2, \ldots, v_{|V|}\}$. Qwen3-4B uses a vocabulary of $|V| = 151{,}936$ tokens constructed via Byte-Pair Encoding (BPE) (Sennrich et al., 2016).

**BPE algorithm.** Starting from individual bytes (or characters), BPE iteratively merges the most frequent adjacent pair of tokens into a new token. Formally, at each step:

1. Compute pair frequencies: $\text{freq}(t_i, t_j) = |\{k : s_k = t_i \wedge s_{k+1} = t_j\}|$ over a training corpus $s$.
2. Merge: $(t^*, t^{**}) = \arg\max_{(t_i, t_j)} \text{freq}(t_i, t_j)$.
3. Replace all occurrences of $(t^*, t^{**})$ with a new token $t_{\text{new}}$ and add $t_{\text{new}}$ to $V$.
4. Repeat until $|V|$ reaches the desired size.

The result is a deterministic, lossless encoding where common words map to single tokens and rare words decompose into subword pieces.

### 1.2 Token Representation

Given an input string, the tokenizer produces a sequence of integer token IDs:

$$\mathbf{t} = (t_1, t_2, \ldots, t_T), \quad t_i \in \{0, 1, \ldots, |V|-1\}$$

where $T$ is the sequence length. Qwen3-4B supports sequences up to $T = 262{,}144$.

---

## 2. Embedding

### 2.1 Token Embedding

An **embedding matrix** $\mathbf{W}_E \in \mathbb{R}^{|V| \times d}$ maps each token ID to a dense vector in $d$-dimensional space, where $d = d_{\text{model}}$ is the hidden dimension of the transformer. For Qwen3-4B, $d = 2{,}560$.

$$\mathbf{e}_i = \mathbf{W}_E[t_i] \in \mathbb{R}^d$$

The full input embedding sequence is $\mathbf{E} = (\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_T) \in \mathbb{R}^{T \times d}$.

### 2.2 Rotary Position Embedding (RoPE)

Unlike absolute position embeddings that add a position-dependent vector to $\mathbf{e}_i$, Qwen3 uses **Rotary Position Embedding** (Su et al., 2021), which encodes position information as rotations in the complex plane applied within the attention mechanism.

For a vector $\mathbf{x} \in \mathbb{R}^d$, partition it into $d/2$ pairs $(x_{2k}, x_{2k+1})$ for $k = 0, \ldots, d/2 - 1$. Define frequency bases:

$$\theta_k = 10{,}000^{-2k/d}$$

The RoPE transformation at position $m$ is:

$$\text{RoPE}(\mathbf{x}, m)_{2k} = x_{2k} \cos(m \theta_k) - x_{2k+1} \sin(m \theta_k)$$
$$\text{RoPE}(\mathbf{x}, m)_{2k+1} = x_{2k} \sin(m \theta_k) + x_{2k+1} \cos(m \theta_k)$$

Equivalently, treating each pair as a complex number $z_k = x_{2k} + i\, x_{2k+1}$:

$$\text{RoPE}(z_k, m) = z_k \cdot e^{i\, m\theta_k}$$

**Key property:** The inner product $\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle$ depends only on the relative position $m - n$, giving the model translation-invariant attention patterns without explicit position vectors in the residual stream.

---

## 3. Transformer Architecture

### 3.1 The Residual Stream

The central organizing principle of the transformer is the **residual stream** (Elhage et al., 2021). The hidden state at position $i$ after layer $\ell$ is:

$$\mathbf{x}_i^{(\ell)} = \mathbf{x}_i^{(\ell-1)} + \text{Attn}^{(\ell)}(\mathbf{x}^{(\ell-1)})_i + \text{MLP}^{(\ell)}(\mathbf{x}^{(\ell-1)})_i$$

with $\mathbf{x}_i^{(0)} = \mathbf{e}_i$. Each layer reads from and writes to this shared communication channel. Qwen3-4B has $L = 36$ layers (to be confirmed at runtime).

### 3.2 Layer Normalization (RMSNorm)

Before each attention and MLP block, Qwen3 applies **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

where

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$

and $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learnable scale parameter. Unlike standard LayerNorm, RMSNorm does not subtract the mean, which simplifies computation and improves stability. $\epsilon > 0$ (typically $10^{-6}$) prevents division by zero.

### 3.3 Multi-Head Self-Attention

#### 3.3.1 Query, Key, Value Projections

For layer $\ell$ with $H$ attention heads (Qwen3-4B uses $H = 32$) and head dimension $d_h = d/H = 80$, each head $h$ computes:

$$\mathbf{q}_i^{(h)} = \mathbf{W}_Q^{(h)} \cdot \text{RMSNorm}(\mathbf{x}_i^{(\ell-1)}) \in \mathbb{R}^{d_h}$$
$$\mathbf{k}_i^{(h)} = \mathbf{W}_K^{(h)} \cdot \text{RMSNorm}(\mathbf{x}_i^{(\ell-1)}) \in \mathbb{R}^{d_h}$$
$$\mathbf{v}_i^{(h)} = \mathbf{W}_V^{(h)} \cdot \text{RMSNorm}(\mathbf{x}_i^{(\ell-1)}) \in \mathbb{R}^{d_h}$$

where $\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)} \in \mathbb{R}^{d_h \times d}$.

**Grouped-Query Attention (GQA).** Qwen3-4B uses GQA with $H_{\text{KV}} = 8$ key-value heads shared across groups of $H / H_{\text{KV}} = 4$ query heads. This reduces the KV cache memory from $O(H \cdot T \cdot d_h)$ to $O(H_{\text{KV}} \cdot T \cdot d_h)$ without significant quality loss.

#### 3.3.2 RoPE Application

RoPE is applied to queries and keys (but not values) before computing attention:

$$\hat{\mathbf{q}}_i^{(h)} = \text{RoPE}(\mathbf{q}_i^{(h)}, i), \quad \hat{\mathbf{k}}_j^{(h)} = \text{RoPE}(\mathbf{k}_j^{(h)}, j)$$

#### 3.3.3 Scaled Dot-Product Attention

The attention weights for head $h$ at position $i$ attending to position $j$ are:

$$\alpha_{ij}^{(h)} = \frac{\exp\left(\hat{\mathbf{q}}_i^{(h)\top} \hat{\mathbf{k}}_j^{(h)} / \sqrt{d_h}\right)}{\sum_{j'=1}^{i} \exp\left(\hat{\mathbf{q}}_i^{(h)\top} \hat{\mathbf{k}}_{j'}^{(h)} / \sqrt{d_h}\right)}$$

The causal mask ensures $\alpha_{ij}^{(h)} = 0$ for $j > i$ (autoregressive constraint). The scaling factor $\sqrt{d_h}$ prevents the dot products from growing too large in magnitude, which would push the softmax into regions of extremely small gradient.

#### 3.3.4 Attention Output

$$\text{Attn}^{(\ell)}(\mathbf{x})_i = \sum_{h=1}^{H} \mathbf{W}_O^{(h)} \left( \sum_{j=1}^{i} \alpha_{ij}^{(h)} \mathbf{v}_j^{(h)} \right)$$

where $\mathbf{W}_O^{(h)} \in \mathbb{R}^{d \times d_h}$ is the output projection for head $h$.

### 3.4 MLP Block (SwiGLU)

Qwen3 uses the **SwiGLU** activation function (Shazeer, 2020), a gated variant that outperforms standard ReLU or GELU MLPs:

$$\text{MLP}^{(\ell)}(\mathbf{x})_i = \mathbf{W}_{\text{down}} \left[ \left(\mathbf{W}_{\text{gate}} \cdot \mathbf{n}_i\right) \odot \sigma\left(\mathbf{W}_{\text{up}} \cdot \mathbf{n}_i\right) \right]$$

where $\mathbf{n}_i = \text{RMSNorm}(\mathbf{x}_i^{(\ell-1)} + \text{Attn}^{(\ell)}_i)$ is the post-attention normalized representation, and:

- $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d}$: gate projection
- $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d}$: up projection
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}$: down projection
- $d_{\text{ff}}$: intermediate dimension (typically $\approx 8/3 \cdot d$ for SwiGLU)
- $\sigma(\cdot) = \text{SiLU}(\cdot) = x \cdot \text{sigmoid}(x)$: the Sigmoid Linear Unit
- $\odot$: element-wise multiplication (gating)

The SiLU activation is defined as:

$$\text{SiLU}(x) = x \cdot \frac{1}{1 + e^{-x}}$$

Note that SwiGLU uses two separate weight matrices ($\mathbf{W}_{\text{gate}}$ and $\mathbf{W}_{\text{up}}$) for the gating mechanism, which increases the parameter count of the MLP by 50% compared to a standard two-matrix MLP, but empirically yields better performance.

### 3.5 Full Transformer Layer

Combining the above, a single layer computes (with pre-norm architecture):

$$\mathbf{a}_i^{(\ell)} = \mathbf{x}_i^{(\ell-1)} + \text{Attn}^{(\ell)}\big(\text{RMSNorm}(\mathbf{x}^{(\ell-1)})\big)_i$$
$$\mathbf{x}_i^{(\ell)} = \mathbf{a}_i^{(\ell)} + \text{MLP}^{(\ell)}\big(\text{RMSNorm}(\mathbf{a}^{(\ell)})\big)_i$$

This pre-norm formulation (applying normalization before each sub-layer rather than after) improves training stability, especially in deep networks.

---

## 4. Next-Token Prediction

### 4.1 Unembedding and Logits

After the final layer $L$, a final RMSNorm and the **unembedding matrix** $\mathbf{W}_U \in \mathbb{R}^{|V| \times d}$ map the residual stream to a distribution over the vocabulary:

$$\boldsymbol{\ell}_i = \mathbf{W}_U \cdot \text{RMSNorm}(\mathbf{x}_i^{(L)}) \in \mathbb{R}^{|V|}$$

The vector $\boldsymbol{\ell}_i$ is the **logit vector** at position $i$. Each component $\ell_{i,v}$ measures the model's unnormalized preference for token $v$ as the next token.

### 4.2 Probability Distribution

The predicted probability of token $v$ at position $i$ is:

$$P(t_{i+1} = v \mid t_1, \ldots, t_i) = \text{softmax}(\boldsymbol{\ell}_i)_v = \frac{\exp(\ell_{i,v})}{\sum_{v'=1}^{|V|} \exp(\ell_{i,v'})}$$

### 4.3 Training Objective

The model is trained to minimize the cross-entropy loss over a corpus:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{T} \sum_{i=1}^{T} \log P(t_{i+1} = t_{i+1}^* \mid t_1, \ldots, t_i)$$

where $t_{i+1}^*$ is the ground-truth next token.

### 4.4 Logit Difference as Behavioural Metric

For our experiments, we evaluate model behaviour using the **logit difference** between a correct answer token $t^+$ and an incorrect answer token $t^-$:

$$\Delta \ell = \ell_{T, t^+} - \ell_{T, t^-}$$

A positive $\Delta \ell$ indicates the model prefers the correct token. We require $\Delta \ell > \tau$ (a behaviour-dependent threshold; see Table 1) to confirm the model "clearly" performs the behaviour.

| Behaviour | Correct ($t^+$) | Incorrect ($t^-$) | Threshold $\tau$ |
|-----------|-----|-----|---|
| Grammar agreement | "are" / "is" | "is" / "are" | 2.0 |
| Factual recall | correct capital | distractor capital | 2.0 |
| Sentiment continuation | matching sentiment word | opposing sentiment word | 1.5 |
| Arithmetic | correct sum digit | plausible wrong digit | 2.0 |

---

## 5. Superposition and the Need for Sparse Autoencoders

### 5.1 The Superposition Hypothesis

A transformer with hidden dimension $d$ has $d$ dimensions available per residual stream position. However, the number of distinct **features** (concepts, facts, syntactic roles, etc.) that the model needs to represent vastly exceeds $d$. Elhage et al. (2022) and Anthropic (Bricken et al., 2023) hypothesized that models represent more features than dimensions by **superposing** them — storing $M \gg d$ features as approximately orthogonal directions in $\mathbb{R}^d$.

Formally, suppose the model needs to represent $M$ features $f_1, \ldots, f_M$ with associated directions $\mathbf{d}_1, \ldots, \mathbf{d}_M \in \mathbb{R}^d$. When $M > d$, these cannot be mutually orthogonal. Superposition exploits the fact that most features are sparse (rarely active simultaneously): if at most $k \ll M$ features are active at any time, approximate recovery is possible when the directions are **incoherent** (nearly orthogonal):

$$\max_{i \neq j} |\mathbf{d}_i^\top \mathbf{d}_j| < \delta$$

for small $\delta > 0$. The interference from inactive features contributes noise proportional to $\delta \sqrt{k}$.

### 5.2 Polysemanticity

A consequence of superposition is **polysemanticity**: individual neurons (components of $\mathbf{x}$) respond to multiple, seemingly unrelated concepts. This makes direct interpretation of neurons unreliable. The goal of sparse autoencoders is to "undo" the superposition by projecting activations into a higher-dimensional space where each direction corresponds to a single feature.

### 5.3 Implications for Scale

At smaller model scales ($d = 2{,}560$ for Qwen3-4B vs. $d \geq 8{,}192$ for models with 70B+ parameters), the degree of superposition is expected to be higher:

- **More polysemantic neurons**: each dimension must serve more features.
- **Higher feature interference**: the incoherence bound $\delta$ is larger, making decomposition harder.
- **Less redundancy**: fewer duplicate features encoding the same concept.

These considerations directly affect the quality of SAE decompositions at our scale.

---

## 6. Sparse Autoencoders

### 6.1 Architecture

A **Sparse Autoencoder** (SAE) maps activations from $\mathbb{R}^d$ to a sparse code in $\mathbb{R}^m$ (where $m = \kappa \cdot d$ with expansion factor $\kappa$; we use $\kappa = 4$, giving $m = 10{,}240$) and back to $\mathbb{R}^d$.

Given an activation vector $\mathbf{x} \in \mathbb{R}^d$:

**Step 1 — Centering (pre-bias subtraction):**

$$\tilde{\mathbf{x}} = \mathbf{x} - \mathbf{b}_{\text{pre}}$$

where $\mathbf{b}_{\text{pre}} \in \mathbb{R}^d$ is a learnable bias that approximates the mean activation, centering the data.

**Step 2 — Encoding:**

$$\mathbf{z} = \text{ReLU}\!\left(\mathbf{W}_{\text{enc}} \tilde{\mathbf{x}} + \mathbf{b}_{\text{enc}}\right)$$

where $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{m \times d}$ is the encoder weight matrix and $\mathbf{b}_{\text{enc}} \in \mathbb{R}^m$ is the encoder bias. The ReLU activation $\text{ReLU}(x) = \max(0, x)$ enforces non-negativity, contributing to sparsity.

Each component $z_i$ is called a **feature activation**. The index $i$ identifies the feature, and $z_i > 0$ means feature $i$ is "active" for this input.

**Step 3 — Decoding (reconstruction):**

$$\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{pre}}$$

where $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d \times m}$ is the decoder weight matrix. Note that $\mathbf{b}_{\text{pre}}$ is added back, so the decoder reconstructs the original (uncentered) activation.

The decoder can also be written as a sum over active features:

$$\hat{\mathbf{x}} = \sum_{i=1}^m z_i \, \mathbf{d}_i + \mathbf{b}_{\text{pre}}$$

where $\mathbf{d}_i \in \mathbb{R}^d$ is the $i$-th column of $\mathbf{W}_{\text{dec}}$ — the **feature direction** in activation space.

### 6.2 Loss Function

The SAE is trained to minimize:

$$\mathcal{L}_{\text{SAE}} = \underbrace{\frac{1}{N}\sum_{n=1}^N \|\mathbf{x}_n - \hat{\mathbf{x}}_n\|_2^2}_{\text{reconstruction (MSE)}} + \underbrace{\lambda \cdot \frac{1}{N}\sum_{n=1}^N \|\mathbf{z}_n\|_1}_{\text{sparsity (L1)}}$$

where $\lambda > 0$ is the sparsity coefficient (we use $\lambda = 0.005$).

**Reconstruction term:** Ensures the autoencoder faithfully captures the information in $\mathbf{x}$.

**Sparsity term:** The L1 norm $\|\mathbf{z}\|_1 = \sum_i |z_i| = \sum_i z_i$ (since $z_i \geq 0$ after ReLU) penalizes the total magnitude of active features, encouraging most $z_i$ to be zero. This is the key mechanism that forces the SAE to find a sparse, interpretable decomposition rather than a trivially dense one.

### 6.3 Decoder Normalization Constraint

After each gradient step, we normalize each column of $\mathbf{W}_{\text{dec}}$ to unit norm:

$$\mathbf{d}_i \leftarrow \frac{\mathbf{d}_i}{\|\mathbf{d}_i\|_2 + \epsilon}$$

Without this constraint, the SAE could satisfy the sparsity penalty by shrinking $z_i$ while proportionally scaling up $\mathbf{d}_i$, which defeats the purpose of the L1 penalty. Unit-norm columns ensure that $z_i$ represents the true magnitude of feature $i$'s contribution.

### 6.4 Training Details

- **Optimizer:** Adam (Kingma & Ba, 2015) with learning rate $\eta = 3 \times 10^{-4}$
- **Batch size:** 256 activation vectors
- **Data:** residual stream activations at layers 10–24, last 5 token positions per prompt
- **Initialization:** Xavier uniform (Glorot & Bengio, 2010) for weights; zeros for biases
- **Early stopping:** when validation $R^2 \geq 0.85$
- **Maximum steps:** 50,000

### 6.5 Quality Metrics

**Coefficient of determination ($R^2$):**

$$R^2 = 1 - \frac{\sum_n \|\mathbf{x}_n - \hat{\mathbf{x}}_n\|^2}{\sum_n \|\mathbf{x}_n - \bar{\mathbf{x}}\|^2}$$

where $\bar{\mathbf{x}}$ is the mean activation vector. $R^2 = 1$ means perfect reconstruction; $R^2 = 0$ means the SAE is no better than predicting the mean. We target $R^2 \geq 0.85$.

**Sparsity ($L_0$):**

$$L_0 = \frac{1}{N \cdot m} \sum_{n=1}^N \sum_{i=1}^m \mathbb{1}[z_{n,i} > 0]$$

This measures the fraction of features active on average. Lower is sparser. Typical values: $L_0 \in [0.01, 0.05]$.

**Dead feature fraction:**

$$f_{\text{dead}} = \frac{1}{m} \sum_{i=1}^m \mathbb{1}\!\left[\max_n z_{n,i} = 0\right]$$

Features that never activate on any input are "dead" and waste capacity. We require $f_{\text{dead}} < 0.20$.

---

## 7. Cross-Layer Transcoders: Anthropic's Approach

To contextualize our methodology, we describe Anthropic's more sophisticated architecture.

### 7.1 CLT Architecture

A **Cross-Layer Transcoder** (CLT) differs from a standard SAE in two key ways:

1. **Single-layer encoder, multi-layer decoder.** The encoder reads from one layer, but the decoder writes to multiple downstream layers.
2. **JumpReLU activation** instead of ReLU.

Formally, given input $\mathbf{x}^{(\ell)}$ from layer $\ell$:

$$\mathbf{z} = \text{JumpReLU}_\theta\!\left(\mathbf{W}_{\text{enc}} (\mathbf{x}^{(\ell)} - \mathbf{b}_{\text{pre}}) + \mathbf{b}_{\text{enc}}\right)$$

where JumpReLU is defined as:

$$\text{JumpReLU}_\theta(x) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{otherwise} \end{cases}$$

The threshold $\theta > 0$ is a learnable parameter. Unlike ReLU (which is JumpReLU with $\theta = 0$), JumpReLU creates a hard gap between zero and the smallest non-zero activation, producing cleaner sparsity patterns.

The decoder writes to multiple layers $\ell' > \ell$:

$$\hat{\mathbf{x}}^{(\ell')} = \mathbf{W}_{\text{dec}}^{(\ell')} \mathbf{z} + \mathbf{b}_{\text{pre}}^{(\ell')}$$

### 7.2 CLT Training Loss

Anthropic trains CLTs using a combined loss with a tanh-based sparsity penalty:

$$\mathcal{L}_{\text{CLT}} = \sum_{\ell'} \left\| \mathbf{x}^{(\ell')} - \hat{\mathbf{x}}^{(\ell')} \right\|_2^2 + \lambda \sum_i \tanh\!\left(\frac{c \cdot z_i}{\theta_i}\right)$$

where $c$ is a scaling constant. The $\tanh$ penalty is approximately linear for small activations (like L1) but saturates for large activations, avoiding excessive penalization of strongly active features.

### 7.3 Comparison: Our SAEs vs. CLTs

| Aspect | Our SAEs | Anthropic CLTs |
|--------|----------|----------------|
| Encoder input | Single layer | Single layer |
| Decoder output | Same layer | Multiple downstream layers |
| Activation | ReLU | JumpReLU (learnable threshold) |
| Sparsity penalty | L1 | tanh-based |
| Features | $\sim$10K per layer | $\sim$30M total |
| Cross-layer info | Not captured directly | Native (multi-layer decoder) |

**Impact on our work:** Because our SAEs decompose one layer at a time, cross-layer dependencies must be inferred via gradient-based attribution between layers (§8), which is a first-order approximation of the direct cross-layer decomposition that CLTs provide.

---

## 8. Feature Interpretation

### 8.1 Top-Activating Examples

For each feature $i$, we compute $z_i$ over all prompts and token positions, then rank by activation magnitude:

$$\text{TopExamples}(i, K) = \text{argtop}_K \{z_{n,i}\}_{n=1}^N$$

By inspecting the top-$K$ examples, we assign a human-readable label to each feature.

### 8.2 Feature–Category Association

For behaviours with categorical metadata (e.g., singular vs. plural for grammar), we measure feature selectivity:

$$\text{Selectivity}(i, c) = \frac{\mathbb{E}[z_i \mid \text{category} = c]}{\mathbb{E}[z_i]}$$

A feature with $\text{Selectivity}(i, c) \gg 1$ is strongly associated with category $c$.

### 8.3 Polysemanticity Score

A feature is **monosemantic** if it activates for a single coherent concept. We quantify polysemanticity via the entropy of the category distribution over top-activating examples:

$$H(i) = -\sum_c p_c^{(i)} \log p_c^{(i)}$$

where $p_c^{(i)}$ is the fraction of top-$K$ examples for feature $i$ belonging to category $c$. Low entropy indicates monosemanticity.

---

## 9. Attribution Graphs

### 9.1 Overview

An **attribution graph** is a directed acyclic graph $G = (\mathcal{N}, \mathcal{E})$ that traces the causal pathway from input tokens through intermediate features to output logits. Nodes represent input tokens, SAE features at various layers, and output logits. Edges represent directed influence, weighted by attribution scores.

### 9.2 Gradient-Based Attribution (Our Method)

We compute a first-order approximation of each feature's contribution to the target logit.

**Step 1: Compute the target logit gradient.** For the logit difference $\Delta \ell = \ell_{t^+} - \ell_{t^-}$, compute the gradient of this scalar with respect to the residual stream at layer $\ell$:

$$\mathbf{g}^{(\ell)} = \frac{\partial \Delta \ell}{\partial \mathbf{x}^{(\ell)}} \in \mathbb{R}^d$$

This requires a backward pass from the output through all layers above $\ell$.

**Step 2: Project into feature space.** The gradient tells us how each dimension of $\mathbf{x}^{(\ell)}$ affects $\Delta \ell$. To attribute to SAE features, we project through the encoder:

$$g_i^{\text{feat}} = \left[\mathbf{W}_{\text{enc}} \left(\mathbf{g}^{(\ell)} - \mathbf{b}_{\text{pre}}\right) + \mathbf{b}_{\text{enc}}\right]_i$$

**Step 3: Attribution score.** The attribution of feature $i$ at layer $\ell$ to the target logit is:

$$a_i^{(\ell)} = z_i \cdot g_i^{\text{feat}}$$

This is the **activation × gradient** formula: $z_i$ measures the feature's current activation strength, and $g_i^{\text{feat}}$ measures how sensitive the output is to this feature. Their product gives a first-order estimate of the feature's contribution.

**Justification via Taylor expansion:** For a differentiable function $f$, the first-order Taylor expansion gives:

$$f(\mathbf{z}) \approx f(\mathbf{0}) + \sum_i z_i \frac{\partial f}{\partial z_i}\bigg|_{\mathbf{z}}$$

The term $z_i \cdot \partial f / \partial z_i$ is exactly our attribution score, representing the linear contribution of feature $i$ to the output.

### 9.3 Feature-to-Feature Attribution (Virtual Weights)

To build edges between features at adjacent layers $\ell$ and $\ell+1$, we use the **virtual weight matrix**:

$$\mathbf{V} = \mathbf{W}_{\text{enc}}^{(\ell+1)} \cdot \mathbf{W}_{\text{dec}}^{(\ell)\top} \in \mathbb{R}^{m \times m}$$

This matrix approximates the linear pathway from the decoder of the source SAE (which writes feature directions into the residual stream) to the encoder of the target SAE (which reads from the residual stream).

The attribution from source feature $j$ (layer $\ell$) to target feature $i$ (layer $\ell+1$) is:

$$a_{j \to i} = z_j^{(\ell)} \cdot V_{ij} \cdot z_i^{(\ell+1)}$$

This three-way product captures: (1) the source is active ($z_j$), (2) there exists a pathway ($V_{ij}$), and (3) the target is also active ($z_i$).

### 9.4 Anthropic's Backward Jacobian Method

For comparison, Anthropic's attribution method uses **backward Jacobians with frozen nonlinearities** (stop-gradients). Given the complete computation graph with CLT features:

$$\text{Attribution}(j \to i) = z_j \cdot \left[\frac{\partial z_i}{\partial z_j}\bigg|_{\text{frozen}}\right] \cdot z_i$$

where the Jacobian $\partial z_i / \partial z_j$ is computed with all nonlinearities (ReLU/JumpReLU gates, attention softmax) frozen at their forward-pass values. This "linearizes" the network around the current input, giving exact (not approximate) linear attribution through the frozen computation graph.

Additionally, Anthropic includes **error nodes** that account for the SAE reconstruction error at each layer, ensuring that unmodeled variance is explicitly tracked.

**Our approximation vs. Anthropic's method:**
- We use standard backpropagation (nonlinearities are not frozen), introducing second-order effects.
- We do not include error nodes, so reconstruction error is unaccounted.
- We attribute layer-by-layer rather than through a unified cross-layer decoder.
- We expect noisier graphs with potentially more false edges as a result.

### 9.5 Graph Construction Algorithm

Given a set of $N$ prompts for a behaviour:

1. **Per-prompt graph:** For each prompt $n$:
   - Forward pass to obtain $\mathbf{x}_n^{(\ell)}$ at all layers $\ell \in [10, 24]$.
   - Encode through SAEs: $\mathbf{z}_n^{(\ell)} = \text{SAE}^{(\ell)}_{\text{enc}}(\mathbf{x}_n^{(\ell)})$.
   - Backward pass: compute $\mathbf{g}_n^{(\ell)}$ for all layers.
   - Compute feature-to-output attributions: $a_{n,i}^{(\ell)} = z_{n,i}^{(\ell)} \cdot g_{n,i}^{(\ell, \text{feat})}$.
   - Compute feature-to-feature attributions via virtual weights.
   - Add nodes and edges above threshold to prompt-level graph.

2. **Aggregation:** Across $N$ prompts, retain features that appear in at least $\lfloor 0.2 N \rfloor$ prompt-level graphs (20% frequency threshold). Edge weights are averaged over prompts where both endpoints are active.

3. **Pruning:** For each node, keep only the top-$k$ outgoing edges by absolute attribution ($k = 10$). Remove edges below the absolute threshold $\tau_{\text{attr}} = 0.01$. Remove isolated feature nodes (degree 0).

### 9.6 Graph Properties

The resulting graph $G = (\mathcal{N}, \mathcal{E})$ has three types of nodes:

- **Input nodes** $\mathcal{N}_{\text{in}} = \{t_1, \ldots, t_T\}$: token embeddings at input positions.
- **Feature nodes** $\mathcal{N}_{\text{feat}} = \{(ℓ, i) : z_i^{(ℓ)} > 0\}$: active SAE features across layers.
- **Output nodes** $\mathcal{N}_{\text{out}} = \{t^+, t^-\}$: target logit tokens.

Edges flow forward through layers: $\mathcal{N}_{\text{in}} \to \mathcal{N}_{\text{feat}}^{(10)} \to \cdots \to \mathcal{N}_{\text{feat}}^{(24)} \to \mathcal{N}_{\text{out}}$.

---

## 10. Causal Interventions

Attribution graphs identify correlational structure. To establish **causal** relationships, we intervene on the model's internal computations and measure the effect on behaviour. We implement three intervention types.

### 10.1 Feature Ablation

**Zero ablation:** Set a feature's activation to zero and observe the effect on the logit difference.

$$z_i' = 0$$

$$\hat{\mathbf{x}}_{\text{ablated}} = \mathbf{W}_{\text{dec}} \mathbf{z}' + \mathbf{b}_{\text{pre}}$$

The effect is measured as:

$$\Delta_{\text{ablation}} = \Delta \ell_{\text{original}} - \Delta \ell_{\text{ablated}}$$

If $\Delta_{\text{ablation}} > 0$, the ablated feature was contributing positively to the correct answer. If $\Delta_{\text{ablation}} \approx 0$, the feature is irrelevant to the behaviour.

**Inhibition:** A stronger intervention that negates the feature:

$$z_i' = -\alpha \cdot z_i, \quad \alpha > 0$$

This not only removes the feature's contribution but actively pushes in the opposite direction. By varying $\alpha$, we can measure the dose-response curve of the intervention.

### 10.2 Activation Patching

Given a **source prompt** $A$ and a **target prompt** $B$ that differ in some property (e.g., singular vs. plural subject for grammar agreement), we swap specific features from $B$ into $A$'s computation:

**Step 1:** Encode both activations:

$$\mathbf{z}_A = \text{SAE}_{\text{enc}}(\mathbf{x}_A), \quad \mathbf{z}_B = \text{SAE}_{\text{enc}}(\mathbf{x}_B)$$

**Step 2:** Construct patched activation for a feature set $S$:

$$z_{\text{patched}, i} = \begin{cases} z_{B,i} & \text{if } i \in S \\ z_{A,i} & \text{otherwise} \end{cases}$$

**Step 3:** Decode and continue the forward pass:

$$\hat{\mathbf{x}}_{\text{patched}} = \mathbf{W}_{\text{dec}} \mathbf{z}_{\text{patched}} + \mathbf{b}_{\text{pre}}$$

**Step 4:** Measure the effect:

$$\Delta_{\text{patch}} = \Delta \ell_{\text{patched}} - \Delta \ell_A$$

If patching features from a plural prompt into a singular prompt causes the model to predict a plural verb, those features causally encode number information.

### 10.3 Feature Steering

Rather than ablating or swapping, we can **inject** feature activations to steer behaviour:

$$z_i' = z_i + \delta$$

where $\delta > 0$ amplifies the feature and $\delta < 0$ suppresses it. This tests whether activating a feature is **sufficient** (not just necessary) for a behaviour.

### 10.4 Control Experiments

To establish that intervention effects are specific to the attributed features and not artifacts:

1. **Random ablation control:** Ablate the same number of randomly selected features and measure the effect. The attributed features should produce significantly larger effects.

2. **Baseline logit difference:** Measure $\Delta \ell$ without any intervention to establish the reference.

3. **Effect normalization:**

$$\text{Relative effect} = \frac{\Delta_{\text{intervention}}}{\Delta \ell_{\text{baseline}}}$$

A relative effect of 1.0 means the intervention completely eliminates the behaviour.

---

## 11. Evaluation Framework

### 11.1 Completeness Score

Following Anthropic, the **completeness** of a circuit measures what fraction of the model's behaviour is explained by the identified features:

$$\text{Completeness} = 1 - \frac{\Delta \ell_{\text{after ablating all circuit features}}}{\Delta \ell_{\text{baseline}}}$$

Completeness $= 1$ means the circuit fully explains the behaviour. Completeness $= 0$ means the circuit explains none of it.

### 11.2 Minimality

A circuit is **minimal** if removing any single feature significantly reduces the completeness. We measure this by ablating features one at a time and checking that each removal has a non-negligible effect.

### 11.3 Faithfulness

The circuit is **faithful** if the attribution scores predict the intervention effects. We measure the correlation between:

- Attribution magnitude $|a_i|$ for each feature $i$
- Ablation effect $|\Delta_{\text{ablation}, i}|$ for each feature $i$

A Spearman rank correlation $\rho > 0.5$ indicates that the attribution graph is a reliable predictor of causal importance.

---

## 12. Scale Considerations: 4B vs. 70B+

### 12.1 Hypotheses

We hypothesize four scale-dependent effects:

1. **Higher polysemanticity.** With $d = 2{,}560$ vs. $d \geq 8{,}192$, Qwen3-4B must superpose more features per dimension, leading to more polysemantic SAE features (higher entropy $H(i)$ on average).

2. **Shallower circuits.** With 36 layers vs. 80+, there are fewer layers available for multi-step reasoning, so circuits may involve fewer intermediate features.

3. **Less redundancy.** Larger models often have multiple features encoding the same concept. At 4B, we expect fewer redundant features and more single-point-of-failure circuits.

4. **Weaker intervention effects.** Smaller models may be more robust to single-feature ablation if information is distributed across more polysemantic features, or conversely, less robust if circuits lack redundancy.

### 12.2 Measurable Indicators

| Indicator | Expected at 4B | Expected at 70B+ |
|-----------|----------------|-------------------|
| Mean polysemanticity entropy $\bar{H}$ | Higher | Lower |
| Circuit depth (max graph path length) | Shorter | Longer |
| Features per concept (redundancy) | Fewer | More |
| Single-feature ablation effect | Variable | Smaller |
| SAE dead feature fraction $f_{\text{dead}}$ | Potentially higher | Lower |
| SAE $R^2$ at same expansion factor | Potentially lower | Higher |

---

## 13. Summary of the Complete Pipeline

The mathematical pipeline from raw text to causal circuit analysis is:

$$\boxed{\text{String} \xrightarrow{\text{BPE}} \text{Tokens} \xrightarrow{\mathbf{W}_E} \text{Embeddings} \xrightarrow{\text{RoPE + Attn + MLP} \times L} \text{Residual stream} \xrightarrow{\mathbf{W}_U} \text{Logits} \xrightarrow{\text{softmax}} P(t_{\text{next}})}$$

At selected layers, we branch off the residual stream:

$$\mathbf{x}^{(\ell)} \xrightarrow{\text{SAE}_{\text{enc}}} \mathbf{z}^{(\ell)} \xrightarrow{\text{interpret}} \text{Features} \xrightarrow{\text{attribution}} \text{Graph } G \xrightarrow{\text{intervene}} \text{Causal validation}$$

Each arrow corresponds to a mathematically defined operation described in this chapter, and each is implemented in our experimental pipeline.

---

## References

1. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic*.
2. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024*.
3. Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
4. Elhage, N., et al. (2022). Toy Models of Superposition. *Anthropic*.
5. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
6. Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
7. Lindsey, J., Gurnee, W., et al. (2025). On the Biology of a Large Language Model. *Anthropic*.
8. Sennrich, R., Haddow, B. & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*.
9. Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv:2002.05202*.
10. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
11. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
12. Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.
