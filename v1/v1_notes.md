# V1: TinyGPT - Mathematical Foundations

## Model Overview
- **Parameters**: 129,089 (~129k)
- **Embedding Dimension ($d_{model}$)**: 64
- **Attention Heads ($h$)**: 4
- **Head Dimension ($d_k$)**: 16 ($64 / 4$)
- **Layers / Blocks ($N$)**: 2
- **Context Length ($T$)**: 64
- **Feed-Forward Hidden Dim**: 256 ($4 \times 64$)

---

## Parameter Breakdown Math
1. **Token Embeddings**: $V \times d_{model} = 65 \times 64 = 4,160$
2. **Position Embeddings**: $T \times d_{model} = 64 \times 64 = 4,096$
3. **Transformer Block (x2)**:
   - Multi-Head Attention: $4 \times d_{model} \times d_{model} = 4 \times 64 \times 64 = 16,384$ (plus $4 \times 64 = 256$ biases)
   - LayerNorms: $2 \times (2 \times 64) = 256$
   - FFN: $2 \times (64 \times 256 + 256) + (256 \times 64 + 64) = 33,088$
   - Total per block $\approx 49,984$ parameters.
4. **Final LayerNorm & LM Head**: $2 \times 64 + 64 \times 65 + 65 = 4,353$

---

## Key Mathematical Observations

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- $\sqrt{d_k} = \sqrt{16} = 4.0$. Scaling scores by $1/4$ prevents dot products from growing excessively large, avoiding extreme softmax gradients that cause vanishing gradients early in training.

### 2. Causal Masking
The triangular mask $M_{i,j}$ sets $M_{i,j} = -\infty$ for $j > i$. This forces probabilities to $0.0$ for future tokens, preserving the auto-regressive constraint.

### 3. Pre-Norm Residual Architecture
Using Pre-LayerNorm:
$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$
allows direct gradient flow through the identity shortcut $x_l$, stabilizing optimization without warmups.

---

## Training Performance
- **Dataset**: Tiny Shakespeare (~1.1M characters, character-level tokenization)
- **Train Loss Progression**: $4.17 \xrightarrow{} 1.40$
- **Val Loss Progression**: $4.17 \xrightarrow{} 1.43$
- **Observation**: Train loss and validation loss closely track each other ($\Delta \approx 0.03$), indicating severe **underfitting** due to high bias/limited capacity rather than overfitting.

---

## What We Learned
- The 129k parameter model captures basic character frequencies and vowel/consonant alternating structures, but lacks context capacity to generate long-range coherent words or grammar.
- Needs scaling up depth, width, and context length in V2.
