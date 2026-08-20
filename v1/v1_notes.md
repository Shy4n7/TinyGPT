# V1: TinyGPT - Mathematical Foundations

## Why I Built V1
I wanted to build a decoder-only GPT model from raw PyTorch operations rather than using high-level abstractions. V1 focuses on implementing `MultiHeadAttention`, `FeedForward`, and `TransformerBlock` while validating the core math behind causal masking and scaled dot-product attention.

---

## Model Specifications
- **Parameters**: 112,577 (~112k)
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
3. **Transformer Blocks (x2)**:
   - Multi-Head Attention: $4 \times d_{model} \times d_{model} = 16,384$ (plus biases)
   - LayerNorms: $2 \times (2 \times 64) = 256$
   - FFN: $2 \times (64 \times 256 + 256) + (256 \times 64 + 64) = 33,088$
   - Total per block $\approx 49,984$ parameters.
4. **Final LayerNorm & LM Head**: $2 \times 64 + 64 \times 65 + 65 = 4,353$

---

## Key Mathematical Observations I Made

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- Scaling by $\sqrt{d_k} = \sqrt{16} = 4.0$ prevented dot-product scores from growing too large, avoiding extreme softmax saturation.

### 2. Causal Masking
Setting $M_{i,j} = -\infty$ for $j > i$ ensured tokens could only attend to past and current positions, maintaining the auto-regressive property.

### 3. Pre-Norm vs Post-Norm
Using Pre-LayerNorm ($x + \text{Attention}(\text{LN}(x))$) provided an uninhibited identity shortcut for gradient propagation during backpropagation.

---

## Why I Chose Epoch-Based Training for V1
- **Choice**: I used an **epoch-based training loop** (20 epochs) with `DataLoader`.
- **Reason**: Epochs made it straightforward to verify that every single character sequence in the dataset was passed through the model during training.
- **Trade-off I Discovered**: Fixed window chunking in standard data loaders limits sample diversity across epochs, and loss updates happen in rigid passes.

---

## What I Learned from Results
- **Train Loss**: $4.17 \rightarrow 1.40$
- **Val Loss**: $4.17 \rightarrow 1.43$
- **Observation**: Train and validation loss tracked closely together ($\Delta \approx 0.03$). The model suffered from **underfitting**—it was simply too small (112k params) to capture complex text structure.
