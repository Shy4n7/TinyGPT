# V2: TinyGPT - Scaling Model Capacity

## Why I Scaled to V2
In V1, the model plateaued at a validation loss of 1.43 due to underfitting. In V2, I wanted to test if increasing parameter capacity alone would improve text coherence and lower perplexity.

---

## Architecture Scaling Changes
- **Parameters**: 818,241 (~818k, 7x scale over V1)
- **Embedding Dimension ($d_{model}$)**: 128 (doubled from 64)
- **Attention Heads ($h$)**: 8 (doubled from 4)
- **Layers / Blocks ($N$)**: 4 (doubled from 2)
- **Feed-Forward Hidden Dim**: 512 ($4 \times 128$)
- **Context Length ($T$)**: 64

---

## What Changed: V1 vs V2 Comparison

| Feature / Metric | V1 (Small) | V2 (Scaled) | What I Observed |
|:---|:---:|:---:|:---|
| **Parameter Count** | 112k | 818k | 7x parameter scaling |
| **Embedding Dim** | 64 | 128 | 2x wider feature space |
| **Blocks / Layers** | 2 | 4 | 2x deeper representation |
| **Final Train Loss** | 1.40 | 1.21 | **-0.19 (-13.6%) drop** |
| **Final Val Loss** | 1.43 | 1.34 | **-0.09 (-6.3%) drop** |
| **Train/Val Gap ($\Delta$)** | 0.03 | 0.13 | Overfitting gap began to appear |

---

## Generation Quality Progression

### V1 Generated Output:
```text
ROMEO:
Whan look user test e thin for an,
Wer shat hand me that call I thin non.
```

### V2 Generated Output:
```text
ROMEO:
What sayst thou, my Lord?
I will return thee to the king's chamber,
And think'st thou of these fearful words.
```

*My Observation*: V2 learned actual words ("sayst", "chamber", "return"), punctuation, and Shakespearean dialog formatting.

---

## What I Learned & Trade-offs in V2
1. **Capacity Solves Underfitting**: Increasing depth and width lowered validation loss from 1.43 to 1.34 and dramatically improved output quality.
2. **Emerging Overfitting**: The gap between train loss (1.21) and val loss (1.34) grew to 0.13. This taught me that scaling model capacity without adding regularization (like dropout and weight decay) leads to memorization.
3. **Training Time Trade-off**: V2 required ~4x more computation per epoch compared to V1, highlighting the need for optimization techniques in V3.
