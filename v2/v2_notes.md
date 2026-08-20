# V2: TinyGPT - Scaled Model

## Model Overview
- **Parameters**: 811,713 (~800k / scaled config)
- **Embedding Dimension ($d_{model}$)**: 128 (2x V1)
- **Attention Heads ($h$)**: 8 (2x V1)
- **Head Dimension ($d_k$)**: 16 ($128 / 8$)
- **Layers / Blocks ($N$)**: 4 (2x V1)
- **Context Length ($T$)**: 64
- **Feed-Forward Hidden Dim**: 512 ($4 \times 128$)

---

## Comparison: V1 vs V2

| Metric / Architectural Feature | V1 (Small) | V2 (Scaled) | Delta / Change |
|:---|:---:|:---:|:---:|
| **Parameters** | 129k | 811k | **~6.3x scaling** |
| **Embedding Dimension** | 64 | 128 | 2x width |
| **Attention Heads** | 4 | 8 | 2x heads |
| **Transformer Blocks** | 2 | 4 | 2x depth |
| **Final Train Loss** | 1.40 | 1.21 | **-0.19 (-13.6%)** |
| **Final Val Loss** | 1.43 | 1.34 | **-0.09 (-6.3%)** |
| **Train/Val Gap ($\Delta$)** | 0.03 | 0.13 | Slight gap emerging |

---

## Generation Quality Progression

### V1 Sample Output:
```text
ROMEO:
Whan look user test e thin for an,
Wer shat hand me that call I thin non.
```
*Observation*: Learned individual characters and basic whitespace, but fails on word structure and long dependencies.

### V2 Sample Output:
```text
ROMEO:
What sayst thou, my Lord?
I will return thee to the king's chamber,
And think'st thou of these fearful words.
```
*Observation*: Captures Shakespearean vocabulary ("sayst", "thou", "chamber"), character names, colon dialogues, and partial syntax structure.

---

## What I Learned
1. **Depth and Width Scaling**: Increasing layer depth from 2 to 4 allowed hierarchical feature composition (lower layers learn syntax/characters, upper layers capture semantic relations).
2. **Emerging Overfitting / Need for Regularization**: Unlike V1 where train and val loss tracked identically, V2 begins showing a small validation gap ($\Delta = 0.13$). This hints that larger models require **dropout** and **weight decay** to generalize even better.
3. **Training Dynamics**: V2 requires slightly more compute per step, but converges to significantly lower perplexity within 20 epochs.
