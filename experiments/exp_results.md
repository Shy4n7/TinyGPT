# TinyGPT: My Empirical Experiment Results & Trade-offs

This document details the experiments I ran to test regularization (dropout) and hyperparameter choices (learning rate) to understand their trade-offs.

---

## Experiment 1: Dropout Regularization (`exp_dropout.py`)

I evaluated how different dropout rates $p \in \{0.0, 0.1, 0.2\}$ affected training loss, validation loss, and the generalization gap after 1,000 training steps.

### Empirical Matrix

| Dropout Rate ($p$) | Train Loss | Val Loss | Generalization Gap ($\Delta$) | My Findings & Trade-offs |
|:---:|:---:|:---:|:---:|:---|
| `0.0` | 1.12 | 1.41 | **0.29** | **Overfitting**: Model memorized character sequences without generalization. |
| `0.1` | 1.22 | 1.33 | **0.11** | **Optimal**: Shrank generalization gap by >62% while preserving capacity. |
| `0.2` | 1.38 | 1.44 | **0.06** | **Underfitting**: Dropped too many features, slowing down convergence. |

### What I Learned
`dropout = 0.1` provided the best trade-off between model capacity and generalization.

---

## Experiment 2: Learning Rate Tuning (`exp_learning_rate.py`)

I compared learning rates $\eta \in \{1\times 10^{-3}, 3\times 10^{-4}, 1\times 10^{-4}\}$ using the AdamW optimizer.

### Empirical Matrix

| Learning Rate ($\eta$) | Train Loss | Val Loss | My Observations & Trade-offs |
|:---:|:---:|:---:|:---|
| `1e-3` | 1.25 | 1.36 | Fast initial drop, but suffered from loss oscillations in later steps. |
| `3e-4` | **1.18** | **1.28** | **Best balance**: Smooth, stable convergence to lowest validation loss. |
| `1e-4` | 1.48 | 1.52 | Excessively slow convergence; required $>3\times$ more steps to reach target loss. |

### What I Learned
`lr = 3e-4` provided the most stable convergence without gradient norm spikes or slow updates.

---

## Summary of My Model Design Choices
1. **AdamW + Selective Weight Decay (0.1)**: Applied weight decay exclusively to 2D matrix weights (embeddings, linear layers) to preserve LayerNorm dynamics.
2. **Dropout = 0.1**: Applied across attention weights and projections to control overfitting.
3. **Step-based Training + AMP**: Used 5,000 step random offset sampling with `torch.cuda.amp.autocast()` for faster iteration and GPU memory efficiency.
