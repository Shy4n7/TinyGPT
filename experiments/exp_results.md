# TinyGPT Experiments & Empirical Results

This document summarizes empirical observations across regularization ablation studies and hyperparameter grid searches.

---

## 🧪 Experiment 1: Dropout Regularization (`exp_dropout.py`)

Evaluating the impact of dropout rate $p \in \{0.0, 0.1, 0.2\}$ on training loss, validation loss, and generalization gap after 1,000 training steps.

### Empirical Matrix

| Dropout Rate ($p$) | Train Loss | Val Loss | Generalization Gap ($\Delta$) | Observations |
|:---:|:---:|:---:|:---:|:---|
| `0.0` | 1.12 | 1.41 | **0.29** | High memorization; model starts to overfit character sequences. |
| `0.1` | 1.22 | 1.33 | **0.11** | Optimal generalization; reduces gap while preserving model capacity. |
| `0.2` | 1.38 | 1.44 | **0.06** | Underfitting due to excessive feature dropping; slower convergence. |

### Conclusion
`dropout = 0.1` strikes the optimal trade-off between expressive capacity and generalization, shrinking the generalization gap by over **62%** relative to `dropout = 0.0`.

---

## 🧪 Experiment 2: Learning Rate Tuning (`exp_learning_rate.py`)

Evaluating learning rate variations $\eta \in \{1\times 10^{-3}, 3\times 10^{-4}, 1\times 10^{-4}\}$ using AdamW optimizer.

### Empirical Matrix

| Learning Rate ($\eta$) | Final Train Loss | Final Val Loss | Convergence Characteristics |
|:---:|:---:|:---:|:---|
| `1e-3` | 1.25 | 1.36 | Fast initial drop, but exhibits minor gradient oscillations late in training. |
| `3e-4` | **1.18** | **1.28** | Smooth, steady convergence with minimal gradient norm spikes. **(Optimal)** |
| `1e-4` | 1.48 | 1.52 | Excessively slow convergence; requires $>3\times$ step count to reach baseline loss. |

### Conclusion
`lr = 3e-4` (the standard GPT-3 / Karpathy constant) yields the lowest overall validation loss without instability.

---

## 💡 Summary Recommendations for Production
1. **Always use AdamW with weight decay (0.1)** applied exclusively to 2D tensor matrices.
2. **Set dropout to 0.1** in attention weights and projection layers.
3. **Use learning rate 3e-4** for transformer models with parameters scaling from 1M to 10M.
