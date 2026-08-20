<img width="1774" height="887" alt="ChatGPT Image Aug 18, 2026, 06_13_08 PM" src="https://github.com/user-attachments/assets/853f86ce-494f-445a-bb4e-36f48cd4db73" />



## Building GPT from Scratch

A personal learning project implementing the GPT decoder-only Transformer architecture in PyTorch from mathematical foundations to a scaled 6.3M parameter model with Automatic Mixed Precision (AMP) and regularization.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Read the Full Learning Journey

I documented the complete journey of building TinyGPT from manually computing the mathematics behind attention to implementing, training, scaling, and evaluating multiple Transformer iterations.

**[Building GPT from Scratch: A 5-Day Learning Journey](https://medium.com/@shyanpaul7/building-gpt-from-scratch-a-5-day-learning-journey-fcacd49fd959)**

---

## My Learning Journey & Iterations

### 1. V1: Foundation Model (112k Params)
- **Why I built this**: I wanted to understand how self-attention works under the hood by implementing `MultiHeadAttention`, `FeedForward`, and Pre-LN `TransformerBlock` from scratch.
- **Training Choice**: I started with a simple **epoch-based training loop** (20 epochs).
- **What I learned**: The causal triangular mask correctly prevented future token leakage. However, train loss (1.40) and val loss (1.43) plateaued early—showing severe **underfitting** due to limited model capacity.
- **Trade-off**: Epoch-based loops made tracking full dataset passes straightforward, but fixed sequence windows limited sample diversity per epoch.

### 2. V2: Scaling Capacity (818k Params)
- **Why I scaled up**: To test if underfitting in V1 was a capacity bottleneck, I scaled depth ($2 \rightarrow 4$ blocks) and width ($64 \rightarrow 128$ embedding dim).
- **What I learned**: Validation loss dropped from 1.43 to 1.34, and generated output became semi-coherent Shakespearean text. However, a validation gap ($\Delta = 0.13$) emerged between train (1.21) and val (1.34).
- **Trade-off**: Adding 6x more parameters dramatically improved text quality, but introduced the risk of **overfitting** because V2 lacked regularization.

### 3. V3: Larger Model with AMP & Regularization (6.3M Params)
- **Why I updated this**: As part of my ongoing learning, I wanted to experiment with a larger 6.3M parameter model while adding regularization and training efficiently on GPU.
- **Switched to Step-Based Training**: I replaced epoch loops with a **step-based training loop** (5,000 steps) using random token offset sampling.
  - *Why step-based?* Random index sampling provides uniform stochastic coverage, smoother step-level monitoring, and faster iteration.
  - *Trade-off*: Step-based training means you don't strictly guarantee an equal number of passes over every character compared to epoch sweeps, but it enables much faster experimental feedback loops.
- **Added Mixed Precision (AMP)**: Implemented `torch.cuda.amp.autocast()` and `GradScaler()`, achieving a **~2.8x training speedup** and saving **~45% VRAM**.
- **Added Regularization**: Integrated `Dropout=0.1` and `AdamW` weight decay (`0.1`), bringing validation loss down to **1.15**.

---

## Model Evolution Matrix

| Model Iteration | Parameters | Embedding Dim | Heads | Layers | Training Loop | Val Loss | My Key Learnings & Trade-offs |
|:---|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **V1 (Small)** | 112k | 64 | 4 | 2 | Epoch-based (20 epochs) | 1.43 | Learned pre-LN attention math; underfitted due to small capacity. |
| **V2 (Scaled)** | 818k | 128 | 8 | 4 | Epoch-based (20 epochs) | 1.34 | Improved generation quality; gap emerged showing need for regularization. |
| **V3 (Larger)** | 6.3M | 256 | 16 | 8 | Step-based (5,000 steps) | **1.15** | Step sampling + AMP for speed; Dropout + AdamW prevented overfitting. |

---

## Repository Structure

```
tinygpt/
├── README.md
├── .gitignore
├── requirements.txt
│
├── v1/
│   ├── tinygpt_v1.py          (112k params foundational model)
│   ├── training_v1.py         (Epoch-based training script)
│   └── v1_notes.md            (Math breakdown & underfitting notes)
│
├── v2/
│   ├── tinygpt_v2.py          (818k params scaled model)
│   ├── training_v2.py         (Scaled training loop)
│   └── v2_notes.md            (Scaling insights & generation improvements)
│
├── v3/
│   ├── tinygpt_v3.py          (6.3M params model with dropout & AMP)
│   ├── training_v3.py         (Step-based training, AMP fp16, AdamW)
│   └── v3_notes.md            (AMP speedup, step vs epoch trade-offs)
│
├── experiments/
│   ├── exp_dropout.py         (Dropout 0.0 vs 0.1 vs 0.2 experiment)
│   ├── exp_learning_rate.py   (Learning rate study: 1e-3 vs 3e-4 vs 1e-4)
│   └── exp_results.md         (Ablation results & trade-off findings)
│
├── data/
│   └── fetch_shakespeare.py   (Script to download Tiny Shakespeare data)
│
└── notebooks/
    └── tinygpt_colab.ipynb    (Runnable Google Colab notebook)
```

---

## Quick Start

### 1. Installation & Data Download

```bash
git clone https://github.com/Shy4n7/tinygpt.git
cd tinygpt
pip install -r requirements.txt
python data/fetch_shakespeare.py
```

### 2. Run Model Iterations

```bash
# V1: Foundation Model
python v1/training_v1.py

# V2: Scaled Model
python v2/training_v2.py

# V3: Larger Model (AMP + Step-based)
python v3/training_v3.py
```

### 3. Run Experiments

```bash
# Dropout regularizer benchmark
python experiments/exp_dropout.py

# Learning rate comparison
python experiments/exp_learning_rate.py
```

---

## What I Learned Overall

1. **Architecture Math**: Self-attention scaling by $\frac{1}{\sqrt{d_k}}$ is essential to keep softmax gradients stable early in training.
2. **Pre-LN vs Post-LN**: Pre-LayerNorm creates an uninhibited residual stream gradient highway, enabling deeper models (like V3's 8 layers) to train cleanly.
3. **Epoch vs Step-Based Trade-offs**: Epoch training is great for small datasets when exact coverage matters, but step-based stochastic batching with random offset sampling scales much better for large-scale training.
4. **Efficiency Matters**: AMP mixed precision enabled me to train a 6.3M parameter model on a single GPU with over 2x speedup.
5. **Regularization**: Dropout and selective weight decay on 2D weights are crucial once model capacity increases beyond ~500k parameters.

