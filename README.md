# TinyGPT: Building GPT from Scratch

A hands-on, educational project implementing the GPT decoder-only Transformer architecture from mathematical foundations to a production-quality 4M parameter model.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Journey & Progression

This repository breaks down GPT building into 3 distinct iterations:

- **V1: Small Model (129k params)** – Hand-computed math foundation, base Multi-Head Attention, epoch-based training.
- **V2: Scaled Model (600k params)** – Increased depth & width, improved generation capabilities, 4x parameter scaling.
- **V3: Production Model (4M params)** – Regularization (dropout & weight decay), Automatic Mixed Precision (AMP), step-based training loop, and model checkpointing.

---

## 📊 Version Comparison Matrix

| Version | Parameters | Embedding Dim | Heads | Layers | Training Loop | Train Loss | Val Loss | Key Highlights |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **V1** | **129k** | 64 | 4 | 2 | Epoch-based (20 epochs) | 1.40 | 1.43 | Math foundation, simple causal mask |
| **V2** | **600k** | 128 | 8 | 4 | Epoch-based (20 epochs) | 1.21 | 1.34 | 4x scale, semi-coherent Shakespeare |
| **V3** | **4.1M** | 256 | 16 | 8 | Step-based (5,000 steps) | 0.98 | 1.15 | Dropout=0.1, AMP, Weight Decay=0.1 |

---

## 📂 Repository Structure

```
tinygpt/
├── README.md
├── .gitignore
├── requirements.txt
│
├── v1/
│   ├── tinygpt_v1.py          (129k params model architecture)
│   ├── training_v1.py         (Epoch-based training script)
│   └── v1_notes.md            (Foundational notes & math observations)
│
├── v2/
│   ├── tinygpt_v2.py          (600k params model architecture)
│   ├── training_v2.py         (Scaled training script)
│   └── v2_notes.md            (Scaling analysis & V1 comparison)
│
├── v3/
│   ├── tinygpt_v3.py          (4M params model + dropout)
│   ├── training_v3.py         (Step-based training, AMP, AdamW)
│   └── v3_notes.md            (Production analysis & optimizations)
│
├── experiments/
│   ├── exp_dropout.py         (Dropout 0.0 vs 0.1 vs 0.2 benchmark)
│   ├── exp_learning_rate.py   (LR study: 1e-3 vs 3e-4 vs 1e-4)
│   └── exp_results.md         (Experimental findings & loss tables)
│
├── data/
│   └── fetch_shakespeare.py   (Script to download Tiny Shakespeare data)
│
└── notebooks/
    └── tinygpt_colab.ipynb    (Runnable Google Colab Notebook)
```

---

## 🛠️ Quick Start

### 1. Installation

```bash
git clone https://github.com/Shy4n7/tinygpt.git
cd tinygpt
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/fetch_shakespeare.py
```

### 3. Run Training Loops

#### Train V1 (129k params)

```bash
python v1/training_v1.py
```

#### Train V2 (600k params)

```bash
python v2/training_v2.py
```

#### Train V3 (4.1M params with AMP & Regularization)

```bash
python v3/training_v3.py
```

### 4. Interactive Experiments

Run dropout impact benchmarks:

```bash
python experiments/exp_dropout.py
```

Run learning rate comparison study:

```bash
python experiments/exp_learning_rate.py
```

---

## 🧠 What You'll Learn

1. **Self-Attention & Causal Masking**: Understanding how $Q, K, V$ matrices interact and why causal triangular masking prevents future token leakage.
2. **Pre-LayerNorm vs Post-LayerNorm**: Why placing LayerNorm before Multi-Head Attention and FFN improves gradient flow in deeper Transformer blocks.
3. **Scaling Dynamics**: How parameter scaling affects loss reduction, vocabulary capture, and sample text quality.
4. **Regularization**: Mitigating overfitting using Dropout and AdamW weight decay.
5. **Modern PyTorch Optimizations**: Speeding up training with `torch.cuda.amp.autocast` and `GradScaler`.

---
