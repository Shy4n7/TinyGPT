# V3: TinyGPT - Scaling & Regularization Experiments

## Why I Built V3
After seeing V2 improve in quality but start to overfit, I wanted to scale up to a **6.3M parameter model** while tackling two key learning objectives:
1. **Preventing Overfitting**: Using dropout and weight decay.
2. **Improving Training Speed & Memory Efficiency**: Switching to step-based training and Automatic Mixed Precision (AMP).

---

## Architectural Scaling
- **Parameters**: 6,384,705 (~6.3M parameters)
- **Embedding Dim ($d_{model}$)**: 256
- **Attention Heads ($h$)**: 16
- **Blocks ($N$)**: 8
- **Context Length ($T$)**: 128 (doubled from 64)
- **Dropout Rate**: 0.1 throughout

---

## My Key Architectural & Training Choices

### 1. Step-Based Training vs Epoch-Based Training
- **My Choice**: Switched from epoch-based training to **step-based training** (5,000 steps) with random sequence offset sampling.
- **Why I Chose Step-Based**:
  - Random indexing `torch.randint(0, len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))` provides uniform stochastic sampling across arbitrary character offsets rather than static dataset chunks.
  - Step counts give a direct, constant-time metric for progress and evaluation logging.
- **Trade-off I Discovered**:
  - *Epoch-based*: Guarantees every single character is passed through the model per epoch, but progress updates are coarse and tied to dataset size.
  - *Step-based*: Enables rapid iterations and smooth loss tracking, but doesn't strictly guarantee equal pass counts for every token unless total steps match full epoch sweeps.

### 2. Automatic Mixed Precision (AMP)
- **My Choice**: Wrapped the forward pass in `torch.cuda.amp.autocast()` and used `GradScaler()`.
- **Why I Chose AMP**: Large matrix multiplications ($Q K^T$) in FP16 compute significantly faster on Tensor Cores while maintaining master weights in FP32.
- **What I Observed**: Reduced VRAM memory usage by **~45%** and boosted training speed by **~2.8x**.

### 3. Regularization: Dropout + Selective AdamW Weight Decay
- **My Choice**: Added `Dropout(0.1)` to attention/embeddings and applied `AdamW` with `weight_decay=0.1` *only* to 2D weight matrices (excluding 1D biases and LayerNorms).
- **What I Learned**: Regularization controlled the generalization gap ($\Delta$ dropped to 0.11), bringing validation loss down to an all-time low of **1.15**.

---

## Evolution Summary Across Iterations

| Feature / Metric | V1 (Small) | V2 (Scaled) | V3 (Larger Model) |
|:---|:---:|:---:|:---:|
| **Parameters** | 112k | 818k | **6.3M** |
| **Embedding Dim** | 64 | 128 | 256 |
| **Heads / Layers** | 4 / 2 | 8 / 4 | 16 / 8 |
| **Context Length** | 64 | 64 | 128 |
| **Training Loop** | Epoch-based (20 epochs) | Epoch-based (20 epochs) | **Step-based (5,000 steps)** |
| **Precision** | FP32 | FP32 | **AMP (FP16/FP32 mixed)** |
| **Val Loss** | 1.43 | 1.34 | **1.15** |
