import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tinygpt_v3 import TinyGPT

# Hyperparameters (Production V3 Config)
EMBEDDING_DIM = 256
NUM_HEADS = 16
NUM_BLOCKS = 8
CONTEXT_LENGTH = 128
HIDDEN_DIM = 1024
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
TOTAL_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_STEPS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for V3: {device}")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")

if not os.path.exists(DATA_PATH):
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    response = requests.get(url)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
char_to_id = {char: i for i, char in enumerate(unique_chars)}
id_to_char = {i: char for i, char in enumerate(unique_chars)}

# Train / Val Split (90% train, 10% val)
n = len(text)
train_data = text[:int(n * 0.9)]
val_data = text[int(n * 0.9):]

train_tokens = torch.tensor([char_to_id[c] for c in train_data], dtype=torch.long)
val_tokens = torch.tensor([char_to_id[c] for c in val_data], dtype=torch.long)

def get_batch(split):
    data = train_tokens if split == 'train' else val_tokens
    ix = torch.randint(len(data) - CONTEXT_LENGTH - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    return x.to(device), y.to(device)

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    max_sequence_length=CONTEXT_LENGTH,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"TinyGPT V3 initialized with {param_count:,} parameters.")

# Configure AdamW with Weight Decay on 2D parameters only (exclude 1D bias / LayerNorms)
decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

optim_groups = [
    {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))
loss_fn = nn.CrossEntropyLoss()

# Automatic Mixed Precision (AMP) setup
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_STEPS)
        for k in range(EVAL_STEPS):
            X, Y = get_batch(split)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(X)
                loss = loss_fn(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate_sample(prompt="KING RICHARD:"):
    model.eval()
    prompt_tokens = [char_to_id.get(c, 0) for c in prompt]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=10)
    out_text = "".join([id_to_char[i.item()] for i in generated[0]])
    model.train()
    return out_text

if __name__ == "__main__":
    print(f"\n=== Training Production TinyGPT V3 ({TOTAL_STEPS:,} Steps with AMP) ===")
    best_val_loss = float('inf')
    start_time = time.time()
    
    for step in range(1, TOTAL_STEPS + 1):
        X, Y = get_batch('train')
        
        optimizer.zero_grad(set_to_none=True)
        
        # AMP Autocast context
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(X)
            loss = loss_fn(logits.view(-1, vocab_size), Y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS:
            losses = estimate_loss()
            dt = time.time() - start_time
            print(f"Step {step:04d}/{TOTAL_STEPS} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Time: {dt:.1f}s")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinygpt_v3_best.pt")
                torch.save(model.state_dict(), save_path)
                print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {best_val_loss:.4f})")
                
    print("\n=== Final Text Generation Sample ===")
    print(generate_sample("KING RICHARD:"))
