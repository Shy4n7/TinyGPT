import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v3.tinygpt_v3 import TinyGPT

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Please run `python data/fetch_shakespeare.py` first.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
char_to_id = {char: i for i, char in enumerate(unique_chars)}

n = len(text)
train_data = text[:int(n * 0.9)]
val_data = text[int(n * 0.9):]

train_tokens = torch.tensor([char_to_id[c] for c in train_data], dtype=torch.long)
val_tokens = torch.tensor([char_to_id[c] for c in val_data], dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_LENGTH = 64
BATCH_SIZE = 64
STEPS = 1000

def get_batch(split):
    data = train_tokens if split == 'train' else val_tokens
    ix = torch.randint(len(data) - CONTEXT_LENGTH - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    return x.to(device), y.to(device)

def run_lr_experiment(learning_rate):
    print(f"\n--- Running Experiment: Learning Rate = {learning_rate} ---")
    model = TinyGPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=8,
        num_blocks=4,
        max_sequence_length=CONTEXT_LENGTH,
        hidden_dim=512,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    for step in range(1, STEPS + 1):
        X, Y = get_batch('train')
        logits = model(X)
        loss = loss_fn(logits.view(-1, vocab_size), Y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        X_tr, Y_tr = get_batch('train')
        train_loss = loss_fn(model(X_tr).view(-1, vocab_size), Y_tr.view(-1)).item()
        
        X_va, Y_va = get_batch('val')
        val_loss = loss_fn(model(X_va).view(-1, vocab_size), Y_va.view(-1)).item()
        
    print(f"Result [LR={learning_rate}]: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
    return train_loss, val_loss

if __name__ == "__main__":
    lrs = [1e-3, 3e-4, 1e-4]
    results = {}
    for lr in lrs:
        results[lr] = run_lr_experiment(lr)
        
    print("\n" + "="*50)
    print("LEARNING RATE EXPERIMENT SUMMARY")
    print("="*50)
    print(f"{'Learning Rate':<15} | {'Train Loss':<12} | {'Val Loss':<12}")
    print("-"*50)
    for lr, (tr, va) in results.items():
        print(f"{lr:<15} | {tr:<12.4f} | {va:<12.4f}")
