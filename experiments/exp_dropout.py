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

def run_dropout_experiment(dropout_rate):
    print(f"\n--- Running Experiment: Dropout = {dropout_rate} ---")
    model = TinyGPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=8,
        num_blocks=4,
        max_sequence_length=CONTEXT_LENGTH,
        hidden_dim=512,
        dropout=dropout_rate
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for step in range(1, STEPS + 1):
        X, Y = get_batch('train')
        logits = model(X)
        loss = loss_fn(logits.view(-1, vocab_size), Y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluate final train and val loss
    model.eval()
    with torch.no_grad():
        X_tr, Y_tr = get_batch('train')
        train_loss = loss_fn(model(X_tr).view(-1, vocab_size), Y_tr.view(-1)).item()
        
        X_va, Y_va = get_batch('val')
        val_loss = loss_fn(model(X_va).view(-1, vocab_size), Y_va.view(-1)).item()
        
    gap = val_loss - train_loss
    print(f"Result [Dropout={dropout_rate}]: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Gap={gap:.4f}")
    return train_loss, val_loss, gap

if __name__ == "__main__":
    rates = [0.0, 0.1, 0.2]
    results = {}
    for r in rates:
        results[r] = run_dropout_experiment(r)
        
    print("\n" + "="*50)
    print("DROPOUT EXPERIMENT SUMMARY")
    print("="*50)
    print(f"{'Dropout':<10} | {'Train Loss':<12} | {'Val Loss':<12} | {'Gap':<12}")
    print("-"*50)
    for r, (tr, va, g) in results.items():
        print(f"{r:<10} | {tr:<12.4f} | {va:<12.4f} | {g:<12.4f}")
