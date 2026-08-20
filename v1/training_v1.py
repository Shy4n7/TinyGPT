import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tinygpt_v1 import TinyGPT

# Hyperparameters
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
CONTEXT_LENGTH = 64
HIDDEN_DIM = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset loading
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")

if not os.path.exists(DATA_PATH):
    print("Dataset not found locally. Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    response = requests.get(url)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer mapping
unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
char_to_id = {char: i for i, char in enumerate(unique_chars)}
id_to_char = {i: char for i, char in enumerate(unique_chars)}

tokens = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
print(f"Tokenized dataset: {len(tokens):,} tokens | Vocab size: {vocab_size}")

class CharDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length
    
    def __len__(self):
        return len(self.tokens) - self.context_length
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_length]
        y = self.tokens[idx + 1:idx + self.context_length + 1]
        return x, y

dataset = CharDataset(tokens, CONTEXT_LENGTH)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TinyGPT(vocab_size, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS, CONTEXT_LENGTH, HIDDEN_DIM).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"TinyGPT V1 initialized with {param_count:,} parameters.")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
    return total_loss / len(val_loader)

def generate_sample(prompt="ROMEO:"):
    model.eval()
    prompt_tokens = [char_to_id.get(c, 0) for c in prompt]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=5)
    out_text = "".join([id_to_char[i.item()] for i in generated[0]])
    return out_text

if __name__ == "__main__":
    print("\n=== Training TinyGPT V1 (20 Epochs) ===")
    for epoch in range(EPOCHS):
        train_loss = train_epoch()
        val_loss = evaluate()
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinygpt_v1.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print("\n--- Sample Generation ---")
    print(generate_sample("ROMEO:"))
