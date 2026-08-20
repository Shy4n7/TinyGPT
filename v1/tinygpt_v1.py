import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Attention module for decoder-only Transformer.
    """
    def __init__(self, embedding_dim, num_heads, max_sequence_length):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
        self.scale = self.head_dim ** -0.5
        
        # Lower triangular matrix for causal masking
        causal_mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length))
        self.register_buffer('causal_mask', causal_mask)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split into heads: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) * self.scale
        
        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, 0.0)
        
        context = weights @ V
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embedding_dim)
        
        output = self.fc_out(context)
        return output, weights

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Standard pre-LN Transformer Decoder Block.
    """
    def __init__(self, embedding_dim, num_heads, max_sequence_length, hidden_dim):
        super().__init__()
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, max_sequence_length)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.ffn = FeedForward(embedding_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x, attn_weights

class TinyGPT(nn.Module):
    """
    TinyGPT V1 model (~129k parameters).
    """
    def __init__(self, vocab_size, embedding_dim=64, num_heads=4, num_blocks=2, max_sequence_length=64, hidden_dim=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, max_sequence_length, hidden_dim)
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.max_sequence_length, f"Sequence length {seq_len} exceeds max {self.max_sequence_length}"
        
        token_emb = self.token_embedding(tokens)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + pos_emb
        
        for block in self.blocks:
            x, _ = block(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate tokens given a prompt tensor idx of shape (1, T).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_sequence_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
