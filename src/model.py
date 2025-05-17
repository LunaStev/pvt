import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention import MaskedMultiHeadSelfAttention

class PVT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, seq_len=5, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        self.attn = MaskedMultiHeadSelfAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding     # (B, T, C)
        x = self.attn(x)                               # (B, T, C)
        x = x.view(x.size(0), -1)                      # (B, T*C)
        return self.ff(x)