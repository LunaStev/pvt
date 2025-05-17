import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, T, T)

        # Causal mask (upper triangular part = -inf)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, T, T)
        out = attn_weights @ v                        # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out_proj(out)