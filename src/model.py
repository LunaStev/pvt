import torch.nn as nn
import torch.nn.functional as F

class PVT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * 3, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits