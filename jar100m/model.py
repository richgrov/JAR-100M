import math

import torch
from torch import nn
import torch.nn.functional as F

from jar100m.device import device

EMBED_DIMENSIONS = 1024

class SelfAttentionHead(nn.Module):
    def __init__(self, in_size: int, head_size: int, context_window_len: int) -> None:
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(in_size, head_size, bias=False)
        self.key = nn.Linear(in_size, head_size, bias=False)
        self.value = nn.Linear(in_size, head_size, bias=False)

        affinity_tri = torch.tril(torch.ones(context_window_len, context_window_len, device=device)) == 0
        self.register_buffer("affinity_tri", affinity_tri)

    def forward(self, x: torch.Tensor):
        k = self.key(x)
        q = self.query(x)

        affinities = (q @ k.transpose(1, 2)) / math.sqrt(self.head_size)

        num_toks = x.shape[1]
        affinity_tri = self.affinity_tri[:num_toks, :num_toks]
        decoder_affinities = affinities.masked_fill(affinity_tri, float("-inf"))

        normalized = F.softmax(decoder_affinities, dim=-1)
        return normalized @ self.value(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, in_size: int, head_size: int, context_window_len: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(
                in_size=in_size,
                head_size=head_size,
                context_window_len=context_window_len,
            ) for _ in range(num_heads)]
        )

    def forward(self, x) -> int:
        return torch.cat([head(x) for head in self.heads], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, context_window_len: int) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            num_heads=16,
            in_size=EMBED_DIMENSIONS,
            head_size=EMBED_DIMENSIONS // 16,
            context_window_len=context_window_len,
        )

        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIMENSIONS, EMBED_DIMENSIONS * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIMENSIONS * 4, EMBED_DIMENSIONS),
            nn.ReLU(),
        ) 

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

class Model(nn.Module):
    def __init__(self, vocab_len: int, context_window_len: int) -> None:
        super().__init__()
        self.context_window_len = context_window_len

        self.info_embedding = nn.Embedding(vocab_len, EMBED_DIMENSIONS)
        self.position_embedding = nn.Embedding(context_window_len, EMBED_DIMENSIONS)
        self.blocks = nn.Sequential(
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
            TransformerBlock(context_window_len),
        )

        self.unembed = nn.Linear(EMBED_DIMENSIONS, vocab_len)

    def forward(self, x):
        num_toks = x.shape[1]

        embedding = self.info_embedding(x[:, -self.context_window_len:])
        arange = torch.arange(min(num_toks, self.context_window_len), device=device)
        position_embeddings = self.position_embedding(arange)

        x = self.blocks(embedding + position_embeddings)

        logits = self.unembed(x)
        return logits
