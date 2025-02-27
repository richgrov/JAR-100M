import torch
from torch import nn

EMBED_DIMENSIONS = 8

class Model(nn.Module):
    def __init__(self, vocab_len: int, context_window_len: int) -> None:
        super().__init__()
        self.context_window_len = context_window_len

        self.info_embedding = nn.Embedding(vocab_len, EMBED_DIMENSIONS)
        self.position_embedding = nn.Embedding(context_window_len, EMBED_DIMENSIONS)
        self.unembed = nn.Linear(EMBED_DIMENSIONS, vocab_len)

    def forward(self, x):
        num_toks = x.shape[0]

        embedding = self.info_embedding(x[-self.context_window_len:num_toks])
        position_embeddings = self.position_embedding(torch.arange(min(num_toks, self.context_window_len)))
        logits = self.unembed(embedding + position_embeddings)
        return logits
