import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from jar100m.dataset import Dataset
from jar100m.device import device

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 3
EPOCHS = 10
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

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

dataset = Dataset(shakespeare[:50000], CONTEXT_WINDOW_SIZE)

model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE).to(device)
optimizer = Adam(model.parameters(), lr=0.01)

for _ in range(EPOCHS):
    total_loss = 0
    for i in range(len(dataset)):
        inp, expected_outp = dataset[i]
        pred_logits = model(inp)

        loss = F.cross_entropy(pred_logits, expected_outp)
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    print(total_loss/len(dataset))

def generate(sequence, n):
    for _ in range(n):
        logits = model(sequence)[-1]
        probs = F.softmax(logits, dim=0)
        next = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, next))

    return sequence

inp = dataset.encode("\n")
outp = generate(inp, 20)
print(dataset.decode(outp))
