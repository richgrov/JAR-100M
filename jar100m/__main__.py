import torch
import torch.nn.functional as F
from torch.optim import Adam

from jar100m.dataset import Dataset
from jar100m.device import device
from jar100m.model import Model

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 3
EPOCHS = 10

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

dataset = Dataset(shakespeare[:50000], CONTEXT_WINDOW_SIZE)

model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

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
