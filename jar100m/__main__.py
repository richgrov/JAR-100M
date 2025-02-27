import torch
import torch.nn.functional as F
from torch.optim import Adam

from jar100m.dataset import Dataset
from jar100m.device import device
from jar100m.model import Model

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 8
EPOCHS = 5
LOSS_REPORT_INTERVAL = 10000

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

dataset = Dataset(shakespeare, CONTEXT_WINDOW_SIZE)

model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters")

for epoch in range(EPOCHS):
    total_loss = 0

    for i in range(100000):
        inp, expected_outp = dataset[i]
        pred_logits = model(inp)

        loss = F.cross_entropy(pred_logits, expected_outp)
        total_loss += loss.item()

        if i % LOSS_REPORT_INTERVAL == 0 and i > 0:
            average_loss = total_loss / LOSS_REPORT_INTERVAL
            print(f"Epoch {epoch}, step {i}: loss {average_loss}")
            total_loss = 0

        model.zero_grad()
        loss.backward()
        optimizer.step()

def generate(sequence, n):
    for _ in range(n):
        logits = model(sequence)[-1]
        probs = F.softmax(logits, dim=0)
        next = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, next))

    return sequence

inp = dataset.encode("\n")
outp = generate(inp, 1000)
print(dataset.decode(outp))
