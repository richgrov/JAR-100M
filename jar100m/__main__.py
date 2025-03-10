import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time

from jar100m.dataset import Dataset
from jar100m.device import device
from jar100m.model import Model

CONTEXT_WINDOW_SIZE = 64
EPOCHS = 3
LOSS_REPORT_INTERVAL = 10

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

dataset = Dataset(shakespeare, CONTEXT_WINDOW_SIZE)
train_data, validate_data, _ = random_split(dataset, [0.2, 0.1, 0.7])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
validate_loader = DataLoader(validate_data, batch_size=16, shuffle=True)

train_loss_history = []
validate_loss_history = []

model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE).to(device)

if len(sys.argv) > 1:
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    EPOCHS = 0
    model.eval()

optimizer = Adam(model.parameters(), lr=0.001)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters")

def cross_entropy_loss(logits, classes):
    batches, context_size, probs = logits.shape
    logits = torch.reshape(logits, (batches*context_size, probs))
    classes = torch.reshape(classes, (batches*context_size,))
    return F.cross_entropy(logits, classes)

def validate():
    total_loss = 0

    with torch.no_grad():
        for inp, expected_outp in validate_loader:
            pred_logits = model(inp)
            total_loss += cross_entropy_loss(pred_logits, expected_outp)

    return total_loss / len(validate_loader)

for epoch in range(EPOCHS):
    total_loss = 0
    timestamp = time.monotonic()

    for i, (inp, expected_outp) in enumerate(train_loader):
        pred_logits = model(inp)
        loss = cross_entropy_loss(pred_logits, expected_outp)
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % LOSS_REPORT_INTERVAL == 0 and i > 0:
            now = time.monotonic()

            average_loss = total_loss / LOSS_REPORT_INTERVAL
            validate_loss = validate()
            print(f"Epoch {epoch}, step {i}: train loss {average_loss}, validate loss {validate_loss}, elapsed {now - timestamp:.2f}s")
            train_loss_history.append(average_loss)
            validate_loss_history.append(validate_loss)
            total_loss = 0
            timestamp = now

    torch.save(model.state_dict(), f"model-{epoch}.pt")

def generate(sequence, n):
    for _ in range(n):
        logits = model(sequence)[:, -1]
        probs = F.softmax(logits, dim=1)
        next = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, next), dim=1)

    return sequence

inp = torch.stack([dataset.encode("\n")])
outp = generate(inp, 1000)
print(dataset.decode(outp[0]))

plt.plot(train_loss_history, label="Train loss")
plt.plot(validate_loss_history, label="Validate loss")
plt.legend()
plt.show()
