import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from jar100m.dataset import Dataset
from jar100m.device import device
from jar100m.model import Model

CONTEXT_WINDOW_SIZE = 16
EPOCHS = 5
LOSS_REPORT_INTERVAL = 1000

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

dataset = Dataset(shakespeare, CONTEXT_WINDOW_SIZE)
train_data, validate_data, _ = random_split(dataset, [0.1, 0.1, 0.8])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters")

for epoch in range(EPOCHS):
    total_loss = 0

    for i, (inp, expected_outp) in enumerate(train_loader):
        pred_logits = model(inp)

        batches, context_size, probs = pred_logits.shape
        pred_logits = torch.reshape(pred_logits, (batches*context_size, probs))
        expected_outp = torch.reshape(expected_outp, (batches*context_size,))

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
        logits = model(sequence)[:, -1]
        probs = F.softmax(logits, dim=1)
        next = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, next), dim=1)

    return sequence

inp = torch.stack([dataset.encode("\n")])
outp = generate(inp, 1000)
print(dataset.decode(outp[0]))
