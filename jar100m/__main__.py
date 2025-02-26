import torch
from torch import nn
from torch.optim import Adam

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.01

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

loss_fn = nn.MSELoss()
model = Model()
optimizer = Adam(model.parameters(), lr=0.1)

"""
dataset = Dataset(shakespeare[:1000], CONTEXT_WINDOW_SIZE)
rng = jax.random.key(0)
"""

inputs = []
outputs = []
for i in range(100):
    for j in range(100):
        inputs.append(torch.tensor([i/100, j/100]))
        outputs.append(torch.tensor([(i/100 + j/100)/2]))

for _ in range(EPOCHS):
    total_loss = 0
    for i in range(len(inputs)):
        #inp, expected_outp = dataset[i]
        inp = inputs[i]
        pred = model(inp)

        loss = loss_fn(pred, outputs[i])
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss += loss
    print(total_loss/len(inputs))

with torch.no_grad():
    a = float(input())
    b = float(input())
    inp = torch.tensor([a, b])
    out = model(inp)
    print(out)
