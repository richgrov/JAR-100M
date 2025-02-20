import random as rand
from jax import grad
import jax.numpy as np
from jax.example_libraries import optimizers

from jar100m.dataset import Dataset
from jar100m.functions import relu, mse
from jar100m.layers import fully_connected

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

dataset = Dataset(train)

def model(params, x):
    for weights, biases in params:
        x = relu(np.dot(x, weights) + biases)

    return x

def loss_fn(params, inp, expected_outp):
    predictions = model(params, inp)
    return mse(predictions, expected_outp)

params = [
    fully_connected(2, 2),
    fully_connected(2, 1),
]

adam_init, adam_update, get_params = optimizers.adam(LEARNING_RATE)
optimizer_state = adam_init(params)

for _ in range(EPOCHS):
    total_loss = 0
    for i in range(len(dataset)):
        inp, expected_outp = dataset[i]

        grads = grad(loss_fn)(params, inp, expected_outp)
        optimizer_state = adam_update(0, grads, optimizer_state)
        params = get_params(optimizer_state)
        
        loss = loss_fn(params, inp, expected_outp)
    total_loss += loss
    print(total_loss/len(dataset))
