from jax import grad
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import softmax
import jax.random

from jar100m.dataset import Dataset
from jar100m.functions import relu, mse
from jar100m.layers import *

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.01

with open("dataset.txt", 'r') as file:
    shakespeare = file.read()

spliceIndex = int(len(shakespeare) * TRAIN_SPLIT)
train = shakespeare[:spliceIndex]
validate = shakespeare[spliceIndex:]

dataset = Dataset(shakespeare[:1000], CONTEXT_WINDOW_SIZE)
rng = jax.random.key(0)

def model(params, x):
    logits = params[x[-1]]
    return logits

def generate(params, x: np.ndarray, n):
    global rng

    for _ in range(n):
        logits = model(params, x)
        #probs = softmax(logits)
        next_idx = jax.random.categorical(rng, logits, shape=(1,))[0]
        rng, _ = jax.random.split(rng)
        x = np.concatenate((x, np.array([next_idx])))

    return x

def loss_fn(params, inp, expected_outp):
    predictions = model(params, inp)
    return mse(predictions, expected_outp)

params = embedding(len(dataset.vocab), len(dataset.vocab))

tokens = generate(params, dataset.encode("hi"), 20)
print(dataset.decode(tokens))

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

new_tokens = generate(params, dataset.encode("\n"), 20)
print(dataset.decode(new_tokens))
