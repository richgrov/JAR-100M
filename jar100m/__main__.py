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

dataset = Dataset()

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
        print(params[0], params[1])

def main():
    file = open("dataset.txt", 'r')
    shakespear = file.read()
    spliceIndex = int(len(shakespear) * TRAIN_SPLIT)

    first = shakespear[:spliceIndex]
    last = shakespear[spliceIndex:]
    
    vocab = set(shakespear)
    
    randomized_vocab_vectors = {}
    
    for character in vocab:
        random_vec = []
        for vec_idx in range(63):
            random_vec.append(rand.uniform(-1.0, 1.0))
        randomized_vocab_vectors[character] = random_vec
    
    # for idx in range(len(shakespear)):
    #     print(shakespear[idx-31:idx+32])
    print(randomized_vocab_vectors)
    print(vocab)


#if __name__ == "__main__":
    #main()
