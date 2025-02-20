import random as rand
from jax import grad
import jax.numpy as np

from jar100m.dataset import Dataset
from jar100m.functions import relu, mse

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01

dataset = Dataset()

def model(params, inp):
    weights, biases = params
    return relu(np.dot(inp, weights) + biases)

def loss_fn(params, inp, expected_outp):
    predictions = model(params, inp)
    return mse(predictions, expected_outp)

params = (np.array([1.0, 1.0]), np.array([1.0]))

for _ in range(EPOCHS):
    total_loss = 0
    for i in range(len(dataset)):
        inp, expected_outp = dataset[i]

        grads = grad(loss_fn)(params, inp, expected_outp)
        print(grads)
        weights, biases = params

        params = (
            weights - LEARNING_RATE * grads[0],
            biases - LEARNING_RATE * grads[1],
        )
        
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
