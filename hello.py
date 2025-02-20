import random as rand
from jax import grad
import jax.numpy as np

TRAIN_SPLIT = 0.9
CONTEXT_WINDOW_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01

input_array = []
output_array = []

for x in range(20):
    for y in range(20):
        input_array.append([x/20, y/20])
        output_array.append([(x/20+y/20)/2+1.0])

inputs = np.array(input_array)
outputs = np.array(output_array)

def model(params, inp):
    weights, biases = params
    return np.dot(inp, weights) + biases

def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)

def loss_fn(params, inp, expected_outp):
    predictions = model(params, inp)
    return mse(predictions, expected_outp)

params = (np.array([1.0, 1.0]), np.array([1.0]))

for _ in range(EPOCHS):
    total_loss = 0
    for i, inp in enumerate(inputs):
        expected_outp = outputs[i]
        grads = grad(loss_fn)(params, inp, expected_outp)
        weights, biases = params

        params = (
            weights - LEARNING_RATE * grads[0],
            biases - LEARNING_RATE * grads[1],
        )
        
        loss = loss_fn(params, inp, expected_outp)
        total_loss += loss
        print(params[0], params[1])

    print(f"Average Loss: {total_loss/len(inputs)}")

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
