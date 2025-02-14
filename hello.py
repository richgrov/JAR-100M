import random as rand
from jax import grad
import jax.numpy as np
from data import Data
import math

CONTEXT_WINDOW_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.01

dataset = [[x/20] for x in range(20)]
inputs = np.array(dataset)
outputs = np.array(dataset)

def model(params, inp):
    weights, biases = params
    return np.dot(inp, weights) + biases

def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)

def loss_fn(params, inp, expected_outp):
    predictions = model(params, inp)
    return mse(predictions, expected_outp)

params = (np.array([0.5]), np.array([0.5]))

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
    thing = Data("dataset.txt")
    this = thing.getRandomContextWindow(CONTEXT_WINDOW_SIZE)
    that = thing.randomVecsInVocab(CONTEXT_WINDOW_SIZE)
    
    # print(this[0][0])
    # print(that[this[0][0]])
    # print(np.dot())
    mat1 = makeMatrix(CONTEXT_WINDOW_SIZE, 8)
    mat2 = makeMatrix(CONTEXT_WINDOW_SIZE, 8)
    mat3 = makeMatrix(CONTEXT_WINDOW_SIZE, 8)
    those = dict.copy(that)
    things = []
    for i in range(CONTEXT_WINDOW_SIZE):
        that[this[0][i]] = multiplyMatrix(that[this[0][i]], mat1)
        those[this[0][i]] = multiplyMatrix(those[this[0][i]], mat2)
    for i in range(CONTEXT_WINDOW_SIZE):
        array_thing = []
        for j in range(CONTEXT_WINDOW_SIZE):
            array_thing.append(np.dot(that[this[0][j]], those[this[0][i]]))
        things.append(np.array(array_thing))
    things = np.array(things)
    # print(things[0])
    for column in range(len(things)):
        for what in range(len(things)):
            if what > column:
                things[column][what] = -1000
        things[column] = softMax(things[column], 5)
        print(things[column])
    # for i in range(len(this)):
    #     print(len(this[i][0]))
    # print(this)
    # print(that)
    
    # for idx in range(len(shakespear)):
    #     print(shakespear[idx-31:idx+32])
    # print(randomized_vocab_vectors)
    # print(vocab)

def multiplyMatrix(vector: list, matrix: list):
    new_vector = []
    for row in range(len(matrix)):
        sum = 0
        for column in range(len(vector)):
            sum += matrix[row][column] * vector[column]
        new_vector.append(sum)
    return new_vector

def makeMatrix(cw: int, rows: int) -> list:
    matrix = []
    for j in range(rows):
        row = []
        for i in range(cw):
            row.append(rand.uniform(-1.0, 1.0))
        matrix.append(row)
    return matrix

def softMax(values: list, temperature: float=1):
    positive_list = []
    length = len(values)
    for i in range(length):
        positive_list.append(np.pow(math.e, values[i]/temperature))
    new_values = []
    sum = 0
    for i in range(length):
        sum += positive_list[i]
    for i in range(length):
        new_values.append(positive_list[i]/sum)
    return new_values

if __name__ == "__main__":
    main()
