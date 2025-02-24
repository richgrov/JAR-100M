from jax import grad, random
import jax.numpy as np
import numpy as nmp
from jar100m.dataset import Dataset

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
    data = Dataset("dataset.txt", CONTEXT_WINDOW_SIZE)
    rand_window = data.getRandomContextWindow(CONTEXT_WINDOW_SIZE)
    rand_vec_vocab = data.randomVecsInVocab(CONTEXT_WINDOW_SIZE)
    print(rand_window)
    print(rand_vec_vocab)
    # rand_window_copy = list.copy(rand_window)
    
    # query_matrix = makeMatrix(CONTEXT_WINDOW_SIZE, 8)
    # key_matrix = nmp.array(makeMatrix(CONTEXT_WINDOW_SIZE, 8))
    # value_matrix = makeMatrix(CONTEXT_WINDOW_SIZE, 8)
    # temp = [[0] for _ in range(CONTEXT_WINDOW_SIZE)]
    # window = []
    # for _ in range(CONTEXT_WINDOW_SIZE):
    #     temp = []
    #     for _ in range(CONTEXT_WINDOW_SIZE):
    #         temp.append(0)
    #     window.append(temp)
    # key_vector = list.copy(temp)
    # query_vector = list.copy(temp)
    # for i in range(CONTEXT_WINDOW_SIZE):
    #     vec = rand_vec_vocab[rand_window[i]]
    #     print(key_matrix.shape)
    #     print(vec.shape)
    #     query_vector[i] = nmp.array(multiplyMatrix(query_matrix, vec))
    #     key_vector[i] = nmp.array(multiplyMatrix(key_matrix, vec))
    #     print(key_vector[i])
    # for i in range(CONTEXT_WINDOW_SIZE):
    #     for j in range(CONTEXT_WINDOW_SIZE):
    #         window[i][j] = dot(key_vector[i], query_vector[j])
    #         # print(window[i][j])
    # for column in range(CONTEXT_WINDOW_SIZE):
    #     thing = None
    #     for what in range(CONTEXT_WINDOW_SIZE):
    #         if what > column:
    #             window[column][what] = -1000000.0
    #     thing = softMax(window[column])
    #     window[column] = thing[0]
    # for i in range(len(window)):
    #     for j in range(len(window[0])):
    #         print(window[i][j])
    
    # #     print(things[column])

# def softMax(values: list, temperature: float=1):
#     values = np.array(values)
#     values = values - np.max(values)
#     exp_values = np.exp(values / temperature)
#     return (exp_values / np.sum(exp_values)).tolist()

if __name__ == "__main__":
    main()
