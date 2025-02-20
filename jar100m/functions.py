import jax.numpy as np

def relu(x):
    return np.maximum(x, 0)

def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)
