import jax.numpy as np
from jax.nn import log_softmax

def relu(x):
    return np.maximum(x, 0)

def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)

def cross_entropy_loss(logits, labels):
    log_probs = log_softmax(logits)
    return -np.mean(np.sum(labels * log_probs, axis=-1))
