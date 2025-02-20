import random as rand
import jax.numpy as np
from jax import random

TRAIN_SPLIT = 0.9

class Data:
    def __init__(self, file_name: str):
        file = open(file_name, 'r')
        self.data = file.read()
        self.vocab = set(self.data)
        splice_index = int(len(self.data) * TRAIN_SPLIT)
        self.training_list = self.data[:splice_index]
        self.validate_list = self.data[splice_index:]

    def getRandomContextWindow(self, context_size: int, training_list: bool = True):
        data = None
        if training_list:
            data = self.training_list
        else:
            data = self.validate_list
        random_position = rand.randint(0, len(data) - 1)
        return [data[i] for i in range(random_position, random_position + context_size)]

    def randomVecsInVocab(self, context_size: int):
        key = random.PRNGKey(0)
        keys = random.split(key, len(self.vocab))
        randomized_vocab_vectors = {
            char: random.uniform(keys[i], shape=(1, context_size), minval=-1.0, maxval=1.0)
            for i, char in enumerate(self.vocab)
        }
        return randomized_vocab_vectors