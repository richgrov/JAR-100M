import random as rand
import jax.numpy as np

TRAIN_SPLIT = 0.9

class Data:
    def __init__(self, file_name: str):
        # global TRAIN_SPLIT
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
        randomized_vocab_vectors = {}
        for character in self.vocab:
            random_vec = []
            for _ in range(context_size):
                random_vec.append(rand.uniform(-1.0, 1.0))
            randomized_vocab_vectors[character] = np.array(random_vec)
        return randomized_vocab_vectors