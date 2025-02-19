import random as rand
import numpy as np

TRAIN_SPLIT = 0.9

class Data:
    def __init__(self, file_name: str):
        global TRAIN_SPLIT
        file = open(file_name, 'r')
        self.data = file.read()
        self.vocab = set(self.data)
        splice_index = int(len(self.data) * TRAIN_SPLIT)
        self.training_list = self.data[:splice_index]
        self.validate_list = self.data[splice_index:]

    def getRandomContextWindow(self, context_size: int):
        random_position = rand.randint(0, len(self.training_list) - 1)
        random_slice = np.array([self.data[random_position: random_position + context_size], self.data[random_position + context_size]])
        # for idx in range(context_size):
        #     inputs = []
        #     for idx_2 in range(idx + 1):
        #         idx_3 = random_position + idx_2
        #         inputs.append(data[idx_3])
        #     random_slice.append(
        #         (inputs, data[idx_3 + 1])
        #     )
        return random_slice
    
    def randomVecsInVocab(self, context_size: int):
        randomized_vocab_vectors = {}
        for character in self.vocab:
            random_vec = []
            for vec_idx in range(context_size):
                random_vec.append(rand.uniform(-1.0, 1.0))
            randomized_vocab_vectors[character] = np.array(random_vec)
        return randomized_vocab_vectors