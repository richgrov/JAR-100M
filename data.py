import random as rand
import numpy as np

class Data:
    data = None
    vocab = None
    training_list = None
    validate_list = None
    TRAIN_SPLIT = None
    
    def __init__(self, file_name: str):
        global data
        global vocab
        global training_list
        global validate_list
        global TRAIN_SPLIT
        TRAIN_SPLIT = 0.9
        file = open(file_name, 'r')
        data = file.read()
        vocab = set(data)
        splice_index = int(len(data) * TRAIN_SPLIT)
        training_list = data[:splice_index]
        validate_list = data[splice_index:]

    def getRandomContextWindow(self, context_size: int):
        random_position = rand.randint(0, len(training_list) - 1)
        random_slice = np.array([data[random_position: random_position + context_size + 1]])
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
        for character in vocab:
            random_vec = []
            for vec_idx in range(context_size):
                random_vec.append(rand.uniform(-1.0, 1.0))
            randomized_vocab_vectors[character] = np.array(random_vec)
        return randomized_vocab_vectors