from typing import List, Tuple
from jax import random
import random as rand
import jax.numpy as np

class Dataset:
    def __init__(self, text, context_size) -> None:
        self.id_char_map = list(set(text))
        self.char_id_map = self.randomVecsInVocab(context_size)

        self.text = self.encode(text)
        
        inputs = []
        outputs = []

        for x in range(20):
            for y in range(20):
                inputs.append([x/20, y/20])
                outputs.append([(x/20+y/20)/2+1.0])

        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
    
    def getRandomContextWindow(self, context_size: int):
        random_position = rand.randint(0, len(self.text) - 1)
        return np.array([self.text[i] for i in range(random_position, random_position + context_size)])

    def randomVecsInVocab(self, context_size: int):
        key = random.PRNGKey(0)
        keys = random.split(key, len(self.id_char_map))
        randomized_vocab_vectors = {
            char: random.uniform(keys[i], shape=(1, context_size), minval=-1.0, maxval=1.0)
            for i, char in enumerate(self.id_char_map)
        }
        return randomized_vocab_vectors

    def encode(self, text: str) -> List[int]:
        return np.array([self.char_id_map[char] for char in text])

    def decode(self, ids) -> str:
        return "".join([self.id_char_map[id] for id in ids])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)
