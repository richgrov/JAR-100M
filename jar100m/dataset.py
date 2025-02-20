from typing import List, Tuple
import jax.numpy as np

class Dataset:
    def __init__(self, text) -> None:
        self.id_char_map = list(set(text))
        self.char_id_map = { char: id for id, char in enumerate(self.id_char_map) }

        self.text = self.encode(text)
        
        inputs = []
        outputs = []

        for x in range(20):
            for y in range(20):
                inputs.append([x/20, y/20])
                outputs.append([(x/20+y/20)/2+1.0])

        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def encode(self, text: str) -> List[int]:
        return np.array([self.char_id_map[char] for char in text])

    def decode(self, ids) -> str:
        return "".join([self.id_char_map[id] for id in ids])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)
