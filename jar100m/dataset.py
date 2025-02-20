from typing import Tuple
import jax.numpy as np

class Dataset:
    def __init__(self) -> None:
        inputs = []
        outputs = []

        for x in range(20):
            for y in range(20):
                inputs.append([x/20, y/20])
                outputs.append([(x/20+y/20)/2+1.0])

        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)

