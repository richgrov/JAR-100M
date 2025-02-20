from typing import List, Tuple
import jax.numpy as np

class Dataset:
    def __init__(self, text: str, context_window: int) -> None:
        self.id_char_map = list(set(text))
        self.char_id_map = { char: id for id, char in enumerate(self.id_char_map) }

        self.encoded_text = self.encode(text)
        self.context_window = context_window

    def encode(self, text: str) -> List[int]:
        return np.array([self.char_id_map[char] for char in text])

    def decode(self, ids) -> str:
        return "".join([self.id_char_map[id] for id in ids])

    @property
    def vocab(self):
        return self.id_char_map

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        window = index // self.context_window
        window_len = index % self.context_window + 1
        return (np.array(self.encoded_text[window:window+window_len]), self.encoded_text[window+window_len])

    def __len__(self):
        windows = (len(self.encoded_text) - self.context_window - 1)
        return windows * self.context_window

